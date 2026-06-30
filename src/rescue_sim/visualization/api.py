"""FastAPI backend for the rescue simulation visualization.

Uses the environment modules directly:
  - rescue_sim.config.settings.GridSettings
  - rescue_sim.environment.generator.generate_grid
  - rescue_sim.environment.grid.Grid, Position
  - rescue_sim.environment.movement.MovementModel
  - rescue_sim.environment.sensors.CentralSensor
  - rescue_sim.agents.single_agent.SingleAgent

No simulation logic is duplicated — the API simply drives the existing
environment layer and streams the state to the frontend over WebSocket.
"""

import asyncio
import json
import math
import os
import random
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rescue_sim.config.settings import FleetSettings, GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.Qlearning.multi_agent_baseline import default_start_positions
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning
from rescue_sim.simulation.evaluation import (
    EvaluationScenario,
    evaluate_simulation_grid,
)
from rescue_sim.shared import (
    CARDINAL_ACTIONS,
    GossipConfig,
    HystereticConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    TargetType,
    calculate_reward,
)

app = FastAPI(title="Rescue Sim Visualization API")

# TODO(security): Restrict allow_origins to the actual frontend origin
# instead of wildcard once a deployment domain is determined.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve the built frontend if it exists (for Docker production mode)
_FRONTEND_DIST_ENV = os.environ.get("FRONTEND_DIST_DIR")
if _FRONTEND_DIST_ENV:
    _FRONTEND_DIST = Path(_FRONTEND_DIST_ENV)
else:
    _THIS_DIR = Path(__file__).resolve().parent
    _FRONTEND_DIST = _THIS_DIR / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount(
        "/app",
        StaticFiles(directory=str(_FRONTEND_DIST), html=True),
        name="frontend",
    )



# ── Pydantic models for config ─────────────────────────────────────────────
def _load_default_scenario() -> dict:
    path = Path("configs/default_scenario.yaml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


_DEFAULT_SCENARIO = _load_default_scenario()
_DEFAULT_GRID = _DEFAULT_SCENARIO.get("grid", {})
_DEFAULT_AGENT = _DEFAULT_SCENARIO.get("agent", {})
_DEFAULT_SIMULATION = _DEFAULT_SCENARIO.get("simulation", {})
_DEFAULT_FLEET = FleetSettings()
_CHECKPOINT_DIR = Path(os.environ.get("RESCUE_SIM_CHECKPOINT_DIR", "checkpoints"))
_CHECKPOINTS = {
    "qmix": _CHECKPOINT_DIR / "qmix.pt",
    "transfqmix": _CHECKPOINT_DIR / "transfqmix.pt",
    "mappo": _CHECKPOINT_DIR / "mappo.pt",
}


class SimConfig(BaseModel):
    grid_width: int = _DEFAULT_GRID.get("width", 10)
    grid_height: int = _DEFAULT_GRID.get("height", 10)
    obstacle_probability: float = _DEFAULT_GRID.get("obstacle_probability", 0.15)
    target_count: int = _DEFAULT_GRID.get("target_a_count", 2) + _DEFAULT_GRID.get("target_b_count", 2)
    num_agents: int = _DEFAULT_FLEET.num_agents
    sensor_range: int = _DEFAULT_AGENT.get("sensor_range", 3)
    max_steps: int = _DEFAULT_SIMULATION.get("max_steps", 500)
    num_episodes: int = 10
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 1.0
    speed_ms: int = 100  # delay between steps in ms
    run_mode: str = "train"  # train | evaluate | instant_train


# ── Global state ────────────────────────────────────────────────────────────
current_config = SimConfig()
trained_visual_fleet: EpidemicHystereticQLearning | None = None
trained_visual_seed: int | None = None
trained_visual_shape: tuple[int, int, int] | None = None
trained_hybrid_policy: dict | None = None


# ── REST endpoints ──────────────────────────────────────────────────────────
@app.get("/")
async def root_redirect():
    """Redirect root to /app."""
    return RedirectResponse(url="/app")


@app.get("/api/config")
async def get_config():
    return current_config


@app.post("/api/config")
async def set_config(config: SimConfig):
    global current_config
    current_config = config
    return {"status": "ok", "config": current_config}


@app.get("/api/health")
async def health():
    """Health check endpoint for Docker."""
    return {"status": "ok"}


# ── WebSocket simulation stream ────────────────────────────────────────────
@app.websocket("/ws/simulation")
async def simulation_ws(websocket: WebSocket):
    await websocket.accept()
    global current_config, trained_visual_fleet, trained_visual_seed, trained_visual_shape

    try:
        while True:
            # Wait for a "start" or "config" command from frontend
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "config":
                current_config = SimConfig(**msg.get("data", {}))
                await websocket.send_json(
                    {"type": "config_ack", "config": current_config.model_dump()}
                )
                continue

            if msg.get("type") != "start":
                continue

            config = current_config
            run_mode = config.run_mode if config.run_mode in {"train", "evaluate", "instant_train"} else "train"

            # ── Validation ────────────────────────────────────────────────
            if (
                config.grid_width < 4
                or config.grid_width > 100
                or config.grid_height < 4
                or config.grid_height > 100
            ):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Grid dimensions must be between 4x4 and 100x100.",
                    }
                )
                continue
            if config.obstacle_probability < 0.0 or config.obstacle_probability > 0.9:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Obstacle probability must be between 0.0 and 0.9.",
                    }
                )
                continue
            if config.target_count < 1:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "At least 1 target is required.",
                    }
                )
                continue
            if config.num_agents < 1:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "At least 1 agent is required.",
                    }
                )
                continue

            await _run_hybrid_simulation(websocket, config)
            continue

            episode_metrics: list[dict] = []
            should_stop = False
            scenario_seed = random.randint(0, 999999)
            fleet = trained_visual_fleet
            expected_shape = (config.grid_width, config.grid_height, config.num_agents)
            if run_mode == "evaluate":
                if (
                    fleet is None
                    or trained_visual_seed is None
                    or trained_visual_shape != expected_shape
                ):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": (
                                "No compatible trained multi-agent policy is available yet. "
                                "Run Train first with the same grid size and agent count."
                            ),
                        }
                    )
                    continue
                scenario_seed = trained_visual_seed
                fleet.epsilon = 0.0
            else:
                fleet = None

            total_episodes = 1 if run_mode == "evaluate" else config.num_episodes
            last_grid = None
            last_settings = None
            last_starts = None
            for episode in range(total_episodes):
                if should_stop:
                    break

                seed = scenario_seed

                # ── Generate grid using environment.generator ─────────────
                target_a = math.ceil(config.target_count / 2)
                target_b = config.target_count - target_a

                settings = GridSettings(
                    width=config.grid_width,
                    height=config.grid_height,
                    obstacle_probability=config.obstacle_probability,
                    target_a_count=target_a,
                    target_b_count=target_b,
                    random_seed=seed,
                )

                start_pos = Position(
                    _DEFAULT_AGENT.get("start_x", 0),
                    _DEFAULT_AGENT.get("start_y", 0),
                )

                try:
                    grid = generate_grid(settings, start_pos)
                    starts = default_start_positions(grid, config.num_agents, start_pos)
                except ValueError as e:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": (
                                f"Environment generation error: {e!s}. "
                                "Try reducing target count or obstacles."
                            ),
                        }
                    )
                    should_stop = True
                    break

                if fleet is None:
                    alpha = config.learning_rate
                    fleet = EpidemicHystereticQLearning(
                        grid=grid,
                        config=HystereticConfig(
                            alpha=alpha,
                            beta=min(0.1, alpha),
                            discount_factor=config.discount_factor,
                            epsilon=config.exploration_rate,
                        ),
                        gossip=GossipConfig(comm_radius=float(config.sensor_range)),
                        max_agents=max(20, config.num_agents),
                        seed=seed,
                    )
                    for agent_id, start in starts.items():
                        fleet.add_agent(agent_id, start)
                else:
                    fleet.reset_positions(starts)

                # Build JSON-serializable obstacle & target lists
                obstacles = [{"x": p.x, "y": p.y} for p in grid.obstacles]
                targets = []
                for p in grid.target_a_positions:
                    targets.append({"x": p.x, "y": p.y, "type": "A"})
                for p in grid.target_b_positions:
                    targets.append({"x": p.x, "y": p.y, "type": "B"})

                movement = MovementModel()
                sensor = CentralSensor(grid)
                positions = dict(starts)
                active_targets = set(grid.target_a_positions) | set(grid.target_b_positions)
                rescued: list[dict] = []
                rescued_positions: set[Position] = set()
                visited_by_agent = {
                    agent_id: {position}
                    for agent_id, position in positions.items()
                }
                visited_positions = set(positions.values())
                total_reward = 0.0

                # Send initial state
                if run_mode != "instant_train":
                    await websocket.send_json(
                        {
                            "type": "episode_start",
                            "episode": episode,
                            "grid": {
                                "width": config.grid_width,
                                "height": config.grid_height,
                                "obstacles": obstacles,
                                "targets": targets,
                            },
                            "agents": _agent_payloads(positions),
                        }
                    )

                steps = 0

                for step in range(config.max_steps):
                    if not active_targets:
                        break

                    # Check for cancellation or speed update (throttled in instant_train to avoid latency)
                    check_cancel = True
                    if run_mode == "instant_train" and step % 20 != 0:
                        check_cancel = False

                    if check_cancel:
                        try:
                            cancel_check = await asyncio.wait_for(
                                websocket.receive_text(), timeout=0.001
                            )
                            cancel_msg = json.loads(cancel_check)
                            if cancel_msg.get("type") == "stop":
                                await websocket.send_json({"type": "stopped"})
                                should_stop = True
                                break
                            if cancel_msg.get("type") == "config":
                                new_cfg = SimConfig(**cancel_msg.get("data", {}))
                                current_config = new_cfg
                                config = new_cfg
                        except asyncio.TimeoutError:
                            pass

                    action_indices = fleet.select_actions()
                    actions = {
                        agent_id: CARDINAL_ACTIONS[action_index]
                        for agent_id, action_index in action_indices.items()
                    }
                    rewards: dict[str, float] = {}
                    next_positions: dict[str, Position] = {}
                    dones: dict[str, bool] = {}
                    agent_states: list[dict] = []

                    for agent_id in sorted(positions):
                        action = actions[agent_id]
                        before = positions[agent_id]
                        movement_result = movement.apply(grid, before, action.value)
                        after = movement_result.end
                        positions[agent_id] = after
                        next_positions[agent_id] = after

                        next_observation = sensor.observe(
                            agent_id,
                            after,
                            config.sensor_range,
                        )
                        target_type = grid.target_type_at(after)
                        rescued_target_type = None
                        if target_type is not None and after not in rescued_positions:
                            rescued_positions.add(after)
                            active_targets.discard(after)
                            rescued_target_type = TargetType(target_type)
                            rescued.append(
                                {
                                    "x": after.x,
                                    "y": after.y,
                                    "step": step,
                                    "type": target_type,
                                }
                            )

                        done = not active_targets
                        reward = calculate_reward(
                            RewardEvent(
                                moved=movement_result.moved,
                                move=action.value,
                                newly_discovered_cells=len(next_observation.newly_discovered_cells),
                                rescued_target_type=rescued_target_type,
                                completed_episode=done,
                                repeated_cell=after in visited_by_agent[agent_id],
                            ),
                            SPRINT3_REWARD_CONFIG,
                        )
                        rewards[agent_id] = reward
                        dones[agent_id] = done
                        total_reward += reward
                        visited_by_agent[agent_id].add(after)
                        visited_positions.add(after)
                        agent_states.append(
                            {
                                "id": int(agent_id.split("-")[-1]),
                                "x": after.x,
                                "y": after.y,
                                "action": action.value,
                                "reward": round(reward, 2),
                            }
                        )

                    if run_mode in {"train", "instant_train"}:
                        fleet.record_transitions(action_indices, rewards, next_positions, dones)
                        fleet.gossip()
                    else:
                        fleet.reset_positions(next_positions)
                    steps = step + 1

                    if run_mode != "instant_train":
                        await websocket.send_json(
                            {
                                "type": "step",
                                "episode": episode,
                                "step": steps,
                                "agents": agent_states,
                                "rescued": rescued,
                                "active_targets": len(active_targets),
                            }
                        )

                        await asyncio.sleep(config.speed_ms / 1000.0)

                if should_stop:
                    break

                success = not active_targets
                metric = {
                    "episode": episode,
                    "steps": steps,
                    "rescued_count": len(rescued),
                    "target_count": config.target_count,
                    "success": success,
                    "total_reward": round(total_reward, 2),
                    "exploration_rate": round(fleet.epsilon, 4),
                }
                episode_metrics.append(metric)
                if run_mode in {"train", "instant_train"}:
                    fleet.epsilon = max(0.05, fleet.epsilon * 0.85)

                success_rate = sum(
                    1 for m in episode_metrics if m["success"]
                ) / len(episode_metrics)

                await websocket.send_json(
                    {
                        "type": "episode_end",
                        **metric,
                        "success_rate": round(success_rate, 4),
                        "avg_steps": round(
                            sum(m["steps"] for m in episode_metrics)
                            / len(episode_metrics),
                            1,
                        ),
                    }
                )
                if run_mode == "instant_train":
                    await asyncio.sleep(0)
                last_grid = grid
                last_settings = settings
                last_starts = starts

            if not should_stop:
                if run_mode in {"train", "instant_train"}:
                    trained_visual_fleet = fleet
                    trained_visual_seed = scenario_seed
                    trained_visual_shape = expected_shape
                if run_mode == "evaluate" and episode_metrics:
                    await websocket.send_json(
                        {
                            "type": "baseline_comparison",
                            "report": _build_run_comparison_report(
                                grid=last_grid,
                                grid_settings=last_settings,
                                starts=last_starts,
                                config=config,
                            ),
                        }
                    )
                await websocket.send_json(
                    {
                        "type": "training_complete",
                        "total_episodes": len(episode_metrics),
                        "final_success_rate": round(
                            sum(1 for m in episode_metrics if m["success"])
                            / max(len(episode_metrics), 1),
                            4,
                        ),
                        "metrics": episode_metrics,
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as error:
        try:
            await websocket.send_json({"type": "error", "message": f"Simulation error: {error!s}"})
        except Exception:
            pass


async def _run_hybrid_simulation(websocket: WebSocket, config: SimConfig) -> None:
    global trained_hybrid_policy

    seed = _DEFAULT_GRID.get("random_seed", 42)
    target_a = math.ceil(config.target_count / 2)
    target_b = config.target_count - target_a
    settings = GridSettings(
        width=config.grid_width,
        height=config.grid_height,
        obstacle_probability=config.obstacle_probability,
        target_a_count=target_a,
        target_b_count=target_b,
        random_seed=seed,
    )
    start_pos = Position(_DEFAULT_AGENT.get("start_x", 0), _DEFAULT_AGENT.get("start_y", 0))

    try:
        grid = generate_grid(settings, start_pos)
        starts = default_start_positions(grid, config.num_agents, start_pos)
    except ValueError as error:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Environment generation error: {error!s}. Try reducing target count or obstacles.",
            }
        )
        return

    obstacles = [{"x": p.x, "y": p.y} for p in grid.obstacles]
    targets = [{"x": p.x, "y": p.y, "type": "A"} for p in grid.target_a_positions]
    targets += [{"x": p.x, "y": p.y, "type": "B"} for p in grid.target_b_positions]

    if config.run_mode not in {"evaluate", "instant_train"}:
        await websocket.send_json(
            {
                "type": "episode_start",
                "episode": 0,
                "grid": {
                    "width": config.grid_width,
                    "height": config.grid_height,
                    "obstacles": obstacles,
                    "targets": targets,
                },
                "agents": _agent_payloads(starts),
                "algorithm": "hybrid_moe:building",
            }
        )
        await asyncio.sleep(0)

    expected_shape = (config.grid_width, config.grid_height, config.num_agents)
    if config.run_mode == "evaluate":
        if trained_hybrid_policy is None or trained_hybrid_policy.get("shape") != expected_shape:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "No compatible trained hybrid policy is available yet. "
                        "Run Train first with the same grid size and agent count."
                    ),
                }
            )
            return
        hybrid = trained_hybrid_policy
        grid = hybrid["grid"]
        settings = hybrid["settings"]
        starts = hybrid["starts"]
        obstacles = [{"x": p.x, "y": p.y} for p in grid.obstacles]
        targets = [{"x": p.x, "y": p.y, "type": "A"} for p in grid.target_a_positions]
        targets += [{"x": p.x, "y": p.y, "type": "B"} for p in grid.target_b_positions]
    else:
        try:
            hybrid = await _build_live_hybrid_policy_async(
                websocket, settings, grid, starts, config, seed
            )
            if hybrid is None:
                return
        except ModuleNotFoundError as error:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Hybrid/MoE needs the optional torch dependency: {error!s}",
                }
            )
            return
        except FileNotFoundError as error:
            await websocket.send_json({"type": "error", "message": str(error)})
            return
        except RuntimeError as error:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "Hybrid checkpoints could not be loaded. "
                        "They may have been trained with different agents/view radius/settings. "
                        f"Details: {error!s}"
                    ),
                }
            )
            return
        if config.run_mode in {"train", "instant_train"}:
            trained_hybrid_policy = {**hybrid, "shape": expected_shape}

    if config.run_mode == "instant_train":
        await websocket.send_json(
            {
                "type": "training_complete",
                "total_episodes": 1,
                "final_success_rate": 0.0,
                "metrics": [],
            }
        )
        return

    moe = hybrid["moe"]
    leader = hybrid["leader"]
    policy = hybrid["policy"]
    env = moe.env
    flat_obs = env.reset()

    if config.run_mode == "evaluate":
        await websocket.send_json(
            {
                "type": "episode_start",
                "episode": 0,
                "grid": {
                    "width": config.grid_width,
                    "height": config.grid_height,
                    "obstacles": obstacles,
                    "targets": targets,
                },
                "agents": _agent_payloads(starts),
                "algorithm": f"hybrid_moe:{leader}",
            }
        )

    rescued: list[dict] = []
    rescued_positions: set[Position] = set()
    total_reward = 0.0
    info = {"success": False, "rescued": 0, "targets": config.target_count, "steps": 0}

    for step in range(1, config.max_steps + 1):
        actions = _select_hybrid_actions(moe, leader, policy, flat_obs)
        flat_obs, reward, done, info = env.step(actions)
        total_reward += float(reward)

        agent_states = []
        for index, position in enumerate(env.positions):
            target_type = grid.target_type_at(position)
            if target_type is not None and position not in rescued_positions:
                rescued_positions.add(position)
                rescued.append({"x": position.x, "y": position.y, "step": step, "type": target_type})
            agent_states.append(
                {
                    "id": index,
                    "x": position.x,
                    "y": position.y,
                    "action": CARDINAL_ACTIONS[int(actions[index])].value,
                    "reward": round(float(reward) / max(config.num_agents, 1), 2),
                }
            )

        await websocket.send_json(
            {
                "type": "step",
                "episode": 0,
                "step": step,
                "agents": agent_states,
                "rescued": rescued,
                "active_targets": max(config.target_count - len(rescued_positions), 0),
                "algorithm": f"hybrid_moe:{leader}",
            }
        )
        if done:
            break
        await asyncio.sleep(config.speed_ms / 1000.0)

    success = bool(info.get("success", False))
    metric = {
        "episode": 0,
        "steps": int(info.get("steps", step)),
        "rescued_count": int(info.get("rescued", len(rescued_positions))),
        "target_count": int(info.get("targets", config.target_count)),
        "success": success,
        "total_reward": round(total_reward, 2),
        "exploration_rate": round(float(moe.fleet.epsilon), 4),
    }
    await websocket.send_json(
        {
            "type": "episode_end",
            **metric,
            "success_rate": 1.0 if success else 0.0,
            "avg_steps": metric["steps"],
            "algorithm": f"hybrid_moe:{leader}",
        }
    )
    await websocket.send_json(
        {
            "type": "baseline_comparison",
            "report": _build_run_comparison_report(
                grid=grid,
                grid_settings=settings,
                starts=starts,
                config=config,
                hybrid_report=hybrid.get("report"),
            ),
        }
    )
    await websocket.send_json(
        {
            "type": "training_complete",
            "total_episodes": 1,
            "final_success_rate": 1.0 if success else 0.0,
            "metrics": [metric],
        }
    )


def _build_live_hybrid_policy(settings, grid, starts, config, seed):
    from rescue_sim.config.settings import MappoSettings, MoeSettings, QmixSettings, TransfQmixSettings
    from rescue_sim.Ensemble import ValueEnsemble, performance_weights
    from rescue_sim.MAPPO import MAPPO, RescueEnv
    from rescue_sim.MoE import MixtureOfExperts
    from rescue_sim.MoE.moe import FLEET
    from rescue_sim.QMIX import QMIX
    from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX

    episodes = max(1, int(config.num_episodes))
    view_radius = max(1, int(config.sensor_range))
    _require_hybrid_checkpoints()

    def env():
        return EntityRescueEnv(
            settings,
            num_agents=config.num_agents,
            max_steps=config.max_steps,
            view_radius=view_radius,
            seed=seed,
        )

    qmix = QMIX(env(), QmixSettings(num_agents=config.num_agents, random_seed=seed))
    qmix.load_checkpoint(_CHECKPOINTS["qmix"])

    transf = TransfQMIX(env(), TransfQmixSettings(num_agents=config.num_agents, random_seed=seed))
    transf.load_checkpoint(_CHECKPOINTS["transfqmix"])

    mappo = MAPPO(
        RescueEnv(
            settings,
            num_agents=config.num_agents,
            max_steps=config.max_steps,
            view_radius=view_radius,
            seed=seed,
        ),
        MappoSettings(num_agents=config.num_agents, random_seed=seed),
    )
    mappo.load_checkpoint(_CHECKPOINTS["mappo"])

    w_qmix, w_transf = performance_weights(
        qmix.evaluate(episodes=2)["success_rate"],
        transf.evaluate(episodes=2)["success_rate"],
    )
    ensemble = ValueEnsemble(qmix, transf, env(), w_qmix, w_transf)

    moe = MixtureOfExperts.from_models(
        qmix=qmix,
        transf=transf,
        mappo=mappo,
        ensemble=ensemble,
        settings=MoeSettings(
            num_trials=episodes,
            random_seed=seed,
            grid_seed=seed,
            comm_radius=float(config.sensor_range),
            epsilon_start=float(config.exploration_rate),
            alpha=float(config.learning_rate),
            discount_factor=float(config.discount_factor),
        ),
        grid=grid,
        baselines={},
        start_positions=starts,
    )
    report = moe.run(num_trials=episodes)
    leader = report.final_leader or FLEET
    policy = None if leader == FLEET else moe.deep_experts[leader]
    return {
        "moe": moe,
        "leader": leader,
        "policy": policy,
        "report": report.to_dict(),
        "grid": grid,
        "settings": settings,
        "starts": starts,
    }


def _require_hybrid_checkpoints() -> None:
    missing = [path for path in _CHECKPOINTS.values() if not path.exists()]
    if not missing:
        return
    commands = (
        "Missing hybrid checkpoints: "
        + ", ".join(str(path) for path in missing)
        + ". Run: python scripts/train_qmix.py --checkpoint checkpoints/qmix.pt; "
        "python scripts/train_transfqmix.py --checkpoint checkpoints/transfqmix.pt; "
        "python scripts/train_mappo.py --checkpoint checkpoints/mappo.pt"
    )
    raise FileNotFoundError(commands)


def _select_hybrid_actions(moe, leader, policy, flat_obs):
    from rescue_sim.MoE.moe import FLEET

    if leader == FLEET:
        policies = {agent_id: moe.fleet.greedy_policy(agent_id) for agent_id in moe.agent_ids}
        return [
            int(policies[moe.agent_ids[i]][moe.env.positions[i].y, moe.env.positions[i].x])
            for i in range(moe.num_agents)
        ]
    return policy(moe.env, flat_obs)


async def _build_live_hybrid_policy_async(websocket, settings, grid, starts, config, seed):
    task = asyncio.create_task(
        asyncio.to_thread(_build_live_hybrid_policy, settings, grid, starts, config, seed)
    )
    while not task.done():
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.2)
            msg = json.loads(raw)
            if msg.get("type") == "stop":
                await websocket.send_json({"type": "stopped"})
                return None
        except asyncio.TimeoutError:
            pass
    return await task


def _agent_payloads(positions: dict[str, Position]) -> list[dict]:
    return [
        {"id": int(agent_id.split("-")[-1]), "x": position.x, "y": position.y}
        for agent_id, position in sorted(positions.items())
    ]


def _build_run_comparison_report(
    grid: object,
    grid_settings: GridSettings,
    starts: dict[str, Position],
    config: SimConfig,
    hybrid_report: dict | None = None,
) -> dict:
    if grid is None or grid_settings is None or starts is None:
        raise ValueError("evaluation comparison needs a completed simulation grid")

    scenario = EvaluationScenario(
        name="visualization_current_grid",
        grid_settings=grid_settings,
        max_steps=config.max_steps,
        start=starts["agent-0"],
        num_agents=config.num_agents,
        communication_range=float(config.sensor_range),
    )
    report = evaluate_simulation_grid(
        scenario=scenario,
        grid=grid,
        start_positions=starts,
    ).__dict__
    if hybrid_report:
        report["hybrid_report"] = hybrid_report
        report["deep_benchmark"] = _deep_rows_from_hybrid_report(
            hybrid_report,
            config.num_agents,
        )
        report["deep_benchmark_note"] = (
            "Deep RL metrics come from the loaded checkpoint models scored by the MoE "
            "on this same live grid."
        )
    return report


def _deep_rows_from_hybrid_report(hybrid_report: dict, num_agents: int) -> list[dict]:
    rows = []
    leaderboard = hybrid_report.get("leaderboard", {})
    for name in ("MAPPO", "QMIX", "TransfQMix", "Ensemble", "Distilled"):
        metrics = leaderboard.get(name)
        if metrics is None:
            rows.append(
                {
                    "agent_name": name,
                    "algorithm_group": "deep_rl_benchmark",
                    "status": "unavailable",
                    "success_rate": None,
                    "average_steps": None,
                    "average_accumulated_reward": None,
                    "average_rescued_targets": None,
                    "num_agents": num_agents,
                    "error": "No checkpoint-backed MoE metrics for this algorithm.",
                }
            )
            continue
        rows.append(
            {
                "agent_name": name,
                "algorithm_group": "deep_rl_benchmark",
                "status": "ok",
                "success_rate": 1.0 if metrics.get("success") else 0.0,
                "average_steps": float(metrics.get("steps", 0)),
                "average_accumulated_reward": float(metrics.get("score", 0.0)),
                "average_rescued_targets": float(metrics.get("rescued", 0)),
                "num_agents": num_agents,
            }
        )
    return rows
