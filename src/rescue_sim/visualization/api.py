"""FastAPI backend for the rescue simulation visualization.

Uses the environment modules directly:
  - rescue_sim.config.settings.GridSettings
  - rescue_sim.environment.generator.generate_grid
  - rescue_sim.environment.grid.Grid, Position
  - rescue_sim.environment.movement.MovementModel
  - rescue_sim.environment.sensors.CentralSensor
  - rescue_sim.Qlearning.q_learning.EpidemicHystereticQLearning

No simulation logic is duplicated — the API simply drives the existing
environment layer (an Epidemic Hysteretic Q-learning fleet) and streams the
state to the frontend over WebSocket.
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
from rescue_sim.Qlearning.baseline import default_start_positions
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

# No cookies/auth are used, so credentials stay disabled — the wildcard
# origin + credentials combination is rejected by browsers and flagged by
# security scanners. Restrict allow_origins to the deployment domain via
# RESCUE_SIM_CORS_ORIGINS once one is determined.
_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("RESCUE_SIM_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
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
    num_episodes: int = 5
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 1.0
    speed_ms: int = 100  # delay between steps in ms
    run_mode: str = "train"  # train | evaluate | instant_train
    algorithm: str = "neural_moe"  # neural_moe | epidemic_fleet


# ── Global state ────────────────────────────────────────────────────────────
current_config = SimConfig()
trained_visual_fleet: EpidemicHystereticQLearning | None = None
trained_visual_seed: int | None = None
trained_visual_shape: tuple[int, int, int] | None = None
trained_moe_policy: object | None = None
trained_moe_shape: tuple[int, int, int, int] | None = None
trained_moe_epochs: int = 0


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

            if config.algorithm == "neural_moe":
                await _run_moe_simulation(websocket, config, run_mode)
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


# ── Neural MoE simulation mode ──────────────────────────────────────────────
#
# Three experts drive every step, blended by the attention router:
#   exploration  — behavioral-cloned from the non-AI frontier heuristic
#   coordination — behavioral-cloned from the target-greedy deep-RL style teacher
#   fallback     — recurrent (GRU) local head for isolated agents, the
#                  neural counterpart of the Hysteretic Q baseline
_MOE_EXPERT_LABELS = ("exploration", "coordination", "fallback")
_MOE_BASELINES = {
    "hysteretic_alpha": 0.1,
    "hysteretic_beta": 0.01,
    "frontier_decay": 0.95,
}


async def _run_moe_simulation(websocket: WebSocket, config: SimConfig, run_mode: str) -> None:
    """Trains (if needed) and rolls out the neural MoE on one fixed grid.

    Unlike the fleet mode, training is offline (behavioral cloning + router
    optimization) and streams as ``moe_training`` progress messages. The
    rendered rollout then runs ``num_episodes`` tries on the *same* grid so
    the routing evolution is comparable try-to-try.
    """
    global trained_moe_policy, trained_moe_shape, trained_moe_epochs

    try:
        from rescue_sim.MoE.pipeline import FixedGridRescueEnv  # noqa: F401 (torch probe)
    except ModuleNotFoundError as error:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Neural MoE needs the optional torch dependency: {error!s}",
            }
        )
        return

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
    view_radius = max(1, config.sensor_range)
    expected_shape = (config.grid_width, config.grid_height, config.num_agents, view_radius)

    compatible_cache = trained_moe_policy if trained_moe_shape == expected_shape else None

    if run_mode == "evaluate":
        if compatible_cache is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "No compatible trained MoE policy is available yet. "
                        "Run Train first with the same grid size, agent count, and sensor range."
                    ),
                }
            )
            return
        policy = compatible_cache
    else:
        # Train presses accumulate: an existing compatible policy keeps
        # training on freshly collected data instead of starting over.
        policy = await _train_moe_policy_async(
            websocket, settings, config, view_radius, seed, existing=compatible_cache
        )
        if policy is None:
            return
        if compatible_cache is None:
            trained_moe_epochs = _MOE_TRAIN_EPOCHS
        else:
            trained_moe_epochs += _MOE_TRAIN_EPOCHS
        trained_moe_policy = policy
        trained_moe_shape = expected_shape

    await websocket.send_json(
        {"type": "moe_status", "trained_epochs": trained_moe_epochs}
    )

    if run_mode == "instant_train":
        await websocket.send_json(
            {
                "type": "training_complete",
                "total_episodes": 0,
                "final_success_rate": 0.0,
                "metrics": [],
            }
        )
        return

    await _run_moe_rollout(websocket, policy, settings, config, view_radius)


# One "Train" press worth of MoE training (accumulates across presses).
# Kept modest so a press finishes in a handful of seconds; use "Train More"
# to accumulate quality. The GRU fallback head (TBPTT) dominates BC cost.
_MOE_TRAIN_EPOCHS = 12

# Per-try step cap for the live MoE rollout. On the demo grids agents rescue
# or plateau well before this, so a lower cap keeps the animation short
# without changing outcomes (the user's max_steps can still lower it further).
_MOE_ROLLOUT_MAX_STEPS = 200


async def _train_moe_policy_async(
    websocket: WebSocket,
    settings: GridSettings,
    config: SimConfig,
    view_radius: int,
    seed: int,
    existing: object | None = None,
):
    """Runs the MoE training pipeline in a worker thread, streaming progress.

    Progress callbacks fire on the worker thread and are forwarded to the
    websocket as ``moe_training`` messages via a thread-safe queue. A "stop"
    message from the client abandons the result (the small training job is
    left to finish in the background).
    """
    from rescue_sim.MAPPO import RescueEnv
    from rescue_sim.MoE.pipeline import train_moe_policy

    loop = asyncio.get_running_loop()
    progress: asyncio.Queue = asyncio.Queue()

    def stage_reporter(stage: str, every: int = 1):
        def report(current: int, total: int, loss: float, acc: float, grad_norm: float) -> None:
            if current == 1 or current % every == 0 or current == total:
                loop.call_soon_threadsafe(
                    progress.put_nowait,
                    {
                        "type": "moe_training",
                        "stage": stage,
                        "epoch": current,
                        "total": total,
                        "loss": round(loss, 6),
                        "accuracy": round(acc, 2),
                    },
                )
        return report

    # Vary the seed per training round so "Train More" sees fresh grids/data
    round_seed = seed + trained_moe_epochs * 31

    def train():
        env = RescueEnv(
            settings,
            num_agents=config.num_agents,
            max_steps=80,
            view_radius=view_radius,
            seed=round_seed,
        )
        return train_moe_policy(
            env,
            episodes_per_head=6,
            collect_steps=70,
            epochs=_MOE_TRAIN_EPOCHS,
            router_steps=150,
            seed=round_seed,
            on_distill_epoch=stage_reporter("distillation"),
            on_router_step=stage_reporter("router", every=10),
            policy=existing,
        )

    task = asyncio.create_task(asyncio.to_thread(train))
    stopped = False
    while not task.done():
        try:
            update = await asyncio.wait_for(progress.get(), timeout=0.15)
            await websocket.send_json(update)
        except asyncio.TimeoutError:
            pass
        if not stopped:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if json.loads(raw).get("type") == "stop":
                    stopped = True
            except asyncio.TimeoutError:
                pass
    if stopped:
        await websocket.send_json({"type": "stopped"})
        return None
    while not progress.empty():
        await websocket.send_json(progress.get_nowait())
    try:
        return await task
    except ValueError as error:
        await websocket.send_json(
            {
                "type": "error",
                "message": (
                    f"Environment generation error: {error!s}. "
                    "Try reducing target count or obstacles."
                ),
            }
        )
        return None


async def _run_moe_rollout(
    websocket: WebSocket,
    policy,
    settings: GridSettings,
    config: SimConfig,
    view_radius: int,
) -> None:
    """Streams ``num_episodes`` policy-driven tries on the same fixed grid.

    Every step message carries a ``moe`` payload with the per-agent softmax
    routing vector, dominant expert, peer links, and GRU hidden norm, so the
    frontend can show routing flips the moment agents leave the 3-block
    communication radius.
    """
    global current_config

    import torch

    from rescue_sim.MoE.pipeline import FixedGridRescueEnv, build_peer_matrix

    # Training may have run on the GPU; single-step inference is trivial, so
    # serve the rollout on CPU and keep the per-step input tensors local.
    policy = policy.to("cpu")

    try:
        env = FixedGridRescueEnv(
            settings,
            num_agents=config.num_agents,
            max_steps=config.max_steps,
            view_radius=view_radius,
            grid_seed=settings.random_seed or 0,
        )
        obs = env.reset()
    except ValueError as error:
        await websocket.send_json(
            {
                "type": "error",
                "message": (
                    f"Environment generation error: {error!s}. "
                    "Try reducing target count or obstacles."
                ),
            }
        )
        return

    grid = env.grid
    obstacles = [{"x": p.x, "y": p.y} for p in grid.obstacles]
    targets = [{"x": p.x, "y": p.y, "type": "A"} for p in grid.target_a_positions]
    targets += [{"x": p.x, "y": p.y, "type": "B"} for p in grid.target_b_positions]
    total_targets = len(targets)

    num_agents = env.num_agents
    episode_metrics: list[dict] = []
    should_stop = False

    rollout_rng = random.Random(settings.random_seed)
    rollout_steps = min(config.max_steps, _MOE_ROLLOUT_MAX_STEPS)

    for episode in range(max(1, config.num_episodes)):
        obs = env.reset()  # identical grid every try (fixed competition grid)
        hidden = None      # GRU temporal memory resets per try
        rescued_records: list[dict] = []
        rescued_seen: set[Position] = set()
        total_reward = 0.0
        explore_sum = 0.0
        explore_samples = 0
        usage = {name: 0 for name in _MOE_EXPERT_LABELS}
        rescues_by_expert = {name: 0 for name in _MOE_EXPERT_LABELS}
        switches = 0
        prev_dominant: list[int] | None = None
        info = {"rescued": 0, "targets": total_targets, "success": False, "steps": 0}

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
                "agents": [
                    {"id": i, "x": p.x, "y": p.y} for i, p in enumerate(env.positions)
                ],
                "algorithm": "neural_moe",
            }
        )

        for step in range(1, rollout_steps + 1):
            # Cancellation / live config (speed) updates
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                cancel_msg = json.loads(raw)
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

            peer_np = build_peer_matrix(env.positions)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            peer_t = torch.tensor(peer_np, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.tensor(env.valid_action_mask(), dtype=torch.bool).unsqueeze(0)

            # Tensor shape assertions before the routing step
            assert obs_t.shape == (1, num_agents, env.obs_dim), f"obs shape {obs_t.shape}"
            assert peer_t.shape == (1, num_agents, num_agents), f"peer shape {peer_t.shape}"
            assert mask_t.shape == (1, num_agents, policy.action_dim), f"mask shape {mask_t.shape}"

            with torch.no_grad():
                y_final, weights, hidden = policy(obs_t, peer_t, mask_t, hidden)
                actions = torch.argmax(y_final.squeeze(0), dim=-1)  # greedy [A]

            # Small epsilon keeps tries from being carbon copies of each
            # other and breaks feed-forward oscillation loops.
            valid_np = mask_t.squeeze(0).numpy()
            act_np = actions.numpy().copy()
            for i in range(num_agents):
                if rollout_rng.random() < 0.1:
                    valid = [a for a in range(policy.action_dim) if valid_np[i, a]]
                    if valid:
                        act_np[i] = rollout_rng.choice(valid)

            weights_step = weights.squeeze(0)                    # [A, 3]
            dominant = torch.argmax(weights_step, dim=-1).tolist()
            gru_norm = torch.norm(hidden.squeeze(0), dim=-1)     # [A]
            peer_counts = peer_np.sum(axis=1).astype(int).tolist()
            explore_sum += float(weights_step[:, 0].mean().item())
            explore_samples += 1
            for d in dominant:
                usage[_MOE_EXPERT_LABELS[d]] += 1
            if prev_dominant is not None:
                switches += sum(1 for a, b in zip(prev_dominant, dominant) if a != b)
            prev_dominant = dominant

            obs, reward, done, info = env.step(act_np)
            total_reward += float(reward)

            for i, pos in enumerate(env.positions):
                if grid.has_target(pos) and pos not in rescued_seen:
                    rescued_seen.add(pos)
                    rescued_records.append(
                        {"x": pos.x, "y": pos.y, "step": step, "type": grid.target_type_at(pos)}
                    )
                    rescues_by_expert[_MOE_EXPERT_LABELS[dominant[i]]] += 1

            agent_states = [
                {
                    "id": i,
                    "x": env.positions[i].x,
                    "y": env.positions[i].y,
                    "action": CARDINAL_ACTIONS[int(act_np[i])].value,
                    "reward": round(float(reward) / num_agents, 2),
                    "expert": _MOE_EXPERT_LABELS[dominant[i]],
                }
                for i in range(num_agents)
            ]
            moe_payload = {
                "weights": [
                    [round(float(w), 4) for w in weights_step[i]] for i in range(num_agents)
                ],
                "active_expert": [_MOE_EXPERT_LABELS[d] for d in dominant],
                "peer_count": peer_counts,
                "peers": [
                    [j for j in range(num_agents) if j != i and peer_np[i, j] > 0]
                    for i in range(num_agents)
                ],
                "gru_norm": [round(float(gru_norm[i].item()), 3) for i in range(num_agents)],
                "baselines": _MOE_BASELINES,
            }

            await websocket.send_json(
                {
                    "type": "step",
                    "episode": episode,
                    "step": step,
                    "agents": agent_states,
                    "rescued": rescued_records,
                    "active_targets": int(info["targets"] - info["rescued"]),
                    "moe": moe_payload,
                }
            )
            await asyncio.sleep(config.speed_ms / 1000.0)
            if done:
                break

        if should_stop:
            break

        usage_total = max(sum(usage.values()), 1)
        metric = {
            "episode": episode,
            "steps": int(info["steps"]),
            "rescued_count": len(rescued_records),
            "target_count": total_targets,
            "success": bool(info["success"]),
            "total_reward": round(total_reward, 2),
            # For the MoE the "exploration rate" is the mean exploration
            # gating weight over the try — comparable across tries.
            "exploration_rate": round(explore_sum / max(explore_samples, 1), 4),
            "moe": {
                "expert_share": {
                    name: round(count / usage_total, 3) for name, count in usage.items()
                },
                "switches": switches,
                "rescues_by_expert": rescues_by_expert,
            },
        }
        episode_metrics.append(metric)
        success_rate = sum(1 for m in episode_metrics if m["success"]) / len(episode_metrics)
        await websocket.send_json(
            {
                "type": "episode_end",
                **metric,
                "success_rate": round(success_rate, 4),
                "avg_steps": round(
                    sum(m["steps"] for m in episode_metrics) / len(episode_metrics), 1
                ),
            }
        )

    if not should_stop:
        n = max(len(episode_metrics), 1)
        moe_summary = {
            "tries": len(episode_metrics),
            "successes": sum(1 for m in episode_metrics if m["success"]),
            "avg_rescued": round(sum(m["rescued_count"] for m in episode_metrics) / n, 2),
            "expert_share": {
                name: round(
                    sum(m["moe"]["expert_share"][name] for m in episode_metrics) / n, 3
                )
                for name in _MOE_EXPERT_LABELS
            },
            "rescues_by_expert": {
                name: sum(m["moe"]["rescues_by_expert"][name] for m in episode_metrics)
                for name in _MOE_EXPERT_LABELS
            },
            "avg_switches": round(sum(m["moe"]["switches"] for m in episode_metrics) / n, 1),
        }
        await websocket.send_json(
            {
                "type": "training_complete",
                "total_episodes": len(episode_metrics),
                "final_success_rate": round(
                    sum(1 for m in episode_metrics if m["success"]) / n, 4
                ),
                "metrics": episode_metrics,
                "moe_summary": moe_summary,
            }
        )


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
    report["deep_benchmark"] = _deep_benchmark_rows(grid_settings, config)
    report["deep_benchmark_note"] = (
        "Deep RL rows are greedy evaluations of the saved checkpoints on fresh "
        "grids with the same settings as this run. Train them first: "
        "docker compose run --rm train-qmix / train-transfqmix / train-mappo."
    )
    return report


def _deep_benchmark_rows(grid_settings: GridSettings, config: SimConfig) -> list[dict]:
    """Greedy checkpoint evaluations of the deep MARL models (QMIX/TransfQMix/MAPPO).

    Each model is loaded from its saved checkpoint and evaluated on fresh grids
    generated with the same settings as the live run. Models without a matching
    checkpoint (or with an incompatible one) produce an "unavailable" row with
    the reason instead of failing the whole report.
    """
    rows = []
    for name, key, service in (
        ("QMIX", "qmix", "train-qmix"),
        ("TransfQMix", "transfqmix", "train-transfqmix"),
        ("MAPPO", "mappo", "train-mappo"),
    ):
        path = _CHECKPOINTS[key]
        error: str | None = None
        metrics: dict | None = None
        if not path.exists():
            error = f"No checkpoint at {path}. Train it: docker compose run --rm {service}"
        else:
            try:
                metrics = _evaluate_deep_checkpoint(name, path, grid_settings, config)
            except ModuleNotFoundError as exc:
                error = f"Deep RL needs the optional torch dependency: {exc!s}"
            except Exception as exc:  # incompatible agents/sensor settings, corrupt file
                error = (
                    "Checkpoint could not be evaluated with the current agent count / "
                    f"sensor range: {exc!s}"
                )

        if metrics is None:
            rows.append(
                {
                    "agent_name": name,
                    "algorithm_group": "deep_rl_benchmark",
                    "status": "unavailable",
                    "success_rate": None,
                    "average_steps": None,
                    "average_rescued_targets": None,
                    "average_accumulated_reward": None,
                    "num_agents": config.num_agents,
                    "error": error,
                }
            )
        else:
            rows.append(
                {
                    "agent_name": name,
                    "algorithm_group": "deep_rl_benchmark",
                    "status": "ok",
                    "success_rate": metrics["success_rate"],
                    "average_steps": metrics["avg_steps"],
                    "average_rescued_targets": metrics["avg_rescued"],
                    "average_accumulated_reward": None,
                    "num_agents": config.num_agents,
                }
            )
    return rows


def _evaluate_deep_checkpoint(
    name: str,
    path: Path,
    grid_settings: GridSettings,
    config: SimConfig,
    episodes: int = 3,
) -> dict:
    """Loads one deep-model checkpoint and runs greedy evaluation episodes."""
    from rescue_sim.config.settings import MappoSettings, QmixSettings, TransfQmixSettings
    from rescue_sim.MAPPO import MAPPO, RescueEnv
    from rescue_sim.QMIX import QMIX
    from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX

    view_radius = max(1, config.sensor_range)
    seed = grid_settings.random_seed

    if name == "QMIX":
        env = RescueEnv(grid_settings, num_agents=config.num_agents,
                        max_steps=config.max_steps, view_radius=view_radius, seed=seed)
        trainer = QMIX(env, QmixSettings(num_agents=config.num_agents, random_seed=seed))
    elif name == "TransfQMix":
        env = EntityRescueEnv(grid_settings, num_agents=config.num_agents,
                              max_steps=config.max_steps, view_radius=view_radius, seed=seed)
        trainer = TransfQMIX(env, TransfQmixSettings(num_agents=config.num_agents, random_seed=seed))
    else:
        env = RescueEnv(grid_settings, num_agents=config.num_agents,
                        max_steps=config.max_steps, view_radius=view_radius, seed=seed)
        trainer = MAPPO(env, MappoSettings(num_agents=config.num_agents, random_seed=seed))

    trainer.load_checkpoint(path)
    return trainer.evaluate(episodes=episodes)
