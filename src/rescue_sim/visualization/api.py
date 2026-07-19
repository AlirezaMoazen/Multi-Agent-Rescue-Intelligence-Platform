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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

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
    # Bounds are enforced by pydantic so the public API rejects malformed or
    # resource-exhausting requests (e.g. a huge grid/fleet) before any grid or
    # Q-table is allocated, rather than trusting client input.
    grid_width: int = Field(default=_DEFAULT_GRID.get("width", 10), ge=4, le=100)
    grid_height: int = Field(default=_DEFAULT_GRID.get("height", 10), ge=4, le=100)
    obstacle_probability: float = Field(
        default=_DEFAULT_GRID.get("obstacle_probability", 0.15), ge=0.0, le=0.9
    )
    target_count: int = Field(
        default=_DEFAULT_GRID.get("target_a_count", 2) + _DEFAULT_GRID.get("target_b_count", 2),
        ge=1,
        le=100,
    )
    num_agents: int = Field(default=_DEFAULT_FLEET.num_agents, ge=1, le=50)
    sensor_range: int = Field(default=_DEFAULT_AGENT.get("sensor_range", 3), ge=1, le=20)
    max_steps: int = Field(default=_DEFAULT_SIMULATION.get("max_steps", 500), ge=1, le=5000)
    num_episodes: int = Field(default=30, ge=1, le=500)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    discount_factor: float = Field(default=0.9, ge=0.0, le=1.0)
    exploration_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    speed_ms: int = Field(default=100, ge=10, le=500)  # delay between steps in ms
    skip_playback: bool = False  # skip per-step animation: run at full speed, stream only per-try results
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

# Pretrained MoE persistence: loaded at startup, re-saved after every Train
# press so the policy keeps improving across sessions/restarts.
_MOE_CKPT_PATH = os.environ.get("MOE_CHECKPOINT", "checkpoints/moe.pt")


def _load_pretrained_moe() -> None:
    global trained_moe_policy, trained_moe_shape, trained_moe_epochs
    try:
        from rescue_sim.MoE.pipeline import load_moe_policy

        loaded = load_moe_policy(_MOE_CKPT_PATH)
    except Exception as exc:  # noqa: BLE001 - missing torch / stale ckpt
        print(f"[MoE] pretrained checkpoint unavailable ({exc})")
        return
    if loaded is not None:
        trained_moe_policy, trained_moe_shape, trained_moe_epochs = loaded
        print(
            f"[MoE] loaded pretrained policy from {_MOE_CKPT_PATH} "
            f"(shape={trained_moe_shape}, epochs={trained_moe_epochs})"
        )


_load_pretrained_moe()


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


# ── Standalone-experts vs. blended-MoE comparison ───────────────────────────

# Fixed labels + brand colors shared with the frontend comparison panel.
_COMPARE_POLICIES = [
    ("Non-AI (APF)", "apf", "#94a3b8"),  # raw potential-fields baseline (no ML)
    ("Expert 1", 0, "#3b82f6"),      # exploration (APF clone)
    ("Expert 2", 1, "#10b981"),      # ensemble coordination
    ("Expert 3", 2, "#f59e0b"),      # GRU fallback / hysteretic
    ("MoE", "moe", "#8b5cf6"),       # attention-router blend
]
_COMPARE_SEED = 20260707  # same grids for every policy => fair comparison


def _policy_mode_actions(policy, mode, obs_t, peer_t, mask_t, hidden):
    """Greedy per-agent actions for one policy 'mode' (expert index 0/1/2 or 'moe').

    Standalone experts bypass the router: encode with the (frozen) expert
    encoder and run the single head. 'moe' runs the full router-blended policy.
    Returns ``(actions[A], hidden)`` — hidden carries the GRU state for the
    recurrent fallback head / the MoE.
    """
    import torch

    if mode == "moe":
        y_final, _weights, hidden = policy(obs_t, peer_t, mask_t, hidden)
        logits = y_final.squeeze(0)                                   # [A, act]
    else:
        obs_flat = obs_t.squeeze(0)                                   # [A, obs]
        peer_count = peer_t.squeeze(0).sum(dim=-1, keepdim=True)      # [A, 1]
        z = policy.expert_encoder(obs_flat, peer_count)               # [A, latent]
        if mode == 0:
            logits = policy.expert_exploration(z)
        elif mode == 1:
            logits = policy.expert_coordination(z)
        else:  # recurrent fallback head
            logits, hidden = policy.expert_fallback(z, hidden)
    logits = logits.masked_fill(~mask_t.squeeze(0), -1e9)
    return torch.argmax(logits, dim=-1), hidden


def _evaluate_policy_mode(policy, settings, num_agents, max_steps, view_radius, mode, episodes,
                          seed=_COMPARE_SEED):
    """Greedy rollouts of one policy mode on a fixed seeded grid sequence."""
    import torch

    from rescue_sim.MAPPO import RescueEnv
    from rescue_sim.MoE.pipeline import build_peer_matrix

    env = RescueEnv(settings, num_agents=num_agents, max_steps=max_steps,
                    view_radius=view_radius, seed=seed)
    successes, rescued, steps = [], [], []
    connected_steps, agent_steps = 0, 0
    total_targets = settings.target_a_count + settings.target_b_count

    # mode "apf": drive the raw non-ML Artificial Potential Fields baseline
    # (E1's teacher) so the panel shows how far no-ML gets on the same grids.
    apf_teacher = None
    if mode == "apf":
        import numpy as np

        from rescue_sim.MoE.pipeline import ExplorationTeacher

        apf_teacher = ExplorationTeacher(np.random.default_rng(seed))

    for _ in range(episodes):
        obs = env.reset()
        if apf_teacher is not None:
            apf_teacher.reset(env)
        hidden = None
        done = False
        info = {"success": False, "rescued": 0, "steps": 0}
        while not done:
            peer_np = build_peer_matrix(env.positions)
            if apf_teacher is not None:
                act_np = apf_teacher.act(env, env.valid_action_mask())
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                peer_t = torch.tensor(peer_np, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.tensor(env.valid_action_mask(), dtype=torch.bool).unsqueeze(0)
                with torch.no_grad():
                    actions, hidden = _policy_mode_actions(policy, mode, obs_t, peer_t, mask_t, hidden)
                act_np = actions.numpy()
            # Connectivity: an agent is "connected" when it has >=1 peer (row sum > 1).
            peer_counts = peer_np.sum(axis=1)
            connected_steps += int((peer_counts > 1).sum())
            agent_steps += num_agents
            obs, _reward, done, info = env.step(act_np)
            if apf_teacher is not None:
                apf_teacher.observe(env)
        successes.append(bool(info["success"]))
        rescued.append(int(info["rescued"]))
        steps.append(int(info["steps"]))

    n = max(len(successes), 1)
    return {
        "success_rate": round(sum(successes) / n, 4),
        "avg_steps": round(sum(steps) / n, 1),
        "avg_rescued": round(sum(rescued) / n, 3),
        "targets": total_targets,
        "peer_connectivity": round(connected_steps / max(agent_steps, 1), 4),
    }


_compare_lock = asyncio.Lock()


@app.get("/api/compare_policies")
async def compare_policies(episodes: int = 30):
    """Compare the three standalone experts against the blended MoE.

    Runs greedy rollouts of Expert 1/2/3 and the full MoE on the *same* seeded
    14×14 grids (using the currently cached trained MoE policy) and returns
    averaged metrics per policy for the frontend comparison dashboard.
    """
    if _compare_lock.locked():
        return JSONResponse(
            status_code=429,
            content={
                "error": "compare_running",
                "message": "A comparison is already running — wait for it to finish.",
            },
        )
    if trained_moe_policy is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "no_trained_moe",
                "message": "Train the Neural MoE first, then run the comparison.",
            },
        )

    episodes = max(3, min(int(episodes), 60))
    cfg = current_config
    target_a = min(cfg.target_count // 2, cfg.target_count)
    target_b = cfg.target_count - target_a
    # New random grid sequence on every Compare press; the same seed is shared
    # by all four policies within a press, so the comparison stays fair.
    compare_seed = random.randint(0, 2**31 - 1)
    settings = GridSettings(
        width=cfg.grid_width, height=cfg.grid_height,
        obstacle_probability=cfg.obstacle_probability,
        target_a_count=target_a, target_b_count=target_b,
        random_seed=compare_seed,
    )
    view_radius = max(1, cfg.sensor_range)
    # Comparison rollouts need far fewer steps than a full training episode;
    # capping keeps the 4-policy sweep interactive on CPU.
    compare_steps = min(cfg.max_steps, 200)

    async with _compare_lock:
        policies = []
        for name, mode, color in _COMPARE_POLICIES:
            metrics = await asyncio.to_thread(
                _evaluate_policy_mode, trained_moe_policy, settings,
                cfg.num_agents, compare_steps, view_radius, mode, episodes,
                compare_seed,
            )
            policies.append({"name": name, "color": color, **metrics})

    # Category winners for the metric cards.
    def _winner(key, better_low=False):
        best = min(policies, key=lambda p: p[key]) if better_low else max(policies, key=lambda p: p[key])
        return {"name": best["name"], "value": best[key], "color": best["color"]}

    return {
        "episodes": episodes,
        "grid": f"{cfg.grid_width}x{cfg.grid_height}",
        "num_agents": cfg.num_agents,
        "policies": policies,
        "winners": {
            "success": _winner("success_rate"),
            "efficiency": _winner("avg_steps", better_low=True),
            "rescued": _winner("avg_rescued"),
            "connectivity": _winner("peer_connectivity"),
        },
    }


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
                try:
                    current_config = SimConfig(**msg.get("data", {}))
                except ValidationError as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Invalid configuration: {exc.error_count()} field(s) out of range.",
                        }
                    )
                    continue
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

                    # Check for cancellation or speed update (throttled when not
                    # animating — instant_train / skip_playback — to avoid latency)
                    check_cancel = True
                    if (run_mode == "instant_train" or config.skip_playback) and step % 20 != 0:
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
                                try:
                                    new_cfg = SimConfig(**cancel_msg.get("data", {}))
                                except ValidationError:
                                    pass  # ignore an invalid mid-run config; keep running
                                else:
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
                        if config.skip_playback:
                            # Results-only mode: no per-step stream, no pacing —
                            # just yield so stop/config messages still get through.
                            await asyncio.sleep(0)
                        else:
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

                # Skipped playback still gets one snapshot per episode so the
                # grid shows the final positions and rescued targets.
                if run_mode != "instant_train" and config.skip_playback and steps:
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
# CARDINAL_ACTIONS order is (UP, DOWN, RIGHT, LEFT): index of each action's reverse.
_REVERSE_ACTION = (1, 0, 3, 2)
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

    # Fresh random grid on every run/restart; tries within one run still share
    # the grid so routing evolution stays comparable try-to-try.
    seed = random.randint(0, 2**31 - 1)
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

        # Persist so restarts (and future sessions) start from this policy.
        try:
            from rescue_sim.MAPPO import RescueEnv as _ObsEnv
            from rescue_sim.MoE.pipeline import save_moe_policy

            _obs_env = _ObsEnv(
                settings, num_agents=config.num_agents,
                max_steps=80, view_radius=view_radius, seed=seed,
            )
            save_moe_policy(
                policy, _MOE_CKPT_PATH, _obs_env.obs_dim,
                view_radius, expected_shape, trained_moe_epochs,
            )
            print(f"[MoE] saved trained policy -> {_MOE_CKPT_PATH}")
        except Exception as exc:  # noqa: BLE001 - saving must never kill a run
            print(f"[MoE] could not save policy ({exc})")

    await websocket.send_json(
        {"type": "moe_status", "trained_epochs": trained_moe_epochs}
    )

    if run_mode == "instant_train":
        # "Train More": accumulate training only — no rollout. Running is the
        # main button's job (evaluate mode, pure playback of the saved policy).
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
            # Expert 2 = trained QMIX+TransfQMix+MAPPO teachers via
            # state-conditioned gated reverse-KL (falls back to the heuristic
            # teacher automatically if the checkpoints don't match the config).
            e2_gated=True,
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

    from rescue_sim.MoE.pipeline import (
        ExplorationMemory,
        FixedGridRescueEnv,
        build_peer_matrix,
    )

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
    rollout_steps = config.max_steps  # honor the scenario's step budget

    # Live Expert 3: a genuine Epidemic Hysteretic Q fleet on THIS grid. The
    # grid is fixed across tries, so tabular Q-learning compounds: the fleet
    # learns from every transition (whoever is in control), gossips Q-tables
    # when agents meet, and keeps its Q across tries — by the later tries it
    # is near-certain on this map. Whenever the router routes an agent to the
    # fallback expert, the live learner acts instead of the frozen GRU clone.
    live_fleet = None
    comms_bus = None
    fleet_ids: list[str] = []
    try:
        from rescue_sim.Qlearning.communications import DefaultCommsBus
        from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning
        from rescue_sim.shared import GossipConfig, HystereticConfig

        live_fleet = EpidemicHystereticQLearning(
            grid,
            HystereticConfig(epsilon=0.3),
            GossipConfig(),
            max_agents=num_agents,
            seed=settings.random_seed or 0,
        )
        fleet_ids = [f"a{i}" for i in range(num_agents)]
        for i, aid in enumerate(fleet_ids):
            live_fleet.add_agent(aid, env.positions[i], fresh=True)
        comms_bus = DefaultCommsBus()
    except Exception as exc:  # noqa: BLE001 - live E3 is an enhancement, not a dependency
        print(f"[MoE] live E3 epidemic fleet unavailable ({exc})")

    # Live E1/E2: real APF and the real TransfQMix checkpoint act whenever the
    # router routes an agent to exploration/coordination — the router decides,
    # the genuine experts move (measured: bare 72% -> 84% on unseen grids).
    from rescue_sim.MoE.live_moe import LiveExperts

    live_experts = LiveExperts.from_checkpoints(
        settings, num_agents, view_radius, config.max_steps,
        seed=settings.random_seed or 0,
    )
    if live_experts is None:
        print("[MoE] live E1/E2 experts unavailable — distilled heads stay in charge")

    # Online scoreboard adaptation (learns WITHIN a run, like E3's Q-table):
    # log-space per-expert routing bias, updated after every try from who
    # actually delivered rescues on THIS grid. E2 starts with a positive
    # prior — it carries the distilled deep-RL knowledge and tends to be
    # under-routed by the generic gate.
    import torch as _torch

    # E2 gets a strong prior (it carries the distilled deep-RL knowledge and
    # the generic gate under-routes it); E1 a slight negative one — dispersal
    # is only valuable early, and the scoreboard restores it if it delivers.
    # bias = prior + mean-centered score: centering keeps trust RELATIVE, so
    # one expert collecting every rescue can't run away to the cap.
    expert_prior = [-0.2, 0.6, 0.0]
    expert_score = [0.0, 0.0, 0.0]
    adapt_bias = list(expert_prior)

    for episode in range(max(1, config.num_episodes)):
        obs = env.reset()  # identical grid every try (fixed competition grid)
        if live_experts is not None:
            live_experts.reset(env)  # re-anchor APF's live-target tracking
        hidden = None      # GRU temporal memory resets per try
        # Anti-revisit memory: penalizes moves onto already-visited cells while
        # no target is in the ego window, so targets far from the start corner
        # are reached instead of the team looping over searched ground.
        memory = ExplorationMemory(num_agents)
        memory.observe(env.positions)
        last_action: list[int | None] = [None] * num_agents
        policy.route_bias = _torch.tensor(adapt_bias, dtype=_torch.float32)
        last_rescue_step = 0
        fleet_base_eps = live_fleet.epsilon if live_fleet is not None else 0.0
        # Rolling routing history per agent: rescue credit goes to the experts
        # that steered the APPROACH (last 8 steps), not just the final step —
        # otherwise whoever holds the wheel at the finish line steals credit.
        recent_dominant: list[list[int]] = [[] for _ in range(num_agents)]
        if live_fleet is not None:
            # Q-tables PERSIST across tries (that's the point of tabular
            # learning on a fixed grid); only positions and exploration reset.
            live_fleet.reset_positions(
                {aid: env.positions[i] for i, aid in enumerate(fleet_ids)}
            )
            live_fleet.decay_epsilon(
                0.3 / max(1, config.num_episodes), floor=0.02
            )
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
        step_message: dict | None = None

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
            # Cancellation / live config (speed) updates (throttled when the
            # per-step animation is skipped, to keep the fast path fast)
            if not config.skip_playback or step % 20 == 1:
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

            valid_np = mask_t.squeeze(0).numpy()
            with torch.no_grad():
                y_final, weights, hidden = policy(obs_t, peer_t, mask_t, hidden)
                # Anti-revisit bias: log-prob scores penalized by per-agent
                # visit counts (only while no target is visible), so the team
                # expands toward unexplored ground and far targets get found.
                logits = memory.bias_logits(
                    env, obs, y_final.squeeze(0), valid_np
                ).clone()                                         # [A, act]

            # No-backtrack rule: mask the exact reverse of each agent's
            # previous move (feed-forward heads have no memory, so without
            # this they ping-pong east/west forever). Skipped when the
            # reverse is the only valid option.
            for i in range(num_agents):
                prev = last_action[i]
                if prev is None:
                    continue
                rev = _REVERSE_ACTION[prev]
                if valid_np[i, rev] and int(valid_np[i].sum()) > 1:
                    logits[i, rev] = -1e9
            actions = torch.argmax(logits, dim=-1)                # greedy [A]

            # Stuck detector: rescued some targets but no progress for 60
            # steps -> the last target is hiding; re-open exploration.
            stuck = (
                len(rescued_seen) > 0
                and len(rescued_seen) < total_targets
                and step - last_rescue_step > 60
            )
            if live_fleet is not None:
                live_fleet.epsilon = 0.25 if stuck else fleet_base_eps

            # Small epsilon keeps tries from being carbon copies of each
            # other and breaks feed-forward oscillation loops (raised while
            # stuck so the team actually widens its search).
            jitter = 0.25 if stuck else 0.1
            act_np = actions.numpy().copy()
            for i in range(num_agents):
                if rollout_rng.random() < jitter:
                    valid = [a for a in range(policy.action_dim) if valid_np[i, a]]
                    if valid:
                        act_np[i] = rollout_rng.choice(valid)

            weights_step = weights.squeeze(0)                    # [A, 3]
            dominant = torch.argmax(weights_step, dim=-1).tolist()

            # Live E1/E2: the router decides, the real experts move — APF on
            # exploration-routed agents, TransfQMix on coordination-routed ones.
            if live_experts is not None:
                proposals = live_experts.actions(env, valid_np)
                act_np = live_experts.apply(act_np, dominant, proposals, valid_np)

            # Live E3 takes over agents the router assigns to the fallback
            # expert: the tabular fleet acts from its (ever-improving) Q-table.
            if live_fleet is not None:
                live_fleet.reset_positions(
                    {aid: env.positions[i] for i, aid in enumerate(fleet_ids)}
                )
                fleet_actions = live_fleet.select_actions()
                for i, aid in enumerate(fleet_ids):
                    if dominant[i] == 2 and aid in fleet_actions:
                        a = int(fleet_actions[aid])
                        # Mission-aware fix for position-keyed tabular Q: the
                        # table still holds +10 routes to targets already
                        # rescued THIS try. If the chosen move walks into a
                        # rescued cell, re-pick the best Q action that doesn't.
                        if live_fleet.peek_next(aid, a) in rescued_seen:
                            slot = live_fleet._id_to_slot[aid]
                            p = env.positions[i]
                            alt, alt_q = None, -float("inf")
                            for b in range(policy.action_dim):
                                if not valid_np[i, b]:
                                    continue
                                if live_fleet.peek_next(aid, b) in rescued_seen:
                                    continue
                                qv = float(live_fleet.q[slot, p.y, p.x, b])
                                if qv > alt_q:
                                    alt, alt_q = b, qv
                            if alt is not None:
                                a = alt
                        if valid_np[i, a]:
                            act_np[i] = a

            gru_norm = torch.norm(hidden.squeeze(0), dim=-1)     # [A]
            peer_counts = peer_np.sum(axis=1).astype(int).tolist()
            explore_sum += float(weights_step[:, 0].mean().item())
            explore_samples += 1
            for d in dominant:
                usage[_MOE_EXPERT_LABELS[d]] += 1
            if prev_dominant is not None:
                switches += sum(1 for a, b in zip(prev_dominant, dominant) if a != b)
            prev_dominant = dominant
            for i in range(num_agents):
                recent_dominant[i].append(dominant[i])
                if len(recent_dominant[i]) > 8:
                    recent_dominant[i].pop(0)

            obs, reward, done, info = env.step(act_np)
            memory.observe(env.positions)
            if live_experts is not None:
                live_experts.observe(env)  # APF drops just-rescued targets
            total_reward += float(reward)
            last_action = [int(a) for a in act_np]

            newly_rescued: set[int] = set()
            for i, pos in enumerate(env.positions):
                if grid.has_target(pos) and pos not in rescued_seen:
                    rescued_seen.add(pos)
                    newly_rescued.add(i)
                    last_rescue_step = step
                    rescued_records.append(
                        {"x": pos.x, "y": pos.y, "step": step, "type": grid.target_type_at(pos)}
                    )
                    window = recent_dominant[i] or [dominant[i]]
                    for d in window:
                        rescues_by_expert[_MOE_EXPERT_LABELS[d]] += 1.0 / len(window)

            # Live E3 learns from EVERY transition (whichever expert drove),
            # then runs one epidemic gossip round over the comms layer — so
            # its map knowledge compounds across steps and tries.
            if live_fleet is not None:
                live_fleet.record_transitions(
                    {aid: int(act_np[i]) for i, aid in enumerate(fleet_ids)},
                    {aid: (10.0 if i in newly_rescued else -0.05)
                     for i, aid in enumerate(fleet_ids)},
                    {aid: env.positions[i] for i, aid in enumerate(fleet_ids)},
                )
                if comms_bus is not None:
                    comms_bus.exchange(live_fleet)

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

            step_message = {
                "type": "step",
                "episode": episode,
                "step": step,
                "agents": agent_states,
                "rescued": rescued_records,
                "active_targets": int(info["targets"] - info["rescued"]),
                "moe": moe_payload,
            }
            if config.skip_playback:
                # Results-only mode: no per-step stream, no pacing — just
                # yield so stop/config messages still get through.
                await asyncio.sleep(0)
            else:
                await websocket.send_json(step_message)
                await asyncio.sleep(config.speed_ms / 1000.0)
            if done:
                break

        if should_stop:
            break

        # Skipped playback still gets one snapshot per try so the grid shows
        # the final positions/rescues and the MoE panel gets routing weights.
        if config.skip_playback and step_message is not None:
            await websocket.send_json(step_message)

        usage_total = max(sum(usage.values()), 1)

        # Scoreboard update: credit experts that steered rescues this try;
        # debit experts that consumed routing share in a failed try. Scores
        # are mean-centered (relative trust) and added onto the fixed priors,
        # feeding back into routing from the next try on.
        for j, name in enumerate(_MOE_EXPERT_LABELS):
            expert_score[j] += 0.25 * rescues_by_expert[name]
            if not info["success"]:
                expert_score[j] -= 0.4 * (usage[name] / usage_total)
        score_mean = sum(expert_score) / len(expert_score)
        expert_score = [s - score_mean for s in expert_score]
        adapt_bias = [
            max(-1.2, min(1.2, expert_prior[j] + expert_score[j]))
            for j in range(len(_MOE_EXPERT_LABELS))
        ]

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
                "rescues_by_expert": {
                    name: round(v, 1) for name, v in rescues_by_expert.items()
                },
                "adaptation_bias": {
                    name: round(adapt_bias[j], 3)
                    for j, name in enumerate(_MOE_EXPERT_LABELS)
                },
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

    # The scoreboard bias is per-run knowledge; never leak it into the cached
    # policy used by Compare / later runs.
    policy.route_bias = None

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
                name: round(
                    sum(m["moe"]["rescues_by_expert"][name] for m in episode_metrics), 1
                )
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
