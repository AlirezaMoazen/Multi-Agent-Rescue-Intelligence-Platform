"""FastAPI backend for the rescue simulation visualization."""

import asyncio
import json
import random
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Resolve project root reliably (works in Docker, locally, and in any CWD)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[3]  # src/rescue_sim/visualization -> project root
_SHARED_DIR = _PROJECT_ROOT / "shared"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SHARED_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR.parent))

from shared.shared import Grid as SharedGrid, Agent, RLAgent  # noqa: E402

app = FastAPI(title="Rescue Sim Visualization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the built frontend if it exists (for Docker production mode)
_FRONTEND_DIST = _THIS_DIR / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


# ── Pydantic models for config ─────────────────────────────────────────────
class SimConfig(BaseModel):
    grid_width: int = 20
    grid_height: int = 20
    obstacle_probability: float = 0.15
    target_count: int = 4
    num_agents: int = 2
    sensor_range: int = 3
    max_steps: int = 500
    num_episodes: int = 50
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 1.0
    speed_ms: int = 100  # delay between steps in ms


# ── Global state ────────────────────────────────────────────────────────────
current_config = SimConfig()


# ── Grid generation helper ──────────────────────────────────────────────────
def generate_random_grid(config: SimConfig, seed: int | None = None):
    """Generate a grid with random obstacles and targets using shared.py Grid."""
    rng = random.Random(seed)
    grid = SharedGrid(config.grid_width, config.grid_height)

    # Place obstacles
    obstacle_positions = []
    for y in range(config.grid_height):
        for x in range(config.grid_width):
            if (x, y) == (0, 0):
                continue  # keep start open
            if rng.random() < config.obstacle_probability:
                grid.set_wall(x, y)
                obstacle_positions.append({"x": x, "y": y})

    # Pick target positions from non-obstacle, non-start cells
    candidates = []
    for y in range(config.grid_height):
        for x in range(config.grid_width):
            if grid.get_cell(x, y) == 0 and (x, y) != (0, 0):
                candidates.append((x, y))

    if config.target_count > len(candidates):
        raise ValueError("target_count exceeds available cells")

    target_coords = rng.sample(candidates, config.target_count)
    target_positions = []
    for tx, ty in target_coords:
        grid.set_special(tx, ty)
        target_positions.append({"x": tx, "y": ty})

    return grid, obstacle_positions, target_positions


def pick_agent_starts(config: SimConfig, grid: SharedGrid, rng: random.Random):
    """Pick random non-wall starting positions for agents."""
    candidates = []
    for y in range(config.grid_height):
        for x in range(config.grid_width):
            if grid.get_cell(x, y) != 1:
                candidates.append((x, y))
    starts = rng.sample(candidates, min(config.num_agents, len(candidates)))
    return starts


# ── REST endpoints ──────────────────────────────────────────────────────────
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
    global current_config

    try:
        while True:
            # Wait for a "start" or "config" command from frontend
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "config":
                current_config = SimConfig(**msg.get("data", {}))
                await websocket.send_json({"type": "config_ack", "config": current_config.model_dump()})
                continue

            if msg.get("type") != "start":
                continue

            config = current_config

            # ── Validation Safeguards ─────────────────────────────────────────
            total_cells = config.grid_width * config.grid_height
            max_obstacles = int(total_cells * config.obstacle_probability)
            est_available = total_cells - max_obstacles - 2

            if config.grid_width < 4 or config.grid_width > 100 or config.grid_height < 4 or config.grid_height > 100:
                await websocket.send_json({"type": "error", "message": "Grid dimensions must be between 4x4 and 100x100."})
                continue
            if config.obstacle_probability < 0.0 or config.obstacle_probability > 0.9:
                await websocket.send_json({"type": "error", "message": "Obstacle probability must be between 0.0 and 0.9."})
                continue
            if config.target_count < 1:
                await websocket.send_json({"type": "error", "message": "At least 1 target is required."})
                continue
            if config.num_agents < 1 or config.num_agents > 10:
                await websocket.send_json({"type": "error", "message": "Number of agents must be between 1 and 10."})
                continue
            if config.target_count + config.num_agents >= est_available:
                await websocket.send_json({"type": "error", "message": f"Too many targets ({config.target_count}) and agents ({config.num_agents}) for a {config.grid_width}x{config.grid_height} grid with {int(config.obstacle_probability*100)}% obstacles."})
                continue

            actions = ["forward", "down", "left", "right"]
            rl_agents_data = [
                RLAgent(actions, config.learning_rate, config.discount_factor, config.exploration_rate)
                for _ in range(config.num_agents)
            ]

            episode_metrics = []
            should_stop = False

            for episode in range(config.num_episodes):
                if should_stop:
                    break

                seed = random.randint(0, 999999)
                rng = random.Random(seed)

                try:
                    grid, obstacles, targets = generate_random_grid(config, seed)
                    starts = pick_agent_starts(config, grid, rng)
                except ValueError as e:
                    await websocket.send_json({"type": "error", "message": f"Environment generation error: {str(e)}. Try reducing target count or obstacles."})
                    should_stop = True
                    break

                agents = [Agent(sx, sy, grid) for sx, sy in starts]

                # Track which targets are still active
                active_targets = set((t["x"], t["y"]) for t in targets)
                rescued = []

                # Send initial state
                await websocket.send_json({
                    "type": "episode_start",
                    "episode": episode,
                    "grid": {
                        "width": config.grid_width,
                        "height": config.grid_height,
                        "obstacles": obstacles,
                        "targets": targets,
                    },
                    "agents": [{"x": a.x, "y": a.y, "id": i} for i, a in enumerate(agents)],
                })

                total_reward = 0
                steps = 0

                for step in range(config.max_steps):
                    if not active_targets:
                        break

                    # Check for cancellation or speed update
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
                            config = new_cfg  # Update speed_ms mid-run
                    except asyncio.TimeoutError:
                        pass

                    agent_states = []
                    for i, agent in enumerate(agents):
                        rl = rl_agents_data[i]
                        state = agent.sensor.sense_environment(agent.grid)

                        action = rl.choose_action(state)

                        # Execute action
                        moved = getattr(agent, action)()

                        # Calculate reward
                        reward = -0.1  # small penalty per step
                        if not moved:
                            reward = -1.0  # penalty for hitting wall

                        pos = (agent.x, agent.y)
                        if pos in active_targets:
                            reward = 10.0
                            active_targets.discard(pos)
                            rescued.append({"x": pos[0], "y": pos[1], "step": step})

                        total_reward += reward
                        next_state = agent.sensor.sense_environment(agent.grid)
                        rl.learn(state, action, reward, next_state)

                        agent_states.append({
                            "id": i,
                            "x": agent.x,
                            "y": agent.y,
                            "action": action,
                            "reward": round(reward, 2),
                        })

                    steps = step + 1

                    await websocket.send_json({
                        "type": "step",
                        "episode": episode,
                        "step": steps,
                        "agents": agent_states,
                        "rescued": rescued,
                        "active_targets": len(active_targets),
                    })

                    await asyncio.sleep(config.speed_ms / 1000.0)

                if should_stop:
                    break

                # Decay exploration
                for rl in rl_agents_data:
                    rl.epsilon = max(0.01, rl.epsilon * 0.95)

                success = len(active_targets) == 0
                metric = {
                    "episode": episode,
                    "steps": steps,
                    "rescued_count": len(rescued),
                    "target_count": config.target_count,
                    "success": success,
                    "total_reward": round(total_reward, 2),
                    "exploration_rate": round(rl_agents_data[0].epsilon, 4),
                }
                episode_metrics.append(metric)

                success_rate = sum(1 for m in episode_metrics if m["success"]) / len(episode_metrics)

                await websocket.send_json({
                    "type": "episode_end",
                    **metric,
                    "success_rate": round(success_rate, 4),
                    "avg_steps": round(
                        sum(m["steps"] for m in episode_metrics) / len(episode_metrics), 1
                    ),
                })

            if not should_stop:
                await websocket.send_json({
                    "type": "training_complete",
                    "total_episodes": len(episode_metrics),
                    "final_success_rate": round(
                        sum(1 for m in episode_metrics if m["success"]) / max(len(episode_metrics), 1), 4
                    ),
                    "metrics": episode_metrics,
                })

    except WebSocketDisconnect:
        pass
    except Exception:
        # Prevent uncaught errors from crashing the server
        try:
            await websocket.close()
        except Exception:
            pass
