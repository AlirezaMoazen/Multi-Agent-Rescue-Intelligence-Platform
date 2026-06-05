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
import random
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Position
from rescue_sim.environment.helper import EnvironmentHelper

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
_THIS_DIR = Path(__file__).resolve().parent
_FRONTEND_DIST = _THIS_DIR / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount(
        "/app",
        StaticFiles(directory=str(_FRONTEND_DIST), html=True),
        name="frontend",
    )



# ── Pydantic models for config ─────────────────────────────────────────────
class SimConfig(BaseModel):
    grid_width: int = 20
    grid_height: int = 20
    obstacle_probability: float = 0.15
    target_count: int = 4
    num_agents: int = 1
    sensor_range: int = 3
    max_steps: int = 500
    num_episodes: int = 50
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 1.0
    speed_ms: int = 100  # delay between steps in ms


# ── Global state ────────────────────────────────────────────────────────────
current_config = SimConfig()


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
    global current_config

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

            episode_metrics: list[dict] = []
            should_stop = False

            for episode in range(config.num_episodes):
                if should_stop:
                    break

                seed = random.randint(0, 999999)

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

                start_pos = Position(0, 0)

                try:
                    grid = generate_grid(settings, start_pos)
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

                # Build JSON-serializable obstacle & target lists
                obstacles = [{"x": p.x, "y": p.y} for p in grid.obstacles]
                all_targets = (
                    list(grid.target_a_positions) + list(grid.target_b_positions)
                )
                targets = [{"x": p.x, "y": p.y} for p in all_targets]

                # Create our EnvironmentHelper to run sensors, movement, and grid logic
                helper = EnvironmentHelper(grid, start_pos, config.sensor_range)

                # Send initial state
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
                            {"x": start_pos.x, "y": start_pos.y, "id": 0}
                        ],
                    }
                )

                steps = 0

                for step in range(config.max_steps):
                    if not helper.has_active_targets():
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
                            config = new_cfg
                    except asyncio.TimeoutError:
                        pass

                    # Step the environment using the helper
                    agent_state = helper.step(step)
                    steps = step + 1

                    await websocket.send_json(
                        {
                            "type": "step",
                            "episode": episode,
                            "step": steps,
                            "agents": [agent_state],
                            "rescued": helper.get_rescued_list(),
                            "active_targets": helper.get_active_targets_count(),
                        }
                    )

                    await asyncio.sleep(config.speed_ms / 1000.0)

                if should_stop:
                    break

                success = not helper.has_active_targets()
                metric = {
                    "episode": episode,
                    "steps": steps,
                    "rescued_count": len(helper.get_rescued_list()),
                    "target_count": config.target_count,
                    "success": success,
                    "total_reward": helper.get_total_reward(),
                    "exploration_rate": round(config.exploration_rate, 4),
                }
                episode_metrics.append(metric)

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

            if not should_stop:
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
    except Exception:
        # Prevent uncaught errors from crashing the server
        try:
            await websocket.close()
        except Exception:
            pass
