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
from dataclasses import replace
import json
import math
import os
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
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.Qlearning.baseline import (
    BaselineExplorer,
    DFSExplorer,
    PrioritizedPlanningExplorer,
    CBSExplorer,
    ICBSExplorer,
    ECBSExplorer,
    MStarExplorer,
)
from rescue_sim.Qlearning.q_learning import QLearningAgent
from rescue_sim.shared import (
    Action,
    LearningState,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    TargetType,
    Transition,
    calculate_reward,
    StrategyInterface,
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
class SimConfig(BaseModel):
    grid_width: int = 10
    grid_height: int = 10
    obstacle_probability: float = 0.15
    target_count: int = 4
    num_agents: int = 1
    sensor_range: int = 3
    max_steps: int = 500
    num_episodes: int = 80
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 1.0
    speed_ms: int = 100  # delay between steps in ms
    run_mode: str = "train"  # train | evaluate | instant_train


# ── Global state ────────────────────────────────────────────────────────────
current_config = SimConfig()
trained_visual_learner: QLearningAgent | None = None
trained_visual_seed: int | None = None


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


def _random_start_from_seed(config: SimConfig, seed: int) -> Position:
    rng = random.Random(seed + 50_000)
    return Position(
        rng.randrange(config.grid_width),
        rng.randrange(config.grid_height),
    )


# ── WebSocket simulation stream ────────────────────────────────────────────
@app.websocket("/ws/simulation")
async def simulation_ws(websocket: WebSocket):
    await websocket.accept()
    global current_config, trained_visual_learner, trained_visual_seed

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

            episode_metrics: list[dict] = []
            should_stop = False
            scenario_seed = random.randint(0, 999999)
            if run_mode == "evaluate":
                if trained_visual_learner is None or trained_visual_seed is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "No trained policy is available yet. Run Train first.",
                        }
                    )
                    continue
                scenario_seed = trained_visual_seed
                learner = trained_visual_learner
                learner.epsilon = 0.0
            else:
                learner = QLearningAgent(
                    learning_rate=config.learning_rate,
                    discount_factor=config.discount_factor,
                    epsilon=config.exploration_rate,
                    rng=random.Random(0),
                )

            total_episodes = 1 if run_mode == "evaluate" else config.num_episodes
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

                start_pos = _random_start_from_seed(config, seed)

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
                targets = []
                for p in grid.target_a_positions:
                    targets.append({"x": p.x, "y": p.y, "type": "A"})
                for p in grid.target_b_positions:
                    targets.append({"x": p.x, "y": p.y, "type": "B"})

                movement = MovementModel()
                sensor = CentralSensor(grid)
                position = start_pos
                found_targets: set[Position] = set()
                active_targets = set(grid.target_a_positions) | set(grid.target_b_positions)
                rescued: list[dict] = []
                visited_positions = {start_pos}
                total_reward = 0.0
                observation = sensor.observe("agent-1", position, config.sensor_range)

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
                            "agents": [
                                {"x": start_pos.x, "y": start_pos.y, "id": 0}
                            ],
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

                    state = _visual_learning_state(
                        learner=learner,
                        observation=observation,
                        grid=grid,
                        found_targets=frozenset(found_targets),
                    )
                    valid_actions = learner.valid_actions(movement, grid, position)
                    valid_actions = _movement_actions_first(valid_actions)
                    if not valid_actions:
                        valid_actions = (Action.WAIT,)

                    action = learner.choose_action(state, valid_actions)
                    movement_result = movement.apply(grid, position, action.value)
                    next_position = movement_result.end
                    next_observation = sensor.observe(
                        "agent-1",
                        next_position,
                        config.sensor_range,
                    )

                    target_type = grid.target_type_at(next_position)
                    rescued_target_type = None
                    if target_type is not None and next_position not in found_targets:
                        found_targets.add(next_position)
                        active_targets.discard(next_position)
                        rescued_target_type = TargetType(target_type)
                        rescued.append(
                            {
                                "x": next_position.x,
                                "y": next_position.y,
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
                            repeated_cell=next_position in visited_positions,
                        ),
                        learner.reward_config,
                    )
                    total_reward += reward

                    next_state = _visual_learning_state(
                        learner=learner,
                        observation=next_observation,
                        grid=grid,
                        found_targets=frozenset(found_targets),
                    )
                    next_valid_actions = learner.valid_actions(movement, grid, next_position)
                    next_valid_actions = _movement_actions_first(next_valid_actions)
                    if not next_valid_actions:
                        next_valid_actions = (Action.WAIT,)
                    if run_mode in {"train", "instant_train"}:
                        learner.update_q_value(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            next_valid_actions=next_valid_actions,
                        )

                    position = next_position
                    observation = next_observation
                    visited_positions.add(position)
                    agent_state = {
                        "id": 0,
                        "x": position.x,
                        "y": position.y,
                        "action": action.value,
                        "reward": round(reward, 2),
                    }
                    steps = step + 1

                    if run_mode != "instant_train":
                        await websocket.send_json(
                            {
                                "type": "step",
                                "episode": episode,
                                "step": steps,
                                "agents": [agent_state],
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
                    "exploration_rate": round(learner.epsilon, 4),
                }
                episode_metrics.append(metric)
                if run_mode in {"train", "instant_train"}:
                    learner.epsilon = max(0.05, learner.epsilon * 0.85)

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

            if not should_stop:
                if run_mode in {"train", "instant_train"}:
                    trained_visual_learner = learner
                    trained_visual_seed = scenario_seed
                if run_mode == "evaluate" and episode_metrics:
                    await websocket.send_json(
                        {
                            "type": "baseline_comparison",
                            "report": _build_run_comparison_report(
                                grid=grid,
                                start=start_pos,
                                config=config,
                                trained_metric=episode_metrics[-1],
                                trained_explored_cells=len(visited_positions),
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
    except Exception:
        # Prevent uncaught errors from crashing the server
        try:
            await websocket.close()
        except Exception:
            pass


def _visual_learning_state(
    learner: QLearningAgent,
    observation: object,
    grid: object,
    found_targets: frozenset[Position],
) -> LearningState:
    """Build a reusable visual-training state for the Q-table.

    The learner's state builder includes ``steps_taken`` for terminal checks in
    offline training. The live visualization already checks max steps in its
    loop, so keeping the step count in the Q-table would make the same cell look
    like a different state at every time step and hide learning across episodes.
    """

    state = learner.state_from_observation(
        observation=observation,
        grid=grid,
        found_targets=found_targets,
        steps_taken=0,
    )
    return replace(state, steps_taken=0)


def _movement_actions_first(actions: tuple[Action, ...]) -> tuple[Action, ...]:
    """Avoid no-op exploration in the live demo unless the agent is stuck."""

    moving_actions = tuple(action for action in actions if action != Action.WAIT)
    return moving_actions or actions


def _build_run_comparison_report(
    grid: object,
    start: Position,
    config: SimConfig,
    trained_metric: dict,
    trained_explored_cells: int,
) -> dict:
    frontier = _run_baseline_on_visual_grid(
        grid=grid,
        start=start,
        config=config,
        strategy=BaselineExplorer(seed=0),
        agent_name="Frontier Greedy",
    )
    bfs = _run_baseline_on_visual_grid(
        grid=grid,
        start=start,
        config=config,
        strategy=DFSExplorer(seed=0),
        agent_name="DFS Explorer",
    )
    prioritized = _run_baseline_on_visual_grid(
        grid=grid,
        start=start,
        config=config,
        strategy=PrioritizedPlanningExplorer(seed=0),
        agent_name="Prioritized Planning",
    )
    cbs = _run_baseline_on_visual_grid(
        grid=grid,
        start=start,
        config=config,
        strategy=CBSExplorer(seed=0),
        agent_name="CBS",
    )
    icbs = _run_baseline_on_visual_grid(
        grid=grid,
        start=start,
        config=config,
        strategy=ICBSExplorer(seed=0),
        agent_name="ICBS",
    )
    ecbs = _run_baseline_on_visual_grid(
        grid=grid,
        start=start,
        config=config,
        strategy=ECBSExplorer(seed=0),
        agent_name="ECBS",
    )
    mstar = _run_baseline_on_visual_grid(
        grid=grid,
        start=start,
        config=config,
        strategy=MStarExplorer(seed=0),
        agent_name="M*",
    )
    trained = _metric_to_comparison_row(
        agent_name="Q-Learning Model",
        metric=trained_metric,
        explored_cells=trained_explored_cells,
        explorable_cells=_explorable_cell_count(grid),
    )

    return {
        "aggregates": [frontier, bfs, cbs, icbs, ecbs, prioritized, mstar, trained],
        "sprint_demo_summary": (
            "Baseline comparison for the latest Run Learned Policy execution.\n"
            f"CBS: success={cbs['success_rate']:.2f}, steps={cbs['average_steps']:.1f}, reward={cbs['average_accumulated_reward']:.1f}\n"
            f"ICBS: success={icbs['success_rate']:.2f}, steps={icbs['average_steps']:.1f}, reward={icbs['average_accumulated_reward']:.1f}\n"
            f"ECBS: success={ecbs['success_rate']:.2f}, steps={ecbs['average_steps']:.1f}, reward={ecbs['average_accumulated_reward']:.1f}\n"
            f"Prioritized: success={prioritized['success_rate']:.2f}, steps={prioritized['average_steps']:.1f}, reward={prioritized['average_accumulated_reward']:.1f}\n"
            f"M*: success={mstar['success_rate']:.2f}, steps={mstar['average_steps']:.1f}, reward={mstar['average_accumulated_reward']:.1f}\n"
            f"Q-Learning: success={trained['success_rate']:.2f}, steps={trained['average_steps']:.1f}, reward={trained['average_accumulated_reward']:.1f}"
        ),
    }


def _run_baseline_on_visual_grid(
    grid: object,
    start: Position,
    config: SimConfig,
    strategy: StrategyInterface,
    agent_name: str,
) -> dict:
    movement = MovementModel()
    sensor = CentralSensor(grid)
    position = start
    visited = {start}
    found_targets: set[Position] = set()
    all_targets = set(grid.target_a_positions) | set(grid.target_b_positions)
    total_reward = 0.0
    steps = 0
    observation = sensor.observe("baseline", position, config.sensor_range)

    for step in range(config.max_steps):
        if found_targets == all_targets:
            break

        state = _baseline_learning_state(
            observation=observation,
            grid=grid,
            discovered_cells=sensor.discovered_cells,
            found_targets=frozenset(found_targets),
            steps_taken=step,
        )
        valid_actions = tuple(Action(move) for move in movement.allowed_moves(grid, position))
        if not valid_actions:
            valid_actions = (Action.WAIT,)

        action = strategy.select_action("baseline", state, valid_actions)
        movement_result = movement.apply(grid, position, action.value)
        next_position = movement_result.end
        next_observation = sensor.observe("baseline", next_position, config.sensor_range)

        repeated_cell = next_position in visited
        newly_discovered_cells = 0
        if next_position not in visited:
            visited.add(next_position)
            newly_discovered_cells = 1

        rescued_target_type = None
        target_type = grid.target_type_at(next_position)
        if target_type is not None and next_position not in found_targets:
            found_targets.add(next_position)
            rescued_target_type = TargetType(target_type)

        step_reward = calculate_reward(
            RewardEvent(
                moved=movement_result.moved,
                move=action.value,
                newly_discovered_cells=len(next_observation.newly_discovered_cells)
                or newly_discovered_cells,
                rescued_target_type=rescued_target_type,
                completed_episode=found_targets == all_targets,
                repeated_cell=repeated_cell,
            ),
            SPRINT3_REWARD_CONFIG,
        )
        total_reward += step_reward

        next_state = _baseline_learning_state(
            observation=next_observation,
            grid=grid,
            discovered_cells=sensor.discovered_cells,
            found_targets=frozenset(found_targets),
            steps_taken=step + 1,
        )
        strategy.update(
            Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=step_reward,
                done=found_targets == all_targets,
                movement=movement_result,
                observation=next_observation,
            )
        )

        position = next_position
        observation = next_observation
        steps = step + 1

    del observation
    success = found_targets == all_targets
    metric = {
        "success": success,
        "steps": steps,
        "rescued_count": len(found_targets),
        "target_count": len(all_targets),
        "total_reward": round(total_reward, 2),
    }
    return _metric_to_comparison_row(
        agent_name=agent_name,
        metric=metric,
        explored_cells=len(visited),
        explorable_cells=_explorable_cell_count(grid),
    )


def _baseline_learning_state(
    observation: object,
    grid: object,
    discovered_cells: frozenset[Position],
    found_targets: frozenset[Position],
    steps_taken: int,
) -> LearningState:
    found_target_a = frozenset(
        position for position in found_targets if position in grid.target_a_positions
    )
    found_target_b = frozenset(
        position for position in found_targets if position in grid.target_b_positions
    )
    visible_target_a = frozenset(
        position
        for position, target_type in observation.target_types.items()
        if target_type == "A"
    )
    visible_target_b = frozenset(
        position
        for position, target_type in observation.target_types.items()
        if target_type == "B"
    )

    return LearningState(
        agent_id=observation.agent_id,
        agent_position=observation.agent_position,
        visible_cells=observation.visible_cells,
        visible_obstacles=observation.obstacles,
        visible_target_a_positions=visible_target_a,
        visible_target_b_positions=visible_target_b,
        discovered_cells=discovered_cells,
        discovered_target_a_positions=visible_target_a,
        discovered_target_b_positions=visible_target_b,
        rescued_target_a_positions=found_target_a,
        rescued_target_b_positions=found_target_b,
        remaining_target_a_positions=grid.target_a_positions - found_target_a,
        remaining_target_b_positions=grid.target_b_positions - found_target_b,
        steps_taken=steps_taken,
    )


def _metric_to_comparison_row(
    agent_name: str,
    metric: dict,
    explored_cells: int,
    explorable_cells: int,
) -> dict:
    explored = explored_cells / explorable_cells * 100 if explorable_cells else 0.0
    success = 1.0 if metric["success"] else 0.0
    return {
        "agent_name": agent_name,
        "num_agents": 1,
        "scenario_count": 1,
        "success_rate": round(success, 4),
        "average_steps": float(metric["steps"]),
        "average_accumulated_reward": float(metric["total_reward"]),
        "average_rescued_targets": float(metric["rescued_count"]),
        "average_explored_area_percentage": round(explored, 4),
    }


def _baseline_action(grid: object, position: Position, visited: set[Position]) -> Action:
    for action in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
        next_position = _next_position(position, action)
        if grid.is_valid_position(next_position) and next_position not in visited:
            return action

    for action in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
        if grid.is_valid_position(_next_position(position, action)):
            return action

    return Action.WAIT


def _explorable_cell_count(grid: object) -> int:
    return grid.width * grid.height - len(grid.obstacles)


def _next_position(position: Position, action: Action) -> Position:
    if action == Action.RIGHT:
        return Position(position.x + 1, position.y)
    if action == Action.DOWN:
        return Position(position.x, position.y + 1)
    if action == Action.LEFT:
        return Position(position.x - 1, position.y)
    if action == Action.UP:
        return Position(position.x, position.y - 1)
    return position
