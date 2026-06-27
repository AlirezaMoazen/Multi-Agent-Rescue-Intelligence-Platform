"""Multi-agent adapters for the non-ML baseline strategies.

This module keeps the existing baseline algorithms unchanged and runs them in a
shared multi-agent rescue episode. It gives the MARL models a fair classical
comparison point: several agents, shared sensor memory, collision handling, and
team-level metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Mapping

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor, Observation
from rescue_sim.Qlearning.baseline import (
    BaselineExplorer,
    CBSExplorer,
    DFSExplorer,
    ECBSExplorer,
    ICBSExplorer,
    MStarExplorer,
    PrioritizedPlanningExplorer,
)
from rescue_sim.shared import (
    Action,
    LearningState,
    MovementResult,
    RewardConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    StrategyInterface,
    TargetType,
    Transition,
    calculate_reward,
)


BaselineFactory = Callable[[int | None], StrategyInterface]


DEFAULT_MULTI_AGENT_BASELINES: Mapping[str, BaselineFactory] = {
    "frontier": BaselineExplorer,
    "dfs": DFSExplorer,
    "prioritized_planning": PrioritizedPlanningExplorer,
    "cbs": CBSExplorer,
    "icbs": ICBSExplorer,
    "ecbs": ECBSExplorer,
    "mstar": MStarExplorer,
}


@dataclass(frozen=True, slots=True)
class MultiAgentBaselineStep:
    """One synchronized step for all baseline agents."""

    step: int
    actions: dict[str, Action]
    positions: dict[str, Position]
    rewards: dict[str, float]
    rescued_targets: int
    collisions: int


@dataclass(frozen=True, slots=True)
class MultiAgentBaselineMetrics:
    """Report-ready result for one multi-agent non-ML baseline run."""

    strategy_name: str
    num_agents: int
    steps: int
    success: bool
    rescued_targets: int
    total_targets: int
    total_reward: float
    discovered_cells: int
    explored_cells: int
    collisions: int
    invalid_moves: int
    final_positions: dict[str, Position]
    reward_by_agent: dict[str, float]
    trace: tuple[MultiAgentBaselineStep, ...] = field(default_factory=tuple)

    @property
    def success_rate(self) -> float:
        return 1.0 if self.success else 0.0


def run_multi_agent_baseline(
    strategy: StrategyInterface,
    grid: Grid,
    start_positions: Mapping[str, Position],
    max_steps: int,
    sensor_range: int = 3,
    reward_config: RewardConfig = SPRINT3_REWARD_CONFIG,
    strategy_name: str | None = None,
) -> MultiAgentBaselineMetrics:
    """Run one non-ML strategy with several agents on the same grid.

    The strategy object is shared across agents so algorithms with per-agent
    state can coordinate through their own ``agent_id`` keys.
    """
    if not start_positions:
        raise ValueError("at least one agent start position is required")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    _validate_start_positions(grid, start_positions)
    _clear_strategy_state(strategy)

    movement = MovementModel()
    sensor = CentralSensor(grid)
    positions = dict(start_positions)
    visited_by_agent = {
        agent_id: {position}
        for agent_id, position in positions.items()
    }
    explored_cells = set(positions.values())
    rescued: set[Position] = set()
    all_targets = grid.target_a_positions | grid.target_b_positions
    reward_by_agent = {agent_id: 0.0 for agent_id in positions}
    trace: list[MultiAgentBaselineStep] = []
    collisions = 0
    invalid_moves = 0

    for step in range(1, max_steps + 1):
        if rescued == all_targets:
            break

        observations = {
            agent_id: sensor.observe(agent_id, position, sensor_range)
            for agent_id, position in positions.items()
        }
        states = {
            agent_id: _state_from_observation(
                agent_id=agent_id,
                observation=observation,
                sensor=sensor,
                grid=grid,
                rescued=rescued,
                steps_taken=step - 1,
            )
            for agent_id, observation in observations.items()
        }
        actions = {
            agent_id: strategy.select_action(
                agent_id,
                states[agent_id],
                _valid_actions(movement, grid, positions[agent_id]),
            )
            for agent_id in sorted(positions)
        }
        movements, step_collisions = _resolve_movements(movement, grid, positions, actions)
        collisions += step_collisions

        next_positions: dict[str, Position] = {}
        step_rewards: dict[str, float] = {}
        for agent_id in sorted(positions):
            result = movements[agent_id]
            next_position = result.end
            next_positions[agent_id] = next_position
            explored_cells.add(next_position)

            newly_rescued_type = None
            target_type = grid.target_type_at(next_position)
            if target_type is not None and next_position not in rescued:
                rescued.add(next_position)
                newly_rescued_type = TargetType(target_type)

            next_observation = sensor.observe(agent_id, next_position, sensor_range)
            repeated_cell = next_position in visited_by_agent[agent_id]
            visited_by_agent[agent_id].add(next_position)
            done = rescued == all_targets
            reward = calculate_reward(
                RewardEvent(
                    moved=result.moved,
                    move=actions[agent_id].value,
                    newly_discovered_cells=len(next_observation.newly_discovered_cells),
                    rescued_target_type=newly_rescued_type,
                    completed_episode=done,
                    repeated_cell=repeated_cell,
                ),
                reward_config,
            )
            reward_by_agent[agent_id] += reward
            step_rewards[agent_id] = round(reward, 4)
            if not result.moved and actions[agent_id] != Action.WAIT:
                invalid_moves += 1

            next_state = _state_from_observation(
                agent_id=agent_id,
                observation=next_observation,
                sensor=sensor,
                grid=grid,
                rescued=rescued,
                steps_taken=step,
            )
            strategy.update(
                Transition(
                    state=states[agent_id],
                    action=actions[agent_id],
                    next_state=next_state,
                    reward=reward,
                    done=done,
                    movement=result,
                    observation=next_observation,
                )
            )

        positions = next_positions
        trace.append(
            MultiAgentBaselineStep(
                step=step,
                actions=actions,
                positions=dict(positions),
                rewards=step_rewards,
                rescued_targets=len(rescued),
                collisions=step_collisions,
            )
        )

    return MultiAgentBaselineMetrics(
        strategy_name=strategy_name or strategy.__class__.__name__,
        num_agents=len(start_positions),
        steps=len(trace),
        success=rescued == all_targets,
        rescued_targets=len(rescued),
        total_targets=len(all_targets),
        total_reward=round(sum(reward_by_agent.values()), 4),
        discovered_cells=len(sensor.discovered_cells),
        explored_cells=len(explored_cells),
        collisions=collisions,
        invalid_moves=invalid_moves,
        final_positions=dict(positions),
        reward_by_agent={agent_id: round(reward, 4) for agent_id, reward in reward_by_agent.items()},
        trace=tuple(trace),
    )


def compare_multi_agent_baselines(
    grid_settings: GridSettings,
    num_agents: int,
    max_steps: int,
    sensor_range: int = 3,
    seed: int | None = None,
    baseline_factories: Mapping[str, BaselineFactory] = DEFAULT_MULTI_AGENT_BASELINES,
) -> dict[str, MultiAgentBaselineMetrics]:
    """Run all configured non-ML baselines on one shared multi-agent scenario."""
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")

    anchor = Position(0, 0)
    grid = generate_grid(grid_settings, start=anchor)
    starts = default_start_positions(grid, num_agents, anchor)
    results: dict[str, MultiAgentBaselineMetrics] = {}

    for index, (name, factory) in enumerate(baseline_factories.items()):
        strategy_seed = None if seed is None else seed + index
        strategy = factory(strategy_seed)
        results[name] = run_multi_agent_baseline(
            strategy=strategy,
            grid=grid,
            start_positions=starts,
            max_steps=max_steps,
            sensor_range=sensor_range,
            strategy_name=name,
        )

    return results


def default_start_positions(
    grid: Grid,
    num_agents: int,
    anchor: Position = Position(0, 0),
) -> dict[str, Position]:
    """Choose deterministic, reachable, non-target starts for a multi-agent run."""
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if not grid.is_valid_position(anchor):
        raise ValueError("anchor must be a valid grid position")

    targets = grid.target_a_positions | grid.target_b_positions
    reachable = _reachable_positions(grid, anchor)
    preferred = (
        anchor,
        Position(grid.width - 1, grid.height - 1),
        Position(grid.width - 1, 0),
        Position(0, grid.height - 1),
    )
    ordered: list[Position] = []
    for position in preferred + tuple(sorted(reachable, key=lambda pos: (pos.y, pos.x))):
        if position in ordered or position not in reachable or position in targets:
            continue
        ordered.append(position)

    if len(ordered) < num_agents:
        raise ValueError("not enough reachable free cells for all agents")

    return {
        f"agent-{index}": ordered[index]
        for index in range(num_agents)
    }


def _state_from_observation(
    agent_id: str,
    observation: Observation,
    sensor: CentralSensor,
    grid: Grid,
    rescued: set[Position],
    steps_taken: int,
) -> LearningState:
    discovered_targets = sensor.discovered_targets
    discovered_a = frozenset(
        position
        for position, target_type in discovered_targets.items()
        if target_type == "A"
    )
    discovered_b = frozenset(
        position
        for position, target_type in discovered_targets.items()
        if target_type == "B"
    )
    visible_a = frozenset(
        position
        for position, target_type in observation.target_types.items()
        if target_type == "A"
    )
    visible_b = frozenset(
        position
        for position, target_type in observation.target_types.items()
        if target_type == "B"
    )
    rescued_a = frozenset(position for position in rescued if position in grid.target_a_positions)
    rescued_b = frozenset(position for position in rescued if position in grid.target_b_positions)

    return LearningState(
        agent_id=agent_id,
        agent_position=observation.agent_position,
        visible_cells=observation.visible_cells,
        visible_obstacles=observation.obstacles,
        visible_target_a_positions=visible_a,
        visible_target_b_positions=visible_b,
        discovered_cells=sensor.discovered_cells,
        discovered_target_a_positions=discovered_a,
        discovered_target_b_positions=discovered_b,
        rescued_target_a_positions=rescued_a,
        rescued_target_b_positions=rescued_b,
        remaining_target_a_positions=grid.target_a_positions - rescued_a,
        remaining_target_b_positions=grid.target_b_positions - rescued_b,
        steps_taken=steps_taken,
    )


def _valid_actions(
    movement: MovementModel,
    grid: Grid,
    position: Position,
) -> tuple[Action, ...]:
    actions: list[Action] = []
    for move in movement.allowed_moves(grid, position):
        try:
            actions.append(Action(move))
        except ValueError:
            continue
    return tuple(actions) or (Action.WAIT,)


def _validate_start_positions(grid: Grid, start_positions: Mapping[str, Position]) -> None:
    seen: set[Position] = set()
    targets = grid.target_a_positions | grid.target_b_positions

    for agent_id, position in start_positions.items():
        if not grid.is_valid_position(position):
            raise ValueError(f"invalid start position for {agent_id}")
        if position in targets:
            raise ValueError(f"start position for {agent_id} overlaps a target")
        if position in seen:
            raise ValueError("agent start positions must be unique")
        seen.add(position)


def _resolve_movements(
    movement: MovementModel,
    grid: Grid,
    positions: Mapping[str, Position],
    actions: Mapping[str, Action],
) -> tuple[dict[str, MovementResult], int]:
    current_positions = set(positions.values())
    reserved_positions: set[Position] = set()
    movements: dict[str, MovementResult] = {}
    collisions = 0

    for agent_id in sorted(positions):
        action = actions[agent_id]
        result = movement.apply(grid, positions[agent_id], action.value)
        wants_occupied_cell = result.end in current_positions and result.end != positions[agent_id]
        wants_reserved_cell = result.end in reserved_positions

        if result.moved and (wants_occupied_cell or wants_reserved_cell):
            collisions += 1
            result = MovementResult(
                start=positions[agent_id],
                requested=result.requested,
                end=positions[agent_id],
                move=action.value,
                moved=False,
                reason="collision",
            )

        movements[agent_id] = result
        reserved_positions.add(result.end)

    return movements, collisions


def _reachable_positions(grid: Grid, start: Position) -> set[Position]:
    queue: deque[Position] = deque([start])
    seen = {start}

    while queue:
        position = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            neighbor = Position(position.x + dx, position.y + dy)
            if neighbor in seen or not grid.is_valid_position(neighbor):
                continue
            seen.add(neighbor)
            queue.append(neighbor)

    return seen


def _clear_strategy_state(strategy: StrategyInterface) -> None:
    clear = getattr(strategy, "clear_reservations", None)
    if callable(clear):
        clear()
