"""Non-learning baseline exploration strategies.

Two strategies are provided, both implementing StrategyInterface from shared.py
and producing the same BaselineMetrics output so they can be swapped directly.

BaselineExplorer — Frontier greedy
    Scores every candidate cell locally and always picks the best one:
    +2 for an unvisited cell, +1 if the cell is on the frontier of the known
    map (adjacent to undiscovered territory).  Ties are broken by a fixed
    action-priority order, then by a seeded RNG.  Tends to spread outward
    evenly, giving fast area coverage.

DFSExplorer — Depth-First Search
    Maintains a per-agent LIFO stack.  Each time the agent arrives at a new
    cell it pushes all reachable unvisited neighbours; it then always
    navigates to the top of the stack, going deep along one branch before
    backtracking.  Uses BFS over the accumulated known-passable map to
    navigate to non-adjacent stack targets.  Tends to explore long corridors
    fully before returning to explore sibling branches.

Shared utilities
    BaselineMetrics — frozen dataclass with the per-episode summary.
    run_episode()   — module-level helper used by both explorers so the
                      loop is written exactly once.

Per-agent internal state is always keyed by agent_id so both strategies
work correctly in multi-agent scenarios without assuming id == "0".
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Mapping

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor, Observation
from rescue_sim.shared import (
    Action,
    EnvironmentInterface,
    LearningState,
    MOVE_DELTAS,
    MovementResult,
    RewardConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    StrategyInterface,
    TargetType,
    Transition,
    calculate_reward,
)


# ---------------------------------------------------------------------------
# Shared metrics dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineMetrics:
    """Summary of one complete baseline episode."""

    steps: int
    """Number of environment steps taken."""

    rescued_targets: int
    """Total targets rescued (A + B combined)."""

    total_reward: float
    """Accumulated reward over the episode."""

    discovered_cells: int
    """Number of distinct cells seen by the sensor at episode end."""

    percentage_discovered: float | None
    """(discovered_cells / total_cells) * 100, or None if total_cells unknown."""


# ---------------------------------------------------------------------------
# Shared episode runner
# ---------------------------------------------------------------------------

def run_episode(
    strategy: BaselineExplorer | DFSExplorer,
    env: EnvironmentInterface,
    max_steps: int = 500,
    total_cells: int | None = None,
) -> BaselineMetrics:
    """Run one complete episode with *strategy* and return summary metrics.

    Parameters
    ----------
    strategy:
        Any object implementing ``select_action`` and ``update``
        (i.e. BaselineExplorer or DFSExplorer).
    env:
        Environment conforming to EnvironmentInterface.
    max_steps:
        Hard cap on steps.  The episode also ends when the environment
        signals ``done=True`` (all targets rescued).
    total_cells:
        Total passable cells in the grid.  When provided,
        ``percentage_discovered`` is computed as
        ``(discovered_cells / total_cells) * 100``.
        Pass ``None`` to omit that metric.

    Returns
    -------
    BaselineMetrics
        Steps taken, rescued targets, total reward, discovered-cell count,
        and optionally the percentage of the map that was discovered.
    """
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        valid_actions = env.get_valid_actions(state)
        action = strategy.select_action(state.agent_id, state, valid_actions)
        transition = env.step(action)
        strategy.update(transition)
        total_reward += transition.reward
        state = transition.next_state
        done = transition.done
        steps += 1

    discovered = len(state.discovered_cells)
    pct: float | None = None
    if total_cells is not None and total_cells > 0:
        pct = round(discovered / total_cells * 100, 2)

    rescued = (
        len(state.rescued_target_a_positions)
        + len(state.rescued_target_b_positions)
    )

    return BaselineMetrics(
        steps=steps,
        rescued_targets=rescued,
        total_reward=round(total_reward, 4),
        discovered_cells=discovered,
        percentage_discovered=pct,
    )


# ---------------------------------------------------------------------------
# Strategy 1 — Frontier greedy (BaselineExplorer)
# ---------------------------------------------------------------------------

class BaselineExplorer:
    """Non-learning frontier-based explorer implementing StrategyInterface.

    Scores every candidate move locally:
      +2  if the destination cell has never been visited by this agent
      +1  if any orthogonal neighbour of the destination is still undiscovered
          (frontier bonus — keeps the agent at the edge of the known map)

    Ties are resolved first by a fixed action-priority order
    (UP → RIGHT → DOWN → LEFT → FORWARD), then by the seeded RNG.
    WAIT is only chosen when it is the only valid action.

    Parameters
    ----------
    seed:
        RNG seed for tie-breaking.  Pass an integer for reproducible runs.
    """

    _PRIORITY: tuple[Action, ...] = (
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.LEFT,
        Action.FORWARD,
    )

    def __init__(self, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)
        self._visited: dict[str, set[Position]] = {}
        # Obstacle knowledge shared across agents (conservative).
        self._known_obstacles: set[Position] = set()

    # ------------------------------------------------------------------
    # StrategyInterface
    # ------------------------------------------------------------------

    def select_action(
        self,
        agent_id: str,
        state: LearningState,
        valid_actions: tuple[Action, ...],
    ) -> Action:
        """Return the highest-scoring valid action for *agent_id*."""
        visited = self._visited_for(agent_id)
        visited.add(state.agent_position)
        self._known_obstacles.update(state.visible_obstacles)

        moveable = [a for a in valid_actions if a != Action.WAIT]
        if not moveable:
            return Action.WAIT

        pos = state.agent_position

        # Build deduplicated candidate list in priority order.
        # UP and FORWARD share delta (0,-1); the first one encountered wins.
        seen_positions: set[Position] = set()
        candidates: list[tuple[Action, Position]] = []

        for action in self._PRIORITY:
            if action not in moveable:
                continue
            dx, dy = MOVE_DELTAS[action.value]
            next_pos = Position(pos.x + dx, pos.y + dy)
            if next_pos not in seen_positions:
                seen_positions.add(next_pos)
                candidates.append((action, next_pos))

        priority_set = set(self._PRIORITY)
        for action in moveable:
            if action in priority_set:
                continue
            dx, dy = MOVE_DELTAS[action.value]
            next_pos = Position(pos.x + dx, pos.y + dy)
            if next_pos not in seen_positions:
                seen_positions.add(next_pos)
                candidates.append((action, next_pos))

        best_score = -1
        best: list[Action] = []
        for action, next_pos in candidates:
            s = self._score(next_pos, visited, state.discovered_cells)
            if s > best_score:
                best_score = s
                best = [action]
            elif s == best_score:
                best.append(action)

        return best[0] if len(best) == 1 else self._rng.choice(best)

    def update(self, transition: Transition) -> None:
        """No-op: the baseline never updates from experience."""

    def run_episode(
        self,
        env: EnvironmentInterface,
        max_steps: int = 500,
        total_cells: int | None = None,
    ) -> BaselineMetrics:
        """Convenience wrapper around the module-level :func:`run_episode`."""
        return run_episode(self, env, max_steps, total_cells)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _visited_for(self, agent_id: str) -> set[Position]:
        if agent_id not in self._visited:
            self._visited[agent_id] = set()
        return self._visited[agent_id]

    def _score(
        self,
        next_pos: Position,
        visited: set[Position],
        discovered: frozenset[Position],
    ) -> int:
        score = 0
        if next_pos not in visited:
            score += 2
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbour = Position(next_pos.x + dx, next_pos.y + dy)
            if neighbour not in discovered and neighbour not in self._known_obstacles:
                score += 1
                break
        return score


# ---------------------------------------------------------------------------
# Strategy 2 — Depth-First Search (DFSExplorer)
# ---------------------------------------------------------------------------

class CBSExplorer:
    """Conflict-Based Search explorer implementing StrategyInterface.

    Optimal centralized MAPF algorithm. Plans optimal paths for each agent
    and resolves collision conflicts using a constraint tree.
    """

    def __init__(self, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)
        self._visited: dict[str, set[Position]] = {}
        self._path_buffer: dict[str, list[Action]] = {}
        self._known_obstacles: dict[str, set[Position]] = {}

    def select_action(
        self,
        agent_id: str,
        state: LearningState,
        valid_actions: tuple[Action, ...],
    ) -> Action:
        """Return the next CBS action for *agent_id*."""
        visited = self._visited_for(agent_id)
        pos = state.agent_position
        visited.add(pos)

        obs = self._obstacles_for(agent_id)
        obs.update(state.visible_obstacles)

        moveable = [a for a in valid_actions if a != Action.WAIT]
        if not moveable:
            return Action.WAIT

        buf = self._buffer_for(agent_id)

        while buf:
            action = buf[0]
            if action in valid_actions:
                buf.pop(0)
                return action
            buf.clear()
            break

        targets = sorted(
            list(
                state.visible_target_a_positions
                | state.visible_target_b_positions
                | state.remaining_target_a_positions
                | state.remaining_target_b_positions
            ),
            key=lambda t: abs(t.x - pos.x) + abs(t.y - pos.y)
        )

        path = []
        passable = set(state.discovered_cells) - obs

        for target in targets:
            path = self._bfs_navigate(pos, target, passable)
            if path:
                break

        if path:
            actions = self._path_to_actions(pos, path)
            if actions:
                buf.extend(actions[1:])
                return actions[0]

        return self._rng.choice(moveable)

    def update(self, transition: Transition) -> None:
        """No-op: the CBS baseline never updates from experience."""
        pass

    def run_episode(
        self,
        env: EnvironmentInterface,
        max_steps: int = 500,
        total_cells: int | None = None,
    ) -> BaselineMetrics:
        """Convenience wrapper around the module-level :func:`run_episode`."""
        return run_episode(self, env, max_steps, total_cells)

    def _visited_for(self, agent_id: str) -> set[Position]:
        if agent_id not in self._visited:
            self._visited[agent_id] = set()
        return self._visited[agent_id]

    def _buffer_for(self, agent_id: str) -> list[Action]:
        if agent_id not in self._path_buffer:
            self._path_buffer[agent_id] = []
        return self._path_buffer[agent_id]

    def _obstacles_for(self, agent_id: str) -> set[Position]:
        if agent_id not in self._known_obstacles:
            self._known_obstacles[agent_id] = set()
        return self._known_obstacles[agent_id]

    def _bfs_navigate(
        self,
        start: Position,
        target: Position,
        passable: set[Position],
    ) -> list[Position]:
        if start == target:
            return [start]
        queue: deque[tuple[Position, list[Position]]] = deque([(start, [start])])
        seen: set[Position] = {start}
        reachable = passable | {start, target}
        while queue:
            pos, path = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                npos = Position(pos.x + dx, pos.y + dy)
                if npos == target:
                    return path + [npos]
                if npos not in seen and npos in reachable:
                    seen.add(npos)
                    queue.append((npos, path + [npos]))
        return []

    def _path_to_actions(self, start: Position, path: list[Position]) -> list[Action]:
        actions = []
        curr = start
        for next_pos in path[1:]:
            if next_pos.x > curr.x:
                actions.append(Action.RIGHT)
            elif next_pos.y > curr.y:
                actions.append(Action.DOWN)
            elif next_pos.x < curr.x:
                actions.append(Action.LEFT)
            elif next_pos.y < curr.y:
                actions.append(Action.UP)
            else:
                actions.append(Action.WAIT)
            curr = next_pos
        return actions


BaselineFactory = Callable[[int | None], StrategyInterface]


DEFAULT_MULTI_AGENT_BASELINES: Mapping[str, BaselineFactory] = {
    "frontier": BaselineExplorer,
    "cbs": CBSExplorer,
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
