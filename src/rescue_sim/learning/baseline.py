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
from dataclasses import dataclass

from rescue_sim.environment.grid import Position
from rescue_sim.shared import (
    Action,
    EnvironmentInterface,
    LearningState,
    MOVE_DELTAS,
    Transition,
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

class DFSExplorer:
    """Depth-First Search explorer implementing StrategyInterface.

    Maintains a per-agent LIFO stack.  Each time the agent arrives at a new
    cell it pushes all immediately reachable unvisited neighbours; the agent
    always navigates to the top of the stack (going deep before backtracking).

    When the stack target is not directly adjacent the explorer uses BFS over
    the accumulated known-passable map to compute a navigation path, which is
    then buffered and consumed one action at a time.

    Direction exploration order: UP → RIGHT → DOWN → LEFT (first direction
    in the list is pushed last so it ends up on top of the LIFO stack).

    Parameters
    ----------
    seed:
        RNG seed used as a fallback when the stack is exhausted or a target
        is unreachable.  Pass an integer for reproducible runs.
    """

    # Primary directions only; "forward" is an alias for "up" and is omitted
    # to avoid duplicate stack entries.
    _DIRS: tuple[tuple[str, int, int], ...] = (
        ("up",    0, -1),
        ("right", 1,  0),
        ("down",  0,  1),
        ("left", -1,  0),
    )

    def __init__(self, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)
        self._visited: dict[str, set[Position]] = {}
        # LIFO stack of cells to visit next, per agent.
        self._stack: dict[str, list[Position]] = {}
        # Buffered navigation path (actions), per agent.
        self._path_buffer: dict[str, list[Action]] = {}
        # Accumulated obstacle knowledge, per agent.
        self._known_obstacles: dict[str, set[Position]] = {}

    # ------------------------------------------------------------------
    # StrategyInterface
    # ------------------------------------------------------------------

    def select_action(
        self,
        agent_id: str,
        state: LearningState,
        valid_actions: tuple[Action, ...],
    ) -> Action:
        """Return the next DFS action for *agent_id*."""
        visited = self._visited_for(agent_id)
        pos = state.agent_position
        just_arrived = pos not in visited
        visited.add(pos)

        obs = self._obstacles_for(agent_id)
        obs.update(state.visible_obstacles)

        moveable = [a for a in valid_actions if a != Action.WAIT]
        if not moveable:
            return Action.WAIT

        stack = self._stack_for(agent_id)
        buf = self._buffer_for(agent_id)

        # Node expansion: push unvisited reachable neighbours (first arrival only).
        # Appended in reverse _DIRS order so the first direction ends up on top.
        if just_arrived:
            to_push: list[Position] = []
            for name, dx, dy in self._DIRS:
                try:
                    action = Action(name)
                except ValueError:
                    continue
                if action not in moveable:
                    continue
                npos = Position(pos.x + dx, pos.y + dy)
                if npos not in visited and npos not in stack:
                    to_push.append(npos)
            for npos in reversed(to_push):
                stack.append(npos)

        # Execute buffered navigation path if still valid.
        while buf:
            action = buf[0]
            if action in valid_actions:
                buf.pop(0)
                return action
            buf.clear()   # Path blocked by newly discovered obstacle — replan.
            break

        # Discard stack entries that are already visited.
        while stack and stack[-1] in visited:
            stack.pop()

        if not stack:
            # Stack exhausted — fall back to random valid move.
            return self._rng.choice(moveable)

        target = stack[-1]

        # If target is directly adjacent, move there immediately.
        for name, dx, dy in self._DIRS:
            try:
                action = Action(name)
            except ValueError:
                continue
            if action not in moveable:
                continue
            if Position(pos.x + dx, pos.y + dy) == target:
                stack.pop()
                return action

        # Not adjacent — navigate via BFS over known passable cells.
        passable = set(state.discovered_cells) - obs
        path = self._bfs_navigate(pos, target, passable)
        if path:
            stack.pop()
            buf.extend(path[1:])
            return path[0]

        # Target unreachable — discard and fall back to random.
        stack.pop()
        return self._rng.choice(moveable)

    def update(self, transition: Transition) -> None:
        """No-op: the DFS baseline never updates from experience."""

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

    def _bfs_navigate(
        self,
        start: Position,
        target: Position,
        passable: set[Position],
    ) -> list[Action]:
        """BFS over *passable* cells; returns the action sequence start→target."""
        if start == target:
            return []
        reachable = passable | {start, target}
        queue: deque[tuple[Position, list[Action]]] = deque([(start, [])])
        seen: set[Position] = {start}
        while queue:
            pos, path = queue.popleft()
            for name, dx, dy in self._DIRS:
                npos = Position(pos.x + dx, pos.y + dy)
                if npos == target:
                    return path + [Action(name)]
                if npos not in seen and npos in reachable:
                    seen.add(npos)
                    queue.append((npos, path + [Action(name)]))
        return []

    def _visited_for(self, agent_id: str) -> set[Position]:
        if agent_id not in self._visited:
            self._visited[agent_id] = set()
        return self._visited[agent_id]

    def _stack_for(self, agent_id: str) -> list[Position]:
        if agent_id not in self._stack:
            self._stack[agent_id] = []
        return self._stack[agent_id]

    def _buffer_for(self, agent_id: str) -> list[Action]:
        if agent_id not in self._path_buffer:
            self._path_buffer[agent_id] = []
        return self._path_buffer[agent_id]

    def _obstacles_for(self, agent_id: str) -> set[Position]:
        if agent_id not in self._known_obstacles:
            self._known_obstacles[agent_id] = set()
        return self._known_obstacles[agent_id]
