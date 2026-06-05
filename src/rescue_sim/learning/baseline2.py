"""DFS-based exploration baseline.

The DFSExplorer implements StrategyInterface from shared.py using a classic
iterative Depth-First Search over the grid.

How it works
------------
Each time the agent arrives at a cell for the first time it *expands* that
cell: all immediately reachable unvisited neighbours are pushed onto a
per-agent LIFO stack.  The agent then always navigates toward the cell at
the top of the stack, going as deep as possible before backtracking — the
defining property of DFS.

When the top of the stack is not directly adjacent (because the agent had
to backtrack through already-visited cells), the explorer uses BFS over
the accumulated known-passable map to find a navigation path.  This path
is buffered and consumed one action at a time.

Comparison with BaselineExplorer (baseline.py)
-----------------------------------------------
* BaselineExplorer (frontier greedy) always picks the locally best cell.
  It tends to sweep outward evenly and covers wide areas quickly.
* DFSExplorer follows one branch to its end before backtracking.  It may
  take longer to cover the whole map but tends to escape dead-ends faster
  once they are identified.

Both share the same StrategyInterface so they can be swapped directly.

Per-agent internal state is keyed by agent_id so multi-agent scenarios
work without assuming a fixed agent "0".
"""

from __future__ import annotations

import random
from collections import deque

from rescue_sim.environment.grid import Position
from rescue_sim.learning.baseline import BaselineMetrics
from rescue_sim.shared import (
    Action,
    EnvironmentInterface,
    LearningState,
    Transition,
)


class DFSExplorer:
    """Depth-First Search explorer implementing StrategyInterface.

    Parameters
    ----------
    seed:
        Seed for the internal RNG used only as a fallback when the DFS
        stack is exhausted (all known cells visited) or when a target
        becomes unreachable.  Pass an integer for reproducible runs; pass
        ``None`` for non-deterministic behaviour.
    """

    # Primary movement directions in exploration-priority order.
    # "forward" is omitted because it is an alias for "up".
    # Direction order determines DFS traversal order: the first direction
    # listed ends up on top of the LIFO stack and is explored first.
    _DIRS: tuple[tuple[str, int, int], ...] = (
        ("up",    0, -1),
        ("right", 1,  0),
        ("down",  0,  1),
        ("left", -1,  0),
    )
    _DIR_NAMES: frozenset[str] = frozenset(name for name, *_ in _DIRS)

    def __init__(self, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)
        # Per-agent visited cells.
        self._visited: dict[str, set[Position]] = {}
        # Per-agent DFS stack (list used as LIFO: append = push, pop() = pop).
        self._stack: dict[str, list[Position]] = {}
        # Per-agent buffered navigation path (actions to reach current target).
        self._path_buffer: dict[str, list[Action]] = {}
        # Per-agent accumulated obstacle knowledge.
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
        """Return the next DFS action for *agent_id* given *state*."""
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

        # ----------------------------------------------------------------
        # Node expansion: push unvisited reachable neighbours onto the stack.
        # Only done the first time we arrive at a position.
        # Append in reverse _DIRS order so the first direction in _DIRS ends
        # up on top of the LIFO stack and is explored first.
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Execute buffered navigation path if still valid.
        # ----------------------------------------------------------------
        while buf:
            action = buf[0]
            if action in valid_actions:
                buf.pop(0)
                return action
            # A new obstacle has blocked the path — clear the buffer and replan.
            buf.clear()
            break

        # ----------------------------------------------------------------
        # Find the next unvisited target from the DFS stack.
        # ----------------------------------------------------------------
        while stack and stack[-1] in visited:
            stack.pop()

        if not stack:
            # Stack exhausted: all reachable cells have been visited.
            # Fall back to a random valid movement (handles partial maps).
            return self._rng.choice(moveable)

        target = stack[-1]

        # ----------------------------------------------------------------
        # If the target is directly adjacent, move there in one step.
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Target is not adjacent — navigate via BFS over the known map.
        # ----------------------------------------------------------------
        passable = set(state.discovered_cells) - obs
        path = self._bfs_navigate(pos, target, passable)
        if path:
            stack.pop()
            buf.extend(path[1:])   # Buffer remaining steps; take first now.
            return path[0]

        # Target is unreachable through the known map — discard and move on.
        stack.pop()
        return self._rng.choice(moveable)

    def update(self, transition: Transition) -> None:
        """No-op: the DFS baseline never updates its policy from experience."""

    # ------------------------------------------------------------------
    # Episode runner (identical interface to BaselineExplorer)
    # ------------------------------------------------------------------

    def run_episode(
        self,
        env: EnvironmentInterface,
        max_steps: int = 500,
        total_cells: int | None = None,
    ) -> BaselineMetrics:
        """Run one complete episode and return summary metrics.

        Parameters
        ----------
        env:
            Environment conforming to EnvironmentInterface.
        max_steps:
            Hard cap on steps.  The episode also ends when the environment
            signals ``done=True`` (all targets rescued).
        total_cells:
            Total passable cells in the grid.  When provided,
            ``percentage_discovered`` is computed as
            ``(discovered_cells / total_cells) * 100``.
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            valid_actions = env.get_valid_actions(state)
            action = self.select_action(state.agent_id, state, valid_actions)
            transition = env.step(action)
            self.update(transition)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bfs_navigate(
        self,
        start: Position,
        target: Position,
        passable: set[Position],
    ) -> list[Action]:
        """BFS over *passable* cells; returns the action sequence start→target.

        Both *start* and *target* are always included in the search space
        even if they are not yet in *passable* (e.g. the start cell may be
        an obstacle we are standing on due to placement rules).
        """
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

        return []  # Target not reachable through the known map.

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
