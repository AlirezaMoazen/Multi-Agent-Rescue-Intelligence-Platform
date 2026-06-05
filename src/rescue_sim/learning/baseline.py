"""Baseline exploration strategy: frontier-based, deterministic, reproducible.

The BaselineExplorer implements StrategyInterface from shared.py without any
learning.  It drives exploration via a greedy scoring rule:

  score(candidate_pos) =
      +2  if the cell has never been visited by this agent
      +1  if any orthogonal neighbour of the cell is still undiscovered
            (i.e. the cell is on the "frontier" of the known map)

Ties are resolved first by a fixed action-priority order (UP → RIGHT → DOWN
→ LEFT → FORWARD), then by the seeded RNG.  WAIT is only chosen when it is
the only valid action.

Per-agent internal state (visited cells) is keyed by agent_id so the class
works correctly in multi-agent scenarios without assuming id == "0".
"""

from __future__ import annotations

import random
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
# Metrics dataclass
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
# Explorer
# ---------------------------------------------------------------------------

class BaselineExplorer:
    """Non-learning frontier-based explorer that implements StrategyInterface.

    Parameters
    ----------
    seed:
        Seed for the internal RNG used to break ties when the scoring heuristic
        cannot distinguish between candidates.  Pass an integer for
        reproducible runs; pass ``None`` for non-deterministic behaviour.
    """

    # Canonical action priority for deterministic tie-breaking.
    # FORWARD maps to the same delta as UP so it appears last to avoid
    # picking an alias when the primary UP direction is available.
    _PRIORITY: tuple[Action, ...] = (
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.LEFT,
        Action.FORWARD,
    )

    def __init__(self, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)
        # Per-agent visited-cell memory; keyed by agent_id string.
        self._visited: dict[str, set[Position]] = {}
        # Accumulated obstacle knowledge shared across agents (conservative:
        # any agent seeing an obstacle records it here so all agents benefit).
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
        """Return the best action for *agent_id* given *state* and *valid_actions*.

        The agent never waits unless WAIT is the only valid action.
        """
        visited = self._visited_for(agent_id)
        # Mark the current cell as visited before scoring candidates so that
        # standing still is never rewarded as "new".
        visited.add(state.agent_position)
        # Grow the shared obstacle map from what this agent can currently see.
        self._known_obstacles.update(state.visible_obstacles)

        moveable = [a for a in valid_actions if a != Action.WAIT]
        if not moveable:
            return Action.WAIT

        pos = state.agent_position

        # Build a deduplicated candidate list in priority order.
        # UP and FORWARD share the same delta (0,-1); the first one wins.
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

        # Include any valid moveable actions not covered by _PRIORITY.
        priority_set = set(self._PRIORITY)
        for action in moveable:
            if action in priority_set:
                continue
            dx, dy = MOVE_DELTAS[action.value]
            next_pos = Position(pos.x + dx, pos.y + dy)
            if next_pos not in seen_positions:
                seen_positions.add(next_pos)
                candidates.append((action, next_pos))

        # Score and select.
        best_score = -1
        best: list[Action] = []

        for action, next_pos in candidates:
            s = self._score(next_pos, visited, state.discovered_cells)
            if s > best_score:
                best_score = s
                best = [action]
            elif s == best_score:
                best.append(action)

        # Candidates were inserted in priority order so best[0] is already the
        # highest-priority tied action; the RNG is only needed as a last resort.
        return best[0] if len(best) == 1 else self._rng.choice(best)

    def update(self, transition: Transition) -> None:
        """No-op: the baseline never updates from experience."""

    # ------------------------------------------------------------------
    # Episode runner
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
            Hard cap on steps; the episode also ends when the environment
            signals ``done=True`` (all targets rescued).
        total_cells:
            Total passable cells in the grid.  When provided,
            ``percentage_discovered`` in the returned metrics is computed as
            ``(discovered_cells / total_cells) * 100``.  Pass ``None`` to
            skip that metric.

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

    def _visited_for(self, agent_id: str) -> set[Position]:
        """Return (and lazily create) the visited-cell set for *agent_id*."""
        if agent_id not in self._visited:
            self._visited[agent_id] = set()
        return self._visited[agent_id]

    def _score(
        self,
        next_pos: Position,
        visited: set[Position],
        discovered: frozenset[Position],
    ) -> int:
        """Score a candidate next position.

        +2  cell has never been visited by this agent
        +1  at least one orthogonal neighbour is not yet in the discovered map
            (frontier bonus — keeps the agent at the edge of explored territory)
        """
        score = 0

        if next_pos not in visited:
            score += 2

        # Frontier bonus: look for an orthogonal neighbour that is neither
        # in the discovered set nor a known obstacle.
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbour = Position(next_pos.x + dx, next_pos.y + dy)
            if (
                neighbour not in discovered
                and neighbour not in self._known_obstacles
            ):
                score += 1
                break

        return score
