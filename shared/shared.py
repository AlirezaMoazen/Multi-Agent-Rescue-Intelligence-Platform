"""Shared project reference for the rescue simulation.

This file mirrors the main names used across the project so the team can check
the common data structures, settings, movement names, and result fields without
reading every module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Grid and positions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class Grid:
    x_size: int
    y_size: int
    walls: frozenset[Position]
    target_a_positions: frozenset[Position]
    target_b_positions: frozenset[Position]

    def contains(self, position: Position) -> bool:
        return 0 <= position.x < self.x_size and 0 <= position.y < self.y_size

    def is_blocked(self, position: Position) -> bool:
        return position in self.walls

    def is_valid_position(self, position: Position) -> bool:
        return self.contains(position) and not self.is_blocked(position)

    def has_target(self, position: Position) -> bool:
        return (
            position in self.target_a_positions
            or position in self.target_b_positions
        )

    def target_type_at(self, position: Position) -> str | None:
        if position in self.target_a_positions:
            return "A"
        if position in self.target_b_positions:
            return "B"
        return None


# ---------------------------------------------------------------------------
# Scenario settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSettings:
    width: int
    height: int
    obstacle_probability: float
    target_a_count: int
    target_b_count: int
    random_seed: int | None = None


@dataclass(frozen=True)
class AgentSettings:
    start_x: int
    start_y: int
    sensor_range: int


@dataclass(frozen=True)
class SimulationSettings:
    max_steps: int


# ---------------------------------------------------------------------------
# Central sensor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    agent_id: str
    agent_position: Position
    visible_cells: frozenset[Position]
    newly_discovered_cells: frozenset[Position]
    obstacles: frozenset[Position]
    targets: frozenset[Position]
    target_types: dict[Position, str]
    newly_discovered_targets: frozenset[Position]


class CentralSensor:
    """Centralized sensor model with shared discovered-map memory."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self._discovered_cells: set[Position] = set()
        self._discovered_targets: dict[Position, str] = {}
        self._agent_positions: dict[str, Position] = {}
        self._latest_observations: dict[str, Observation] = {}

    @property
    def discovered_cells(self) -> frozenset[Position]:
        return frozenset(self._discovered_cells)

    @property
    def discovered_targets(self) -> dict[Position, str]:
        return dict(self._discovered_targets)

    @property
    def agent_positions(self) -> dict[str, Position]:
        return dict(self._agent_positions)

    @property
    def latest_observations(self) -> dict[str, Observation]:
        return dict(self._latest_observations)

    def observe(self, agent_id: str | int, position: Position, sensor_range: int) -> Observation:
        if sensor_range < 0:
            raise ValueError("sensor_range must be non-negative")
        if not self.grid.contains(position):
            raise ValueError("agent position must be inside the grid")

        normalized_agent_id = str(agent_id)
        visible_cells = self._visible_cells_from(position, sensor_range)
        previously_discovered_cells = set(self._discovered_cells)
        previously_discovered_targets = set(self._discovered_targets)

        obstacles = frozenset(cell for cell in visible_cells if self.grid.is_blocked(cell))
        target_types = {
            cell: target_type
            for cell in visible_cells
            if (target_type := self.grid.target_type_at(cell)) is not None
        }
        targets = frozenset(target_types)

        self._discovered_cells.update(visible_cells)
        self._discovered_targets.update(target_types)
        self._agent_positions[normalized_agent_id] = position

        observation = Observation(
            agent_id=normalized_agent_id,
            agent_position=position,
            visible_cells=visible_cells,
            newly_discovered_cells=frozenset(
                cell for cell in visible_cells if cell not in previously_discovered_cells
            ),
            obstacles=obstacles,
            targets=targets,
            target_types=target_types,
            newly_discovered_targets=frozenset(
                target for target in targets if target not in previously_discovered_targets
            ),
        )
        self._latest_observations[normalized_agent_id] = observation
        return observation

    def _visible_cells_from(self, center: Position, sensor_range: int) -> frozenset[Position]:
        visible_cells: set[Position] = set()
        for y in range(center.y - sensor_range, center.y + sensor_range + 1):
            for x in range(center.x - sensor_range, center.x + sensor_range + 1):
                position = Position(x, y)
                if self.grid.contains(position) and self._distance(center, position) <= sensor_range:
                    visible_cells.add(position)
        return frozenset(visible_cells)

    @staticmethod
    def _distance(first: Position, second: Position) -> int:
        return abs(first.x - second.x) + abs(first.y - second.y)


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "forward": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "wait": (0, 0),
}


@dataclass(frozen=True, slots=True)
class MovementResult:
    start: Position
    requested: Position
    end: Position
    move: str
    moved: bool
    reason: str


class MovementModel:
    """Validates and applies deterministic grid movements."""

    def next_position(self, position: Position, move: str) -> Position:
        if move not in MOVE_DELTAS:
            raise ValueError(f"unknown move: {move}")

        dx, dy = MOVE_DELTAS[move]
        return Position(position.x + dx, position.y + dy)

    def is_allowed(self, grid: Grid, position: Position, move: str) -> bool:
        requested = self.next_position(position, move)
        return grid.contains(requested) and not grid.is_blocked(requested)

    def allowed_moves(self, grid: Grid, position: Position) -> dict[str, Position]:
        return {
            move: self.next_position(position, move)
            for move in MOVE_DELTAS
            if self.is_allowed(grid, position, move)
        }

    def apply(self, grid: Grid, position: Position, move: str) -> MovementResult:
        requested = self.next_position(position, move)

        if not grid.contains(requested):
            return MovementResult(position, requested, position, move, False, "out_of_bounds")

        if grid.is_blocked(requested):
            return MovementResult(position, requested, position, move, False, "blocked")

        return MovementResult(position, requested, requested, move, move != "wait", "ok")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


DEFAULT_MOVES: tuple[str, ...] = ("right", "down", "left", "up", "wait")


@dataclass(frozen=True, slots=True)
class SimulationStep:
    step: int
    start: Position
    move: str
    requested: Position
    end: Position
    moved: bool
    reason: str
    target_found: str | None
    observation: Observation


@dataclass(frozen=True, slots=True)
class SimulationResult:
    grid: Grid
    start_position: Position
    final_position: Position
    initial_observation: Observation
    steps_taken: int
    targets_found: int
    found_targets: frozenset[Position]
    success: bool
    history: tuple[SimulationStep, ...]


# ---------------------------------------------------------------------------
# Agent and learning helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleAgent:
    x: int
    y: int
    sensor_range: int


class BaselineExplorer:
    """Initial deterministic explorer to be replaced or compared with RL methods."""

    pass


class Sensor:
    """Local sensor that reports neighboring cells as numeric states."""

    def __init__(self, agent: "Agent"):
        self.agent = agent

    def get_location(self) -> Tuple[int, int]:
        return self.agent.x, self.agent.y

    def sense_environment(self, grid: object) -> Dict[str, int]:
        x, y = self.get_location()
        return {
            "forward": cell_value_at(grid, x, y - 1),
            "down": cell_value_at(grid, x, y + 1),
            "left": cell_value_at(grid, x - 1, y),
            "right": cell_value_at(grid, x + 1, y),
        }


class Agent:
    """Mutable agent helper used by simple simulations and visualization."""

    def __init__(self, start_x: int, start_y: int, grid: object):
        self.x = start_x
        self.y = start_y
        self.grid = grid
        self.sensor = Sensor(self)
        self.history = [(start_x, start_y)]

    def forward(self) -> bool:
        return self._try_move(0, -1)

    def down(self) -> bool:
        return self._try_move(0, 1)

    def left(self) -> bool:
        return self._try_move(-1, 0)

    def right(self) -> bool:
        return self._try_move(1, 0)

    def _try_move(self, dx: int, dy: int) -> bool:
        next_x = self.x + dx
        next_y = self.y + dy
        if cell_value_at(self.grid, next_x, next_y) == 1:
            return False

        self.x = next_x
        self.y = next_y
        self.history.append((self.x, self.y))
        return True


def cell_value_at(grid: object, x: int, y: int) -> int:
    """Return 0 empty, 1 wall/out of bounds, 2 target A, or 3 target B."""
    if hasattr(grid, "get_cell"):
        return grid.get_cell(x, y)

    position = Position(x, y)
    if not grid.contains(position) or grid.is_blocked(position):
        return 1
    if position in grid.target_a_positions:
        return 2
    if position in grid.target_b_positions:
        return 3
    return 0


class RLAgent:
    """Q-learning helper used by the visualization."""

    def __init__(
        self,
        actions: List[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0,
    ):
        self.q_table = {}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def get_state_key(self, state_dict: Dict[str, int]) -> str:
        return str(sorted(state_dict.items()))

    def choose_action(self, state_dict: Dict[str, int]) -> str:
        state = self.get_state_key(state_dict)
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state_dict: Dict[str, int], action: str, reward: float, next_state_dict: Dict[str, int]):
        state = self.get_state_key(state_dict)
        next_state = self.get_state_key(next_state_dict)

        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
