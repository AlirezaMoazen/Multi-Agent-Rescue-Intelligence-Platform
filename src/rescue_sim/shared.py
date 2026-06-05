"""Reference contracts for independent Sprint 3 development.

This file is intentionally self-contained: it does not import or call other
project modules. The first section mirrors the existing Sprint 2 code. The
second section defines the new Sprint 3 contract that exploration, learning,
and evaluation work should follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, TypedDict, runtime_checkable


# Existing Sprint 2 contracts


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


@dataclass
class ApiSimConfig:
    """Field mirror of visualization.api.SimConfig without its Pydantic dependency."""

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
    speed_ms: int = 100


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    obstacles: frozenset[Position]
    target_a_positions: frozenset[Position]
    target_b_positions: frozenset[Position]

    def contains(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def is_blocked(self, position: Position) -> bool:
        return position in self.obstacles

    def is_valid_position(self, position: Position) -> bool:
        return self.contains(position) and not self.is_blocked(position)

    def has_target(self, position: Position) -> bool:
        return position in self.target_a_positions or position in self.target_b_positions

    def target_type_at(self, position: Position) -> str | None:
        if position in self.target_a_positions:
            return "A"
        if position in self.target_b_positions:
            return "B"
        return None


# Reuse Grid as GridState to avoid duplicating the data contract and behavior
GridState = Grid


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


ScenarioGenerator = Callable[[GridSettings, Position], Grid]
Pathfinder = Callable[[Grid, Position, Position], list[str]]


@dataclass(frozen=True)
class SingleAgent:
    x: int
    y: int
    sensor_range: int


class RescuedTargetRecord(TypedDict):
    """Current item format returned by EnvironmentHelper.get_rescued_list()."""

    x: int
    y: int
    step: int


class AgentStepPayload(TypedDict):
    """Current format returned by EnvironmentHelper.step()."""

    id: int
    x: int
    y: int
    action: str
    reward: float


class EpisodeMetric(TypedDict):
    """Current per-episode format produced by the visualization API."""

    episode: int
    steps: int
    rescued_count: int
    target_count: int
    success: bool
    total_reward: float
    exploration_rate: float


@runtime_checkable
class ExistingMovementModelInterface(Protocol):
    """Current MovementModel public methods."""

    def next_position(self, position: Position, move: str) -> Position:
        ...

    def is_allowed(self, grid: object, position: Position, move: str) -> bool:
        ...

    def allowed_moves(self, grid: object, position: Position) -> dict[str, Position]:
        ...

    def apply(self, grid: object, position: Position, move: str) -> MovementResult:
        ...

    def apply_to_agent(self, agent: object, move: str) -> MovementResult:
        ...


@runtime_checkable
class ExistingCentralSensorInterface(Protocol):
    """Current CentralSensor public properties and observation method."""

    @property
    def discovered_cells(self) -> frozenset[Position]:
        ...

    @property
    def discovered_targets(self) -> dict[Position, str]:
        ...

    @property
    def agent_positions(self) -> dict[str, Position]:
        ...

    @property
    def latest_observations(self) -> dict[str, Observation]:
        ...

    def observe(
        self,
        agent_id: str | int,
        position: Position,
        sensor_range: int,
    ) -> Observation:
        ...


@runtime_checkable
class ExistingEnvironmentHelperInterface(Protocol):
    """Current EnvironmentHelper public methods used by the API."""

    def step(self, step_idx: int) -> AgentStepPayload:
        ...

    def has_active_targets(self) -> bool:
        ...

    def get_active_targets_count(self) -> int:
        ...

    def get_rescued_list(self) -> list[RescuedTargetRecord]:
        ...

    def get_total_reward(self) -> float:
        ...


# New Sprint 3 contracts


class Action(str, Enum):
    """Movement actions accepted by the existing movement model."""

    UP = "up"
    FORWARD = "forward"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"


class TargetType(str, Enum):
    """Target types already represented separately by the existing Grid."""

    A = "A"
    B = "B"


class TypedRescuedTargetRecord(TypedDict):
    """Sprint 3 rescue record: add the type missing from the current helper."""

    x: int
    y: int
    step: int
    target_type: TargetType


class PositionPayload(TypedDict):
    x: int
    y: int


class ScenarioMetric(TypedDict):
    """Evaluation output aligned with docs/requirements.yaml."""

    scenario_id: str
    num_agents: int
    success: bool
    steps_taken: int
    targets_found: int
    target_a_count: int
    target_b_count: int
    explored_cells: int
    final_position: PositionPayload
    random_seed: int | None


@dataclass(frozen=True, slots=True)
class LearningState:
    """Hashable Q-table state preserving separate Target A and Target B data."""

    agent_id: str
    agent_position: Position
    visible_cells: frozenset[Position] = frozenset()
    visible_obstacles: frozenset[Position] = frozenset()
    visible_target_a_positions: frozenset[Position] = frozenset()
    visible_target_b_positions: frozenset[Position] = frozenset()
    discovered_cells: frozenset[Position] = frozenset()
    discovered_target_a_positions: frozenset[Position] = frozenset()
    discovered_target_b_positions: frozenset[Position] = frozenset()
    rescued_target_a_positions: frozenset[Position] = frozenset()
    rescued_target_b_positions: frozenset[Position] = frozenset()
    remaining_target_a_positions: frozenset[Position] = frozenset()
    remaining_target_b_positions: frozenset[Position] = frozenset()
    steps_taken: int = 0

    @property
    def remaining_targets(self) -> int:
        return len(self.remaining_target_a_positions) + len(self.remaining_target_b_positions)

    def is_terminal(self, max_steps: int) -> bool:
        """Check if the state is terminal (episode complete).

        An episode ends when either:
        1. all targets are rescued (remaining_targets == 0)
        2. max_steps is reached (steps_taken >= max_steps)
        """
        return self.remaining_targets == 0 or self.steps_taken >= max_steps


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Sprint 3 reward values; defaults preserve the current helper behavior."""

    move: float = -0.1
    invalid_move: float = -1.0
    wait: float = -1.0
    discovered_cell_bonus: float = 0.0
    rescued_target_a: float = 10.0
    rescued_target_b: float = 10.0
    completed_episode_bonus: float = 0.0
    repeated_cell: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardEvent:
    """Facts produced by one environment step for reward calculation."""

    moved: bool
    move: str
    newly_discovered_cells: int = 0
    rescued_target_type: TargetType | None = None
    completed_episode: bool = False
    repeated_cell: bool = False


# Standard Sprint 3 reward configuration for Q-learning.
SPRINT3_REWARD_CONFIG = RewardConfig(
    move=-1.0,
    invalid_move=-5.0,
    wait=-2.0,
    discovered_cell_bonus=2.0,
    repeated_cell=-1.5,
    rescued_target_a=150.0,
    rescued_target_b=100.0,
    completed_episode_bonus=50.0,
)


def calculate_reward(
    event: RewardEvent,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Calculate reward while preserving the current rescue reward override."""
    if event.move == Action.WAIT:
        reward = config.wait
    elif event.moved:
        reward = config.move
    else:
        reward = config.invalid_move

    if event.rescued_target_type == TargetType.A:
        reward = config.rescued_target_a
    elif event.rescued_target_type == TargetType.B:
        reward = config.rescued_target_b

    reward += event.newly_discovered_cells * config.discovered_cell_bonus
    
    if event.repeated_cell:
        reward += config.repeated_cell
        
    if event.completed_episode:
        reward += config.completed_episode_bonus
        
    return reward


@dataclass(frozen=True, slots=True)
class Transition:
    """Result expected from a Sprint 3 environment after one action."""

    state: LearningState
    action: Action
    next_state: LearningState
    reward: float
    done: bool
    movement: MovementResult
    observation: Observation


@runtime_checkable
class EnvironmentInterface(Protocol):
    """Interface that the real environment and temporary test doubles follow."""

    def reset(self) -> LearningState:
        ...

    def get_valid_actions(self, state: LearningState) -> tuple[Action, ...]:
        ...

    def step(self, action: Action) -> Transition:
        ...


@runtime_checkable
class StrategyInterface(Protocol):
    """Common API for baseline, Q-learning, and future strategies."""

    def select_action(
        self,
        agent_id: str,
        state: LearningState,
        valid_actions: tuple[Action, ...],
    ) -> Action:
        ...

    def update(self, transition: Transition) -> None:
        ...
