"""Reference contracts for independent Sprint 3 development.

This file is intentionally self-contained: it does not import or call other
project modules. The first section mirrors the existing Sprint 2 code. The
second section defines the Sprint 3 contract that exploration, learning, and
evaluation work should follow. The third section defines the minimum Sprint 4
contract needed for independent multi-agent work.
"""

from __future__ import annotations

import math
from collections import deque
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


# New Sprint 4 multi-agent contracts


AgentId = str
JointAction = dict[AgentId, Action]
CostByAgent = dict[AgentId, int | float]


@dataclass(frozen=True, slots=True)
class AgentStart:
    """Configuration for one agent at scenario reset."""

    agent_id: AgentId
    position: Position
    sensor_range: int


@dataclass(frozen=True, slots=True)
class MultiAgentSettings:
    """Scenario-level settings needed by the multi-agent environment task."""

    agents: tuple[AgentStart, ...]
    communication_range: int = 0

    @property
    def num_agents(self) -> int:
        return len(self.agents)


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """Shared target knowledge used by communication, reward, and evaluation."""

    position: Position
    target_type: TargetType
    discovered_by: AgentId | None = None
    discovered_step: int | None = None
    rescued_by: AgentId | None = None
    rescued_step: int | None = None
    cost_by_agent: CostByAgent | None = None


@dataclass(frozen=True, slots=True)
class AgentState:
    """Runtime state for one rescue agent."""

    agent_id: AgentId
    position: Position
    sensor_range: int
    active: bool = True
    total_reward: float = 0.0
    visited_cells: frozenset[Position] = frozenset()
    last_action: Action | None = None


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """Information sent between nearby agents."""

    sender_id: AgentId
    receiver_id: AgentId | None
    step: int
    sender_position: Position
    discovered_cells: frozenset[Position] = frozenset()
    targets: tuple[TargetInfo, ...] = ()
    target_costs: dict[Position, CostByAgent] | None = None
    known_agent_positions: dict[AgentId, Position] | None = None


@dataclass(frozen=True, slots=True)
class MultiAgentState:
    """Full environment state shared by multi-agent environment and strategies."""

    agents: dict[AgentId, AgentState]
    shared_discovered_cells: frozenset[Position] = frozenset()
    shared_obstacles: frozenset[Position] = frozenset()
    shared_targets: dict[Position, TargetInfo] | None = None
    rescued_targets: frozenset[Position] = frozenset()
    remaining_target_a_positions: frozenset[Position] = frozenset()
    remaining_target_b_positions: frozenset[Position] = frozenset()
    messages: tuple[AgentMessage, ...] = ()
    steps_taken: int = 0

    @property
    def remaining_targets(self) -> int:
        return len(self.remaining_target_a_positions) + len(self.remaining_target_b_positions)

    def is_terminal(self, max_steps: int) -> bool:
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
    collision: float = 0.0
    communication_cost: float = 0.0
    useful_communication_bonus: float = 0.0
    duplicate_target_penalty: float = 0.0
    team_target_rescue_bonus: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardEvent:
    """Facts produced by one environment step for reward calculation."""

    moved: bool
    move: str
    agent_id: AgentId | None = None
    newly_discovered_cells: int = 0
    rescued_target_type: TargetType | None = None
    completed_episode: bool = False
    repeated_cell: bool = False
    collision: bool = False
    communication_sent: bool = False
    useful_communication: bool = False
    duplicate_target: bool = False
    team_target_rescued: bool = False


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

    if event.collision:
        reward += config.collision

    if event.communication_sent:
        reward += config.communication_cost

    if event.useful_communication:
        reward += config.useful_communication_bonus

    if event.duplicate_target:
        reward += config.duplicate_target_penalty

    if event.team_target_rescued:
        reward += config.team_target_rescue_bonus
        
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


@dataclass(frozen=True, slots=True)
class MultiAgentTransition:
    """Result expected from a Sprint 4 environment after one joint action."""

    state: MultiAgentState
    joint_action: JointAction
    next_state: MultiAgentState
    rewards: dict[AgentId, float]
    team_reward: float
    done: bool
    movements: dict[AgentId, MovementResult]
    observations: dict[AgentId, Observation]
    messages: tuple[AgentMessage, ...] = ()


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
class MultiAgentEnvironmentInterface(Protocol):
    """Interface for central or distributed multi-agent environments."""

    def reset(self) -> MultiAgentState:
        ...

    def get_valid_actions(
        self,
        agent_id: AgentId,
        state: MultiAgentState,
    ) -> tuple[Action, ...]:
        ...

    def step(self, joint_action: JointAction) -> MultiAgentTransition:
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


@runtime_checkable
class MultiAgentStrategyInterface(Protocol):
    """Common API for central and distributed multi-agent strategies."""

    def select_actions(
        self,
        state: MultiAgentState,
        valid_actions: dict[AgentId, tuple[Action, ...]],
    ) -> JointAction:
        ...

    def update(self, transition: MultiAgentTransition) -> None:
        ...


# ---------------------------------------------------------------------------
# Decentralized fleet contracts (Epidemic Hysteretic Q-Learning)
# ---------------------------------------------------------------------------

# Cardinal action order used by the vectorized fleet learner: index 0..3.
# North = up (y - 1), South = down (y + 1), East = right (x + 1), West = left (x - 1).
# Kept consistent with MOVE_DELTAS so the rest of the system (MovementModel,
# visualization) can translate an index back to a move string via ``.value``.
CARDINAL_ACTIONS: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT)


@dataclass(frozen=True, slots=True)
class HystereticConfig:
    """Hysteretic Q-learning hyper-parameters.

    Two learning rates make cooperative agents *optimistic*: a positive TD error
    is applied with ``alpha`` while a negative TD error is applied with a heavily
    muted ``beta`` (``beta << alpha``).  This stops a teammate's exploration from
    erasing an already-good policy. ``beta <= alpha`` is required.
    """

    alpha: float = 0.5
    beta: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.2


@dataclass(frozen=True, slots=True)
class GossipConfig:
    """Ad-hoc peer-to-peer epidemic synchronization parameters."""

    comm_radius: float = 3.0          # Euclidean distance that opens a link
    cooldown: int = 5                 # steps before the same pair may re-sync
    max_links_per_step: int = 2       # per-agent handshake budget (congestion control)
    utility_threshold: float = 0.0    # only gossip |Q| at or above this value
    clear_dirty_on_export: bool = True


# ---------------------------------------------------------------------------
# Deep-RL shared utilities (used by MAPPO / QMIX / TransfQMix / Ensemble)
# ---------------------------------------------------------------------------
# torch and numpy are imported lazily *inside* the functions below so that this
# module stays importable without the optional deep-RL extras.  The api,
# visualization, and baseline code import `shared` but never need torch.


def resolve_device(device: str | None = None):
    """Selects the compute device: explicit override, else CUDA if available, else CPU.

    The deep-RL networks here are small and the wall-clock cost is dominated by
    CPU-side environment stepping, so a GPU helps only modestly (most on the
    TransfQMix transformer) -- but honoring one when present is free and keeps
    the trainers portable to a CUDA host.
    """
    import torch

    if device is not None and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_eval_hook(trainer, checkpoint_path, time_budget_s: float, eval_episodes: int = 20):
    """Periodic-eval hook for the deep-RL trainers.

    RL performance is noisy and can regress, so this saves a checkpoint only
    when greedy eval improves (`avg_rescued`, tie-broken by `success_rate`) and
    signals the training loop to stop once ``time_budget_s`` wall-clock elapses.
    Returns ``(hook, state)``; pass ``hook`` as ``eval_hook`` to ``train()``.
    """
    import time

    state = {"best_rescued": -1.0, "best_success": -1.0, "start": time.time(), "saved": 0}

    def hook(step: int) -> bool:
        metrics = trainer.evaluate(episodes=eval_episodes)
        score = (metrics["avg_rescued"], metrics["success_rate"])
        improved = score > (state["best_rescued"], state["best_success"])
        if improved:
            state["best_rescued"], state["best_success"] = score
            trainer.save_checkpoint(checkpoint_path)
            state["saved"] += 1
        elapsed = time.time() - state["start"]
        print(
            f"  [eval @ {step:>5}] success {metrics['success_rate']:.2f} "
            f"rescued {metrics['avg_rescued']:.2f} steps {metrics['avg_steps']:.0f} "
            f"| best_rescued {state['best_rescued']:.2f} | {elapsed / 60:.1f}min"
            + ("  <== saved best" if improved else ""),
            flush=True,
        )
        return elapsed >= time_budget_s  # True => stop training

    return hook, state


def orthogonal_init(layer, gain: float = 2 ** 0.5):
    """Orthogonal weights + zero bias -- a standard PPO/DQN stability trick."""
    from torch import nn

    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def hard_update(target, source) -> None:
    """Copy every weight from `source` into `target` (target-network sync)."""
    target.load_state_dict(source.state_dict())


class RunningMeanStd:
    """Running mean/variance to normalize value targets (MAPPO trick #1).

    Welford-style batched update; keeps statistics stable as rewards drift.
    Works on any object with `.mean()` / `.var()` / `.numel()` (e.g. a tensor),
    so it needs no torch import of its own.
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x) -> None:
        batch_mean = float(x.mean())
        batch_var = float(x.var(unbiased=False))
        batch_count = x.numel()
        # A single non-finite target (e.g. masked -inf reaching the Bellman
        # backup) would poison mean/var permanently; skip such batches.
        if not (math.isfinite(batch_mean) and math.isfinite(batch_var)):
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return self.var**0.5 + 1e-8


class ReplayBuffer:
    """Fixed-size buffer of single-step team transitions (off-policy methods).

    Each transition is a dict of NumPy arrays; `sample` stacks a random batch
    into torch tensors with the right dtype per field.  Shared by QMIX and
    TransfQMix.
    """

    # Field name -> torch dtype name (resolved lazily) for the batched tensor.
    _DTYPES = {
        "obs": "float32", "state": "float32", "actions": "long", "avail": "bool",
        "reward": "float32", "next_obs": "float32", "next_state": "float32",
        "next_avail": "bool", "done": "float32",
    }

    def __init__(self, capacity: int, rng) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = rng

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: dict) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict:
        import numpy as np
        import torch

        batch = self.rng.sample(self.buffer, batch_size)
        return {
            key: torch.as_tensor(np.stack([t[key] for t in batch])).to(getattr(torch, dt))
            for key, dt in self._DTYPES.items()
        }
