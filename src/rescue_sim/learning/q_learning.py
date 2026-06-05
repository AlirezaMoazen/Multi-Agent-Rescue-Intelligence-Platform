"""Single-agent Q-learning connected to the Sprint 3 shared contract."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

from rescue_sim.config.settings import AgentSettings, GridSettings, SimulationSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor, Observation
from rescue_sim.shared import (
    Action,
    LearningState,
    RewardConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    TargetType,
    calculate_reward,
)


DEFAULT_ACTIONS: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.WAIT,
)
# Q-table format: one shared LearningState maps to one value per action.
QTable = dict[LearningState, dict[Action, float]]


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Training result for one episode."""

    steps: int
    total_reward: float
    targets_found: int
    success: bool


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Summary returned after several training episodes."""

    episodes: int
    successes: int
    average_reward: float
    average_steps: float
    episode_metrics: tuple[EpisodeMetrics, ...]


class QLearningAgent:
    """Q-table learner for one rescue agent."""

    def __init__(
        self,
        actions: Sequence[Action] = DEFAULT_ACTIONS,
        learning_rate: float = 0.2,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        reward_config: RewardConfig = SPRINT3_REWARD_CONFIG,
        rng: Random | None = None,
    ) -> None:
        if not actions:
            raise ValueError("at least one action is required")
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")

        self.actions = tuple(actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.reward_config = reward_config
        self.rng = rng or Random()
        self.q_table: QTable = defaultdict(self._new_action_values)

    def _new_action_values(self) -> dict[Action, float]:
        return {action: 0.0 for action in self.actions}

    def state_from_observation(
        self,
        observation: Observation,
        grid: Grid,
        found_targets: frozenset[Position],
        steps_taken: int,
    ) -> LearningState:
        """Build the shared Sprint 3 LearningState from current environment data."""
        # Separate visible targets by type so rewards/evaluation can distinguish A and B.
        visible_target_a = frozenset(
            position
            for position, target_type in observation.target_types.items()
            if target_type == TargetType.A
        )
        visible_target_b = frozenset(
            position
            for position, target_type in observation.target_types.items()
            if target_type == TargetType.B
        )
        found_target_a = frozenset(
            position for position in found_targets if position in grid.target_a_positions
        )
        found_target_b = frozenset(
            position for position in found_targets if position in grid.target_b_positions
        )

        return LearningState(
            agent_id=observation.agent_id,
            agent_position=observation.agent_position,
            visible_cells=observation.visible_cells,
            visible_obstacles=observation.obstacles,
            visible_target_a_positions=visible_target_a,
            visible_target_b_positions=visible_target_b,
            discovered_cells=observation.visible_cells,
            discovered_target_a_positions=visible_target_a,
            discovered_target_b_positions=visible_target_b,
            rescued_target_a_positions=found_target_a,
            rescued_target_b_positions=found_target_b,
            remaining_target_a_positions=grid.target_a_positions - found_target_a,
            remaining_target_b_positions=grid.target_b_positions - found_target_b,
            steps_taken=steps_taken,
        )

    def choose_action(
        self,
        state: LearningState,
        valid_actions: Sequence[Action],
    ) -> Action:
        """Choose an action with epsilon-greedy exploration."""
        if not valid_actions:
            raise ValueError("at least one valid action is required")

        actions = tuple(valid_actions)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        # Otherwise use the currently best known Q-value.
        action_values = self.q_table[state]
        return max(actions, key=lambda action: action_values[action])

    def update_q_value(
        self,
        state: LearningState,
        action: Action,
        reward: float,
        next_state: LearningState,
        next_valid_actions: Sequence[Action],
    ) -> None:
        """Apply the standard Q-learning update rule."""
        if action not in self.actions:
            raise ValueError(f"unknown action: {action}")
        if not next_valid_actions:
            raise ValueError("at least one next action is required")

        # Q-learning update: old value moves toward reward plus best future value.
        best_next_value = max(self.q_table[next_state][action] for action in next_valid_actions)
        old_value = self.q_table[state][action]
        target = reward + self.discount_factor * best_next_value
        self.q_table[state][action] = old_value + self.learning_rate * (target - old_value)

    def valid_actions(
        self,
        movement_model: MovementModel,
        grid: Grid,
        position: Position,
    ) -> tuple[Action, ...]:
        """Return legal actions that are part of this learner's action list."""
        allowed_moves = movement_model.allowed_moves(grid, position)
        # MovementModel returns strings; the learning contract uses Action values.
        return tuple(action for action in self.actions if action.value in allowed_moves)

    def train_episode(
        self,
        grid: Grid,
        start_position: Position,
        sensor_range: int,
        max_steps: int,
        movement_model: MovementModel | None = None,
    ) -> EpisodeMetrics:
        """Train the learner on one scenario."""
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        movement = movement_model or MovementModel()
        sensor = CentralSensor(grid)
        position = start_position
        found_targets: set[Position] = set()
        visited_positions: set[Position] = {start_position}
        all_targets = grid.target_a_positions | grid.target_b_positions
        total_reward = 0.0

        observation = sensor.observe("agent-1", position, sensor_range)

        for step_index in range(max_steps):
            state = self.state_from_observation(
                observation,
                grid,
                frozenset(found_targets),
                step_index,
            )

            # Stop when all targets are rescued or the step limit is reached.
            if state.is_terminal(max_steps):
                return EpisodeMetrics(
                    step_index,
                    total_reward,
                    len(found_targets),
                    found_targets == all_targets,
                )

            action = self.choose_action(state, self.valid_actions(movement, grid, position))
            movement_result = movement.apply(grid, position, action.value)
            next_position = movement_result.end
            next_observation = sensor.observe("agent-1", next_position, sensor_range)

            # Check whether the move rescued a new target.
            target_type = grid.target_type_at(next_position)
            new_target_type = None
            if target_type is not None and next_position not in found_targets:
                found_targets.add(next_position)
                new_target_type = TargetType(target_type)

            done = found_targets == all_targets
            reward = calculate_reward(
                RewardEvent(
                    moved=movement_result.moved,
                    move=action.value,
                    newly_discovered_cells=len(next_observation.newly_discovered_cells),
                    rescued_target_type=new_target_type,
                    completed_episode=done,
                    repeated_cell=next_position in visited_positions,
                ),
                self.reward_config,
            )
            total_reward += reward

            # Learn from this transition.
            next_state = self.state_from_observation(
                next_observation,
                grid,
                frozenset(found_targets),
                step_index + 1,
            )
            next_valid_actions = self.valid_actions(movement, grid, next_position)
            self.update_q_value(state, action, reward, next_state, next_valid_actions)

            position = next_position
            observation = next_observation
            visited_positions.add(position)

        return EpisodeMetrics(max_steps, total_reward, len(found_targets), found_targets == all_targets)

    def train(
        self,
        grid_settings: GridSettings,
        agent_settings: AgentSettings,
        simulation_settings: SimulationSettings,
        episodes: int,
        movement_model: MovementModel | None = None,
    ) -> TrainingMetrics:
        """Generate scenarios and train for several episodes."""
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        start_position = Position(agent_settings.start_x, agent_settings.start_y)
        episode_metrics: list[EpisodeMetrics] = []

        for _ in range(episodes):
            # Each episode starts with a fresh generated grid using the given settings.
            grid = generate_grid(grid_settings, start=start_position)
            episode_metrics.append(
                self.train_episode(
                    grid=grid,
                    start_position=start_position,
                    sensor_range=agent_settings.sensor_range,
                    max_steps=simulation_settings.max_steps,
                    movement_model=movement_model,
                )
            )

        successes = sum(metrics.success for metrics in episode_metrics)
        average_reward = sum(metrics.total_reward for metrics in episode_metrics) / episodes
        average_steps = sum(metrics.steps for metrics in episode_metrics) / episodes

        return TrainingMetrics(
            episodes=episodes,
            successes=successes,
            average_reward=average_reward,
            average_steps=average_steps,
            episode_metrics=tuple(episode_metrics),
        )

    def best_policy(self) -> dict[LearningState, Action]:
        """Return the currently best action for every known state."""
        return {
            state: max(action_values, key=action_values.get)
            for state, action_values in self.q_table.items()
        }

    def q_values(self) -> Mapping[LearningState, Mapping[Action, float]]:
        """Expose a copy of the learned Q-table."""
        return {state: dict(action_values) for state, action_values in self.q_table.items()}
