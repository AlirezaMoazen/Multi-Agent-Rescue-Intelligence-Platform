from rescue_sim.shared import Action, Grid, LearningState, Position, RewardConfig, RewardEvent
from rescue_sim.shared import TargetType, calculate_reward, GridState, SPRINT3_REWARD_CONFIG


def test_reference_grid_preserves_separate_target_types() -> None:
    target_a = Position(1, 0)
    target_b = Position(2, 0)
    grid = Grid(
        width=3,
        height=1,
        obstacles=frozenset(),
        target_a_positions=frozenset({target_a}),
        target_b_positions=frozenset({target_b}),
    )

    assert grid.target_type_at(target_a) == "A"
    assert grid.target_type_at(target_b) == "B"


def test_actions_match_existing_movement_commands() -> None:
    assert [action.value for action in Action] == [
        "up",
        "forward",
        "down",
        "left",
        "right",
        "wait",
    ]


def test_learning_state_is_hashable_and_preserves_both_target_types() -> None:
    state = LearningState(
        agent_id="agent-1",
        agent_position=Position(1, 2),
        discovered_target_a_positions=frozenset({Position(2, 2)}),
        discovered_target_b_positions=frozenset({Position(3, 2)}),
        remaining_target_a_positions=frozenset({Position(4, 2)}),
        remaining_target_b_positions=frozenset({Position(5, 2), Position(6, 2)}),
    )

    q_table = {(state, Action.RIGHT): 2.5}

    assert q_table[(state, Action.RIGHT)] == 2.5
    assert state.remaining_targets == 3


def test_default_reward_preserves_current_helper_values() -> None:
    assert calculate_reward(RewardEvent(moved=True, move="right")) == -0.1
    assert calculate_reward(RewardEvent(moved=False, move="right")) == -1.0
    assert calculate_reward(RewardEvent(moved=False, move="wait")) == -1.0
    assert (
        calculate_reward(
            RewardEvent(moved=True, move="right", rescued_target_type=TargetType.A)
        )
        == 10.0
    )


def test_target_types_can_have_different_rewards_in_sprint_3() -> None:
    config = RewardConfig(rescued_target_a=10.0, rescued_target_b=25.0)

    assert (
        calculate_reward(
            RewardEvent(moved=True, move="right", rescued_target_type=TargetType.A),
            config,
        )
        == 10.0
    )
    assert (
        calculate_reward(
            RewardEvent(moved=True, move="right", rescued_target_type=TargetType.B),
            config,
        )
        == 25.0
    )


def test_grid_state_behavior() -> None:
    obstacles = frozenset({Position(1, 1)})
    target_a = frozenset({Position(0, 1)})
    target_b = frozenset({Position(2, 0)})
    grid_state = GridState(
        width=3,
        height=3,
        obstacles=obstacles,
        target_a_positions=target_a,
        target_b_positions=target_b,
    )

    assert grid_state.width == 3
    assert grid_state.height == 3
    assert grid_state.obstacles == obstacles
    assert grid_state.contains(Position(0, 0)) is True
    assert grid_state.contains(Position(3, 3)) is False
    assert grid_state.is_blocked(Position(1, 1)) is True
    assert grid_state.is_blocked(Position(0, 0)) is False
    assert grid_state.is_valid_position(Position(1, 1)) is False
    assert grid_state.is_valid_position(Position(0, 0)) is True
    assert grid_state.has_target(Position(0, 1)) is True
    assert grid_state.has_target(Position(0, 0)) is False
    assert grid_state.target_type_at(Position(0, 1)) == "A"
    assert grid_state.target_type_at(Position(2, 0)) == "B"
    assert grid_state.target_type_at(Position(0, 0)) is None


def test_learning_state_termination_semantics() -> None:
    # Scenario: Not terminal (targets remain, steps within max)
    state = LearningState(
        agent_id="agent-0",
        agent_position=Position(0, 0),
        remaining_target_a_positions=frozenset({Position(0, 1)}),
        steps_taken=5,
    )
    assert state.is_terminal(max_steps=10) is False

    # Scenario: Terminal (all targets rescued)
    terminal_state_no_targets = LearningState(
        agent_id="agent-0",
        agent_position=Position(0, 1),
        remaining_target_a_positions=frozenset(),
        remaining_target_b_positions=frozenset(),
        steps_taken=5,
    )
    assert terminal_state_no_targets.is_terminal(max_steps=10) is True

    # Scenario: Terminal (max steps reached)
    terminal_state_max_steps = LearningState(
        agent_id="agent-0",
        agent_position=Position(0, 0),
        remaining_target_a_positions=frozenset({Position(0, 1)}),
        steps_taken=10,
    )
    assert terminal_state_max_steps.is_terminal(max_steps=10) is True


def test_sprint3_reward_config_values() -> None:
    config = SPRINT3_REWARD_CONFIG
    assert config.move == -1.0
    assert config.invalid_move == -5.0
    assert config.wait == -2.0
    assert config.discovered_cell_bonus == 2.0
    assert config.repeated_cell == -1.5
    assert config.rescued_target_a == 150.0
    assert config.rescued_target_b == 100.0
    assert config.completed_episode_bonus == 50.0


def test_sprint3_reward_calculation_behavior() -> None:
    config = SPRINT3_REWARD_CONFIG

    # Valid Move
    assert calculate_reward(RewardEvent(moved=True, move="right"), config) == -1.0

    # Invalid Move
    assert calculate_reward(RewardEvent(moved=False, move="right"), config) == -5.0

    # Wait Action
    assert calculate_reward(RewardEvent(moved=False, move="wait"), config) == -2.0

    # Discovered Cells
    assert (
        calculate_reward(RewardEvent(moved=True, move="right", newly_discovered_cells=3), config)
        == -1.0 + 3 * 2.0
    )

    # Repeated Cell Penalty
    assert (
        calculate_reward(RewardEvent(moved=True, move="right", repeated_cell=True), config)
        == -1.0 - 1.5
    )

    # Rescue Target A
    assert (
        calculate_reward(
            RewardEvent(moved=True, move="right", rescued_target_type=TargetType.A), config
        )
        == 150.0
    )

    # Rescue Target B
    assert (
        calculate_reward(
            RewardEvent(moved=True, move="right", rescued_target_type=TargetType.B), config
        )
        == 100.0
    )

    # Completed Episode
    assert (
        calculate_reward(RewardEvent(moved=True, move="right", completed_episode=True), config)
        == -1.0 + 50.0
    )

    # Complex combination: Rescue target A + discover 1 cell + completed episode + repeated cell
    event = RewardEvent(
        moved=True,
        move="right",
        newly_discovered_cells=1,
        rescued_target_type=TargetType.A,
        completed_episode=True,
        repeated_cell=True,
    )
    # expected: base (rescued_target_a = 150) + discovered_cell_bonus (2.0) + repeated_cell (-1.5) + completed_episode_bonus (50.0) = 200.5
    assert calculate_reward(event, config) == 200.5
