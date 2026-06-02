from rescue_sim.shared import Action, Grid, LearningState, Position, RewardConfig, RewardEvent
from rescue_sim.shared import TargetType, calculate_reward


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
