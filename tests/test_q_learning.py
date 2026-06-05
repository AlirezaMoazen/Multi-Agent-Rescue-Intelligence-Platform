from random import Random

from rescue_sim.config.settings import AgentSettings, GridSettings, SimulationSettings
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.learning.q_learning import QLearningAgent
from rescue_sim.shared import Action, LearningState


def make_grid() -> Grid:
    return Grid(
        width=2,
        height=1,
        obstacles=frozenset(),
        target_a_positions=frozenset({Position(1, 0)}),
        target_b_positions=frozenset(),
    )


def make_state() -> LearningState:
    return LearningState(agent_id="agent-1", agent_position=Position(0, 0))


def test_choose_action_uses_best_known_valid_action() -> None:
    agent = QLearningAgent(actions=(Action.RIGHT, Action.WAIT), epsilon=0.0)
    state = make_state()
    agent.q_table[state][Action.RIGHT] = 2.0
    agent.q_table[state][Action.WAIT] = 1.0

    action = agent.choose_action(state, valid_actions=(Action.RIGHT, Action.WAIT))

    assert action == Action.RIGHT


def test_update_q_value_applies_learning_rule() -> None:
    agent = QLearningAgent(
        actions=(Action.RIGHT, Action.WAIT),
        learning_rate=0.5,
        discount_factor=0.5,
        epsilon=0.0,
    )
    state = make_state()
    next_state = LearningState(agent_id="agent-1", agent_position=Position(1, 0))
    agent.q_table[next_state][Action.WAIT] = 4.0

    agent.update_q_value(
        state=state,
        action=Action.RIGHT,
        reward=2.0,
        next_state=next_state,
        next_valid_actions=(Action.WAIT,),
    )

    assert agent.q_table[state][Action.RIGHT] == 2.0


def test_state_from_observation_uses_shared_learning_state() -> None:
    grid = make_grid()
    sensor = CentralSensor(grid)
    observation = sensor.observe("agent-1", Position(0, 0), sensor_range=1)
    agent = QLearningAgent(actions=(Action.RIGHT, Action.WAIT), epsilon=0.0)

    state = agent.state_from_observation(
        observation=observation,
        grid=grid,
        found_targets=frozenset(),
        steps_taken=0,
    )

    assert isinstance(state, LearningState)
    assert state.agent_position == Position(0, 0)
    assert state.visible_target_a_positions == frozenset({Position(1, 0)})
    assert state.remaining_target_a_positions == frozenset({Position(1, 0)})


def test_training_episode_updates_q_table_and_records_success() -> None:
    agent = QLearningAgent(actions=(Action.RIGHT, Action.WAIT), epsilon=0.0)

    metrics = agent.train_episode(
        grid=make_grid(),
        start_position=Position(0, 0),
        sensor_range=0,
        max_steps=3,
    )

    assert metrics.success is True
    assert metrics.targets_found == 1
    assert metrics.steps == 1
    assert any(
        values[Action.RIGHT] > 0
        for state, values in agent.q_table.items()
        if state.agent_position == Position(0, 0)
    )


def test_training_returns_aggregate_metrics() -> None:
    agent = QLearningAgent(actions=(Action.RIGHT, Action.WAIT), epsilon=0.0, rng=Random(1))

    metrics = agent.train(
        grid_settings=GridSettings(
            width=2,
            height=1,
            obstacle_probability=0.0,
            target_a_count=1,
            target_b_count=0,
            random_seed=1,
        ),
        agent_settings=AgentSettings(start_x=0, start_y=0, sensor_range=0),
        simulation_settings=SimulationSettings(max_steps=3),
        episodes=3,
    )

    assert metrics.episodes == 3
    assert metrics.successes == 3
    assert metrics.average_steps == 1
    assert len(metrics.episode_metrics) == 3
    assert list(agent.best_policy().values()) == [Action.RIGHT, Action.RIGHT]
