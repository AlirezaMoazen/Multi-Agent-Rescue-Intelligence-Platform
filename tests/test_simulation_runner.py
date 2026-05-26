from rescue_sim.config.settings import AgentSettings, GridSettings, SimulationSettings
from rescue_sim.environment.grid import Position
from rescue_sim.simulation.runner import SimulationRunner


def test_runner_succeeds_immediately_when_no_targets_exist() -> None:
    runner = SimulationRunner()

    result = runner.run(
        GridSettings(
            width=3,
            height=3,
            obstacle_probability=0.0,
            target_a_count=0,
            target_b_count=0,
            random_seed=1,
        ),
        AgentSettings(start_x=0, start_y=0, sensor_range=1),
        SimulationSettings(max_steps=5),
    )

    assert result.success is True
    assert result.steps_taken == 0
    assert result.targets_found == 0
    assert result.initial_observation.agent_position == Position(0, 0)


def test_runner_applies_movement_plan() -> None:
    runner = SimulationRunner(moves=("right",))

    result = runner.run(
        GridSettings(
            width=3,
            height=3,
            obstacle_probability=0.0,
            target_a_count=0,
            target_b_count=1,
            random_seed=2,
        ),
        AgentSettings(start_x=0, start_y=0, sensor_range=1),
        SimulationSettings(max_steps=1),
    )

    assert result.steps_taken == 1
    assert result.final_position.x == 1
    assert result.final_position.y == 0
    assert result.history[0].move == "right"
    assert result.history[0].reason == "ok"
    assert result.history[0].observation.agent_position == Position(1, 0)


def test_runner_records_found_target() -> None:
    runner = SimulationRunner(moves=("right",))

    result = runner.run(
        GridSettings(
            width=2,
            height=1,
            obstacle_probability=0.0,
            target_a_count=1,
            target_b_count=0,
            random_seed=1,
        ),
        AgentSettings(start_x=0, start_y=0, sensor_range=1),
        SimulationSettings(max_steps=1),
    )

    assert result.success is True
    assert result.targets_found == 1
    assert result.history[0].target_found == "A"
    assert result.history[0].observation.target_types[Position(1, 0)] == "A"


def test_runner_uses_central_sensor_range() -> None:
    runner = SimulationRunner(moves=("wait",))

    result = runner.run(
        GridSettings(
            width=3,
            height=3,
            obstacle_probability=0.0,
            target_a_count=1,
            target_b_count=0,
            random_seed=1,
        ),
        AgentSettings(start_x=0, start_y=0, sensor_range=0),
        SimulationSettings(max_steps=1),
    )

    assert result.initial_observation.visible_cells == frozenset({Position(0, 0)})
    assert result.history[0].observation.visible_cells == frozenset({Position(0, 0)})


def test_runner_rejects_empty_movement_plan() -> None:
    runner = SimulationRunner(moves=())

    try:
        runner.run(
            GridSettings(
                width=3,
                height=3,
                obstacle_probability=0.0,
                target_a_count=1,
                target_b_count=0,
                random_seed=1,
            ),
            AgentSettings(start_x=0, start_y=0, sensor_range=1),
            SimulationSettings(max_steps=5),
        )
    except ValueError as error:
        assert str(error) == "at least one movement command is required"
    else:
        raise AssertionError("Expected ValueError for empty movement plan")
