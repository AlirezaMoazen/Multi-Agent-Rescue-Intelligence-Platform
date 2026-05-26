"""Central sensor tests"""
#CRISTINA MARCOS ALONSO (task 2, ST02)

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.sensors import CentralSensor


def make_grid() -> Grid:
    # Small fixed grid used in all central sensor tests.
    return Grid(
        width=5,
        height=5,
        obstacles=frozenset({Position(1, 2), Position(3, 3)}),
        target_a_positions=frozenset({Position(2, 1)}),
        target_b_positions=frozenset({Position(4, 4)}),
    )


def test_central_sensor_returns_visible_cells_inside_range() -> None:
    # The sensor should return only the cells inside the agent sensor range.
    sensor = CentralSensor(make_grid())

    observation = sensor.observe(agent_id="agent-1", position=Position(2, 2), sensor_range=1)

    assert observation.agent_id == "agent-1"
    assert observation.agent_position == Position(2, 2)
    assert observation.visible_cells == frozenset(
        {
            Position(2, 2),
            Position(2, 1),
            Position(2, 3),
            Position(1, 2),
            Position(3, 2),
        }
    )


def test_central_sensor_excludes_cells_outside_grid() -> None:
    # If the agent is close to a border, positions outside the grid are ignored.
    sensor = CentralSensor(make_grid())

    observation = sensor.observe(agent_id="agent-1", position=Position(0, 0), sensor_range=2)

    assert all(0 <= cell.x < 5 and 0 <= cell.y < 5 for cell in observation.visible_cells)
    assert Position(-1, 0) not in observation.visible_cells
    assert Position(0, -1) not in observation.visible_cells


def test_central_sensor_detects_visible_obstacles_and_targets() -> None:
    # The observation should separate visible obstacles and visible targets.
    sensor = CentralSensor(make_grid())

    observation = sensor.observe(agent_id="agent-1", position=Position(2, 2), sensor_range=1)

    assert observation.obstacles == frozenset({Position(1, 2)})
    assert observation.targets == frozenset({Position(2, 1)})
    assert observation.target_types == {Position(2, 1): "A"}


def test_central_sensor_keeps_shared_discovered_memory() -> None:
    # The central sensor keeps one shared memory for all agents.
    sensor = CentralSensor(make_grid())

    first = sensor.observe(agent_id="agent-1", position=Position(2, 2), sensor_range=1)
    second = sensor.observe(agent_id="agent-2", position=Position(4, 3), sensor_range=1)

    assert Position(2, 1) in sensor.discovered_targets
    assert Position(4, 4) in sensor.discovered_targets
    assert sensor.discovered_targets[Position(2, 1)] == "A"
    assert sensor.discovered_targets[Position(4, 4)] == "B"
    assert sensor.discovered_cells == first.visible_cells | second.visible_cells
    assert sensor.agent_positions == {
        "agent-1": Position(2, 2),
        "agent-2": Position(4, 3),
    }


def test_central_sensor_reports_only_new_discoveries_once() -> None:
    # If the same agent observes the same place twice, nothing is new the second time.
    sensor = CentralSensor(make_grid())

    first = sensor.observe(agent_id="agent-1", position=Position(2, 2), sensor_range=1)
    second = sensor.observe(agent_id="agent-1", position=Position(2, 2), sensor_range=1)

    assert first.newly_discovered_cells == first.visible_cells
    assert first.newly_discovered_targets == frozenset({Position(2, 1)})
    assert second.newly_discovered_cells == frozenset()
    assert second.newly_discovered_targets == frozenset()


def test_central_sensor_rejects_invalid_input() -> None:
    # Invalid agent positions and invalid sensor ranges should fail clearly.
    sensor = CentralSensor(make_grid())

    try:
        sensor.observe(agent_id="agent-1", position=Position(5, 0), sensor_range=1)
    except ValueError as error:
        assert str(error) == "agent position must be inside the grid"
    else:
        raise AssertionError("Expected ValueError for out-of-bounds agent position")

    try:
        sensor.observe(agent_id="agent-1", position=Position(0, 0), sensor_range=-1)
    except ValueError as error:
        assert str(error) == "sensor_range must be non-negative"
    else:
        raise AssertionError("Expected ValueError for negative sensor range")
