from dataclasses import fields

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel


def make_grid() -> Grid:
    grid_fields = {field.name for field in fields(Grid)}
    size_kwargs = (
        {"x_size": 4, "y_size": 3}
        if "x_size" in grid_fields
        else {"width": 4, "height": 3}
    )
    obstacle_kwargs = (
        {"walls": frozenset({Position(2, 1)})}
        if "walls" in grid_fields
        else {"obstacles": frozenset({Position(2, 1)})}
    )
    target_kwargs = (
        {
            "target_a_positions": frozenset({Position(3, 2)}),
            "target_b_positions": frozenset(),
        }
        if "target_a_positions" in grid_fields
        else {"targets": frozenset({Position(3, 2)})}
    )

    return Grid(
        **size_kwargs,
        **obstacle_kwargs,
        **target_kwargs,
    )


class SharedStyleGrid:
    def __init__(self) -> None:
        self.blocked = {(2, 1)}

    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < 4 and 0 <= y < 3

    def get_cell(self, x: int, y: int) -> int:
        if not self.is_within_bounds(x, y) or (x, y) in self.blocked:
            return 1
        return 0


class SharedStyleAgent:
    def __init__(self) -> None:
        self.x = 1
        self.y = 1
        self.grid = SharedStyleGrid()
        self.history = [(1, 1)]


def test_valid_move_updates_position() -> None:
    model = MovementModel()

    result = model.apply(make_grid(), Position(1, 1), "forward")

    assert result.moved is True
    assert result.end == Position(1, 0)
    assert result.reason == "ok"


def test_move_into_obstacle_is_rejected() -> None:
    model = MovementModel()

    result = model.apply(make_grid(), Position(1, 1), "right")

    assert result.moved is False
    assert result.requested == Position(2, 1)
    assert result.end == Position(1, 1)
    assert result.reason == "blocked"


def test_move_outside_grid_is_rejected() -> None:
    model = MovementModel()

    result = model.apply(make_grid(), Position(0, 0), "up")

    assert result.moved is False
    assert result.requested == Position(0, -1)
    assert result.end == Position(0, 0)
    assert result.reason == "out_of_bounds"


def test_wait_is_allowed_without_changing_position() -> None:
    model = MovementModel()

    result = model.apply(make_grid(), Position(1, 1), "wait")

    assert result.moved is False
    assert result.end == Position(1, 1)
    assert result.reason == "ok"


def test_allowed_moves_are_ready_for_sensor_or_simulation_use() -> None:
    model = MovementModel()

    allowed = model.allowed_moves(make_grid(), Position(1, 1))

    assert allowed == {
        "up": Position(1, 0),
        "forward": Position(1, 0),
        "down": Position(1, 2),
        "left": Position(0, 1),
        "wait": Position(1, 1),
    }


def test_can_move_shared_style_agent_used_by_visualization() -> None:
    model = MovementModel()
    agent = SharedStyleAgent()

    result = model.apply_to_agent(agent, "forward")

    assert result.moved is True
    assert (agent.x, agent.y) == (1, 0)
    assert agent.history[-1] == (1, 0)
