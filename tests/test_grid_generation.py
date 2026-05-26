from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
#ADRIANA HERRERO (Task 1, ST02)

# Checks that the agent start cell is not used for obstacles or targets.
def test_generated_grid_keeps_start_free() -> None:
    start = Position(0, 0)
    grid = generate_grid(
        GridSettings(
            width=5,
            height=5,
            obstacle_probability=0.2,
            target_a_count=2,
            target_b_count=1,
            random_seed=1,
        ),
        start=start,
    )

    assert start not in grid.walls
    assert start not in grid.target_a_positions
    assert start not in grid.target_b_positions
    assert len(grid.target_a_positions) == 2
    assert len(grid.target_b_positions) == 1


# Checks that Target A and Target B positions are generated separately.
def test_target_a_and_b_do_not_overlap() -> None:
    start = Position(0, 0)
    grid = generate_grid(
        GridSettings(
            width=5,
            height=5,
            obstacle_probability=0.1,
            target_a_count=3,
            target_b_count=3,
            random_seed=2,
        ),
        start=start,
    )

    assert grid.target_a_positions.isdisjoint(grid.target_b_positions)


# Checks that targets are only placed on free cells.
def test_targets_do_not_overlap_obstacles() -> None:
    start = Position(0, 0)
    grid = generate_grid(
        GridSettings(
            width=6,
            height=6,
            obstacle_probability=0.3,
            target_a_count=2,
            target_b_count=2,
            random_seed=3,
        ),
        start=start,
    )

    all_targets = grid.target_a_positions | grid.target_b_positions

    assert all_targets.isdisjoint(grid.walls)


# Checks that positions outside the grid bounds are not valid.
def test_grid_rejects_out_of_bounds_position() -> None:
    grid = Grid(
        x_size=3,
        y_size=3,
        walls=frozenset(),
        target_a_positions=frozenset(),
        target_b_positions=frozenset(),
    )

    assert not grid.is_valid_position(Position(3, 0))
    assert not grid.is_valid_position(Position(0, 3))
    assert not grid.is_valid_position(Position(-1, 0))
    assert not grid.is_valid_position(Position(0, -1))


# Checks that obstacle cells are not valid movement destinations.
def test_grid_rejects_obstacle_position() -> None:
    obstacle = Position(1, 1)
    grid = Grid(
        x_size=3,
        y_size=3,
        walls=frozenset({obstacle}),
        target_a_positions=frozenset(),
        target_b_positions=frozenset(),
    )

    assert not grid.is_valid_position(obstacle)


# Checks that an empty in-bounds cell is a valid position.
def test_grid_accepts_free_position() -> None:
    grid = Grid(
        x_size=3,
        y_size=3,
        walls=frozenset({Position(1, 1)}),
        target_a_positions=frozenset(),
        target_b_positions=frozenset(),
    )

    assert grid.is_valid_position(Position(0, 0))


# Checks that using the same random seed generates the same grid.
def test_generated_grid_is_reproducible_with_seed() -> None:
    start = Position(0, 0)
    settings = GridSettings(
        width=6,
        height=6,
        obstacle_probability=0.25,
        target_a_count=2,
        target_b_count=2,
        random_seed=4,
    )

    first_grid = generate_grid(settings, start=start)
    second_grid = generate_grid(settings, start=start)

    assert first_grid == second_grid


# Checks that generation fails when targets cannot fit in free cells.
def test_generate_grid_rejects_too_many_targets() -> None:
    start = Position(0, 0)

    try:
        generate_grid(
            GridSettings(
                width=2,
                height=2,
                obstacle_probability=0.0,
                target_a_count=2,
                target_b_count=2,
                random_seed=5,
            ),
            start=start,
        )
    except ValueError as error:
        assert str(error) == "target counts exceed available non-obstacle cells"
    else:
        raise AssertionError("Expected ValueError for too many targets")


# Checks that the grid reports whether a position contains any target.
def test_grid_detects_target_positions() -> None:
    target_a = Position(1, 0)
    target_b = Position(2, 0)
    empty = Position(0, 0)
    grid = Grid(
        x_size=3,
        y_size=3,
        walls=frozenset(),
        target_a_positions=frozenset({target_a}),
        target_b_positions=frozenset({target_b}),
    )

    assert grid.has_target(target_a)
    assert grid.has_target(target_b)
    assert not grid.has_target(empty)


# Checks that the grid returns the correct target type at a position.
def test_grid_returns_target_type_at_position() -> None:
    target_a = Position(1, 0)
    target_b = Position(2, 0)
    empty = Position(0, 0)
    grid = Grid(
        x_size=3,
        y_size=3,
        walls=frozenset(),
        target_a_positions=frozenset({target_a}),
        target_b_positions=frozenset({target_b}),
    )

    assert grid.target_type_at(target_a) == "A"
    assert grid.target_type_at(target_b) == "B"
    assert grid.target_type_at(empty) is None


# Checks that grid generation rejects a start position outside the grid.
def test_generate_grid_rejects_start_outside_grid() -> None:
    start = Position(5, 5)

    try:
        generate_grid(
            GridSettings(
                width=3,
                height=3,
                obstacle_probability=0.0,
                target_a_count=1,
                target_b_count=1,
                random_seed=6,
            ),
            start=start,
        )
    except ValueError as error:
        assert str(error) == "start position must be inside the grid"
    else:
        raise AssertionError("Expected ValueError for start outside grid")


# Checks that grid generation supports scenarios with no targets.
def test_generate_grid_allows_zero_targets() -> None:
    start = Position(0, 0)
    grid = generate_grid(
        GridSettings(
            width=3,
            height=3,
            obstacle_probability=0.0,
            target_a_count=0,
            target_b_count=0,
            random_seed=7,
        ),
        start=start,
    )

    assert grid.target_a_positions == frozenset()
    assert grid.target_b_positions == frozenset()
