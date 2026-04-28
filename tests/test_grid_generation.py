from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Position


def test_generated_grid_keeps_start_free() -> None:
    start = Position(0, 0)
    grid = generate_grid(
        GridSettings(
            width=5,
            height=5,
            obstacle_probability=0.2,
            target_count=3,
            random_seed=1,
        ),
        start=start,
    )

    assert start not in grid.obstacles
    assert start not in grid.targets
    assert len(grid.targets) == 3

