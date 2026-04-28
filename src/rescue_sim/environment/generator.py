"""Scenario generation for configurable ST04 grids."""

from random import Random

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.grid import Grid, Position


def generate_grid(settings: GridSettings, start: Position) -> Grid:
    """Generate a grid with random obstacles and targets."""
    rng = Random(settings.random_seed)
    obstacles: set[Position] = set()
    candidates: list[Position] = []

    for y in range(settings.height):
        for x in range(settings.width):
            position = Position(x, y)
            if position == start:
                continue
            if rng.random() < settings.obstacle_probability:
                obstacles.add(position)
            else:
                candidates.append(position)

    if settings.target_count > len(candidates):
        raise ValueError("target_count exceeds available non-obstacle cells")

    targets = frozenset(rng.sample(candidates, settings.target_count))
    return Grid(
        width=settings.width,
        height=settings.height,
        obstacles=frozenset(obstacles),
        targets=targets,
    )

