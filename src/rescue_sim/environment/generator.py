"""Scenario generation for configurable ST02 grids."""
#ADRIANA HERRERO (Task 1, ST02)

from random import Random

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.grid import Grid, Position


def generate_grid(settings: GridSettings, start: Position) -> Grid:
    """Generate a grid with random obstacles and targets."""
    rng = Random(settings.random_seed)

    #Initial position in grid.
    if not (0 <= start.x < settings.width and 0 <= start.y < settings.height):
        raise ValueError("start position must be inside the grid")

    walls: set[Position] = set()
    candidates: list[Position] = []     

    for y in range(settings.height):
        for x in range(settings.width):
            position = Position(x, y)
            if position == start:
                continue
            if rng.random() < settings.obstacle_probability:
                walls.add(position)
            else:
                candidates.append(position)

    total_targets = settings.target_a_count + settings.target_b_count

    if total_targets > len(candidates):
        raise ValueError("target counts exceed available non-obstacle cells")

    #Target b never on top of target a and viceversa
    target_a_positions = set(rng.sample(candidates, settings.target_a_count))
    remaining_candidates = [
        position for position in candidates if position not in target_a_positions
    ]
    target_b_positions = set(rng.sample(remaining_candidates, settings.target_b_count))

    return Grid(
        x_size=settings.width,
        y_size=settings.height,
        walls=frozenset(walls),
        target_a_positions=frozenset(target_a_positions),
        target_b_positions=frozenset(target_b_positions),
    )
