"""Scenario generation for configurable ST02 grids."""
#ADRIANA HERRERO (Task 1, ST02)

from collections import deque
from random import Random

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.grid import Grid, Position

MAX_GENERATION_ATTEMPTS = 100


class _GenerationRetry(Exception):
    """Internal signal for an invalid random candidate."""


def generate_grid(settings: GridSettings, start: Position) -> Grid:
    """Generate a reachable grid with random obstacles and targets."""
    rng = Random(settings.random_seed)

    #Initial position in grid.
    if not (0 <= start.x < settings.width and 0 <= start.y < settings.height):
        raise ValueError("start position must be inside the grid")

    total_targets = settings.target_a_count + settings.target_b_count
    if total_targets > settings.width * settings.height - 1:
        raise ValueError("target counts exceed available non-obstacle cells")

    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _generate_candidate_grid(settings, start, rng)
        except _GenerationRetry:
            continue
        if _targets_are_reachable(grid, start):
            return grid

    raise ValueError("could not generate a reachable grid with the given settings")


def _generate_candidate_grid(settings: GridSettings, start: Position, rng: Random) -> Grid:
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

    total_targets = settings.target_a_count + settings.target_b_count

    if total_targets > len(candidates):
        raise _GenerationRetry

    #Target b never on top of target a and viceversa
    target_a_positions = set(rng.sample(candidates, settings.target_a_count))
    remaining_candidates = [
        position for position in candidates if position not in target_a_positions
    ]
    target_b_positions = set(rng.sample(remaining_candidates, settings.target_b_count))

    return Grid(
        width=settings.width,
        height=settings.height,
        obstacles=frozenset(obstacles),
        target_a_positions=frozenset(target_a_positions),
        target_b_positions=frozenset(target_b_positions),
    )


def _targets_are_reachable(grid: Grid, start: Position) -> bool:
    targets = grid.target_a_positions | grid.target_b_positions
    if not targets:
        return True

    reachable = _reachable_positions(grid, start)
    return targets.issubset(reachable)


def _reachable_positions(grid: Grid, start: Position) -> set[Position]:
    queue: deque[Position] = deque([start])
    seen = {start}

    while queue:
        current = queue.popleft()
        for neighbor in _neighbors(current):
            if neighbor in seen or not grid.is_valid_position(neighbor):
                continue
            seen.add(neighbor)
            queue.append(neighbor)

    return seen


def _neighbors(position: Position) -> tuple[Position, Position, Position, Position]:
    return (
        Position(position.x + 1, position.y),
        Position(position.x, position.y + 1),
        Position(position.x - 1, position.y),
        Position(position.x, position.y - 1),
    )
