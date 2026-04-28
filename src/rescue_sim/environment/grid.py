"""Grid-world primitives for rescue scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    obstacles: frozenset[Position]
    targets: frozenset[Position]

    def contains(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def is_blocked(self, position: Position) -> bool:
        return position in self.obstacles

