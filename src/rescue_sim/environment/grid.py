"""Grid-world primitives for rescue scenarios."""

from dataclasses import dataclass
#ADRIANA HERRERO (task 1, ST02)

@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class Grid:
    x_size: int
    y_size: int
    walls: frozenset[Position]
    target_a_positions: frozenset[Position]
    target_b_positions: frozenset[Position]

    def contains(self, position: Position) -> bool:
        return 0 <= position.x < self.x_size and 0 <= position.y < self.y_size

    def is_blocked(self, position: Position) -> bool:
        return position in self.walls

    def is_valid_position(self, position: Position) -> bool:
        return self.contains(position) and not self.is_blocked(position)

    def has_target(self, position: Position) -> bool:
        return (
            position in self.target_a_positions
            or position in self.target_b_positions
        )

    def target_type_at(self, position: Position) -> str | None:
        if position in self.target_a_positions:
            return "A"
        if position in self.target_b_positions:
            return "B"
        return None
