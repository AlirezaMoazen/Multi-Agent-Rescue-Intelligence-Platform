"""Deterministic movement model for rescue agents."""

from dataclasses import dataclass

from rescue_sim.environment.grid import Position


MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "forward": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "wait": (0, 0),
}


@dataclass(frozen=True, slots=True)
class MovementResult:
    """Result of applying a movement command."""

    start: Position
    requested: Position
    end: Position
    move: str
    moved: bool
    reason: str


class MovementModel:
    """Validates and applies deterministic grid movements."""

    def next_position(self, position: Position, move: str) -> Position:
        if move not in MOVE_DELTAS:
            raise ValueError(f"unknown move: {move}")

        dx, dy = MOVE_DELTAS[move]
        return Position(position.x + dx, position.y + dy)

    def _contains(self, grid: object, position: Position) -> bool:
        if hasattr(grid, "contains"):
            return grid.contains(position)
        if hasattr(grid, "is_within_bounds"):
            return grid.is_within_bounds(position.x, position.y)
        return 0 <= position.x < grid.width and 0 <= position.y < grid.height

    def _is_blocked(self, grid: object, position: Position) -> bool:
        if hasattr(grid, "is_blocked"):
            return grid.is_blocked(position)
        if hasattr(grid, "get_cell"):
            return grid.get_cell(position.x, position.y) == 1
        return position in grid.obstacles

    def is_allowed(self, grid: object, position: Position, move: str) -> bool:
        requested = self.next_position(position, move)
        return self._contains(grid, requested) and not self._is_blocked(grid, requested)

    def allowed_moves(self, grid: object, position: Position) -> dict[str, Position]:
        """Return moves accepted by the grid and their resulting positions."""
        return {
            move: self.next_position(position, move)
            for move in MOVE_DELTAS
            if self.is_allowed(grid, position, move)
        }

    def apply(self, grid: object, position: Position, move: str) -> MovementResult:
        """Apply a move; invalid moves leave the agent at the original position."""
        requested = self.next_position(position, move)

        if not self._contains(grid, requested):
            return MovementResult(
                start=position,
                requested=requested,
                end=position,
                move=move,
                moved=False,
                reason="out_of_bounds",
            )

        if self._is_blocked(grid, requested):
            return MovementResult(
                start=position,
                requested=requested,
                end=position,
                move=move,
                moved=False,
                reason="blocked",
            )

        return MovementResult(
            start=position,
            requested=requested,
            end=requested,
            move=move,
            moved=move != "wait",
            reason="ok",
        )

    def apply_to_agent(self, agent: object, move: str) -> MovementResult:
        """Move an object with x, y, grid, and optional history attributes."""
        result = self.apply(agent.grid, Position(agent.x, agent.y), move)
        if result.moved:
            agent.x = result.end.x
            agent.y = result.end.y
            if hasattr(agent, "history"):
                agent.history.append((agent.x, agent.y))
        return result
