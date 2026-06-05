"""Central sensor observations for rescue agents."""
#CRISTINA MARCOS ALONSO (task 2, ST02)
from __future__ import annotations

from dataclasses import dataclass
from rescue_sim.environment.grid import Grid, Position


@dataclass(frozen=True)
class Observation:
    """Information sent from the central sensor back to one agent."""

    # Agent ID that asked the central sensor for information (Initially we only have one agents)
    agent_id: str
    # Position from where the agent is observing the grid 
    agent_position: Position
    # All cells the agent can see from its current position
    visible_cells: frozenset[Position]
    # Cells that were seen for the first time in this observation 
    newly_discovered_cells: frozenset[Position]
    # Visible cells which are ocupied by obstacles
    obstacles: frozenset[Position]
    # Visible cells that contain a target.
    targets: frozenset[Position]
    # Type of every visible target, for example A or B.
    target_types: dict[Position, str]
    # Targets that were discovered for the first time in this observation.
    newly_discovered_targets: frozenset[Position]


class CentralSensor:
    """Centralized sensor model with shared discovered-map memory.

    The central sensor has access to the real grid, but each agent only receives
    the cells that are visible from its current position and sensor range.
    """

    def __init__(self, grid: Grid):
        # Central sensor receives the complete grid from the simulator
        self.grid = grid
        # Memory shared by all agents: cells already seen by at least one agent - We want to keep track of the discovered area
        self._discovered_cells: set[Position] = set()
        # Memory shared by all agents: targets already found and their type - Again we want to keep track of the targets discovered
        self._discovered_targets: dict[Position, str] = {}
        # Last known position reported by each agent
        self._agent_positions: dict[str, Position] = {}
        # Last observation generated for each agent - So the agent knows the obstacles/targets and already discovered cells
        self._latest_observations: dict[str, Observation] = {}

    @property
    def discovered_cells(self) -> frozenset[Position]:
        """All cells that have been observed by any agent."""
        return frozenset(self._discovered_cells)

    @property
    def discovered_targets(self) -> dict[Position, str]:
        """Targets discovered by any agent, indexed by position."""
        return dict(self._discovered_targets)

    @property
    def agent_positions(self) -> dict[str, Position]:
        """Latest position reported by each agent."""
        return dict(self._agent_positions)

    @property
    def latest_observations(self) -> dict[str, Observation]:
        """Latest observation sent to each agent."""
        return dict(self._latest_observations)

    def observe(self, agent_id: str | int, position: Position, sensor_range: int) -> Observation:
        """Receive an agent position and return the visible local observation."""
        # The sensor range cannot be negative because it represents a distance.
        if sensor_range < 0:
            raise ValueError("sensor_range must be non-negative")
        # The agent must be inside the grid before asking for an observation.
        if not self.grid.contains(position):
            raise ValueError("agent position must be inside the grid")

        # We store all agent ids as strings so ids have a consistent format.
        normalized_agent_id = str(agent_id)
        # First we calculate which cells are visible from the agent position.
        visible_cells = self._visible_cells_from(position, sensor_range)
        # We keep the old memory to know what is new in this observation.
        previously_discovered_cells = set(self._discovered_cells)
        previously_discovered_targets = set(self._discovered_targets)

        # From the visible cells, we separate obstacles and targets.
        obstacles = frozenset(cell for cell in visible_cells if self.grid.is_blocked(cell))
        target_types = {
            cell: target_type
            for cell in visible_cells
            if (target_type := self.grid.target_type_at(cell)) is not None
        }
        targets = frozenset(target_types)

        # Now the central memory is updated with the new information.
        self._discovered_cells.update(visible_cells)
        self._discovered_targets.update(target_types)
        self._agent_positions[normalized_agent_id] = position

        # This is the message that the central sensor sends back to the agent.
        observation = Observation(
            agent_id=normalized_agent_id,
            agent_position=position,
            visible_cells=visible_cells,
            newly_discovered_cells=frozenset(
                cell for cell in visible_cells if cell not in previously_discovered_cells
            ),
            obstacles=obstacles,
            targets=targets,
            target_types=target_types,
            newly_discovered_targets=frozenset(
                target for target in targets if target not in previously_discovered_targets
            ),
        )
        self._latest_observations[normalized_agent_id] = observation

        return observation

    def _visible_cells_from(self, center: Position, sensor_range: int) -> frozenset[Position]:
        # We use Manhattan distance because agents move in four grid directions.
        visible_cells: set[Position] = set()
        for y in range(center.y - sensor_range, center.y + sensor_range + 1):
            for x in range(center.x - sensor_range, center.x + sensor_range + 1):
                position = Position(x, y)
                is_visible = self._distance(center, position) <= sensor_range
                # Cells outside the grid are ignored.
                if self.grid.contains(position) and is_visible:
                    visible_cells.add(position)
        return frozenset(visible_cells)

    @staticmethod
    def _distance(first: Position, second: Position) -> int:
        # Manhattan distance is the number of horizontal and vertical moves.
        return abs(first.x - second.x) + abs(first.y - second.y)
