"""
Simulation runner for ST02 scenarios.

1. Generates a grid.
2. Places the agent at the start position.
3. Repeats movement commands.
4. Validates each movement with MovementModel.
5. Updates the agent position.
6. Checks whether Target A or Target B was found.
7. Stores the simulation history.
8. Returns basic metrics.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rescue_sim.config.settings import AgentSettings, GridSettings, SimulationSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor, Observation


DEFAULT_MOVES: tuple[str, ...] = ("right", "down", "left", "up", "wait") #cannot be modified after creation


@dataclass(frozen=True, slots=True)
class SimulationStep:
    step: int
    start: Position
    move: str
    requested: Position
    end: Position
    moved: bool
    reason: str
    target_found: str | None
    observation: Observation


@dataclass(frozen=True, slots=True)
class SimulationResult:
    grid: Grid
    start_position: Position
    final_position: Position
    initial_observation: Observation
    steps_taken: int
    targets_found: int
    found_targets: frozenset[Position]
    success: bool
    history: tuple[SimulationStep, ...]


class SimulationRunner:
    """Coordinates one generated scenario, one agent, and one strategy."""

    def __init__(
        self,
        movement_model: MovementModel | None = None,
        moves: Sequence[str] = DEFAULT_MOVES,
    ) -> None:
        self.movement_model = movement_model or MovementModel()
        self.moves = tuple(moves)

    def run(
        self,
        grid_settings: GridSettings,
        agent_settings: AgentSettings,
        simulation_settings: SimulationSettings,
        moves: Sequence[str] | None = None,
    ) -> SimulationResult:
        start_position = Position(agent_settings.start_x, agent_settings.start_y)
        grid = generate_grid(grid_settings, start=start_position)
        sensor = CentralSensor(grid)

        #this part is to get what the sensor sees
        initial_observation = sensor.observe(
            agent_id="agent-1",
            position=start_position,
            sensor_range=agent_settings.sensor_range,
        )
        
        all_targets = grid.target_a_positions | grid.target_b_positions

        position = start_position
        found_targets: set[Position] = set()
        history: list[SimulationStep] = []
        move_plan = tuple(moves) if moves is not None else self.moves

        if not move_plan:
            raise ValueError("at least one movement command is required")

        for step_index in range(simulation_settings.max_steps):
            if found_targets == all_targets:
                break

            move = move_plan[step_index % len(move_plan)]
            movement_result = self.movement_model.apply(grid, position, move)
            position = movement_result.end
            target_found = grid.target_type_at(position)
            observation = sensor.observe(
                agent_id="agent-1",
                position=position,
                sensor_range=agent_settings.sensor_range,
            )

            if target_found is not None:
                found_targets.add(position)

            history.append(
                SimulationStep(
                    step=step_index + 1,
                    start=movement_result.start,
                    move=move,
                    requested=movement_result.requested,
                    end=movement_result.end,
                    moved=movement_result.moved,
                    reason=movement_result.reason,
                    target_found=target_found,
                    observation=observation,
                )
            )

        return SimulationResult(
            grid=grid,
            start_position=start_position,
            final_position=position,
            initial_observation=initial_observation,
            steps_taken=len(history),
            targets_found=len(found_targets),
            found_targets=frozenset(found_targets),
            success=found_targets == all_targets,
            history=tuple(history),
        )
