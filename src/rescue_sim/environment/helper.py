"""Helper for coordinating environment components without learning logic."""

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor


class EnvironmentHelper:
    """Coordinates grid, sensor, and movement logic with a deterministic strategy."""

    def __init__(self, grid: Grid, start_pos: Position, sensor_range: int):
        self.grid = grid
        self.sensor = CentralSensor(grid)
        self.movement = MovementModel()
        self.agent_position = start_pos
        self.sensor_range = sensor_range

        # Track active targets as a set for O(1) lookups and updates
        self.active_targets = set(grid.target_a_positions) | set(grid.target_b_positions)
        self.rescued = []
        self.total_reward = 0.0

        # Deterministic cycle of directions: "just goes till its wall then it goes right down left"
        self.direction_sequence = ["forward", "right", "down", "left"]
        self.current_dir_index = 0

    def step(self, step_idx: int) -> dict:
        """Executes one simulation step using environment modules and a deterministic strategy."""
        # 1. Observe the environment using CentralSensor
        self.sensor.observe("0", self.agent_position, self.sensor_range)

        # 2. Select next movement direction deterministically
        action = self.direction_sequence[self.current_dir_index]

        # Check if the current direction is blocked or out of bounds.
        # If so, cycle through directions until we find one that is allowed.
        attempts = 0
        while not self.movement.is_allowed(self.grid, self.agent_position, action) and attempts < 4:
            self.current_dir_index = (self.current_dir_index + 1) % 4
            action = self.direction_sequence[self.current_dir_index]
            attempts += 1

        # If all 4 directions are blocked, wait
        if attempts == 4 and not self.movement.is_allowed(self.grid, self.agent_position, action):
            action = "wait"

        # 3. Apply the movement using MovementModel
        result = self.movement.apply(self.grid, self.agent_position, action)
        self.agent_position = result.end

        # 4. Reward calculation (purely for stats/frontend compatibility)
        reward = -0.1 if result.moved else -1.0
        if self.agent_position in self.active_targets:
            reward = 10.0
            self.active_targets.discard(self.agent_position)
            self.rescued.append(
                {"x": self.agent_position.x, "y": self.agent_position.y, "step": step_idx}
            )

        self.total_reward += reward

        return {
            "id": 0,
            "x": self.agent_position.x,
            "y": self.agent_position.y,
            "action": action,
            "reward": round(reward, 2),
        }

    def has_active_targets(self) -> bool:
        """Returns True if there are active targets remaining on the grid."""
        return len(self.active_targets) > 0

    def get_active_targets_count(self) -> int:
        """Returns the number of active targets remaining."""
        return len(self.active_targets)

    def get_rescued_list(self) -> list[dict]:
        """Returns the list of rescued targets."""
        return self.rescued

    def get_total_reward(self) -> float:
        """Returns the total accumulated reward."""
        return round(self.total_reward, 2)
