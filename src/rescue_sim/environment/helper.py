"""Helper for coordinating environment components with systematic grid coverage."""

from collections import deque

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor


def find_path_bfs(grid: Grid, start: Position, target: Position) -> list[str]:
    """Finds the shortest list of movement directions from start to target using BFS."""
    if start == target:
        return []

    # Using deque for efficient O(1) pops and appends in BFS
    queue: deque[tuple[Position, list[str]]] = deque([(start, [])])
    visited = {start}
    movement = MovementModel()

    while queue:
        current_pos, path_actions = queue.popleft()

        if current_pos == target:
            return path_actions

        allowed = movement.allowed_moves(grid, current_pos)
        for move, next_pos in allowed.items():
            if move == "wait":
                continue
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path_actions + [move]))

    return []  # No path found (e.g. if the target cell is fully enclosed by obstacles)


class EnvironmentHelper:
    """Coordinates grid, sensor, and movement logic with a systematic snake-sweep strategy."""

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

        # Build Boustrophedon (snake-like) sweep list of coordinates to cover the entire grid
        self.visit_order: list[Position] = []
        for y in range(grid.height):
            # Sweep left-to-right on even rows, right-to-left on odd rows
            if y % 2 == 0:
                x_range = range(grid.width)
            else:
                x_range = range(grid.width - 1, -1, -1)

            for x in x_range:
                pos = Position(x, y)
                if pos not in grid.obstacles:
                    self.visit_order.append(pos)

        # Track visited cells to avoid backtracking to cells we've already searched
        self.visited_cells = set()
        self.current_path_actions: list[str] = []

    def step(self, step_idx: int) -> dict:
        """Executes one simulation step using environment modules and a systematic snake-sweep strategy."""
        # 1. Observe the environment using CentralSensor
        self.sensor.observe("0", self.agent_position, self.sensor_range)

        # Mark current position as visited
        self.visited_cells.add(self.agent_position)

        # 2. Get next movement action
        action = None

        # Execute any pre-planned moves along our path
        while self.current_path_actions:
            next_action = self.current_path_actions.pop(0)
            if self.movement.is_allowed(self.grid, self.agent_position, next_action):
                action = next_action
                break

        # If we don't have a plan, find the next unvisited cell in our sweep sequence
        if action is None:
            for target_cell in self.visit_order:
                if target_cell not in self.visited_cells:
                    # Find a BFS path to it around obstacles
                    path = find_path_bfs(self.grid, self.agent_position, target_cell)
                    if path:
                        self.current_path_actions = path
                        break

            if self.current_path_actions:
                action = self.current_path_actions.pop(0)
            else:
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
