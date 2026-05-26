import random
import numpy as np
from typing import Tuple, Dict, List

class Grid:
    """
    Represents the environment grid.
    Values:
        0: Empty space
        1: Wall
        2: Special/Goal (or other designated entity)
    """
    def __init__(self, x_size: int, y_size: int):
        self.x_size = x_size
        self.y_size = y_size
        self.data = np.zeros((y_size, x_size), dtype=int)
        
    def set_wall(self, x: int, y: int):
        if self.is_within_bounds(x, y):
            self.data[y, x] = 1
            
    def set_special(self, x: int, y: int):
        if self.is_within_bounds(x, y):
            self.data[y, x] = 2

    def set_target_a(self, x: int, y: int):
        if self.is_within_bounds(x, y):
            self.data[y, x] = 2

    def set_target_b(self, x: int, y: int):
        if self.is_within_bounds(x, y):
            self.data[y, x] = 3
            
    def get_cell(self, x: int, y: int) -> int:
        if self.is_within_bounds(x, y):
            return self.data[y, x]
        return 1  # Treat out of bounds as a wall
        
    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.x_size and 0 <= y < self.y_size


class Sensor:
    """
    Sensor class that knows its agent's location and can sense the surrounding grid.
    """
    def __init__(self, agent: 'Agent'):
        self.agent = agent
        
    def get_location(self) -> Tuple[int, int]:
        """Returns the current (x, y) location of the agent."""
        return self.agent.x, self.agent.y
        
    def sense_environment(self, grid: Grid) -> Dict[str, int]:
        """Returns the state of neighboring cells."""
        x, y = self.get_location()
        return {
            'forward': grid.get_cell(x, y - 1),  # Assuming forward is "up" (decreasing y)
            'down': grid.get_cell(x, y + 1),
            'left': grid.get_cell(x - 1, y),
            'right': grid.get_cell(x + 1, y)
        }


class Agent:
    """
    Agent class representing the entity moving in the Grid.
    """
    def __init__(self, start_x: int, start_y: int, grid: Grid):
        self.x = start_x
        self.y = start_y
        self.grid = grid
        self.sensor = Sensor(self)
        self.history = [(start_x, start_y)]  # Keeps track of where the robot has been
        
    def forward(self) -> bool:
        """Move forward/up (decrease y) if not a wall."""
        if self.grid.get_cell(self.x, self.y - 1) != 1:
            self.y -= 1
            self.history.append((self.x, self.y))
            return True
        return False

    def down(self) -> bool:
        """Move down (increase y) if not a wall."""
        if self.grid.get_cell(self.x, self.y + 1) != 1:
            self.y += 1
            self.history.append((self.x, self.y))
            return True
        return False
        
    def left(self) -> bool:
        """Move left (decrease x) if not a wall."""
        if self.grid.get_cell(self.x - 1, self.y) != 1:
            self.x -= 1
            self.history.append((self.x, self.y))
            return True
        return False
        
    def right(self) -> bool:
        """Move right (increase x) if not a wall."""
        if self.grid.get_cell(self.x + 1, self.y) != 1:
            self.x += 1
            self.history.append((self.x, self.y))
            return True
        return False


class RLAgent:
    """
    A class handling the reinforcement learning part (Q-Learning implementation).
    """
    def __init__(self, actions: List[str], learning_rate: float = 0.1, discount_factor: float = 0.9, exploration_rate: float = 1.0):
        self.q_table = {}
        self.actions = actions  # e.g., ['forward', 'down', 'left', 'right']
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
    def get_state_key(self, state_dict: Dict[str, int]) -> str:
        """Convert state dictionary from sensor to a string key for the Q-table."""
        # Sort items so identical states produce the same string key
        return str(sorted(state_dict.items()))
        
    def choose_action(self, state_dict: Dict[str, int]) -> str:
        """Choose an action using the epsilon-greedy policy."""
        state = self.get_state_key(state_dict)
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
            
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
            
    def learn(self, state_dict: Dict[str, int], action: str, reward: float, next_state_dict: Dict[str, int]):
        """Update the Q-value using the standard Q-learning algorithm."""
        state = self.get_state_key(state_dict)
        next_state = self.get_state_key(next_state_dict)
        
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
            
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning formula
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
