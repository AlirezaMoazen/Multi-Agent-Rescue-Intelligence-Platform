"""Cooperative multi-agent rescue environment for MAPPO (CTDE).

Pure NumPy -- no torch here, so it can be tested on its own. It reuses the
existing grid, movement, and reward contracts (`generate_grid`, `MovementModel`,
`calculate_reward`) so MAPPO results stay comparable to the Q-learning baselines.

Conventions
-----------
* Actions are indices 0..3 into ``CARDINAL_ACTIONS`` (N, S, E, W).
* Each agent sees a small egocentric window (partial observability -> the actor
  is decentralized).  The critic instead receives the global state (all agent
  observations concatenated) -- this is what makes it *centralized* training.
* The task is cooperative: every agent receives the same team reward each step.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.shared import (
    CARDINAL_ACTIONS,
    RewardConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    TargetType,
    calculate_reward,
)

N_ACTIONS = len(CARDINAL_ACTIONS)  # N, S, E, W


class RescueEnv:
    """A small cooperative grid-rescue environment for a fleet of agents."""

    def __init__(
        self,
        grid_settings: GridSettings,
        num_agents: int = 4,
        max_steps: int = 200,
        view_radius: int = 2,
        reward_config: RewardConfig = SPRINT3_REWARD_CONFIG,
        seed: int | None = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        self.grid_settings = grid_settings
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.view_radius = view_radius
        self.reward_config = reward_config
        self.rng = np.random.default_rng(seed)
        self.movement = MovementModel()

        win = 2 * view_radius + 1
        self._channels = 4  # [blocked, target-A, target-B, other-agent]
        self.obs_dim = win * win * self._channels + 4 + num_agents
        self.state_dim = self.obs_dim * num_agents
        self.n_actions = N_ACTIONS

        self.grid: Grid | None = None
        self.positions: list[Position] = []
        self._rescued: set[Position] = set()
        self._discovered: set[Position] = set()
        self._visited: set[Position] = set()
        self._steps = 0

    # -- gym-style API ------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Generate a fresh grid and place all agents at the start; returns obs."""
        seed = int(self.rng.integers(0, 2**31 - 1))
        start = Position(0, 0)
        self.grid = generate_grid(replace(self.grid_settings, random_seed=seed), start)
        self.positions = [start for _ in range(self.num_agents)]
        self._rescued = set()
        self._discovered = set()
        self._visited = {start}
        self._steps = 0
        return self._observations()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply one action per agent; returns (obs, team_reward, done, info)."""
        assert self.grid is not None, "call reset() before step()"
        all_targets = self.grid.target_a_positions | self.grid.target_b_positions
        team_reward = 0.0

        for i, action in enumerate(actions):
            move = CARDINAL_ACTIONS[int(action)].value
            result = self.movement.apply(self.grid, self.positions[i], move)
            self.positions[i] = result.end

            rescued_type = None
            target_type = self.grid.target_type_at(result.end)
            if target_type is not None and result.end not in self._rescued:
                self._rescued.add(result.end)
                rescued_type = TargetType(target_type)

            newly_discovered = self._discover(result.end)
            done_now = self._rescued == all_targets

            team_reward += calculate_reward(
                RewardEvent(
                    moved=result.moved,
                    move=move,
                    newly_discovered_cells=newly_discovered,
                    rescued_target_type=rescued_type,
                    completed_episode=done_now,
                    repeated_cell=result.end in self._visited,
                ),
                self.reward_config,
            )
            self._visited.add(result.end)

        self._steps += 1
        done = self._rescued == all_targets or self._steps >= self.max_steps
        info = {
            "rescued": len(self._rescued),
            "targets": len(all_targets),
            "success": self._rescued == all_targets,
            "steps": self._steps,
        }
        return self._observations(), team_reward, done, info

    # -- observations -------------------------------------------------------

    def global_state(self) -> np.ndarray:
        """Centralized critic input: all agent observations concatenated."""
        return self._observations().reshape(-1)

    def valid_action_mask(self) -> np.ndarray:
        """Per-agent boolean mask of moves that do not hit a wall/edge."""
        assert self.grid is not None
        mask = np.zeros((self.num_agents, self.n_actions), dtype=bool)
        for i, pos in enumerate(self.positions):
            for a, action in enumerate(CARDINAL_ACTIONS):
                mask[i, a] = self.movement.is_allowed(self.grid, pos, action.value)
        return mask

    def _observations(self) -> np.ndarray:
        return np.stack([self._agent_obs(i) for i in range(self.num_agents)])

    def _agent_obs(self, index: int) -> np.ndarray:
        assert self.grid is not None
        pos = self.positions[index]
        others = {p for j, p in enumerate(self.positions) if j != index}
        radius = self.view_radius
        window = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cell = Position(pos.x + dx, pos.y + dy)
                blocked = not self.grid.is_valid_position(cell)
                target_a = cell in self.grid.target_a_positions and cell not in self._rescued
                target_b = cell in self.grid.target_b_positions and cell not in self._rescued
                window.extend((float(blocked), float(target_a), float(target_b), float(cell in others)))

        total = len(self.grid.target_a_positions | self.grid.target_b_positions)
        remaining = (total - len(self._rescued)) / total if total else 0.0
        scalars = [
            pos.x / self.grid.width,
            pos.y / self.grid.height,
            self._steps / self.max_steps,
            remaining,
        ]
        agent_id = [1.0 if i == index else 0.0 for i in range(self.num_agents)]
        return np.array(window + scalars + agent_id, dtype=np.float32)

    def _discover(self, center: Position) -> int:
        assert self.grid is not None
        radius = self.view_radius
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cell = Position(center.x + dx, center.y + dy)
                if self.grid.contains(cell) and cell not in self._discovered:
                    self._discovered.add(cell)
                    count += 1
        return count
