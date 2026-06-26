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
    MOVE_DELTAS,
    RewardConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    TargetType,
    calculate_reward,
)

N_ACTIONS = len(CARDINAL_ACTIONS)  # N, S, E, W
# (dx, dy) per cardinal action, in CARDINAL_ACTIONS order.
_ACTION_DELTAS = [MOVE_DELTAS[a.value] for a in CARDINAL_ACTIONS]


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
        self._win = win
        self._pad = view_radius
        self._channels = 4  # [blocked, target-A, target-B, other-agent]
        self.obs_dim = win * win * self._channels + 4 + num_agents
        self.state_dim = self.obs_dim * num_agents
        self.n_actions = N_ACTIONS

        # Precompute the per-cell relative offsets in the view window (constant).
        rr = max(1, view_radius)
        offsets = np.arange(win) - view_radius                 # -r .. r
        self._rel_x = np.tile(offsets, (win, 1)).astype(np.float32) / rr           # varies by column
        self._rel_y = np.tile(offsets.reshape(-1, 1), (1, win)).astype(np.float32) / rr  # by row

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
        self._build_grid_arrays()
        return self._observations()

    def _build_grid_arrays(self) -> None:
        """Precompute padded NumPy maps so observations are slices, not loops.

        Each map is padded by `view_radius` so a view window is always a plain
        array slice; the border padding counts as walls (out of bounds).
        Actual cell (x, y) lives at padded index [y + pad, x + pad].
        """
        assert self.grid is not None
        r, h, w = self._pad, self.grid.height, self.grid.width
        self._blocked = np.ones((h + 2 * r, w + 2 * r), dtype=np.float32)  # padding = wall
        self._inbounds = np.zeros_like(self._blocked)
        self._ta = np.zeros_like(self._blocked)
        self._tb = np.zeros_like(self._blocked)
        self._blocked[r:r + h, r:r + w] = 0.0
        self._inbounds[r:r + h, r:r + w] = 1.0
        for p in self.grid.obstacles:
            self._blocked[p.y + r, p.x + r] = 1.0
        for p in self.grid.target_a_positions:
            self._ta[p.y + r, p.x + r] = 1.0
        for p in self.grid.target_b_positions:
            self._tb[p.y + r, p.x + r] = 1.0
        self._rescued_mask = np.zeros_like(self._blocked)
        self._discovered_mask = np.zeros_like(self._blocked)
        self._agent_count = np.zeros_like(self._blocked)

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
                self._rescued_mask[result.end.y + self._pad, result.end.x + self._pad] = 1.0
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
        r = self._pad
        mask = np.zeros((self.num_agents, self.n_actions), dtype=bool)
        for i, pos in enumerate(self.positions):
            for a, (dx, dy) in enumerate(_ACTION_DELTAS):
                mask[i, a] = self._blocked[pos.y + dy + r, pos.x + dx + r] == 0.0
        return mask

    def _refresh_agent_count(self) -> None:
        """Rebuild the padded map of how many agents occupy each cell."""
        self._agent_count[:] = 0.0
        for p in self.positions:
            self._agent_count[p.y + self._pad, p.x + self._pad] += 1.0

    def _observations(self) -> np.ndarray:
        self._refresh_agent_count()
        return np.stack([self._agent_obs(i) for i in range(self.num_agents)])

    def _window(self, x: int, y: int) -> tuple[slice, slice]:
        win = self._win
        return slice(y, y + win), slice(x, x + win)

    def _view_channels(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Shared view tensors for agent `index`: (blocked, target-A, target-B, other-agent)."""
        r = self._pad
        pos = self.positions[index]
        rows, cols = self._window(pos.x, pos.y)
        remaining = 1.0 - self._rescued_mask[rows, cols]
        blocked = self._blocked[rows, cols]
        target_a = self._ta[rows, cols] * remaining
        target_b = self._tb[rows, cols] * remaining
        other = (self._agent_count[rows, cols] > 0).astype(np.float32)
        # The agent itself sits at the window centre; only count it as "other"
        # if a teammate shares the same cell.
        other[r, r] = 1.0 if self._agent_count[pos.y + r, pos.x + r] - 1.0 > 0 else 0.0
        return blocked, target_a, target_b, other

    def _scalars(self, index: int) -> np.ndarray:
        assert self.grid is not None
        pos = self.positions[index]
        total = len(self.grid.target_a_positions | self.grid.target_b_positions)
        remaining = (total - len(self._rescued)) / total if total else 0.0
        return np.array(
            [pos.x / self.grid.width, pos.y / self.grid.height,
             self._steps / self.max_steps, remaining],
            dtype=np.float32,
        )

    def _agent_obs(self, index: int) -> np.ndarray:
        blocked, target_a, target_b, other = self._view_channels(index)
        window = np.stack([blocked, target_a, target_b, other], axis=-1).reshape(-1)
        agent_id = np.zeros(self.num_agents, dtype=np.float32)
        agent_id[index] = 1.0
        return np.concatenate([window, self._scalars(index), agent_id]).astype(np.float32)

    def _discover(self, center: Position) -> int:
        rows, cols = self._window(center.x, center.y)
        newly = (self._inbounds[rows, cols] > 0) & (self._discovered_mask[rows, cols] == 0)
        count = int(newly.sum())
        if count:
            self._discovered_mask[rows, cols] = np.where(newly, 1.0, self._discovered_mask[rows, cols])
        return count
