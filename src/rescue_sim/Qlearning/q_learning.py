"""Q-learning strategies for the rescue simulator.

Two learners live here:

* ``QLearningAgent`` -- **legacy single-agent** tabular Q-learner.  It is NOT
  part of the multi-agent model line-up or the Mixture-of-Experts; it is kept
  only because the visualization API's live single-agent demo and the evaluation
  panel still depend on it.  The multi-agent successor is the fleet below.
* ``EpidemicHystereticQLearning`` -- a vectorized NumPy learner for a
  *decentralized* multi-robot fleet.  It adds hysteretic updates and epidemic
  peer-to-peer max-sync on top of Q-learning.  See the section header lower in
  this file and ``rescue_sim.Qlearning.communications`` for the comms boundary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

import numpy as np

from rescue_sim.config.settings import AgentSettings, GridSettings, SimulationSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor, Observation
from rescue_sim.shared import (
    Action,
    CARDINAL_ACTIONS,
    GossipConfig,
    HystereticConfig,
    LearningState,
    RewardConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    TargetType,
    calculate_reward,
)


DEFAULT_ACTIONS: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.WAIT,
)
# Q-table format: one shared LearningState maps to one value per action.
QTable = dict[LearningState, dict[Action, float]]


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Training result for one episode."""

    steps: int
    total_reward: float
    targets_found: int
    success: bool


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Summary returned after several training episodes."""

    episodes: int
    successes: int
    average_reward: float
    average_steps: float
    episode_metrics: tuple[EpisodeMetrics, ...]


class QLearningAgent:
    """Q-table learner for one rescue agent."""

    def __init__(
        self,
        actions: Sequence[Action] = DEFAULT_ACTIONS,
        learning_rate: float = 0.2,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        reward_config: RewardConfig = SPRINT3_REWARD_CONFIG,
        rng: Random | None = None,
    ) -> None:
        if not actions:
            raise ValueError("at least one action is required")
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")

        self.actions = tuple(actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.reward_config = reward_config
        self.rng = rng or Random()
        self.q_table: QTable = defaultdict(self._new_action_values)

    def _new_action_values(self) -> dict[Action, float]:
        return {action: 0.0 for action in self.actions}

    def state_from_observation(
        self,
        observation: Observation,
        grid: Grid,
        found_targets: frozenset[Position],
        steps_taken: int,
    ) -> LearningState:
        """Build the shared Sprint 3 LearningState from current environment data."""
        # Separate visible targets by type so rewards/evaluation can distinguish A and B.
        visible_target_a = frozenset(
            position
            for position, target_type in observation.target_types.items()
            if target_type == TargetType.A
        )
        visible_target_b = frozenset(
            position
            for position, target_type in observation.target_types.items()
            if target_type == TargetType.B
        )
        found_target_a = frozenset(
            position for position in found_targets if position in grid.target_a_positions
        )
        found_target_b = frozenset(
            position for position in found_targets if position in grid.target_b_positions
        )

        return LearningState(
            agent_id=observation.agent_id,
            agent_position=observation.agent_position,
            visible_cells=observation.visible_cells,
            visible_obstacles=observation.obstacles,
            visible_target_a_positions=visible_target_a,
            visible_target_b_positions=visible_target_b,
            discovered_cells=observation.visible_cells,
            discovered_target_a_positions=visible_target_a,
            discovered_target_b_positions=visible_target_b,
            rescued_target_a_positions=found_target_a,
            rescued_target_b_positions=found_target_b,
            remaining_target_a_positions=grid.target_a_positions - found_target_a,
            remaining_target_b_positions=grid.target_b_positions - found_target_b,
            steps_taken=steps_taken,
        )

    def choose_action(
        self,
        state: LearningState,
        valid_actions: Sequence[Action],
    ) -> Action:
        """Choose an action with epsilon-greedy exploration."""
        if not valid_actions:
            raise ValueError("at least one valid action is required")

        actions = tuple(valid_actions)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        # Otherwise use the currently best known Q-value.
        action_values = self.q_table[state]
        return max(actions, key=lambda action: action_values[action])

    def update_q_value(
        self,
        state: LearningState,
        action: Action,
        reward: float,
        next_state: LearningState,
        next_valid_actions: Sequence[Action],
    ) -> None:
        """Apply the standard Q-learning update rule."""
        if action not in self.actions:
            raise ValueError(f"unknown action: {action}")
        if not next_valid_actions:
            raise ValueError("at least one next action is required")

        # Q-learning update: old value moves toward reward plus best future value.
        best_next_value = max(self.q_table[next_state][action] for action in next_valid_actions)
        old_value = self.q_table[state][action]
        target = reward + self.discount_factor * best_next_value
        self.q_table[state][action] = old_value + self.learning_rate * (target - old_value)

    def valid_actions(
        self,
        movement_model: MovementModel,
        grid: Grid,
        position: Position,
    ) -> tuple[Action, ...]:
        """Return legal actions that are part of this learner's action list."""
        allowed_moves = movement_model.allowed_moves(grid, position)
        # MovementModel returns strings; the learning contract uses Action values.
        return tuple(action for action in self.actions if action.value in allowed_moves)

    def train_episode(
        self,
        grid: Grid,
        start_position: Position,
        sensor_range: int,
        max_steps: int,
        movement_model: MovementModel | None = None,
    ) -> EpisodeMetrics:
        """Train the learner on one scenario."""
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        movement = movement_model or MovementModel()
        sensor = CentralSensor(grid)
        position = start_position
        found_targets: set[Position] = set()
        visited_positions: set[Position] = {start_position}
        all_targets = grid.target_a_positions | grid.target_b_positions
        total_reward = 0.0

        observation = sensor.observe("agent-1", position, sensor_range)

        for step_index in range(max_steps):
            state = self.state_from_observation(
                observation,
                grid,
                frozenset(found_targets),
                step_index,
            )

            # Stop when all targets are rescued or the step limit is reached.
            if state.is_terminal(max_steps):
                return EpisodeMetrics(
                    step_index,
                    total_reward,
                    len(found_targets),
                    found_targets == all_targets,
                )

            action = self.choose_action(state, self.valid_actions(movement, grid, position))
            movement_result = movement.apply(grid, position, action.value)
            next_position = movement_result.end
            next_observation = sensor.observe("agent-1", next_position, sensor_range)

            # Check whether the move rescued a new target.
            target_type = grid.target_type_at(next_position)
            new_target_type = None
            if target_type is not None and next_position not in found_targets:
                found_targets.add(next_position)
                new_target_type = TargetType(target_type)

            done = found_targets == all_targets
            reward = calculate_reward(
                RewardEvent(
                    moved=movement_result.moved,
                    move=action.value,
                    newly_discovered_cells=len(next_observation.newly_discovered_cells),
                    rescued_target_type=new_target_type,
                    completed_episode=done,
                    repeated_cell=next_position in visited_positions,
                ),
                self.reward_config,
            )
            total_reward += reward

            # Learn from this transition.
            next_state = self.state_from_observation(
                next_observation,
                grid,
                frozenset(found_targets),
                step_index + 1,
            )
            next_valid_actions = self.valid_actions(movement, grid, next_position)
            self.update_q_value(state, action, reward, next_state, next_valid_actions)

            position = next_position
            observation = next_observation
            visited_positions.add(position)

        return EpisodeMetrics(max_steps, total_reward, len(found_targets), found_targets == all_targets)

    def train(
        self,
        grid_settings: GridSettings,
        agent_settings: AgentSettings,
        simulation_settings: SimulationSettings,
        episodes: int,
        movement_model: MovementModel | None = None,
    ) -> TrainingMetrics:
        """Generate scenarios and train for several episodes."""
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        start_position = Position(agent_settings.start_x, agent_settings.start_y)
        episode_metrics: list[EpisodeMetrics] = []

        for _ in range(episodes):
            # Each episode starts with a fresh generated grid using the given settings.
            grid = generate_grid(grid_settings, start=start_position)
            episode_metrics.append(
                self.train_episode(
                    grid=grid,
                    start_position=start_position,
                    sensor_range=agent_settings.sensor_range,
                    max_steps=simulation_settings.max_steps,
                    movement_model=movement_model,
                )
            )

        successes = sum(metrics.success for metrics in episode_metrics)
        average_reward = sum(metrics.total_reward for metrics in episode_metrics) / episodes
        average_steps = sum(metrics.steps for metrics in episode_metrics) / episodes

        return TrainingMetrics(
            episodes=episodes,
            successes=successes,
            average_reward=average_reward,
            average_steps=average_steps,
            episode_metrics=tuple(episode_metrics),
        )

    def best_policy(self) -> dict[LearningState, Action]:
        """Return the currently best action for every known state."""
        return {
            state: max(action_values, key=action_values.get)
            for state, action_values in self.q_table.items()
        }

    def q_values(self) -> Mapping[LearningState, Mapping[Action, float]]:
        """Expose a copy of the learned Q-table."""
        return {state: dict(action_values) for state, action_values in self.q_table.items()}


# ===========================================================================
# Epidemic Hysteretic Q-Learning  (decentralized multi-robot fleet)
# ===========================================================================
#
# Why a separate, NumPy-vectorized design instead of extending QLearningAgent?
# A fleet of up to 20 robots, each learning every step, makes a Python
# per-agent loop the bottleneck.  Here the *entire fleet* is one contiguous
# array and every operation (action selection, the TD update, proximity
# detection, the gossip merge) is a single vectorized NumPy expression.
#
# Memory layout
# -------------
#   q[slot, y, x, action]      float32   one dense Q-table per agent slot
#   dirty[slot, y, x, action]  bool      "changed since last export" delta mask
#   active[slot]               bool      membership gate (dynamic 1..max_agents)
#   pos[slot]                  int16     (y, x) of each agent
#   last_sync[slot_a, slot_b]  int64     step of the last pairwise sync (cooldown)
#
# State = the agent's grid cell (y, x); actions = N, S, E, W (CARDINAL_ACTIONS).
# Indexing a (slot, y, x) triple is an O(1) view, so a step touches only the
# rows it needs.  Slots are pre-allocated to capacity and gated by ``active`` so
# add/remove is O(1) and never reallocates -- robots can fail or join mid-run.
#
# Algorithm
# ---------
# 1. Local hysteretic update:  delta = r + gamma * max_a' Q(s', a') - Q(s, a);
#    apply alpha if delta >= 0 else a muted beta (beta << alpha).  Optimism
#    keeps a good policy from being erased by a teammate's exploration.
# 2. Epidemic max-sync: when two robots are within ``comm_radius`` they merge
#    Q-tables with an element-wise max:  Q_local = max(Q_local, Q_peer).
# 3. Bandwidth minimization: only *dirty, high-utility* entries are serialized
#    into a GossipMessage (delta), not the whole table.
# 4. Congestion control: a per-pair cooldown plus a per-agent link budget cap
#    the chatter when robots cluster together.
#
# The physical transport of a GossipMessage is intentionally NOT decided here --
# that is the comms developer's job; see ``rescue_sim.communications``.

N_ACTIONS = 4
# (dy, dx) for each cardinal action in (row=y, col=x) order: N, S, E, W.
_ACTION_DELTAS: np.ndarray = np.array([(-1, 0), (1, 0), (0, 1), (0, -1)], dtype=np.int16)


@dataclass(frozen=True, slots=True)
class GossipMessage:
    """Bandwidth-minimized Q-table delta exchanged over one peer-to-peer link.

    Only *modified, high-utility* entries travel: ``indices`` are flat offsets
    into an agent's ``(H * W * A)`` Q-table and ``values`` are the matching
    Q-values.  This is the wire format the communications layer moves between
    robots; the receiver merges it with an element-wise max (``import_delta``).
    """

    sender: int
    indices: np.ndarray
    values: np.ndarray

    @property
    def size(self) -> int:
        """Number of Q entries carried -- the message's bandwidth cost."""
        return int(self.indices.size)


def _build_grid_maps(
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute passability and per-cell transitions for O(1) steps.

    Returns ``(passable, next_y, next_x, valid)``.  ``next_y``/``next_x`` give
    the cell reached by each action from every cell (an invalid move keeps the
    agent in place, matching ``MovementModel``); ``valid`` marks the actions
    that actually move the agent.
    """
    height, width = grid.height, grid.width
    passable = np.ones((height, width), dtype=bool)
    for obstacle in grid.obstacles:
        passable[obstacle.y, obstacle.x] = False

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    next_y = np.empty((height, width, N_ACTIONS), dtype=np.int16)
    next_x = np.empty((height, width, N_ACTIONS), dtype=np.int16)
    valid = np.zeros((height, width, N_ACTIONS), dtype=bool)

    for action in range(N_ACTIONS):
        dy = int(_ACTION_DELTAS[action, 0])
        dx = int(_ACTION_DELTAS[action, 1])
        cand_y, cand_x = rows + dy, cols + dx
        in_bounds = (cand_y >= 0) & (cand_y < height) & (cand_x >= 0) & (cand_x < width)
        ok = in_bounds.copy()
        ok[in_bounds] &= passable[cand_y[in_bounds], cand_x[in_bounds]]
        next_y[..., action] = np.where(ok, cand_y, rows)
        next_x[..., action] = np.where(ok, cand_x, cols)
        valid[..., action] = ok

    return passable, next_y, next_x, valid


class EpidemicHystereticQLearning:
    """Vectorized Epidemic Hysteretic Q-Learning for a decentralized fleet."""

    def __init__(
        self,
        grid: Grid,
        config: HystereticConfig = HystereticConfig(),
        gossip: GossipConfig = GossipConfig(),
        max_agents: int = 20,
        seed: int | None = None,
    ) -> None:
        if not 1 <= max_agents <= 64:
            raise ValueError("max_agents must be between 1 and 64")
        if not 0.0 <= config.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= config.beta <= config.alpha:
            raise ValueError("beta must satisfy 0 <= beta <= alpha (hysteretic)")
        if not 0.0 <= config.discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0.0 <= config.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        self.height = grid.height
        self.width = grid.width
        self.capacity = max_agents
        self.cfg = config
        self.gossip_cfg = gossip
        self.epsilon = config.epsilon
        self.step_count = 0
        self.rng = np.random.default_rng(seed)

        self._passable, self._next_y, self._next_x, self._valid = _build_grid_maps(grid)

        shape = (self.capacity, self.height, self.width, N_ACTIONS)
        self.q = np.zeros(shape, dtype=np.float32)
        self.dirty = np.zeros(shape, dtype=bool)
        self.active = np.zeros(self.capacity, dtype=bool)
        self.pos = np.zeros((self.capacity, 2), dtype=np.int16)
        self.last_sync = np.full((self.capacity, self.capacity), -(10**9), dtype=np.int64)

        self._id_to_slot: dict[str, int] = {}
        self._slot_to_id: dict[int, str] = {}

    # -- fleet membership ---------------------------------------------------

    def add_agent(self, agent_id: str, start: Position, *, fresh: bool | None = None) -> int:
        """Activate ``agent_id`` at ``start``; returns its slot index.

        A previously removed id is reactivated and *keeps* its learned Q-table
        (set ``fresh=True`` to wipe it).  A brand-new id claims a free slot
        and starts from zeros.  Raises if the fleet is at capacity.
        """
        if agent_id in self._id_to_slot:
            slot = self._id_to_slot[agent_id]
            if fresh:
                self._wipe_slot(slot)
        else:
            slot = self._free_slot()
            self._wipe_slot(slot)
            self._id_to_slot[agent_id] = slot
            self._slot_to_id[slot] = agent_id
        self.active[slot] = True
        self._set_pos(slot, start)
        return slot

    def remove_agent(self, agent_id: str) -> None:
        """Mark an agent inactive (a failure/dropout); its Q-table is retained."""
        slot = self._id_to_slot.get(agent_id)
        if slot is not None:
            self.active[slot] = False

    def forget_agent(self, agent_id: str) -> None:
        """Remove an agent *and* free its slot (its learned Q-table is lost)."""
        slot = self._id_to_slot.pop(agent_id, None)
        if slot is not None:
            self._slot_to_id.pop(slot, None)
            self.active[slot] = False
            self._wipe_slot(slot)

    def fleet_size(self) -> int:
        return int(self.active.sum())

    def positions(self) -> dict[str, Position]:
        return {
            self._slot_to_id[s]: Position(int(self.pos[s, 1]), int(self.pos[s, 0]))
            for s in np.flatnonzero(self.active)
        }

    def reset_positions(self, starts: Mapping[str, Position]) -> None:
        """Reset agent positions for a new episode (learned Q-tables persist)."""
        for agent_id, start in starts.items():
            if agent_id in self._id_to_slot:
                self._set_pos(self._id_to_slot[agent_id], start)

    # -- action selection ---------------------------------------------------

    def select_actions(self) -> dict[str, int]:
        """Epsilon-greedy action index per active agent, vectorized over the fleet."""
        slots = np.flatnonzero(self.active)
        if slots.size == 0:
            return {}
        ys, xs = self.pos[slots, 0], self.pos[slots, 1]
        q_here = self.q[slots, ys, xs, :]
        valid = self._valid[ys, xs, :]
        greedy = np.where(valid, q_here, -np.inf).argmax(axis=1)

        explore = self.rng.random(slots.size) < self.epsilon
        if explore.any():
            greedy[explore] = self._random_valid_actions(valid[explore])
        return {self._slot_to_id[int(s)]: int(a) for s, a in zip(slots, greedy)}

    def select_actions_enum(self) -> dict[str, Action]:
        """Same as ``select_actions`` but values are ``Action`` enums."""
        return {aid: CARDINAL_ACTIONS[idx] for aid, idx in self.select_actions().items()}

    def peek_next(self, agent_id: str, action: int) -> Position:
        """Resulting position of ``action`` from the agent's cell (matches MovementModel)."""
        slot = self._id_to_slot[agent_id]
        y, x = int(self.pos[slot, 0]), int(self.pos[slot, 1])
        return Position(int(self._next_x[y, x, action]), int(self._next_y[y, x, action]))

    # -- learning -----------------------------------------------------------

    def record_transitions(
        self,
        actions: Mapping[str, int],
        rewards: Mapping[str, float],
        next_positions: Mapping[str, Position],
        dones: Mapping[str, bool] | None = None,
    ) -> None:
        """Apply one hysteretic TD update for every acting agent, then advance.

        Call once per environment timestep with all agents that acted.  Updates
        are vectorized; ``dirty`` is marked so the changes can be gossiped.
        """
        if not actions:
            self.step_count += 1
            return

        ids = list(actions)
        slots = np.array([self._id_to_slot[i] for i in ids], dtype=np.intp)
        acts = np.array([actions[i] for i in ids], dtype=np.intp)
        rew = np.array([rewards[i] for i in ids], dtype=np.float32)
        next_y = np.array([next_positions[i].y for i in ids], dtype=np.intp)
        next_x = np.array([next_positions[i].x for i in ids], dtype=np.intp)
        cur_y = self.pos[slots, 0].astype(np.intp)
        cur_x = self.pos[slots, 1].astype(np.intp)

        next_q = self.q[slots, next_y, next_x, :]
        next_valid = self._valid[next_y, next_x, :]
        best_next = np.where(next_valid, next_q, -np.inf).max(axis=1)
        best_next = np.where(np.isfinite(best_next), best_next, 0.0).astype(np.float32)
        if dones is not None:
            terminal = np.array([bool(dones.get(i, False)) for i in ids])
            best_next = np.where(terminal, 0.0, best_next).astype(np.float32)

        current = self.q[slots, cur_y, cur_x, acts]
        delta = rew + self.cfg.discount_factor * best_next - current
        rate = np.where(delta >= 0.0, self.cfg.alpha, self.cfg.beta).astype(np.float32)
        self.q[slots, cur_y, cur_x, acts] = current + rate * delta
        self.dirty[slots, cur_y, cur_x, acts] = True

        self.pos[slots, 0] = next_y.astype(np.int16)
        self.pos[slots, 1] = next_x.astype(np.int16)
        self.step_count += 1

    def decay_epsilon(self, amount: float, floor: float = 0.0) -> None:
        """Reduce exploration after an episode."""
        self.epsilon = max(floor, self.epsilon - amount)

    # -- epidemic communication --------------------------------------------

    def neighbors(self, radius: float | None = None) -> list[tuple[str, str, float]]:
        """All active agent pairs within ``radius`` (Euclidean), with distances.

        This is the proximity trigger ("they can communicate when they meet").
        It does *not* perform any sync -- the comms layer decides what to do
        with the candidate links.
        """
        limit = self.gossip_cfg.comm_radius if radius is None else radius
        slots = np.flatnonzero(self.active)
        if slots.size < 2:
            return []
        coords = self.pos[slots].astype(np.float64)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        iu, ju = np.triu_indices(slots.size, k=1)
        pairs: list[tuple[str, str, float]] = []
        for a_local, b_local in zip(iu, ju):
            d = float(dist[a_local, b_local])
            if d < limit:
                pairs.append(
                    (self._slot_to_id[int(slots[a_local])],
                     self._slot_to_id[int(slots[b_local])], d)
                )
        return pairs

    def can_sync(self, id_a: str, id_b: str) -> bool:
        """True if the pair is off cooldown and may exchange this step."""
        slot_a, slot_b = self._id_to_slot[id_a], self._id_to_slot[id_b]
        return self.step_count - self.last_sync[slot_a, slot_b] >= self.gossip_cfg.cooldown

    def sync_pair(self, id_a: str, id_b: str) -> int:
        """Force a delta max-sync between two agents; returns entries improved.

        Records the sync time (for cooldown).  The comms layer can call this for
        any pair it decides should exchange -- after a line-of-sight or channel
        check of its own.
        """
        slot_a, slot_b = self._id_to_slot[id_a], self._id_to_slot[id_b]
        improved = self._sync_slots(slot_a, slot_b)
        self.last_sync[slot_a, slot_b] = self.step_count
        self.last_sync[slot_b, slot_a] = self.step_count
        return improved

    def gossip(self) -> int:
        """Built-in epidemic round: proximity + cooldown + link budget + max-sync.

        Returns the number of pairwise syncs performed.  Closest pairs get
        priority; each agent participates in at most ``max_links_per_step`` syncs
        so a cluster of robots cannot saturate the channel.
        """
        slots = np.flatnonzero(self.active)
        if slots.size < 2:
            return 0
        coords = self.pos[slots].astype(np.float64)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        iu, ju = np.triu_indices(slots.size, k=1)
        close = dist[iu, ju] < self.gossip_cfg.comm_radius
        cand_i, cand_j = iu[close], ju[close]
        if cand_i.size == 0:
            return 0

        order = np.argsort(dist[cand_i, cand_j])  # handshake priority: closest first
        budget = np.full(slots.size, self.gossip_cfg.max_links_per_step, dtype=np.int64)
        syncs = 0
        for k in order:
            a_local, b_local = int(cand_i[k]), int(cand_j[k])
            if budget[a_local] <= 0 or budget[b_local] <= 0:
                continue
            slot_a, slot_b = int(slots[a_local]), int(slots[b_local])
            if self.step_count - self.last_sync[slot_a, slot_b] < self.gossip_cfg.cooldown:
                continue
            self._sync_slots(slot_a, slot_b)
            self.last_sync[slot_a, slot_b] = self.step_count
            self.last_sync[slot_b, slot_a] = self.step_count
            budget[a_local] -= 1
            budget[b_local] -= 1
            syncs += 1
        return syncs

    def export_delta(self, agent_id: str) -> GossipMessage:
        """Serialize an agent's dirty, high-utility Q entries into a delta."""
        slot = self._id_to_slot[agent_id]
        return self._export_slot(slot)

    def import_delta(self, agent_id: str, message: GossipMessage) -> int:
        """Merge an incoming delta via element-wise max; returns entries improved."""
        return self._import_slot(self._id_to_slot[agent_id], message)

    # -- introspection / metrics -------------------------------------------

    def q_table(self, agent_id: str) -> np.ndarray:
        """Copy of one agent's ``(H, W, A)`` Q-table."""
        return self.q[self._id_to_slot[agent_id]].copy()

    def greedy_policy(self, agent_id: str) -> np.ndarray:
        """Best action index per cell for one agent (masked to valid moves)."""
        slot = self._id_to_slot[agent_id]
        return np.where(self._valid, self.q[slot], -np.inf).argmax(axis=-1)

    def mean_q(self) -> float:
        """Mean Q-value across active agents -- a coarse learning-progress signal."""
        slots = np.flatnonzero(self.active)
        return float(self.q[slots].mean()) if slots.size else 0.0

    # -- internals ----------------------------------------------------------

    def _sync_slots(self, slot_a: int, slot_b: int) -> int:
        delta_a = self._export_slot(slot_a)
        delta_b = self._export_slot(slot_b)
        improved = self._import_slot(slot_b, delta_a)
        improved += self._import_slot(slot_a, delta_b)
        return improved

    def _export_slot(self, slot: int) -> GossipMessage:
        mask = self.dirty[slot]
        if self.gossip_cfg.utility_threshold > 0.0:
            mask = mask & (np.abs(self.q[slot]) >= self.gossip_cfg.utility_threshold)
        idx = np.flatnonzero(mask.reshape(-1))
        values = self.q[slot].reshape(-1)[idx].copy()
        if self.gossip_cfg.clear_dirty_on_export and idx.size:
            self.dirty[slot].reshape(-1)[idx] = False
        return GossipMessage(sender=slot, indices=idx.astype(np.int64), values=values)

    def _import_slot(self, slot: int, message: GossipMessage) -> int:
        if message.indices.size == 0:
            return 0
        flat_q = self.q[slot].reshape(-1)
        current = flat_q[message.indices]
        improved = message.values > current
        if not improved.any():
            return 0
        winners = message.indices[improved]
        flat_q[winners] = message.values[improved]
        # Re-mark improved entries dirty so new knowledge keeps spreading (epidemic).
        self.dirty[slot].reshape(-1)[winners] = True
        return int(improved.sum())

    def _free_slot(self) -> int:
        used = set(self._slot_to_id)
        for slot in range(self.capacity):
            if slot not in used:
                return slot
        raise RuntimeError(f"fleet at capacity ({self.capacity} agents)")

    def _wipe_slot(self, slot: int) -> None:
        self.q[slot] = 0.0
        self.dirty[slot] = False
        self.last_sync[slot, :] = -(10**9)
        self.last_sync[:, slot] = -(10**9)

    def _set_pos(self, slot: int, position: Position) -> None:
        if not (0 <= position.x < self.width and 0 <= position.y < self.height):
            raise ValueError(f"start position {position} is outside the grid")
        self.pos[slot, 0] = position.y
        self.pos[slot, 1] = position.x

    def _random_valid_actions(self, valid_rows: np.ndarray) -> np.ndarray:
        # Pick a uniformly random valid action per row (invalid -> -1 so never chosen).
        noise = self.rng.random(valid_rows.shape)
        return np.where(valid_rows, noise, -1.0).argmax(axis=1)
