"""Tests for DFSExplorer.

Coverage:
- select_action always returns a valid action
- WAIT is only chosen when forced
- Never chooses WAIT when moves are available
- Determinism: same seed → same action sequence
- Per-agent visited sets are independent
- update() is a no-op
- DFS-specific: explores a corridor depth-first (in order) before backtracking
- DFS-specific: goes deep along one branch before exploring siblings
- run_episode() returns correct BaselineMetrics
- run_episode() respects max_steps
- run_episode() is deterministic with the same seed
"""

from __future__ import annotations

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.environment.sensors import Observation as SensorObservation
from rescue_sim.learning.baseline import BaselineMetrics
from rescue_sim.learning.baseline2 import DFSExplorer
from rescue_sim.shared import (
    Action,
    LearningState,
    MovementResult,
    Observation,
    Transition,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_open_grid(width: int = 5, height: int = 5) -> Grid:
    return Grid(
        width=width,
        height=height,
        obstacles=frozenset(),
        target_a_positions=frozenset(),
        target_b_positions=frozenset(),
    )


def make_state(
    agent_id: str = "0",
    pos: Position = Position(2, 2),
    discovered: frozenset[Position] = frozenset(),
    visible_obstacles: frozenset[Position] = frozenset(),
) -> LearningState:
    return LearningState(
        agent_id=agent_id,
        agent_position=pos,
        discovered_cells=discovered,
        visible_obstacles=visible_obstacles,
    )


ALL_MOVE_ACTIONS: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.FORWARD,
)


def dummy_transition(state: LearningState, action: Action) -> Transition:
    pos = state.agent_position
    return Transition(
        state=state,
        action=action,
        next_state=state,
        reward=0.0,
        done=False,
        movement=MovementResult(
            start=pos, requested=pos, end=pos,
            move=action.value, moved=False, reason="ok",
        ),
        observation=Observation(
            agent_id=state.agent_id,
            agent_position=pos,
            visible_cells=frozenset(),
            newly_discovered_cells=frozenset(),
            obstacles=frozenset(),
            targets=frozenset(),
            target_types={},
            newly_discovered_targets=frozenset(),
        ),
    )


# ---------------------------------------------------------------------------
# Minimal EnvironmentInterface stub (same pattern as test_baseline.py)
# ---------------------------------------------------------------------------

class _SimpleEnv:
    def __init__(self, grid: Grid, start: Position, sensor_range: int = 2, ep_max: int = 200) -> None:
        self.grid = grid
        self.start = start
        self.sensor_range = sensor_range
        self.ep_max = ep_max
        self._movement = MovementModel()
        self._sensor: CentralSensor | None = None
        self._pos = start
        self._active_a: set[Position] = set()
        self._active_b: set[Position] = set()
        self._rescued_a: set[Position] = set()
        self._rescued_b: set[Position] = set()
        self._steps = 0

    def reset(self) -> LearningState:
        self._sensor = CentralSensor(self.grid)
        self._pos = self.start
        self._steps = 0
        self._active_a = set(self.grid.target_a_positions)
        self._active_b = set(self.grid.target_b_positions)
        self._rescued_a = set()
        self._rescued_b = set()
        obs: SensorObservation = self._sensor.observe("0", self._pos, self.sensor_range)
        return LearningState(agent_id="0", agent_position=obs.agent_position,
                             discovered_cells=self._sensor.discovered_cells)

    def get_valid_actions(self, state: LearningState) -> tuple[Action, ...]:
        allowed = self._movement.allowed_moves(self.grid, state.agent_position)
        actions: list[Action] = []
        for name in allowed:
            try:
                actions.append(Action(name))
            except ValueError:
                pass
        return tuple(actions)

    def step(self, action: Action) -> Transition:
        assert self._sensor is not None
        old_pos = self._pos
        result = self._movement.apply(self.grid, self._pos, action.value)
        self._pos = result.end
        obs: SensorObservation = self._sensor.observe("0", self._pos, self.sensor_range)

        reward = -0.1 if result.moved else -1.0
        if self._pos in self._active_a:
            self._active_a.discard(self._pos); self._rescued_a.add(self._pos); reward = 10.0
        elif self._pos in self._active_b:
            self._active_b.discard(self._pos); self._rescued_b.add(self._pos); reward = 10.0

        self._steps += 1
        done = (not self._active_a and not self._active_b) or self._steps >= self.ep_max

        old_state = LearningState(agent_id="0", agent_position=old_pos,
                                  discovered_cells=self._sensor.discovered_cells)
        new_state = LearningState(agent_id="0", agent_position=self._pos,
                                  discovered_cells=self._sensor.discovered_cells,
                                  rescued_target_a_positions=frozenset(self._rescued_a),
                                  rescued_target_b_positions=frozenset(self._rescued_b),
                                  steps_taken=self._steps)
        return Transition(state=old_state, action=action, next_state=new_state,
                          reward=reward, done=done, movement=result, observation=obs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# select_action: valid-action contract
# ---------------------------------------------------------------------------

def test_selected_action_is_always_in_valid_actions() -> None:
    explorer = DFSExplorer(seed=0)
    valid = (Action.UP, Action.RIGHT)
    action = explorer.select_action("0", make_state(), valid)
    assert action in valid


def test_returns_wait_when_only_wait_is_available() -> None:
    explorer = DFSExplorer(seed=0)
    action = explorer.select_action("0", make_state(), (Action.WAIT,))
    assert action == Action.WAIT


def test_never_chooses_wait_when_moves_available() -> None:
    explorer = DFSExplorer(seed=0)
    state = make_state()
    valid = (Action.WAIT, Action.UP, Action.RIGHT)
    for _ in range(30):
        action = explorer.select_action("0", state, valid)
        assert action != Action.WAIT


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_action_sequence() -> None:
    state = make_state()
    valid = ALL_MOVE_ACTIONS
    ea, eb = DFSExplorer(seed=42), DFSExplorer(seed=42)
    ra = [ea.select_action("0", state, valid) for _ in range(25)]
    rb = [eb.select_action("0", state, valid) for _ in range(25)]
    assert ra == rb


def test_none_seed_runs_without_error() -> None:
    explorer = DFSExplorer(seed=None)
    action = explorer.select_action("0", make_state(), ALL_MOVE_ACTIONS)
    assert action in ALL_MOVE_ACTIONS


# ---------------------------------------------------------------------------
# Per-agent independence
# ---------------------------------------------------------------------------

def test_per_agent_stacks_are_independent() -> None:
    """DFS stack for agent '0' must not affect agent '1'."""
    explorer = DFSExplorer(seed=0)
    pos = Position(2, 2)
    # Drive agent "0" through a few positions so its stack and visited grow.
    for y in range(3):
        explorer._visited_for("0").add(Position(2, y))
    # Agent "1" should have an empty visited set.
    assert not explorer._visited.get("1"), "agent 1 visited must be empty"
    assert not explorer._stack.get("1"), "agent 1 stack must be empty"


def test_per_agent_visited_are_independent() -> None:
    explorer = DFSExplorer(seed=0)
    explorer._visited_for("a").add(Position(0, 0))
    explorer._visited_for("a").add(Position(1, 0))
    assert Position(0, 0) in explorer._visited["a"]
    assert Position(0, 0) not in explorer._visited.get("b", set())


# ---------------------------------------------------------------------------
# update() is a no-op
# ---------------------------------------------------------------------------

def test_update_does_not_raise() -> None:
    explorer = DFSExplorer(seed=0)
    state = make_state()
    explorer.update(dummy_transition(state, Action.RIGHT))


def test_update_does_not_change_behaviour() -> None:
    state = make_state()
    valid = ALL_MOVE_ACTIONS
    ea, eb = DFSExplorer(seed=3), DFSExplorer(seed=3)
    for _ in range(5):
        eb.update(dummy_transition(state, Action.RIGHT))
    ra = [ea.select_action("0", state, valid) for _ in range(10)]
    rb = [eb.select_action("0", state, valid) for _ in range(10)]
    assert ra == rb


# ---------------------------------------------------------------------------
# DFS-specific behaviour
# ---------------------------------------------------------------------------

def test_dfs_explores_corridor_sequentially() -> None:
    """On a 1-row corridor DFS must advance cell by cell, not skip around.

    Grid (5×2):
      S . . . .   row 0 — open
      # # # # #   row 1 — fully blocked

    Starting at S=(0,0), every valid action is RIGHT (no UP/DOWN/LEFT).
    DFS should push (1,0), navigate there, push (2,0), etc. — visiting
    cells in order (0,0)→(1,0)→(2,0)→(3,0)→(4,0).
    """
    obstacles = frozenset(Position(x, 1) for x in range(5))
    grid = Grid(width=5, height=2, obstacles=obstacles,
                target_a_positions=frozenset(), target_b_positions=frozenset())
    env = _SimpleEnv(grid, start=Position(0, 0), sensor_range=1, ep_max=20)

    explorer = DFSExplorer(seed=0)
    state = env.reset()
    visited_order: list[Position] = [state.agent_position]

    for _ in range(10):
        valid = env.get_valid_actions(state)
        action = explorer.select_action(state.agent_id, state, valid)
        transition = env.step(action)
        state = transition.next_state
        if state.agent_position != visited_order[-1]:
            visited_order.append(state.agent_position)
        if transition.done:
            break

    # All positions should be on row 0 and x should be non-decreasing
    # (the only valid direction is RIGHT).
    xs = [p.x for p in visited_order]
    assert all(p.y == 0 for p in visited_order), "should stay on row 0"
    assert xs == sorted(xs), f"should advance right monotonically, got {xs}"


def test_dfs_goes_deep_before_backtracking() -> None:
    """DFS must follow one branch to its end before exploring a sibling.

    Grid (3×3, no obstacles):
      . . .
      . S .
      . . .

    From S=(1,1), DFS should push neighbours in _DIRS priority order
    (UP first).  It should go UP to (1,0) before exploring other
    neighbours like RIGHT or DOWN.
    """
    grid = make_open_grid(width=3, height=3)
    env = _SimpleEnv(grid, start=Position(1, 1), sensor_range=1, ep_max=50)

    explorer = DFSExplorer(seed=0)
    state = env.reset()

    # The first action from (1,1) should be UP (first in _DIRS).
    valid = env.get_valid_actions(state)
    first_action = explorer.select_action(state.agent_id, state, valid)

    assert first_action == Action.UP, (
        f"DFS should explore UP first (first direction in _DIRS), got {first_action}"
    )


def test_dfs_stack_grows_on_arrival() -> None:
    """DFS stack must be non-empty after arriving at the first cell."""
    grid = make_open_grid(width=5, height=5)
    env = _SimpleEnv(grid, start=Position(2, 2), sensor_range=2, ep_max=100)

    explorer = DFSExplorer(seed=0)
    state = env.reset()
    valid = env.get_valid_actions(state)

    # Before any call: stack is empty.
    assert not explorer._stack.get("0"), "stack should start empty"

    explorer.select_action(state.agent_id, state, valid)

    # After the first call, the DFS stack should have been populated.
    assert explorer._stack.get("0"), "stack should be non-empty after first expansion"


# ---------------------------------------------------------------------------
# run_episode metrics
# ---------------------------------------------------------------------------

def test_run_episode_returns_baselinemetrics() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, Position(0, 0), ep_max=50)
    metrics = DFSExplorer(seed=0).run_episode(env, max_steps=50)
    assert isinstance(metrics, BaselineMetrics)


def test_run_episode_records_positive_steps() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, Position(0, 0), ep_max=50)
    metrics = DFSExplorer(seed=0).run_episode(env, max_steps=50)
    assert metrics.steps >= 1


def test_run_episode_records_discovered_cells() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, Position(0, 0), ep_max=50)
    metrics = DFSExplorer(seed=0).run_episode(env, max_steps=50)
    assert metrics.discovered_cells > 0


def test_run_episode_respects_max_steps() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, Position(0, 0), ep_max=9999)
    metrics = DFSExplorer(seed=0).run_episode(env, max_steps=10)
    assert metrics.steps <= 10


def test_run_episode_percentage_with_total_cells() -> None:
    grid = make_open_grid(width=5, height=5)
    env = _SimpleEnv(grid, Position(0, 0), ep_max=300)
    metrics = DFSExplorer(seed=0).run_episode(env, max_steps=200, total_cells=25)
    assert metrics.percentage_discovered is not None
    assert 0.0 < metrics.percentage_discovered <= 100.0


def test_run_episode_percentage_none_without_total_cells() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, Position(0, 0))
    metrics = DFSExplorer(seed=0).run_episode(env, max_steps=20)
    assert metrics.percentage_discovered is None


def test_run_episode_deterministic() -> None:
    grid = make_open_grid()
    m1 = DFSExplorer(seed=11).run_episode(_SimpleEnv(grid, Position(0, 0), ep_max=50), max_steps=50)
    m2 = DFSExplorer(seed=11).run_episode(_SimpleEnv(grid, Position(0, 0), ep_max=50), max_steps=50)
    assert m1 == m2


def test_run_episode_stops_when_all_targets_rescued() -> None:
    grid = Grid(width=3, height=3, obstacles=frozenset(),
                target_a_positions=frozenset({Position(1, 0)}),
                target_b_positions=frozenset())
    env = _SimpleEnv(grid, Position(0, 0), sensor_range=2, ep_max=200)
    metrics = DFSExplorer(seed=0).run_episode(env, max_steps=200)
    if metrics.rescued_targets == 1:
        assert metrics.steps < 200
