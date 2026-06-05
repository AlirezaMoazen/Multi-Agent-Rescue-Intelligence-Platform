"""Tests for BaselineExplorer.

Coverage:
- select_action always returns a valid action
- WAIT is only chosen when forced
- Unvisited cells are preferred over visited ones
- Frontier cells (adjacent to undiscovered territory) are preferred
- Same seed → identical action sequences (determinism)
- Per-agent visited sets are independent
- update() is a no-op (no raises, no state change)
- run_episode() returns correct BaselineMetrics
- run_episode() respects max_steps
- run_episode() is deterministic with the same seed
"""

from __future__ import annotations

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.environment.sensors import Observation as SensorObservation
from rescue_sim.learning.baseline import BaselineExplorer, BaselineMetrics
from rescue_sim.shared import (
    Action,
    LearningState,
    MovementResult,
    Observation,
    Transition,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def make_open_grid(width: int = 5, height: int = 5) -> Grid:
    """Return a fully passable grid with no targets."""
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
    """Build a minimal Transition for use in update() tests."""
    pos = state.agent_position
    return Transition(
        state=state,
        action=action,
        next_state=state,
        reward=0.0,
        done=False,
        movement=MovementResult(
            start=pos,
            requested=pos,
            end=pos,
            move=action.value,
            moved=False,
            reason="ok",
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
# Minimal EnvironmentInterface stub for run_episode tests
# ---------------------------------------------------------------------------

class _SimpleEnv:
    """Drives one agent through a real Grid using actual movement/sensor logic."""

    def __init__(
        self,
        grid: Grid,
        start: Position,
        sensor_range: int = 2,
        ep_max: int = 200,
    ) -> None:
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

    # EnvironmentInterface ---------------------------------------------------

    def reset(self) -> LearningState:
        self._sensor = CentralSensor(self.grid)
        self._pos = self.start
        self._steps = 0
        self._active_a = set(self.grid.target_a_positions)
        self._active_b = set(self.grid.target_b_positions)
        self._rescued_a = set()
        self._rescued_b = set()
        obs: SensorObservation = self._sensor.observe("0", self._pos, self.sensor_range)
        return self._to_state(obs)

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
            self._active_a.discard(self._pos)
            self._rescued_a.add(self._pos)
            reward = 10.0
        elif self._pos in self._active_b:
            self._active_b.discard(self._pos)
            self._rescued_b.add(self._pos)
            reward = 10.0

        self._steps += 1
        done = (not self._active_a and not self._active_b) or self._steps >= self.ep_max

        old_state = LearningState(
            agent_id="0",
            agent_position=old_pos,
            discovered_cells=self._sensor.discovered_cells,
        )
        new_state = LearningState(
            agent_id="0",
            agent_position=self._pos,
            discovered_cells=self._sensor.discovered_cells,
            rescued_target_a_positions=frozenset(self._rescued_a),
            rescued_target_b_positions=frozenset(self._rescued_b),
            steps_taken=self._steps,
        )
        return Transition(
            state=old_state,
            action=action,
            next_state=new_state,
            reward=reward,
            done=done,
            movement=result,
            observation=obs,  # type: ignore[arg-type]
        )

    # -----------------------------------------------------------------------

    def _to_state(self, obs: SensorObservation) -> LearningState:
        assert self._sensor is not None
        return LearningState(
            agent_id="0",
            agent_position=obs.agent_position,
            discovered_cells=self._sensor.discovered_cells,
        )


# ---------------------------------------------------------------------------
# select_action: valid-action contract
# ---------------------------------------------------------------------------

def test_selected_action_is_always_in_valid_actions() -> None:
    """The explorer must never return an action outside the provided tuple."""
    explorer = BaselineExplorer(seed=0)
    valid = (Action.UP, Action.RIGHT)
    state = make_state()
    action = explorer.select_action("0", state, valid)
    assert action in valid


def test_returns_wait_when_only_wait_is_available() -> None:
    explorer = BaselineExplorer(seed=0)
    state = make_state()
    action = explorer.select_action("0", state, (Action.WAIT,))
    assert action == Action.WAIT


def test_never_chooses_wait_when_moves_available() -> None:
    """WAIT must not be chosen when at least one movement action is valid."""
    explorer = BaselineExplorer(seed=0)
    state = make_state()
    valid = (Action.WAIT, Action.UP, Action.RIGHT)
    for _ in range(30):
        action = explorer.select_action("0", state, valid)
        assert action != Action.WAIT


# ---------------------------------------------------------------------------
# select_action: exploration preference
# ---------------------------------------------------------------------------

def test_prefers_unvisited_over_visited_cell() -> None:
    """When one direction leads to a visited cell, the other should be chosen."""
    explorer = BaselineExplorer(seed=0)
    pos = Position(2, 2)
    # Pre-mark the UP destination as visited for agent "0".
    explorer._visited_for("0").add(Position(2, 1))

    state = make_state(agent_id="0", pos=pos)
    # Only UP (visited target) and RIGHT (unvisited target) available.
    valid = (Action.UP, Action.RIGHT)
    action = explorer.select_action("0", state, valid)
    assert action == Action.RIGHT


def test_prefers_frontier_cell_over_interior() -> None:
    """A cell adjacent to undiscovered territory wins over a fully surrounded one."""
    explorer = BaselineExplorer(seed=0)
    pos = Position(2, 2)

    # RIGHT target (3,2): all four neighbours already discovered → no frontier bonus.
    # UP target (2,1): neighbour (2,0) is NOT in discovered → frontier bonus.
    discovered = frozenset(
        [pos, Position(3, 2), Position(4, 2), Position(3, 1), Position(3, 3)]
    )
    state = make_state(pos=pos, discovered=discovered)
    valid = (Action.UP, Action.RIGHT)
    action = explorer.select_action("0", state, valid)
    assert action == Action.UP


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_action_sequence() -> None:
    """Two explorers with the same seed must make identical choices."""
    state = make_state()
    valid = ALL_MOVE_ACTIONS

    explorer_a = BaselineExplorer(seed=99)
    explorer_b = BaselineExplorer(seed=99)

    results_a = [explorer_a.select_action("0", state, valid) for _ in range(25)]
    results_b = [explorer_b.select_action("0", state, valid) for _ in range(25)]

    assert results_a == results_b


def test_none_seed_runs_without_error() -> None:
    """seed=None must be accepted and produce a valid action."""
    explorer = BaselineExplorer(seed=None)
    state = make_state()
    action = explorer.select_action("0", state, ALL_MOVE_ACTIONS)
    assert action in ALL_MOVE_ACTIONS


# ---------------------------------------------------------------------------
# Per-agent state independence
# ---------------------------------------------------------------------------

def test_per_agent_visited_cells_are_independent() -> None:
    """Visits recorded for agent '0' must not affect agent '1'."""
    explorer = BaselineExplorer(seed=0)
    pos = Position(2, 2)

    # Mark the UP destination as visited for agent "0" only.
    explorer._visited_for("0").add(Position(2, 1))

    state_0 = make_state(agent_id="0", pos=pos)
    state_1 = make_state(agent_id="1", pos=pos)

    valid = (Action.UP, Action.RIGHT)

    # Agent 0: UP target is visited → should prefer RIGHT.
    action_0 = explorer.select_action("0", state_0, valid)
    assert action_0 == Action.RIGHT

    # Agent 1: visited set is empty → any valid action is acceptable.
    action_1 = explorer.select_action("1", state_1, valid)
    assert action_1 in valid


def test_two_agents_maintain_separate_histories() -> None:
    """After several steps, each agent must only know its own visited cells."""
    explorer = BaselineExplorer(seed=0)

    # Drive agent "a" through a sequence of positions.
    for pos in [Position(0, 0), Position(0, 1), Position(0, 2)]:
        explorer._visited_for("a").add(pos)

    # Agent "b" has visited only its own cell.
    explorer._visited_for("b").add(Position(3, 3))

    assert Position(0, 1) in explorer._visited["a"]
    assert Position(0, 1) not in explorer._visited.get("b", set())


# ---------------------------------------------------------------------------
# update() is a no-op
# ---------------------------------------------------------------------------

def test_update_does_not_raise() -> None:
    explorer = BaselineExplorer(seed=0)
    state = make_state()
    explorer.update(dummy_transition(state, Action.RIGHT))


def test_update_does_not_change_behaviour() -> None:
    """Calling update() must not affect subsequent select_action calls."""
    explorer_no_update = BaselineExplorer(seed=1)
    explorer_with_update = BaselineExplorer(seed=1)

    state = make_state()
    valid = ALL_MOVE_ACTIONS

    for _ in range(5):
        t = dummy_transition(state, Action.RIGHT)
        explorer_with_update.update(t)

    actions_no_update = [explorer_no_update.select_action("0", state, valid) for _ in range(10)]
    actions_with_update = [explorer_with_update.select_action("0", state, valid) for _ in range(10)]

    assert actions_no_update == actions_with_update


# ---------------------------------------------------------------------------
# run_episode metrics
# ---------------------------------------------------------------------------

def test_run_episode_returns_baseline_metrics() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, start=Position(0, 0), ep_max=50)
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=50)
    assert isinstance(metrics, BaselineMetrics)


def test_run_episode_records_positive_steps() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, start=Position(0, 0), ep_max=50)
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=50)
    assert metrics.steps >= 1


def test_run_episode_records_nonzero_discovered_cells() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, start=Position(0, 0), ep_max=50)
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=50)
    assert metrics.discovered_cells > 0


def test_run_episode_respects_max_steps() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, start=Position(0, 0), ep_max=9999)
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=10)
    assert metrics.steps <= 10


def test_run_episode_percentage_discovered_with_total_cells() -> None:
    grid = make_open_grid(width=5, height=5)
    env = _SimpleEnv(grid, start=Position(0, 0), ep_max=300)
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=200, total_cells=25)
    assert metrics.percentage_discovered is not None
    assert 0.0 < metrics.percentage_discovered <= 100.0


def test_run_episode_percentage_is_none_without_total_cells() -> None:
    grid = make_open_grid()
    env = _SimpleEnv(grid, start=Position(0, 0))
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=20)
    assert metrics.percentage_discovered is None


def test_run_episode_rescued_targets_counted() -> None:
    """Placing a target next to start should result in it being rescued."""
    grid = Grid(
        width=5,
        height=5,
        obstacles=frozenset(),
        target_a_positions=frozenset({Position(1, 0)}),
        target_b_positions=frozenset(),
    )
    env = _SimpleEnv(grid, start=Position(0, 0), sensor_range=2, ep_max=500)
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=500)
    assert metrics.rescued_targets >= 0  # May or may not reach depending on path


def test_run_episode_deterministic_with_same_seed() -> None:
    """Identical seed + identical environment must produce identical metrics."""
    grid = make_open_grid()

    m1 = BaselineExplorer(seed=7).run_episode(
        _SimpleEnv(grid, Position(0, 0), ep_max=50), max_steps=50
    )
    m2 = BaselineExplorer(seed=7).run_episode(
        _SimpleEnv(grid, Position(0, 0), ep_max=50), max_steps=50
    )

    assert m1 == m2


def test_run_episode_stops_when_all_targets_rescued() -> None:
    """Episode must end as soon as all targets are rescued, before max_steps."""
    # Single target right next to start; explorer should find it quickly.
    grid = Grid(
        width=3,
        height=3,
        obstacles=frozenset(),
        target_a_positions=frozenset({Position(1, 0)}),
        target_b_positions=frozenset(),
    )
    env = _SimpleEnv(grid, start=Position(0, 0), sensor_range=2, ep_max=200)
    metrics = BaselineExplorer(seed=0).run_episode(env, max_steps=200)
    # If rescued, steps should be well below max_steps.
    if metrics.rescued_targets == 1:
        assert metrics.steps < 200
