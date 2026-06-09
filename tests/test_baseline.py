"""Tests for both baseline exploration strategies.

Sections:
  1. Shared helpers (SimpleEnv stub, factory functions)
  2. BaselineExplorer tests   — valid-action contract, preferences, determinism,
                                per-agent independence, update no-op, run_episode
  3. DFSExplorer tests        — same contract tests + DFS-specific behaviour
  4. Shared run_episode() function tests
"""

from __future__ import annotations

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.environment.sensors import Observation as SensorObservation
from rescue_sim.learning.baseline import (
    BaselineExplorer,
    BaselineMetrics,
    DFSExplorer,
    run_episode,
)
from rescue_sim.shared import (
    Action,
    LearningState,
    MovementResult,
    Observation,
    Transition,
)


# ===========================================================================
# 1. Shared helpers
# ===========================================================================

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


class _SimpleEnv:
    """Minimal EnvironmentInterface stub backed by real grid/sensor/movement."""

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
            self._active_a.discard(self._pos)
            self._rescued_a.add(self._pos)
            reward = 10.0
        elif self._pos in self._active_b:
            self._active_b.discard(self._pos)
            self._rescued_b.add(self._pos)
            reward = 10.0

        self._steps += 1
        done = (not self._active_a and not self._active_b) or self._steps >= self.ep_max

        old_state = LearningState(agent_id="0", agent_position=old_pos,
                                  discovered_cells=self._sensor.discovered_cells)
        new_state = LearningState(
            agent_id="0", agent_position=self._pos,
            discovered_cells=self._sensor.discovered_cells,
            rescued_target_a_positions=frozenset(self._rescued_a),
            rescued_target_b_positions=frozenset(self._rescued_b),
            steps_taken=self._steps,
        )
        return Transition(state=old_state, action=action, next_state=new_state,
                          reward=reward, done=done, movement=result, observation=obs)  # type: ignore[arg-type]


# ===========================================================================
# 2. BaselineExplorer tests
# ===========================================================================

class TestBaselineExplorer:

    # --- valid-action contract ---

    def test_selected_action_is_always_in_valid_actions(self) -> None:
        exp = BaselineExplorer(seed=0)
        valid = (Action.UP, Action.RIGHT)
        assert exp.select_action("0", make_state(), valid) in valid

    def test_returns_wait_when_only_wait_is_available(self) -> None:
        exp = BaselineExplorer(seed=0)
        assert exp.select_action("0", make_state(), (Action.WAIT,)) == Action.WAIT

    def test_never_chooses_wait_when_moves_available(self) -> None:
        exp = BaselineExplorer(seed=0)
        for _ in range(30):
            a = exp.select_action("0", make_state(), (Action.WAIT, Action.UP, Action.RIGHT))
            assert a != Action.WAIT

    # --- exploration preference ---

    def test_prefers_unvisited_over_visited(self) -> None:
        exp = BaselineExplorer(seed=0)
        exp._visited_for("0").add(Position(2, 1))   # UP destination already visited
        a = exp.select_action("0", make_state(pos=Position(2, 2)), (Action.UP, Action.RIGHT))
        assert a == Action.RIGHT

    def test_prefers_frontier_cell_over_interior(self) -> None:
        exp = BaselineExplorer(seed=0)
        # RIGHT target (3,2): all neighbours discovered → no frontier bonus.
        # UP target (2,1): neighbour (2,0) undiscovered → frontier bonus.
        disc = frozenset([Position(2,2), Position(3,2), Position(4,2), Position(3,1), Position(3,3)])
        a = exp.select_action("0", make_state(pos=Position(2,2), discovered=disc),
                               (Action.UP, Action.RIGHT))
        assert a == Action.UP

    # --- determinism ---

    def test_same_seed_produces_same_action_sequence(self) -> None:
        state = make_state()
        ea, eb = BaselineExplorer(seed=99), BaselineExplorer(seed=99)
        ra = [ea.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(25)]
        rb = [eb.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(25)]
        assert ra == rb

    def test_none_seed_runs_without_error(self) -> None:
        exp = BaselineExplorer(seed=None)
        assert exp.select_action("0", make_state(), ALL_MOVE_ACTIONS) in ALL_MOVE_ACTIONS

    # --- per-agent independence ---

    def test_per_agent_visited_cells_are_independent(self) -> None:
        exp = BaselineExplorer(seed=0)
        exp._visited_for("0").add(Position(2, 1))
        a0 = exp.select_action("0", make_state(agent_id="0"), (Action.UP, Action.RIGHT))
        a1 = exp.select_action("1", make_state(agent_id="1"), (Action.UP, Action.RIGHT))
        assert a0 == Action.RIGHT   # UP visited for agent 0
        assert a1 in (Action.UP, Action.RIGHT)   # agent 1 unaffected

    def test_two_agents_maintain_separate_histories(self) -> None:
        exp = BaselineExplorer(seed=0)
        for pos in [Position(0,0), Position(0,1), Position(0,2)]:
            exp._visited_for("a").add(pos)
        exp._visited_for("b").add(Position(3, 3))
        assert Position(0, 1) in exp._visited["a"]
        assert Position(0, 1) not in exp._visited.get("b", set())

    # --- update is a no-op ---

    def test_update_does_not_raise(self) -> None:
        exp = BaselineExplorer(seed=0)
        exp.update(dummy_transition(make_state(), Action.RIGHT))

    def test_update_does_not_change_behaviour(self) -> None:
        ea, eb = BaselineExplorer(seed=1), BaselineExplorer(seed=1)
        state = make_state()
        for _ in range(5):
            eb.update(dummy_transition(state, Action.RIGHT))
        ra = [ea.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(10)]
        rb = [eb.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(10)]
        assert ra == rb

    # --- run_episode ---

    def test_run_episode_returns_baselinemetrics(self) -> None:
        m = BaselineExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=50), max_steps=50)
        assert isinstance(m, BaselineMetrics)

    def test_run_episode_records_positive_steps(self) -> None:
        m = BaselineExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=50), max_steps=50)
        assert m.steps >= 1

    def test_run_episode_records_nonzero_discovered_cells(self) -> None:
        m = BaselineExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=50), max_steps=50)
        assert m.discovered_cells > 0

    def test_run_episode_respects_max_steps(self) -> None:
        m = BaselineExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=9999), max_steps=10)
        assert m.steps <= 10

    def test_run_episode_percentage_with_total_cells(self) -> None:
        m = BaselineExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(5,5), Position(0,0), ep_max=300),
            max_steps=200, total_cells=25)
        assert m.percentage_discovered is not None
        assert 0.0 < m.percentage_discovered <= 100.0

    def test_run_episode_percentage_none_without_total_cells(self) -> None:
        m = BaselineExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0)), max_steps=20)
        assert m.percentage_discovered is None

    def test_run_episode_deterministic(self) -> None:
        grid = make_open_grid()
        m1 = BaselineExplorer(seed=7).run_episode(
            _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        m2 = BaselineExplorer(seed=7).run_episode(
            _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        assert m1 == m2

    def test_run_episode_stops_when_all_targets_rescued(self) -> None:
        grid = Grid(width=3, height=3, obstacles=frozenset(),
                    target_a_positions=frozenset({Position(1,0)}),
                    target_b_positions=frozenset())
        m = BaselineExplorer(seed=0).run_episode(
            _SimpleEnv(grid, Position(0,0), sensor_range=2, ep_max=200), max_steps=200)
        if m.rescued_targets == 1:
            assert m.steps < 200


# ===========================================================================
# 3. DFSExplorer tests
# ===========================================================================

class TestDFSExplorer:

    # --- valid-action contract ---

    def test_selected_action_is_always_in_valid_actions(self) -> None:
        exp = DFSExplorer(seed=0)
        valid = (Action.UP, Action.RIGHT)
        assert exp.select_action("0", make_state(), valid) in valid

    def test_returns_wait_when_only_wait_is_available(self) -> None:
        exp = DFSExplorer(seed=0)
        assert exp.select_action("0", make_state(), (Action.WAIT,)) == Action.WAIT

    def test_never_chooses_wait_when_moves_available(self) -> None:
        exp = DFSExplorer(seed=0)
        for _ in range(30):
            a = exp.select_action("0", make_state(), (Action.WAIT, Action.UP, Action.RIGHT))
            assert a != Action.WAIT

    # --- determinism ---

    def test_same_seed_produces_same_action_sequence(self) -> None:
        state = make_state()
        ea, eb = DFSExplorer(seed=42), DFSExplorer(seed=42)
        ra = [ea.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(25)]
        rb = [eb.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(25)]
        assert ra == rb

    def test_none_seed_runs_without_error(self) -> None:
        exp = DFSExplorer(seed=None)
        assert exp.select_action("0", make_state(), ALL_MOVE_ACTIONS) in ALL_MOVE_ACTIONS

    # --- per-agent independence ---

    def test_per_agent_stacks_are_independent(self) -> None:
        exp = DFSExplorer(seed=0)
        for y in range(3):
            exp._visited_for("0").add(Position(2, y))
        assert not exp._visited.get("1"), "agent 1 visited must be empty"
        assert not exp._stack.get("1"), "agent 1 stack must be empty"

    def test_per_agent_visited_are_independent(self) -> None:
        exp = DFSExplorer(seed=0)
        exp._visited_for("a").add(Position(0, 0))
        exp._visited_for("a").add(Position(1, 0))
        assert Position(0, 0) in exp._visited["a"]
        assert Position(0, 0) not in exp._visited.get("b", set())

    # --- update is a no-op ---

    def test_update_does_not_raise(self) -> None:
        exp = DFSExplorer(seed=0)
        exp.update(dummy_transition(make_state(), Action.RIGHT))

    def test_update_does_not_change_behaviour(self) -> None:
        state = make_state()
        ea, eb = DFSExplorer(seed=3), DFSExplorer(seed=3)
        for _ in range(5):
            eb.update(dummy_transition(state, Action.RIGHT))
        ra = [ea.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(10)]
        rb = [eb.select_action("0", state, ALL_MOVE_ACTIONS) for _ in range(10)]
        assert ra == rb

    # --- DFS-specific behaviour ---

    def test_dfs_explores_corridor_sequentially(self) -> None:
        """On a 1-row corridor DFS must advance cell by cell without skipping.

        Grid (5×2): row 0 open, row 1 fully blocked.
        The only valid move is always RIGHT, so DFS visits (0,0)→(1,0)→…→(4,0).
        """
        obstacles = frozenset(Position(x, 1) for x in range(5))
        grid = Grid(width=5, height=2, obstacles=obstacles,
                    target_a_positions=frozenset(), target_b_positions=frozenset())
        env = _SimpleEnv(grid, Position(0, 0), sensor_range=1, ep_max=20)

        exp = DFSExplorer(seed=0)
        state = env.reset()
        visited_order: list[Position] = [state.agent_position]

        for _ in range(10):
            valid = env.get_valid_actions(state)
            action = exp.select_action(state.agent_id, state, valid)
            transition = env.step(action)
            state = transition.next_state
            if state.agent_position != visited_order[-1]:
                visited_order.append(state.agent_position)
            if transition.done:
                break

        xs = [p.x for p in visited_order]
        assert all(p.y == 0 for p in visited_order), "should stay on row 0"
        assert xs == sorted(xs), f"should advance right monotonically, got {xs}"

    def test_dfs_goes_deep_first(self) -> None:
        """From the centre of a 3×3 grid DFS should go UP first (first in _DIRS)."""
        grid = make_open_grid(width=3, height=3)
        env = _SimpleEnv(grid, Position(1, 1), sensor_range=1, ep_max=50)

        exp = DFSExplorer(seed=0)
        state = env.reset()
        first = exp.select_action(state.agent_id, state, env.get_valid_actions(state))
        assert first == Action.UP, f"DFS should go UP first, got {first}"

    def test_dfs_stack_grows_on_arrival(self) -> None:
        """After the first select_action call the DFS stack must be non-empty."""
        env = _SimpleEnv(make_open_grid(), Position(2, 2), sensor_range=2, ep_max=100)
        exp = DFSExplorer(seed=0)
        state = env.reset()
        assert not exp._stack.get("0"), "stack should start empty"
        exp.select_action(state.agent_id, state, env.get_valid_actions(state))
        assert exp._stack.get("0"), "stack should be non-empty after first expansion"

    # --- run_episode ---

    def test_run_episode_returns_baselinemetrics(self) -> None:
        m = DFSExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=50), max_steps=50)
        assert isinstance(m, BaselineMetrics)

    def test_run_episode_records_positive_steps(self) -> None:
        m = DFSExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=50), max_steps=50)
        assert m.steps >= 1

    def test_run_episode_records_discovered_cells(self) -> None:
        m = DFSExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=50), max_steps=50)
        assert m.discovered_cells > 0

    def test_run_episode_respects_max_steps(self) -> None:
        m = DFSExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0), ep_max=9999), max_steps=10)
        assert m.steps <= 10

    def test_run_episode_percentage_with_total_cells(self) -> None:
        m = DFSExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(5,5), Position(0,0), ep_max=300),
            max_steps=200, total_cells=25)
        assert m.percentage_discovered is not None
        assert 0.0 < m.percentage_discovered <= 100.0

    def test_run_episode_percentage_none_without_total_cells(self) -> None:
        m = DFSExplorer(seed=0).run_episode(
            _SimpleEnv(make_open_grid(), Position(0,0)), max_steps=20)
        assert m.percentage_discovered is None

    def test_run_episode_deterministic(self) -> None:
        grid = make_open_grid()
        m1 = DFSExplorer(seed=11).run_episode(
            _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        m2 = DFSExplorer(seed=11).run_episode(
            _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        assert m1 == m2

    def test_run_episode_stops_when_all_targets_rescued(self) -> None:
        grid = Grid(width=3, height=3, obstacles=frozenset(),
                    target_a_positions=frozenset({Position(1,0)}),
                    target_b_positions=frozenset())
        m = DFSExplorer(seed=0).run_episode(
            _SimpleEnv(grid, Position(0,0), sensor_range=2, ep_max=200), max_steps=200)
        if m.rescued_targets == 1:
            assert m.steps < 200


# ===========================================================================
# 4. Module-level run_episode() function
# ===========================================================================

class TestRunEpisodeFunction:

    def test_works_with_baseline_explorer(self) -> None:
        grid = make_open_grid()
        m = run_episode(BaselineExplorer(seed=0),
                        _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        assert isinstance(m, BaselineMetrics)
        assert m.steps >= 1

    def test_works_with_dfs_explorer(self) -> None:
        grid = make_open_grid()
        m = run_episode(DFSExplorer(seed=0),
                        _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        assert isinstance(m, BaselineMetrics)
        assert m.steps >= 1

    def test_both_strategies_produce_same_metric_shape(self) -> None:
        grid = make_open_grid()
        m1 = run_episode(BaselineExplorer(seed=0),
                         _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        m2 = run_episode(DFSExplorer(seed=0),
                         _SimpleEnv(grid, Position(0,0), ep_max=50), max_steps=50)
        # Same fields, possibly different values — just check the shape.
        assert type(m1) is type(m2)
        assert hasattr(m1, "steps") and hasattr(m2, "steps")
