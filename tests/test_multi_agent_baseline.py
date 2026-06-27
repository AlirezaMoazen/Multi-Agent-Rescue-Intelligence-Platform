from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.learning.baseline import BaselineExplorer, DFSExplorer
from rescue_sim.learning.multi_agent_baseline import (
    compare_multi_agent_baselines,
    default_start_positions,
    run_multi_agent_baseline,
)
from rescue_sim.shared import Action, LearningState, Transition


class AlwaysRightStrategy:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def select_action(
        self,
        agent_id: str,
        state: LearningState,
        valid_actions: tuple[Action, ...],
    ) -> Action:
        del agent_id, state
        return Action.RIGHT if Action.RIGHT in valid_actions else Action.WAIT

    def update(self, transition: Transition) -> None:
        self.transitions.append(transition)


def make_grid() -> Grid:
    return Grid(
        width=3,
        height=1,
        obstacles=frozenset(),
        target_a_positions=frozenset({Position(2, 0)}),
        target_b_positions=frozenset(),
    )


def test_default_start_positions_returns_unique_valid_non_target_cells() -> None:
    grid = make_grid()

    starts = default_start_positions(grid, num_agents=2)

    assert len(starts) == 2
    assert len(set(starts.values())) == 2
    assert all(grid.is_valid_position(position) for position in starts.values())
    assert Position(2, 0) not in starts.values()


def test_multi_agent_runner_rescues_targets_and_prevents_overlap() -> None:
    strategy = AlwaysRightStrategy()
    starts = {
        "agent-0": Position(0, 0),
        "agent-1": Position(1, 0),
    }

    metrics = run_multi_agent_baseline(
        strategy=strategy,
        grid=make_grid(),
        start_positions=starts,
        max_steps=2,
        sensor_range=1,
        strategy_name="always_right",
    )

    assert metrics.num_agents == 2
    assert metrics.success is True
    assert metrics.rescued_targets == 1
    assert metrics.collisions == 1
    assert metrics.invalid_moves == 1
    assert metrics.final_positions["agent-0"] == Position(0, 0)
    assert metrics.final_positions["agent-1"] == Position(2, 0)
    assert len(strategy.transitions) == 2


def test_compare_multi_agent_baselines_runs_existing_non_ml_strategies() -> None:
    results = compare_multi_agent_baselines(
        grid_settings=GridSettings(
            width=4,
            height=4,
            obstacle_probability=0.0,
            target_a_count=1,
            target_b_count=1,
            random_seed=4,
        ),
        num_agents=2,
        max_steps=20,
        sensor_range=2,
        seed=3,
        baseline_factories={
            "frontier": BaselineExplorer,
            "dfs": DFSExplorer,
        },
    )

    assert set(results) == {"frontier", "dfs"}
    assert all(metrics.num_agents == 2 for metrics in results.values())
    assert all(metrics.total_targets == 2 for metrics in results.values())
    assert all(metrics.steps <= 20 for metrics in results.values())
