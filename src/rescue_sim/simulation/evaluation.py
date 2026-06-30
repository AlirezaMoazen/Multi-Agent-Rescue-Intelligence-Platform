"""Evaluation tools for rescue strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
from random import Random
from typing import Iterable

from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.Qlearning.multi_agent_baseline import (
    DEFAULT_MULTI_AGENT_BASELINES,
    default_start_positions,
    run_multi_agent_baseline,
)
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning, QLearningAgent
from rescue_sim.shared import (
    Action,
    CARDINAL_ACTIONS,
    GossipConfig,
    GridSettings,
    HystereticConfig,
    LearningState,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    StrategyInterface,
    TargetType,
    calculate_reward,
)

DEFAULT_NUM_AGENTS = 4
DEFAULT_TRAINING_EPISODES = 25
EVALUATION_SENSOR_RANGE = 3
ACTIONS: tuple[Action, ...] = (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP, Action.WAIT)

BASELINE_STRATEGIES = tuple(DEFAULT_MULTI_AGENT_BASELINES.items())

DEEP_BENCHMARK_ALGORITHMS = ("MAPPO", "QMIX", "TransfQMix", "Ensemble", "Distilled")
DEEP_BENCHMARK_NOTE = (
    "Deep RL methods use their own RescueEnv benchmark with fresh generated grids. "
    "These results are useful as an additional benchmark, but they are not directly "
    "comparable to the same-grid simulation runs."
)


@dataclass(frozen=True)
class EvaluationScenario:
    """Configuration for one reproducible evaluation scenario."""

    name: str
    grid_settings: GridSettings
    max_steps: int
    start: Position = Position(0, 0)
    num_agents: int = DEFAULT_NUM_AGENTS
    communication_range: float = 3.0


@dataclass(frozen=True)
class LearningFeedback:
    """Small training summary for the Q-learning comparator."""

    scenario_name: str
    training_episodes: int
    first_episode_steps: int
    last_episode_steps: int
    first_episode_reward: float
    last_episode_reward: float
    learned_state_actions: int


@dataclass(frozen=True)
class RunTrace:
    """Raw result for one algorithm on one scenario."""

    scenario_name: str
    agent_name: str
    seed: int | None
    num_agents: int
    steps_taken: int
    max_steps: int
    total_reward: float
    rescued_targets: int
    total_targets: int
    explored_cells: int
    explorable_cells: int
    algorithm_group: str = "baseline"
    status: str = "ok"
    communication_events: int = 0
    error: str | None = None


@dataclass(frozen=True)
class RunMetrics:
    """Report-ready metrics for one scenario and one algorithm."""

    scenario_name: str
    agent_name: str
    seed: int | None
    num_agents: int
    success: bool
    success_rate: float
    steps_taken: int
    average_steps: float
    accumulated_reward: float
    average_accumulated_reward: float
    rescued_targets: int
    total_targets: int
    explored_area_percentage: float
    algorithm_group: str = "baseline"
    status: str = "ok"
    communication_events: int = 0
    error: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Full comparison report for JSON, CSV, or visualization."""

    training_scenarios: list[dict]
    scenarios: list[dict]
    runs: list[dict]
    aggregates: list[dict]
    learning_feedback: list[dict]
    sprint_demo_summary: str
    deep_benchmark: list[dict]
    deep_benchmark_note: str


def default_evaluation_scenarios() -> list[EvaluationScenario]:
    specs = [
        ("small_open_seed_7", 5, 5, 0.05, 1, 1, 7, 35),
        ("medium_mixed_seed_13", 8, 8, 0.18, 2, 2, 13, 80),
        ("large_dense_seed_23", 10, 10, 0.25, 3, 2, 23, 130),
        ("wide_sparse_seed_31", 12, 6, 0.12, 2, 3, 31, 95),
    ]
    return [_scenario(*spec) for spec in specs]


def default_training_scenarios() -> list[EvaluationScenario]:
    specs = [
        ("train_small_seed_101", 5, 5, 0.05, 1, 1, 101, 35, Position(1, 1)),
        ("train_medium_seed_113", 8, 8, 0.18, 2, 2, 113, 80, Position(2, 2)),
        ("train_large_seed_123", 10, 10, 0.25, 3, 2, 123, 130, Position(3, 3)),
        ("train_wide_seed_131", 12, 6, 0.12, 2, 3, 131, 95, Position(4, 2)),
    ]
    return [_scenario(*spec) for spec in specs]


def evaluate_agents(
    scenarios: Iterable[EvaluationScenario] | None = None,
    training_scenarios: Iterable[EvaluationScenario] | None = None,
    training_episodes: int = DEFAULT_TRAINING_EPISODES,
) -> EvaluationReport:
    """Compatibility wrapper that builds demo grids from scenario settings.

    The real simulation/visualization path should call evaluate_simulation_grid()
    with the grid that the main simulation already generated.
    """

    selected = list(scenarios or default_evaluation_scenarios())
    train_selected = list(
        training_scenarios
        if training_scenarios is not None
        else (selected if scenarios is not None else default_training_scenarios())
    )

    learner = QLearningAgent(
        actions=ACTIONS,
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=0.35,
        reward_config=SPRINT3_REWARD_CONFIG,
        rng=Random(10_000),
    )
    feedback = [
        train_q_learning_agent(
            learner,
            training_scenario,
            generate_grid(training_scenario.grid_settings, start=training_scenario.start),
            training_episodes,
        )
        for training_scenario in train_selected
    ]

    all_runs: list[dict] = []
    scenario_dicts: list[dict] = []
    for scenario in selected:
        grid = generate_grid(scenario.grid_settings, start=scenario.start)
        starts = _agent_starts(grid, scenario)
        report = evaluate_simulation_grid(
            scenario=scenario,
            grid=grid,
            learner=learner,
            start_positions=starts,
        )
        all_runs.extend(report.runs)
        scenario_dicts.extend(report.scenarios)

    runs = [RunMetrics(**run) for run in all_runs]
    aggregates = _aggregate_by_agent(runs)
    return EvaluationReport(
        training_scenarios=[_scenario_to_dict(s, "train") for s in train_selected],
        scenarios=scenario_dicts,
        runs=all_runs,
        aggregates=aggregates,
        learning_feedback=[asdict(item) for item in feedback],
        sprint_demo_summary=build_sprint_demo_summary(aggregates),
        deep_benchmark=_deep_benchmark_rows(),
        deep_benchmark_note=DEEP_BENCHMARK_NOTE,
    )


def evaluate_simulation_grid(
    scenario: EvaluationScenario,
    grid: Grid,
    start_positions: dict[str, Position],
    learner: QLearningAgent | None = None,
) -> EvaluationReport:
    """Evaluate algorithms on the exact grid used by the main simulation."""

    q_learner = learner or QLearningAgent(
        actions=ACTIONS,
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=0.0,
        reward_config=SPRINT3_REWARD_CONFIG,
        rng=Random(_policy_seed(scenario.grid_settings.random_seed, "trained")),
    )

    runs = [calculate_run_metrics(run_main_algorithm(scenario, grid, start_positions))]
    for name, strategy_type in BASELINE_STRATEGIES:
        if hasattr(strategy_type, "clear_reservations"):
            strategy_type.clear_reservations()
        strategy = strategy_type(seed=_policy_seed(scenario.grid_settings.random_seed, name))
        runs.append(
            calculate_run_metrics(
                run_strategy_on_grid(scenario, grid, name, strategy, start_positions)
            )
        )
    runs.append(
        calculate_run_metrics(
            run_q_learning_agent_on_grid(scenario, grid, q_learner, start_positions)
        )
    )

    aggregates = _aggregate_by_agent(runs)
    return EvaluationReport(
        training_scenarios=[],
        scenarios=[_scenario_to_dict(scenario, start_positions=start_positions)],
        runs=[asdict(run) for run in runs],
        aggregates=aggregates,
        learning_feedback=[],
        sprint_demo_summary=build_sprint_demo_summary(aggregates),
        deep_benchmark=_deep_benchmark_rows(),
        deep_benchmark_note=DEEP_BENCHMARK_NOTE,
    )


def run_main_algorithm(
    scenario: EvaluationScenario,
    grid: Grid,
    starts: dict[str, Position],
) -> RunTrace:
    """Run the main cooperative Q-learning algorithm."""

    fleet = EpidemicHystereticQLearning(
        grid=grid,
        config=HystereticConfig(),
        gossip=GossipConfig(comm_radius=scenario.communication_range),
        max_agents=max(20, scenario.num_agents),
        seed=scenario.grid_settings.random_seed,
    )
    for agent_id, start in starts.items():
        fleet.add_agent(agent_id, start)

    return _run_fleet_loop(
        scenario=scenario,
        grid=grid,
        starts=starts,
        group="main",
        name="epidemic_hysteretic_q",
        action_picker=lambda: {
            agent_id: CARDINAL_ACTIONS[index] for agent_id, index in fleet.select_actions().items()
        },
        after_step=lambda actions, rewards, positions, dones: (
            fleet.record_transitions(
                {agent_id: CARDINAL_ACTIONS.index(action) for agent_id, action in actions.items()},
                rewards,
                positions,
                dones,
            ),
            fleet.gossip(),
        )[1],
    )


def run_strategy_on_grid(
    scenario: EvaluationScenario,
    grid: Grid,
    name: str,
    strategy: StrategyInterface,
    starts: dict[str, Position],
) -> RunTrace:
    """Run one baseline-style strategy as a small cooperative fleet."""

    metrics = run_multi_agent_baseline(
        strategy=strategy,
        grid=grid,
        start_positions=starts,
        max_steps=scenario.max_steps,
        sensor_range=EVALUATION_SENSOR_RANGE,
        reward_config=SPRINT3_REWARD_CONFIG,
        strategy_name=name,
    )
    return RunTrace(
        scenario_name=scenario.name,
        agent_name=name,
        seed=scenario.grid_settings.random_seed,
        num_agents=scenario.num_agents,
        steps_taken=metrics.steps,
        max_steps=scenario.max_steps,
        total_reward=metrics.total_reward,
        rescued_targets=metrics.rescued_targets,
        total_targets=metrics.total_targets,
        explored_cells=metrics.explored_cells,
        explorable_cells=_explorable_cell_count(grid),
        algorithm_group="baseline",
    )


def train_q_learning_agent(
    learner: QLearningAgent,
    scenario: EvaluationScenario,
    grid: Grid,
    training_episodes: int,
) -> LearningFeedback:
    # Legacy single-agent Q-learning path: retained for the visualization API and
    # the evaluation panel. The multi-agent line-up (Epidemic fleet, QMIX,
    # TransfQMix, MAPPO, MoE) does not go through here.
    starts = _agent_starts(grid, scenario)
    traces = [
        run_q_learning_agent_on_grid(scenario, grid, learner, starts, training=True)
        for _ in range(training_episodes)
    ]
    first, last = traces[0], traces[-1]
    return LearningFeedback(
        scenario.name,
        training_episodes,
        first.steps_taken,
        last.steps_taken,
        first.total_reward,
        last.total_reward,
        len(learner.q_table),
    )


def run_q_learning_agent_on_grid(
    scenario: EvaluationScenario,
    grid: Grid,
    learner: QLearningAgent,
    starts: dict[str, Position],
    training: bool = False,
) -> RunTrace:
    """Run the regular Q-learning policy as another comparator."""

    previous_epsilon = learner.epsilon
    if not training:
        learner.epsilon = 0.0

    sensor = CentralSensor(grid)

    def pick_actions() -> dict[str, Action]:
        actions: dict[str, Action] = {}
        for agent_id, position in positions.items():
            observation = sensor.observe(agent_id, position, EVALUATION_SENSOR_RANGE)
            state = learner.state_from_observation(observation, grid, frozenset(), 0)
            actions[agent_id] = learner.choose_action(state, _learner_actions(learner, grid, position))
        return actions

    def learn(
        actions: dict[str, Action],
        rewards: dict[str, float],
        next_positions: dict[str, Position],
        dones: dict[str, bool],
    ) -> int:
        if not training:
            return 0
        for agent_id, action in actions.items():
            before = positions[agent_id]
            after = next_positions[agent_id]
            state_obs = sensor.observe(agent_id, before, EVALUATION_SENSOR_RANGE)
            next_obs = sensor.observe(agent_id, after, EVALUATION_SENSOR_RANGE)
            state = learner.state_from_observation(state_obs, grid, frozenset(), 0)
            next_state = learner.state_from_observation(next_obs, grid, frozenset(), 0)
            learner.update_q_value(
                state,
                action,
                rewards[agent_id],
                next_state,
                _learner_actions(learner, grid, after),
            )
        return 0

    positions = starts
    trace = _run_fleet_loop(
        scenario,
        grid,
        positions,
        group="q_learning",
        name="trained",
        action_picker=pick_actions,
        after_step=learn,
    )
    learner.epsilon = previous_epsilon
    return trace


def _run_fleet_loop(
    scenario: EvaluationScenario,
    grid: Grid,
    starts: dict[str, Position],
    group: str,
    name: str,
    action_picker,
    after_step,
) -> RunTrace:
    movement = MovementModel()
    sensor = CentralSensor(grid)
    positions = dict(starts)
    visited = set(starts.values())
    visited_by_agent = {agent_id: {start} for agent_id, start in starts.items()}
    rescued: set[Position] = set()
    targets = grid.target_a_positions | grid.target_b_positions
    steps = 0
    reward_total = 0.0
    communication_events = 0

    for step in range(1, scenario.max_steps + 1):
        if rescued == targets:
            break

        actions = action_picker()
        rewards: dict[str, float] = {}
        next_positions: dict[str, Position] = {}
        dones: dict[str, bool] = {}

        for agent_id, action in actions.items():
            before = positions[agent_id]
            result = movement.apply(grid, before, action.value)
            after = result.end
            positions[agent_id] = after
            next_positions[agent_id] = after

            observation = sensor.observe(agent_id, after, EVALUATION_SENSOR_RANGE)
            target_type = _target_type_enum(grid, after) if after not in rescued else None
            if target_type is not None:
                rescued.add(after)

            new_cell = after not in visited
            visited.add(after)
            repeated = after in visited_by_agent[agent_id]
            visited_by_agent[agent_id].add(after)
            done = rescued == targets
            reward = calculate_reward(
                RewardEvent(
                    moved=result.moved,
                    move=action.value,
                    newly_discovered_cells=len(observation.newly_discovered_cells) or int(new_cell),
                    rescued_target_type=target_type,
                    completed_episode=done,
                    repeated_cell=repeated,
                ),
                SPRINT3_REWARD_CONFIG,
            )
            rewards[agent_id] = reward
            dones[agent_id] = done
            reward_total += reward

        communication_events += int(after_step(actions, rewards, next_positions, dones))
        steps = step

    return RunTrace(
        scenario_name=scenario.name,
        agent_name=name,
        seed=scenario.grid_settings.random_seed,
        num_agents=scenario.num_agents,
        steps_taken=steps,
        max_steps=scenario.max_steps,
        total_reward=round(reward_total, 4),
        rescued_targets=len(rescued),
        total_targets=len(targets),
        explored_cells=len(visited),
        explorable_cells=_explorable_cell_count(grid),
        algorithm_group=group,
        communication_events=communication_events,
    )


def calculate_run_metrics(trace: RunTrace) -> RunMetrics:
    success = trace.status == "ok" and trace.rescued_targets == trace.total_targets
    explored = trace.explored_cells / trace.explorable_cells * 100 if trace.explorable_cells else 0.0
    return RunMetrics(
        scenario_name=trace.scenario_name,
        agent_name=trace.agent_name,
        seed=trace.seed,
        num_agents=trace.num_agents,
        success=success,
        success_rate=1.0 if success else 0.0,
        steps_taken=trace.steps_taken,
        average_steps=float(trace.steps_taken),
        accumulated_reward=trace.total_reward,
        average_accumulated_reward=trace.total_reward,
        rescued_targets=trace.rescued_targets,
        total_targets=trace.total_targets,
        explored_area_percentage=round(explored, 2),
        algorithm_group=trace.algorithm_group,
        status=trace.status,
        communication_events=trace.communication_events,
        error=trace.error,
    )


def report_to_json(report: EvaluationReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def report_to_csv(report: EvaluationReport) -> str:
    if not report.runs:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(report.runs[0]))
    writer.writeheader()
    writer.writerows(report.runs)
    return output.getvalue()


def build_sprint_demo_summary(aggregates: list[dict]) -> str:
    lines = ["Evaluation compares real strategy runs on the same generated grids."]
    for row in aggregates:
        error = f", error={row['error']}" if row.get("error") else ""
        lines.append(
            f"{row['agent_name']} ({row['algorithm_group']}): "
            f"success_rate={row['success_rate']:.2f}, "
            f"average_steps={row['average_steps']:.1f}, "
            f"average_reward={row['average_accumulated_reward']:.1f}, "
            f"average_rescued_targets={row['average_rescued_targets']:.1f}, "
            f"average_explored_area={row['average_explored_area_percentage']:.1f}%, "
            f"num_agents={row['num_agents']}, "
            f"communication_events={row['communication_events']:.1f}, "
            f"status={row['status']}{error}"
        )
    return "\n".join(lines)


def _learning_state(
    observation: object,
    grid: Grid,
    discovered_cells: frozenset[Position],
    rescued: frozenset[Position],
    steps: int,
) -> LearningState:
    rescued_a = frozenset(position for position in rescued if position in grid.target_a_positions)
    rescued_b = frozenset(position for position in rescued if position in grid.target_b_positions)
    visible_a = frozenset(
        position for position, target_type in observation.target_types.items() if target_type == "A"
    )
    visible_b = frozenset(
        position for position, target_type in observation.target_types.items() if target_type == "B"
    )
    return LearningState(
        agent_id=observation.agent_id,
        agent_position=observation.agent_position,
        visible_cells=observation.visible_cells,
        visible_obstacles=observation.obstacles,
        visible_target_a_positions=visible_a,
        visible_target_b_positions=visible_b,
        discovered_cells=discovered_cells,
        discovered_target_a_positions=visible_a,
        discovered_target_b_positions=visible_b,
        rescued_target_a_positions=rescued_a,
        rescued_target_b_positions=rescued_b,
        remaining_target_a_positions=grid.target_a_positions - rescued_a,
        remaining_target_b_positions=grid.target_b_positions - rescued_b,
        steps_taken=steps,
    )


def _valid_actions(grid: Grid, position: Position) -> tuple[Action, ...]:
    movement = MovementModel()
    valid = tuple(Action(move) for move in movement.allowed_moves(grid, position))
    return valid or (Action.WAIT,)


def _learner_actions(
    learner: QLearningAgent,
    grid: Grid,
    position: Position,
) -> tuple[Action, ...]:
    valid = tuple(action for action in _valid_actions(grid, position) if action in learner.actions)
    return valid or (Action.WAIT,)


def _target_type_enum(grid: Grid, position: Position) -> TargetType | None:
    target_type = grid.target_type_at(position)
    if target_type == "A":
        return TargetType.A
    if target_type == "B":
        return TargetType.B
    return None


def _explorable_cell_count(grid: Grid) -> int:
    return grid.width * grid.height - len(grid.obstacles)


def _aggregate_by_agent(runs: list[RunMetrics]) -> list[dict]:
    aggregates = []
    for name in dict.fromkeys(run.agent_name for run in runs):
        agent_runs = [run for run in runs if run.agent_name == name]
        aggregates.append(
            {
                "agent_name": name,
                "algorithm_group": agent_runs[0].algorithm_group,
                "status": "ok" if all(run.status == "ok" for run in agent_runs) else "unavailable",
                "error": next((run.error for run in agent_runs if run.error), None),
                "num_agents": agent_runs[0].num_agents,
                "scenario_count": len(agent_runs),
                "success_rate": round(_average(run.success_rate for run in agent_runs), 4),
                "average_steps": round(_average(run.steps_taken for run in agent_runs), 4),
                "average_accumulated_reward": round(
                    _average(run.accumulated_reward for run in agent_runs), 4
                ),
                "average_rescued_targets": round(
                    _average(run.rescued_targets for run in agent_runs), 4
                ),
                "average_explored_area_percentage": round(
                    _average(run.explored_area_percentage for run in agent_runs), 4
                ),
                "communication_events": round(
                    _average(run.communication_events for run in agent_runs), 4
                ),
            }
        )
    return aggregates


def _average(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _scenario_to_dict(
    scenario: EvaluationScenario,
    split: str = "test",
    start_positions: dict[str, Position] | None = None,
) -> dict:
    grid = generate_grid(scenario.grid_settings, start=scenario.start)
    starts = start_positions or _agent_starts(grid, scenario)
    return {
        "name": scenario.name,
        "split": split,
        "grid": asdict(scenario.grid_settings),
        "max_steps": scenario.max_steps,
        "start": asdict(scenario.start),
        "num_agents": scenario.num_agents,
        "communication_range": scenario.communication_range,
        "agent_starts": {
            agent_id: asdict(position) for agent_id, position in starts.items()
        },
    }


def _scenario(
    name: str,
    width: int,
    height: int,
    obstacle_probability: float,
    target_a_count: int,
    target_b_count: int,
    seed: int,
    max_steps: int,
    start: Position = Position(0, 0),
) -> EvaluationScenario:
    return EvaluationScenario(
        name,
        GridSettings(width, height, obstacle_probability, target_a_count, target_b_count, seed),
        max_steps,
        start=start,
    )


def _agent_starts(grid: Grid, scenario: EvaluationScenario) -> dict[str, Position]:
    return default_start_positions(grid, scenario.num_agents, scenario.start)


def _deep_benchmark_rows() -> list[dict]:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        status = "requires_torch"
    else:
        status = "external_benchmark"

    return [
        {
            "agent_name": name,
            "algorithm_group": "deep_rl_benchmark",
            "evaluation_mode": "fresh_grid_benchmark",
            "status": status,
            "success_rate": None,
            "average_steps": None,
            "average_rescued_targets": None,
            "source": "scripts/compare_all.py",
        }
        for name in DEEP_BENCHMARK_ALGORITHMS
    ]


def _policy_seed(seed: int | None, name: str) -> int:
    return (seed or 0) + (0 if name == "baseline" else 10_000)
