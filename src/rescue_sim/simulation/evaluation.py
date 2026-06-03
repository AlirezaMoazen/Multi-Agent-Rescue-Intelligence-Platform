"""Evaluation tools for single-agent rescue strategies."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import csv
import io
import json
from random import Random
from typing import Callable, Iterable, Literal

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position

AgentName = Literal["baseline", "trained"]
Policy = Callable[[Grid, Position, frozenset[Position], frozenset[Position], Random], Position]

NUM_AGENTS = 1
STEP_REWARD = -1.0
NEW_CELL_REWARD = 0.2
TARGET_REWARD = 10.0
INVALID_MOVE_REWARD = -2.0


@dataclass(frozen=True)
class EvaluationScenario:
    """Configuration for one reproducible evaluation scenario."""

    name: str
    grid_settings: GridSettings
    max_steps: int
    start: Position = Position(0, 0)


@dataclass(frozen=True)
class RunTrace:
    """Raw information collected during one strategy run."""

    scenario_name: str
    agent_name: AgentName
    seed: int | None
    num_agents: int
    steps_taken: int
    max_steps: int
    total_reward: float
    rescued_targets: int
    total_targets: int
    explored_cells: int
    explorable_cells: int


@dataclass(frozen=True)
class RunMetrics:
    """Report-ready metrics for one scenario and one strategy."""

    scenario_name: str
    agent_name: AgentName
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


@dataclass(frozen=True)
class EvaluationReport:
    """Full comparison report for visualization, JSON export, or sprint demos."""

    scenarios: list[dict]
    runs: list[dict]
    aggregates: list[dict]
    sprint_demo_summary: str


def default_evaluation_scenarios() -> list[EvaluationScenario]:
    """Define varied scenarios across seeds, grid sizes, obstacles, and targets."""

    return [
        EvaluationScenario(
            name="small_open_seed_7",
            grid_settings=GridSettings(
                width=5,
                height=5,
                obstacle_probability=0.05,
                target_a_count=1,
                target_b_count=1,
                random_seed=7,
            ),
            max_steps=35,
        ),
        EvaluationScenario(
            name="medium_mixed_seed_13",
            grid_settings=GridSettings(
                width=8,
                height=8,
                obstacle_probability=0.18,
                target_a_count=2,
                target_b_count=2,
                random_seed=13,
            ),
            max_steps=80,
        ),
        EvaluationScenario(
            name="large_dense_seed_23",
            grid_settings=GridSettings(
                width=10,
                height=10,
                obstacle_probability=0.25,
                target_a_count=3,
                target_b_count=2,
                random_seed=23,
            ),
            max_steps=130,
        ),
        EvaluationScenario(
            name="wide_sparse_seed_31",
            grid_settings=GridSettings(
                width=12,
                height=6,
                obstacle_probability=0.12,
                target_a_count=2,
                target_b_count=3,
                random_seed=31,
            ),
            max_steps=95,
        ),
    ]


def evaluate_agents(scenarios: Iterable[EvaluationScenario] | None = None) -> EvaluationReport:
    """Run baseline and trained mock agents under identical scenario conditions."""

    selected_scenarios = list(scenarios or default_evaluation_scenarios())
    runs: list[RunMetrics] = []

    for scenario in selected_scenarios:
        grid = generate_grid(scenario.grid_settings, start=scenario.start)
        for agent_name, policy in _default_policies().items():
            trace = run_agent_on_grid(
                scenario=scenario,
                grid=grid,
                agent_name=agent_name,
                policy=policy,
            )
            runs.append(calculate_run_metrics(trace))

    run_dicts = [asdict(run) for run in runs]
    aggregate_dicts = _aggregate_by_agent(runs)

    return EvaluationReport(
        scenarios=[_scenario_to_dict(scenario) for scenario in selected_scenarios],
        runs=run_dicts,
        aggregates=aggregate_dicts,
        sprint_demo_summary=build_sprint_demo_summary(aggregate_dicts),
    )


def run_agent_on_grid(
    scenario: EvaluationScenario,
    grid: Grid,
    agent_name: AgentName,
    policy: Policy,
) -> RunTrace:
    """Execute one policy on a generated grid and collect raw trace values."""

    rng = Random(_policy_seed(scenario.grid_settings.random_seed, agent_name))
    position = scenario.start
    visited = {position}
    rescued: set[Position] = set()
    all_targets = grid.target_a_positions | grid.target_b_positions
    total_reward = 0.0
    steps_taken = 0

    for step in range(1, scenario.max_steps + 1):
        if rescued == all_targets:
            break

        next_position = policy(grid, position, frozenset(visited), frozenset(rescued), rng)
        steps_taken = step

        if grid.is_valid_position(next_position):
            position = next_position
            total_reward += STEP_REWARD
        else:
            total_reward += INVALID_MOVE_REWARD

        if position not in visited:
            visited.add(position)
            total_reward += NEW_CELL_REWARD

        if position in all_targets and position not in rescued:
            rescued.add(position)
            total_reward += TARGET_REWARD

    return RunTrace(
        scenario_name=scenario.name,
        agent_name=agent_name,
        seed=scenario.grid_settings.random_seed,
        num_agents=NUM_AGENTS,
        steps_taken=steps_taken,
        max_steps=scenario.max_steps,
        total_reward=round(total_reward, 4),
        rescued_targets=len(rescued),
        total_targets=len(all_targets),
        explored_cells=len(visited),
        explorable_cells=_explorable_cell_count(grid),
    )


def calculate_run_metrics(trace: RunTrace) -> RunMetrics:
    """Convert raw trace data into normalized evaluation metrics."""

    success = trace.total_targets == 0 or trace.rescued_targets == trace.total_targets
    explored_percentage = (
        trace.explored_cells / trace.explorable_cells * 100 if trace.explorable_cells else 0.0
    )

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
        explored_area_percentage=round(explored_percentage, 2),
    )


def report_to_json(report: EvaluationReport) -> str:
    """Serialize the evaluation report as deterministic pretty JSON."""

    return json.dumps(asdict(report), indent=2, sort_keys=True)


def report_to_csv(report: EvaluationReport) -> str:
    """Serialize per-run metrics as CSV for spreadsheets or dashboards."""

    rows = report.runs
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def build_sprint_demo_summary(aggregates: list[dict]) -> str:
    """Create a concise text summary for the sprint demo."""

    lines = ["Single-agent evaluation compares the baseline and trained mock agent."]
    for aggregate in aggregates:
        lines.append(
            (
                f"{aggregate['agent_name']}: success_rate={aggregate['success_rate']:.2f}, "
                f"average_steps={aggregate['average_steps']:.1f}, "
                f"average_reward={aggregate['average_accumulated_reward']:.1f}, "
                f"average_rescued_targets={aggregate['average_rescued_targets']:.1f}, "
                f"average_explored_area={aggregate['average_explored_area_percentage']:.1f}%, "
                f"num_agents={aggregate['num_agents']}"
            )
        )
    return "\n".join(lines)


def _default_policies() -> dict[AgentName, Policy]:
    return {
        "baseline": _baseline_policy,
        "trained": _trained_mock_policy,
    }


def _baseline_policy(
    grid: Grid,
    position: Position,
    visited: frozenset[Position],
    rescued: frozenset[Position],
    rng: Random,
) -> Position:
    del rescued, rng

    for neighbor in _neighbors(position):
        if grid.is_valid_position(neighbor) and neighbor not in visited:
            return neighbor

    valid_neighbors = [neighbor for neighbor in _neighbors(position) if grid.is_valid_position(neighbor)]
    return valid_neighbors[0] if valid_neighbors else position


def _trained_mock_policy(
    grid: Grid,
    position: Position,
    visited: frozenset[Position],
    rescued: frozenset[Position],
    rng: Random,
) -> Position:
    del rng

    remaining_targets = (grid.target_a_positions | grid.target_b_positions) - rescued
    next_step = _next_step_towards_any(grid, position, remaining_targets)
    if next_step is not None:
        return next_step

    unvisited = frozenset(_all_free_positions(grid)) - visited
    next_step = _next_step_towards_any(grid, position, unvisited)
    if next_step is not None:
        return next_step

    valid_neighbors = [neighbor for neighbor in _neighbors(position) if grid.is_valid_position(neighbor)]
    return valid_neighbors[0] if valid_neighbors else position


def _next_step_towards_any(
    grid: Grid,
    start: Position,
    goals: frozenset[Position],
) -> Position | None:
    if not goals:
        return None

    queue: deque[tuple[Position, Position | None]] = deque([(start, None)])
    seen = {start}

    while queue:
        current, first_step = queue.popleft()
        if current in goals:
            return first_step

        for neighbor in _neighbors(current):
            if neighbor in seen or not grid.is_valid_position(neighbor):
                continue
            seen.add(neighbor)
            queue.append((neighbor, neighbor if first_step is None else first_step))

    return None


def _neighbors(position: Position) -> tuple[Position, Position, Position, Position]:
    return (
        Position(position.x + 1, position.y),
        Position(position.x, position.y + 1),
        Position(position.x - 1, position.y),
        Position(position.x, position.y - 1),
    )


def _all_free_positions(grid: Grid) -> list[Position]:
    return [
        Position(x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.is_valid_position(Position(x, y))
    ]


def _explorable_cell_count(grid: Grid) -> int:
    return grid.width * grid.height - len(grid.obstacles)


def _aggregate_by_agent(runs: list[RunMetrics]) -> list[dict]:
    aggregates: list[dict] = []

    for agent_name in ("baseline", "trained"):
        agent_runs = [run for run in runs if run.agent_name == agent_name]
        if not agent_runs:
            continue

        aggregates.append(
            {
                "agent_name": agent_name,
                "num_agents": NUM_AGENTS,
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
            }
        )

    return aggregates


def _average(values: Iterable[float]) -> float:
    value_list = list(values)
    return sum(value_list) / len(value_list)


def _scenario_to_dict(scenario: EvaluationScenario) -> dict:
    return {
        "name": scenario.name,
        "grid": asdict(scenario.grid_settings),
        "max_steps": scenario.max_steps,
        "start": asdict(scenario.start),
        "num_agents": NUM_AGENTS,
    }


def _policy_seed(seed: int | None, agent_name: AgentName) -> int:
    base_seed = seed or 0
    agent_offset = 0 if agent_name == "baseline" else 10_000
    return base_seed + agent_offset
