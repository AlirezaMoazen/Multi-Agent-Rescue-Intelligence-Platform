from rescue_sim.environment.grid import Position
from rescue_sim.simulation.evaluation import (
    EvaluationScenario,
    RunTrace,
    calculate_run_metrics,
    evaluate_agents,
    report_to_csv,
    report_to_json,
)
from rescue_sim.shared import GridSettings


def test_calculate_run_metrics_reports_success_and_exploration() -> None:
    trace = RunTrace(
        scenario_name="unit",
        agent_name="baseline",
        algorithm_group="baseline",
        status="ok",
        seed=1,
        num_agents=1,
        steps_taken=12,
        max_steps=20,
        total_reward=5.5,
        rescued_targets=2,
        total_targets=2,
        explored_cells=8,
        explorable_cells=10,
    )

    metrics = calculate_run_metrics(trace)

    assert metrics.success
    assert metrics.success_rate == 1.0
    assert metrics.average_steps == 12.0
    assert metrics.average_accumulated_reward == 5.5
    assert metrics.rescued_targets == 2
    assert metrics.explored_area_percentage == 80.0
    assert metrics.num_agents == 1


def test_evaluation_is_reproducible_for_same_scenarios() -> None:
    scenarios = [
        EvaluationScenario(
            name="reproducible",
            grid_settings=GridSettings(
                width=5,
                height=5,
                obstacle_probability=0.1,
                target_a_count=1,
                target_b_count=1,
                random_seed=11,
            ),
            max_steps=40,
            start=Position(0, 0),
        )
    ]

    first_report = evaluate_agents(scenarios)
    second_report = evaluate_agents(scenarios)

    assert first_report == second_report
    assert report_to_json(first_report) == report_to_json(second_report)
    assert report_to_csv(first_report) == report_to_csv(second_report)


def test_evaluation_includes_single_agent_count_everywhere() -> None:
    report = evaluate_agents(
        [
            EvaluationScenario(
                name="single-agent-format",
                grid_settings=GridSettings(
                    width=4,
                    height=4,
                    obstacle_probability=0.0,
                    target_a_count=1,
                    target_b_count=0,
                    random_seed=3,
                ),
                max_steps=20,
                num_agents=1,
            )
        ]
    )

    assert {scenario["num_agents"] for scenario in report.scenarios} == {1}
    assert {run["num_agents"] for run in report.runs} == {1}
    assert {aggregate["num_agents"] for aggregate in report.aggregates} == {1}


def test_default_evaluation_separates_training_and_test_scenarios() -> None:
    report = evaluate_agents()

    training_seeds = {
        scenario["grid"]["random_seed"] for scenario in report.training_scenarios
    }
    test_seeds = {scenario["grid"]["random_seed"] for scenario in report.scenarios}

    assert report.training_scenarios
    assert report.scenarios
    assert training_seeds.isdisjoint(test_seeds)
    assert {scenario["split"] for scenario in report.training_scenarios} == {"train"}
    assert {scenario["split"] for scenario in report.scenarios} == {"test"}
