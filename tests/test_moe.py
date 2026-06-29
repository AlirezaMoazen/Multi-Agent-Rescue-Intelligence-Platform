"""Tests for the Mixture-of-Experts gate (rescue_sim.MoE).

torch is an optional dependency, so the whole module is skipped when it is
missing -- exactly like the other deep-RL test modules.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from rescue_sim.config.settings import (  # noqa: E402
    GridSettings,
    MoeSettings,
    QmixSettings,
    TransfQmixSettings,
)
from rescue_sim.Ensemble import ValueEnsemble  # noqa: E402
from rescue_sim.MoE import FixedGridEntityEnv, MixtureOfExperts, solve_score  # noqa: E402
from rescue_sim.QMIX import QMIX  # noqa: E402
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX  # noqa: E402


GRID = GridSettings(width=5, height=5, obstacle_probability=0.1,
                    target_a_count=1, target_b_count=1)


def _make_moe(num_agents: int = 2, trials: int = 4, grid: GridSettings = GRID,
              max_steps: int = 40, **moe_kwargs) -> MixtureOfExperts:
    """A tiny MoE with *untrained* deep nets -- enough to exercise the gate fast."""
    def env() -> EntityRescueEnv:
        return EntityRescueEnv(grid, num_agents=num_agents, max_steps=max_steps,
                               view_radius=1, seed=0)

    qmix = QMIX(env(), QmixSettings(num_agents=num_agents, random_seed=0))
    transf = TransfQMIX(env(), TransfQmixSettings(num_agents=num_agents, random_seed=0))
    deep = ValueEnsemble(qmix, transf, env())
    kwargs = dict(num_trials=trials, random_seed=0, grid_seed=0,
                  epsilon_start=0.4, epsilon_decay=0.05)
    kwargs.update(moe_kwargs)
    return MixtureOfExperts(deep, MoeSettings(**kwargs))


def test_solve_score_orders_success_above_failure():
    success = {"success": True, "rescued": 2, "targets": 2, "steps": 30}
    failure = {"success": False, "rescued": 2, "targets": 2, "steps": 5}
    # A full success must outrank any failure, even a fast one.
    assert solve_score(success, max_steps=40) > solve_score(failure, max_steps=40)
    # Among successes, fewer steps scores higher.
    faster = {"success": True, "rescued": 2, "targets": 2, "steps": 10}
    assert solve_score(faster, max_steps=40) > solve_score(success, max_steps=40)


def test_fixed_grid_env_does_not_regenerate():
    moe = _make_moe()
    assert isinstance(moe.env, FixedGridEntityEnv)
    obs = moe.env.reset()
    grid1 = moe.env.grid
    moe.env.reset()
    grid2 = moe.env.grid
    assert grid1 is grid2          # same fixed grid object across resets
    assert grid1 is moe.grid
    assert obs.shape[0] == moe.num_agents


def test_moe_run_produces_consistent_history():
    moe = _make_moe(num_agents=2, trials=5)
    report = moe.run()
    assert len(report.history) == 5
    # Trials are numbered 1..N and leader is always a known expert.
    for i, row in enumerate(report.history, start=1):
        assert row.trial == i
        assert row.leader in ("deep", "adaptive")
        assert 0.0 <= row.epsilon <= 1.0
        assert set(row.deep) == {"success", "rescued", "targets", "steps", "score"}
    # Once handed over, the leader never reverts to deep (monotone takeover).
    leaders = [row.leader for row in report.history]
    if "adaptive" in leaders:
        first = leaders.index("adaptive")
        assert all(lead == "adaptive" for lead in leaders[first:])


def test_moe_report_is_json_serializable():
    import json

    report = _make_moe(trials=3).run()
    payload = report.to_dict()
    # Round-trips through JSON -- ready for an API response.
    assert json.loads(json.dumps(payload))["final_leader"] in ("deep", "adaptive")
    assert len(payload["history"]) == 3


def test_adaptive_expert_learns_over_trials():
    """The fleet's mean Q-value should grow as it learns the fixed grid."""
    moe = _make_moe(num_agents=2, trials=6)
    report = moe.run()
    early = report.history[0].mean_q
    late = report.history[-1].mean_q
    assert late >= early  # learning never makes the mean Q strictly worse here


def test_specialist_overtakes_a_weak_generalist():
    """On a single-target grid an untrained deep expert cannot reach the target,
    so the tabular specialist must learn it and take over (the headline MoE story)."""
    single = GridSettings(width=8, height=8, obstacle_probability=0.15,
                          target_a_count=1, target_b_count=0)
    moe = _make_moe(num_agents=2, trials=25, grid=single, max_steps=100,
                    grid_seed=4, epsilon_start=0.6, epsilon_decay=0.03)
    report = moe.run()
    assert report.surpassed_at is not None            # a handover happened
    assert report.final_leader == "adaptive"          # specialist ended in charge
    assert report.best_adaptive_score > report.deep_score
    # Once handed over, the leader is monotone (never reverts to deep).
    leaders = [row.leader for row in report.history]
    first = leaders.index("adaptive")
    assert all(lead == "adaptive" for lead in leaders[first:])
