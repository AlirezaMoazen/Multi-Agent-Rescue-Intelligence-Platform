"""Tests for the Mixture-of-Experts router (rescue_sim.MoE).

torch is an optional dependency, so the whole module is skipped when it is
missing -- exactly like the other deep-RL test modules.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from rescue_sim.config.settings import (  # noqa: E402
    GridSettings,
    MappoSettings,
    MoeSettings,
    QmixSettings,
    TransfQmixSettings,
)
from rescue_sim.Ensemble import ValueEnsemble  # noqa: E402
from rescue_sim.MAPPO import MAPPO, RescueEnv  # noqa: E402
from rescue_sim.MoE import FixedGridEntityEnv, MixtureOfExperts, solve_score  # noqa: E402
from rescue_sim.QMIX import QMIX  # noqa: E402
from rescue_sim.Qlearning.multi_agent_baseline import DEFAULT_MULTI_AGENT_BASELINES  # noqa: E402
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX  # noqa: E402


GRID = GridSettings(width=5, height=5, obstacle_probability=0.1,
                    target_a_count=1, target_b_count=1)


def _make_moe(num_agents: int = 2, trials: int = 4, grid: GridSettings = GRID,
              max_steps: int = 40, baselines=None, with_mappo: bool = False,
              **moe_kwargs) -> MixtureOfExperts:
    """A tiny MoE with *untrained* deep nets -- enough to exercise the gate fast."""
    def env() -> EntityRescueEnv:
        return EntityRescueEnv(grid, num_agents=num_agents, max_steps=max_steps,
                               view_radius=1, seed=0)

    qmix = QMIX(env(), QmixSettings(num_agents=num_agents, random_seed=0))
    transf = TransfQMIX(env(), TransfQmixSettings(num_agents=num_agents, random_seed=0))
    ensemble = ValueEnsemble(qmix, transf, env())
    mappo = None
    if with_mappo:
        mappo = MAPPO(
            RescueEnv(grid, num_agents=num_agents, max_steps=max_steps, view_radius=1, seed=0),
            MappoSettings(num_agents=num_agents, random_seed=0),
        )
    kwargs = dict(num_trials=trials, random_seed=0, grid_seed=0,
                  epsilon_start=0.4, epsilon_decay=0.05)
    kwargs.update(moe_kwargs)
    bl = DEFAULT_MULTI_AGENT_BASELINES if baselines is None else baselines
    return MixtureOfExperts.from_models(
        qmix=qmix, transf=transf, mappo=mappo, ensemble=ensemble,
        settings=MoeSettings(**kwargs), baselines=bl,
    )


def test_solve_score_orders_success_above_failure():
    success = {"success": True, "rescued": 2, "targets": 2, "steps": 30}
    failure = {"success": False, "rescued": 2, "targets": 2, "steps": 5}
    assert solve_score(success, max_steps=40) > solve_score(failure, max_steps=40)
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


def test_leaderboard_scores_all_experts():
    """Every classical baseline and deep expert (incl. MAPPO) appears, scored once."""
    moe = _make_moe(with_mappo=True, trials=3)
    report = moe.run()
    # 7 classical baselines + QMIX + TransfQMix + MAPPO + Ensemble.
    for name in DEFAULT_MULTI_AGENT_BASELINES:
        assert name in report.leaderboard
    for name in ("QMIX", "TransfQMix", "MAPPO", "Ensemble"):
        assert name in report.leaderboard
    for m in report.leaderboard.values():
        assert set(m) == {"success", "rescued", "targets", "steps", "score"}
    assert report.teacher in ("QMIX", "TransfQMix", "MAPPO", "Ensemble")


def test_moe_run_produces_consistent_history():
    moe = _make_moe(num_agents=2, trials=5)
    report = moe.run()
    assert len(report.history) == 5
    known = set(report.leaderboard) | {"EpidemicFleet"}
    for i, row in enumerate(report.history, start=1):
        assert row.trial == i
        assert row.leader in known
        assert 0.0 <= row.epsilon <= 1.0


def test_moe_report_is_json_serializable():
    import json

    report = _make_moe(trials=3).run()
    payload = report.to_dict()
    assert json.loads(json.dumps(payload))["final_leader"]
    assert len(payload["history"]) == 3
    assert payload["leaderboard"]  # non-empty


def test_specialist_overtakes_weak_generalists():
    """On a single-target grid with only weak (untrained) deep experts in the
    pool, the fleet must learn the grid and lead -- the headline MoE story."""
    single = GridSettings(width=10, height=10, obstacle_probability=0.2,
                          target_a_count=1, target_b_count=0)
    # A grid the untrained deep experts cannot solve (long path behind obstacles);
    # isolate the deep-vs-fleet gate by excluding the classical baselines.
    moe = _make_moe(num_agents=2, trials=25, grid=single, max_steps=120,
                    baselines={}, grid_seed=1, epsilon_start=0.6, epsilon_decay=0.03)
    report = moe.run()
    assert report.surpassed_at is not None
    assert report.final_leader == "EpidemicFleet"
    leaders = [row.leader for row in report.history]
    first = leaders.index("EpidemicFleet")
    assert all(lead == "EpidemicFleet" for lead in leaders[first:])  # monotone takeover


def test_adaptive_expert_learns_over_trials():
    moe = _make_moe(num_agents=2, trials=6, baselines={})
    report = moe.run()
    assert report.history[-1].mean_q >= report.history[0].mean_q
