"""Tests for the value ensemble and distillation.

Skips automatically if torch is not installed (optional: pip install -e ".[ensemble]").
"""

from __future__ import annotations

import numpy as np
import pytest

from rescue_sim.config.settings import (
    DistillSettings,
    GridSettings,
    QmixSettings,
    TransfQmixSettings,
)

torch = pytest.importorskip("torch")

from rescue_sim.Ensemble.distill import Distiller  # noqa: E402
from rescue_sim.Ensemble.ensemble import ValueEnsemble, performance_weights  # noqa: E402
from rescue_sim.QMIX.qmix import QMIX  # noqa: E402
from rescue_sim.TransfQMix.transf_qmix import EntityRescueEnv, TransfQMIX  # noqa: E402


def make_env(num_agents: int = 2, seed: int = 0) -> EntityRescueEnv:
    grid = GridSettings(width=6, height=6, obstacle_probability=0.1,
                        target_a_count=1, target_b_count=1, random_seed=0)
    return EntityRescueEnv(grid, num_agents=num_agents, max_steps=15, view_radius=2, seed=seed)


def make_trainers(num_agents: int = 2):
    qmix = QMIX(make_env(num_agents), QmixSettings(
        num_agents=num_agents, hidden_dim=32, mixing_embed_dim=16, max_steps=15,
        batch_size=8, buffer_size=200, target_update_interval=5,
        epsilon_anneal_episodes=3, random_seed=0))
    transf = TransfQMIX(make_env(num_agents), TransfQmixSettings(
        num_agents=num_agents, d_model=16, n_heads=2, n_agent_layers=1, n_mixer_layers=1,
        ff_dim=32, mixing_embed_dim=16, max_steps=15, batch_size=8, buffer_size=200,
        target_update_interval=5, epsilon_anneal_episodes=3, random_seed=0))
    qmix.train(num_episodes=2, log_every=0)
    transf.train(num_episodes=2, log_every=0)
    return qmix, transf


# ---------------------------------------------------------------------------
# 1. Weights
# ---------------------------------------------------------------------------

def test_performance_weights_normalize():
    w_q, w_t = performance_weights(0.6, 0.4)
    assert abs((w_q + w_t) - 1.0) < 1e-6
    assert w_q > w_t   # stronger method gets more weight


def test_performance_weights_zero_fallback():
    assert performance_weights(0.0, 0.0) == (0.5, 0.5)


# ---------------------------------------------------------------------------
# 2. Value ensemble
# ---------------------------------------------------------------------------

def test_ensemble_combined_q_shape():
    qmix, transf = make_trainers(num_agents=2)
    env = make_env(2)
    flat = env.reset()
    tokens = env.entity_obs()
    ens = ValueEnsemble(qmix, transf, env, 0.5, 0.5)
    q = ens.combined_q(flat, tokens)
    assert q.shape == (2, env.n_actions)


def test_ensemble_weights_sum_to_one():
    qmix, transf = make_trainers()
    ens = ValueEnsemble(qmix, transf, make_env(2), 0.8, 0.2)
    assert abs((ens.w_qmix + ens.w_transf) - 1.0) < 1e-6


def test_ensemble_respects_action_mask():
    qmix, transf = make_trainers(num_agents=1)
    env = make_env(1)
    ens = ValueEnsemble(qmix, transf, env, 0.5, 0.5)
    for _ in range(20):
        flat = env.reset()
        actions = ens.select_actions(flat, env.entity_obs(), env.valid_action_mask())
        assert env.valid_action_mask()[0, int(actions[0])]


def test_ensemble_evaluate_metrics():
    qmix, transf = make_trainers()
    ens = ValueEnsemble(qmix, transf, make_env(2), 0.5, 0.5)
    result = ens.evaluate(episodes=3)
    assert {"success_rate", "avg_rescued", "avg_steps"} <= result.keys()
    assert 0.0 <= result["success_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 3. Distillation
# ---------------------------------------------------------------------------

def test_distiller_collect_shapes():
    qmix, transf = make_trainers(num_agents=2)
    env = make_env(2)
    ens = ValueEnsemble(qmix, transf, env, 0.5, 0.5)
    distiller = Distiller(ens, env, DistillSettings(collect_steps=20, random_seed=0))
    x, y = distiller.collect(20)
    assert x.shape[1] == env.obs_dim
    assert y.shape[1] == env.n_actions
    assert x.shape[0] == y.shape[0]


def test_distiller_train_returns_losses():
    qmix, transf = make_trainers(num_agents=2)
    env = make_env(2)
    ens = ValueEnsemble(qmix, transf, env, 0.5, 0.5)
    distiller = Distiller(ens, env, DistillSettings(collect_steps=40, epochs=3,
                                                    batch_size=8, random_seed=0))
    losses = distiller.train()
    assert len(losses) == 3
    assert all(np.isfinite(loss) for loss in losses)


def test_distiller_student_is_single_network_and_evaluates():
    qmix, transf = make_trainers(num_agents=2)
    env = make_env(2)
    ens = ValueEnsemble(qmix, transf, env, 0.5, 0.5)
    distiller = Distiller(ens, env, DistillSettings(collect_steps=40, epochs=2,
                                                    batch_size=8, random_seed=0))
    distiller.train()
    # The student uses only the flat observation -> input dim == env.obs_dim.
    first_layer = distiller.student.net[0]
    assert first_layer.in_features == env.obs_dim
    result = distiller.evaluate(episodes=3)
    assert 0.0 <= result["success_rate"] <= 1.0
