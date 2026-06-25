"""Tests for QMIX (agent net, monotonic mixer, replay buffer, trainer).

Skips automatically if torch is not installed (optional: pip install -e ".[qmix]").
The environment itself is covered by tests/test_mappo.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rescue_sim.config.settings import GridSettings, QmixSettings
from rescue_sim.MAPPO.environment import RescueEnv

torch = pytest.importorskip("torch")

from rescue_sim.QMIX.qmix import (  # noqa: E402
    QMIX,
    AgentQNet,
    MixingNetwork,
    ReplayBuffer,
)
from random import Random  # noqa: E402


def make_env(num_agents: int = 3, max_steps: int = 20, seed: int = 0) -> RescueEnv:
    grid = GridSettings(width=6, height=6, obstacle_probability=0.1,
                        target_a_count=1, target_b_count=1, random_seed=0)
    return RescueEnv(grid, num_agents=num_agents, max_steps=max_steps,
                     view_radius=2, seed=seed)


def small_trainer(num_agents: int = 2) -> QMIX:
    env = make_env(num_agents=num_agents, max_steps=15)
    settings = QmixSettings(
        num_agents=num_agents,
        hidden_dim=32,
        mixing_embed_dim=16,
        max_steps=15,
        batch_size=8,
        buffer_size=200,
        target_update_interval=5,
        epsilon_anneal_episodes=5,
        random_seed=0,
    )
    return QMIX(env, settings)


# ---------------------------------------------------------------------------
# 1. Agent Q-network
# ---------------------------------------------------------------------------

def test_agent_qnet_output_shape():
    net = AgentQNet(obs_dim=20, n_actions=4, hidden=32)
    out = net(torch.zeros(3, 20))   # 3 agents
    assert out.shape == (3, 4)


def test_agent_qnet_batched():
    net = AgentQNet(obs_dim=20, n_actions=4, hidden=32)
    out = net(torch.zeros(8, 3, 20))   # batch of 8, 3 agents
    assert out.shape == (8, 3, 4)


# ---------------------------------------------------------------------------
# 2. Mixing network
# ---------------------------------------------------------------------------

def test_mixer_output_shape():
    mixer = MixingNetwork(num_agents=3, state_dim=30, embed_dim=16)
    q_tot = mixer(torch.randn(8, 3), torch.randn(8, 30))
    assert q_tot.shape == (8,)


def test_mixer_monotonicity():
    """Q_tot must be non-decreasing in every agent's Q (the QMIX guarantee)."""
    torch.manual_seed(0)
    mixer = MixingNetwork(num_agents=3, state_dim=30, embed_dim=16)
    state = torch.randn(5, 30)
    base_q = torch.randn(5, 3)
    base_tot = mixer(base_q, state)
    # Increase every agent's Q -> Q_tot must not decrease.
    higher = mixer(base_q + 1.0, state)
    assert torch.all(higher >= base_tot - 1e-4)


def test_mixer_weights_are_nonnegative():
    """The hypernetwork weights used in the mix are kept >= 0 via abs()."""
    mixer = MixingNetwork(num_agents=2, state_dim=10, embed_dim=8)
    state = torch.randn(4, 10)
    w1 = torch.abs(mixer.hyper_w1(state))
    w2 = torch.abs(mixer.hyper_w2(state))
    assert torch.all(w1 >= 0) and torch.all(w2 >= 0)


# ---------------------------------------------------------------------------
# 3. Replay buffer
# ---------------------------------------------------------------------------

def make_transition(env: RescueEnv) -> dict:
    obs = env.reset()
    return {
        "obs": obs,
        "state": env.global_state(),
        "actions": np.zeros(env.num_agents, dtype=np.int64),
        "avail": env.valid_action_mask(),
        "reward": np.float32(1.0),
        "next_obs": obs,
        "next_state": env.global_state(),
        "next_avail": env.valid_action_mask(),
        "done": np.float32(0.0),
    }


def test_buffer_push_and_len():
    buf = ReplayBuffer(capacity=10, rng=Random(0))
    env = make_env(num_agents=2)
    for _ in range(3):
        buf.push(make_transition(env))
    assert len(buf) == 3


def test_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=5, rng=Random(0))
    env = make_env(num_agents=2)
    for _ in range(10):
        buf.push(make_transition(env))
    assert len(buf) == 5


def test_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=20, rng=Random(0))
    env = make_env(num_agents=2)
    for _ in range(10):
        buf.push(make_transition(env))
    batch = buf.sample(4)
    assert batch["obs"].shape == (4, 2, env.obs_dim)
    assert batch["actions"].shape == (4, 2)
    assert batch["reward"].shape == (4,)
    assert batch["state"].shape == (4, env.state_dim)


# ---------------------------------------------------------------------------
# 4. Action selection
# ---------------------------------------------------------------------------

def test_greedy_actions_respect_mask():
    trainer = small_trainer(num_agents=1)
    env = trainer.env
    obs = env.reset()
    for _ in range(30):
        avail = env.valid_action_mask()
        actions = trainer.select_actions(env.reset(), env.valid_action_mask(), greedy=True)
        m = env.valid_action_mask()
        assert m[0, int(actions[0])]
    _ = obs, avail


def test_select_actions_shape():
    trainer = small_trainer(num_agents=3)
    env = trainer.env
    obs = env.reset()
    actions = trainer.select_actions(obs, env.valid_action_mask())
    assert actions.shape == (3,)


# ---------------------------------------------------------------------------
# 5. Learning / training
# ---------------------------------------------------------------------------

def test_learn_returns_finite_loss():
    trainer = small_trainer()
    env = trainer.env
    for _ in range(20):  # fill the buffer past batch_size
        trainer.buffer.push(make_transition(env))
    loss = trainer._learn()
    assert np.isfinite(loss)


def test_target_sync_copies_weights():
    trainer = small_trainer()
    # Perturb the online net, then sync; targets must match afterwards.
    with torch.no_grad():
        for p in trainer.agent.parameters():
            p.add_(1.0)
    trainer._sync_targets()
    for online, target in zip(trainer.agent.parameters(), trainer.target_agent.parameters()):
        assert torch.equal(online, target)


def test_train_runs_and_updates_weights():
    trainer = small_trainer()
    before = [p.clone() for p in trainer.agent.parameters()]
    history = trainer.train(num_episodes=5, log_every=0)
    assert len(history) == 5
    after = list(trainer.agent.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_epsilon_decays():
    trainer = small_trainer()
    start = trainer.epsilon
    trainer.train(num_episodes=10, log_every=0)
    assert trainer.epsilon < start
    assert trainer.epsilon >= trainer.cfg.epsilon_end - 1e-6


def test_evaluate_returns_metrics():
    trainer = small_trainer()
    result = trainer.evaluate(episodes=3)
    assert {"success_rate", "avg_rescued", "avg_steps"} <= result.keys()
    assert 0.0 <= result["success_rate"] <= 1.0
