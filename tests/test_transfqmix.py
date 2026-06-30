"""Tests for TransfQMix (entity tokens, transformer agent, transformer mixer).

Skips automatically if torch is not installed (optional: pip install -e ".[transfqmix]").
"""

from __future__ import annotations

import numpy as np
import pytest

from rescue_sim.config.settings import GridSettings, TransfQmixSettings

torch = pytest.importorskip("torch")

from rescue_sim.TransfQMix.transf_qmix import (  # noqa: E402
    TOKEN_DIM,
    AgentTransformer,
    EntityRescueEnv,
    TransfQMIX,
    TransformerMixer,
)


def make_env(num_agents: int = 3, max_steps: int = 15, seed: int = 0) -> EntityRescueEnv:
    grid = GridSettings(width=6, height=6, obstacle_probability=0.1,
                        target_a_count=1, target_b_count=1, random_seed=0)
    return EntityRescueEnv(grid, num_agents=num_agents, max_steps=max_steps,
                           view_radius=2, seed=seed)


def small_settings(num_agents: int = 2) -> TransfQmixSettings:
    return TransfQmixSettings(
        num_agents=num_agents,
        d_model=16,
        n_heads=2,
        n_agent_layers=1,
        n_mixer_layers=1,
        ff_dim=32,
        mixing_embed_dim=16,
        max_steps=12,
        batch_size=8,
        buffer_size=200,
        target_update_interval=5,
        epsilon_anneal_episodes=5,
        random_seed=0,
    )


def small_trainer(num_agents: int = 2) -> TransfQMIX:
    env = make_env(num_agents=num_agents, max_steps=12)
    return TransfQMIX(env, small_settings(num_agents))


# ---------------------------------------------------------------------------
# 1. Entity tokenization
# ---------------------------------------------------------------------------

def test_entity_obs_shape():
    env = make_env(num_agents=3)
    env.reset()
    tokens = env.entity_obs()
    assert tokens.shape == (3, env.n_tokens, env.token_dim)


def test_token_dim_and_count():
    env = make_env(num_agents=2)
    win = 2 * env.view_radius + 1
    assert env.token_dim == TOKEN_DIM == 9
    assert env.n_tokens == win * win + 1  # window cells + self token


def test_self_token_flag_set():
    env = make_env(num_agents=1)
    env.reset()
    tokens = env.entity_obs()
    # Last token is the self token: is_self flag (index 6) == 1.
    assert tokens[0, -1, 6] == 1.0
    # Window-cell tokens have is_self == 0.
    assert np.all(tokens[0, :-1, 6] == 0.0)


# ---------------------------------------------------------------------------
# 2. Agent transformer
# ---------------------------------------------------------------------------

def test_agent_transformer_outputs():
    cfg = small_settings()
    net = AgentTransformer(token_dim=TOKEN_DIM, n_actions=4, cfg=cfg)
    tokens = torch.zeros(5, 26, TOKEN_DIM)   # 5 agents, 26 tokens
    q, h = net(tokens)
    assert q.shape == (5, 4)
    assert h.shape == (5, cfg.d_model)


# ---------------------------------------------------------------------------
# 3. Transformer mixer
# ---------------------------------------------------------------------------

def test_mixer_output_shape():
    cfg = small_settings()
    env = make_env(num_agents=3)
    env.reset()
    mixer = TransformerMixer(state_dim=env.state_dim, cfg=cfg)
    agent_qs = torch.randn(4, 3)
    agent_h = torch.randn(4, 3, cfg.d_model)
    state = torch.randn(4, env.state_dim)
    q_tot = mixer(agent_qs, agent_h, state)
    assert q_tot.shape == (4,)


def test_mixer_monotonicity():
    """Q_tot must be non-decreasing in every agent's Q (the QMIX guarantee)."""
    torch.manual_seed(0)
    cfg = small_settings()
    env = make_env(num_agents=3)
    env.reset()
    mixer = TransformerMixer(state_dim=env.state_dim, cfg=cfg)
    agent_h = torch.randn(5, 3, cfg.d_model)
    state = torch.randn(5, env.state_dim)
    base_q = torch.randn(5, 3)
    base = mixer(base_q, agent_h, state)
    higher = mixer(base_q + 1.0, agent_h, state)
    assert torch.all(higher >= base - 1e-4)


def test_mixer_transferable_across_agent_count():
    """Same mixer parameters accept different numbers of agents (transferability)."""
    cfg = small_settings()
    env = make_env(num_agents=3)
    env.reset()
    mixer = TransformerMixer(state_dim=env.state_dim, cfg=cfg)
    state = torch.randn(2, env.state_dim)
    for n in (2, 3, 5):  # different team sizes, no parameter change
        q_tot = mixer(torch.randn(2, n), torch.randn(2, n, cfg.d_model), state)
        assert q_tot.shape == (2,)


# ---------------------------------------------------------------------------
# 4. Action selection
# ---------------------------------------------------------------------------

def test_greedy_actions_respect_mask():
    trainer = small_trainer(num_agents=1)
    env = trainer.env
    for _ in range(20):
        env.reset()
        actions = trainer.select_actions(env.entity_obs(), env.valid_action_mask(), greedy=True)
        assert env.valid_action_mask()[0, int(actions[0])]


def test_select_actions_shape():
    trainer = small_trainer(num_agents=3)
    env = trainer.env
    env.reset()
    actions = trainer.select_actions(env.entity_obs(), env.valid_action_mask())
    assert actions.shape == (3,)


# ---------------------------------------------------------------------------
# 5. Learning / training
# ---------------------------------------------------------------------------

def test_learn_returns_finite_loss():
    trainer = small_trainer()
    env = trainer.env
    for _ in range(20):
        env.reset()
        trainer.buffer.push({
            "obs": env.entity_obs(), "state": env.global_state(),
            "actions": np.zeros(env.num_agents, dtype=np.int64),
            "avail": env.valid_action_mask(), "reward": np.float32(1.0),
            "next_obs": env.entity_obs(), "next_state": env.global_state(),
            "next_avail": env.valid_action_mask(), "done": np.float32(0.0),
        })
    loss = trainer._learn()
    assert np.isfinite(loss)


def test_target_sync_copies_weights():
    trainer = small_trainer()
    with torch.no_grad():
        for p in trainer.agent.parameters():
            p.add_(0.5)
    trainer._sync_targets()
    for online, target in zip(trainer.agent.parameters(), trainer.target_agent.parameters()):
        assert torch.equal(online, target)


def test_train_runs_and_updates_weights():
    trainer = small_trainer()
    before = [p.clone() for p in trainer.agent.parameters()]
    history = trainer.train(num_episodes=4, log_every=0)
    assert len(history) == 4
    after = list(trainer.agent.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_returns_metrics():
    trainer = small_trainer()
    result = trainer.evaluate(episodes=2)
    assert {"success_rate", "avg_rescued", "avg_steps"} <= result.keys()
    assert 0.0 <= result["success_rate"] <= 1.0
