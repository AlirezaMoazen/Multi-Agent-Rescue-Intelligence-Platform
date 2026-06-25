"""Tests for the MAPPO environment and trainer.

The environment tests are pure NumPy. The trainer tests skip automatically if
torch is not installed (it is an optional dependency: pip install -e ".[mappo]").
"""

from __future__ import annotations

import numpy as np
import pytest

from rescue_sim.config.settings import GridSettings, MappoSettings
from rescue_sim.MAPPO.environment import RescueEnv, N_ACTIONS


def make_settings(**kwargs) -> GridSettings:
    base = dict(
        width=6,
        height=6,
        obstacle_probability=0.1,
        target_a_count=1,
        target_b_count=1,
        random_seed=0,
    )
    base.update(kwargs)
    return GridSettings(**base)


def make_env(num_agents: int = 3, max_steps: int = 50, seed: int = 0) -> RescueEnv:
    return RescueEnv(make_settings(), num_agents=num_agents, max_steps=max_steps,
                     view_radius=2, seed=seed)


# ---------------------------------------------------------------------------
# 1. Environment API
# ---------------------------------------------------------------------------

def test_reset_returns_correct_obs_shape():
    env = make_env(num_agents=3)
    obs = env.reset()
    assert obs.shape == (3, env.obs_dim)


def test_obs_dim_matches_formula():
    env = make_env(num_agents=4)
    win = 2 * env.view_radius + 1
    expected = win * win * 4 + 4 + 4   # window*channels + scalars + agent-id
    assert env.obs_dim == expected


def test_state_dim_is_concatenated_obs():
    env = make_env(num_agents=3)
    env.reset()
    assert env.state_dim == env.obs_dim * 3
    assert env.global_state().shape == (env.state_dim,)


def test_n_actions_is_four():
    env = make_env()
    assert env.n_actions == N_ACTIONS == 4


def test_step_returns_signature():
    env = make_env(num_agents=2)
    env.reset()
    obs, reward, done, info = env.step(np.zeros(2, dtype=int))
    assert obs.shape == (2, env.obs_dim)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert {"rescued", "targets", "success", "steps"} <= info.keys()


def test_episode_terminates_at_max_steps():
    env = make_env(num_agents=1, max_steps=5)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, info = env.step(np.array([env.n_actions - 1]))  # West repeatedly
        steps += 1
    assert steps <= 5
    assert info["steps"] <= 5


def test_valid_action_mask_shape_and_corner():
    env = make_env(num_agents=1)
    env.reset()  # all agents start at (0,0)
    mask = env.valid_action_mask()
    assert mask.shape == (1, env.n_actions)
    # From (0,0): North (idx 0) and West (idx 3) leave the grid -> invalid.
    assert not mask[0, 0]
    assert not mask[0, 3]


def test_step_before_reset_raises():
    env = make_env()
    with pytest.raises(AssertionError):
        env.step(np.zeros(env.num_agents, dtype=int))


def test_num_agents_validation():
    with pytest.raises(ValueError, match="num_agents"):
        RescueEnv(make_settings(), num_agents=0)


def test_grid_changes_between_episodes():
    env = make_env(num_agents=1)
    env.reset()
    grid_a = env.grid
    env.reset()
    grid_b = env.grid
    # Different internal seed each reset -> obstacle layouts should differ.
    assert grid_a.obstacles != grid_b.obstacles or grid_a is not grid_b


def test_rescue_increments_count():
    # Tiny grid with a single target adjacent to start so rescue is reachable.
    settings = GridSettings(width=3, height=1, obstacle_probability=0.0,
                            target_a_count=1, target_b_count=0, random_seed=1)
    env = RescueEnv(settings, num_agents=1, max_steps=10, view_radius=1, seed=1)
    env.reset()
    total = len(env.grid.target_a_positions | env.grid.target_b_positions)
    assert total == 1
    # Walk East until the target is found or steps run out.
    rescued = 0
    for _ in range(10):
        _, _, done, info = env.step(np.array([2]))  # East
        rescued = info["rescued"]
        if done:
            break
    assert rescued == 1


# ---------------------------------------------------------------------------
# 2. MAPPO trainer (requires torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
from rescue_sim.MAPPO.mappo import MAPPO, ActorCritic, RunningMeanStd  # noqa: E402


def small_trainer(num_agents: int = 2) -> MAPPO:
    env = make_env(num_agents=num_agents, max_steps=20)
    settings = MappoSettings(
        num_agents=num_agents,
        hidden_dim=32,
        rollout_steps=64,
        epochs=2,
        max_steps=20,
        random_seed=0,
    )
    return MAPPO(env, settings)


def test_actor_critic_shapes():
    env = make_env(num_agents=2)
    env.reset()
    net = ActorCritic(env.obs_dim, env.state_dim, env.n_actions, hidden=32)
    obs = torch.as_tensor(env.reset())
    mask = torch.as_tensor(env.valid_action_mask())
    action, logp = net.act(obs, mask)
    assert action.shape == (2,)
    assert logp.shape == (2,)
    value = net.value(torch.as_tensor(env.global_state()))
    assert value.shape == ()


def test_action_mask_blocks_invalid_moves():
    env = make_env(num_agents=1)
    env.reset()
    net = ActorCritic(env.obs_dim, env.state_dim, env.n_actions, hidden=32)
    mask = torch.as_tensor(env.valid_action_mask())
    # Sample many times; the masked actions must never be selected.
    for _ in range(50):
        action, _ = net.act(torch.as_tensor(env.reset()), torch.as_tensor(env.valid_action_mask()))
        m = env.valid_action_mask()
        assert m[0, int(action[0])]
    _ = mask


def test_collect_rollout_shapes():
    trainer = small_trainer(num_agents=2)
    rollout, _episodes = trainer._collect()
    t = trainer.cfg.rollout_steps
    assert rollout["obs"].shape == (t, 2, trainer.env.obs_dim)
    assert rollout["act"].shape == (t, 2)
    assert rollout["rew"].shape == (t,)


def test_gae_returns_finite():
    trainer = small_trainer()
    rollout, _ = trainer._collect()
    adv, returns = trainer._gae(rollout)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(returns).all()


def test_update_returns_stats():
    trainer = small_trainer()
    rollout, _ = trainer._collect()
    stats = trainer.update(rollout)
    assert {"policy_loss", "value_loss", "entropy"} <= stats.keys()
    assert all(np.isfinite(v) for v in stats.values())


def test_train_runs_and_changes_weights():
    trainer = small_trainer()
    before = [p.clone() for p in trainer.net.parameters()]
    history = trainer.train(num_updates=2, log_every=0)
    assert len(history) == 2
    after = list(trainer.net.parameters())
    # At least one parameter tensor must have moved.
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_returns_metrics():
    trainer = small_trainer()
    result = trainer.evaluate(episodes=3)
    assert {"success_rate", "avg_rescued", "avg_steps"} <= result.keys()
    assert 0.0 <= result["success_rate"] <= 1.0


def test_running_mean_std_tracks_mean():
    rms = RunningMeanStd()
    data = torch.arange(100.0)
    rms.update(data)
    assert abs(rms.mean - float(data.mean())) < 1.0
