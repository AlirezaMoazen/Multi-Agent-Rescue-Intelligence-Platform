"""MAPPO -- Multi-Agent PPO with a centralized critic (CTDE).

Implements the recipe from Yu et al. 2022, "The Surprising Effectiveness of PPO
in Cooperative Multi-Agent Games", kept as small as possible:

* parameter sharing  -- one actor-critic network is shared by every agent;
* centralized critic -- the value function sees the global state (all agents),
  while each actor sees only its own local observation;
* GAE, clipped policy + value losses, advantage normalization, value-target
  normalization, an entropy bonus, and gradient clipping.

Runs on CPU; `torch` is the only extra dependency (install with `pip install
-e ".[mappo]"`).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rescue_sim.config.settings import MappoSettings
from rescue_sim.shared import RunningMeanStd, orthogonal_init
from rescue_sim.MAPPO.environment import RescueEnv


class ActorCritic(nn.Module):
    """Shared policy (local obs -> action) + centralized value (global state)."""

    def __init__(self, obs_dim: int, state_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            orthogonal_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            orthogonal_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            orthogonal_init(nn.Linear(hidden, n_actions), gain=0.01),
        )
        self.critic = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden)), nn.Tanh(),
            orthogonal_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            orthogonal_init(nn.Linear(hidden, 1), gain=1.0),
        )

    def _dist(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = self.actor(obs)
        logits = logits.masked_fill(~mask, -1e8)  # forbid walls/edges
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs, mask)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(
        self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs, mask)
        return dist.log_prob(actions), dist.entropy()

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


class MAPPO:
    """Collects rollouts from a RescueEnv and trains a shared actor-critic."""

    def __init__(self, env: RescueEnv, settings: MappoSettings = MappoSettings()) -> None:
        self.env = env
        self.cfg = settings
        if settings.random_seed is not None:
            torch.manual_seed(settings.random_seed)
        self.net = ActorCritic(env.obs_dim, env.state_dim, env.n_actions, settings.hidden_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=settings.learning_rate)
        self.value_norm = RunningMeanStd() if settings.normalize_value else None

    # -- rollout ------------------------------------------------------------

    def _collect(self) -> tuple[dict, list[dict]]:
        cfg = self.cfg
        obs = self.env.reset()
        buf_obs, buf_mask, buf_act, buf_logp = [], [], [], []
        buf_state, buf_val, buf_rew, buf_done = [], [], [], []
        episodes: list[dict] = []

        for _ in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(obs)
            mask_t = torch.as_tensor(self.env.valid_action_mask())
            state_t = torch.as_tensor(self.env.global_state())
            action, logp = self.net.act(obs_t, mask_t)
            # Rollout values are constants for GAE / value-clipping -> detach.
            value = self.net.value(state_t).detach()

            next_obs, reward, done, info = self.env.step(action.numpy())

            buf_obs.append(obs_t)
            buf_mask.append(mask_t)
            buf_act.append(action)
            buf_logp.append(logp)
            buf_state.append(state_t)
            buf_val.append(value)
            buf_rew.append(reward)
            buf_done.append(done)

            obs = next_obs
            if done:
                episodes.append(info)
                obs = self.env.reset()

        # Bootstrap value of the final state for GAE.
        last_value = self.net.value(torch.as_tensor(self.env.global_state())).detach()
        rollout = {
            "obs": torch.stack(buf_obs),                       # (T, n, obs)
            "mask": torch.stack(buf_mask),                     # (T, n, A)
            "act": torch.stack(buf_act),                       # (T, n)
            "logp": torch.stack(buf_logp),                     # (T, n)
            "state": torch.stack(buf_state),                   # (T, state)
            "val": torch.stack(buf_val),                       # (T,)
            "rew": torch.as_tensor(buf_rew, dtype=torch.float32),   # (T,)
            "done": torch.as_tensor(buf_done, dtype=torch.float32), # (T,)
            "last_value": last_value,
        }
        return rollout, episodes

    def _gae(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        rew, val, done = rollout["rew"], rollout["val"], rollout["done"]
        values = val
        if self.value_norm is not None:  # critic predicts normalized values
            values = values * self.value_norm.std + self.value_norm.mean
        advantages = torch.zeros_like(rew)
        last_gae = 0.0
        next_value = rollout["last_value"]
        for t in reversed(range(len(rew))):
            non_terminal = 1.0 - done[t]
            delta = rew[t] + cfg.gamma * next_value * non_terminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    # -- training -----------------------------------------------------------

    def update(self, rollout: dict) -> dict:
        cfg = self.cfg
        advantages, returns = self._gae(rollout)
        if self.value_norm is not None:
            self.value_norm.update(returns)
            norm_returns = (returns - self.value_norm.mean) / self.value_norm.std
        else:
            norm_returns = returns

        # Per-(timestep, agent) tensors for the shared policy.
        obs = rollout["obs"].reshape(-1, self.env.obs_dim)
        mask = rollout["mask"].reshape(-1, self.env.n_actions)
        actions = rollout["act"].reshape(-1)
        old_logp = rollout["logp"].reshape(-1)
        n = self.env.num_agents
        adv_agents = advantages.unsqueeze(1).expand(-1, n).reshape(-1)
        adv_agents = (adv_agents - adv_agents.mean()) / (adv_agents.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(cfg.epochs):
            new_logp, entropy = self.net.evaluate(obs, mask, actions)
            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
            policy_loss = -torch.min(ratio * adv_agents, clipped * adv_agents).mean()

            value = self.net.value(rollout["state"])
            value_clipped = rollout["val"] + (value - rollout["val"]).clamp(-cfg.clip, cfg.clip)
            v_loss = torch.max(
                (value - norm_returns) ** 2, (value_clipped - norm_returns) ** 2
            ).mean()

            loss = policy_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            stats["policy_loss"] = float(policy_loss.detach())
            stats["value_loss"] = float(v_loss.detach())
            stats["entropy"] = float(entropy.mean().detach())
        return stats

    def train(self, num_updates: int, log_every: int = 1) -> list[dict]:
        """Run `num_updates` rollout+update cycles; returns per-update metrics."""
        history: list[dict] = []
        for update in range(1, num_updates + 1):
            rollout, episodes = self._collect()
            stats = self.update(rollout)
            success = np.mean([e["success"] for e in episodes]) if episodes else 0.0
            rescued = np.mean([e["rescued"] for e in episodes]) if episodes else 0.0
            record = {
                "update": update,
                "episodes": len(episodes),
                "success_rate": float(success),
                "avg_rescued": float(rescued),
                **stats,
            }
            history.append(record)
            if log_every and update % log_every == 0:
                print(
                    f"update {update:>4} | eps {record['episodes']:>3} "
                    f"| success {success:5.2f} | rescued {rescued:4.1f} "
                    f"| pi {stats['policy_loss']:+.3f} | v {stats['value_loss']:.3f}"
                )
        return history

    @torch.no_grad()
    def evaluate(self, episodes: int = 10) -> dict:
        """Greedy roll-outs for reporting (no exploration, no learning)."""
        successes, rescued, steps = [], [], []
        for _ in range(episodes):
            obs = self.env.reset()
            done = False
            while not done:
                mask_t = torch.as_tensor(self.env.valid_action_mask())
                logits = self.net.actor(torch.as_tensor(obs)).masked_fill(~mask_t, -1e8)
                action = logits.argmax(dim=-1)
                obs, _, done, info = self.env.step(action.numpy())
            successes.append(info["success"])
            rescued.append(info["rescued"])
            steps.append(info["steps"])
        return {
            "success_rate": float(np.mean(successes)),
            "avg_rescued": float(np.mean(rescued)),
            "avg_steps": float(np.mean(steps)),
        }
