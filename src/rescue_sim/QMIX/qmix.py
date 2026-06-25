"""QMIX -- monotonic value-function factorization for cooperative MARL (CTDE).

Implements Rashid et al. 2018, "QMIX: Monotonic Value Function Factorisation
for Deep Multi-Agent RL", kept as small as possible:

* each agent has a shared Q-network (parameter sharing) over its local obs;
* a *mixing network* combines the per-agent Q-values into a team value Q_tot,
  using weights produced by a hypernetwork from the global state;
* the mixing weights are kept non-negative (monotonicity), which guarantees
  argmax over Q_tot equals the per-agent argmaxes -> decentralized execution;
* trained off-policy from a replay buffer with a Double-DQN target.

This is a feed-forward QMIX (no RNN) with a per-transition replay buffer -- the
smallest version that still trains well on a fully observable grid.  It reuses
`RescueEnv` from the MAPPO package so results are directly comparable.

Runs on CPU; `torch` is the only extra dependency (`pip install -e ".[qmix]"`).
"""

from __future__ import annotations

from collections import deque
from random import Random

import numpy as np
import torch
from torch import nn

from rescue_sim.config.settings import QmixSettings
from rescue_sim.MAPPO.environment import RescueEnv


class AgentQNet(nn.Module):
    """Shared per-agent Q-network: local observation -> Q-value per action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MixingNetwork(nn.Module):
    """Monotonic mixer: per-agent Qs + global state -> Q_tot (non-negative weights)."""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        # Hypernetworks produce the mixing weights/biases from the global state.
        self.hyper_w1 = nn.Linear(state_dim, num_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = agent_qs.size(0)
        agent_qs = agent_qs.view(batch, 1, self.num_agents)
        # abs() keeps weights >= 0 -> Q_tot is monotonic in each agent's Q.
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_qs, w1) + b1)  # (B, 1, embed)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)                  # V(state)
        q_tot = torch.bmm(hidden, w2) + b2                          # (B, 1, 1)
        return q_tot.view(batch)


class ReplayBuffer:
    """Fixed-size buffer of single-step team transitions."""

    def __init__(self, capacity: int, rng: Random) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = rng

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: dict) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict:
        batch = self.rng.sample(self.buffer, batch_size)
        stack = lambda key: torch.as_tensor(np.stack([t[key] for t in batch]))  # noqa: E731
        return {
            "obs": stack("obs").float(),
            "state": stack("state").float(),
            "actions": stack("actions").long(),
            "avail": stack("avail").bool(),
            "reward": stack("reward").float(),
            "next_obs": stack("next_obs").float(),
            "next_state": stack("next_state").float(),
            "next_avail": stack("next_avail").bool(),
            "done": stack("done").float(),
        }


class QMIX:
    """Trains a shared agent Q-network and a monotonic mixer on a RescueEnv."""

    def __init__(self, env: RescueEnv, settings: QmixSettings = QmixSettings()) -> None:
        self.env = env
        self.cfg = settings
        self.rng = Random(settings.random_seed)
        if settings.random_seed is not None:
            torch.manual_seed(settings.random_seed)

        self.agent = AgentQNet(env.obs_dim, env.n_actions, settings.hidden_dim)
        self.mixer = MixingNetwork(env.num_agents, env.state_dim, settings.mixing_embed_dim)
        self.target_agent = AgentQNet(env.obs_dim, env.n_actions, settings.hidden_dim)
        self.target_mixer = MixingNetwork(env.num_agents, env.state_dim, settings.mixing_embed_dim)
        self._sync_targets()

        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=settings.learning_rate)
        self.buffer = ReplayBuffer(settings.buffer_size, self.rng)
        self.epsilon = settings.epsilon_start
        self.learn_steps = 0

    # -- action selection ---------------------------------------------------

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray, avail: np.ndarray, greedy: bool = False) -> np.ndarray:
        q = self.agent(torch.as_tensor(obs).float())
        q = q.masked_fill(~torch.as_tensor(avail), -float("inf"))
        greedy_actions = q.argmax(dim=-1).numpy()
        if greedy:
            return greedy_actions
        actions = greedy_actions.copy()
        for i in range(self.env.num_agents):
            if self.rng.random() < self.epsilon:
                valid = np.flatnonzero(avail[i])
                actions[i] = self.rng.choice(valid.tolist())
        return actions

    # -- learning -----------------------------------------------------------

    def _sync_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _learn(self) -> float:
        cfg = self.cfg
        batch = self.buffer.sample(cfg.batch_size)

        # Current Q_tot for the actions actually taken.
        q = self.agent(batch["obs"])                                    # (B, n, A)
        chosen = q.gather(2, batch["actions"].unsqueeze(2)).squeeze(2)  # (B, n)
        q_tot = self.mixer(chosen, batch["state"])                      # (B,)

        with torch.no_grad():
            target_q = self.target_agent(batch["next_obs"])
            target_q = target_q.masked_fill(~batch["next_avail"], -float("inf"))
            if cfg.double_q:  # pick next actions with the online net, value with target
                online_next = self.agent(batch["next_obs"])
                online_next = online_next.masked_fill(~batch["next_avail"], -float("inf"))
                next_actions = online_next.argmax(dim=2, keepdim=True)
                next_q = target_q.gather(2, next_actions).squeeze(2)
            else:
                next_q = target_q.max(dim=2).values
            q_tot_next = self.target_mixer(next_q, batch["next_state"])
            y = batch["reward"] + cfg.gamma * (1 - batch["done"]) * q_tot_next

        loss = ((q_tot - y) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % cfg.target_update_interval == 0:
            self._sync_targets()
        return float(loss.detach())

    # -- training -----------------------------------------------------------

    def _run_episode(self, learn: bool = True) -> dict:
        obs = self.env.reset()
        avail = self.env.valid_action_mask()
        state = self.env.global_state()
        loss_sum, n_learn = 0.0, 0
        done = False
        info: dict = {}
        while not done:
            actions = self.select_actions(obs, avail, greedy=not learn)
            next_obs, reward, done, info = self.env.step(actions)
            next_avail = self.env.valid_action_mask()
            next_state = self.env.global_state()
            if learn:
                self.buffer.push({
                    "obs": obs, "state": state, "actions": actions, "avail": avail,
                    "reward": np.float32(reward), "next_obs": next_obs,
                    "next_state": next_state, "next_avail": next_avail,
                    "done": np.float32(done),
                })
                if len(self.buffer) >= self.cfg.batch_size:
                    loss_sum += self._learn()
                    n_learn += 1
            obs, avail, state = next_obs, next_avail, next_state
        info["loss"] = loss_sum / n_learn if n_learn else 0.0
        return info

    def train(self, num_episodes: int, log_every: int = 10) -> list[dict]:
        """Run `num_episodes` of collect+learn; returns per-episode metrics."""
        cfg = self.cfg
        history: list[dict] = []
        for episode in range(1, num_episodes + 1):
            info = self._run_episode(learn=True)
            # Linear epsilon decay.
            frac = min(1.0, episode / max(1, cfg.epsilon_anneal_episodes))
            self.epsilon = cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)
            record = {
                "episode": episode,
                "success": bool(info["success"]),
                "rescued": info["rescued"],
                "steps": info["steps"],
                "loss": info["loss"],
                "epsilon": self.epsilon,
            }
            history.append(record)
            if log_every and episode % log_every == 0:
                print(
                    f"ep {episode:>4} | success {int(info['success'])} "
                    f"| rescued {info['rescued']}/{info['targets']} "
                    f"| steps {info['steps']:>3} | loss {info['loss']:.3f} "
                    f"| eps {self.epsilon:.2f}"
                )
        return history

    @torch.no_grad()
    def evaluate(self, episodes: int = 10) -> dict:
        """Greedy roll-outs for reporting (no exploration, no learning)."""
        successes, rescued, steps = [], [], []
        for _ in range(episodes):
            info = self._run_episode(learn=False)
            successes.append(info["success"])
            rescued.append(info["rescued"])
            steps.append(info["steps"])
        return {
            "success_rate": float(np.mean(successes)),
            "avg_rescued": float(np.mean(rescued)),
            "avg_steps": float(np.mean(steps)),
        }
