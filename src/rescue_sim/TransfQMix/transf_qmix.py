"""TransfQMix -- transformers for cooperative MARL value factorization (CTDE).

Implements the core idea of Gallici et al. 2023, "TransfQMix: Transformers for
Leveraging the Graph Structure of MARL Problems" (AAMAS), kept as small as
possible:

* observations are a *set of entity tokens* (one per visible cell + a self
  token) instead of a flat vector;
* the agent network is a **transformer** over those tokens -> Q-values;
* the mixing network is also a **transformer** over the agents' hidden states
  (+ a global-state token) that produces *non-negative* (monotonic) mixing
  weights -> Q_tot.

Because both networks are transformers over a variable token set, the same
parameters transfer to any number of agents/entities -- TransfQMix's headline
property.

This is QMIX with transformer networks: it reuses `RescueEnv` (via an entity
subclass) and QMIX's `ReplayBuffer`, and trains off-policy with a Double-DQN
target.  Runs on CPU (`pip install -e ".[transfqmix]"`).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from random import Random

import numpy as np
import torch
from torch import nn

from rescue_sim.config.settings import TransfQmixSettings
from rescue_sim.MAPPO.environment import RescueEnv
from rescue_sim.shared import ReplayBuffer, hard_update, resolve_device

# Entity-token features: [blocked, target-A, target-B, other-agent,
#                         rel_x, rel_y, is_self, step_frac, remaining_frac]
TOKEN_DIM = 9


class EntityRescueEnv(RescueEnv):
    """RescueEnv that also emits entity-token observations for TransfQMix."""

    @property
    def n_tokens(self) -> int:
        win = 2 * self.view_radius + 1
        return win * win + 1  # window cells + a self token

    @property
    def token_dim(self) -> int:
        return TOKEN_DIM

    def entity_obs(self) -> np.ndarray:
        """Per-agent token sets: shape (num_agents, n_tokens, TOKEN_DIM)."""
        self._refresh_agent_count()
        return np.stack([self._agent_tokens(i) for i in range(self.num_agents)])

    def _agent_tokens(self, index: int) -> np.ndarray:
        # Reuse the parent's vectorized view channels; add relative offsets so
        # each visible cell becomes one entity token, then a final self token.
        blocked, target_a, target_b, other = self._view_channels(index)
        zeros = np.zeros_like(blocked)
        window = np.stack(
            [blocked, target_a, target_b, other, self._rel_x, self._rel_y, zeros, zeros, zeros],
            axis=-1,
        ).reshape(self._win * self._win, TOKEN_DIM)
        sx, sy, step_frac, remaining = self._scalars(index)
        self_token = np.array(
            [[0.0, 0.0, 0.0, 0.0, sx, sy, 1.0, step_frac, remaining]], dtype=np.float32
        )
        return np.concatenate([window, self_token], axis=0).astype(np.float32)


class AgentTransformer(nn.Module):
    """Transformer over entity tokens -> Q-values + a hidden embedding (CLS)."""

    def __init__(self, token_dim: int, n_actions: int, cfg: TransfQmixSettings) -> None:
        super().__init__()
        self.embed = nn.Linear(token_dim, cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.ff_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_agent_layers)
        self.q_head = nn.Linear(cfg.d_model, n_actions)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(tokens)                              # (B, T, d_model)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, x], dim=1))        # (B, T+1, d_model)
        cls_out = h[:, 0]                                   # (B, d_model)
        return self.q_head(cls_out), cls_out


class TransformerMixer(nn.Module):
    """Transformer over agent hiddens + a state token -> monotonic Q_tot."""

    def __init__(self, state_dim: int, cfg: TransfQmixSettings) -> None:
        super().__init__()
        self.embed_dim = cfg.mixing_embed_dim
        self.state_proj = nn.Linear(state_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.ff_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_mixer_layers)
        self.w1_head = nn.Linear(cfg.d_model, self.embed_dim)   # per-agent weight (>=0)
        self.w2_head = nn.Linear(cfg.d_model, self.embed_dim)   # state weight (>=0)
        self.b1_head = nn.Linear(cfg.d_model, self.embed_dim)
        self.v_head = nn.Sequential(
            nn.Linear(cfg.d_model, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, agent_h: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        state_token = self.state_proj(state).unsqueeze(1)       # (B, 1, d_model)
        h = self.encoder(torch.cat([state_token, agent_h], dim=1))
        state_h, agents_h = h[:, 0], h[:, 1:]                   # (B, d_model), (B, N, d_model)
        # abs() -> non-negative weights -> Q_tot monotonic in each agent's Q.
        w1 = torch.abs(self.w1_head(agents_h))                  # (B, N, embed)
        b1 = self.b1_head(state_h)                             # (B, embed)
        hidden = torch.nn.functional.elu(torch.einsum("bn,bne->be", agent_qs, w1) + b1)
        w2 = torch.abs(self.w2_head(state_h))                  # (B, embed)
        v = self.v_head(state_h).squeeze(-1)                  # (B,)
        return (hidden * w2).sum(dim=1) + v


class TransfQMIX:
    """Trains transformer agent + transformer mixer on an EntityRescueEnv."""

    def __init__(
        self,
        env: EntityRescueEnv,
        settings: TransfQmixSettings = TransfQmixSettings(),
        device: str | None = None,
    ):
        self.env = env
        self.cfg = settings
        self.rng = Random(settings.random_seed)
        if settings.random_seed is not None:
            torch.manual_seed(settings.random_seed)

        self.device = resolve_device(device)
        self.agent = AgentTransformer(env.token_dim, env.n_actions, settings).to(self.device)
        self.mixer = TransformerMixer(env.state_dim, settings).to(self.device)
        self.target_agent = AgentTransformer(env.token_dim, env.n_actions, settings).to(self.device)
        self.target_mixer = TransformerMixer(env.state_dim, settings).to(self.device)
        self._sync_targets()

        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=settings.learning_rate)
        self.buffer = ReplayBuffer(settings.buffer_size, self.rng)
        self.epsilon = settings.epsilon_start
        self.learn_steps = 0

    # -- action selection ---------------------------------------------------

    @torch.no_grad()
    def select_actions(self, tokens: np.ndarray, avail: np.ndarray, greedy: bool = False):
        q, _ = self.agent(torch.as_tensor(tokens).float().to(self.device))
        q = q.masked_fill(~torch.as_tensor(avail).to(self.device), -1e9)
        greedy_actions = q.argmax(dim=-1).cpu().numpy()
        if greedy:
            return greedy_actions
        actions = greedy_actions.copy()
        for i in range(self.env.num_agents):
            if self.rng.random() < self.epsilon:
                actions[i] = self.rng.choice(np.flatnonzero(avail[i]).tolist())
        return actions

    # -- learning -----------------------------------------------------------

    def _sync_targets(self) -> None:
        hard_update(self.target_agent, self.agent)
        hard_update(self.target_mixer, self.mixer)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights and training state for later visualization/API use."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "settings": asdict(self.cfg),
                "agent": self.agent.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_agent": self.target_agent.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load weights into an already-created TransfQMIX trainer."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_agent.load_state_dict(checkpoint.get("target_agent", checkpoint["agent"]))
        self.target_mixer.load_state_dict(checkpoint.get("target_mixer", checkpoint["mixer"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.learn_steps = int(checkpoint.get("learn_steps", self.learn_steps))
        for net in (self.agent, self.mixer, self.target_agent, self.target_mixer):
            net.to(self.device)

    def _agent_forward(self, net: AgentTransformer, tokens: torch.Tensor):
        """Apply a per-agent transformer to a (B, N, T, D) batch."""
        batch, n_agents = tokens.shape[0], tokens.shape[1]
        flat = tokens.reshape(batch * n_agents, tokens.shape[2], tokens.shape[3])
        q, h = net(flat)
        return q.view(batch, n_agents, -1), h.view(batch, n_agents, -1)

    def _learn(self) -> float:
        cfg = self.cfg
        batch = self.buffer.sample(cfg.batch_size)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        q, h = self._agent_forward(self.agent, batch["obs"])           # (B,N,A), (B,N,d)
        chosen = q.gather(2, batch["actions"].unsqueeze(2)).squeeze(2)  # (B,N)
        q_tot = self.mixer(chosen, h, batch["state"])

        with torch.no_grad():
            tgt_q, tgt_h = self._agent_forward(self.target_agent, batch["next_obs"])
            tgt_q = tgt_q.masked_fill(~batch["next_avail"], -1e9)
            if cfg.double_q:
                online_q, _ = self._agent_forward(self.agent, batch["next_obs"])
                online_q = online_q.masked_fill(~batch["next_avail"], -1e9)
                next_actions = online_q.argmax(dim=2, keepdim=True)
                next_q = tgt_q.gather(2, next_actions).squeeze(2)
            else:
                next_q = tgt_q.max(dim=2).values
            q_tot_next = self.target_mixer(next_q, tgt_h, batch["next_state"])
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
        self.env.reset()
        tokens = self.env.entity_obs()
        avail = self.env.valid_action_mask()
        state = self.env.global_state()
        loss_sum, n_learn = 0.0, 0
        done = False
        info: dict = {}
        while not done:
            actions = self.select_actions(tokens, avail, greedy=not learn)
            _, reward, done, info = self.env.step(actions)
            next_tokens = self.env.entity_obs()
            next_avail = self.env.valid_action_mask()
            next_state = self.env.global_state()
            if learn:
                self.buffer.push({
                    "obs": tokens, "state": state, "actions": actions, "avail": avail,
                    "reward": np.float32(reward), "next_obs": next_tokens,
                    "next_state": next_state, "next_avail": next_avail,
                    "done": np.float32(done),
                })
                if len(self.buffer) >= self.cfg.batch_size:
                    loss_sum += self._learn()
                    n_learn += 1
            tokens, avail, state = next_tokens, next_avail, next_state
        info["loss"] = loss_sum / n_learn if n_learn else 0.0
        return info

    def train(
        self,
        num_episodes: int,
        log_every: int = 10,
        eval_hook=None,
        hook_every: int = 0,
    ) -> list[dict]:
        """Run `num_episodes` of collect+learn; returns per-episode metrics.

        ``eval_hook`` (called every ``hook_every`` episodes) may return True to
        stop training early (e.g. on a wall-clock budget).
        """
        cfg = self.cfg
        history: list[dict] = []
        for episode in range(1, num_episodes + 1):
            info = self._run_episode(learn=True)
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
            if eval_hook and hook_every and episode % hook_every == 0:
                if eval_hook(episode):
                    break
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
