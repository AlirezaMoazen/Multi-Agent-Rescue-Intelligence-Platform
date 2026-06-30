"""Distill the QMIX+TransfQMix ensemble into a single student network.

The ensemble is great but runs *two* networks at test time.  Distillation trains
one small student (an MLP over the local observation) to copy the ensemble's
per-agent Q-values -- giving ensemble-level behaviour at single-network cost.

This is supervised learning, not RL:

1. roll the ensemble (the *teacher*) through the env and record, at each step,
   the local observations and the teacher's Q-values;
2. train the student by regression (MSE) to predict those Q-values;
3. the student then acts greedily from its own Q-values -- one network, local
   observation only.

The student reuses ``AgentQNet`` from QMIX, so there is no new network code.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rescue_sim.config.settings import DistillSettings
from rescue_sim.Ensemble.ensemble import ValueEnsemble
from rescue_sim.QMIX.qmix import AgentQNet
from rescue_sim.TransfQMix.transf_qmix import EntityRescueEnv


class Distiller:
    """Trains a single student network to imitate a ValueEnsemble teacher."""

    def __init__(
        self,
        ensemble: ValueEnsemble,
        env: EntityRescueEnv,
        settings: DistillSettings = DistillSettings(),
    ) -> None:
        self.ensemble = ensemble
        self.env = env
        self.cfg = settings
        if settings.random_seed is not None:
            torch.manual_seed(settings.random_seed)
        self.student = AgentQNet(env.obs_dim, env.n_actions, settings.hidden_dim)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=settings.learning_rate)

    @torch.no_grad()
    def collect(self, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather (local observation, teacher Q-values) pairs by rolling the teacher."""
        obs_rows, q_rows = [], []
        flat_obs = self.env.reset()
        for _ in range(steps):
            tokens = self.env.entity_obs()
            avail = self.env.valid_action_mask()
            teacher_q = self.ensemble.combined_q(flat_obs, tokens)          # (n, A)
            obs_rows.append(np.asarray(flat_obs, dtype=np.float32))         # (n, obs)
            q_rows.append(teacher_q.numpy())
            masked = teacher_q.masked_fill(~torch.as_tensor(avail), -float("inf"))
            actions = masked.argmax(dim=-1).numpy()                         # follow the teacher
            flat_obs, _, done, _ = self.env.step(actions)
            if done:
                flat_obs = self.env.reset()
        x = torch.as_tensor(np.concatenate(obs_rows))   # (M, obs_dim)
        y = torch.as_tensor(np.concatenate(q_rows))     # (M, A)
        return x, y

    def train(self) -> list[float]:
        """Supervised regression of the student onto the teacher's Q-values."""
        x, y = self.collect(self.cfg.collect_steps)
        losses: list[float] = []
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(x.size(0))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, x.size(0), self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                pred = self.student(x[idx])
                loss = nn.functional.mse_loss(pred, y[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1
            losses.append(epoch_loss / max(1, batches))
        return losses

    @torch.no_grad()
    def evaluate(self, episodes: int = 20) -> dict:
        """Greedy roll-outs using only the student (one network, local obs)."""
        successes, rescued, steps = [], [], []
        for _ in range(episodes):
            flat_obs = self.env.reset()
            done = False
            info: dict = {}
            while not done:
                avail = self.env.valid_action_mask()
                q = self.student(torch.as_tensor(flat_obs).float())
                q = q.masked_fill(~torch.as_tensor(avail), -float("inf"))
                actions = q.argmax(dim=-1).numpy()
                flat_obs, _, done, info = self.env.step(actions)
            successes.append(info["success"])
            rescued.append(info["rescued"])
            steps.append(info["steps"])
        return {
            "success_rate": float(np.mean(successes)),
            "avg_rescued": float(np.mean(rescued)),
            "avg_steps": float(np.mean(steps)),
        }
