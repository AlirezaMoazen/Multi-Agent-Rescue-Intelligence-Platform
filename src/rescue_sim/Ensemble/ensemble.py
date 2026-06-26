"""Value ensemble of a trained QMIX + TransfQMix (CTDE, value-based).

Both methods output *comparable* per-agent Q-values, so we combine them at
decision time: average their Q-values (weighted by how good each is on its own)
and take the best valid action.  This is the only principled "use two methods
together" combination here --

* the tabular methods only know one grid (useless on fresh test grids), and
* MAPPO outputs action probabilities, not Q-values, so it does not mix cleanly.

The ensemble needs no retraining.  A single ``EntityRescueEnv`` feeds both: QMIX
reads the flat observation, TransfQMix reads the entity tokens of the same step.
"""

from __future__ import annotations

import numpy as np
import torch

from rescue_sim.QMIX.qmix import QMIX
from rescue_sim.TransfQMix.transf_qmix import EntityRescueEnv, TransfQMIX


def performance_weights(qmix_success: float, transf_success: float) -> tuple[float, float]:
    """Weights proportional to each method's success (stronger method dominates).

    Avoids the weaker model dragging down the stronger one; falls back to equal
    weights if both scored zero.
    """
    total = qmix_success + transf_success
    if total <= 0:
        return 0.5, 0.5
    return qmix_success / total, transf_success / total


class ValueEnsemble:
    """Weighted average of a trained QMIX and TransfQMix per-agent Q-values."""

    def __init__(
        self,
        qmix: QMIX,
        transf: TransfQMIX,
        env: EntityRescueEnv,
        w_qmix: float = 0.5,
        w_transf: float = 0.5,
    ) -> None:
        self.qmix = qmix
        self.transf = transf
        self.env = env
        total = w_qmix + w_transf
        self.w_qmix = w_qmix / total      # normalize so the weights sum to 1
        self.w_transf = w_transf / total

    @torch.no_grad()
    def combined_q(self, flat_obs: np.ndarray, tokens: np.ndarray) -> torch.Tensor:
        """Weighted per-agent Q-values from both methods: shape (num_agents, A)."""
        q_qmix = self.qmix.agent(torch.as_tensor(flat_obs).float())
        q_transf, _ = self.transf.agent(torch.as_tensor(tokens).float())
        return self.w_qmix * q_qmix + self.w_transf * q_transf

    @torch.no_grad()
    def select_actions(self, flat_obs: np.ndarray, tokens: np.ndarray, avail: np.ndarray):
        """Greedy action per agent from the weighted-average Q-values."""
        q = self.combined_q(flat_obs, tokens)
        q = q.masked_fill(~torch.as_tensor(avail), -float("inf"))   # forbid walls/edges
        return q.argmax(dim=-1).numpy()

    @torch.no_grad()
    def evaluate(self, episodes: int = 20) -> dict:
        """Greedy roll-outs on fresh grids; same metrics as the single methods."""
        successes, rescued, steps = [], [], []
        for _ in range(episodes):
            flat_obs = self.env.reset()
            done = False
            info: dict = {}
            while not done:
                tokens = self.env.entity_obs()
                avail = self.env.valid_action_mask()
                actions = self.select_actions(flat_obs, tokens, avail)
                flat_obs, _, done, info = self.env.step(actions)
            successes.append(info["success"])
            rescued.append(info["rescued"])
            steps.append(info["steps"])
        return {
            "success_rate": float(np.mean(successes)),
            "avg_rescued": float(np.mean(rescued)),
            "avg_steps": float(np.mean(steps)),
        }
