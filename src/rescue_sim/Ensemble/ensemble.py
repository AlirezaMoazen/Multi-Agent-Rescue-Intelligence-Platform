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
import torch.nn.functional as F

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


class PolicyEnsemble:
    """Probability-space blend of trained QMIX + TransfQMix + MAPPO policies.

    Unlike ``ValueEnsemble`` (which averages Q-values of the two value-based
    methods), this brings MAPPO in by working in *probability* space:

    * QMIX / TransfQMix Q-values -> Boltzmann action distributions
      ``softmax(Q / tau)``,
    * MAPPO actor logits -> ``softmax(logits)`` directly,
    * a linear blend with weights ``w`` (equal thirds by default, or
      performance-weighted) gives one action distribution per agent.

    Only the per-agent policy networks are used (``qmix.agent``,
    ``transf.agent``, ``mappo.net.actor``), so the blend is agent-count
    agnostic; it just needs observations of the shape the checkpoints were
    trained on (same ``view_radius`` / ``num_agents``).
    """

    def __init__(
        self,
        qmix: QMIX,
        transf: TransfQMIX,
        mappo,
        weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        temperature: float = 1.0,
    ) -> None:
        self.qmix = qmix
        self.transf = transf
        self.mappo = mappo
        w = np.asarray(weights, dtype=np.float64)
        self.weights = tuple((w / w.sum()).tolist())   # normalized, sum to 1
        self.tau = temperature

    @classmethod
    def from_checkpoints(
        cls,
        env: EntityRescueEnv,
        qmix_path: str = "checkpoints/qmix.pt",
        transf_path: str = "checkpoints/transfqmix.pt",
        mappo_path: str = "checkpoints/mappo.pt",
        weights: tuple[float, float, float] | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> "PolicyEnsemble":
        """Loads the three checkpoints, sizing each net to ``env``'s dimensions.

        ``env`` must be an ``EntityRescueEnv`` (feeds flat obs to QMIX/MAPPO and
        entity tokens to TransfQMix). Its ``view_radius`` / ``num_agents`` must
        match what the checkpoints were trained on, or ``load_state_dict`` will
        raise a size-mismatch error.
        """
        from rescue_sim.config.settings import (
            MappoSettings,
            QmixSettings,
            TransfQmixSettings,
        )
        from rescue_sim.MAPPO import MAPPO

        common = dict(num_agents=env.num_agents, view_radius=env.view_radius,
                      max_steps=env.max_steps, random_seed=0)
        qmix = QMIX(env, QmixSettings(**common), device=device)
        qmix.load_checkpoint(qmix_path)
        transf = TransfQMIX(env, TransfQmixSettings(**common), device=device)
        transf.load_checkpoint(transf_path)
        mappo = MAPPO(env, MappoSettings(**common), device=device)
        mappo.load_checkpoint(mappo_path)
        return cls(qmix, transf, mappo, weights or (1 / 3, 1 / 3, 1 / 3), temperature)

    @torch.no_grad()
    def action_probs(self, flat_obs: np.ndarray, tokens: np.ndarray) -> torch.Tensor:
        """Blended per-agent action distribution: shape (num_agents, n_actions)."""
        fo = torch.as_tensor(flat_obs).float()
        tk = torch.as_tensor(tokens).float()
        q_qmix = self.qmix.agent(fo.to(self.qmix.device)).cpu()
        q_transf, _ = self.transf.agent(tk.to(self.transf.device))
        logits_mappo = self.mappo.net.actor(fo.to(self.mappo.device)).cpu()

        p_qmix = F.softmax(q_qmix / self.tau, dim=-1)
        p_transf = F.softmax(q_transf.cpu() / self.tau, dim=-1)
        p_mappo = F.softmax(logits_mappo, dim=-1)

        w_q, w_t, w_m = self.weights
        return w_q * p_qmix + w_t * p_transf + w_m * p_mappo

    @torch.no_grad()
    def select_actions(self, flat_obs: np.ndarray, tokens: np.ndarray, avail: np.ndarray):
        """Greedy action per agent from the blended distribution (walls masked out)."""
        probs = self.action_probs(flat_obs, tokens)
        probs = probs.masked_fill(~torch.as_tensor(avail), 0.0)
        return probs.argmax(dim=-1).numpy()

    @torch.no_grad()
    def evaluate(self, episodes: int = 20) -> dict:
        """Greedy roll-outs on fresh grids using the blended policy."""
        env = self.transf.env  # an EntityRescueEnv
        successes, rescued, steps = [], [], []
        for _ in range(episodes):
            flat_obs = env.reset()
            done = False
            info: dict = {}
            while not done:
                tokens = env.entity_obs()
                avail = env.valid_action_mask()
                actions = self.select_actions(flat_obs, tokens, avail)
                flat_obs, _, done, info = env.step(actions)
            successes.append(info["success"])
            rescued.append(info["rescued"])
            steps.append(info["steps"])
        return {
            "success_rate": float(np.mean(successes)),
            "avg_rescued": float(np.mean(rescued)),
            "avg_steps": float(np.mean(steps)),
        }
