# Copyright 2026 TUHH Group 05 — A. Herrero Callejo, C. Marcos Alonso,
# M. M. Orfany, A. Moazzen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH).
#
# http://www.apache.org/licenses/LICENSE-2.0

"""State-Conditioned Gated Teacher Selection for Expert 2 distillation.

Instead of collapsing the three trained teachers (MAPPO / QMIX / TransfQMix)
into one fixed-weight blend *before* training -- which mixes Q-values (absolute
expected return) with policy logits (relative preference) under a single shared
temperature, a category error that also throws away teacher diversity -- this
module keeps the teachers separate and lets a small learned router decide, per
state, how much to trust each one. The student (the MoE ``expert_coordination``
head) is then distilled with a per-teacher **reverse-KL** objective that is
mode-seeking, so it ignores a teacher whose distribution is noisy/uncertain.

Three phases (Gallici-style ensemble policy distillation, done right):

1. **Pseudo-oracle labels** -- for each collected sample compute each teacher's
   cross-entropy against the action actually taken; oracle weights are
   ``softmax(-[ce_mappo, ce_qmix, ce_transf])`` (the teacher that best predicts
   a good action gets the most weight *in that state*).
2. **Gating router** -- a 2-layer MLP maps interpretable state features
   (per-teacher policy entropy, nearby-agent count, target-visible flag,
   distance-to-wall) to teacher weights, trained (MSE) toward the oracle labels.
3. **Gated reverse-KL distillation** -- ``P_target = sum_k router(s)_k P_k`` and
   ``lambda_kd = 1/(1+beta*disagreement)`` (disagreement = mean pairwise KL
   between teachers) down-weights states where the teachers conflict. The
   student minimises ``lambda_kd * RKL(P_student || P_target)`` on
   no-visible-target states; visible-target states keep the greedy label.

Temperatures are calibrated **per teacher** (QMIX/TransfQMix matched to MAPPO's
entropy; ``tau_mappo = 1.0``) so no shared-temperature scale mismatch remains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rescue_sim.MoE.pipeline import EnsembleTeacher, build_peer_matrix

# Teacher order is fixed everywhere: loss vectors, oracle weights, router output.
TEACHER_ORDER = ("mappo", "qmix", "transf")
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Teacher bank: the three trained policies, per-teacher distributions (no blend)
# ---------------------------------------------------------------------------
class TeacherBank:
    """Holds the three trained teachers and exposes their *separate* per-agent
    action distributions on a shared ``EntityRescueEnv`` state.

    QMIX / TransfQMix Q-values are turned into Boltzmann distributions with
    their own calibrated temperature; MAPPO uses its actor logits directly
    (``tau = 1``).
    """

    def __init__(self, mappo, qmix, transf, tau_qmix: float, tau_transf: float) -> None:
        self.mappo = mappo
        self.qmix = qmix
        self.transf = transf
        self.tau = {"mappo": 1.0, "qmix": tau_qmix, "transf": tau_transf}

    @classmethod
    def from_checkpoints(
        cls,
        env,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        calib_states: int = 400,
    ) -> "TeacherBank":
        from rescue_sim.config.settings import (
            MappoSettings,
            QmixSettings,
            TransfQmixSettings,
        )
        from rescue_sim.MAPPO import MAPPO
        from rescue_sim.QMIX import QMIX
        from rescue_sim.TransfQMix import TransfQMIX

        d = Path(checkpoint_dir)
        common = dict(num_agents=env.num_agents, view_radius=env.view_radius,
                      max_steps=env.max_steps, random_seed=0)
        qmix = QMIX(env, QmixSettings(**common), device=device)
        qmix.load_checkpoint(str(d / "qmix.pt"))
        transf = TransfQMIX(env, TransfQmixSettings(**common), device=device)
        transf.load_checkpoint(str(d / "transfqmix.pt"))
        mappo = MAPPO(env, MappoSettings(**common), device=device)
        mappo.load_checkpoint(str(d / "mappo.pt"))

        bank = cls(mappo, qmix, transf, tau_qmix=1.0, tau_transf=1.0)
        bank._calibrate(env, calib_states)
        return bank

    # -- temperature calibration -------------------------------------------
    def _calibrate(self, env, n_states: int) -> None:
        """Match QMIX/TransfQMix softmax entropy to MAPPO's on sampled states."""
        flat, tokens = _sample_states(env, n_states)
        with torch.no_grad():
            logits_m = self.mappo.net.actor(flat)
            target_H = float(_entropy(F.softmax(logits_m, dim=-1)).mean())
            q_qmix = self.qmix.agent(flat)
            q_transf, _ = self.transf.agent(tokens)
        self.tau["qmix"] = _match_temperature(q_qmix, target_H)
        self.tau["transf"] = _match_temperature(q_transf, target_H)

    # -- distributions ------------------------------------------------------
    @torch.no_grad()
    def dists(self, flat_obs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Per-teacher distributions stacked in ``TEACHER_ORDER``: [3, A, n_act]."""
        p_mappo = F.softmax(self.mappo.net.actor(flat_obs), dim=-1)
        p_qmix = F.softmax(self.qmix.agent(flat_obs) / self.tau["qmix"], dim=-1)
        q_transf, _ = self.transf.agent(tokens)
        p_transf = F.softmax(q_transf / self.tau["transf"], dim=-1)
        return torch.stack([p_mappo, p_qmix, p_transf], dim=0)


def _sample_states(env, n: int):
    """Roll a random policy to gather a batch of (flat_obs, tokens) states."""
    flats, toks = [], []
    obs = env.reset()
    for _ in range(n):
        flats.append(np.asarray(obs, dtype=np.float32))
        toks.append(env.entity_obs().astype(np.float32))
        mask = env.valid_action_mask()
        actions = np.array([np.random.choice(np.flatnonzero(mask[i])) for i in range(env.num_agents)])
        obs, _, done, _ = env.step(actions)
        if done:
            obs = env.reset()
    return (torch.as_tensor(np.concatenate(flats)),
            torch.as_tensor(np.concatenate(toks)))


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p + _EPS)).sum(dim=-1)


def _match_temperature(q_values: torch.Tensor, target_H: float,
                       lo: float = 0.05, hi: float = 25.0, iters: int = 40) -> float:
    """Bisection for tau so mean entropy of softmax(Q/tau) ~= ``target_H``.

    Entropy is monotone increasing in tau, so bisection is exact.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        H = float(_entropy(F.softmax(q_values / mid, dim=-1)).mean())
        if H < target_H:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# State features for the gating router
# ---------------------------------------------------------------------------
def _state_features(flat_obs: np.ndarray, teacher_dists: torch.Tensor,
                    peer_row_sums: np.ndarray, view_radius: int) -> np.ndarray:
    """Per-agent features: [H_mappo, H_qmix, H_transf, nearby, target_vis, wall_dist].

    ``flat_obs``: [A, obs_dim]; ``teacher_dists``: [3, A, n_act]; ``peer_row_sums``:
    [A] (row sums of the peer matrix, i.e. 1 + number of linked peers).
    """
    win = 2 * view_radius + 1
    channels = 4
    A = flat_obs.shape[0]
    window = flat_obs[:, : win * win * channels].reshape(A, win, win, channels)

    ent = _entropy(teacher_dists).numpy().T            # [A, 3] in TEACHER_ORDER
    nearby = (peer_row_sums - 1.0).reshape(A, 1)       # linked peers (excl. self)
    target_vis = (window[:, :, :, 1:3].max(axis=(1, 2, 3)) > 0.5).astype(np.float32).reshape(A, 1)

    # Distance from the ego centre to the nearest visible wall (normalized).
    cy = cx = view_radius
    wall_dist = np.full((A, 1), 1.0, dtype=np.float32)
    for a in range(A):
        blocked = np.argwhere(window[a, :, :, 0] > 0.5)
        if len(blocked):
            d = np.abs(blocked[:, 0] - cy) + np.abs(blocked[:, 1] - cx)
            wall_dist[a, 0] = float(d.min()) / (2 * view_radius)
    return np.concatenate([ent, nearby, target_vis, wall_dist], axis=1).astype(np.float32)


FEATURE_DIM = 6


# ---------------------------------------------------------------------------
# Phase 1 data collection: samples with teacher dists, features, oracle labels
# ---------------------------------------------------------------------------
class GatedSamples:
    """Flat arrays over all (agent, step) samples collected for distillation."""

    def __init__(self) -> None:
        self.flat_obs: list[np.ndarray] = []      # [obs_dim]
        self.peer_count: list[float] = []         # scalar row-sum
        self.actions: list[int] = []              # action actually taken
        self.features: list[np.ndarray] = []      # [FEATURE_DIM]
        self.dists: list[np.ndarray] = []         # [3, n_act] teacher dists
        self.target_visible: list[bool] = []

    def stack(self):
        return (
            torch.as_tensor(np.stack(self.flat_obs)),
            torch.as_tensor(np.asarray(self.peer_count, dtype=np.float32)),
            torch.as_tensor(np.asarray(self.actions, dtype=np.int64)),
            torch.as_tensor(np.stack(self.features)),
            torch.as_tensor(np.stack(self.dists)),
            torch.as_tensor(np.asarray(self.target_visible, dtype=np.bool_)),
        )


def collect_gated_samples(env, bank: TeacherBank, ensemble, episodes: int,
                          steps: int, seed: int = 0) -> GatedSamples:
    """Rolls a neutral behavior policy (visible-greedy + equal-blend argmax) and
    records per-agent teacher distributions, state features and the action taken.
    """
    rng = np.random.default_rng(seed)
    behavior = EnsembleTeacher(rng, ensemble)   # visible-greedy + blend argmax
    out = GatedSamples()

    for _ in range(episodes):
        obs = env.reset()
        behavior.reset(env)
        for _ in range(steps):
            mask = env.valid_action_mask()
            flat = np.asarray(env._observations(), dtype=np.float32)
            tokens = env.entity_obs().astype(np.float32)
            peer = build_peer_matrix(env.positions)
            peer_rows = peer.sum(axis=1)

            dists = bank.dists(torch.as_tensor(flat), torch.as_tensor(tokens))  # [3,A,n]
            feats = _state_features(flat, dists, peer_rows, env.view_radius)     # [A,6]
            actions = behavior.act(env, mask)                                    # [A]

            for a in range(env.num_agents):
                out.flat_obs.append(flat[a])
                out.peer_count.append(float(peer_rows[a]))
                out.actions.append(int(actions[a]))
                out.features.append(feats[a])
                out.dists.append(dists[:, a, :].numpy())
                out.target_visible.append(behavior._visible_target(env, a) is not None)

            obs, _, done, _ = env.step(actions)
            behavior.observe(env)
            if done:
                break
    return out


def oracle_weights(dists: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Phase 1 labels: softmax over negative per-teacher CE on the taken action.

    ``dists``: [N, 3, n_act]; ``actions``: [N]. Returns [N, 3] in TEACHER_ORDER.
    """
    taken = dists.gather(2, actions.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)  # [N,3]
    ce = -torch.log(taken + _EPS)                                                 # [N,3]
    return F.softmax(-ce, dim=-1)


# ---------------------------------------------------------------------------
# Phase 2: gating router (features -> teacher weights)
# ---------------------------------------------------------------------------
class TeacherGatingRouter(nn.Module):
    """2-layer MLP: interpretable state features -> per-teacher weights (softmax)."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64, n_teachers: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_teachers),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(feats), dim=-1)


def train_gating_router(features: torch.Tensor, oracle: torch.Tensor,
                        epochs: int = 200, lr: float = 1e-3, seed: int = 0) -> TeacherGatingRouter:
    """Fits the router to the pseudo-oracle weights with an MSE objective."""
    torch.manual_seed(seed)
    router = TeacherGatingRouter()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    for _ in range(epochs):
        pred = router(features)
        loss = F.mse_loss(pred, oracle)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return router


# ---------------------------------------------------------------------------
# Phase 3: gated reverse-KL distillation into expert_coordination
# ---------------------------------------------------------------------------
def _pairwise_disagreement(dists: torch.Tensor) -> torch.Tensor:
    """Mean pairwise KL over ordered teacher pairs: [N, 3, n] -> [N]."""
    n_t = dists.shape[1]
    total = torch.zeros(dists.shape[0])
    count = 0
    logd = torch.log(dists + _EPS)
    for i in range(n_t):
        for j in range(n_t):
            if i == j:
                continue
            total = total + (dists[:, i] * (logd[:, i] - logd[:, j])).sum(dim=-1)
            count += 1
    return total / max(count, 1)


def gated_rkl_distill(policy, samples: GatedSamples, router: TeacherGatingRouter,
                      epochs: int = 12, batch_size: int = 128, lr: float = 1e-3,
                      beta: float = 2.0, seed: int = 0) -> dict:
    """Trains ``policy.expert_coordination`` (encoder frozen) with:

      * greedy cross-entropy on visible-target states, and
      * ``lambda_kd * RKL(P_student || P_target)`` on no-target states, where
        ``P_target`` is the router-gated teacher blend and ``lambda_kd`` shrinks
        when the teachers disagree.
    """
    torch.manual_seed(seed)
    flat, peer_count, actions, feats, dists, visible = samples.stack()

    # Router gates + gated target distribution (teachers frozen -> detach).
    with torch.no_grad():
        gates = router(feats)                                   # [N, 3]
        p_target = (gates.unsqueeze(-1) * dists).sum(dim=1)     # [N, n_act]
        p_target = p_target / p_target.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        disagree = _pairwise_disagreement(dists)                # [N]
        lambda_kd = 1.0 / (1.0 + beta * disagree)               # [N]

    # Freeze everything except the coordination head.
    for p in policy.parameters():
        p.requires_grad = False
    for p in policy.expert_coordination.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(policy.expert_coordination.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    n = flat.shape[0]
    gen = torch.Generator().manual_seed(seed)
    last = {"loss": 0.0, "rkl": 0.0, "ce": 0.0, "acc": 0.0}
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        tot, tot_rkl, tot_ce, nb = 0.0, 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            with torch.no_grad():  # encoder is frozen -> no graph needed for z
                z = policy.expert_encoder(flat[idx], peer_count[idx].unsqueeze(-1))
            logits = policy.expert_coordination(z)
            logp = F.log_softmax(logits, dim=-1)

            vis = visible[idx]
            loss = torch.zeros(())
            if vis.any():                                   # greedy hard label
                loss = loss + ce(logits[vis], actions[idx][vis])
                tot_ce += float(ce(logits[vis], actions[idx][vis]).item())
            nvis = ~vis
            if nvis.any():                                  # gated reverse-KL
                sp = logp[nvis].exp()
                rkl = (sp * (logp[nvis] - torch.log(p_target[idx][nvis] + _EPS))).sum(dim=-1)
                rkl = (lambda_kd[idx][nvis] * rkl).mean()
                loss = loss + rkl
                tot_rkl += float(rkl.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1

        # Distillation accuracy vs the gated target's argmax (no-target states).
        with torch.no_grad():
            z = policy.expert_encoder(flat, peer_count.unsqueeze(-1))
            pred = policy.expert_coordination(z).argmax(dim=-1)
            tgt = p_target.argmax(dim=-1)
            m = ~visible
            acc = float((pred[m] == tgt[m]).float().mean()) if m.any() else 0.0
        last = {"loss": tot / max(nb, 1), "rkl": tot_rkl / max(nb, 1),
                "ce": tot_ce / max(nb, 1), "acc": 100.0 * acc}

    # Restore grad flags so downstream training (e.g. router optimization) is unaffected.
    for p in policy.parameters():
        p.requires_grad = True
    return last


def train_gated_expert2(policy, env, checkpoint_dir: str = "checkpoints",
                        episodes: int = 8, steps: int = 60, epochs: int = 12,
                        seed: int = 0, weights: Optional[tuple] = None) -> dict:
    """End-to-end Expert 2 distillation: build teachers, collect samples, fit the
    gating router on pseudo-oracle labels, then gated reverse-KL into E2.

    ``env`` must be an ``EntityRescueEnv`` matching the checkpoints' dims. Returns
    the final distillation metrics plus the mean learned teacher weights.
    """
    from rescue_sim.Ensemble.ensemble import PolicyEnsemble

    bank = TeacherBank.from_checkpoints(env, checkpoint_dir=checkpoint_dir)
    ensemble = PolicyEnsemble.from_checkpoints(env, weights=weights, temperature=1.0)

    samples = collect_gated_samples(env, bank, ensemble, episodes, steps, seed)
    _, _, actions, feats, dists, _ = samples.stack()

    oracle = oracle_weights(dists, actions)                 # Phase 1
    router = train_gating_router(feats, oracle, seed=seed)  # Phase 2
    metrics = gated_rkl_distill(policy, samples, router, epochs=epochs, seed=seed)  # Phase 3

    with torch.no_grad():
        mean_gates = router(feats).mean(dim=0).tolist()
    metrics["teacher_weights"] = dict(zip(TEACHER_ORDER, [round(w, 3) for w in mean_gates]))
    metrics["temperatures"] = {"qmix": round(bank.tau["qmix"], 3),
                               "transf": round(bank.tau["transf"], 3)}
    return metrics


# ---------------------------------------------------------------------------
# Outcome-labeled MoE router training (real-MoE gate, not rule-based)
# ---------------------------------------------------------------------------
def train_outcome_router(
    policy,
    env,
    checkpoint_dir: str = "checkpoints",
    seed: int = 0,
    episodes: int = 4,
    max_steps: int = 70,
    steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    gate_tau: float = 0.25,
) -> dict:
    """Retrains the MoE gating router on OUTCOME labels instead of hand rules.

    For every state the MoE itself visits, each expert head proposes its greedy
    action, and the proposals are judged by the calibrated teacher mixture's
    probability for that action — the router learns to hand each state to the
    expert whose action the trained deep-RL models rate best. Isolated agents
    (no peers) are always labelled fallback, since coordination needs comms.
    The expert heads stay frozen throughout. Finally the gate temperature is
    sharpened (``gate_tau``) so routing is near winner-take-all rather than a
    logit blend of incompatible scales.

    Returns ``{"acc": ..., "gate_tau": ..., "n_states": ..., "expert_share": ...}``.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bank = TeacherBank.from_checkpoints(env, checkpoint_dir)
    A = env.num_agents
    neg = -1e9

    # -- collect (state, best-expert) pairs by rolling the current MoE -------
    obs_buf, peer_buf, label_buf = [], [], []
    policy.gate_tau = 1.0  # soft gate while labelling (sharpened at the end)
    policy.eval()
    for _ in range(episodes):
        env.reset()
        hidden = None
        for _ in range(max_steps):
            flat = np.asarray(env._observations(), dtype=np.float32)     # [A, obs]
            tokens = env.entity_obs().astype(np.float32)
            mask_np = env.valid_action_mask()
            peer = build_peer_matrix(env.positions)                       # [A, A]

            obs_t = torch.as_tensor(flat).unsqueeze(0)
            peer_t = torch.as_tensor(peer, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask_np, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                # Per-head greedy proposals (same frozen encoder + GRU state).
                peer_count = peer_t.sum(-1).view(A, 1)
                z = policy.expert_encoder(obs_t.view(A, -1), peer_count)
                y0 = policy.expert_exploration(z)
                y1 = policy.expert_coordination(z)
                h_flat = hidden.view(A, -1) if hidden is not None else None
                y2, _ = policy.expert_fallback(z, h_flat)
                m = mask_t.view(A, -1)
                proposals = torch.stack([
                    y0.masked_fill(~m, neg).argmax(-1),
                    y1.masked_fill(~m, neg).argmax(-1),
                    y2.masked_fill(~m, neg).argmax(-1),
                ], dim=1)                                                 # [A, 3]

                # Judge proposals with the calibrated teacher mixture.
                dists = bank.dists(torch.as_tensor(flat), torch.as_tensor(tokens))
                p_mix = dists.mean(dim=0)                                 # [A, n_act]
                scores = p_mix.gather(1, proposals)                       # [A, 3]
                labels = scores.argmax(dim=1)                             # [A]
                labels[peer_t.view(A, A).sum(-1) <= 1.0] = 2              # comms rule

                # Step with the MoE's own greedy action (distribution match).
                y_final, _w, hidden = policy(obs_t, peer_t, mask_t, hidden)
                actions = y_final.view(A, -1).masked_fill(~m, neg).argmax(-1)

            obs_buf.append(flat)
            peer_buf.append(peer.astype(np.float32))
            label_buf.append(labels.numpy())
            _, _, done, _ = env.step(actions.numpy())
            if done:
                break

    obs_all = torch.as_tensor(np.stack(obs_buf))                          # [N, A, obs]
    peer_all = torch.as_tensor(np.stack(peer_buf))                        # [N, A, A]
    lab_all = torch.as_tensor(np.stack(label_buf), dtype=torch.long)      # [N, A]
    n_states = obs_all.shape[0]

    # -- train only the router branch on the outcome labels ------------------
    for p in policy.parameters():
        p.requires_grad = False
    router_params = list(policy.router_encoder.parameters()) + list(policy.router.parameters())
    for p in router_params:
        p.requires_grad = True
    opt = torch.optim.Adam(router_params, lr=lr)

    acc = 0.0
    policy.train()
    for _ in range(steps):
        idx = rng.integers(0, n_states, size=min(batch_size, n_states))
        ob, pe, lb = obs_all[idx], peer_all[idx], lab_all[idx]
        mask = torch.ones(len(idx), A, policy.action_dim, dtype=torch.bool)
        _, weights, _ = policy(ob, pe, mask)
        loss = F.nll_loss(torch.log(weights.reshape(-1, 3) + _EPS), lb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = float((weights.reshape(-1, 3).argmax(-1) == lb.reshape(-1)).float().mean() * 100)

    for p in policy.parameters():
        p.requires_grad = True
    policy.gate_tau = gate_tau
    policy.eval()

    share = torch.bincount(lab_all.reshape(-1), minlength=3).float()
    share = (share / share.sum()).tolist()
    return {
        "acc": acc,
        "gate_tau": gate_tau,
        "n_states": int(n_states),
        "expert_share": {
            "explore": round(share[0], 3),
            "coord": round(share[1], 3),
            "fallback": round(share[2], 3),
        },
    }
