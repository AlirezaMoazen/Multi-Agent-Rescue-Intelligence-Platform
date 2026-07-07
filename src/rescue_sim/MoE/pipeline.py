# Copyright 2026 Alireza Moazen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Training pipeline for the Neural MoE policy on the real rescue environment.

Shared by ``demo_moe.py`` (terminal dashboard) and the visualization API
(live web dashboard): heuristic expert teachers label trajectories on real
grids, the expert heads are behavioral-cloned, and the attention router is
fine-tuned with communication-blackout penalties. Progress is reported
through optional callbacks so each caller renders it its own way.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rescue_sim.MAPPO.environment import RescueEnv
from rescue_sim.MoE.moe import NeuralMoEPolicy
from rescue_sim.shared import CARDINAL_ACTIONS, MOVE_DELTAS, Position

ACTION_DIM: int = 4       # N, S, E, W
COMM_RADIUS: int = 3      # Manhattan distance threshold for peer links
EXPERT_NAMES: tuple[str, str, str] = ("exploration", "coordination", "fallback")

# (dx, dy) per action index, in CARDINAL_ACTIONS order (N, S, E, W).
ACTION_DELTAS: list[tuple[int, int]] = [MOVE_DELTAS[a.value] for a in CARDINAL_ACTIONS]

# One team transition: (obs [A, obs_dim], peer_matrix [A, A], actions [A]).
Transition = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Progress callbacks: (current, total, loss, accuracy_pct, grad_norm)
ProgressCallback = Callable[[int, int, float, float, float], None]


def build_peer_matrix(positions: list[Position]) -> np.ndarray:
    """Builds the dynamic peer adjacency matrix from agent positions.

    A link exists where the Manhattan distance is below ``COMM_RADIUS``;
    the diagonal (self link) is always 1.

    Args:
        positions: Current agent positions.

    Returns:
        peer_matrix: [A, A] float32 adjacency matrix.
    """
    n = len(positions)
    peer = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            dist = abs(positions[i].x - positions[j].x) + abs(positions[i].y - positions[j].y)
            if dist < COMM_RADIUS:
                peer[i, j] = 1.0
    return peer


class FixedGridRescueEnv(RescueEnv):
    """RescueEnv that regenerates the *same* grid on every reset.

    The base env draws a fresh random grid per episode (right for training
    generalization). The live dashboard instead shows the MoE solving one
    fixed competition grid over repeated tries, so the routing evolution is
    comparable try-to-try.
    """

    def __init__(self, *args: object, grid_seed: int = 0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._grid_seed = grid_seed

    def reset(self) -> np.ndarray:
        self.rng = np.random.default_rng(self._grid_seed)
        return super().reset()


# ===========================================================================
# Heuristic expert teachers (trajectory labellers on the real grid)
# ===========================================================================

class ExplorationTeacher:
    """Frontier-style teacher: prefer moves onto cells the team has not visited."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.visited: set[Position] = set()

    def reset(self, env: RescueEnv) -> None:
        self.visited = set(env.positions)

    def act(self, env: RescueEnv, valid_mask: np.ndarray) -> np.ndarray:
        actions = np.zeros(env.num_agents, dtype=np.int64)
        for i, pos in enumerate(env.positions):
            best_score, best_actions = -1.0, [int(self.rng.integers(0, ACTION_DIM))]
            for a, (dx, dy) in enumerate(ACTION_DELTAS):
                if not valid_mask[i, a]:
                    continue
                dest = Position(pos.x + dx, pos.y + dy)
                score = 2.0 if dest not in self.visited else 1.0
                if score > best_score:
                    best_score, best_actions = score, [a]
                elif score == best_score:
                    best_actions.append(a)
            actions[i] = int(self.rng.choice(best_actions))
        self.visited.update(env.positions)
        return actions


class CoordinationTeacher:
    """Target-greedy teacher: descend Manhattan distance to the nearest live target."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.remaining: set[Position] = set()

    def reset(self, env: RescueEnv) -> None:
        assert env.grid is not None
        self.remaining = set(env.grid.target_a_positions | env.grid.target_b_positions)

    def act(self, env: RescueEnv, valid_mask: np.ndarray) -> np.ndarray:
        actions = np.zeros(env.num_agents, dtype=np.int64)
        for i, pos in enumerate(env.positions):
            if not self.remaining:
                valid = [a for a in range(ACTION_DIM) if valid_mask[i, a]]
                actions[i] = int(self.rng.choice(valid)) if valid else 0
                continue
            target = min(self.remaining, key=lambda t: abs(t.x - pos.x) + abs(t.y - pos.y))
            best_dist, best_actions = 10 ** 9, [0]
            for a, (dx, dy) in enumerate(ACTION_DELTAS):
                if not valid_mask[i, a]:
                    continue
                dist = abs(target.x - (pos.x + dx)) + abs(target.y - (pos.y + dy))
                if dist < best_dist:
                    best_dist, best_actions = dist, [a]
                elif dist == best_dist:
                    best_actions.append(a)
            actions[i] = int(self.rng.choice(best_actions))
        return actions

    def observe(self, env: RescueEnv) -> None:
        """Removes targets an agent has just rescued (landed on)."""
        for pos in env.positions:
            self.remaining.discard(pos)


class FallbackTeacher:
    """Direction-persistence teacher for isolated agents.

    Keeps the previous heading while it stays valid and rotates clockwise on
    walls (wall-following), producing the temporally correlated labels the GRU
    fallback head needs to escape dead-ends instead of blind looping.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.headings: np.ndarray = np.zeros(0, dtype=np.int64)

    def reset(self, env: RescueEnv) -> None:
        self.headings = self.rng.integers(0, ACTION_DIM, size=env.num_agents)

    def act(self, env: RescueEnv, valid_mask: np.ndarray) -> np.ndarray:
        # Clockwise scan order per heading: N -> E -> S -> W (indices 0, 2, 1, 3).
        clockwise = [0, 2, 1, 3]
        actions = np.zeros(env.num_agents, dtype=np.int64)
        for i in range(env.num_agents):
            heading = int(self.headings[i])
            if valid_mask[i, heading] and self.rng.random() < 0.85:
                actions[i] = heading
                continue
            start = clockwise.index(heading)
            for offset in range(1, ACTION_DIM + 1):
                candidate = clockwise[(start + offset) % ACTION_DIM]
                if valid_mask[i, candidate]:
                    actions[i] = candidate
                    break
            else:
                actions[i] = heading
            self.headings[i] = actions[i]
        return actions


Teacher = Union[ExplorationTeacher, CoordinationTeacher, FallbackTeacher]


def make_teachers(rng: np.random.Generator) -> list[Teacher]:
    """The three teachers in expert-head order (exploration, coordination, fallback)."""
    return [ExplorationTeacher(rng), CoordinationTeacher(rng), FallbackTeacher(rng)]


def collect_expert_dataset(
    env: RescueEnv,
    teacher: Teacher,
    episodes: int,
    max_steps: int,
) -> list[Transition]:
    """Rolls the teacher on the real grid and records labelled transitions.

    Args:
        env: The rescue environment (a fresh random grid per episode).
        teacher: Heuristic expert producing target action labels.
        episodes: Number of full episodes to collect.
        max_steps: Step cap per collection episode.

    Returns:
        Dataset of (obs [A, obs_dim], peer_matrix [A, A], actions [A]) tuples.
    """
    dataset: list[Transition] = []
    for _ in range(episodes):
        obs = env.reset()  # [A, obs_dim]
        teacher.reset(env)
        for _ in range(max_steps):
            valid_mask = env.valid_action_mask()
            actions = teacher.act(env, valid_mask)
            peer = build_peer_matrix(env.positions)

            dataset.append((
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(peer, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
            ))

            obs, _, done, _ = env.step(actions)
            if isinstance(teacher, CoordinationTeacher):
                teacher.observe(env)
            if done:
                break
    return dataset


def _stack_dataset(dataset: list[Transition]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacks a transition list into batched tensors [N, A, ...]."""
    obs = torch.stack([t[0] for t in dataset])
    peer = torch.stack([t[1] for t in dataset])
    acts = torch.stack([t[2] for t in dataset])
    return obs, peer, acts


# ===========================================================================
# Stage 1: behavioral cloning of the expert heads
# ===========================================================================

def run_expert_distillation(
    policy: NeuralMoEPolicy,
    datasets: list[list[Transition]],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    on_epoch: Optional[ProgressCallback] = None,
) -> dict[str, float]:
    """Offline behavioral cloning of the three expert heads.

    Full mini-batched training with a 90/10 train/validation split. The gating
    router stays frozen; validation accuracy is computed under
    ``torch.no_grad()``.

    Args:
        policy: The NeuralMoEPolicy to train.
        datasets: [exploration, coordination, fallback] trajectory datasets.
        epochs: Number of full training epochs.
        batch_size: Mini-batch size (in team transitions).
        lr: Adam learning rate.
        seed: Shuffle seed for reproducible batching.
        on_epoch: Called after each epoch with (epoch, epochs, loss, acc, grad_norm).

    Returns:
        Final metrics: {"loss": ..., "accuracy": ...}.
    """
    for p in policy.router_encoder.parameters():
        p.requires_grad = False
    for p in policy.router.parameters():
        p.requires_grad = False
    trainable = (
        list(policy.expert_encoder.parameters())
        + list(policy.expert_exploration.parameters())
        + list(policy.expert_coordination.parameters())
        + list(policy.expert_fallback.parameters())
    )
    for p in trainable:
        p.requires_grad = True

    optimizer = optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()
    heads = [policy.expert_exploration, policy.expert_coordination, policy.expert_fallback]
    generator = torch.Generator().manual_seed(seed)

    # 90/10 train/validation split per head
    splits: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = []
    for dataset in datasets:
        obs, peer, acts = _stack_dataset(dataset)
        n_val = max(1, len(dataset) // 10)
        perm = torch.randperm(len(dataset), generator=generator)
        tr, va = perm[n_val:], perm[:n_val]
        splits.append(
            ((obs[tr], peer[tr], acts[tr]), (obs[va], peer[va], acts[va]))
        )

    avg_loss, val_acc = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        total_loss, num_batches, grad_norm = 0.0, 0, 0.0

        for head_idx, ((obs, peer, acts), _) in enumerate(splits):
            head = heads[head_idx]
            perm = torch.randperm(obs.size(0), generator=generator)

            for start in range(0, obs.size(0), batch_size):
                idx = perm[start:start + batch_size]
                obs_b, peer_b, act_b = obs[idx], peer[idx], acts[idx]

                B, A, obs_dim = obs_b.shape
                assert peer_b.shape == (B, A, A), f"peer batch shape {peer_b.shape}"
                peer_count = torch.sum(peer_b, dim=-1, keepdim=True)

                optimizer.zero_grad()
                z = policy.expert_encoder(
                    obs_b.view(B * A, obs_dim), peer_count.view(B * A, 1)
                )
                if head_idx == 2:
                    logits, _ = head(z, None)
                else:
                    logits = head(z)

                loss = criterion(logits, act_b.view(B * A))
                loss.backward()
                grad_norm = float(
                    torch.norm(
                        torch.stack([
                            p.grad.detach().norm(2)
                            for p in policy.expert_encoder.parameters()
                            if p.grad is not None
                        ])
                    )
                )
                optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

        # Validation accuracy — memory-efficient, no autograd graph
        correct, total = 0, 0
        with torch.no_grad():
            for head_idx, (_, (obs_v, peer_v, act_v)) in enumerate(splits):
                head = heads[head_idx]
                B, A, obs_dim = obs_v.shape
                peer_count = torch.sum(peer_v, dim=-1, keepdim=True)
                z = policy.expert_encoder(
                    obs_v.view(B * A, obs_dim), peer_count.view(B * A, 1)
                )
                if head_idx == 2:
                    logits, _ = head(z, None)
                else:
                    logits = head(z)
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == act_v.view(B * A)).sum().item())
                total += B * A

        avg_loss = total_loss / max(num_batches, 1)
        val_acc = 100.0 * correct / max(total, 1)
        if on_epoch is not None:
            on_epoch(epoch, epochs, avg_loss, val_acc, grad_norm)

    return {"loss": avg_loss, "accuracy": val_acc}


# ===========================================================================
# Stage 2: attention-router optimization with blackout penalties
# ===========================================================================

def run_router_optimization(
    policy: NeuralMoEPolicy,
    datasets: list[list[Transition]],
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    on_step: Optional[ProgressCallback] = None,
) -> dict[str, float]:
    """Fine-tunes the attention router with Conditional Indicator Mask penalties.

    Samples real grid states and augments each batch with forced blackout
    (identity peer matrix) and forced full-connectivity copies so the
    penalties always have coverage:

        isolated  (peer_count == 1)  -> push g_fallback -> 1
        connected (peer_count == A)  -> push g_coord    -> 1

    Args:
        policy: The NeuralMoEPolicy whose router is fine-tuned.
        datasets: The collected expert datasets (real-state pool).
        steps: Number of optimization steps.
        batch_size: States sampled per step (before augmentation).
        lr: Adam learning rate.
        seed: Sampling seed.
        on_step: Called per step with (step, steps, loss, 100.0, grad_norm).

    Returns:
        Final metrics: {"loss": ...}.
    """
    for p in policy.expert_encoder.parameters():
        p.requires_grad = False
    for p in policy.expert_exploration.parameters():
        p.requires_grad = False
    for p in policy.expert_coordination.parameters():
        p.requires_grad = False
    for p in policy.expert_fallback.parameters():
        p.requires_grad = False
    router_params = list(policy.router_encoder.parameters()) + list(policy.router.parameters())
    for p in router_params:
        p.requires_grad = True

    optimizer = optim.Adam(router_params, lr=lr)
    rng = np.random.default_rng(seed)

    pool = [t for dataset in datasets for t in dataset]
    A = policy.num_agents

    loss_value = 0.0
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(pool), size=batch_size)
        obs = torch.stack([pool[i][0] for i in idx])        # [B, A, obs_dim]
        peer_real = torch.stack([pool[i][1] for i in idx])  # [B, A, A]

        # Connectivity augmentation: real / forced blackout / forced connected
        peer_blackout = torch.eye(A).expand(batch_size, A, A)
        peer_connected = torch.ones(batch_size, A, A)
        obs_all = torch.cat([obs, obs, obs], dim=0)
        peer_all = torch.cat([peer_real, peer_blackout, peer_connected], dim=0)
        action_mask = torch.ones(obs_all.size(0), A, policy.action_dim, dtype=torch.bool)

        assert obs_all.shape[1:] == (A, obs.shape[-1]), f"obs batch shape {obs_all.shape}"
        assert peer_all.shape[1:] == (A, A), f"peer batch shape {peer_all.shape}"

        optimizer.zero_grad()
        _, weights, _ = policy(obs_all, peer_all, action_mask)

        peer_count = torch.sum(peer_all, dim=-1)  # [3B, A]
        is_isolated = (peer_count == 1.0).float()
        is_connected = (peer_count == float(A)).float()

        g_coord = weights[:, :, 1]
        g_fallback = weights[:, :, 2]

        blackout_penalty = (
            (is_isolated * (1.0 - g_fallback) ** 2).sum() / is_isolated.sum().clamp(min=1.0)
        )
        connected_penalty = (
            (is_connected * (1.0 - g_coord) ** 2).sum() / is_connected.sum().clamp(min=1.0)
        )
        loss = 10.0 * blackout_penalty + 10.0 * connected_penalty

        loss.backward()
        grad_norm = float(
            torch.norm(
                torch.stack([
                    p.grad.detach().norm(2) for p in router_params if p.grad is not None
                ])
            )
        )
        optimizer.step()

        loss_value = float(loss.item())
        if on_step is not None:
            on_step(step, steps, loss_value, 100.0, grad_norm)

    return {"loss": loss_value}


def train_moe_policy(
    env: RescueEnv,
    episodes_per_head: int = 6,
    collect_steps: int = 80,
    epochs: int = 20,
    batch_size: int = 64,
    router_steps: int = 120,
    router_batch: int = 16,
    lr: float = 1e-3,
    seed: int = 7,
    on_distill_epoch: Optional[ProgressCallback] = None,
    on_router_step: Optional[ProgressCallback] = None,
) -> NeuralMoEPolicy:
    """End-to-end pipeline: collect teacher trajectories, clone experts, tune router.

    Args:
        env: The rescue environment to train against.
        episodes_per_head: Collection episodes per expert teacher.
        collect_steps: Step cap per collection episode.
        epochs: Behavioral cloning epochs.
        batch_size: BC mini-batch size.
        router_steps: Router optimization steps.
        router_batch: Real states sampled per router step.
        lr: Adam learning rate (both stages).
        seed: Global random seed.
        on_distill_epoch: Progress callback for the BC stage.
        on_router_step: Progress callback for the router stage.

    Returns:
        The trained NeuralMoEPolicy.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    policy = NeuralMoEPolicy(env.obs_dim, env.num_agents, env.view_radius, ACTION_DIM)
    datasets = [
        collect_expert_dataset(env, teacher, episodes_per_head, collect_steps)
        for teacher in make_teachers(rng)
    ]
    run_expert_distillation(policy, datasets, epochs, batch_size, lr, seed, on_distill_epoch)
    run_router_optimization(policy, datasets, router_steps, router_batch, lr, seed, on_router_step)
    return policy
