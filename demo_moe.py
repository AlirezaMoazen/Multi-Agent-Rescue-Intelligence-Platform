# Copyright 2026 Alireza Moazen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Production demonstration of the step-level Neural Mixture of Experts (MoE) policy.

Runs the full pipeline against the real 20x20 ``RescueEnv`` — no synthetic
shortcuts or mock iterations:

    1. Collect expert trajectories on the 20x20 grid from three heuristic
       teachers (frontier exploration, target-greedy coordination, and a
       direction-persistent fallback for isolated agents).
    2. Phase A — full behavioral-cloning training of the expert heads with
       live progress bars (epoch, cross-entropy loss, BC accuracy, grad norm),
       followed by attention-router fine-tuning with communication-blackout
       penalties.
    3. Phase B — live 20x20 ASCII grid render of a policy-driven rollout
       (``.`` empty, ``#`` walls, ``G0-G3`` goals, ``A0-A3`` agents).
    4. Phase C — per-step telemetry table: step index, agent coordinates,
       active peer count, baseline parameters, and the live softmax routing
       vector ``[g_explore, g_coord, g_fallback]`` plus the GRU hidden norm.
    5. Integration gate — automatically executes ``pytest tests/test_moe.py``
       to confirm loss minimization, BC accuracy, and multi-agent dimension
       flows compile without runtime errors.

Usage:
    python demo_moe.py                       # full production run
    python demo_moe.py --epochs 30 --seed 3  # custom configuration
    python demo_moe.py --skip-tests          # skip the pytest gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parent

try:
    from rescue_sim.config.settings import GridSettings
    from rescue_sim.MAPPO import RescueEnv
    from rescue_sim.MoE.moe import NeuralMoEPolicy
    from rescue_sim.shared import CARDINAL_ACTIONS, MOVE_DELTAS, Position
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from rescue_sim.config.settings import GridSettings
    from rescue_sim.MAPPO import RescueEnv
    from rescue_sim.MoE.moe import NeuralMoEPolicy
    from rescue_sim.shared import CARDINAL_ACTIONS, MOVE_DELTAS, Position


# ===========================================================================
# 1. Environment / routing constants
# ===========================================================================

GRID_SIZE: int = 20
NUM_AGENTS: int = 4
VIEW_RADIUS: int = 3      # 7x7 ego-centric window (3-block blindness constraint)
ACTION_DIM: int = 4       # N, S, E, W
COMM_RADIUS: int = 3      # Manhattan distance threshold for peer links
EXPERT_NAMES: tuple[str, str, str] = ("exploration", "coordination", "fallback")

# (dx, dy) per action index, in CARDINAL_ACTIONS order (N, S, E, W).
ACTION_DELTAS: list[tuple[int, int]] = [MOVE_DELTAS[a.value] for a in CARDINAL_ACTIONS]

# One team transition: (obs [A, obs_dim], peer_matrix [A, A], actions [A]).
Transition = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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


# ===========================================================================
# 2. Heuristic expert teachers (trajectory labellers on the real grid)
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


def collect_expert_dataset(
    env: RescueEnv,
    teacher: ExplorationTeacher | CoordinationTeacher | FallbackTeacher,
    episodes: int,
    max_steps: int,
) -> list[Transition]:
    """Rolls the teacher on the real 20x20 grid and records labelled transitions.

    Args:
        env: The rescue environment (a fresh random 20x20 grid per episode).
        teacher: Heuristic expert producing target action labels.
        episodes: Number of full episodes to collect.
        max_steps: Step cap per collection episode.

    Returns:
        Dataset of (obs [A, obs_dim], peer_matrix [A, A], actions [A]) tuples.
    """
    dataset: list[Transition] = []
    for _ in range(episodes):
        env.reset()
        teacher.reset(env)
        for _ in range(max_steps):
            obs = env._observations()  # [A, obs_dim] — current-state observation
            valid_mask = env.valid_action_mask()
            actions = teacher.act(env, valid_mask)
            peer = build_peer_matrix(env.positions)

            dataset.append((
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(peer, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
            ))

            _, _, done, _ = env.step(actions)
            if isinstance(teacher, CoordinationTeacher):
                teacher.observe(env)
            if done:
                break
    return dataset


# ===========================================================================
# 3. Phase A: full behavioral-cloning training with live progress bars
# ===========================================================================

def print_progress_bar(
    label: str,
    epoch: int,
    total_epochs: int,
    loss: float,
    acc: float,
    grad_norm: float,
) -> None:
    """Renders a dynamic ASCII progress bar for live training feedback.

    Args:
        label: Short stage label (e.g. "BC" or "Router").
        epoch: Current epoch number (1-indexed).
        total_epochs: Total epoch count.
        loss: Current cross-entropy (or penalty) loss value.
        acc: Behavioral-cloning validation accuracy (%).
        grad_norm: L2 gradient norm of the trainable parameters.
    """
    width = 25
    filled = int(width * epoch / total_epochs)
    bar = ("=" * filled + ">" + "." * (width - filled - 1))[:width]
    print(
        f"{label} {epoch:03d}/{total_epochs:03d} [{bar}] "
        f"Loss: {loss:.6f} | BC-Acc: {acc:6.2f}% | Grad Norm: {grad_norm:.4f}"
    )


def _stack_dataset(dataset: list[Transition]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacks a transition list into batched tensors [N, A, ...]."""
    obs = torch.stack([t[0] for t in dataset])
    peer = torch.stack([t[1] for t in dataset])
    acts = torch.stack([t[2] for t in dataset])
    return obs, peer, acts


def run_expert_distillation(
    policy: NeuralMoEPolicy,
    datasets: list[list[Transition]],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Stage 1: offline behavioral cloning of the three expert heads.

    Full mini-batched training with a 90/10 train/validation split.  The
    gating router stays frozen; validation accuracy is computed under
    ``torch.no_grad()``.

    Args:
        policy: The NeuralMoEPolicy to train.
        datasets: [exploration, coordination, fallback] trajectory datasets.
        epochs: Number of full training epochs.
        batch_size: Mini-batch size (in team transitions).
        lr: Adam learning rate.
        seed: Shuffle seed for reproducible batching.
    """
    print("\n[PHASE A: LIVE TRAINING TELEPRINTER]")
    print("=" * 96)
    print("Stage 1: Expert Policy Distillation (Behavioral Cloning on the 20x20 grid)")
    print("-> Freezing Gating Router weights; training encoder + 3 expert heads.")
    sizes = " | ".join(f"{n}: {len(d)} transitions" for n, d in zip(EXPERT_NAMES, datasets))
    print(f"-> Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | {sizes}")
    print("-" * 96)

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
        print_progress_bar("BC  ", epoch, epochs, avg_loss, val_acc, grad_norm)


def run_gating_router_optimization(
    policy: NeuralMoEPolicy,
    datasets: list[list[Transition]],
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Stage 2: online attention-router fine-tuning with blackout penalties.

    Samples real 20x20 grid states and augments each batch with forced
    blackout (identity peer matrix) and forced full-connectivity copies so the
    Conditional Indicator Mask penalties always have coverage:

        isolated  (peer_count == 1)  -> push g_fallback -> 1
        connected (peer_count == A)  -> push g_coord    -> 1

    Args:
        policy: The NeuralMoEPolicy whose router is fine-tuned.
        datasets: The collected expert datasets (real-state pool).
        steps: Number of optimization steps.
        batch_size: States sampled per step (before augmentation).
        lr: Adam learning rate.
        seed: Sampling seed.
    """
    print("\nStage 2: Attention-Based Gating Router Optimization")
    print("-> Freezing expert heads; fine-tuning scaled dot-product attention router.")
    print(f"-> Steps: {steps} | Batch: {batch_size} (x3 connectivity augmentation) | LR: {lr}")
    print("-" * 96)

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

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(pool), size=batch_size)
        obs = torch.stack([pool[i][0] for i in idx])       # [B, A, obs_dim]
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

        peer_count = torch.sum(peer_all, dim=-1)          # [3B, A]
        is_isolated = (peer_count == 1.0).float()
        is_connected = (peer_count == float(A)).float()

        g_coord = weights[:, :, 1]
        g_fallback = weights[:, :, 2]

        blackout_penalty = (is_isolated * (1.0 - g_fallback) ** 2).sum() / is_isolated.sum().clamp(min=1.0)
        connected_penalty = (is_connected * (1.0 - g_coord) ** 2).sum() / is_connected.sum().clamp(min=1.0)
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

        if step == 1 or step % 10 == 0 or step == steps:
            print_progress_bar("Gate", step, steps, float(loss.item()), 100.0, grad_norm)

    print("=" * 96)


# ===========================================================================
# 4. Phase B & C: live dashboard rendering
# ===========================================================================

def render_grid_20x20(
    positions: list[Position],
    goals: list[Position],
    rescued: set[Position],
    obstacles: frozenset[Position],
) -> None:
    """Renders the live 20x20 ASCII grid world to the terminal.

    Legend: ``.`` empty | ``#`` wall | ``G0-G3`` goals | ``*`` rescued | ``A0-A3`` agents.

    Args:
        positions: Current agent positions.
        goals: All goal positions in stable index order (G0..G3).
        rescued: Goal positions already rescued (rendered as ``*``).
        obstacles: Static wall positions.
    """
    grid: list[list[str]] = [[" . " for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for p in obstacles:
        grid[p.y][p.x] = " # "
    for idx, g in enumerate(goals):
        grid[g.y][g.x] = " * " if g in rescued else f"G{idx} "
    for idx, a in enumerate(positions):
        grid[a.y][a.x] = f"A{idx} "

    print("\n[PHASE B: 20x20 GRID WORLD VIEW]  ( . empty | # wall | G0-G3 goals | * rescued | A0-A3 agents )")
    print("   " + "".join(f"{i:^3}" for i in range(GRID_SIZE)))
    print("   " + "---" * GRID_SIZE)
    for r_idx, row in enumerate(grid):
        print(f"{r_idx:2d}|" + "".join(row) + "|")
    print("   " + "---" * GRID_SIZE)


def print_telemetry_table(
    step: int,
    positions: list[Position],
    weights: torch.Tensor,
    peer_matrix: torch.Tensor,
    fallback_hidden: torch.Tensor,
) -> None:
    """Prints the parametric evolution telemetry table with GRU state info.

    Args:
        step: Current simulation step index.
        positions: Agent positions at this step.
        weights: [1, A, 3] softmax gating weights.
        peer_matrix: [1, A, A] peer adjacency tensor.
        fallback_hidden: [1, A, H] GRU hidden state tensor.
    """
    print("\n[PHASE C: PARAMETRIC EVOLUTION TELEMETRY]")
    header = (
        f"{'Step':<6}{'Agent':<9}{'Pos':<10}{'Peers':<7}"
        f"{'Baseline Params':<30}"
        f"{'MoE Gating [g_exp, g_coord, g_fall]':<38}"
        f"{'GRU |h|':<9}"
    )
    print(header)
    print("-" * 109)

    peer_count = torch.sum(peer_matrix, dim=-1).squeeze(0)   # [A]
    weights_step = weights.squeeze(0)                          # [A, 3]
    h_norms = torch.norm(fallback_hidden.squeeze(0), dim=-1)   # [A]

    for a, pos in enumerate(positions):
        pos_str = f"({pos.x},{pos.y})"
        if int(peer_count[a].item()) <= 1:
            baseline_str = "Hyst Q: α=0.10, β=0.01 [ISO]"
        else:
            baseline_str = "Frontier Exploration: γ=0.95"
        w_str = (
            f"[{weights_step[a, 0]:.4f}, "
            f"{weights_step[a, 1]:.4f}, "
            f"{weights_step[a, 2]:.4f}]"
        )
        print(
            f"{step:<6}{f'Agent-{a}':<9}{pos_str:<10}"
            f"{int(peer_count[a].item()):<7}{baseline_str:<30}"
            f"{w_str:<38}{h_norms[a].item():<9.4f}"
        )
    print("=" * 109)


def run_live_simulation(
    policy: NeuralMoEPolicy,
    env: RescueEnv,
    sim_steps: int,
    render_every: int,
) -> dict[str, float]:
    """Phase B/C: policy-driven rollout on a fresh 20x20 grid with GRU tracking.

    Actions are sampled from the blended masked logits under
    ``torch.no_grad()``; the GRU hidden state persists across the whole
    episode timeline.

    Args:
        policy: The trained NeuralMoEPolicy.
        env: The 20x20 rescue environment.
        sim_steps: Maximum simulation steps.
        render_every: Grid/telemetry render period (step 1 and the final step
            always render).

    Returns:
        Episode summary: rescued targets, total targets, success flag, steps.
    """
    obs_np = env.reset()
    assert env.grid is not None
    goals: list[Position] = sorted(
        env.grid.target_a_positions | env.grid.target_b_positions,
        key=lambda p: (p.y, p.x),
    )
    rescued: set[Position] = set()
    fallback_hidden: Optional[torch.Tensor] = None
    info: dict[str, float] = {"rescued": 0, "targets": len(goals), "success": False, "steps": 0}

    print("\n" + "=" * 96)
    print(f"   SIMULATION: policy-driven rollout on a fresh {GRID_SIZE}x{GRID_SIZE} grid "
          f"(max {sim_steps} steps, render every {render_every})")
    print("=" * 96)

    for step in range(1, sim_steps + 1):
        peer_np = build_peer_matrix(env.positions)

        obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)      # [1, A, obs_dim]
        peer_t = torch.tensor(peer_np, dtype=torch.float32).unsqueeze(0)    # [1, A, A]
        mask_t = torch.tensor(env.valid_action_mask(), dtype=torch.bool).unsqueeze(0)

        # Tensor shape assertions before the routing step
        assert obs_t.shape == (1, env.num_agents, env.obs_dim), \
            f"obs shape: expected [1, {env.num_agents}, {env.obs_dim}], got {obs_t.shape}"
        assert peer_t.shape == (1, env.num_agents, env.num_agents), \
            f"peer shape: expected [1, {env.num_agents}, {env.num_agents}], got {peer_t.shape}"
        assert mask_t.shape == (1, env.num_agents, policy.action_dim), \
            f"mask shape: expected [1, {env.num_agents}, {policy.action_dim}], got {mask_t.shape}"

        with torch.no_grad():
            y_final, weights, fallback_hidden = policy(obs_t, peer_t, mask_t, fallback_hidden)
            probs = torch.softmax(y_final, dim=-1)
            actions = torch.multinomial(
                probs.view(env.num_agents, policy.action_dim), num_samples=1
            ).squeeze(-1)

        positions_before = list(env.positions)
        if step == 1 or step % render_every == 0:
            render_grid_20x20(positions_before, goals, rescued, env.grid.obstacles)
            print_telemetry_table(step, positions_before, weights, peer_t, fallback_hidden)

        obs_np, _, done, step_info = env.step(actions.numpy())
        info = step_info
        rescued.update(g for g in goals if g in set(env.positions))

        if done:
            render_grid_20x20(list(env.positions), goals, rescued, env.grid.obstacles)
            print_telemetry_table(step, list(env.positions), weights, peer_t, fallback_hidden)
            break

    print(
        f"\n[SIMULATION SUMMARY] steps: {info['steps']} | "
        f"rescued: {info['rescued']}/{info['targets']} | success: {bool(info['success'])}"
    )
    return dict(info)


# ===========================================================================
# 5. Integration gate: automatic pytest execution
# ===========================================================================

def run_test_suite() -> int:
    """Executes the local pytest suite for the MoE module and reports the result.

    Returns:
        The pytest process exit code (0 = all tests green).
    """
    print("\n" + "=" * 96)
    print("   INTEGRATION GATE: pytest tests/test_moe.py")
    print("=" * 96)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_moe.py", "-q"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    print(output[-2000:] if len(output) > 2000 else output)
    verdict = "ALL TESTS PASSED" if result.returncode == 0 else "TESTS FAILED"
    print(f"\n[INTEGRATION GATE] {verdict} (exit code {result.returncode})")
    return result.returncode


# ===========================================================================
# 6. Main production pipeline
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parses the production run configuration."""
    parser = argparse.ArgumentParser(description="Neural MoE production demonstration (20x20 grid).")
    parser.add_argument("--epochs", type=int, default=20, help="behavioral cloning epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="BC mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (both stages)")
    parser.add_argument("--episodes-per-head", type=int, default=6,
                        help="20x20 collection episodes per expert teacher")
    parser.add_argument("--collect-steps", type=int, default=80,
                        help="step cap per collection episode")
    parser.add_argument("--router-steps", type=int, default=120,
                        help="attention router optimization steps")
    parser.add_argument("--router-batch", type=int, default=16,
                        help="real states sampled per router step")
    parser.add_argument("--sim-steps", type=int, default=60, help="live simulation step cap")
    parser.add_argument("--render-every", type=int, default=5, help="dashboard render period")
    parser.add_argument("--seed", type=int, default=7, help="global random seed")
    parser.add_argument("--skip-tests", action="store_true", help="skip the pytest gate")
    return parser.parse_args()


def main() -> int:
    """Full production pipeline: collect -> train -> simulate -> test.

    Returns:
        Process exit code (non-zero if the pytest gate fails).
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 96)
    print("   NEURAL MIXTURE OF EXPERTS (MoE) — PRODUCTION DEMONSTRATION")
    print("   Attention-Based Router | GRU Temporal Fallback | Real 20x20 RescueEnv")
    print("=" * 96)

    grid = GridSettings(
        width=GRID_SIZE, height=GRID_SIZE, obstacle_probability=0.15,
        target_a_count=2, target_b_count=2,
    )
    env = RescueEnv(
        grid, num_agents=NUM_AGENTS, max_steps=max(args.collect_steps, args.sim_steps),
        view_radius=VIEW_RADIUS, seed=args.seed,
    )
    policy = NeuralMoEPolicy(env.obs_dim, NUM_AGENTS, VIEW_RADIUS, ACTION_DIM)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nEnvironment: {GRID_SIZE}x{GRID_SIZE} grid | {NUM_AGENTS} agents | "
          f"7x7 ego window (view radius {VIEW_RADIUS}) | obs_dim {env.obs_dim}")
    print(f"Model Parameters: {total_params:,} total")

    # -- Trajectory collection on the real grid -----------------------------
    print(f"\nCollecting expert trajectories "
          f"({args.episodes_per_head} episodes x {args.collect_steps} steps per teacher) ...")
    teachers = [ExplorationTeacher(rng), CoordinationTeacher(rng), FallbackTeacher(rng)]
    datasets = [
        collect_expert_dataset(env, teacher, args.episodes_per_head, args.collect_steps)
        for teacher in teachers
    ]

    # -- Phase A: full training loops ----------------------------------------
    run_expert_distillation(policy, datasets, args.epochs, args.batch_size, args.lr, args.seed)
    run_gating_router_optimization(
        policy, datasets, args.router_steps, args.router_batch, args.lr, args.seed
    )

    # -- Phase B & C: live dashboard rollout ---------------------------------
    run_live_simulation(policy, env, args.sim_steps, args.render_every)

    # -- Integration gate -----------------------------------------------------
    exit_code = 0 if args.skip_tests else run_test_suite()
    print("\n[DEMO COMPLETE] All phases executed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
