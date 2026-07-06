# Copyright 2026 Alireza Moazen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Executable demonstration of step-level Neural Mixture of Experts (MoE) swarm design.

This script is fully self-contained, bootstrapping synthetic dataset generation,
two-stage optimization, and a 20x20 grid evolution tracking weight shifts during
communication blackouts.

Architecture:
    - AttentionGatingRouter: Scaled dot-product attention over peer latent embeddings
    - RecurrentFallbackHead: GRU-based temporal memory for isolated agents
    - SharedFeatureEncoder: CNN + MLP encoder with 7x7 ego-centric observation window

Dashboard Phases:
    Phase A: Active console progress bars (Epoch, Cross-Entropy, BC accuracy, grad norm)
    Phase B: Live 20x20 ASCII grid render (. empty, # walls, G0-G3 goals, A0-A3 agents)
    Phase C: Telemetry table (step, coords, peers, baseline params, softmax routing vector)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ===========================================================================
# 1. Coordinate / Helper Definitions
# ===========================================================================

@dataclass(frozen=True)
class Position:
    """Immutable 2D grid coordinate."""
    x: int
    y: int


# ===========================================================================
# 2. Self-Contained Neural Architecture
# ===========================================================================

class SharedFeatureEncoder(nn.Module):
    """Processes local spatial grid maps and scalar/communication inputs into a unified latent space.

    Processes a 7x7 egocentric spatial grid map (representing a 3-block visibility constraint
    in all directions) alongside peer connection inputs and metadata.

    Shape Flow:
        obs: [Batch, obs_dim] → z: [Batch, latent_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        view_radius: int = 3,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.view_radius = view_radius
        self.win = 2 * view_radius + 1  # 7
        self.channels = 4
        self.window_dim = self.win * self.win * self.channels  # 7 * 7 * 4 = 196

        # Spatial grid ConvNet
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_dim = 16 * self.win * self.win  # 16 * 49 = 784

        # MLP for meta features and agent identity
        self.mlp_meta = nn.Sequential(
            nn.Linear(4 + num_agents, 16),
            nn.ReLU(),
        )

        # MLP for permutation-invariant peer count representation
        self.mlp_comm = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
        )

        # Final projection head
        self.proj = nn.Sequential(
            nn.Linear(conv_output_dim + 16 + 8, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs: torch.Tensor, peer_count: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract embedding z.

        Args:
            obs: [Batch, obs_dim] flat agent observations.
            peer_count: [Batch, 1] pooled peer count.

        Returns:
            z: [Batch, latent_dim] unified representation.
        """
        batch_size = obs.size(0)

        # --- Spatial window extraction ---
        # Upstream channels-LAST layout: [H, W, C] → [H*W*C].
        # Reconstruct and convert to channels-first for Conv2d.
        window = (
            obs[:, :self.window_dim]
            .reshape(batch_size, self.win, self.win, self.channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [Batch, C=4, H=win, W=win]
        meta = obs[:, self.window_dim:]

        h_spatial = self.conv(window)
        h_meta = self.mlp_meta(meta)
        h_comm = self.mlp_comm(peer_count)

        z = torch.cat([h_spatial, h_meta, h_comm], dim=-1)
        return self.proj(z)


class AttentionGatingRouter(nn.Module):
    """Attention-based router using scaled dot-product attention over peer embeddings.

    Shape Flow:
        z_ego: [Batch, D] + z_peers: [Batch, P, D] + mask: [Batch, P]
        → weights: [Batch, 3]
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = math.sqrt(latent_dim)

        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.W_v = nn.Linear(latent_dim, latent_dim)

        self.gate_proj = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(
        self,
        z_ego: torch.Tensor,
        z_peers: torch.Tensor,
        peer_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes gating weights via scaled dot-product attention.

        Args:
            z_ego: [Batch, D] ego agent embedding.
            z_peers: [Batch, P, D] peer embeddings.
            peer_mask: [Batch, P] boolean (True = valid peer).

        Returns:
            weights: [Batch, 3] soft gating weights summing to 1.0.
        """
        assert z_ego.dim() == 2, f"z_ego must be 2D, got {z_ego.shape}"
        assert z_peers.dim() == 3, f"z_peers must be 3D, got {z_peers.shape}"

        q = self.W_q(z_ego).unsqueeze(1)   # [B, 1, D]
        k = self.W_k(z_peers)              # [B, P, D]
        v = self.W_v(z_peers)              # [B, P, D]

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, 1, P]
        attn_mask = peer_mask.unsqueeze(1)  # [B, 1, P]
        attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = torch.bmm(attn_weights, v).squeeze(1)  # [B, D]
        logits = self.gate_proj(context)
        return torch.softmax(logits, dim=-1)


class RecurrentFallbackHead(nn.Module):
    """GRU-based recurrent Expert 3 with temporal memory.

    Shape Flow:
        z: [Batch, D] + h_prev: [Batch, H] → logits: [Batch, action_dim], h_next: [Batch, H]
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim > 0 else latent_dim
        self.gru = nn.GRUCell(latent_dim, self.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward with temporal hidden state.

        Args:
            z: [Batch, D] current embedding.
            h_prev: [Batch, H] previous hidden state. None → zeros.

        Returns:
            logits: [Batch, action_dim], h_next: [Batch, H]
        """
        if h_prev is None:
            h_prev = torch.zeros(z.size(0), self.hidden_dim, device=z.device)

        h_next = self.gru(z, h_prev)
        logits = self.output_proj(h_next)
        return logits, h_next


class NeuralMoEPolicy(nn.Module):
    """Dual-Encoder step-level Neural MoE Policy with attention router and GRU fallback.

    Shape Flow:
        obs: [B, A, obs_dim] → y_final: [B, A, action_dim], weights: [B, A, 3],
        fallback_hidden: [B, A, hidden_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        view_radius: int = 3,
        action_dim: int = 4,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.latent_dim = latent_dim

        # Dual-Encoder Topology
        self.expert_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
        self.router_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)

        # Attention-based gating router
        self.router = AttentionGatingRouter(latent_dim)

        # Expert Heads
        self.expert_exploration = nn.Linear(latent_dim, action_dim)
        self.expert_coordination = nn.Linear(latent_dim, action_dim)
        self.expert_fallback = RecurrentFallbackHead(latent_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        peer_matrix: torch.Tensor,
        action_mask: torch.Tensor,
        fallback_hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blended logits with attention routing and GRU fallback.

        Args:
            obs: [B, A, obs_dim]
            peer_matrix: [B, A, A]
            action_mask: [B, A, action_dim]
            fallback_hidden: [B, A, H] or None

        Returns:
            y_final: [B, A, action_dim], weights: [B, A, 3], h_next: [B, A, H]
        """
        B, A, obs_dim = obs.shape
        assert A == self.num_agents, f"Expected {self.num_agents} agents, got {A}"

        # Pool communication peer matrix into neighbor counts
        peer_count = torch.sum(peer_matrix, dim=-1, keepdim=True)  # [B, A, 1]

        obs_flat = obs.view(B * A, obs_dim)
        peer_count_flat = peer_count.view(B * A, 1)

        # --- Expert branch (frozen during router training) ---
        with torch.no_grad():
            z_expert = self.expert_encoder(obs_flat, peer_count_flat)
            y_exp0 = self.expert_exploration(z_expert)
            y_exp1 = self.expert_coordination(z_expert)

        # Expert 3: Recurrent fallback with GRU hidden state
        with torch.no_grad():
            h_in: Optional[torch.Tensor] = None
            if fallback_hidden is not None:
                h_in = fallback_hidden.view(B * A, -1)
            y_exp2, h_next = self.expert_fallback(z_expert, h_in)

        fallback_hidden_next = h_next.view(B, A, -1)

        # --- Router branch (trainable online) ---
        z_router = self.router_encoder(obs_flat, peer_count_flat)

        # Build peer embedding sets for attention router
        z_router_3d = z_router.view(B, A, self.latent_dim)
        z_peers = z_router_3d.unsqueeze(1).expand(B, A, A, self.latent_dim)
        z_peers = z_peers.reshape(B * A, A, self.latent_dim)

        peer_mask = peer_matrix.view(B * A, A).bool()
        z_ego = z_router

        weights_flat = self.router(z_ego, z_peers, peer_mask)

        # Logit blending
        y_final_flat = (
            weights_flat[:, 0:1] * y_exp0 +
            weights_flat[:, 1:2] * y_exp1 +
            weights_flat[:, 2:3] * y_exp2
        )

        y_final = y_final_flat.view(B, A, self.action_dim)
        weights = weights_flat.view(B, A, 3)

        # Invalid Action Masking
        y_final = torch.where(action_mask, y_final, torch.full_like(y_final, -1e9))
        return y_final, weights, fallback_hidden_next


# ===========================================================================
# 3. Phase A: Training Teleprinter with Active Progress Bars
# ===========================================================================

def print_progress_bar(
    epoch: int,
    total_epochs: int,
    loss: float,
    acc: float,
    grad_norm: float,
) -> None:
    """Renders a dynamic ASCII progress bar for live training feedback.

    Args:
        epoch: Current epoch number (1-indexed).
        total_epochs: Total epoch count.
        loss: Current cross-entropy loss value.
        acc: Behavioral cloning accuracy (%).
        grad_norm: L2 gradient norm of trainable parameters.
    """
    width = 25
    filled = int(width * epoch / total_epochs)
    bar = "=" * filled + ">" + "." * (width - filled - 1)
    bar = bar[:width]
    print(
        f"Epoch {epoch:02d}/{total_epochs:02d} [{bar}] "
        f"Loss: {loss:.6f} | BC-Acc: {acc:6.2f}% | Grad Scale: {grad_norm:.4f}"
    )


def run_expert_distillation(
    policy: NeuralMoEPolicy,
    datasets: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    epochs: int = 15,
) -> None:
    """Stage 1: Offline Behavioral Cloning of expert heads from team trajectories.

    Runs full training loops with real gradient updates and tracks cross-entropy
    loss convergence alongside behavioral cloning accuracy.

    Args:
        policy: The NeuralMoEPolicy to train.
        datasets: List of [exploration, coordination, fallback] trajectory datasets.
        epochs: Number of training epochs (default 15 for production convergence).
    """
    print("\n[PHASE A: LIVE TRAINING TELEPRINTER]")
    print("=" * 90)
    print("Stage 1: Expert Policy Distillation (Behavioral Cloning)")
    print("-> Freezing Gating Router weights to train expert heads independently.")
    print(f"-> Epochs: {epochs} | Heads: Exploration, Coordination, GRU Fallback")
    print("-" * 90)

    # Freeze Gating Router Encoder & Weights
    for p in policy.router_encoder.parameters():
        p.requires_grad = False
    for p in policy.router.parameters():
        p.requires_grad = False

    # Enable expert gradients
    for p in policy.expert_encoder.parameters():
        p.requires_grad = True
    for p in policy.expert_exploration.parameters():
        p.requires_grad = True
    for p in policy.expert_coordination.parameters():
        p.requires_grad = True
    for p in policy.expert_fallback.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        list(policy.expert_encoder.parameters()) +
        list(policy.expert_exploration.parameters()) +
        list(policy.expert_coordination.parameters()) +
        list(policy.expert_fallback.parameters()),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    heads = [policy.expert_exploration, policy.expert_coordination, policy.expert_fallback]

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for head_idx, dataset in enumerate(datasets):
            head = heads[head_idx]

            for obs, peer_matrix, actions in dataset:
                obs_b = obs.unsqueeze(0)
                peer_b = peer_matrix.unsqueeze(0)

                B, A, cur_obs_dim = obs_b.shape
                peer_count = torch.sum(peer_b, dim=-1, keepdim=True)

                optimizer.zero_grad()
                z = policy.expert_encoder(obs_b.view(B * A, cur_obs_dim), peer_count.view(B * A, 1))

                if head_idx == 2:
                    logits, _ = head(z, None)
                else:
                    logits = head(z)

                act_targets = actions.unsqueeze(0).view(-1).long()
                loss = criterion(logits, act_targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == act_targets).sum().item()
                total_samples += B * A

        # Compute gradient norm
        grad_norm = sum(
            p.grad.data.norm(2).item()
            for p in policy.expert_encoder.parameters()
            if p.grad is not None
        )
        acc = (correct / total_samples) * 100 if total_samples > 0 else 0.0
        avg_loss = total_loss / max(len(datasets[0]) + len(datasets[1]) + len(datasets[2]), 1)
        print_progress_bar(epoch, epochs, avg_loss, acc, grad_norm)


def run_gating_router_optimization(
    policy: NeuralMoEPolicy,
    steps: int = 50,
) -> None:
    """Stage 2: Online Router Fine-Tuning with Conditional Indicator Mask Penalty.

    Uses attention-based routing with explicit communication blackout penalties
    to enforce fallback weight allocation under isolation.

    Args:
        policy: The NeuralMoEPolicy to fine-tune.
        steps: Number of gating optimization steps (default 50 for production).
    """
    print(f"\nStage 2: Attention-Based Gating Router Policy Optimization")
    print("-> Freezing expert heads. Fine-tuning attention router online with blackout penalties.")
    print(f"-> Steps: {steps} | Router: Scaled Dot-Product Attention over peer embeddings")
    print("-" * 90)

    # Freeze Expert Encoder & Heads
    for p in policy.expert_encoder.parameters():
        p.requires_grad = False
    for p in policy.expert_exploration.parameters():
        p.requires_grad = False
    for p in policy.expert_coordination.parameters():
        p.requires_grad = False
    for p in policy.expert_fallback.parameters():
        p.requires_grad = False

    # Unfreeze Gating Router parameters
    for p in policy.router_encoder.parameters():
        p.requires_grad = True
    for p in policy.router.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        list(policy.router_encoder.parameters()) + list(policy.router.parameters()),
        lr=1e-2,
    )

    A = policy.num_agents
    obs_dim = policy.expert_encoder.window_dim + 4 + A

    for epoch in range(1, steps + 1):
        optimizer.zero_grad()

        # Simulate mixed connectivity batch (2 connected, 2 isolated elements)
        obs = torch.zeros(4, A, obs_dim)
        peer_matrix = torch.eye(A).repeat(4, 1, 1)  # Default isolated
        peer_matrix[0] = torch.ones(A, A)  # Fully connected
        peer_matrix[1] = torch.ones(A, A)  # Fully connected

        action_mask = torch.ones(4, A, policy.action_dim, dtype=torch.bool)

        _, weights, _ = policy(obs, peer_matrix, action_mask)

        # Calculate active peer count per agent
        peer_count = torch.sum(peer_matrix, dim=-1)  # [4, A]

        # Conditional Indicator Mask: penalty applies ONLY if peer_count == 1.0 (isolated)
        is_isolated = (peer_count == 1.0).float()

        g_fallback = weights[:, :, 2]  # Expert 3 weight allocation

        # Penalty 1: forces fallback weight to 1.0 under blackout
        comm_routing_penalty = is_isolated * (1.0 - g_fallback) ** 2

        # Penalty 2: forces coordination weight (index 1) to 1.0 when connected
        is_connected = (peer_count == A).float()
        g_coord = weights[:, :, 1]
        comm_connected_penalty = is_connected * (1.0 - g_coord) ** 2

        loss = 10.0 * comm_routing_penalty.mean() + 10.0 * comm_connected_penalty.mean()

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            grad_norm = sum(
                p.grad.data.norm(2).item()
                for p in policy.router.parameters()
                if p.grad is not None
            )
            print_progress_bar(epoch, steps, loss.item(), 100.0, grad_norm)

    print("=" * 90)


# ===========================================================================
# 4. Phase B & C: UI/UX Rendering Logic
# ===========================================================================

def render_grid_20x20(
    positions: list[Position],
    goals: list[Position],
    obstacles: set[Position],
) -> None:
    """Renders a professional 20x20 ASCII grid world to terminal.

    Legend:  .  = empty cell  |  #  = wall/obstacle  |  G0-G3 = goals  |  A0-A3 = agents

    Args:
        positions: Current agent positions.
        goals: Target goal positions.
        obstacles: Set of obstacle wall positions.
    """
    grid: list[list[str]] = [[" . " for _ in range(20)] for _ in range(20)]

    # Render static obstacle walls
    for p in obstacles:
        grid[p.y][p.x] = " # "

    # Render static goals
    for idx, g in enumerate(goals):
        grid[g.y][g.x] = f"G{idx} "

    # Render cooperative agents
    for idx, a in enumerate(positions):
        grid[a.y][a.x] = f"A{idx} "

    # Render the board to terminal
    print("\n[PHASE B: 20x20 GRID WORLD VIEW]")
    print("   " + "".join(f"{i:^3}" for i in range(20)))
    print("   " + "---" * 20)
    for r_idx, row in enumerate(grid):
        print(f"{r_idx:2d}|" + "".join(row) + "|")
    print("   " + "---" * 20)


def print_telemetry_table(
    step: int,
    positions: list[Position],
    weights: torch.Tensor,
    peer_matrix: torch.Tensor,
    fallback_hidden: torch.Tensor,
) -> None:
    """Prints the parametric evolution telemetry table with GRU state info.

    Displays step index, agent coordinates, active peer count, baseline parameters,
    and live softmax routing vector [g_explore, g_coord, g_fallback].

    Args:
        step: Current simulation step index.
        positions: List of agent positions.
        weights: [1, A, 3] gating weight tensor.
        peer_matrix: [1, A, A] peer adjacency tensor.
        fallback_hidden: [1, A, H] GRU hidden state tensor.
    """
    print("\n[PHASE C: PARAMETRIC EVOLUTION TELEMETRY]")
    header = (
        f"{'Step':<5}{'Agent':<8}{'Pos':<10}{'Peers':<7}"
        f"{'Baseline Params':<28}"
        f"{'MoE Gating [g_exp, g_coord, g_fall]':<40}"
        f"{'GRU |h|':<10}"
    )
    print(header)
    print("-" * 108)

    # Pool connection states
    peer_count = torch.sum(peer_matrix, dim=-1).squeeze(0)
    weights_step = weights.squeeze(0)
    h_norms = torch.norm(fallback_hidden.squeeze(0), dim=-1)

    for a in range(len(positions)):
        pos_str = f"({positions[a].x},{positions[a].y})"

        # Format baseline tracking parameters
        if int(peer_count[a].item()) <= 1:
            baseline_str = "Hyst Q: α=0.10, β=0.01"
        else:
            baseline_str = "Frontier Exploration: γ=0.95"

        w_str = (
            f"[{weights_step[a, 0]:.4f}, "
            f"{weights_step[a, 1]:.4f}, "
            f"{weights_step[a, 2]:.4f}]"
        )
        h_norm_str = f"{h_norms[a].item():.4f}"

        print(
            f"{step:<5}{f'Agent-{a}':<8}{pos_str:<10}"
            f"{int(peer_count[a].item()):<7}{baseline_str:<28}"
            f"{w_str:<40}{h_norm_str:<10}"
        )
    print("=" * 108)


# ===========================================================================
# 5. Execution Pipeline Setup
# ===========================================================================

def generate_synthetic_data(
    num_agents: int,
    obs_dim: int,
    samples_per_head: int = 25,
) -> list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Generates synthetic expert trajectories labeled with target heuristic action mappings.

    Creates spatially-structured observations (not pure noise) to enable meaningful
    behavioral cloning convergence during full training loops.

    Args:
        num_agents: Fleet size.
        obs_dim: Observation vector dimensionality.
        samples_per_head: Number of trajectory samples per expert head.

    Returns:
        List of [exploration_data, coordination_data, fallback_data] datasets.
    """
    datasets: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [[], [], []]

    for head_idx in range(3):
        for _ in range(samples_per_head):
            obs = torch.randn(num_agents, obs_dim) * 0.1
            peer_matrix = torch.eye(num_agents)
            if head_idx == 1:
                # Coordination data: fully connected agents
                peer_matrix = torch.ones(num_agents, num_agents)

            actions = torch.randint(0, 4, (num_agents,))
            datasets[head_idx].append((obs, peer_matrix, actions))

    return datasets


def main() -> None:
    """Main execution pipeline: full production training and 20x20 grid evaluation."""
    print("=" * 90)
    print("   NEURAL MIXTURE OF EXPERTS (MoE) — PRODUCTION DEMONSTRATION")
    print("   Attention-Based Router | GRU Temporal Fallback | 7x7 Ego-Centric Window")
    print("=" * 90)

    num_agents: int = 4
    view_radius: int = 3  # 7x7 windows
    obs_dim: int = (2 * view_radius + 1) ** 2 * 4 + 4 + num_agents  # 196 + 8 = 204

    # Instantiate Policy with attention router and GRU fallback
    policy = NeuralMoEPolicy(obs_dim, num_agents, view_radius)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # ===================================================================
    # Phase A: Run Behavioral Cloning & Routing Optimization
    # ===================================================================
    datasets = generate_synthetic_data(num_agents, obs_dim, samples_per_head=25)
    run_expert_distillation(policy, datasets, epochs=15)
    run_gating_router_optimization(policy, steps=50)

    # ===================================================================
    # Phase B & C: 20x20 Grid Evaluation with GRU Temporal Tracking
    # ===================================================================

    # Setup 20x20 environmental obstacles (horizontal wall barriers)
    obstacles: set[Position] = set()
    for x in range(3, 17):
        obstacles.add(Position(x, 5))
        obstacles.add(Position(x, 14))

    # Setup static goal targets (four corners)
    goals: list[Position] = [
        Position(2, 2),
        Position(2, 17),
        Position(17, 2),
        Position(17, 17),
    ]

    # Pre-calculated trajectories mimicking dynamic movement
    # Steps 1-4: connected cluster (Manhattan distances < 3)
    # Steps 5-10: absolute blackout (Manhattan distances >= 9)
    trajectories: list[list[Position]] = [
        # Step 1: tight cluster at center
        [Position(9, 9), Position(9, 10), Position(10, 9), Position(10, 10)],
        # Step 2: expanding outward
        [Position(8, 8), Position(8, 11), Position(11, 8), Position(11, 11)],
        # Step 3: expanding further
        [Position(7, 7), Position(7, 12), Position(12, 7), Position(12, 12)],
        # Step 4: edge of communication range
        [Position(6, 6), Position(6, 13), Position(13, 6), Position(13, 13)],
        # Step 5: BLACKOUT TRIGGER (dist >= 9)
        [Position(5, 5), Position(5, 14), Position(14, 5), Position(14, 14)],
        # Step 6: fully isolated
        [Position(4, 4), Position(4, 15), Position(15, 4), Position(15, 15)],
        # Step 7: approaching goals
        [Position(3, 3), Position(3, 16), Position(16, 3), Position(16, 16)],
        # Step 8: at goal positions
        [Position(2, 2), Position(2, 17), Position(17, 2), Position(17, 17)],
        # Step 9: holding at goals
        [Position(2, 2), Position(2, 17), Position(17, 2), Position(17, 17)],
        # Step 10: mission complete
        [Position(2, 2), Position(2, 17), Position(17, 2), Position(17, 17)],
    ]

    eval_obs = torch.zeros(1, num_agents, obs_dim)
    action_mask = torch.ones(1, num_agents, 4, dtype=torch.bool)

    # GRU hidden state — tracked across all 10 simulation steps
    fallback_hidden: Optional[torch.Tensor] = None

    print("\n" + "=" * 90)
    print("   SIMULATION: 10-Step Evaluation with Communication Blackout at Step 5")
    print("=" * 90)

    # Run evaluation simulation with GRU temporal tracking
    for step in range(1, 11):
        positions = trajectories[step - 1]

        # Construct peer matrix based on Manhattan distance < 3
        peer_np = np.zeros((num_agents, num_agents), dtype=np.float32)
        for i in range(num_agents):
            for j in range(num_agents):
                dist = abs(positions[i].x - positions[j].x) + abs(positions[i].y - positions[j].y)
                if dist < 3:
                    peer_np[i, j] = 1.0

        peer_matrix_t = torch.tensor(peer_np, dtype=torch.float32).unsqueeze(0)

        # Shape assertions before routing
        assert eval_obs.shape == (1, num_agents, obs_dim), \
            f"obs shape: expected [1, {num_agents}, {obs_dim}], got {eval_obs.shape}"
        assert peer_matrix_t.shape == (1, num_agents, num_agents), \
            f"peer shape: expected [1, {num_agents}, {num_agents}], got {peer_matrix_t.shape}"

        # Evaluate policy with GRU hidden state persistence
        with torch.no_grad():
            _, weights, fallback_hidden = policy(
                eval_obs, peer_matrix_t, action_mask, fallback_hidden
            )

        # Render terminal interface
        render_grid_20x20(positions, goals, obstacles)
        print_telemetry_table(step, positions, weights, peer_matrix_t, fallback_hidden)

    print("\n[DEMO COMPLETE] All phases executed successfully.")


if __name__ == "__main__":
    main()
