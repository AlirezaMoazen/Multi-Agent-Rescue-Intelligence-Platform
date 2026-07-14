# Copyright 2026 TUHH Group 05 — A. Herrero Callejo, C. Marcos Alonso,
# M. M. Orfany, A. Moazzen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Step-level Neural Mixture of Experts (MoE) policy with Dual-Encoder topology.

Architecture highlights:
- **AttentionGatingRouter**: Scaled dot-product attention over peer latent embeddings
  replaces the rigid MLP gate, enabling dynamic routing based on real-time spatial
  proximity without zero-padding or index-ordering failures.
- **RecurrentFallbackHead**: GRU-based Expert 3 maintains temporal hidden state h_t
  across simulation steps, allowing isolated agents to track historical trajectory
  and avoid infinite blind looping under communication blackout.
- **SharedFeatureEncoder**: CNN + MLP encoder with 7×7 ego-centric local observation
  window (3-block visibility radius), channels-last upstream format.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SharedFeatureEncoder(nn.Module):
    """Processes local spatial grid maps and scalar/communication inputs into a unified latent space.

    Processes a 7x7 ego-centric spatial window (representing a 3-block visibility radius
    in all directions) alongside peer connection inputs and metadata.

    Shape Flow:
        obs: [Batch, obs_dim] → z: [Batch, latent_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        view_radius: int = 3,
        latent_dim: int = 128,
    ) -> None:
        """Initializes the encoder layers.

        Args:
            obs_dim: Dimension of the flattened agent observation vector.
            num_agents: Swarm fleet size.
            view_radius: Perception radius (default 3, producing 7x7 windows).
            latent_dim: Target embedding size.
        """
        super().__init__()
        self.num_agents = num_agents
        self.view_radius = view_radius
        self.win = 2 * view_radius + 1  # 7
        self.channels = 4
        self.window_dim = self.win * self.win * self.channels  # 7 * 7 * 4 = 196

        # Spatial CNN for 7x7 local grid maps
        # Input shape: [Batch, Channels, Height, Width]
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_dim = 32 * self.win * self.win  # 32 * 49 = 1568

        # MLP for meta features (pos, steps, targets) and agent ID
        # Input shape: [Batch, 4 + num_agents]
        self.mlp_meta = nn.Sequential(
            nn.Linear(4 + num_agents, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # MLP for permutation-invariant peer count
        # Input shape: [Batch, 1]
        self.mlp_comm = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        # Unified projection head
        self.proj = nn.Sequential(
            nn.Linear(conv_output_dim + 32 + 16, latent_dim),
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
        # Upstream serialization (RescueEnv._agent_obs) stacks 4 channels via
        # np.stack([blocked, target_a, target_b, other], axis=-1).reshape(-1),
        # producing a flattened channels-LAST layout: [H, W, C] → [H*W*C].
        # We reconstruct the spatial tensor and convert to channels-first for
        # Conv2d in a single contiguous operation.
        window = (
            obs[:, :self.window_dim]
            .reshape(batch_size, self.win, self.win, self.channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [Batch, C=4, H=win, W=win]

        # Slice meta features (scalars + agent one-hot ID)
        meta = obs[:, self.window_dim:]

        # Forward pass on spatial and meta layers
        h_spatial = self.conv(window)       # [Batch, conv_output_dim]
        h_meta = self.mlp_meta(meta)        # [Batch, 32]
        h_comm = self.mlp_comm(peer_count)  # [Batch, 16]

        # Concatenate representation and project
        z = torch.cat([h_spatial, h_meta, h_comm], dim=-1)  # [Batch, conv_output_dim + 32 + 16]
        return self.proj(z)                                  # [Batch, latent_dim]


class AttentionGatingRouter(nn.Module):
    """Attention-based router that dynamically allocates expert weights using scaled
    dot-product attention over peer latent embeddings.

    The ego agent's embedding is projected into Query space; all visible peer
    embeddings (including self) are projected into Key and Value spaces.  The
    attended context is projected to 3-dimensional gating logits → softmax →
    routing weights.

    This eliminates rigid MLP gating and zero-padding artifacts — variable-size
    peer sets are handled natively via attention masking.

    Shape Flow:
        z_ego:   [Batch, latent_dim]
        z_peers: [Batch, max_peers, latent_dim]
        peer_mask: [Batch, max_peers] (True = valid peer)
        → weights: [Batch, 3]
    """

    def __init__(self, latent_dim: int) -> None:
        """Initializes the attention-based gating projection layers.

        Args:
            latent_dim: Dimensionality of the latent embeddings from the encoder.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = math.sqrt(latent_dim)

        # Q/K/V projections for single-head scaled dot-product attention
        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.W_v = nn.Linear(latent_dim, latent_dim)

        # Output projection: attended context concatenated with the ego
        # embedding (skip connection) → 3 expert gating logits. The context
        # alone is a convex mix of peer embeddings, which dilutes per-agent
        # signals (e.g. "a target is in MY window") — the skip keeps them.
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(
        self,
        z_ego: torch.Tensor,
        z_peers: torch.Tensor,
        peer_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes gating weights via scaled dot-product attention over peers.

        Args:
            z_ego: [Batch, latent_dim] ego agent latent embedding (query source).
            z_peers: [Batch, max_peers, latent_dim] peer latent embeddings (key/value).
            peer_mask: [Batch, max_peers] boolean mask (True = valid peer, False = pad).

        Returns:
            weights: [Batch, 3] soft gating weights summing to 1.0.
        """
        assert z_ego.dim() == 2, f"z_ego must be 2D [B, D], got {z_ego.shape}"
        assert z_peers.dim() == 3, f"z_peers must be 3D [B, P, D], got {z_peers.shape}"
        assert peer_mask.dim() == 2, f"peer_mask must be 2D [B, P], got {peer_mask.shape}"

        # Project ego to query, peers to key/value
        q = self.W_q(z_ego).unsqueeze(1)   # [B, 1, D]
        k = self.W_k(z_peers)              # [B, P, D]
        v = self.W_v(z_peers)              # [B, P, D]

        # Scaled dot-product attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, 1, P]

        # Apply mask: set invalid peer positions to -inf before softmax
        attn_mask = peer_mask.unsqueeze(1)  # [B, 1, P]
        attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, P]

        # Handle fully-masked rows (all peers invalid → NaN after softmax)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Attended context + ego skip connection
        context = torch.bmm(attn_weights, v).squeeze(1)  # [B, D]

        # Project to gating logits
        logits = self.gate_proj(torch.cat([context, z_ego], dim=-1))  # [B, 3]
        return torch.softmax(logits, dim=-1)


class RecurrentFallbackHead(nn.Module):
    """GRU-based recurrent Expert 3 (Fallback/Adaptation Head).

    Maintains temporal hidden state h_t across simulation steps, enabling
    isolated agents to track their historical trajectory under communication
    blackout and prevent infinite blind looping in dead-ends.

    Shape Flow:
        z: [Batch, latent_dim] + h_prev: [Batch, hidden_dim]
        → logits: [Batch, action_dim], h_next: [Batch, hidden_dim]
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 0) -> None:
        """Initializes the GRU cell and output projection.

        Args:
            latent_dim: Input embedding dimensionality.
            action_dim: Number of discrete actions.
            hidden_dim: GRU hidden state size. Defaults to latent_dim if 0.
        """
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim > 0 else latent_dim
        self.gru = nn.GRUCell(latent_dim, self.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with temporal memory.

        Args:
            z: [Batch, latent_dim] current step embedding.
            h_prev: [Batch, hidden_dim] previous hidden state. None → zeros.

        Returns:
            logits: [Batch, action_dim] action logits.
            h_next: [Batch, hidden_dim] updated hidden state for next step.
        """
        if h_prev is None:
            h_prev = torch.zeros(z.size(0), self.hidden_dim, device=z.device)

        assert z.dim() == 2, f"z must be 2D [B, D], got {z.shape}"
        assert h_prev.shape == (z.size(0), self.hidden_dim), (
            f"h_prev shape mismatch: expected [{z.size(0)}, {self.hidden_dim}], got {h_prev.shape}"
        )

        h_next = self.gru(z, h_prev)               # [Batch, hidden_dim]
        logits = self.output_proj(h_next)            # [Batch, action_dim]
        return logits, h_next


class NeuralMoEPolicy(nn.Module):
    """Step-level Neural Mixture of Experts (MoE) Policy with Dual-Encoder topology.

    Uses distinct feature encoders to isolate learning between the frozen expert
    heads (distilled offline) and the gating router fine-tuned online.

    Architecture:
        - Expert Encoder (frozen): feeds Expert 1 (exploration), Expert 2 (coordination),
          and Expert 3 (GRU fallback)
        - Router Encoder (trainable): feeds AttentionGatingRouter
        - Logit blending: y_final = sum(g_j * y_j) with action masking

    Shape Flow:
        obs: [B, A, obs_dim]
        peer_matrix: [B, A, A]
        action_mask: [B, A, action_dim]
        → y_final: [B, A, action_dim], weights: [B, A, 3], fallback_hidden: [B, A, hidden_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        view_radius: int = 3,
        action_dim: int = 4,
        latent_dim: int = 128,
        coord_hidden: int = 64,
    ) -> None:
        """Initializes encoders, the attention router, and three expert heads.

        Args:
            obs_dim: Size of the flat observations vector.
            num_agents: Swarm agent size.
            view_radius: Sensor range radius (default 3, 7x7 egocentric window).
            action_dim: Total actions in policy distribution.
            latent_dim: Layer size for shared space.
            coord_hidden: Hidden width of the coordination head (Expert 2).
                The default 64 matches historical checkpoints; wider heads give
                the distilled deep-RL student more imitation capacity.
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        self.coord_hidden = coord_hidden
        # Gate sharpening temperature: 1.0 = soft blend; <1 pushes routing
        # toward winner-take-all so the MoE acts like its best expert per
        # state instead of a logit compromise. Set by outcome-based router
        # training; persisted by save_moe_policy.
        self.gate_tau = 1.0
        # Optional per-expert routing bias [3] (log-space), set at rollout time
        # by the online scoreboard adaptation. None = no bias. Not persisted.
        self.route_bias: Optional[torch.Tensor] = None

        # Dual-Encoder Topology
        self.expert_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
        self.router_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)

        # Attention-based gating router
        self.router = AttentionGatingRouter(latent_dim)

        # Expert Head 1: Exploration (feed-forward)
        self.expert_exploration = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        # Expert Head 2: Coordination distilled from QMIX/MAPPO (feed-forward)
        self.expert_coordination = nn.Sequential(
            nn.Linear(latent_dim, coord_hidden),
            nn.ReLU(),
            nn.Linear(coord_hidden, action_dim),
        )

        # Expert Head 3: Recurrent fallback with GRU temporal memory
        self.expert_fallback = RecurrentFallbackHead(latent_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        peer_matrix: torch.Tensor,
        action_mask: torch.Tensor,
        fallback_hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates blended logits and gating assignments over 3D inputs.

        Args:
            obs: [B, A, obs_dim] flat observations.
            peer_matrix: [B, A, A] adjacency peer matrix.
            action_mask: [B, A, action_dim] boolean action mask.
            fallback_hidden: [B, A, hidden_dim] GRU hidden state from previous step.
                None at episode start (will initialize to zeros).

        Returns:
            y_final: [B, A, action_dim] blended action logits with invalid action masking.
            weights: [B, A, 3] expert allocation weights.
            fallback_hidden_next: [B, A, hidden_dim] updated GRU hidden state.
        """
        B, A, obs_dim = obs.shape
        assert A == self.num_agents, f"Expected {self.num_agents} agents, got {A}"
        assert peer_matrix.shape == (B, A, A), (
            f"peer_matrix shape mismatch: expected [{B}, {A}, {A}], got {peer_matrix.shape}"
        )
        assert action_mask.shape == (B, A, self.action_dim), (
            f"action_mask shape mismatch: expected [{B}, {A}, {self.action_dim}], got {action_mask.shape}"
        )

        # Standardize peer matrix to get pooled permutation-invariant peer count
        peer_count = torch.sum(peer_matrix, dim=-1, keepdim=True)  # [B, A, 1]

        # Flatten first two dimensions for spatial encoder processing
        obs_flat = obs.view(B * A, obs_dim)
        peer_count_flat = peer_count.view(B * A, 1)

        # --- Expert branch (frozen during router training) ---
        with torch.no_grad():
            z_expert = self.expert_encoder(obs_flat, peer_count_flat)  # [B*A, D]
            y_exp0 = self.expert_exploration(z_expert)                  # [B*A, action_dim]
            y_exp1 = self.expert_coordination(z_expert)                 # [B*A, action_dim]

        # Expert 3: Recurrent fallback with GRU hidden state
        with torch.no_grad():
            h_in: Optional[torch.Tensor] = None
            if fallback_hidden is not None:
                h_in = fallback_hidden.view(B * A, -1)
            y_exp2, h_next = self.expert_fallback(z_expert, h_in)  # [B*A, action_dim], [B*A, H]

        fallback_hidden_next = h_next.view(B, A, -1)

        # --- Router branch (trainable online) ---
        z_router = self.router_encoder(obs_flat, peer_count_flat)  # [B*A, D]

        # Build peer embedding sets for the attention router
        # z_router reshaped to [B, A, D] so we can index peer sets
        z_router_3d = z_router.view(B, A, self.latent_dim)

        # For each agent, peers are all agents in the fleet (masked by peer_matrix)
        # z_peers: [B*A, A, D] — each agent sees all A potential peers
        z_peers = z_router_3d.unsqueeze(1).expand(B, A, A, self.latent_dim)
        z_peers = z_peers.reshape(B * A, A, self.latent_dim)

        # peer_mask: [B*A, A] — True where peer_matrix has a connection
        peer_mask = peer_matrix.view(B * A, A).bool()

        # z_ego: [B*A, D]
        z_ego = z_router  # already [B*A, D]

        weights_flat = self.router(z_ego, z_peers, peer_mask)  # [B*A, 3]

        # Optional gate sharpening (near winner-take-all routing).
        if self.gate_tau != 1.0:
            weights_flat = torch.softmax(
                torch.log(weights_flat.clamp_min(1e-8)) / self.gate_tau, dim=-1
            )

        # Online scoreboard adaptation: bias routing toward experts that are
        # actually delivering rescues on the current grid (rollout-only).
        if self.route_bias is not None:
            biased = weights_flat * torch.exp(self.route_bias.to(weights_flat.device))
            weights_flat = biased / biased.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Actor-Critic Logit Blending: y_final = sum(g_j * y_j)
        y_final_flat = (
            weights_flat[:, 0:1] * y_exp0 +
            weights_flat[:, 1:2] * y_exp1 +
            weights_flat[:, 2:3] * y_exp2
        )  # [B*A, action_dim]

        # Reshape back to 3D layouts
        y_final = y_final_flat.view(B, A, self.action_dim)
        weights = weights_flat.view(B, A, 3)

        # Apply invalid action mask (large negative value for illegal moves)
        y_final = torch.where(action_mask, y_final, torch.full_like(y_final, -1e9))

        return y_final, weights, fallback_hidden_next

    def get_action(
        self,
        obs: torch.Tensor,
        peer_matrix: torch.Tensor,
        action_mask: torch.Tensor,
        fallback_hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions under valid-action masking.

        Args:
            obs: [B, A, obs_dim] observations.
            peer_matrix: [B, A, A] peer adjacency link matrix.
            action_mask: [B, A, action_dim] boolean valid action masks.
            fallback_hidden: [B, A, hidden_dim] GRU hidden state. None at episode start.

        Returns:
            actions: [B, A] sampled action indices.
            fallback_hidden_next: [B, A, hidden_dim] updated hidden state.
        """
        y_final, _, h_next = self.forward(obs, peer_matrix, action_mask, fallback_hidden)
        probs = torch.softmax(y_final, dim=-1)

        B, A, AD = probs.shape
        # Flatten to sample
        probs_flat = probs.view(B * A, AD)
        sampled = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        return sampled.view(B, A), h_next


def distill_expert_heads(
    policy: NeuralMoEPolicy,
    expert_datasets: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, list[float]]:
    """Trains expert heads and the expert encoder from offline team trajectories.

    Keeps the gating router network completely locked.

    Args:
        policy: The Neural MoE policy instance.
        expert_datasets: Trajectory dataset list: [exploration_data, coordination_data, fallback_data].
            Each element contains (obs_team, peer_matrix_team, actions_team) tuples.
        epochs: Training epochs.
        batch_size: Mini-batch dimensions.
        lr: Adam learning rate.

    Returns:
        loss_history: Dict of average losses per expert head over epochs.
    """
    # Freeze Gating Router parameters
    for p in policy.router_encoder.parameters():
        p.requires_grad = False
    for p in policy.router.parameters():
        p.requires_grad = False

    # Enable gradients for expert parameters
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
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    loss_history: dict[str, list[float]] = {"exploration": [], "coordination": [], "fallback": []}

    heads = [policy.expert_exploration, policy.expert_coordination, policy.expert_fallback]

    for epoch in range(epochs):
        for idx, dataset in enumerate(expert_datasets):
            if not dataset:
                continue

            expert_name = ["exploration", "coordination", "fallback"][idx]
            head = heads[idx]

            epoch_loss = 0.0
            num_batches = 0

            indices = list(range(len(dataset)))
            random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i: i + batch_size]
                batch = [dataset[bi] for bi in batch_idx]

                obs_batch = torch.stack([b[0] for b in batch])          # [B, A, obs_dim]
                peer_matrix_batch = torch.stack([b[1] for b in batch])  # [B, A, A]
                act_batch = torch.stack([b[2] for b in batch])          # [B, A]

                B, A, cur_obs_dim = obs_batch.shape
                peer_count = torch.sum(peer_matrix_batch, dim=-1, keepdim=True)  # [B, A, 1]

                obs_flat = obs_batch.view(B * A, cur_obs_dim)
                peer_count_flat = peer_count.view(B * A, 1)
                act_flat = act_batch.view(B * A).long()

                optimizer.zero_grad()
                z = policy.expert_encoder(obs_flat, peer_count_flat)

                # For the fallback head (GRU), pass None hidden state during distillation
                if idx == 2:
                    logits, _ = head(z, None)
                else:
                    logits = head(z)

                loss = criterion(logits, act_flat)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            loss_history[expert_name].append(avg_loss)

    return loss_history


def train_gating_router(
    policy: NeuralMoEPolicy,
    env: object,
    updates: int = 50,
    lr: float = 1e-3,
    comm_penalty_coef: float = 5.0,
    gamma: float = 0.99,
) -> list[float]:
    """Fine-tunes the gating router online using a policy gradient algorithm.

    Enforces fallback routing weights (g_fallback ≈ 1.0) under blackout.
    Tracks GRU hidden state across episode steps for temporal consistency.

    Args:
        policy: Neural MoE network.
        env: Simulation environment supporting gym-style interface and valid_action_mask().
        updates: Policy gradient iteration steps.
        lr: Adam learning rate.
        comm_penalty_coef: Penalty scale for routing off Expert 3 during blackouts.
        gamma: Returns discount rate.

    Returns:
        losses: Policy gradient losses per training step.
    """
    # Freeze Expert parameters
    for p in policy.expert_encoder.parameters():
        p.requires_grad = False
    for p in policy.expert_exploration.parameters():
        p.requires_grad = False
    for p in policy.expert_coordination.parameters():
        p.requires_grad = False
    for p in policy.expert_fallback.parameters():
        p.requires_grad = False

    # Enable router gradients
    for p in policy.router_encoder.parameters():
        p.requires_grad = True
    for p in policy.router.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        list(policy.router_encoder.parameters()) + list(policy.router.parameters()),
        lr=lr,
    )

    loss_history: list[float] = []

    for update in range(updates):
        obs = env.reset()  # type: ignore[union-attr]
        done = False

        saved_log_probs: list[torch.Tensor] = []
        saved_rewards: list[float] = []
        saved_penalties: list[torch.Tensor] = []

        # Initialize GRU hidden state at episode start
        fallback_hidden: Optional[torch.Tensor] = None

        while not done:
            num_agents = env.num_agents  # type: ignore[union-attr]
            positions = env.positions  # type: ignore[union-attr]

            # Reconstruct the dynamic peer link adjacency mask (d < 3 threshold)
            peer_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
            for i in range(num_agents):
                for j in range(num_agents):
                    dist = abs(positions[i].x - positions[j].x) + abs(positions[i].y - positions[j].y)
                    if dist < 3:
                        peer_matrix[i, j] = 1.0

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)                          # [1, A, obs_dim]
            peer_matrix_t = torch.tensor(peer_matrix, dtype=torch.float32).unsqueeze(0)           # [1, A, A]
            act_mask_t = torch.tensor(env.valid_action_mask(), dtype=torch.bool).unsqueeze(0)  # type: ignore[union-attr]

            y_final, weights, fallback_hidden = policy(obs_t, peer_matrix_t, act_mask_t, fallback_hidden)

            # Detach hidden state from computation graph for next step
            fallback_hidden = fallback_hidden.detach()

            probs = torch.softmax(y_final, dim=-1)

            m = torch.distributions.Categorical(probs)
            actions = m.sample()  # [1, A]

            log_prob = m.log_prob(actions)  # [1, A]

            next_obs, reward, done, _ = env.step(actions.squeeze(0).numpy())  # type: ignore[union-attr]

            # Communication-routing blackout penalty via Conditional Indicator Mask
            peer_count_step = torch.sum(peer_matrix_t, dim=-1)  # [1, A]
            is_isolated = (peer_count_step == 1.0).float()  # [1, A]

            g_fallback = weights[:, :, 2]  # [1, A]
            step_penalty = is_isolated * (1.0 - g_fallback) ** 2  # [1, A]

            saved_log_probs.append(log_prob)
            saved_rewards.append(reward)
            saved_penalties.append(step_penalty)

            obs = next_obs

        # Calculate discounted returns
        discounted_returns: list[float] = []
        R = 0.0
        for r in reversed(saved_rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)

        # Compute policy gradient update
        optimizer.zero_grad()
        policy_loss: list[torch.Tensor] = []
        for log_prob, G, penalty in zip(saved_log_probs, discounted_returns, saved_penalties):
            policy_loss.append((-log_prob.mean() * G) + comm_penalty_coef * penalty.mean())

        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())

    return loss_history
