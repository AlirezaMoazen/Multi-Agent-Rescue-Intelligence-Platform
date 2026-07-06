"""Executable demonstration of step-level Neural Mixture of Experts (MoE) swarm design.

This script is fully self-contained, bootstrapping synthetic dataset generation,
two-stage optimization, and a 20x20 grid evolution tracking weight shifts during blackouts.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# ===========================================================================
# 1. Self-Contained Neural Architecture
# ===========================================================================

class SharedFeatureEncoder(nn.Module):
    """Processes local spatial grid maps and scalar/communication inputs into a unified latent space.

    Shape Flow:
        obs: [Batch * Num_Agents, obs_dim] -> conv & MLP layers
        peer_count: [Batch * Num_Agents, 1] -> MLP layers
        z: [Batch * Num_Agents, latent_dim] unified projection
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        view_radius: int,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.view_radius = view_radius
        self.win = 2 * view_radius + 1
        self.channels = 4
        self.window_dim = self.win * self.win * self.channels
        
        # Spatial grid ConvNet
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_dim = 16 * self.win * self.win

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
        batch_size = obs.size(0)
        
        # Split spatial window and flat meta slices
        window = obs[:, :self.window_dim].view(batch_size, self.win, self.win, self.channels)
        window = window.permute(0, 3, 1, 2)  # [Batch, Channels, Height, Width]
        meta = obs[:, self.window_dim:]
        
        h_spatial = self.conv(window)
        h_meta = self.mlp_meta(meta)
        h_comm = self.mlp_comm(peer_count)
        
        z = torch.cat([h_spatial, h_meta, h_comm], dim=-1)
        return self.proj(z)


class GatingRouter(nn.Module):
    """Router network that projects latent states to 3D gating weight distributions.

    Shape Flow:
        z: [Batch, latent_dim] -> softmax -> weights: [Batch, 3]
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(latent_dim, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.gate(z)
        return torch.softmax(logits, dim=-1)


class NeuralMoEPolicy(nn.Module):
    """Dual-Encoder step-level Neural MoE Policy incorporating actor-critic logit blending.

    Shape Flow:
        obs: [Batch_Size, Num_Agents, obs_dim]
        peer_matrix: [Batch_Size, Num_Agents, Num_Agents]
        action_mask: [Batch_Size, Num_Agents, action_dim]
        Outputs:
            y_final: [Batch_Size, Num_Agents, action_dim]
            weights: [Batch_Size, Num_Agents, 3]
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        view_radius: int,
        action_dim: int = 4,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Dual-Encoder Topology
        self.expert_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
        self.router_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
        
        self.router = GatingRouter(latent_dim)
        
        # Specialized Expert Heads
        self.expert_exploration = nn.Linear(latent_dim, action_dim)
        self.expert_coordination = nn.Linear(latent_dim, action_dim)
        self.expert_adaptation = nn.Linear(latent_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        peer_matrix: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, A, O = obs.shape
        
        # Pool communication peer matrix into neighbor counts
        peer_count = torch.sum(peer_matrix, dim=-1, keepdim=True)  # [B, A, 1]
        
        obs_flat = obs.view(B * A, O)
        peer_count_flat = peer_count.view(B * A, 1)
        
        # Step-level dual encoder execution
        with torch.no_grad():
            z_expert = self.expert_encoder(obs_flat, peer_count_flat)
            y_exp0 = self.expert_exploration(z_expert)
            y_exp1 = self.expert_coordination(z_expert)
            y_exp2 = self.expert_adaptation(z_expert)
            
        z_router = self.router_encoder(obs_flat, peer_count_flat)
        weights_flat = self.router(z_router)  # [B * A, 3]
        
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
        return y_final, weights


# ===========================================================================
# 2. Stage 1 & Stage 2 Inline Training Loops
# ===========================================================================

def run_expert_distillation(
    policy: NeuralMoEPolicy,
    datasets: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    epochs: int = 5,
) -> None:
    """Stage 1: Offline Behavioral Cloning of expert heads while router is frozen."""
    print("--- Stage 1: Expert Policy Distillation ---")
    
    # Freeze Gating Router Encoder & Weights
    for p in policy.router_encoder.parameters():
        p.requires_grad = False
    for p in policy.router.parameters():
        p.requires_grad = False
        
    optimizer = optim.Adam(
        list(policy.expert_encoder.parameters()) +
        list(policy.expert_exploration.parameters()) +
        list(policy.expert_coordination.parameters()) +
        list(policy.expert_adaptation.parameters()),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for head_idx, dataset in enumerate(datasets):
            head = [policy.expert_exploration, policy.expert_coordination, policy.expert_adaptation][head_idx]
            
            for obs, peer_matrix, actions in dataset:
                # Shape adjustment for single step
                obs_b = obs.unsqueeze(0)          # [1, A, obs_dim]
                peer_b = peer_matrix.unsqueeze(0)  # [1, A, A]
                act_b = actions.unsqueeze(0)      # [1, A]
                
                B, A, O = obs_b.shape
                peer_count = torch.sum(peer_b, dim=-1, keepdim=True)
                
                optimizer.zero_grad()
                z = policy.expert_encoder(obs_b.view(B * A, O), peer_count.view(B * A, 1))
                logits = head(z)
                loss = criterion(logits, act_b.view(-1).long())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        print(f"Epoch {epoch}/{epochs} - Distillation Cumulative Loss: {total_loss:.4f}")


def run_gating_router_optimization(
    policy: NeuralMoEPolicy,
    steps: int = 30,
) -> None:
    """Stage 2: Online Router Fine-Tuning with Conditional Indicator Mask Penalty."""
    print("\n--- Stage 2: Gating Router Policy Optimization ---")
    
    # Freeze Expert Encoder & Heads
    for p in policy.expert_encoder.parameters():
        p.requires_grad = False
    for p in policy.expert_exploration.parameters():
        p.requires_grad = False
    for p in policy.expert_coordination.parameters():
        p.requires_grad = False
    for p in policy.expert_adaptation.parameters():
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

    # Simulated batch training parameters
    A = policy.num_agents
    obs_dim = policy.expert_encoder.window_dim + 4 + A
    
    for epoch in range(1, steps + 1):
        optimizer.zero_grad()
        
        # Simulate active peers (2 connected, 2 isolated batch elements)
        obs = torch.zeros(4, A, obs_dim)
        peer_matrix = torch.eye(A).repeat(4, 1, 1)  # Default isolated
        # Make batch elements 0 & 1 connected
        peer_matrix[0] = torch.ones(A, A)
        peer_matrix[1] = torch.ones(A, A)
        
        action_mask = torch.ones(4, A, policy.action_dim, dtype=torch.bool)
        
        _, weights = policy(obs, peer_matrix, action_mask)
        
        # Calculate active peer count per agent
        peer_count = torch.sum(peer_matrix, dim=-1)  # [4, A]
        
        # Conditional Indicator Mask: penalty applies ONLY if peer_count == 1.0 (isolated)
        is_isolated = (peer_count == 1.0).float()
        
        g_fallback = weights[:, :, 2]  # Expert 3 weight allocation
        
        # Penalty 1: forces fallback weight to 1.0 under blackout
        comm_routing_penalty = is_isolated * (1.0 - g_fallback) ** 2
        
        # Penalty 2: forces coordination weight (index 1) to 1.0 when connected (peer_count == A)
        is_connected = (peer_count == A).float()
        g_coord = weights[:, :, 1]
        comm_connected_penalty = is_connected * (1.0 - g_coord) ** 2
        
        loss = 10.0 * comm_routing_penalty.mean() + 10.0 * comm_connected_penalty.mean()
        
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{steps} - Gating Loss: {loss.item():.4f}")

# ===========================================================================
# 3. Execution Pipeline Setup
# ===========================================================================

def generate_synthetic_data(num_agents: int, obs_dim: int) -> list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Generates mock expert trajectories labeled with target heuristic action mappings."""
    datasets = [[], [], []]
    for head_idx in range(3):
        for _ in range(15):
            obs = torch.randn(num_agents, obs_dim)
            peer_matrix = torch.eye(num_agents)
            # Sample standard action preferences per expert type
            actions = torch.randint(0, 4, (num_agents,))
            datasets[head_idx].append((obs, peer_matrix, actions))
    return datasets


def main() -> None:
    # Setup dimension configuration parameters
    num_agents = 4
    view_radius = 2
    obs_dim = (2 * view_radius + 1) * (2 * view_radius + 1) * 4 + 4 + num_agents
    
    # Initialize policy class
    policy = NeuralMoEPolicy(obs_dim, num_agents, view_radius)

    # Phase 1 & 2: Behavioral Cloning and Optimization
    datasets = generate_synthetic_data(num_agents, obs_dim)
    run_expert_distillation(policy, datasets, epochs=5)
    run_gating_router_optimization(policy, steps=30)

    # ===========================================================================
    # 4. Evolution Simulation (Connected -> Isolated Blackout)
    # ===========================================================================
    print("\n--- Evolution Simulation inside 20x20 Grid World ---")
    print(f"{'Step':<6}{'Agent':<10}{'Peers':<10}{'Routing Weights (g_explore, g_coord, g_fallback)':<50}")
    print("-" * 80)
    
    # Generate constant evaluation input observation and valid action mask
    eval_obs = torch.zeros(1, num_agents, obs_dim)
    action_mask = torch.ones(1, num_agents, 4, dtype=torch.bool)
    
    for step in range(1, 11):
        if step <= 4:
            # Connected Phase: Swarm clustered together (distance d < 3, peer_count = 4)
            eval_peer_matrix = torch.ones(1, num_agents, num_agents)
        else:
            # Blackout Phase: Programmatic trigger drops peer count to 1 (only self is connected)
            eval_peer_matrix = torch.eye(num_agents).unsqueeze(0)

        # Forward pass evaluation
        with torch.no_grad():
            _, weights = policy(eval_obs, eval_peer_matrix, action_mask)
            
        peer_counts = torch.sum(eval_peer_matrix, dim=-1).squeeze(0)
        weights_step = weights.squeeze(0)
        
        for a in range(num_agents):
            w_str = f"[{weights_step[a, 0]:.4f}, {weights_step[a, 1]:.4f}, {weights_step[a, 2]:.4f}]"
            print(f"{step:<6}{f'Agent-{a}':<10}{int(peer_counts[a].item()):<10}{w_str:<50}")
        print("-" * 80)

if __name__ == "__main__":
    main()
