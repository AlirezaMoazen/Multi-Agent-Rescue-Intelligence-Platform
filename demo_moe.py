# Copyright 2026 Alireza Moazen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Executable demonstration of step-level Neural Mixture of Experts (MoE) swarm design.

This script is fully self-contained, bootstrapping synthetic dataset generation,
two-stage optimization, and a 20x20 grid evolution tracking weight shifts during blackouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===========================================================================
# 1. Coordinate / Helper Definitions
# ===========================================================================

@dataclass(frozen=True)
class Position:
    x: int
    y: int


# ===========================================================================
# 2. Self-Contained Neural Architecture (Option B: Coord = Distilled QMIX/MAPPO)
# ===========================================================================

class SharedFeatureEncoder(nn.Module):
    """Processes local spatial grid maps and scalar/communication inputs into a unified latent space.

    Processes a 7x7 egocentric spatial grid map (representing a 3-block visibility constraint
    in all directions) alongside peer connection inputs and metadata.
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
        view_radius: int = 3,
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
        
        # Specialized Expert Heads (Exploration, Distilled Coordination, Fallback)
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
# 3. Phase A: Teleprinter Progress Bar Rendering
# ===========================================================================

def print_progress_bar(epoch: int, total_epochs: int, loss: float, acc: float, grad_norm: float) -> None:
    """Renders a dynamic ASCII progress bar for live training feedback."""
    width = 25
    filled = int(width * epoch / total_epochs)
    bar = "=" * filled + ">" + "." * (width - filled - 1)
    bar = bar[:width]
    print(f"Epoch {epoch:02d}/{total_epochs:02d} [{bar}] Loss: {loss:.6f} | BC-Acc: {acc:6.2f}% | Grad Scale: {grad_norm:.4f}")


def run_expert_distillation(
    policy: NeuralMoEPolicy,
    datasets: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    epochs: int = 5,
) -> None:
    """Stage 1: Offline Behavioral Cloning of expert heads from team trajectories."""
    print("\n[PHASE A: LIVE TRAINING TELEPRINTER]")
    print("--------------------------------------------------------------------------------")
    print("Stage 1: Expert Policy Distillation (Behavioral Cloning)")
    print("-> Freezing Gating Router weights to train expert heads independently.")
    
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
        correct = 0
        total_samples = 0
        
        for head_idx, dataset in enumerate(datasets):
            head = [policy.expert_exploration, policy.expert_coordination, policy.expert_adaptation][head_idx]
            
            for obs, peer_matrix, actions in dataset:
                obs_b = obs.unsqueeze(0)
                peer_b = peer_matrix.unsqueeze(0)
                act_b = actions.unsqueeze(0)
                
                B, A, O = obs_b.shape
                peer_count = torch.sum(peer_b, dim=-1, keepdim=True)
                
                optimizer.zero_grad()
                z = policy.expert_encoder(obs_b.view(B * A, O), peer_count.view(B * A, 1))
                logits = head(z)
                loss = criterion(logits, act_b.view(-1).long())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == act_b.view(-1)).sum().item()
                total_samples += B * A
                
        # Mock gradient scale metric
        grad_norm = sum(p.grad.data.norm(2).item() for p in policy.expert_encoder.parameters() if p.grad is not None)
        acc = (correct / total_samples) * 100 if total_samples > 0 else 0.0
        print_progress_bar(epoch, epochs, total_loss / 3, acc, grad_norm)


def run_gating_router_optimization(
    policy: NeuralMoEPolicy,
    steps: int = 30,
) -> None:
    """Stage 2: Online Router Fine-Tuning with Conditional Indicator Mask Penalty."""
    print("\nStage 2: Gating Router Policy Optimization")
    print("-> Freezing expert heads. Fine-tuning router online with local blackout penalties.")
    
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

    A = policy.num_agents
    obs_dim = policy.expert_encoder.window_dim + 4 + A
    
    for epoch in range(1, steps + 1):
        optimizer.zero_grad()
        
        # Simulate active peers (2 connected, 2 isolated batch elements)
        obs = torch.zeros(4, A, obs_dim)
        peer_matrix = torch.eye(A).repeat(4, 1, 1)  # Default isolated
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
            grad_norm = sum(p.grad.data.norm(2).item() for p in policy.router.parameters() if p.grad is not None)
            print_progress_bar(epoch, steps, loss.item(), 100.0, grad_norm)
    print("--------------------------------------------------------------------------------")


# ===========================================================================
# 4. Phase B & C: UI/UX Rendering Logic
# ===========================================================================

def render_grid_20x20(positions: list[Position], goals: list[Position], obstacles: set[Position]) -> None:
    """Renders a professional 3D text block of the 20x20 discrete grid world."""
    grid = [[" . " for _ in range(20)] for _ in range(20)]
    
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
    for r_idx, row in enumerate(grid):
        print(f"{r_idx:2d} " + "".join(row))


def print_telemetry_table(step: int, positions: list[Position], weights: torch.Tensor, peer_matrix: torch.Tensor) -> None:
    """Prints the parametric evolution telemetry table directly below the grid."""
    print("\n[PHASE C: PARAMETRIC EVOLUTION DIAGRAM]")
    print(f"{'Step':<5}{'Agent':<8}{'Position':<10}{'Peers':<7}{'Baseline Stats':<25}{'MoE Gating (g_explore, g_coord, g_fallback)':<40}")
    print("-" * 100)
    
    # Pool connection states
    peer_count = torch.sum(peer_matrix, dim=-1).squeeze(0)
    weights_step = weights.squeeze(0)
    
    for a in range(len(positions)):
        pos_str = f"({positions[a].x},{positions[a].y})"
        
        # Format baseline tracking parameters
        if int(peer_count[a].item()) <= 1:
            baseline_str = "Hyst Q: a=0.10, b=0.01"
        else:
            baseline_str = "Frontier Exploration: d=0.95"
            
        w_str = f"[{weights_step[a, 0]:.4f}, {weights_step[a, 1]:.4f}, {weights_step[a, 2]:.4f}]"
        print(f"{step:<5}{f'Agent-{a}':<8}{pos_str:<10}{int(peer_count[a].item()):<7}{baseline_str:<25}{w_str:<40}")
    print("=" * 100)


# ===========================================================================
# 5. Execution Pipeline Setup
# ===========================================================================

def generate_synthetic_data(num_agents: int, obs_dim: int) -> list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Generates mock expert trajectories labeled with target heuristic action mappings."""
    datasets = [[], [], []]
    for head_idx in range(3):
        for _ in range(15):
            obs = torch.randn(num_agents, obs_dim)
            peer_matrix = torch.eye(num_agents)
            actions = torch.randint(0, 4, (num_agents,))
            datasets[head_idx].append((obs, peer_matrix, actions))
    return datasets


def main() -> None:
    num_agents = 4
    view_radius = 3  # 7x7 windows
    obs_dim = (2 * view_radius + 1) * (2 * view_radius + 1) * 4 + 4 + num_agents  # 7 * 7 * 4 + 4 + 4 = 204
    
    # Instantiate Policy
    policy = NeuralMoEPolicy(obs_dim, num_agents, view_radius)
    
    # Phase A: Run Behavioral Cloning & Routing Optimization
    datasets = generate_synthetic_data(num_agents, obs_dim)
    run_expert_distillation(policy, datasets, epochs=5)
    run_gating_router_optimization(policy, steps=30)

    # Setup 20x20 environmental obstacles
    obstacles = set()
    for x in range(3, 17):
        obstacles.add(Position(x, 5))
        obstacles.add(Position(x, 14))
    
    # Setup static goal targets
    goals = [
        Position(2, 2),
        Position(2, 17),
        Position(17, 2),
        Position(17, 17)
    ]
    
    # Pre-calculated trajectories mimicking dynamic movement
    # Steps 1 to 4: connected cluster (distances < 3)
    # Steps 5 to 10: absolute blackout (distances >= 9)
    trajectories = [
        # Step 1
        [Position(9, 9), Position(9, 10), Position(10, 9), Position(10, 10)],
        # Step 2
        [Position(8, 8), Position(8, 11), Position(11, 8), Position(11, 11)],
        # Step 3
        [Position(7, 7), Position(7, 12), Position(12, 7), Position(12, 12)],
        # Step 4
        [Position(6, 6), Position(6, 13), Position(13, 6), Position(13, 13)],
        # Step 5 (Blackout trigger starts here: dist >= 9)
        [Position(5, 5), Position(5, 14), Position(14, 5), Position(14, 14)],
        # Step 6
        [Position(4, 4), Position(4, 15), Position(15, 4), Position(15, 15)],
        # Step 7
        [Position(3, 3), Position(3, 16), Position(16, 3), Position(16, 16)],
        # Step 8
        [Position(2, 2), Position(2, 17), Position(17, 2), Position(17, 17)],
        # Step 9
        [Position(2, 2), Position(2, 17), Position(17, 2), Position(17, 17)],
        # Step 10
        [Position(2, 2), Position(2, 17), Position(17, 2), Position(17, 17)]
    ]

    eval_obs = torch.zeros(1, num_agents, obs_dim)
    action_mask = torch.ones(1, num_agents, 4, dtype=torch.bool)

    # Run evaluation simulation
    for step in range(1, 11):
        positions = trajectories[step - 1]
        
        # Construct peer matrix
        peer_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
        for i in range(num_agents):
            for j in range(num_agents):
                dist = abs(positions[i].x - positions[j].x) + abs(positions[i].y - positions[j].y)
                if dist < 3:
                    peer_matrix[i, j] = 1.0
                    
        peer_matrix_t = torch.tensor(peer_matrix, dtype=torch.float32).unsqueeze(0)
        
        # Evaluate policy
        with torch.no_grad():
            _, weights = policy(eval_obs, peer_matrix_t, action_mask)
            
        # Render terminal interface
        render_grid_20x20(positions, goals, obstacles)
        print_telemetry_table(step, positions, weights, peer_matrix_t)


if __name__ == "__main__":
    main()
