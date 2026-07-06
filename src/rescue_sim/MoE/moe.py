from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SharedFeatureEncoder(nn.Module):
    """Processes local spatial grid maps and scalar/communication inputs into a unified latent space.

    Processes a 7x7 ego-centric spatial window (representing a 3-block visibility radius
    in all directions) alongside peer connection inputs and metadata.
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
        
        # Slice spatial window and reshape to [Batch, Channels, Height, Width]
        window = obs[:, :self.window_dim].view(batch_size, self.win, self.win, self.channels)
        window = window.permute(0, 3, 1, 2)  # [Batch, Channels, Height, Width]
        
        # Slice meta features and agent ID
        meta = obs[:, self.window_dim:]
        
        # Forward pass on spatial and meta layers
        h_spatial = self.conv(window)  # [Batch, conv_output_dim]
        h_meta = self.mlp_meta(meta)    # [Batch, 32]
        h_comm = self.mlp_comm(peer_count)  # [Batch, 16]
        
        # Concatenate representation and project
        z = torch.cat([h_spatial, h_meta, h_comm], dim=-1)      # [Batch, conv_output_dim + 32 + 16]
        return self.proj(z)                                      # [Batch, latent_dim]


class GatingRouter(nn.Module):
    """Router network that outputs gating weight distributions over the three experts.

    Takes the latent embedding from the trainable router encoder and maps it to
    a 3-dimensional probability vector.
    """

    def __init__(self, latent_dim: int) -> None:
        """Initializes the gating projection layers."""
        super().__init__()
        self.gate = nn.Linear(latent_dim, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass to yield gating weights.

        Args:
            z: [Batch, latent_dim] router representation.

        Returns:
            weights: [Batch, 3] soft gating weights summing to 1.0.
        """
        logits = self.gate(z)
        return torch.softmax(logits, dim=-1)


class NeuralMoEPolicy(nn.Module):
    """Step-level Neural Mixture of Experts (MoE) Policy with Dual-Encoder topology.

    Uses distinct feature encoders to isolate learning between the frozen expert
    heads (distilled offline) and the gating router fine-tuned online.
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        view_radius: int = 3,
        action_dim: int = 4,
        latent_dim: int = 128,
    ) -> None:
        """Initializes encoders, the router, and three expert heads.

        Args:
            obs_dim: Size of the flat observations vector.
            num_agents: Swarm agent size.
            view_radius: Sensor range radius (default 3, 7x7 egocentric window).
            action_dim: Total actions in policy distribution.
            latent_dim: Layer size for shared space.
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Dual-Encoder Topology
        self.expert_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
        self.router_encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
        
        # Gating router network
        self.router = GatingRouter(latent_dim)
        
        # Expert Heads: output unnormalized directional policy logits
        self.expert_exploration = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.expert_coordination = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.expert_adaptation = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        peer_matrix: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates blended logits and gating assignments over 3D inputs.

        Args:
            obs: [Batch_Size, Num_Agents, obs_dim] flat observations.
            peer_matrix: [Batch_Size, Num_Agents, Num_Agents] adjacency peer matrix.
            action_mask: [Batch_Size, Num_Agents, action_dim] boolean action mask.

        Returns:
            y_final: [Batch_Size, Num_Agents, action_dim] blended action logits with invalid action masking.
            weights: [Batch_Size, Num_Agents, 3] expert allocation weights.
        """
        B, A, obs_dim = obs.shape
        assert A == self.num_agents, f"Expected {self.num_agents} agents, got {A}"
        
        # Standardize peer matrix to get pooled permutation-invariant peer count
        peer_count = torch.sum(peer_matrix, dim=-1, keepdim=True)  # [B, A, 1]
        
        # Flatten first two dimensions for spatial encoder processing
        obs_flat = obs.view(B * A, obs_dim)
        peer_count_flat = peer_count.view(B * A, 1)
        
        # Feature extraction from the frozen expert brain (no-grad offline)
        with torch.no_grad():
            z_expert = self.expert_encoder(obs_flat, peer_count_flat)
            y_exp0 = self.expert_exploration(z_expert)
            y_exp1 = self.expert_coordination(z_expert)
            y_exp2 = self.expert_adaptation(z_expert)
            
        # Feature extraction for gating router (online trainable)
        z_router = self.router_encoder(obs_flat, peer_count_flat)
        weights_flat = self.router(z_router)  # [B * A, 3]
        
        # Actor-Critic Logit Blending: y_final = sum(g_j * y_j)
        y_final_flat = (
            weights_flat[:, 0:1] * y_exp0 +
            weights_flat[:, 1:2] * y_exp1 +
            weights_flat[:, 2:3] * y_exp2
        )  # [B * A, action_dim]
        
        # Reshape back to 3D layouts
        y_final = y_final_flat.view(B, A, self.action_dim)
        weights = weights_flat.view(B, A, 3)
        
        # Apply invalid action mask (large negative value for illegal moves)
        y_final = torch.where(action_mask, y_final, torch.full_like(y_final, -1e9))
        
        return y_final, weights

    def get_action(
        self,
        obs: torch.Tensor,
        peer_matrix: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample actions under valid-action masking.

        Args:
            obs: [Batch_Size, Num_Agents, obs_dim] observations.
            peer_matrix: [Batch_Size, Num_Agents, Num_Agents] peer adjacency link matrix.
            action_mask: [Batch_Size, Num_Agents, action_dim] boolean valid action masks.

        Returns:
            actions: [Batch_Size, Num_Agents] sampled action indices.
        """
        y_final, _ = self.forward(obs, peer_matrix, action_mask)
        probs = torch.softmax(y_final, dim=-1)
        
        B, A, AD = probs.shape
        # Flatten to sample
        probs_flat = probs.view(B * A, AD)
        sampled = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        return sampled.view(B, A)


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
        loss_history: List of average losses per expert head over epochs.
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
    for p in policy.expert_adaptation.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        list(policy.expert_encoder.parameters()) +
        list(policy.expert_exploration.parameters()) +
        list(policy.expert_coordination.parameters()) +
        list(policy.expert_adaptation.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    loss_history: dict[str, list[float]] = {"exploration": [], "coordination": [], "fallback": []}

    for epoch in range(epochs):
        for idx, dataset in enumerate(expert_datasets):
            if not dataset:
                continue
            
            expert_name = ["exploration", "coordination", "fallback"][idx]
            head = [policy.expert_exploration, policy.expert_coordination, policy.expert_adaptation][idx]
            
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
                
                B, A, obs_dim = obs_batch.shape
                peer_count = torch.sum(peer_matrix_batch, dim=-1, keepdim=True)  # [B, A, 1]
                
                obs_flat = obs_batch.view(B * A, obs_dim)
                peer_count_flat = peer_count.view(B * A, 1)
                act_flat = act_batch.view(B * A).long()
                
                optimizer.zero_grad()
                z = policy.expert_encoder(obs_flat, peer_count_flat)
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
    env,
    updates: int = 50,
    lr: float = 1e-3,
    comm_penalty_coef: float = 5.0,
    gamma: float = 0.99,
) -> list[float]:
    """Fine-tunes the gating router online using a policy gradient algorithm.

    Enforces fallback routing weights (g_fallback approx 1.0) under blackout.

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
    for p in policy.expert_adaptation.parameters():
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
        obs = env.reset()
        done = False
        
        saved_log_probs: list[torch.Tensor] = []
        saved_rewards: list[float] = []
        saved_penalties: list[torch.Tensor] = []
        
        while not done:
            num_agents = env.num_agents
            positions = env.positions
            
            # Reconstruct the dynamic peer link adjacency mask (d < 3 threshold)
            peer_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
            for i in range(num_agents):
                for j in range(num_agents):
                    dist = abs(positions[i].x - positions[j].x) + abs(positions[i].y - positions[j].y)
                    if dist < 3:
                        peer_matrix[i, j] = 1.0
                        
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)             # [1, A, obs_dim]
            peer_matrix_t = torch.tensor(peer_matrix, dtype=torch.float32).unsqueeze(0)  # [1, A, A]
            act_mask_t = torch.tensor(env.valid_action_mask(), dtype=torch.bool).unsqueeze(0)  # [1, A, action_dim]
            
            y_final, weights = policy(obs_t, peer_matrix_t, act_mask_t)
            probs = torch.softmax(y_final, dim=-1)
            
            m = torch.distributions.Categorical(probs)
            actions = m.sample()  # [1, A]
            
            log_prob = m.log_prob(actions)  # [1, A]
            
            next_obs, reward, done, _ = env.step(actions.squeeze(0).numpy())
            
            # Communication-routing blackout penalty via Conditional Indicator Mask
            peer_count = torch.sum(peer_matrix_t, dim=-1)  # [1, A]
            is_isolated = (peer_count == 1.0).float()  # [1, A]
            
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
        policy_loss = []
        for log_prob, G, penalty in zip(saved_log_probs, discounted_returns, saved_penalties):
            policy_loss.append((-log_prob.mean() * G) + comm_penalty_coef * penalty.mean())
            
        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
    return loss_history
