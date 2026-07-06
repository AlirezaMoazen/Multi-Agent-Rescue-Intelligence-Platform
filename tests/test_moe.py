"""Tests for the step-level Neural Mixture of Experts (MoE) Policy."""

from __future__ import annotations

import pytest
import numpy as np
import torch

pytest.importorskip("torch")

from rescue_sim.MoE.moe import (
    GatingRouter,
    NeuralMoEPolicy,
    SharedFeatureEncoder,
    distill_expert_heads,
    train_gating_router,
)
from rescue_sim.shared import Position


class MockEnv:
    """Mock environment with gym-style interface and blackout state support."""
    def __init__(self, num_agents: int = 2, obs_dim: int = 108):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.positions = [Position(0, 0) for _ in range(num_agents)]
        self.max_steps = 5
        self._steps = 0
        self.n_actions = 4

    def reset(self) -> np.ndarray:
        self._steps = 0
        # Force blackout at reset (dist >= 3)
        self.positions = [Position(0, 0), Position(5, 5)]
        return np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self._steps += 1
        done = self._steps >= self.max_steps
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        # Keep positions isolated (blackout)
        self.positions = [Position(0, 0), Position(10, 10)]
        return obs, 1.0, done, {}

    def valid_action_mask(self) -> np.ndarray:
        return np.ones((self.num_agents, self.n_actions), dtype=bool)


def test_shared_feature_encoder() -> None:
    batch_size = 4
    num_agents = 3
    view_radius = 2
    obs_dim = (2 * view_radius + 1) * (2 * view_radius + 1) * 4 + 4 + num_agents
    latent_dim = 64

    encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
    
    # Encoder expects 2D inputs when processing flattened agent batches
    obs = torch.randn(batch_size * num_agents, obs_dim)
    peer_count = torch.ones(batch_size * num_agents, 1)

    z = encoder(obs, peer_count)
    assert z.shape == (batch_size * num_agents, latent_dim)


def test_permutation_invariant_comm_pooling() -> None:
    num_agents = 3
    view_radius = 2
    obs_dim = (2 * view_radius + 1) * (2 * view_radius + 1) * 4 + 4 + num_agents
    latent_dim = 64

    encoder = SharedFeatureEncoder(obs_dim, num_agents, view_radius, latent_dim)
    
    obs = torch.randn(1, obs_dim)
    
    # Peer count is sum-pooled, so identical neighbor count gives identical outputs
    peer_count1 = torch.tensor([[2.0]], dtype=torch.float32)
    peer_count2 = torch.tensor([[2.0]], dtype=torch.float32)

    z1 = encoder(obs, peer_count1)
    z2 = encoder(obs, peer_count2)
    
    assert torch.allclose(z1, z2, atol=1e-5)


def test_gating_router() -> None:
    latent_dim = 64
    router = GatingRouter(latent_dim)
    z = torch.randn(5, latent_dim)
    weights = router(z)

    assert weights.shape == (5, 3)
    assert torch.allclose(torch.sum(weights, dim=-1), torch.ones(5), atol=1e-5)
    assert torch.all(weights >= 0.0)


def test_neural_moe_policy_forward_and_action() -> None:
    batch_size = 3
    num_agents = 2
    view_radius = 2
    obs_dim = (2 * view_radius + 1) * (2 * view_radius + 1) * 4 + 4 + num_agents
    action_dim = 4

    policy = NeuralMoEPolicy(obs_dim, num_agents, view_radius, action_dim)
    
    # 3D inputs: [Batch_Size, Num_Agents, ...]
    obs = torch.randn(batch_size, num_agents, obs_dim)
    peer_matrix = torch.ones(batch_size, num_agents, num_agents)
    action_mask = torch.ones(batch_size, num_agents, action_dim, dtype=torch.bool)

    y_final, weights = policy(obs, peer_matrix, action_mask)
    assert y_final.shape == (batch_size, num_agents, action_dim)
    assert weights.shape == (batch_size, num_agents, 3)

    # Action selection with custom masking
    custom_mask = torch.tensor([[[True, False, False, False], [False, True, False, False]],
                                [[False, False, True, False], [False, False, False, True]],
                                [[True, False, False, False], [False, True, False, False]]], dtype=torch.bool)
    actions = policy.get_action(obs, peer_matrix, custom_mask)
    assert actions.shape == (batch_size, num_agents)
    assert actions[0, 0].item() == 0
    assert actions[0, 1].item() == 1
    assert actions[1, 0].item() == 2
    assert actions[1, 1].item() == 3


def test_distill_expert_heads() -> None:
    num_agents = 2
    view_radius = 2
    obs_dim = (2 * view_radius + 1) * (2 * view_radius + 1) * 4 + 4 + num_agents
    action_dim = 4

    policy = NeuralMoEPolicy(obs_dim, num_agents, view_radius, action_dim)

    # Build tiny datasets of team transitions: (obs_team, peer_matrix_team, actions_team)
    # obs_team: [num_agents, obs_dim]
    # peer_matrix_team: [num_agents, num_agents]
    # actions_team: [num_agents]
    exploration_data = [(torch.randn(num_agents, obs_dim), torch.ones(num_agents, num_agents), torch.zeros(num_agents)) for _ in range(10)]
    coordination_data = [(torch.randn(num_agents, obs_dim), torch.ones(num_agents, num_agents), torch.ones(num_agents)) for _ in range(10)]
    fallback_data = [(torch.randn(num_agents, obs_dim), torch.ones(num_agents, num_agents), torch.ones(num_agents) * 2) for _ in range(10)]

    loss_history = distill_expert_heads(
        policy,
        [exploration_data, coordination_data, fallback_data],
        epochs=3,
        batch_size=2,
        lr=0.01,
    )

    assert "exploration" in loss_history
    assert "coordination" in loss_history
    assert "fallback" in loss_history
    assert len(loss_history["exploration"]) == 3
    assert loss_history["exploration"][-1] >= 0.0

    # Ensure Gating Router parameters are frozen (grad is False)
    for p in policy.router_encoder.parameters():
        assert not p.requires_grad
    for p in policy.router.parameters():
        assert not p.requires_grad


def test_train_gating_router_penalty() -> None:
    num_agents = 2
    view_radius = 2
    obs_dim = (2 * view_radius + 1) * (2 * view_radius + 1) * 4 + 4 + num_agents
    action_dim = 4

    policy = NeuralMoEPolicy(obs_dim, num_agents, view_radius, action_dim)
    env = MockEnv(num_agents, obs_dim)

    # Train router on the mock environment (isolated agents)
    losses = train_gating_router(
        policy,
        env,
        updates=5,
        lr=0.01,
        comm_penalty_coef=20.0,
    )

    assert len(losses) == 5

    # Gating Router parameters should now have gradients enabled
    for p in policy.router_encoder.parameters():
        assert p.requires_grad
    for p in policy.router.parameters():
        assert p.requires_grad

    # Expert parameters should remain frozen
    for p in policy.expert_encoder.parameters():
        assert not p.requires_grad

    # Verify that in a blackout, the policy assigns high weight to the fallback head (Expert 3)
    obs = torch.zeros(1, num_agents, obs_dim)
    # peer_matrix for isolated agent (only self connection, so identity matrix)
    peer_matrix = torch.eye(num_agents).unsqueeze(0)  # [1, A, A]
    action_mask = torch.ones(1, num_agents, action_dim, dtype=torch.bool)
    
    _, weights = policy(obs, peer_matrix, action_mask)
    
    # Due to penalty during training, Expert 3 (index 2) weight should be significantly positive
    assert weights[0, 0, 2].item() > 0.5
    assert weights[0, 1, 2].item() > 0.5
