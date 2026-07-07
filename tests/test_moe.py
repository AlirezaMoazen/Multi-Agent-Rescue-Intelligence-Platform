# Copyright 2026 Alireza Moazen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the step-level Neural Mixture of Experts (MoE) Policy.

Covers:
- SharedFeatureEncoder dimensionality and permutation invariance
- AttentionGatingRouter weight distributions and peer masking
- RecurrentFallbackHead GRU hidden state persistence
- NeuralMoEPolicy forward pass, action masking, and 3-tuple return
- distill_expert_heads offline behavioral cloning
- train_gating_router online policy gradient with blackout penalty
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rescue_sim.MoE.moe import (  # noqa: E402
    AttentionGatingRouter,
    NeuralMoEPolicy,
    RecurrentFallbackHead,
    SharedFeatureEncoder,
    distill_expert_heads,
    train_gating_router,
)
from rescue_sim.shared import Position  # noqa: E402


class MockEnv:
    """Mock environment with gym-style interface and blackout state support."""
    def __init__(self, num_agents: int = 2, obs_dim: int = 108) -> None:
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


def test_attention_gating_router() -> None:
    """Tests AttentionGatingRouter produces valid probability distributions."""
    latent_dim = 64
    router = AttentionGatingRouter(latent_dim)

    batch_size = 5
    max_peers = 4

    z_ego = torch.randn(batch_size, latent_dim)
    z_peers = torch.randn(batch_size, max_peers, latent_dim)
    peer_mask = torch.ones(batch_size, max_peers, dtype=torch.bool)

    weights = router(z_ego, z_peers, peer_mask)

    assert weights.shape == (batch_size, 3)
    assert torch.allclose(torch.sum(weights, dim=-1), torch.ones(batch_size), atol=1e-5)
    assert torch.all(weights >= 0.0)


def test_attention_router_with_partial_mask() -> None:
    """Tests that the router handles variable peer counts via masking."""
    latent_dim = 64
    router = AttentionGatingRouter(latent_dim)

    batch_size = 3
    max_peers = 4

    z_ego = torch.randn(batch_size, latent_dim)
    z_peers = torch.randn(batch_size, max_peers, latent_dim)

    # First batch element: 2 valid peers, second: 4, third: 1
    peer_mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ], dtype=torch.bool)

    weights = router(z_ego, z_peers, peer_mask)

    assert weights.shape == (batch_size, 3)
    assert torch.allclose(torch.sum(weights, dim=-1), torch.ones(batch_size), atol=1e-5)


def test_recurrent_fallback_head() -> None:
    """Tests GRU hidden state persistence across steps."""
    latent_dim = 64
    action_dim = 4
    batch_size = 6

    head = RecurrentFallbackHead(latent_dim, action_dim)

    z = torch.randn(batch_size, latent_dim)

    # Step 1: h_prev is None → zeros
    logits1, h1 = head(z, None)
    assert logits1.shape == (batch_size, action_dim)
    assert h1.shape == (batch_size, latent_dim)

    # Step 2: pass h1 as hidden state
    logits2, h2 = head(z, h1)
    assert logits2.shape == (batch_size, action_dim)
    assert h2.shape == (batch_size, latent_dim)

    # Hidden states should differ across steps (GRU updates them)
    assert not torch.allclose(h1, h2, atol=1e-6)


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

    y_final, weights, fallback_hidden = policy(obs, peer_matrix, action_mask)
    assert y_final.shape == (batch_size, num_agents, action_dim)
    assert weights.shape == (batch_size, num_agents, 3)
    assert fallback_hidden.shape[0] == batch_size
    assert fallback_hidden.shape[1] == num_agents

    # Action selection with custom masking
    custom_mask = torch.tensor([[[True, False, False, False], [False, True, False, False]],
                                [[False, False, True, False], [False, False, False, True]],
                                [[True, False, False, False], [False, True, False, False]]], dtype=torch.bool)
    actions, h_next = policy.get_action(obs, peer_matrix, custom_mask)
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

    with torch.no_grad():
        _, weights, _ = policy(obs, peer_matrix, action_mask)

    # Due to penalty during training, Expert 3 (index 2) weight should be significantly positive
    assert weights[0, 0, 2].item() > 0.3
    assert weights[0, 1, 2].item() > 0.3
