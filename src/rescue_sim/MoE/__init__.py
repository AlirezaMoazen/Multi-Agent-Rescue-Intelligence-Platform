"""Mixture-of-Experts (MoE) neural routing modules and routines."""

from rescue_sim.MoE.moe import (
    AttentionGatingRouter,
    NeuralMoEPolicy,
    RecurrentFallbackHead,
    SharedFeatureEncoder,
    distill_expert_heads,
    train_gating_router,
)
from rescue_sim.MoE.pipeline import (
    ACTION_DIM,
    COMM_RADIUS,
    EXPERT_NAMES,
    CoordinationTeacher,
    FixedGridRescueEnv,
    ExplorationTeacher,
    FallbackTeacher,
    build_peer_matrix,
    collect_expert_dataset,
    make_teachers,
    run_expert_distillation,
    run_router_optimization,
    train_moe_policy,
)

__all__ = [
    "SharedFeatureEncoder",
    "AttentionGatingRouter",
    "RecurrentFallbackHead",
    "NeuralMoEPolicy",
    "distill_expert_heads",
    "train_gating_router",
    "ACTION_DIM",
    "COMM_RADIUS",
    "EXPERT_NAMES",
    "ExplorationTeacher",
    "CoordinationTeacher",
    "FallbackTeacher",
    "FixedGridRescueEnv",
    "build_peer_matrix",
    "collect_expert_dataset",
    "make_teachers",
    "run_expert_distillation",
    "run_router_optimization",
    "train_moe_policy",
]
