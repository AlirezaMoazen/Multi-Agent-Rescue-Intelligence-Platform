"""Distill the trained QMIX+TransfQMix+MAPPO teachers into MoE Expert 2 via
State-Conditioned Gated Teacher Selection (per-teacher calibrated temperatures,
pseudo-oracle-weighted gating router, gated reverse-KL).

    python scripts/distill_expert2.py --grid 14 --episodes 8

Reports the learned per-state teacher weights (TransfQMix should stay low while
its checkpoint is weak) and the distillation accuracy.
"""

from __future__ import annotations

import argparse

from rescue_sim.config.settings import GridSettings
from rescue_sim.MAPPO import RescueEnv
from rescue_sim.MoE.gated_distill import train_gated_expert2
from rescue_sim.MoE.pipeline import train_moe_policy
from rescue_sim.TransfQMix import EntityRescueEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Gated distillation into MoE Expert 2.")
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--view-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=8, help="distillation collection episodes")
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    grid = GridSettings(width=args.grid, height=args.grid, obstacle_probability=0.15,
                        target_a_count=2, target_b_count=2)

    # 1) Train a base MoE so the shared encoder + E1/E3 heads are meaningful.
    base_env = RescueEnv(grid, num_agents=args.agents, max_steps=args.max_steps,
                         view_radius=args.view_radius, seed=args.seed)
    print("Training base MoE (encoder + experts)…")
    policy = train_moe_policy(base_env, seed=args.seed, use_ensemble=False)

    # 2) Gated Expert-2 distillation against the trained teachers.
    entity_env = EntityRescueEnv(grid, num_agents=args.agents, max_steps=args.max_steps,
                                 view_radius=args.view_radius, seed=args.seed + 1)
    print("Distilling Expert 2 (gated teacher selection)…")
    metrics = train_gated_expert2(
        policy, entity_env, checkpoint_dir=args.checkpoint_dir,
        episodes=args.episodes, steps=args.steps, epochs=args.epochs, seed=args.seed,
    )

    print("\n── Gated Expert-2 distillation ──────────────────────────────")
    print(f"  calibrated temperatures : {metrics['temperatures']}")
    print(f"  mean teacher weights    : {metrics['teacher_weights']}")
    print(f"  reverse-KL (final)      : {metrics['rkl']:.4f}")
    print(f"  distillation accuracy   : {metrics['acc']:.1f}%")


if __name__ == "__main__":
    main()
