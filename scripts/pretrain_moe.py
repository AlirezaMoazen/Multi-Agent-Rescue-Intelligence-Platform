"""Pretrain the Neural MoE offline and save the BEST round to checkpoints/moe.pt.

Runs the same distillation pipeline the dashboard uses (gated E2 from the
trained MAPPO/QMIX/TransfQMix checkpoints) for a wall-clock budget over many
freshly seeded grids. After every round the policy is scored on a fixed
validation grid sequence, and the checkpoint is only overwritten when the
validation success rate improves — so more rounds can never make the saved
policy worse, only better. The dashboard loads the result at startup.

    python scripts/pretrain_moe.py --minutes 5
"""

from __future__ import annotations

import argparse
import time

from rescue_sim.config.settings import GridSettings
from rescue_sim.MAPPO import RescueEnv
from rescue_sim.MoE.pipeline import evaluate_moe_policy, save_moe_policy, train_moe_policy

VAL_SEED = 987_654  # fixed validation grid sequence, disjoint from training seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the Neural MoE policy.")
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--view-radius", type=int, default=3)
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--val-episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="checkpoints/moe.pt")
    args = parser.parse_args()

    grid = GridSettings(
        width=args.grid, height=args.grid, obstacle_probability=0.15,
        target_a_count=2, target_b_count=2, random_seed=args.seed,
    )
    shape = (args.grid, args.grid, args.agents, args.view_radius)
    deadline = time.monotonic() + args.minutes * 60

    def make_val_env() -> RescueEnv:
        # Fresh env each time -> identical grid sequence for every round's eval.
        return RescueEnv(
            grid, num_agents=args.agents, max_steps=200,
            view_radius=args.view_radius, seed=VAL_SEED,
        )

    policy = None
    best = {"success_rate": -1.0, "avg_rescued": -1.0}
    best_round = 0

    # Never regress: score the existing checkpoint first and require any
    # round to beat it before overwriting.
    from rescue_sim.MoE.pipeline import load_moe_policy

    existing = load_moe_policy(args.out)
    if existing is not None and existing[1] == shape:
        prev_score = evaluate_moe_policy(existing[0], make_val_env(), episodes=args.val_episodes)
        best = {"success_rate": prev_score["success_rate"], "avg_rescued": prev_score["avg_rescued"]}
        print(
            f"[pretrain] existing {args.out}: val success={prev_score['success_rate']:.2f} "
            f"rescued={prev_score['avg_rescued']:.2f} — new rounds must beat this"
        )
    epochs_per_round = 12
    total_epochs = 0
    round_idx = 0
    while time.monotonic() < deadline:
        round_idx += 1
        round_seed = args.seed + round_idx * 31  # fresh grids/data every round
        env = RescueEnv(
            grid, num_agents=args.agents, max_steps=80,
            view_radius=args.view_radius, seed=round_seed,
        )
        started = time.monotonic()
        policy = train_moe_policy(
            env,
            episodes_per_head=6,
            collect_steps=70,
            epochs=epochs_per_round,
            router_steps=150,
            seed=round_seed,
            policy=policy,
            e2_gated=True,
        )
        total_epochs += epochs_per_round

        score = evaluate_moe_policy(policy, make_val_env(), episodes=args.val_episodes)
        improved = (score["success_rate"], score["avg_rescued"]) > (
            best["success_rate"], best["avg_rescued"]
        )
        if improved:
            best = score
            best_round = round_idx
            save_moe_policy(
                policy, args.out, env.obs_dim, args.view_radius, shape, total_epochs
            )
        print(
            f"[pretrain] round {round_idx}: {time.monotonic() - started:.0f}s, "
            f"val success={score['success_rate']:.2f} rescued={score['avg_rescued']:.2f} "
            f"steps={score['avg_steps']:.0f}"
            + (f"  -> saved {args.out} (new best)" if improved else "  (kept previous best)")
        )
        # Roll back to the best policy so the next round trains on top of it
        # instead of drifting further from a bad round.
        if not improved and best_round:
            from rescue_sim.MoE.pipeline import load_moe_policy

            loaded = load_moe_policy(args.out)
            if loaded is not None:
                policy = loaded[0]

    if best_round == 0:
        raise SystemExit("time budget too small: no training round completed")
    print(
        f"[pretrain] finished: {round_idx} rounds; best was round {best_round} "
        f"(val success={best['success_rate']:.2f}, rescued={best['avg_rescued']:.2f}) "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
