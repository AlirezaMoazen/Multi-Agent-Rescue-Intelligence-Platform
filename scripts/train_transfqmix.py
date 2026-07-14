"""Train TransfQMix on the cooperative rescue environment.

Usage:
    python scripts/train_transfqmix.py                     # time-boxed default run
    python scripts/train_transfqmix.py --time-budget 5400  # train up to 1.5 h

Requires the optional torch dependency:  pip install -e ".[transfqmix]"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rescue_sim.config.settings import GridSettings, TransfQmixSettings
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX
from rescue_sim.shared import make_eval_hook, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TransfQMix on the rescue grid.")
    parser.add_argument("--episodes", type=int, default=100000, help="hard cap; usually time-limited first")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid", type=int, default=14, help="grid width/height")
    parser.add_argument("--view-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200, help="per-episode step cap during training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--time-budget", type=float, default=5400.0, help="max wall-clock seconds")
    parser.add_argument("--eval-every", type=int, default=50, help="episodes between greedy evals + best-checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=30, help="greedy episodes per eval (best-checkpoint selection)")
    parser.add_argument("--epsilon-anneal", type=int, default=300,
                        help="episodes to anneal epsilon 1.0 -> 0.05. Keep moderate: the "
                             "value-target normalizer is anchored by early greedy successes, "
                             "and very long random phases (e.g. 1500) make it diverge")
    parser.add_argument("--buffer-size", type=int, default=5000,
                        help="replay capacity in transitions (shipped default; larger buffers "
                             "hold too much stale random data for the normalized mixer)")
    parser.add_argument("--checkpoint", default="checkpoints/transfqmix.pt")
    args = parser.parse_args()

    import os
    device = resolve_device(args.device)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS") or torch.get_num_threads()))
    print(f"TransfQMix training on device={device}, grid={args.grid}, max_steps={args.max_steps}, "
          f"budget={args.time_budget / 60:.0f}min")

    grid = GridSettings(
        width=args.grid,
        height=args.grid,
        obstacle_probability=0.15,
        target_a_count=2,
        target_b_count=2,
    )
    settings = TransfQmixSettings(
        num_agents=args.agents,
        view_radius=args.view_radius,
        max_steps=args.max_steps,
        random_seed=args.seed,
        epsilon_anneal_episodes=args.epsilon_anneal,
        buffer_size=args.buffer_size,
    )
    env = EntityRescueEnv(
        grid,
        num_agents=settings.num_agents,
        max_steps=settings.max_steps,
        view_radius=settings.view_radius,
        seed=args.seed,
    )

    trainer = TransfQMIX(env, settings, device=args.device)
    hook, state = make_eval_hook(trainer, args.checkpoint, args.time_budget, eval_episodes=args.eval_episodes)
    trainer.train(num_episodes=args.episodes, eval_hook=hook, hook_every=args.eval_every)

    print("\nFinal greedy evaluation (best checkpoint kept during training):")
    print(f"  best avg_rescued={state['best_rescued']:.2f}  success_rate={state['best_success']:.2f}")
    print(f"  saved {state['saved']} improved checkpoints -> {Path(args.checkpoint)}")


if __name__ == "__main__":
    main()
