"""Train MAPPO on the cooperative rescue environment.

Usage:
    python scripts/train_mappo.py                      # time-boxed default run
    python scripts/train_mappo.py --time-budget 5400   # train up to 1.5 h

Requires the optional torch dependency:  pip install -e ".[mappo]"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rescue_sim.config.settings import GridSettings, MappoSettings
from rescue_sim.MAPPO import MAPPO, RescueEnv
from rescue_sim.shared import make_eval_hook, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO on the rescue grid.")
    parser.add_argument("--updates", type=int, default=1000000, help="hard cap; usually time-limited first")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid", type=int, default=14, help="grid width/height")
    parser.add_argument("--view-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200, help="per-episode step cap during training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--time-budget", type=float, default=5400.0, help="max wall-clock seconds")
    parser.add_argument("--eval-every", type=int, default=10, help="updates between greedy evals + best-checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=30, help="greedy episodes per eval (best-checkpoint selection)")
    parser.add_argument("--checkpoint", default="checkpoints/mappo.pt")
    args = parser.parse_args()

    import os
    device = resolve_device(args.device)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS") or torch.get_num_threads()))
    print(f"MAPPO training on device={device}, grid={args.grid}, max_steps={args.max_steps}, "
          f"budget={args.time_budget / 60:.0f}min")

    grid = GridSettings(
        width=args.grid,
        height=args.grid,
        obstacle_probability=0.15,
        target_a_count=2,
        target_b_count=2,
    )
    settings = MappoSettings(
        num_agents=args.agents,
        view_radius=args.view_radius,
        max_steps=args.max_steps,
        random_seed=args.seed,
    )
    env = RescueEnv(
        grid,
        num_agents=settings.num_agents,
        max_steps=settings.max_steps,
        view_radius=settings.view_radius,
        seed=args.seed,
    )

    trainer = MAPPO(env, settings, device=args.device)
    hook, state = make_eval_hook(trainer, args.checkpoint, args.time_budget, eval_episodes=args.eval_episodes)
    trainer.train(num_updates=args.updates, eval_hook=hook, hook_every=args.eval_every)

    print("\nFinal greedy evaluation (best checkpoint kept during training):")
    print(f"  best avg_rescued={state['best_rescued']:.2f}  success_rate={state['best_success']:.2f}")
    print(f"  saved {state['saved']} improved checkpoints -> {Path(args.checkpoint)}")


if __name__ == "__main__":
    main()
