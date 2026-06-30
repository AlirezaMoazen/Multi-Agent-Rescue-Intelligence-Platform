"""Train TransfQMix on the cooperative rescue environment.

Usage:
    python scripts/train_transfqmix.py                  # quick default run
    python scripts/train_transfqmix.py --episodes 400   # longer training

Requires the optional torch dependency:  pip install -e ".[transfqmix]"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rescue_sim.config.settings import GridSettings, TransfQmixSettings
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TransfQMix on the rescue grid.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid", type=int, default=8, help="grid width/height")
    parser.add_argument("--view-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/transfqmix.pt")
    args = parser.parse_args()

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
    )
    env = EntityRescueEnv(
        grid,
        num_agents=settings.num_agents,
        max_steps=settings.max_steps,
        view_radius=settings.view_radius,
        seed=args.seed,
    )

    trainer = TransfQMIX(env, settings)
    trainer.train(num_episodes=args.episodes)
    trainer.save_checkpoint(args.checkpoint)

    print("\nFinal greedy evaluation:")
    print(trainer.evaluate(episodes=20))
    print(f"Saved checkpoint: {Path(args.checkpoint)}")


if __name__ == "__main__":
    main()
