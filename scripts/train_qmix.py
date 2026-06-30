"""Train QMIX on the cooperative rescue environment.

Usage:
    python scripts/train_qmix.py                  # quick default run
    python scripts/train_qmix.py --episodes 400   # longer training

Requires the optional torch dependency:  pip install -e ".[qmix]"
"""

from __future__ import annotations

import argparse

from rescue_sim.config.settings import GridSettings, QmixSettings
from rescue_sim.MAPPO import RescueEnv
from rescue_sim.QMIX import QMIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QMIX on the rescue grid.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid", type=int, default=8, help="grid width/height")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grid = GridSettings(
        width=args.grid,
        height=args.grid,
        obstacle_probability=0.15,
        target_a_count=2,
        target_b_count=2,
    )
    settings = QmixSettings(num_agents=args.agents, random_seed=args.seed)
    env = RescueEnv(
        grid,
        num_agents=settings.num_agents,
        max_steps=settings.max_steps,
        view_radius=settings.view_radius,
        seed=args.seed,
    )

    trainer = QMIX(env, settings)
    trainer.train(num_episodes=args.episodes)

    print("\nFinal greedy evaluation:")
    print(trainer.evaluate(episodes=20))


if __name__ == "__main__":
    main()
