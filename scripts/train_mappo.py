"""Train MAPPO on the cooperative rescue environment.

Usage:
    python scripts/train_mappo.py                 # quick default run
    python scripts/train_mappo.py --updates 200   # longer training

Requires the optional torch dependency:  pip install -e ".[mappo]"
"""

from __future__ import annotations

import argparse

from rescue_sim.config.settings import GridSettings, MappoSettings
from rescue_sim.MAPPO import MAPPO, RescueEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO on the rescue grid.")
    parser.add_argument("--updates", type=int, default=50, help="rollout+update cycles")
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
    settings = MappoSettings(num_agents=args.agents, random_seed=args.seed)
    env = RescueEnv(
        grid,
        num_agents=settings.num_agents,
        max_steps=settings.max_steps,
        view_radius=settings.view_radius,
        seed=args.seed,
    )

    trainer = MAPPO(env, settings)
    trainer.train(num_updates=args.updates)

    print("\nFinal greedy evaluation:")
    print(trainer.evaluate(episodes=20))


if __name__ == "__main__":
    main()
