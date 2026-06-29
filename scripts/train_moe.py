"""Mixture-of-Experts router: pick the best rescue strategy for one fixed grid.

Trains the deep models (QMIX, TransfQMix, MAPPO) and builds their ValueEnsemble,
then runs the MoE gate on ONE fixed grid against the full expert pool:

    7 classical baselines + QMIX + TransfQMix + MAPPO + Ensemble + the adaptive
    Epidemic Hysteretic fleet.

The frozen experts are scored once (a leaderboard); the fleet then learns the
grid every try -- taught off-policy by the strongest deep expert until it leads
-- and the gate routes to whichever expert solves the grid best. It prints the
leaderboard, the fleet's climb, and when (if) the fleet takes over.

Usage:
    python scripts/train_moe.py                         # 20x20, 4 agents, 2A+2B
    python scripts/train_moe.py --grid 10 --episodes 120 --trials 40

Requires the optional torch dependency:  pip install -e ".[ensemble]"
"""

from __future__ import annotations

import argparse

from rescue_sim.config.settings import (
    GridSettings,
    MappoSettings,
    MoeSettings,
    QmixSettings,
    TransfQmixSettings,
)
from rescue_sim.Ensemble import ValueEnsemble, performance_weights
from rescue_sim.MAPPO import MAPPO, RescueEnv
from rescue_sim.MoE import MixtureOfExperts
from rescue_sim.QMIX import QMIX
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixture-of-Experts router over all rescue strategies.")
    parser.add_argument("--grid", type=int, default=20, help="grid width/height")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--targets-a", type=int, default=2, help="number of Target A")
    parser.add_argument("--targets-b", type=int, default=2, help="number of Target B")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--episodes", type=int, default=250,
                        help="deep-model training episodes (train the generalists well)")
    parser.add_argument("--mappo-updates", type=int, default=80)
    parser.add_argument("--trials", type=int, default=40, help="MoE tries on the fixed grid")
    parser.add_argument("--grid-seed", type=int, default=1, help="seed of the fixed competition grid")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grid = GridSettings(
        width=args.grid, height=args.grid, obstacle_probability=0.15,
        target_a_count=args.targets_a, target_b_count=args.targets_b,
    )

    def env() -> EntityRescueEnv:
        return EntityRescueEnv(grid, num_agents=args.agents, max_steps=args.max_steps,
                               view_radius=2, seed=args.seed)

    print("Training QMIX ...")
    qmix = QMIX(env(), QmixSettings(num_agents=args.agents, random_seed=args.seed))
    qmix.train(num_episodes=args.episodes, log_every=0)

    print("Training TransfQMix ...")
    transf = TransfQMIX(env(), TransfQmixSettings(num_agents=args.agents, random_seed=args.seed))
    transf.train(num_episodes=args.episodes, log_every=0)

    print("Training MAPPO ...")
    mappo = MAPPO(
        RescueEnv(grid, num_agents=args.agents, max_steps=args.max_steps, view_radius=2, seed=args.seed),
        MappoSettings(num_agents=args.agents, random_seed=args.seed),
    )
    mappo.train(num_updates=args.mappo_updates, log_every=0)

    w_qmix, w_transf = performance_weights(
        qmix.evaluate(episodes=20)["success_rate"],
        transf.evaluate(episodes=20)["success_rate"],
    )
    ensemble = ValueEnsemble(qmix, transf, env(), w_qmix, w_transf)

    print("\nRunning the Mixture-of-Experts gate on one fixed grid ...")
    moe = MixtureOfExperts.from_models(
        qmix=qmix, transf=transf, mappo=mappo, ensemble=ensemble,
        settings=MoeSettings(num_trials=args.trials, random_seed=args.seed,
                             grid_seed=args.grid_seed, epsilon_start=0.6, epsilon_decay=0.03),
    )
    report = moe.run()

    print("\n=== Fixed-expert leaderboard on the grid (scored once) ===")
    print(f"{'expert':<22}{'success':>9}{'rescued':>9}{'steps':>8}{'score':>8}")
    for name, m in sorted(report.leaderboard.items(), key=lambda kv: -kv[1]["score"]):
        print(f"{name:<22}{int(m['success']):>9}{m['rescued']:>5}/{m['targets']:<3}{m['steps']:>8}{m['score']:>8.3f}")
    print(f"\nTeacher for the adaptive fleet: {report.teacher}\n")

    print(f"{'try':>4}{'leader':>16}{'fleet_score':>13}{'eps':>7}{'mean_q':>9}{'syncs':>7}")
    for row in report.history:
        print(f"{row.trial:>4}{row.leader:>16}{row.fleet['score']:>13.3f}"
              f"{row.epsilon:>7.2f}{row.mean_q:>9.3f}{row.syncs:>7}")

    if report.surpassed_at is not None:
        print(f"\nThe adaptive fleet overtook every fixed expert on try {report.surpassed_at} "
              f"and led from try {report.surpassed_at + 1} on.")
    else:
        print("\nNo single expert was overtaken by the fleet within the trial budget "
              f"(leader stays '{report.final_leader}').")
    print(f"Final leader: {report.final_leader} | best fleet score: {report.best_fleet_score:.3f}")


if __name__ == "__main__":
    main()
