"""Train every deep method, then build + distill the ensemble, and compare.

Runs the full pipeline end to end and prints one table:

    MAPPO | QMIX | TransfQMix | Ensemble(QMIX+TransfQMix) | Distilled student

All methods train on the same random-grid distribution and are evaluated greedily
on fresh grids, so the numbers are directly comparable.

Requires the optional torch dependency:  pip install -e ".[ensemble]"
"""

from __future__ import annotations

import argparse

from rescue_sim.config.settings import (
    DistillSettings,
    GridSettings,
    MappoSettings,
    QmixSettings,
    TransfQmixSettings,
)
from rescue_sim.Ensemble import Distiller, ValueEnsemble, performance_weights
from rescue_sim.MAPPO import MAPPO, RescueEnv
from rescue_sim.QMIX import QMIX
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all methods + ensemble + distillation.")
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--mappo-updates", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=150, help="episodes for QMIX/TransfQMix")
    parser.add_argument("--eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grid = GridSettings(
        width=args.grid, height=args.grid, obstacle_probability=0.15,
        target_a_count=2, target_b_count=2,
    )

    def env() -> EntityRescueEnv:
        return EntityRescueEnv(grid, num_agents=args.agents, max_steps=200,
                               view_radius=2, seed=args.seed)

    results: dict[str, dict] = {}

    print("Training MAPPO ...")
    mappo = MAPPO(RescueEnv(grid, num_agents=args.agents, max_steps=200, view_radius=2,
                            seed=args.seed),
                  MappoSettings(num_agents=args.agents, random_seed=args.seed))
    mappo.train(num_updates=args.mappo_updates, log_every=0)
    results["MAPPO"] = mappo.evaluate(episodes=args.eval)

    print("Training QMIX ...")
    qmix = QMIX(env(), QmixSettings(num_agents=args.agents, random_seed=args.seed))
    qmix.train(num_episodes=args.episodes, log_every=0)
    results["QMIX"] = qmix.evaluate(episodes=args.eval)

    print("Training TransfQMix ...")
    transf = TransfQMIX(env(), TransfQmixSettings(num_agents=args.agents, random_seed=args.seed))
    transf.train(num_episodes=args.episodes, log_every=0)
    results["TransfQMix"] = transf.evaluate(episodes=args.eval)

    print("Building ensemble ...")
    w_qmix, w_transf = performance_weights(
        results["QMIX"]["success_rate"], results["TransfQMix"]["success_rate"]
    )
    ensemble = ValueEnsemble(qmix, transf, env(), w_qmix, w_transf)
    results["Ensemble"] = ensemble.evaluate(episodes=args.eval)

    print("Distilling ensemble into a single student ...")
    distiller = Distiller(ensemble, env(), DistillSettings(random_seed=args.seed))
    distiller.train()
    results["Distilled"] = distiller.evaluate(episodes=args.eval)

    print("\n=== Greedy evaluation on fresh grids ===")
    print(f"{'method':<26}{'success':>9}{'rescued':>9}{'steps':>8}")
    for name, m in results.items():
        print(f"{name:<26}{m['success_rate']:>9.2f}{m['avg_rescued']:>9.2f}{m['avg_steps']:>8.1f}")


if __name__ == "__main__":
    main()
