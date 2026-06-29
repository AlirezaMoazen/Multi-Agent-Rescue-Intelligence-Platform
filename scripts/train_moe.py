"""Run the Mixture-of-Experts gate: deep ensemble vs. an adaptive tabular fleet.

The script trains a small QMIX + TransfQMix on the random-grid distribution,
freezes them into a ``ValueEnsemble`` (the generalist *deep expert*), then pits
that ensemble against ``EpidemicHystereticQLearning`` (the specialist *adaptive
expert*) on ONE fixed grid. It prints, try by try, each expert's "solve score"
and which one is driving the fleet -- so you can watch the tabular learner catch
up to and overtake the frozen deep model on that specific grid.

Usage:
    python scripts/train_moe.py                      # quick default run
    python scripts/train_moe.py --episodes 300 --trials 40

Requires the optional torch dependency:  pip install -e ".[ensemble]"
"""

from __future__ import annotations

import argparse

from rescue_sim.config.settings import (
    GridSettings,
    MoeSettings,
    QmixSettings,
    TransfQmixSettings,
)
from rescue_sim.Ensemble import ValueEnsemble, performance_weights
from rescue_sim.MoE import MixtureOfExperts
from rescue_sim.QMIX import QMIX
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixture-of-Experts: deep ensemble vs. tabular fleet.")
    parser.add_argument("--grid", type=int, default=20, help="grid width/height")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--targets-a", type=int, default=2, help="number of Target A")
    parser.add_argument("--targets-b", type=int, default=2, help="number of Target B")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--episodes", type=int, default=250,
                        help="deep-model training episodes (train the generalist well)")
    parser.add_argument("--trials", type=int, default=40, help="MoE tries on the fixed grid")
    parser.add_argument("--grid-seed", type=int, default=1, help="seed of the fixed competition grid")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The realistic rescue task: a 20x20 grid, a fleet of agents, and several
    # Target A / Target B sites. The generalist is trained well so the two
    # experts are a fair match; the tabular specialist (spread starts let its
    # memoryless agents divide the targets) learns the grid from the generalist's
    # demonstrations and competes on it.
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

    q_eval = qmix.evaluate(episodes=20)
    t_eval = transf.evaluate(episodes=20)
    w_qmix, w_transf = performance_weights(q_eval["success_rate"], t_eval["success_rate"])
    deep_expert = ValueEnsemble(qmix, transf, env(), w_qmix, w_transf)

    print("\nRunning the Mixture-of-Experts gate on one fixed grid ...")
    moe = MixtureOfExperts(
        deep_expert,
        MoeSettings(num_trials=args.trials, random_seed=args.seed, grid_seed=args.grid_seed,
                    epsilon_start=0.6, epsilon_decay=0.03),
    )
    report = moe.run()

    print(f"\nDeep expert (frozen) solve score on the fixed grid: {report.deep_score:.3f}\n")
    print(f"{'try':>4}{'leader':>10}{'deep':>8}{'adaptive':>10}{'eps':>7}{'mean_q':>9}{'syncs':>7}")
    for row in report.history:
        print(
            f"{row.trial:>4}{row.leader:>10}{row.deep['score']:>8.3f}"
            f"{row.adaptive['score']:>10.3f}{row.epsilon:>7.2f}"
            f"{row.mean_q:>9.3f}{row.syncs:>7}"
        )

    if report.surpassed_at is not None:
        print(
            f"\nThe adaptive fleet overtook the deep expert on try {report.surpassed_at} "
            f"and drove the fleet from try {report.surpassed_at + 1} on."
        )
    else:
        print("\nThe deep expert was never overtaken within the trial budget.")
    print(f"Final leader: {report.final_leader} | best adaptive score: {report.best_adaptive_score:.3f}")


if __name__ == "__main__":
    main()
