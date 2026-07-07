"""Evaluate the *saved* checkpoints (no training) and compare them.

Unlike ``compare_all.py`` (which trains fresh throwaway models), this loads the
trained checkpoints in ``checkpoints/`` and reports greedy rescue performance on
a held-out random-grid distribution, plus the QMIX+TransfQMix value ensemble.

    python scripts/eval_checkpoints.py --grid 14 --episodes 50
"""

from __future__ import annotations

import argparse

from rescue_sim.config.settings import (
    GridSettings,
    MappoSettings,
    QmixSettings,
    TransfQmixSettings,
)
from rescue_sim.Ensemble import ValueEnsemble, performance_weights
from rescue_sim.MAPPO import MAPPO, RescueEnv
from rescue_sim.QMIX import QMIX
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved deep-RL checkpoints.")
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--view-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    grid = GridSettings(
        width=args.grid, height=args.grid, obstacle_probability=0.15,
        target_a_count=2, target_b_count=2,
    )

    def make_env(entity: bool):
        cls = EntityRescueEnv if entity else RescueEnv
        return cls(grid, num_agents=args.agents, max_steps=args.max_steps,
                   view_radius=args.view_radius, seed=args.seed)

    results: dict[str, dict] = {}

    # MAPPO
    m_env = make_env(entity=False)
    mappo = MAPPO(m_env, MappoSettings(num_agents=args.agents, view_radius=args.view_radius,
                                       max_steps=args.max_steps, random_seed=args.seed),
                  device=args.device)
    mappo.load_checkpoint("checkpoints/mappo.pt")
    results["MAPPO"] = mappo.evaluate(episodes=args.episodes)

    # QMIX
    q_env = make_env(entity=False)
    qmix = QMIX(q_env, QmixSettings(num_agents=args.agents, view_radius=args.view_radius,
                                    max_steps=args.max_steps, random_seed=args.seed),
                device=args.device)
    qmix.load_checkpoint("checkpoints/qmix.pt")
    results["QMIX"] = qmix.evaluate(episodes=args.episodes)

    # TransfQMix
    t_env = make_env(entity=True)
    transf = TransfQMIX(t_env, TransfQmixSettings(num_agents=args.agents, view_radius=args.view_radius,
                                                  max_steps=args.max_steps, random_seed=args.seed),
                        device=args.device)
    transf.load_checkpoint("checkpoints/transfqmix.pt")
    results["TransfQMix"] = transf.evaluate(episodes=args.episodes)

    # QMIX + TransfQMix value ensemble (weighted by standalone success)
    w_q, w_t = performance_weights(results["QMIX"]["success_rate"], results["TransfQMix"]["success_rate"])
    e_env = make_env(entity=True)
    ensemble = ValueEnsemble(qmix, transf, e_env, w_qmix=w_q, w_transf=w_t)
    results["Ensemble(QMIX+TransfQMix)"] = ensemble.evaluate(episodes=args.episodes)

    print(f"\nGreedy eval on {args.grid}x{args.grid}, {args.episodes} episodes, "
          f"max_steps={args.max_steps}:\n")
    print(f"{'method':<26} {'success':>8} {'rescued':>9} {'steps':>7}")
    print("-" * 54)
    for name, m in results.items():
        print(f"{name:<26} {m['success_rate']:>8.2f} {m['avg_rescued']:>9.2f} {m['avg_steps']:>7.0f}")


if __name__ == "__main__":
    main()
