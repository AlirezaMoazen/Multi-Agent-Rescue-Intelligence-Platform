"""Mixture-of-Experts router over every rescue strategy in the project.

A performance gate scores a whole pool of experts on **one fixed rescue grid**
and routes the fleet to whichever solves it best -- the per-instance
algorithm-selection problem (Rice 1976; SATzilla, Xu et al. 2008) phrased as a
Mixture-of-Experts (Jacobs et al. 1991). The pool:

* **classical (no-AI)** -- the 7 baselines (frontier, DFS, prioritized planning,
  CBS, ICBS, ECBS, M*), run as a synchronized team;
* **deep (frozen)** -- QMIX, TransfQMix, MAPPO, and their ``ValueEnsemble``;
* **adaptive** -- ``EpidemicHystereticQLearning``, which learns *this* grid a
  little more every try (off-policy from the best deep expert while it is
  behind, then self-play once it leads).

Frozen experts are scored once (a leaderboard); the adaptive fleet is re-scored
each try and can climb to the top. Because the gate only ever serves the
best-scoring expert, the MoE is never worse than the strongest single method.

Build it with ``MixtureOfExperts.from_models(qmix=..., transf=..., mappo=...,
ensemble=...)``. Requires torch (the deep experts): ``pip install -e ".[ensemble]"``.
"""

__all__ = [
    "MixtureOfExperts",
    "MoeReport",
    "MoeTrial",
    "FixedGridEntityEnv",
    "solve_score",
    "qmix_policy",
    "transf_policy",
    "mappo_policy",
    "ensemble_policy",
]


def __getattr__(name: str):
    # Lazy import so ``import rescue_sim.MoE`` does not require torch until the
    # mixture is actually built -- mirrors rescue_sim.Ensemble.
    if name in __all__:
        from rescue_sim.MoE import moe

        return getattr(moe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
