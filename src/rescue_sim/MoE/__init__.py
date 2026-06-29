"""Mixture-of-Experts gate over the deep ensemble and the tabular fleet learner.

Two experts compete on **one fixed rescue grid**:

* the *deep expert* -- the trained ``ValueEnsemble`` of QMIX + TransfQMix, which
  generalizes to any grid but is frozen, and
* the *adaptive expert* -- ``EpidemicHystereticQLearning``, a tabular fleet that
  starts from scratch and learns *this* grid a little more on every try.

A performance gate routes each try to whichever expert currently solves the
grid better. The adaptive expert keeps learning on every try (even while the
deep expert is driving), so on a fixed grid it eventually overtakes the
generalist; from the next try on it drives the fleet -- and keeps learning.

Requires the optional torch dependency (the deep expert is a neural net):
``pip install -e ".[ensemble]"``.
"""

__all__ = [
    "MixtureOfExperts",
    "MoeReport",
    "MoeTrial",
    "FixedGridEntityEnv",
    "solve_score",
]


def __getattr__(name: str):
    # Lazy import so ``import rescue_sim.MoE`` does not require torch until the
    # mixture is actually built -- mirrors rescue_sim.Ensemble.
    if name in __all__:
        from rescue_sim.MoE import moe

        return getattr(moe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
