"""Ensemble + distillation for the deep value-based methods.

* ``ValueEnsemble`` combines a trained QMIX and TransfQMix at decision time.
* ``Distiller`` compresses that ensemble into a single student network.

Requires the optional torch dependency: ``pip install -e ".[ensemble]"``.
"""

__all__ = ["ValueEnsemble", "performance_weights", "Distiller"]


def __getattr__(name: str):
    # Lazy import so torch is only required when the ensemble is actually used.
    if name in ("ValueEnsemble", "performance_weights"):
        from rescue_sim.Ensemble import ensemble

        return getattr(ensemble, name)
    if name == "Distiller":
        from rescue_sim.Ensemble import distill

        return distill.Distiller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
