"""QMIX -- monotonic value-function factorization for cooperative MARL (CTDE).

Requires the optional `torch` dependency: ``pip install -e ".[qmix]"``.
Reuses `RescueEnv` from rescue_sim.MAPPO (pure NumPy, importable without torch).
"""

__all__ = ["AgentQNet", "MixingNetwork", "QMIX", "ReplayBuffer"]


def __getattr__(name: str):
    # Lazy import so torch is only required when QMIX is actually used.
    if name in __all__:
        from rescue_sim.QMIX import qmix

        return getattr(qmix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
