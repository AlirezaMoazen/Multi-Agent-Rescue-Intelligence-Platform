"""TransfQMix -- transformer-based value factorization for cooperative MARL.

Requires the optional `torch` dependency: ``pip install -e ".[transfqmix]"``.
Reuses `RescueEnv` (via EntityRescueEnv) and QMIX's replay buffer.
"""

__all__ = [
    "EntityRescueEnv",
    "AgentTransformer",
    "TransformerMixer",
    "TransfQMIX",
]


def __getattr__(name: str):
    # Lazy import so torch is only required when TransfQMix is actually used.
    if name in __all__:
        from rescue_sim.TransfQMix import transf_qmix

        return getattr(transf_qmix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
