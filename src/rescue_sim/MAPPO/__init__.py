"""MAPPO -- Multi-Agent PPO with a centralized critic (CTDE).

Requires the optional `torch` dependency: ``pip install -e ".[mappo]"``.
The environment (`RescueEnv`) is pure NumPy and can be imported without torch.
"""

from rescue_sim.MAPPO.environment import RescueEnv

__all__ = ["RescueEnv", "ActorCritic", "MAPPO"]


def __getattr__(name: str):
    # Lazy import so `import rescue_sim.MAPPO` (or RescueEnv) works without torch.
    if name in ("ActorCritic", "MAPPO"):
        from rescue_sim.MAPPO import mappo

        return getattr(mappo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
