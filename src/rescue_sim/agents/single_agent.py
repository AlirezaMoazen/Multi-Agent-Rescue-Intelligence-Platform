"""Single rescue agent model (legacy).

Part of the project's original *single-agent* flow, kept only for the legacy
single-agent visualization/evaluation path. Multi-agent work uses
``shared.AgentState`` / ``MultiAgentState`` and the fleet learners instead.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SingleAgent:
    """State for one rescue agent (legacy single-agent flow)."""

    x: int
    y: int
    sensor_range: int

