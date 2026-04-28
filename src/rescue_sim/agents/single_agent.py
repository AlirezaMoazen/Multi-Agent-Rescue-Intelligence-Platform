"""Single rescue agent model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SingleAgent:
    """State for one rescue agent."""

    x: int
    y: int
    sensor_range: int

