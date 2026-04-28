"""Typed settings for ST04 scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSettings:
    width: int
    height: int
    obstacle_probability: float
    target_count: int
    random_seed: int | None = None


@dataclass(frozen=True)
class AgentSettings:
    start_x: int
    start_y: int
    sensor_range: int


@dataclass(frozen=True)
class SimulationSettings:
    max_steps: int

