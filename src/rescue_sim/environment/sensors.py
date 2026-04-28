"""Sensor observations for the rescue agent."""

from dataclasses import dataclass

from rescue_sim.environment.grid import Position


@dataclass(frozen=True)
class Observation:
    obstacles: frozenset[Position]
    targets: frozenset[Position]
    visible_cells: frozenset[Position]

