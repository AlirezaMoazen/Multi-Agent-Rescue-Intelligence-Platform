"""Run one rescue scenario from a YAML config and print basic metrics.

Loads a scenario (grid + agent + simulation settings), runs the tested
``SimulationRunner`` with its default move cycle, and reports the outcome. This
is the plain single-agent entry point; the decentralized fleet, the deep methods,
and the Neural MoE have their own scripts and the live dashboard (see the README).

    python scripts/run_scenario.py [path/to/scenario.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from rescue_sim.config.settings import AgentSettings, GridSettings, SimulationSettings
from rescue_sim.simulation.runner import SimulationRunner

DEFAULT_CONFIG = Path("configs/default_scenario.yaml")


def load_scenario(path: Path) -> tuple[GridSettings, AgentSettings, SimulationSettings]:
    """Parse a scenario YAML file into the three typed settings blocks."""

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (
        GridSettings(**data["grid"]),
        AgentSettings(**data["agent"]),
        SimulationSettings(**data["simulation"]),
    )


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    if not config_path.exists():
        raise SystemExit(f"scenario config not found: {config_path}")

    grid_settings, agent_settings, simulation_settings = load_scenario(config_path)
    result = SimulationRunner().run(grid_settings, agent_settings, simulation_settings)

    print(f"Scenario: {config_path}")
    print(f"  grid            : {grid_settings.width}x{grid_settings.height}")
    print(f"  start           : {result.start_position}")
    print(f"  final position  : {result.final_position}")
    print(f"  steps taken     : {result.steps_taken}")
    print(f"  targets found   : {result.targets_found}")
    print(f"  success         : {result.success}")


if __name__ == "__main__":
    main()
