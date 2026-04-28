# Rescue Sim

Python implementation for backlog task **ST04: Single agent rescue team** from
the Multi-Agent Rescue Teams project.

The simulator models one rescue agent exploring a generated damaged area with
configurable obstacles and rescue targets. The agent knows how many targets
exist and must find them while avoiding blocked cells.

## Current Scope

ST04 focuses on the single-agent version of the rescue task:

- generate several rescue scenarios
- configure grid size, obstacle density, targets, start position, sensor range, and max steps
- place one rescue agent at a valid starting cell
- use a deterministic movement model first
- provide sensor observations around the agent
- explore until all known targets are found or the run reaches `max_steps`
- record metrics so strategies can be compared

Out of scope for ST04:

- multi-agent coordination
- agent-to-agent communication
- central multi-agent optimization
- distributed multi-agent learning
- uncertain movement or uncertain sensor readings

## Technology

Application code is written in **Python**.

Configuration and machine-readable output should use **YAML**.

Main dependencies are declared in [pyproject.toml](pyproject.toml):

- `numpy`
- `pydantic`
- `pyyaml`
- `pytest` for tests
- `ruff` for linting

## Project Layout

```text
.
|-- .gitlab-ci.yml             # GitLab CI pipeline
|-- configs/
|   `-- default_scenario.yaml  # Example YAML scenario configuration
|-- docs/
|   |-- architecture.md        # Architecture overview
|   |-- requirements.yaml      # ST04 requirements specification
|   `-- st04_design.md         # Initial design notes
|-- scripts/
|   `-- run_scenario.py        # Scenario runner entry point
|-- src/rescue_sim/
|   |-- agents/                # Single-agent state and policy logic
|   |-- config/                # YAML loading and typed settings
|   |-- environment/           # Grid, generation, movement, sensing
|   |-- learning/              # Baseline strategy and later learning methods
|   |-- simulation/            # Simulation runner and metrics
|   `-- visualization/         # Optional rendering helpers
`-- tests/                     # Unit and integration tests
```

## Documentation

- [Architecture](docs/architecture.md)
- [Requirements](docs/requirements.yaml)
- [ST04 design notes](docs/st04_design.md)

## Configuration

Scenarios are configured with YAML. The default scenario is in
[configs/default_scenario.yaml](configs/default_scenario.yaml).

Example structure:

```yaml
grid:
  width: 20
  height: 20
  obstacle_probability: 0.15
  target_count: 4
  random_seed: 42

agent:
  start_x: 0
  start_y: 0
  sensor_range: 3

simulation:
  max_steps: 500
```

## Setup

Create and activate a virtual environment, then install the project with
development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run

The scenario runner is prepared as the command-line entry point:

```bash
python scripts/run_scenario.py
```

The current runner is a placeholder while the ST04 simulation loop is being
implemented.

## Test and Lint

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check src tests scripts
```

## GitLab CI

The project includes a GitLab CI pipeline in [.gitlab-ci.yml](.gitlab-ci.yml).

The pipeline uses `python:3.12` and runs:

1. `ruff check src tests scripts`
2. `pytest`

CI installs the package with:

```bash
python -m pip install -e ".[dev]"
```

## Implementation Status

Completed foundation:

- Python package structure
- architecture documentation
- YAML requirements specification
- default YAML scenario file
- GitLab CI pipeline
- initial grid and scenario generation skeleton
- first generation test

Next implementation steps:

- YAML configuration loader
- deterministic movement validation
- sensor observation logic
- baseline exploration strategy
- full simulation runner
- YAML metrics output

