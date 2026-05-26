# Rescue Sim

Python simulator for the Multi-Agent Rescue Teams project.

The project goal is to build a damaged-area rescue simulation where agents can
explore an environment, detect rescue targets, communicate observations, and
improve rescue strategies over time.

## Project Status

Current sprint: Sprint 2, May 6 - May 27

Sprint 2 goal: create the first working damaged-area simulator foundation with
grid environment, Target A/B spawning, valid movement, central sensor
communication, simple visual output, and basic integration.

Planning documents:

- [Product Backlog](docs/product_backlog.md) ([PDF version](docs/product_backlog.pdf))
- [Sprint 2 Backlog](docs/sprints/sprint_2.md) ([PDF version](docs/sprints/sprint_2.pdf))
- [Sprint 3 Backlog](docs/sprints/sprint_3.md) ([PDF version](docs/sprints/sprint_3.pdf))

## Current Scope

The current implementation focuses on the Sprint 2 damaged-area simulator
foundation:

- generate grid-based rescue scenarios
- configure grid size, obstacle density, targets, start positions, sensor range, and max steps
- place obstacles and rescue targets using reproducible random seeds
- distinguish between Target A and Target B
- validate movements against walls, blocked cells, and obstacles
- provide basic sensor observations
- support basic communication between the agent and sensor model
- run a simple scenario loop
- produce basic visual/text feedback and metrics

Future increments will add autonomous exploration, single-agent learning,
multi-agent coordination, distributed learning, uncertainty, validation, and
final graphical/demo improvements.

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
|   |-- product_backlog.md     # Ordered Product Backlog
|   |-- requirements.yaml      # Project requirements
|   `-- sprints/               # Sprint Backlogs and sprint planning
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
- [Product Backlog](docs/product_backlog.md) ([PDF version](docs/product_backlog.pdf))
- [Sprint 2 Backlog](docs/sprints/sprint_2.md) ([PDF version](docs/sprints/sprint_2.pdf))
- [Sprint 3 Backlog](docs/sprints/sprint_3.md) ([PDF version](docs/sprints/sprint_3.pdf))

## Configuration

Scenarios are configured with YAML. The default scenario is in
[configs/default_scenario.yaml](configs/default_scenario.yaml).

Example structure:

```yaml
grid:
  width: 20
  height: 20
  obstacle_probability: 0.15
  target_a_count: 2
  target_b_count: 2
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

Sprint 2 work will turn this into a runnable damaged-area scenario with basic
metrics and visual/text output.

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


