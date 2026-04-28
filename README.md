# Multi-Agent Rescue Teams - ST04

Python implementation for backlog task ST04: a single rescue agent explores a
generated environment with configurable obstacles and targets, then learns or
selects a strategy to find the known number of targets.

## Project Layout

```text
.
├── configs/                  # Scenario and training configuration files
├── data/                     # Generated scenario inputs and run outputs
├── docs/                     # Design notes and task documentation
├── scripts/                  # Command-line entry points for common workflows
├── src/rescue_sim/           # Main Python package
│   ├── agents/               # Single-agent policy and behavior logic
│   ├── environment/          # Grid, obstacles, targets, sensors, movement
│   ├── learning/             # Search/RL algorithms and training loops
│   ├── simulation/           # Scenario orchestration and metrics
│   ├── visualization/        # Plotting or GUI helpers
│   └── config/               # Typed configuration loading
└── tests/                    # Unit and integration tests
```

## Task Scope

ST04 focuses on a single agent rescue team:

- Generate several scenarios with configurable obstacles and targets.
- Let one agent explore the environment.
- Assume the agent knows the number of targets.
- Find the best strategy to discover the defined target set.

