# ST04 Design Notes

## Goal

Build a Python simulator where one rescue agent explores a generated area and
finds a known number of rescue targets while avoiding obstacles.

## Main Components

- Environment generation: creates a grid with obstacles and targets.
- Sensor model: returns observations around the agent.
- Movement model: initially deterministic moves on the grid.
- Agent policy: decides which action to take next.
- Strategy search or learning: compares policies and optimizes target discovery.
- Metrics: records steps, targets found, coverage, and completion status.

## First Implementation Milestone

Create a deterministic grid-world baseline with a simple exploration strategy.
This provides a testable foundation before reinforcement learning is added.

