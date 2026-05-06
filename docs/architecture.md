# Architecture

This document describes the architecture for the Multi-Agent Rescue Teams
simulator. The current focus is Sprint 2: the damaged-area simulator foundation.

## Goal

The system simulates a damaged rescue area where agents can move, observe nearby
cells, detect targets, communicate information, and later improve rescue
strategies through learning methods.

The architecture is built incrementally:

1. damaged-area grid simulator
2. single-agent exploration and learning
3. multi-agent coordination
4. distributed learning and uncertainty
5. graphical/demo improvements

## Package Structure

```text
src/rescue_sim/
|-- agents/          # Agent state, policies, and decision logic
|-- config/          # Scenario and simulation settings
|-- environment/     # Grid, obstacles, targets, movement, and sensing
|-- learning/        # Strategy optimization and reinforcement learning
|-- simulation/      # Scenario execution, run state, and metrics
`-- visualization/   # Rendering and plotting helpers
```

## Main Components

### Environment

The environment represents a rescue scenario as a grid. It owns the static map
data:

- grid width and height
- blocked cells and walls
- obstacle positions
- Target A positions
- Target B positions

The first implementation uses deterministic movement. A move is valid only when
the destination is inside the grid and not blocked by an obstacle.

### Scenario Generator

The generator creates reproducible scenarios from configuration values:

- map size
- obstacle probability
- target count
- random seed
- agent start position

The start position is kept free of obstacles and targets.

### Sensor / Observation Model

The sensor model returns what an agent can observe from its current position.
In Sprint 2, this includes nearby visible cells, obstacles, and targets. Later
increments will extend this to other agents, communication radius, and uncertain
sensor readings.

### Agent

The single agent stores its current position and sensor range. Decision logic is
kept separate from the raw agent state so different strategies can be compared
without changing the environment model.

### Learning and Strategy

The `learning` package contains exploration strategies. The first milestone is a
deterministic baseline explorer. Later milestones can add reinforcement learning
methods and compare them against the baseline.

### Simulation Runner

The simulation runner coordinates one scenario:

1. Load configuration.
2. Generate the grid.
3. Place the agent.
4. Repeatedly collect observations.
5. Ask the strategy for the next action.
6. Apply movement.
7. Record metrics until all targets are found or `max_steps` is reached.

## Data Flow

```text
Configuration
     |
     v
Scenario Generator --> Grid Environment
     |                      |
     v                      v
Single Agent --------> Sensor Model
     |                      |
     v                      v
Strategy / Learning --> Next Action
     |
     v
Simulation Runner --> Metrics / Results
```

## Testing Strategy

Tests should be added close to the behavior they protect:

- environment generation keeps start cells valid
- movement rejects blocked or out-of-bounds cells
- sensor observations match visible nearby cells
- runner stops when all targets are found
- baseline strategy produces deterministic results for fixed seeds

