# Architecture

This document describes the initial architecture for the ST04 single-agent
rescue simulator.

## Goal

The system simulates one rescue agent exploring a generated damaged area. The
agent knows how many targets exist and must find all targets while avoiding
obstacles.

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
- obstacle positions
- target positions

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

### Sensor Model

The sensor model returns the agent's local observation. In the baseline version,
this should include visible cells, obstacles, and targets within the configured
sensor range.

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

