# Rescue Sim — Visualization Frontend

React + Vite dashboard for the rescue simulation. It connects to the FastAPI
backend (`src/rescue_sim/visualization/api.py`) over WebSocket and renders the
live grid, agent trails, MoE expert routing, per-try metrics, and the
Experts-vs-MoE comparison panel.

## Development

```bash
npm install
npm run dev        # Vite dev server on :5173, proxying the API on :8000
```

Run the backend alongside it:

```bash
uvicorn src.rescue_sim.visualization.api:app --reload --port 8000
```

## Production build

```bash
npm run build      # outputs to dist/
```

The Docker image (`Dockerfile` at the repo root) runs this build and serves
`dist/` from the backend at `/app`, so in Docker you never need to run npm
yourself — `docker compose up --build viz` does everything.

## Layout

```text
src/
├── App.jsx                 # page layout, run-mode buttons, config wiring
├── hooks/useSimulation.js  # WebSocket client + simulation state
└── components/
    ├── GridCanvas.jsx      # canvas renderer: grid, agents, trails, rescues
    ├── ControlPanel.jsx    # start/stop/restart, speed, skip-to-results
    ├── ParameterPanel.jsx  # scenario summary
    ├── StatsBar.jsx        # live episode/step/rescued counters
    ├── MoePanel.jsx        # expert routing weights, tries, adaptation board
    ├── MetricsChart.jsx    # per-episode reward/steps chart
    ├── PolicyComparison.jsx# Experts vs. MoE head-to-head (REST)
    └── EvaluationPanel.jsx # baseline comparison table (fleet mode)
```
