# 🚨 Rescue Sim — Multi-Agent Rescue Intelligence Platform

> **Neural Mixture-of-Experts for cooperative MARL disaster response** — from tabular baselines to attention-gated deep coordination with GRU temporal memory.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-193%20passing-brightgreen.svg)](#test-and-lint)
[![CPU Training](https://img.shields.io/badge/GPU-Not%20Required-orange.svg)](#technology)

A grid-based rescue simulator where agents explore a damaged area, detect targets,
communicate under range limits, and learn to coordinate. It implements and
**compares** a full ladder of strategies — classical baselines → decentralized
tabular RL → deep multi-agent RL → an attention-gated **Neural Mixture-of-Experts**
— all sharing one reward/observation contract so the numbers are directly
comparable. Every method trains on a normal **CPU**.

---

## ⚡ Quick Start

```bash
git clone https://github.com/AlirezaMoazen/Multi-Agent-Rescue-Intelligence-Platform.git
cd Multi-Agent-Rescue-Intelligence-Platform

# Option A: Docker (zero setup) — live MoE dashboard demo
docker compose run --rm demo-moe
# ...or the web visualization dashboard:
docker compose up --build viz          # then open http://localhost:8000/app

# Option B: Local install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
python demo_moe.py                     # full Neural MoE demonstration
pytest                                 # run the test suite
```

## 🔍 The Core Problem: Partial Observability Under Communication Blackout

Each agent sees only a **3-block radius** (a 7×7 ego window, 4 channels: blocked,
target-A, target-B, other-agent). When agents drift beyond communication range
(Manhattan distance ≥ 3) they enter **blackout** and must fall back on temporal
memory to avoid blind looping.

```
    CONNECTED (d < 3): peer count high         BLACKOUT (d ≥ 3): isolated
    → route to Expert 2 (coordination)         → route to Expert 3 (GRU fallback)
    Agent A ◄─ d=2 ─► Agent B                  Agent A ◄──── d=9 ────► Agent B
```

The attention router shifts each agent's gating weights in real time based on peer
visibility, via scaled dot-product attention over peer latent embeddings.

## Project Status

Every planned learning strategy is implemented and tested (193 tests green, CPU-only):

- **Classical baselines** (no learning) — frontier-greedy exploration and
  Artificial Potential Fields (APF) swarm navigation, single-agent **or** as a
  synchronized team.
- **Tabular RL** — a decentralized **Epidemic Hysteretic Q-Learning** fleet (up to
  20 robots) with peer-to-peer gossip.
- **Deep MARL (CTDE)** — MAPPO, QMIX, TransfQMix, plus a **ValueEnsemble** and a
  distilled student.
- **Neural Mixture-of-Experts** — attention-gated routing across three specialized
  expert heads with a GRU temporal fallback under communication blackout.

The web dashboard drives the multi-agent methods directly: pick **Epidemic Fleet**
or **Neural MoE** in the UI, watch training stream live, agents colored by dominant
expert, and isolated agents flagged with a blackout badge.

> **Legacy removed.** The original single-agent `QLearningAgent` was deleted — its
> `LearningState` keyed the Q-table on full cell-sets, so it memorized episodes
> instead of generalizing, and nothing in the multi-agent line-up depended on it.
> Collapsing its state to the grid cell `(y,x)` just re-derives the Epidemic fleet
> (same TD rule + hysteresis + gossip), so removal cost nothing.

Docs: [Architecture](docs/architecture.md) · [Requirements](docs/requirements.yaml) ·
[Neural MoE deep-dive](README_moe.md).

## Technology

Python + YAML config. Core deps (see [pyproject.toml](pyproject.toml)): `numpy`
(vectorized environment/tabular maths), `pydantic` (typed, validated API models),
`pyyaml`, `fastapi`+`uvicorn`+`websockets` (visualization server), `pytest`+`ruff`.
`torch` is **optional** — only for the deep methods; CPU-only is enough
(`pip install -e ".[moe]"`).

## Project Layout

```text
.
|-- demo_moe.py                # Full Neural MoE demonstration (train + live dashboard)
|-- configs/default_scenario.yaml
|-- docs/                      # Architecture, requirements, sprint planning
|-- scripts/
|   |-- run_scenario.py        # Run one scenario from YAML, print metrics
|   |-- train_{mappo,qmix,transfqmix}.py
|   |-- compare_all.py         # Train all -> ensemble -> distill -> compare
|   |-- pretrain_moe.py        # Pretrain the Neural MoE, save checkpoints/moe.pt
|   `-- eval_checkpoints.py    # Score saved checkpoints (no training)
|-- src/rescue_sim/
|   |-- shared.py              # Project contract, rewards, shared deep-RL helpers
|   |-- config/ environment/   # Typed settings; grid, generation, movement, sensing
|   |-- Qlearning/             # Baselines + multi-agent runner, Epidemic fleet, comms
|   |-- MAPPO/ QMIX/ TransfQMix/  # Deep MARL trainers (share RescueEnv)
|   |-- Ensemble/ MoE/         # ValueEnsemble + distillation; Neural Mixture-of-Experts
|   |-- simulation/            # Runner, evaluation, metrics
|   `-- visualization/         # FastAPI backend + React frontend
`-- tests/                     # Unit and integration tests
```

## Learning Algorithms

All methods share one contract — the same `Action` set, `Grid`, observation, and
`calculate_reward` — so results are apples-to-apples. The thread: a decentralized
fleet that shares knowledge (Epidemic) → deep nets that generalize across grids
(QMIX/TransfQMix/MAPPO) → combining them (Ensemble) → routing between experts (MoE).

| | Epidemic Hysteretic Q | QMIX | TransfQMix | MAPPO |
|---|---|---|---|---|
| Family | Value-based | Value-based | Value-based | Policy-gradient |
| Function approx. | Tabular (dense NumPy) | Deep (MLP) | Deep (Transformer) | Deep (MLP) |
| Input | Grid cell `(y,x)` | Local window vector | Entity token set | Local window vector |
| Replay buffer | No | Yes (off-policy) | Yes (off-policy) | No (on-policy) |
| Coordination | Peer gossip (max-sync) | Mixer (train only) | Mixer (train only) | Critic (train only) |
| Runtime comms | Yes (when robots meet) | No | No | No |
| Key idea | Optimistic + epidemic max-sync | Monotonic mixing | Attention over entities | Clipped policy update |

### Baselines (no learning) — `Qlearning/baseline.py`

The performance floor. **`frontier`** (frontier-greedy, scores unvisited/frontier
cells) and **`apf`** (Artificial Potential Fields, Khatib 1986: target attraction +
teammate separation + obstacle repulsion). `run_multi_agent_baseline` /
`compare_multi_agent_baselines` run any of them as a synchronized team with shared
sensor memory and deterministic collision resolution — a fair, AI-free multi-agent
reference. APF also teaches the MoE's Expert 1 and is the "Non-AI" competitor in the
dashboard's head-to-head panel.

### Epidemic Hysteretic Q-Learning (tabular, decentralized) — `Qlearning/q_learning.py`

A vectorized NumPy fleet of up to 20 robots with **no central coordinator**. Each
robot keeps its own Q-table (state = grid cell, four cardinal moves) as one
contiguous `q[slot, y, x, action]` array, and:

- **Hysteretic update** (Matignon et al., 2007): two learning rates, α for positive
  TD error and β ≪ α for negative, keeping each robot optimistic against transient
  teammate noise.
- **Epidemic max-sync**: robots within `comm_radius` merge Q-tables element-wise by
  max — monotone, so the fleet converges regardless of meeting order.
- **Bandwidth/congestion control**: dirty-delta sync, pairwise cooldown, per-robot
  link budget. **Dynamic membership**: robots fail/join mid-run in O(1).

The transport layer (`Qlearning/communications.py`) offers `DefaultCommsBus`
(perfect channel) and `ResilientCommsBus` (packet loss, bandwidth cap, delay).

### Deep MARL — QMIX, TransfQMix, MAPPO

All three are CTDE (centralized training, decentralized execution), need the `torch`
extra, train on CPU in minutes, and share one cooperative `RescueEnv` plus the
helpers in `shared.py` (replay buffer, value normalization, weight init, target sync).

- **QMIX** (`QMIX/qmix.py`; Rashid et al., ICML 2018) — value decomposition with a
  hypernetwork mixer under a monotonicity constraint (the IGM principle: per-agent
  greedy actions stay consistent with the team optimum). Double-DQN target.
- **TransfQMix** (`TransfQMix/transf_qmix.py`; Gallici et al., AAMAS 2023) — QMIX
  where both agent net and mixer are transformers over entity tokens, so the same
  parameters transfer to any number of agents.
- **MAPPO** (`MAPPO/`; Yu et al., NeurIPS 2022) — on-policy policy gradient: shared
  actor on local obs, centralized critic on global state, GAE, PPO clipped objective.

```bash
pip install -e ".[qmix]"           # or [transfqmix] / [mappo]
python scripts/train_qmix.py --episodes 200 --grid 8 --agents 4
```

```python
from rescue_sim.config.settings import GridSettings, QmixSettings
from rescue_sim.MAPPO import RescueEnv
from rescue_sim.QMIX import QMIX

grid = GridSettings(width=8, height=8, obstacle_probability=0.15, target_a_count=2, target_b_count=2)
env = RescueEnv(grid, num_agents=4, max_steps=200, view_radius=2, seed=0)
trainer = QMIX(env, QmixSettings(num_agents=4, random_seed=0))
trainer.train(num_episodes=200)
print(trainer.evaluate(episodes=20))
```

### Ensemble + distillation — `Ensemble/`

QMIX and TransfQMix are both value-based, so their Q-values combine. **ValueEnsemble**
averages them weighted by validation success; **Distiller** compresses the ensemble
into one small student net via supervised regression on its Q-values — ensemble
behavior at single-network cost. Run the pipeline (train all → ensemble → distill →
compare): `python scripts/compare_all.py` (or `docker compose run --rm compare-all`).

### 🧠 Neural Mixture-of-Experts — the technical engine — `MoE/`

The top layer: a step-level neural gate (Jacobs et al., 1991) that blends three
specialized expert heads *per agent, per step*, driven by real-time communication
topology. CTDE throughout — experts are distilled centrally from team trajectories,
but at execution each agent routes on its **local** observation and peer set only.

- **Dual-encoder topology** — a frozen *expert encoder* (CNN over the 7×7 window +
  scalar/peer-count MLPs) feeds the three heads; a separate trainable *router
  encoder* feeds the gate, so online router tuning can never corrupt the experts.
- **Attention gating router** — scaled dot-product attention over the *variable-size*
  set of visible peer embeddings (no zero-padding, no index-ordering bugs), masking
  peers outside the 3-block radius to −∞:

  $$\mathbf{g} = \operatorname{softmax}\!\Bigl(W_g \cdot \operatorname{attn}(z_{\text{ego}}, Z_{\text{peers}}, M)\Bigr) \in \Delta^2$$

- **Recurrent temporal fallback (Expert 3)** — a `GRUCell` head carries hidden state
  across the episode so an isolated agent remembers its trajectory and escapes
  dead-ends instead of looping.
- **Expert teachers** — E1 clones the **APF baseline**; E2 is distilled from all three
  trained deep-RL checkpoints (MAPPO+QMIX+TransfQMix) via calibrated-temperature
  gated reverse-KL (`MoE/gated_distill.py`); E3 clones a cold-start sweep policy, and
  during dashboard rollouts the **real Epidemic Hysteretic fleet runs live** and
  takes control whenever the router selects the fallback.
- **Blackout-aware routing** — the router is fine-tuned so `g_fallback → 1` under
  blackout and `g_coord → 1` when connected.

Live telemetry (`demo_moe.py`) shows the routing flip the moment an agent leaves the
3-block radius:

```text
Step  Agent    Pos    Peers  MoE Gating [g_exp, g_coord, g_fall]
5     Agent-0  (1,3)  3      [0.0042, 0.9948, 0.0011]
5     Agent-1  (3,1)  1      [0.0003, 0.0007, 0.9990]   ← blackout → GRU fallback
```

```bash
pip install -e ".[moe]"
python demo_moe.py                     # or: docker compose run --rm demo-moe
```

The same pipeline (`MoE/pipeline.py`) powers the web dashboard's Neural MoE mode.
Full derivations and design notes are in [README_moe.md](README_moe.md).

### Indicative results (short CPU runs)

Greedy success on freshly generated grids (new random map each episode — measures
generalization, not memorization):

| Method | Grid | Agents | Train | Greedy success |
|---|---|---|---|---|
| MAPPO | 6×6 | 3 | 40 updates | ~0.60 |
| QMIX | 6×6 | 3 | 150 episodes | ~0.65 |
| TransfQMix | 6×6 | 3 | 150 episodes | ~0.80 |

Numbers are indicative of short runs; longer training improves all three. The MoE is
measured live by `demo_moe.py` on the full 20×20 grid (behavioral cloning converges,
router blackout penalty → ~2×10⁻⁴, then `pytest tests/test_moe.py` runs as an
integration gate).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`torch` is optional — add only the extras you need (`[qmix]`, `[transfqmix]`,
`[mappo]`, `[ensemble]`, `[moe]`).

## Run

```bash
python scripts/run_scenario.py                 # default configs/default_scenario.yaml
python scripts/run_scenario.py path/to.yaml    # or a custom scenario
```

## Test and Lint

```bash
pytest
ruff check src tests scripts
```

CI ([.gitlab-ci.yml](.gitlab-ci.yml)) runs both on `python:3.12`.

## Configuration

Scenarios are YAML ([configs/default_scenario.yaml](configs/default_scenario.yaml)):

```yaml
grid:
  width: 14
  height: 14
  obstacle_probability: 0.15
  target_a_count: 2
  target_b_count: 2
  random_seed: 42
agent:
  start_x: 0
  start_y: 0
  sensor_range: 3
simulation:
  max_steps: 200

# Decentralized Epidemic Hysteretic fleet (config.settings.FleetSettings)
fleet:
  num_agents: 4           # active at episode start
  max_agents: 20          # pre-allocated capacity (1 <= N <= 20)
  alpha: 0.5              # learning rate for positive TD error
  beta: 0.1              # muted rate for negative TD error (beta << alpha)
  discount_factor: 0.95
  epsilon: 0.2
  comm_radius: 3.0        # Euclidean distance that opens a peer link
  gossip_cooldown: 5      # steps before the same pair may re-sync
  max_links_per_step: 2   # per-robot handshake budget
  utility_threshold: 0.0
  random_seed: 42
```

## Docker

Fully containerized (backend + built React frontend + CPU torch), hot-reloading
backend edits.

```bash
docker compose up --build viz          # dashboard at http://localhost:8000/app
docker compose run --rm demo-moe       # Neural MoE demonstration
docker compose run --rm dev            # interactive shell
docker compose run --rm test           # test suite
docker compose run --rm lint           # ruff
```

## References

- **Hysteretic Q** — Matignon et al., IROS 2007 · **QMIX** — Rashid et al., ICML 2018 ·
  **TransfQMix** — Gallici et al., AAMAS 2023 · **MAPPO** — Yu et al., NeurIPS 2022 ·
  **GAE** — Schulman et al., ICLR 2016.
- **Mixture-of-Experts** — Jacobs et al., 1991; Shazeer et al., ICLR 2017 ·
  **Policy distillation** — Rusu et al., ICLR 2016 · **Knowledge distillation** —
  Hinton et al., 2015 · **Algorithm selection / portfolios** — Rice 1976; SATzilla
  (Xu et al., JAIR 2008).
- **Artificial Potential Fields** — Khatib, ICRA 1985 / IJRR 1986.

## License & Academic Attribution

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).

* **Authors (Group 05)**:
 * Adriana Herrero Callejo ([github.com/adrianaherrerocallejo](https://github.com/adrianaherrerocallejo))
 * Cristina Marcos Alonso ([github.com/CristinaMarcosAlonso](https://github.com/CristinaMarcosAlonso))
 * Mohammad Mustafa Orfany ([github.com/MustafaZo77o](https://github.com/MustafaZo77o))
 * Alireza Moazzen ([alirezamoazen.com](http://alirezamoazen.com))
 * **Institution**: Hamburg University of Technology (TUHH) — Software Development SS26
 * **Supervisor**: Rainer Marrone
