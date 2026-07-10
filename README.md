# 🚨 Rescue Sim — Multi-Agent Rescue Intelligence Platform

> **Neural Mixture-of-Experts for cooperative MARL disaster response** — from tabular baselines to attention-gated deep coordination with GRU temporal memory.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-200%20passing-brightgreen.svg)](#test-and-lint)
[![CPU Training](https://img.shields.io/badge/GPU-Not%20Required-orange.svg)](#technology)

---

## ⚡ 1-Minute Quick Start

```bash
# Clone the repository
git clone https://collaborating.tuhh.de/e16/courses/software-development/ss26/group05.git
cd group05

# Option A: Docker (recommended — zero setup)
# One command: builds the multi-stage container and launches the live 20x20
# MoE simulation (training dashboard + ASCII grid + telemetry + pytest gate)
docker compose run --rm demo-moe

# Or start the web visualization dashboard instead:
docker compose up --build viz
# Open http://localhost:8000/app in your browser

# Option B: Local install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
python demo_moe.py            # Run the full MoE dashboard demonstration
pytest                         # Verify the full test suite passes
```

---

## 🔍 The Core Problem: Partial Observability Under Communication Blackout

Each agent operates within a **3-block visibility radius** — a 7×7 ego-centric observation window.  When agents drift beyond communication range (Manhattan distance ≥ 3), they enter **blackout** and must rely on temporal memory to avoid blind looping.

```
    3-BLOCK BLINDNESS CONSTRAINT              COMMUNICATION BLACKOUT RULES
    (7x7 Ego-Centric Window)                  (Manhattan Distance Threshold)

     . . . . . . . . . . .                   Agent A ◄──── d=2 ────► Agent B
     . . . . . . . . . . .                        (CONNECTED: d < 3)
     . . ┌───────────┐ . .                        Peer count = 2
     . . │ . . . . . │ . .                        → Route to Expert 2 (Coordination)
     . . │ . . . . . │ . .
     . . │ . . A . . │ . .  ← Agent             Agent A ◄────── d=9 ──────► Agent B
     . . │ . . . . . │ . .                        (BLACKOUT: d ≥ 3)
     . . │ . . . . . │ . .                        Peer count = 1
     . . └───────────┘ . .                        → Route to Expert 3 (GRU Fallback)
     . . . . . . . . . . .                        → Temporal memory prevents looping
     . . . . . . . . . . .

    Agent sees ONLY the 7x7 box.             The Attention Router dynamically shifts
    4 channels: blocked, target-A,           gating weights based on real-time
    target-B, other-agent.                   peer visibility via scaled dot-product
    Everything outside = unknown.            attention over peer latent embeddings.
```

---

## Project Status

The simulator foundation is complete, and every planned learning strategy is implemented and tested:

- **Classical baselines** (no learning) — frontier-greedy exploration and Artificial Potential Fields (APF) swarm navigation, runnable single-agent **and** as a synchronized multi-agent team.
- **Tabular RL** — single-agent Q-learning and a decentralized **Epidemic Hysteretic Q-Learning** fleet with peer-to-peer gossip.
- **Deep MARL (CTDE)** — MAPPO, QMIX, and TransfQMix, plus a **ValueEnsemble** and a distilled student that combine the two value methods.
- **Mixture-of-Experts** — attention-gated routing across three specialized expert heads with GRU-based temporal fallback under communication blackout.

The whole test suite is green and every method trains on a normal CPU.

> **Live web dashboard.** The visualization API and React frontend drive the
> multi-agent methods directly: pick **Epidemic Fleet** (tabular hysteretic
> Q-learning with gossip) or **Neural MoE** (three experts blended by the
> attention router) in the UI. The MoE mode streams its training progress
> live, then solves the *same* fixed grid over repeated tries — agents are
> colored by their dominant expert, isolated agents get a dashed blackout
> radius and a BLACKOUT badge, and the router panel shows each agent's
> softmax routing vector in real time.
>
> **Legacy code removed.** The original *single-agent* flow (`QLearningAgent`
> and its evaluation-panel comparator path) has been **deleted** — it was
> superseded by the multi-agent line-up above and nothing production-facing
> depended on it anymore. See
> [Q-Learning (removed)](#q-learning-tabular-single-agent--removed) for what
> the problem was and why removal was the right fix.

Architecture documents:

- [Architecture Design](docs/architecture.md)
- [Requirements Specifications](docs/requirements.yaml)


## Current Scope

The damaged-area simulator foundation provides:

- grid-based rescue scenarios with configurable size, obstacle density, targets,
  start positions, sensor range, and max steps
- reproducible obstacle/target placement via random seeds, with distinct Target
  A and Target B
- movement validation against walls, blocked cells, and obstacles
- central-sensor observations and a scenario/episode loop
- text and web (React) visual feedback plus per-episode metrics

On top of that foundation, the project implements and **compares** a ladder of
rescue strategies — from non-learning baselines, through tabular and
decentralized RL, up to deep multi-agent RL and a mixture-of-experts gate (see
[Learning Algorithms](#learning-algorithms)). Every method scores moves through
the same reward/observation contract, so the comparison is apples-to-apples.

## Technology

Application code is written in **Python**.

Configuration and machine-readable output should use **YAML**.

Main dependencies are declared in [pyproject.toml](pyproject.toml):

- `numpy` — all environment/tabular maths (vectorized)
- `pydantic` — typed API request/response models
- `pyyaml` — YAML scenario configuration
- `fastapi` + `uvicorn` + `websockets` — the visualization API/server
- `pytest` for tests, `ruff` for linting
- `torch` — **optional**, only for the deep methods (MAPPO/QMIX/TransfQMix/
  Ensemble/MoE). CPU-only is enough; install via the relevant extra, e.g.
  `pip install -e ".[ensemble]"`.

## Project Layout

```text
.
|-- .gitlab-ci.yml             # GitLab CI pipeline (ruff + pytest)
|-- Dockerfile                 # Container image (backend + built frontend + CPU torch)
|-- docker-compose.yml         # viz / dev / test / lint / train-* / compare-all / demo-moe
|-- configs/
|   `-- default_scenario.yaml  # Example YAML scenario configuration
|-- docs/                      # Architecture, backlog, requirements, sprint planning
|-- scripts/
|   |-- run_scenario.py        # Scenario runner entry point
|   |-- train_mappo.py         # Train MAPPO
|   |-- train_qmix.py          # Train QMIX
|   |-- train_transfqmix.py    # Train TransfQMix
|   `-- compare_all.py         # Train all -> ensemble -> distill -> compare table
|-- src/rescue_sim/
|   |-- shared.py              # Project contract + shared deep-RL helpers
|   |-- config/                # YAML loading and typed settings
|   |-- environment/           # Grid, generation, movement, sensing
|   |-- Qlearning/             # Tabular: baselines, single-agent Q, Epidemic fleet,
|   |                          #   gossip comms, multi-agent baseline adapter
|   |-- MAPPO/                 # RescueEnv + MAPPO (policy-gradient, CTDE)
|   |-- QMIX/                  # QMIX (monotonic value decomposition, CTDE)
|   |-- TransfQMix/            # TransfQMix (transformer value decomposition, CTDE)
|   |-- Ensemble/              # ValueEnsemble + distillation of QMIX + TransfQMix
|   |-- MoE/                   # Neural MoE: attention router + GRU fallback + CTDE experts
|   |-- simulation/            # Simulation runner, evaluation, metrics
|   `-- visualization/         # FastAPI backend + React frontend
`-- tests/                     # Unit and integration tests
```

## Learning Algorithms

The project compares a ladder of rescue strategies against non-learning
baselines. They all share one contract — the same `Action` set, `Grid`,
observation, and `calculate_reward` — so the numbers are directly comparable.

**Status:** ✅ all implemented — **Baselines** (frontier greedy + APF potential fields, single- and multi-agent), **Epidemic Hysteretic Q-Learning** (decentralized multi-agent), **MAPPO** (deep, policy-gradient, CTDE), **QMIX** (deep, value-decomposition, CTDE), **TransfQMix** (deep, transformer-based, CTDE), a **ValueEnsemble** + distilled student, and a **Mixture-of-Experts** gate. The original single-agent **Q-Learning** rung has been removed (see its section below).

> **Reading guide for juniors.** Each section below states *what problem the
> method solves*, *the key equation in plain symbols*, and *how it differs from
> its neighbours*. The thread connecting them: a single agent that memorizes
> (the removed Q-learning rung) → a decentralized fleet that shares knowledge
> (Epidemic) → deep nets that generalize across grids (QMIX/TransfQMix/MAPPO)
> → combining them (Ensemble) → and finally routing between a generalist and a
> specialist (MoE).

---

### Baseline (no learning)

Defined in `src/rescue_sim/Qlearning/baseline.py`. None of these use a reward
signal — they are the **performance floor** every learning method must beat.

*Non-learning strategies (both honestly decentralized — every decision comes
from the agent's own sensing):*

- **BaselineExplorer** (`frontier`) — frontier-greedy: scores candidate moves +2 for unvisited cells and +1 for frontier adjacency; always picks the best score.
- **APFExplorer** (`apf`) — Artificial Potential Fields (Khatib 1986): per-agent force sum of target attraction, teammate separation (spreads the swarm's joint sensor footprint), obstacle repulsion, and open-space attraction. Also serves as the teacher for the MoE's Expert 1 and as the **"Non-AI (APF)"** competitor in the dashboard's head-to-head panel — the gap between it and the ML policies quantifies what learning buys. (The earlier centralized CBS planner was removed: a central plan cannot be executed from local observations, so it fits neither the CTDE setting nor the MoE.)

**Multi-agent runner.** `run_multi_agent_baseline` /
`compare_multi_agent_baselines` (`src/rescue_sim/Qlearning/multi_agent_baseline.py`)
run **any** of these strategies as a synchronized team on one shared grid:
shared sensor memory, deterministic collision resolution (a move into an
occupied/reserved cell is cancelled and counted), and team-level metrics
(success, rescued, steps, collisions, per-agent reward). This gives the MARL
methods a *fair, AI-free multi-agent* comparison point. Run it:

```python
from rescue_sim.config.settings import GridSettings
from rescue_sim.Qlearning.multi_agent_baseline import compare_multi_agent_baselines

gs = GridSettings(width=8, height=8, obstacle_probability=0.1,
                  target_a_count=2, target_b_count=2, random_seed=7)
results = compare_multi_agent_baselines(gs, num_agents=3, max_steps=200, seed=7)
for name, m in results.items():
    print(name, m.success, f"{m.rescued_targets}/{m.total_targets}", m.steps, m.collisions)
```

---

### Q-Learning (tabular, single-agent) — ❌ removed

> **Status:** the project's *first* RL rung, `QLearningAgent`, has been
> **deleted** from `src/rescue_sim/Qlearning/q_learning.py`, together with its
> comparator path in `simulation/evaluation.py` and the "Legacy Q-learning" row
> in the evaluation panel. Its multi-agent successor is the **Epidemic
> Hysteretic fleet** below, which keeps the exact same TD update at its core.

It was standard temporal-difference learning with an ε-greedy policy:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**What was the problem?**

1. **State explosion → memorisation, not learning.** The learner keyed its
   Q-table on the full `LearningState` — entire *sets* of visible cells,
   discovered cells, and remaining targets. Almost every step produced a
   brand-new, never-seen-again key, so the table memorised individual episodes
   instead of generalising. Its evaluation numbers were effectively noise.
2. **Dead code kept on life support.** Nothing in the multi-agent line-up
   (Epidemic fleet, QMIX, TransfQMix, MAPPO, Ensemble, MoE) used it. It
   survived only because the evaluation panel still rendered a "Legacy
   Q-learning" comparator row — a row whose numbers were misleading (point 1).
3. **Maintenance drag.** Two Q-learners in one module meant duplicated update
   logic, extra tests, extra imports (`MovementModel`, `CentralSensor`,
   reward plumbing), and a confusing "which one is real?" question for every
   new reader.

**Why is deletion the best solution?**

- *Fixing* the state key would just re-derive the Epidemic fleet: collapse the
  state to the grid cell $(y,x)$ and you get exactly the representation the
  fleet already uses — with the identical TD rule, plus hysteresis and gossip
  on top. There is no behaviour the legacy learner offered that the fleet
  does not supersede.
- Removing it deleted an entire misleading comparator from the evaluation
  report, so every remaining row (`epidemic_hysteretic_q`, baselines, deep RL)
  now measures something meaningful on the same grid.
- The single-agent → multi-agent story is preserved *here in the docs* where
  it belongs, instead of as ~300 lines of unused runtime code that every
  refactor had to keep compiling and every test run had to keep training.

---

### Epidemic Hysteretic Q-Learning (tabular, decentralized multi-agent) — ✅ implemented

A vectorized NumPy learner for a **decentralized fleet of up to 20 robots**, implemented as `EpidemicHystereticQLearning` in `src/rescue_sim/Qlearning/q_learning.py`. There is **no central coordinator**: each robot keeps its own Q-table and only shares knowledge with peers it physically meets.

**State and actions.** State is simply the robot's grid cell $s = (y, x)$; actions are the four cardinal moves $\{\text{N}, \text{S}, \text{E}, \text{W}\}$. The whole fleet is stored as one contiguous array `q[slot, y, x, action]` (float32) so every operation below is a single vectorized NumPy expression instead of a per-robot Python loop.

**1. Local hysteretic update (Matignon et al., 2007)**

Two separate learning rates replace the single $\alpha$:

$$Q_i(s, a) \leftarrow Q_i(s, a) + \begin{cases} \alpha \cdot \delta & \text{if } \delta \geq 0 \\ \beta \cdot \delta & \text{if } \delta < 0 \end{cases}, \qquad \beta \ll \alpha$$

where $\delta = r + \gamma \max_{a'} Q_i(s', a') - Q_i(s, a)$ is the TD error.

*Why:* in a cooperative fleet a teammate's exploration (or a dropout) can briefly make a good action look bad (negative $\delta$). Muting negative updates with $\beta$ keeps each robot **optimistic**, so transient teammate noise cannot erase an already-good policy.

**2. Epidemic max-sync**

When two robots come within Euclidean distance $d < r_{\text{comm}}$ (default 3) they merge Q-tables with an **element-wise maximum**:

$$Q_i(s, a) \leftarrow \max\bigl(Q_i(s, a),\ Q_j(s, a)\bigr) \quad \forall (s, a)$$

The max is **monotone** — values only ever increase — so the fleet converges regardless of the order in which robots happen to meet. A robot that discovers a rescue target propagates that high value to every teammate it later encounters, and they pass it on in turn (hence *epidemic*).

**3. Bandwidth minimization (dirty-delta sync)**

Sending a whole Q-table over a brief connection window is wasteful. Each robot keeps a **dirty mask** marking entries changed since its last exchange; only those (optionally filtered by a utility threshold) are serialized into a compact `GossipMessage`:

$$\text{payload} = \{(\text{index}, Q_i[\text{index}]) : \text{dirty}_i[\text{index}] \land |Q_i[\text{index}]| \ge \tau\}$$

Imported improvements are re-marked dirty, so new knowledge keeps spreading without re-sending unchanged entries.

**4. Congestion control**

When robots cluster, the number of candidate links explodes. Two throttles bound the chatter:

- **Pairwise cooldown** — a given pair may re-sync only every $c$ steps.
- **Per-robot link budget** — each robot performs at most $k$ syncs per step; the closest pairs get priority (handshake ordering).

**5. Dynamic membership**

Slots are pre-allocated to capacity and gated by an `active` mask, so robots can **fail or join mid-operation** in $O(1)$ with no reallocation. A removed robot's Q-table is retained for a possible rejoin (`forget_agent` releases the slot entirely).

**Communication boundary.** The learner owns the *mechanism* (proximity detection, delta export/import, max-merge, throttling). The physical *transport* — line-of-sight, packet loss, latency, bandwidth budgeting — lives in `src/rescue_sim/Qlearning/communications.py`, which ships two buses with one `exchange(fleet) -> int` method:

- **`DefaultCommsBus`** — the perfect-channel baseline; delegates straight to `fleet.gossip()`.
- **`ResilientCommsBus`** — a realistic channel with three independent, configurable impairments: probabilistic **packet loss** (`drop_prob`), a **bandwidth cap** (`max_entries_per_message`, keeps the top-|Q| entries), and **transmission delay** (`delay_steps`, store-and-forward). It also records `CommunicationStats` (syncs, drops, entries sent/improved) so the report can quantify the cost of the channel.

**Step-loop contract** (one timestep): `select_actions` → environment applies the moves → `record_transitions` (hysteretic update) → `bus.exchange(fleet)` (epidemic max-sync over the chosen channel).

```python
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning
from rescue_sim.shared import HystereticConfig, GossipConfig

fleet = EpidemicHystereticQLearning(grid, HystereticConfig(), GossipConfig(), max_agents=20, seed=0)
fleet.add_agent("r1", start)          # robots may join (or fail via remove_agent) any time
fleet.add_agent("r2", start2)

from rescue_sim.Qlearning.communications import DefaultCommsBus
bus = DefaultCommsBus()               # or ResilientCommsBus(drop_prob=0.3, delay_steps=2)

for _ in range(max_steps):
    actions = fleet.select_actions()  # {agent_id: action_index in 0..3 = N,S,E,W}
    rewards, next_positions, dones = environment_step(actions)
    fleet.record_transitions(actions, rewards, next_positions, dones)
    bus.exchange(fleet)               # epidemic max-sync over the chosen channel
```

---

### QMIX (deep, multi-agent, value-based) — ✅ implemented

Implemented in `src/rescue_sim/QMIX/qmix.py` (reuses `RescueEnv` from the MAPPO package).
Reference: Rashid et al., *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL*, ICML 2018.

Requires the optional torch dependency: `pip install -e ".[qmix]"`. CPU-only —
trains on a normal laptop in minutes (no GPU needed).

**Core idea — value decomposition under a monotonicity constraint:**

Each agent $i$ has an individual Q-network:

$$Q_i(o_i, a_i;\ \theta_i)$$

A mixing network combines them into a team Q-value:

$$Q_{\text{tot}}(\mathbf{o}, \mathbf{a};\ \phi) = f_\phi\bigl(Q_1, \ldots, Q_n,\ s_{\text{global}}\bigr)$$

subject to the **monotonicity constraint**:

$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0 \quad \forall i$$

This constraint enforces the **Individual-Global-Max (IGM) principle**:

$$\arg\max_{\mathbf{a}} Q_{\text{tot}}(\mathbf{o}, \mathbf{a}) = \bigl(\arg\max_{a_1} Q_1(o_1, a_1),\ \ldots,\ \arg\max_{a_n} Q_n(o_n, a_n)\bigr)$$

i.e. the individual greedy policy $\arg\max_{a_i} Q_i$ is consistent with the joint greedy policy over $Q_{\text{tot}}$ — agents can act locally (decentralized execution) while optimising a shared team objective learned centrally (**CTDE**).

**Mixing network (hypernetwork):**

$$W_1 = \left| \text{Hyper}_1(s)\right|, \quad \mathbf{b}_1 = \text{Hyper}_{b1}(s)$$
$$\mathbf{h} = \text{ELU}(W_1 \cdot \mathbf{Q}_{\text{ind}} + \mathbf{b}_1)$$
$$W_2 = \left| \text{Hyper}_2(s)\right|, \quad b_2 = V(s)$$
$$Q_{\text{tot}} = W_2 \cdot \mathbf{h} + b_2$$

Absolute values $|\cdot|$ enforce non-negative weights (monotonicity).

**Loss (with Double DQN target):**

$$\mathcal{L} = \left( Q_{\text{tot}} - \left[ r + \gamma \cdot \bar{Q}_{\text{tot}}\!\left(\mathbf{o}', \arg\max_{\mathbf{a}'} Q_{\text{ind}}(\mathbf{o}')\right) \right] \right)^2$$

where $\bar{Q}$ is the target network (updated every $N$ steps).

**Observation encoding (shared `RescueEnv`, same as MAPPO):**

Each agent encodes its local observation as a fixed-size vector:

- Egocentric grid window $(2r+1)^2 \times 4$ channels: `[blocked, target-A, target-B, other-agent]`
- Scalars: normalised position $(x/W,\ y/H)$, step fraction $t/t_{\max}$, fraction of targets remaining
- One-hot agent id (so the shared network can tell agents apart)

The mixer's global state is all agent observations concatenated.

**Implementation notes:**

- Feed-forward QMIX (no RNN) with a per-transition replay buffer — the smallest
  variant that trains well on a fully observable grid.
- Double-DQN target (online net selects the next action, target net evaluates it).
- Hard target sync every `target_update_interval` learn steps; linear ε-decay.
- Parameter sharing: one `AgentQNet` for all agents.

**Run it:**

```bash
pip install -e ".[qmix]"
python scripts/train_qmix.py --episodes 200 --grid 8 --agents 4
# or in Docker:  docker compose run --rm train-qmix
```

```python
from rescue_sim.config.settings import GridSettings, QmixSettings
from rescue_sim.MAPPO import RescueEnv
from rescue_sim.QMIX import QMIX

grid = GridSettings(width=8, height=8, obstacle_probability=0.15,
                    target_a_count=2, target_b_count=2)
env = RescueEnv(grid, num_agents=4, max_steps=200, view_radius=2, seed=0)
trainer = QMIX(env, QmixSettings(num_agents=4, random_seed=0))
trainer.train(num_episodes=200)
print(trainer.evaluate(episodes=20))
```

---

### TransfQMix (deep, multi-agent, transformer + value-based) — ✅ implemented

Implemented in `src/rescue_sim/TransfQMix/transf_qmix.py` (reuses `RescueEnv` and QMIX's replay buffer).
Reference: Gallici, Martin, Masmitja, *TransfQMix: Transformers for Leveraging the Graph Structure of MARL Problems*, AAMAS 2023.

Requires the optional torch dependency: `pip install -e ".[transfqmix]"`. CPU-only
(transformers are heavier than the MLP methods, so training is slower but still
runs on a laptop).

**It is QMIX with transformer networks.** Both the agent network *and* the mixer
are transformers over a set of entity tokens — so the **same parameters transfer
to any number of agents/entities** (TransfQMix's headline property).

**Entity tokenisation.** Each agent's observation is a *set of tokens* — one per
visible cell plus a self token — instead of a flat vector:

$$e_i = \bigl[\mathbb{1}_{\text{blocked}},\ \mathbb{1}_{\text{target-A}},\ \mathbb{1}_{\text{target-B}},\ \mathbb{1}_{\text{other-agent}},\ \tfrac{\Delta x}{r},\ \tfrac{\Delta y}{r},\ \mathbb{1}_{\text{self}},\ \tfrac{t}{t_{\max}},\ \rho_{\text{remaining}}\bigr]$$

**Agent transformer.** A learnable CLS token is prepended; multi-head
self-attention pools the entities; the CLS output gives the Q-values and a hidden
embedding:

$$\mathbf{H} = \text{TransformerEncoder}\bigl([\mathbf{z}_{\text{CLS}};\ E W_{\text{in}}]\bigr), \quad Q_i = W_{\text{out}}\,\mathbf{H}[\text{CLS}], \quad h_i = \mathbf{H}[\text{CLS}]$$

**Transformer mixer.** A second transformer runs over the agent hidden states
$h_i$ plus a global-state token, and emits **non-negative** mixing weights (via
$|\cdot|$), so $Q_{\text{tot}}$ stays monotonic in each $Q_i$ — the QMIX
guarantee, but with transformer-generated weights:

$$Q_{\text{tot}} = \mathbf{w}_2^{\top}\,\text{ELU}\!\Bigl(\textstyle\sum_i Q_i\,\mathbf{w}_{1,i} + \mathbf{b}_1\Bigr) + V(s), \qquad \mathbf{w}_1, \mathbf{w}_2 \ge 0$$

**Implementation notes:** Double-DQN target, hard target sync, linear ε-decay,
and parameter sharing — same training loop as QMIX, only the networks differ.

**Run it:**

```bash
pip install -e ".[transfqmix]"
python scripts/train_transfqmix.py --episodes 200 --grid 8 --agents 4
# or in Docker:  docker compose run --rm train-transfqmix
```

```python
from rescue_sim.config.settings import GridSettings, TransfQmixSettings
from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX

grid = GridSettings(width=8, height=8, obstacle_probability=0.15,
                    target_a_count=2, target_b_count=2)
env = EntityRescueEnv(grid, num_agents=4, max_steps=200, view_radius=2, seed=0)
trainer = TransfQMIX(env, TransfQmixSettings(num_agents=4, random_seed=0))
trainer.train(num_episodes=200)
print(trainer.evaluate(episodes=20))
```

---

### MAPPO (deep, multi-agent, policy-gradient) — ✅ implemented

Implemented in `src/rescue_sim/MAPPO/` (`environment.py` + `mappo.py`).
Reference: Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022.

Requires the optional torch dependency: `pip install -e ".[mappo]"`. CPU-only —
trains on a normal laptop in minutes (no GPU needed).

**Architecture — Centralised Training, Decentralised Execution (CTDE):**

- **Actor** $\pi(a \mid o_i;\ \theta)$ — shared across all agents (parameter sharing); takes local observation only.
- **Critic** $V(s;\ \phi)$ — centralised; takes the concatenated global state $s = [o_1, \ldots, o_n]$ during training.

**Advantage estimation (GAE):**

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

$\lambda \in [0,1]$ trades bias vs. variance in the advantage estimate.

**PPO clipped objective:**

$$r_t(\theta) = \frac{\pi(a_t \mid o_t;\ \theta)}{\pi(a_t \mid o_t;\ \theta_{\text{old}})}$$

$$\mathcal{L}_{\text{CLIP}} = \mathbb{E}\!\left[\min\!\left(r_t \hat{A}_t,\ \operatorname{clip}(r_t,\ 1{-}\varepsilon,\ 1{+}\varepsilon)\hat{A}_t\right)\right]$$

**Value loss:**

$$\mathcal{L}_V = \text{MSE}\!\left(V(s_t;\ \phi),\ \hat{V}_t^{\text{target}}\right)$$

**Total loss:**

$$\mathcal{L} = -\mathcal{L}_{\text{CLIP}} + c_v \mathcal{L}_V - c_e \mathcal{H}[\pi]$$

where $\mathcal{H}[\pi]$ is the policy entropy bonus encouraging exploration.

**Key difference from QMIX:** MAPPO is **on-policy** (no replay buffer); it collects full rollouts, updates parameters, then discards the data. This is slower sample-wise but more stable.

**Implementation details (following the 5 MAPPO tricks from Yu et al.):**

- **Parameter sharing** — one `ActorCritic` network shared by all agents.
- **Value normalization** — running mean/std on value targets (`RunningMeanStd`).
- **Centralised value input** — critic sees `[o_1, …, o_n]` concatenated.
- **Clipping** — both the policy ratio *and* the value update are clipped to `±ε`.
- Plus GAE, advantage normalization, an entropy bonus, orthogonal init, and
  gradient clipping. The environment is cooperative (shared team reward) and
  reuses the existing grid/movement/reward contracts so results are comparable
  to the Q-learning baselines.

**Run it:**

```bash
pip install -e ".[mappo]"
python scripts/train_mappo.py --updates 100 --grid 8 --agents 4
# or in Docker:  docker compose run --rm train-mappo
```

```python
from rescue_sim.config.settings import GridSettings, MappoSettings
from rescue_sim.MAPPO import MAPPO, RescueEnv

grid = GridSettings(width=8, height=8, obstacle_probability=0.15,
                    target_a_count=2, target_b_count=2)
env = RescueEnv(grid, num_agents=4, max_steps=200, view_radius=2, seed=0)
trainer = MAPPO(env, MappoSettings(num_agents=4, random_seed=0))
trainer.train(num_updates=100)
print(trainer.evaluate(episodes=20))   # greedy success rate / steps
```

---

### Algorithm Comparison

| | Epidemic Hysteretic Q | QMIX | TransfQMix | MAPPO |
|---|---|---|---|---|
| Status | ✅ implemented | ✅ implemented | ✅ implemented | ✅ implemented |
| Family | Value-based | Value-based | Value-based | Policy-gradient |
| Agents | Multi (decentralized) | Multi | Multi | Multi |
| Function approx. | Tabular (dense NumPy) | Deep (MLP) | Deep (Transformer) | Deep (MLP) |
| State / input | Grid cell $(y,x)$ | Local window vector | Entity token sequence | Local window vector |
| Replay buffer | No | Yes (off-policy) | Yes (off-policy) | No (on-policy) |
| Coordination | Peer gossip (max-sync) | Mixer (training only) | Mixer (training only) | Critic (training only) |
| Runtime comms | Yes (when robots meet) | No | No | No |
| Key innovation | Optimistic + epidemic max-sync | Monotonic mixing | Attention over entities | Clipped policy update |
| PC trainable | Yes | Yes | Yes | Yes |

---

### Code structure (how the methods share code)

The methods are deliberately small because they share their plumbing — one
environment, one reward/observation contract, one set of helpers — so each
algorithm file tells one clear story and there is no copy-paste between them:

```text
src/rescue_sim/
├── shared.py            # the project contract + shared deep-RL helpers:
│                        #   Action, CARDINAL_ACTIONS, RewardConfig, Grid, ... AND
│                        #   ReplayBuffer, RunningMeanStd, orthogonal_init, hard_update
│                        #   (torch is imported lazily so shared.py stays import-light)
├── Qlearning/
│   ├── baseline.py            # frontier greedy + APF potential fields (non-AI)
│   ├── q_learning.py          # single-agent Q + Epidemic Hysteretic fleet
│   ├── communications.py      # Default/Resilient gossip buses (the channel)
│   └── multi_agent_baseline.py# run any baseline as a synchronized team
├── MAPPO/
│   ├── environment.py   # RescueEnv — the cooperative env (pure NumPy, vectorized)
│   └── mappo.py         # policy-gradient trainer
├── QMIX/qmix.py         # value-decomposition trainer (reuses RescueEnv)
├── TransfQMix/transf_qmix.py  # transformer trainer (reuses RescueEnv + buffer)
├── Ensemble/
│   ├── ensemble.py      # ValueEnsemble: combine QMIX + TransfQMix at test time
│   └── distill.py       # Distiller: compress the ensemble into one student net
└── MoE/moe.py           # Neural Mixture-of-Experts: dual encoders, attention
                         #   gating router, GRU temporal fallback head, logit
                         #   blending — demonstrated end-to-end by demo_moe.py
```

- **One environment.** `RescueEnv` is written once and reused by all the deep
  methods; TransfQMix extends it (`EntityRescueEnv`) only to add entity tokens.
- **One set of helpers.** Replay buffer, value normalization, weight init, and
  target-network sync all live in `shared.py` — no copy-paste between methods.
- **One reward/observation contract.** Every method scores moves through
  `shared.calculate_reward`, so results are directly comparable to the baselines.
- **Fast on CPU.** The environment precomputes padded NumPy maps at reset, so an
  observation is an array *slice* rather than a Python per-cell loop (the obs
  output is byte-for-byte identical to the simple version — there's a regression
  test for it).

### Ensemble + distillation (combining the best methods)

QMIX and TransfQMix are both value-based, so their per-agent Q-values are
comparable and can be combined. Implemented in `src/rescue_sim/Ensemble/`:

- **ValueEnsemble** — averages the two methods' Q-values, weighted by each one's
  validation success (so the stronger method dominates), and takes the best valid
  action. No retraining; runs both networks at test time.

  $$a_i = \arg\max_{a\ \text{valid}}\bigl(w_q\,Q^{\text{QMIX}}_i(a) + w_t\,Q^{\text{TransfQMix}}_i(a)\bigr)$$

- **Distiller** — *ensemble policy distillation*: the ensemble is the **teacher**;
  a single small network (the **student**) is trained by supervised regression to
  match the teacher's Q-values from the local observation alone. Result:
  ensemble-level behaviour at **single-network** cost — one deployable policy.

  $$\mathcal{L}_{\text{distill}} = \mathbb{E}\bigl[(Q^{\text{student}}(o) - Q^{\text{ensemble}}(o))^2\bigr]$$

Only QMIX + TransfQMix are combined: the tabular methods know just one grid, and
MAPPO outputs probabilities (not Q-values), so neither mixes cleanly.

Run the whole pipeline (train all → ensemble → distill → compare):

```bash
pip install -e ".[ensemble]"
python scripts/compare_all.py          # or: docker compose run --rm compare-all
```

### Neural Mixture-of-Experts — The Technical Engine — ✅ implemented

Implemented in `src/rescue_sim/MoE/moe.py`, demonstrated live by `demo_moe.py`.
This is the project's **top layer**: a step-level neural gate (Jacobs et al.
1991) that blends three specialized expert heads *per agent, per step*, driven
by real-time communication topology. It follows the **CTDE** paradigm: experts
are distilled centrally from team trajectories, but at execution time each
agent routes on its **local** observation and peer set only.

**Dual-Encoder topology.** Two independent `SharedFeatureEncoder`s (CNN over
the 7×7 ego window + MLPs for scalars/peer count) isolate learning: a frozen
*expert encoder* feeds the three heads, a trainable *router encoder* feeds the
gate — so online router fine-tuning can never corrupt the distilled experts.

**Attention-based gating router.** The rigid MLP gate is replaced with scaled
dot-product attention over the *variable-size set* of visible peer embeddings —
no zero-padding, no index-ordering failures:

$$\mathbf{q} = W_Q z_{\text{ego}}, \quad K = W_K Z_{\text{peers}}, \quad V = W_V Z_{\text{peers}}$$

$$\text{ctx} = \operatorname{softmax}\!\Bigl(\tfrac{\mathbf{q}K^{\top}}{\sqrt{d}} + M\Bigr)V, \qquad \mathbf{g} = \operatorname{softmax}\bigl(W_g\,\text{ctx}\bigr) \in \Delta^{2}$$

where $M$ masks agents outside the 3-block communication radius to $-\infty$.

**Recurrent temporal fallback (Expert 3).** A `GRUCell` head carries a hidden
state $h_t$ across the episode timeline, so an isolated agent remembers its
trajectory history and escapes dead-ends instead of blind looping:

$$h_t = \operatorname{GRU}(z_t,\ h_{t-1}), \qquad y^{(3)} = W_o\,h_t$$

**Expert teachers (what each head is distilled from).** E1 clones the
**APF baseline** (`Qlearning/baseline.py::APFExplorer`) — the real non-AI
swarm algorithm. E2 is distilled from **all three trained deep-RL
checkpoints** (MAPPO + QMIX + TransfQMix) via calibrated-temperature gated
reverse-KL (`MoE/gated_distill.py`), and the router is then retrained on
**outcome labels** (which expert's action the trained teachers rate best per
visited state) with a sharpened, near winner-take-all gate. E3's GRU head
clones a learnable sweep policy for cold-start — and during dashboard
rollouts the **real Epidemic Hysteretic Q fleet runs live**: it learns from
every transition, gossips Q-tables through the comms layer, persists its
tables across tries on the fixed grid, and takes control whenever the router
selects the fallback expert. (Distilling the epidemic learner offline was
tried and measured at 0-8% MoE success vs ~57% — its policy is keyed to a
per-grid Q-table that local observations cannot expose, so it must run live.)

**Logit blending mechanics.** The final policy logits are the gate-weighted sum
of the three expert heads, then invalid moves are masked before the softmax:

$$y_{\text{final}} = \sum_{j=1}^{3} g_j\, y^{(j)}, \qquad y_{\text{final}}[a] \leftarrow -10^{9} \ \ \forall a \notin \mathcal{A}_{\text{valid}}$$

**Blackout-aware router training.** After behavioral cloning of the heads, the
router is fine-tuned with a Conditional Indicator Mask penalty that forces
$g_{\text{fallback}} \to 1$ under blackout and $g_{\text{coord}} \to 1$ when
connected:

$$\mathcal{L}_{\text{route}} = \mathbb{1}[\text{peers}=1]\,(1 - g_{\text{fallback}})^2 + \mathbb{1}[\text{peers}=A]\,(1 - g_{\text{coord}})^2$$

Live telemetry from `demo_moe.py` shows the routing flip the moment an agent
leaves the 3-block radius:

```text
Step  Agent    Pos     Peers  Baseline Params               MoE Gating [g_exp, g_coord, g_fall]
5     Agent-0  (1,3)   3      Frontier Exploration: γ=0.95  [0.0042, 0.9948, 0.0011]
5     Agent-1  (3,1)   1      Hyst Q: α=0.10, β=0.01 [ISO]  [0.0003, 0.0007, 0.9990]   ← blackout
```

```bash
pip install -e ".[moe]"
python demo_moe.py                     # or: docker compose run --rm demo-moe
```

The demo trains on the real 20×20 `RescueEnv` (full behavioral-cloning epochs +
router optimization), renders the live ASCII grid and telemetry dashboard, and
finishes by running `pytest tests/test_moe.py` as an integration gate.
A module-level deep dive lives in [README_moe.md](README_moe.md).

The same pipeline (shared via `rescue_sim/MoE/pipeline.py`) powers the **web
dashboard's Neural MoE mode** (`docker compose up --build viz`, then select
*Neural MoE* and press Start): training progress streams into the browser,
and the rollout runs repeated tries on one fixed 20×20 / 4-agent / 4-goal
grid with live per-agent routing vectors, expert-colored agents, and blackout
badges.

### Indicative results (short CPU training runs)

Greedy success rate on freshly generated grids (each episode is a new random map,
so this measures generalisation, not memorisation):

| Method | Grid | Agents | Episodes/updates | Greedy success |
|---|---|---|---|---|
| MAPPO | 6×6 | 3 | 40 updates | ~0.60 |
| QMIX | 6×6 | 3 | 150 episodes | ~0.65 |
| TransfQMix | 6×6 | 3 | 150 episodes | ~0.80 |

All trained on CPU in minutes (TransfQMix is slowest — transformers are heavier).
Numbers are indicative of short runs; longer training improves all three.

**Neural Mixture-of-Experts** is measured differently — `demo_moe.py` runs the
full production pipeline on the real 20×20 grid and prints the numbers live:

- Behavioral cloning converges (cross-entropy ≈ 1.37 → ≈ 0.63 over 20 epochs on
  three heuristic-teacher datasets; validation accuracy reported per epoch under
  `torch.no_grad()`).
- Router optimization drives the blackout penalty to ≈ 2×10⁻⁴ within 120 steps:
  connected agents route to the coordination head with $g_{\text{coord}} > 0.99$
  and isolated agents flip to the GRU fallback with $g_{\text{fallback}} > 0.99$
  — visible per step in the Phase C telemetry table.
- The run finishes by executing `pytest tests/test_moe.py` as an automatic
  integration gate.

### Sources & further reading

- **Q-learning** — Watkins & Dayan, *Q-learning*, Machine Learning 1992.
- **Hysteretic Q-learning** — Matignon, Laurent & Le Fort-Piat, *Hysteretic
  Q-Learning: an algorithm for decentralized RL in cooperative multi-agent teams*,
  IROS 2007.
- **MAPPO** — Yu et al., *The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games*, NeurIPS 2022 — <https://arxiv.org/abs/2103.01955>
- **QMIX** — Rashid et al., *QMIX: Monotonic Value Function Factorisation for
  Deep Multi-Agent RL*, ICML 2018 — <https://arxiv.org/abs/1803.11485>
- **TransfQMix** — Gallici, Martin & Masmitja, *TransfQMix: Transformers for
  Leveraging the Graph Structure of MARL Problems*, AAMAS 2023 —
  <https://arxiv.org/abs/2301.05334> · code: <https://github.com/mttga/pymarl_transformers>
- **GAE** (used by MAPPO) — Schulman et al., *High-Dimensional Continuous Control
  Using Generalized Advantage Estimation*, ICLR 2016 — <https://arxiv.org/abs/1506.02438>
- **CTDE paradigm** — *Centralized Training, Decentralized Execution* survey —
  <https://arxiv.org/abs/2409.03052>
- **Policy distillation** (used by the Ensemble's `Distiller`) — Rusu et al.,
  *Policy Distillation*, ICLR 2016 — <https://arxiv.org/abs/1511.06295>
- **Knowledge distillation / temperature softening** (TeacherBank calibration in
  `MoE/gated_distill.py`) — Hinton, Vinyals & Dean, *Distilling the Knowledge in
  a Neural Network*, NeurIPS-W 2015 — <https://arxiv.org/abs/1503.02531>
- **Reverse-KL distillation** (the gated E2 objective) — Agarwal et al.,
  *On-Policy Distillation of Language Models (GKD)*, ICLR 2024 —
  <https://arxiv.org/abs/2306.13649>
- **Value-target normalization** (PopArt-style stabilization applied to
  QMIX/TransfQMix TD targets) — van Hasselt et al., *Learning values across
  many orders of magnitude*, NeurIPS 2016 — <https://arxiv.org/abs/1602.07714>
- **Mixture-of-Experts** (the gating idea behind `MoE/`) — Jacobs, Jordan,
  Nowlan & Hinton, *Adaptive Mixtures of Local Experts*, Neural Computation 1991;
  sparsely-gated modern form: Shazeer et al., ICLR 2017 — <https://arxiv.org/abs/1701.06538>
- **Algorithm selection / portfolios** (the `MoE/` gate *is* per-instance
  algorithm selection) — Rice, *The Algorithm Selection Problem*, 1976; SATzilla:
  Xu, Hutter, Hoos & Leyton-Brown, *Portfolio-based Algorithm Selection for SAT*,
  JAIR 2008 — <https://arxiv.org/abs/1111.2249>
- **Artificial Potential Fields** (the `apf` baseline and Expert 1's teacher) —
  Khatib, *Real-Time Obstacle Avoidance for Manipulators and Mobile Robots*,
  ICRA 1985 / IJRR 1986.

---

## Documentation

- [Architecture](docs/architecture.md)
- [Requirements](docs/requirements.yaml)

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

# Decentralized Epidemic Hysteretic fleet (maps to config.settings.FleetSettings)
fleet:
  num_agents: 4           # robots active at episode start
  max_agents: 20          # pre-allocated capacity (1 <= N <= 20)
  alpha: 0.5              # learning rate for positive TD error
  beta: 0.1              # muted rate for negative TD error (beta << alpha)
  discount_factor: 0.95
  epsilon: 0.2
  comm_radius: 3.0        # Euclidean distance that opens a peer link
  gossip_cooldown: 5      # steps before the same pair may re-sync
  max_links_per_step: 2   # per-robot handshake budget (congestion control)
  utility_threshold: 0.0
  random_seed: 42
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

The scenario runner is the command-line entry point for a single damaged-area
scenario with metrics and text output:

```bash
python scripts/run_scenario.py
```

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

## Docker Support

The project is fully containerized using **Docker** and **Docker Compose**. The Docker configuration automatically builds and serves the updated React frontend assets while hot-reloading backend Python files when you make local edits.

### Requirements
Ensure you have Docker and Docker Compose installed.

### Quick Start
To build the image and start the visualization dashboard on port `8000`:
```bash
docker compose up --build viz
```
Open [http://localhost:8000/app](http://localhost:8000/app) in your browser.

### Interactive Development Shell
To open a bash shell inside the container:
```bash
docker compose run --rm dev
```

### Running Tests inside Docker
To execute the test suite in the containerized environment:
```bash
docker compose run --rm test
```

### Running Linting inside Docker
To check the codebase with Ruff:
```bash
docker compose run --rm lint
```

## License & Academic Attribution

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full license text.

* **Authors (Group 05)**:
 * Adriana Herrero Callejo ([github.com/adrianaherrerocallejo](https://github.com/adrianaherrerocallejo))
 * Cristina Marcos Alonso ([github.com/CristinaMarcosAlonso](https://github.com/CristinaMarcosAlonso))
 * Mohammad Mustafa Orfany ([github.com/MustafaZo77o](https://github.com/MustafaZo77o))
 * Alireza Moazzen ([alirezamoazen.com](http://alirezamoazen.com))
 * **Institution**: Hamburg University of Technology (TUHH) — Software Development SS26
 * **Supervisor**: Rainer Marrone


