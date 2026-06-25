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

## Learning Algorithms

The project compares several learning strategies against two non-learning baselines.

**Status:** ✅ implemented — Q-Learning (single-agent baseline), **Epidemic Hysteretic Q-Learning** (decentralized multi-agent), **MAPPO** (deep, policy-gradient, CTDE), and **QMIX** (deep, value-decomposition, CTDE). 🔜 planned — TransfQMix (documented below as the remaining deep-learning roadmap; not yet coded).

---

### Baseline (no learning)

Two deterministic heuristics defined in `src/rescue_sim/Qlearning/baseline.py`:

- **BaselineExplorer** — frontier-greedy: scores candidate moves by +2 for unvisited cells and +1 for frontier adjacency; always picks the best score.
- **DFSExplorer** — depth-first search: maintains a LIFO stack of unvisited neighbors; uses BFS over the discovered map to navigate to non-adjacent targets.

No reward signal is used. These are the performance floor every learning method must beat.

---

### Q-Learning (tabular, single-agent)

Standard temporal-difference learning with an ε-greedy policy, implemented as `QLearningAgent` in `src/rescue_sim/Qlearning/q_learning.py`.

**Update rule:**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Symbol | Meaning |
|--------|---------|
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $r$ | Reward received after action $a$ |
| $s'$ | Next state |

**Limitation:** The `LearningState` encodes full sets of cell positions, so nearly every state is unique. The Q-table memorises episodes rather than generalising — the main motivation for the methods below.

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

**Communication boundary.** The learner owns the *mechanism* (proximity detection, delta export/import, max-merge, throttling). The physical *transport* — line-of-sight, packet loss, latency, bandwidth budgeting — is deliberately left to `src/rescue_sim/communications.py`, which documents the hand-off and ships a working `ProximityGossipBus` reference plus state-of-the-art upgrade suggestions (version-vector anti-entropy, Merkle digests, robust aggregation).

**Step-loop contract** (one timestep): `select_actions` → environment applies the moves → `record_transitions` (hysteretic update) → `gossip` (epidemic max-sync).

```python
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning
from rescue_sim.shared import HystereticConfig, GossipConfig

fleet = EpidemicHystereticQLearning(grid, HystereticConfig(), GossipConfig(), max_agents=20, seed=0)
fleet.add_agent("r1", start)          # robots may join (or fail via remove_agent) any time
fleet.add_agent("r2", start2)

for _ in range(max_steps):
    actions = fleet.select_actions()  # {agent_id: action_index in 0..3 = N,S,E,W}
    rewards, next_positions, dones = environment_step(actions)
    fleet.record_transitions(actions, rewards, next_positions, dones)
    fleet.gossip()                    # or: ProximityGossipBus(...).exchange(fleet)
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

This constraint ensures that the individual greedy policy $\arg\max_{a_i} Q_i$ is consistent with $\arg\max_{\mathbf{a}} Q_{\text{tot}}$ — agents can act locally while optimising a shared team objective.

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

### TransfQMix (deep, multi-agent, transformer + value-based) — 🔜 planned

Planned module: `src/rescue_sim/Qlearning/transf_qmix.py` (not yet implemented).
Reference: Marin Bilos et al. / Hu et al., *TransfQMix: Transformers for Leveraging the Graph Structure of MARL Problems*, 2021.

**Same mixer as QMIX; different agent network.**

Instead of an MLP over a flat observation, each agent encodes its visible environment as a **sequence of entity tokens** fed to a transformer encoder.

**Entity tokenisation:**

Each visible cell becomes one token $e_i \in \mathbb{R}^{d_e}$:

$$e_i = \left[\frac{\Delta x}{r},\ \frac{\Delta y}{r},\ \mathbb{1}_{\text{free}},\ \mathbb{1}_{\text{obstacle}},\ \mathbb{1}_{\text{target-A}},\ \mathbb{1}_{\text{target-B}}\right]$$

**Transformer encoder:**

$$\mathbf{T} = \bigl[\mathbf{z}_{\text{CLS}};\ E \cdot W_{\text{in}}\bigr]$$
$$\mathbf{H} = \text{TransformerEncoder}(\mathbf{T})$$
$$Q_i = W_{\text{out}} \cdot \mathbf{H}[\text{CLS}]$$

The CLS token aggregates entity information via multi-head self-attention.

**Advantages over QMIX:**
- Naturally handles a variable number of visible entities (no padding needed)
- Attention weights reveal which entities the agent focuses on (interpretability)
- Generalises to different map sizes and team sizes without retraining

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

| | Q-Learning | Epidemic Hysteretic Q | QMIX | TransfQMix | MAPPO |
|---|---|---|---|---|---|
| Status | ✅ implemented | ✅ implemented | ✅ implemented | 🔜 planned | ✅ implemented |
| Family | Value-based | Value-based | Value-based | Value-based | Policy-gradient |
| Agents | Single | Multi (decentralized) | Multi | Multi | Multi |
| Function approx. | Tabular | Tabular (dense NumPy) | Deep (MLP) | Deep (Transformer) | Deep (MLP) |
| State / input | Full `LearningState` | Grid cell $(y,x)$ | Local window vector | Entity token sequence | Local window vector |
| Replay buffer | No | No | Yes (off-policy) | Yes (off-policy) | No (on-policy) |
| Coordination | None | Peer gossip (max-sync) | Mixer (training only) | Mixer (training only) | Critic (training only) |
| Runtime comms | No | Yes (when robots meet) | No | No | No |
| Key innovation | Baseline RL | Optimistic + epidemic max-sync | Monotonic mixing | Attention over entities | Clipped policy update |
| PC trainable | Yes | Yes | Yes | Yes | Yes |

---

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


