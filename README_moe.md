# Neural Mixture of Experts (MoE) — Module Deep Dive

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg?logo=pytorch)](https://pytorch.org/)
[![CPU Training](https://img.shields.io/badge/GPU-Not%20Required-orange.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-blue.svg?logo=docker)](https://www.docker.com/)

A step-level **Neural Mixture of Experts (MoE)** policy for cooperative robot
swarms operating under localized communication blackouts (Manhattan distance
$d \ge 3$ grid cells). An attention-based gating router blends three
specialized expert heads per agent, per step, following the **Centralized
Training, Decentralized Execution (CTDE)** paradigm.

> This is the module-level deep dive. For the project-wide overview, quick
> start, and the other learning methods, see the main [README](README.md).

---

## Module Files

```text
├── Dockerfile_moe            # Multi-stage CPU demo container (build + run below)
├── README_moe.md             # This document
├── demo_moe.py               # Production demo: trains on the real 20x20 RescueEnv,
│                             #   renders the live dashboard, runs the pytest gate
├── src/rescue_sim/MoE/
│   ├── __init__.py           # Package exports
│   └── moe.py                # NeuralMoEPolicy, AttentionGatingRouter,
│                             #   RecurrentFallbackHead, SharedFeatureEncoder
└── tests/test_moe.py         # Unit tests (also run by the demo's integration gate)
```

---

## Theoretical Foundations

### 1. Centralized Training, Decentralized Execution (CTDE)
Expert heads are distilled centrally from team trajectories. During execution,
each agent acts independently, mapping its own $7 \times 7$ local ego-centric
observation (visibility radius $r=3$) to action probabilities:
$$\pi_i(a_i \mid o_i) = \text{Softmax}(\mathbf{y}_i^{\text{masked}})$$

### 2. Individual-Global-Max (IGM) Constraint in QMIX
QMIX enforces the IGM constraint, ensuring that the joint argmax action matches
the collection of individual greedy actions:
$$\arg\max_{\mathbf{a}} Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = \left( \arg\max_{a_1} Q_1(s_1, a_1), \dots, \arg\max_{a_n} Q_n(s_n, a_n) \right)$$
This is guaranteed by maintaining monotonicity between the joint utility and individual utilities:
$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0, \quad \forall i \in \{1, \dots, n\}$$
The **coordination head (Expert 2)** is distilled from this monotonic
coordinate policy space.

### 3. Logit Blending & Partial Observability
Under communication blackouts ($d \ge 3$), the attention router shifts
allocation weights from the coordination head to the recurrent fallback head.
Action selection remains robust under partial observability because the router
blends unnormalized policy logits ($\mathbb{R}^4$) directly, preventing the
value-scale mismatches common in state-action utility mixing:
$$y_i^{\text{final}} = \sum_{j=1}^{3} g_{i, j} \cdot y_i^{j}$$

### 4. Temporal Memory Under Blackout
The fallback head (Expert 3) is a `GRUCell` whose hidden state $h_t$ persists
across the episode timeline. An isolated agent therefore remembers where it
has been and escapes dead-ends instead of blind looping:
$$h_t = \operatorname{GRU}(z_t,\ h_{t-1})$$

---

## Comparative Trade-Off Matrix

| Metric | Legacy Planners (CBS, M*) | QMIX / MAPPO Generalists | Step-Level Neural MoE (Ours) |
| :--- | :--- | :--- | :--- |
| **Observation Requirement** | Omniscient Global View | Local / Flattened Window | Rigid $7 \times 7$ Ego-centric View |
| **Coordination Method** | Centralized Tree Branching | Decentralized Monotonic Q-mixing | Dynamic Actor Logit Blending |
| **Blackout Behavior ($d \ge 3$)**| Failure / Total Halt | Suboptimal drift | Smooth transition to GRU fallback |
| **Execution Complexity** | Exponential $O(b^d)$ | Constant $O(1)$ | Constant $O(1)$ |

---

## Quickstart

```bash
# Docker (single command — build + full demo)
docker compose run --rm demo-moe

# Or build the standalone demo image
docker build -t rescue-sim-moe -f Dockerfile_moe .
docker run --rm -it rescue-sim-moe

# Or run locally
pip install -e ".[dev]" && pip install torch --index-url https://download.pytorch.org/whl/cpu
python demo_moe.py
```

The demo collects expert trajectories on the real 20×20 `RescueEnv`, runs full
behavioral-cloning and router-optimization training loops with live progress
bars, renders the ASCII grid world and per-step routing telemetry, and finishes
by executing `pytest tests/test_moe.py` as an integration gate. CPU-only —
no GPU required.
