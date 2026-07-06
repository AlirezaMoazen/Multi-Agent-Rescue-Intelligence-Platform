# Multi-Agent Neural Mixture of Experts (MoE) Policy Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-blue.svg?logo=docker)](https://www.docker.com/)

A state-of-the-art step-level **Neural Mixture of Experts (MoE)** policy implementation for cooperative robot swarms (1 to 20 agents) operating under localized communication blackouts ($d \ge 3$ grid cells). 

This framework replaces omniscient legacy planners with a decentralized neural network matching the **Centralized Training, Decentralized Execution (CTDE)** paradigm.

---

## Directory Structure

```text
├── Dockerfile_moe            # Multi-stage, VRAM-optimized Docker deployment file
├── README_moe.md             # Theoretical documentation and quickstart instructions
├── demo_moe.py               # Self-contained, executable simulation and training dashboard
└── src/
    └── rescue_sim/
        └── MoE/
            ├── __init__.py   # Neural MoE package declarations
            └── moe.py        # Production-grade step-level Neural MoE implementation
```

---

## Theoretical Foundations

### 1. Centralized Training, Decentralized Execution (CTDE)
During training, the critic network uses global states (all agent observations concatenated) to stabilize value estimation. During execution, each agent acts independently, mapping its own $7 \times 7$ local ego-centric observation (visibility radius $r=3$) to action probabilities:
$$\pi_i(a_i \mid o_i) = \text{Softmax}(\mathbf{y}_i^{\text{masked}})$$

### 2. Individual-Global-Max (IGM) Constraint in QMIX
QMIX enforces the IGM constraint, ensuring that the joint argmax action matches the collection of individual greedy actions:
$$\arg\max_{\mathbf{a}} Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = \left( \arg\max_{a_1} Q_1(s_1, a_1), \dots, \arg\max_{a_n} Q_n(s_n, a_n) \right)$$
This is guaranteed by maintaining monotonicity between the joint utility and individual utilities:
$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0, \quad \forall i \in \{1, \dots, n\}$$
Our **Expert 2 Head** is distilled directly from this monotonic coordinate policy space.

### 3. Logit-Blending & Partial Observability
Under communication blackouts ($d \ge 3$), the gating router shifts allocation weights from the coordination head to the decentralized fallback head. Action selection remains robust under partial observability because the Gating Router blends unnormalized policy logits ($\mathbb{R}^4$) directly, preventing the value-scale mismatches common in state-action utility mixing:
$$y_i^{\text{final}} = \sum_{j=1}^{3} g_{i, j} \cdot y_i^{j}$$

---

## Comparative Trade-Off Matrix

| Metric | Legacy Planners (CBS, M*) | QMIX / MAPPO Generalists | Step-Level Neural MoE (Ours) |
| :--- | :--- | :--- | :--- |
| **Observation Requirement** | Omniscient Global View | Local / Flattened Window | Rigid $7 \times 7$ Ego-centric View |
| **Coordination Method** | Centralized Tree Branching | Decentralized Monotonic Q-mixing | Dynamic Actor Logit Blending |
| **Blackout Behavior ($d \ge 3$)**| Failure / Total Halt | Suboptimal drift | Smooth transition to Hysteretic fallback |
| **Execution Complexity** | Exponential $O(b^d)$ | Constant $O(1)$ | Constant $O(1)$ |

---

## Docker Quickstart

### 1. Build the Container
```bash
docker build -t marl-moe -f Dockerfile_moe .
```

### 2. Run the Container (with GPU Acceleration)
```bash
docker run --rm --gpus all marl-moe
```
*(Omit the `--gpus all` flag if running on a CPU-only host; PyTorch will automatically fallback to CPU mode).*
