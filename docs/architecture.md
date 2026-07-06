# Architecture — Multi-Agent Rescue Swarm Simulator

This document describes the design architecture for the Multi-Agent Rescue Teams simulator, incorporating the Neural Mixture of Experts (MoE) swarm design and reinforcement learning baselines.

---

## System Design Paradigm

The simulator operates under the **Centralized Training, Decentralized Execution (CTDE)** paradigm:
1. **Centralized Training**: Value-based and policy-gradient algorithms (QMIX, TransfQMix, MAPPO) access joint observations and rewards during training to optimize the parameters of the actor and critic networks.
2. **Decentralized Execution**: During runtime, each agent acts independently, using only its local egocentric observations (a 7x7 spatial window corresponding to a 3-block visibility range) to sample actions:
   $$\pi_i(a_i \mid o_i) = \text{Softmax}(y_i^{\text{masked}})$$

---

## Package Directory Structure

```text
src/rescue_sim/
├── MoE/                # Neural Mixture of Experts (MoE) framework
│   ├── __init__.py     # Package exports
│   └── moe.py          # Dual-Encoder step-level policy implementation
├── MAPPO/              # Multi-Agent PPO training and policy modules
├── QMIX/               # Multi-Agent Value Factorization QMIX / TransfQMix
├── Qlearning/          # Tabular Hysteretic Q-learning and heuristic explorers
│   ├── __init__.py     # Package exports
│   └── baseline.py     # Greedy Frontier and CBS baseline planners
├── config/             # YAML config loader and settings classes
├── environment/        # Spatial grid, obstacle generator, and sensor models
├── simulation/         # Execution runners and metric collection
└── visualization/      # Dynamic evaluation charts and web rendering api
```

---

## Mixture of Experts (MoE) Architecture

To handle communication disruptions, a **Neural MoE** is deployed to blend predictions from three specialized offline-distilled experts:

```text
                               +----------------------------+
                               |     Local Agent Obs o_i    |
                               +--------------+-------------+
                                              |
                                       z = Encoder(o_i)
                                              |
                       +----------------------+----------------------+
                       |                      |                      |
                       v                      v                      v
                +──────────────+       +──────────────+       +──────────────+
                |   Expert 1   |       |   Expert 2   |       |   Expert 3   |
                | (Exploration |       | (Coordination|       |  (Fallback   |
                |  Heuristic)  |       |  QMIX/MAPPO) |       |  Hysteretic) |
                +──────┬───────+       +──────┬───────+       +──────┬───────+
                       | y_exp0               | y_exp1               | y_exp2
                       |                      |                      |
                       +----------------------+----------------------+
                                              |
                                              v
                              y_final = sum(g_j * y_exp_j)
                                              |
                                              v
                                     [Action Selection]
```

### 1. Dual-Encoder Topology
To prevent representational drift, the MoE separates feature extraction:
* `expert_encoder`: Extracted features feed the expert heads. Frozen during Stage 2.
* `router_encoder`: Extracted features feed the gating router. Trainable during Stage 2.

### 2. Expert Allocations
* **Expert 1 (Exploration)**: Distilled from the greedy `BaselineExplorer` to maximize area coverage.
* **Expert 2 (Coordination)**: Distilled from the team's decentralized `QMIX` or `MAPPO` algorithms to coordinate collision-free paths.
* **Expert 3 (Fallback)**: Distilled from decentralized `Epidemic Hysteretic Q-learning` to handle isolated movement.

### 3. Gating Router
A single linear layer maps the `router_encoder` embedding to 3 gating probabilities using Softmax. Under blackouts ($d \ge 3$), a **Conditional Indicator Mask Penalty** forces the weights to favor the fallback expert:
$$\mathcal{L}_{\text{penalty}} = \lambda \cdot \mathbb{I}(\text{peer\_count} == 1.0) \cdot (1.0 - g_{\text{fallback}})^2$$
