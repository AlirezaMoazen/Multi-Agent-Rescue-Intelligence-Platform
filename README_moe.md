# Multi-Agent Neural Mixture of Experts (MoE) Policy Framework

A state-of-the-art step-level **Neural Mixture of Experts (MoE)** policy implementation for cooperative robot swarms (1 to 20 agents) operating under localized communication blackouts ($d \ge 3$ grid cells). 

This framework operates under the **Centralized Training, Decentralized Execution (CTDE)** paradigm. It utilizes unnormalized policy logit blending to eliminate value-scale mismatches, a dual-encoder topology to prevent representational drift during fine-tuning, and a permutation-invariant communication pooling layer that allows the policy to generalize seamlessly across varying fleet sizes.

---

## System Architecture

```text
                       +-----------------------------------+
                       |    Raw Local Input State (obs)    |
                       +-----------------+-----------------+
                                         |
                       +─────────────────┴─────────────────+
                       | Permutation-Invariant Comms Pool  |
                       +─────────────────┬─────────────────+
                                         |
                                  [z_comm pooled]
                                         |
                 +───────────────────────┴───────────────────────+
                 |                                               |
                 v                                               v
     +───────────────────────+                       +───────────────────────+
     |  Router Encoder (RL)  |                       |  Expert Encoder (Fix) |
     +───────────┬───────────+                       +───────────┬───────────+
                 | [z_router]                                    | [z_expert]
                 v                                               +-----+-----+
     +───────────────────────+                                         |
     |     Gating Router     |                 +───────────────────────┼───────────────────────+
     +───────────┬───────────+                 |                       |                       |
                 |                             v                       v                       v
                 |                      +─────────────+         +─────────────+         +─────────────+
                 |                      |  Expert 1   |         |  Expert 2   |         |  Expert 3   |
                 |                      | (Exploration|         | (Coordination|        | (Fallback   |
                 |                      |   Heuristic)|         |    Planner) |         | Hysteretic) |
                 |                      +──────┬──────+         +──────┬──────+         +──────┬──────+
                 | [Weights g]                 | [Logits y_1]          | [Logits y_2]          | [Logits y_3]
                 |                             +                       +                       +
                 +────────────────────────────>+                      >+                      >+
                                               |                       |                       |
                                               v                       v                       v
                                        +───────────────────────────────────────────────────────+
                                        |                Actor Logit Blending                   |
                                        |      y_final = g_1*y_1 + g_2*y_2 + g_3*y_3            |
                                        +──────────────────────────┬────────────────────────────+
                                                                   |
                                                                   v
                                        +───────────────────────────────────────────────────────+
                                        |                 Invalid Action Mask                   |
                                        +──────────────────────────┬────────────────────────────+
                                                                   |
                                                                   v
                                        +───────────────────────────────────────────────────────+
                                        |                   Softmax & Sample                    |
                                        +-------------------------------------------------------+
```

### 1. Dual-Encoder Topology
To preserve the performance of distilled heuristic experts while optimization fine-tunes the gating router online, the architecture implements two distinct parameter spaces:
* `expert_encoder`: Extracted features feed the expert heads. Locked permanently during Stage 2.
* `router_encoder`: Extracted features feed the gating router. Remains fully trainable during Stage 2 online optimization.

### 2. Permutation-Invariant Communication Tracking
The policy maps neighbor link states to a single pooled active peer count metric:
$$\text{peer\_count} = \sum_{j=1}^{\text{Num\_Agents}} \text{peer\_matrix}_{i, j}$$
Because neighbor identity coordinates are summed, the network processes active neighbors uniformly. This enables the swarm controller to generalize from 1 up to 20 agents out of the box.

### 3. Actor logit Blending & Masking
Value-scale conflicts are eliminated by blending unnormalized directional logits ($\mathbb{R}^4$) instead of utility Q-values:
$$y_i^{\text{final}} = \sum_{j=1}^{3} g_{i, j} \cdot y_i^{j}$$
Where $g_{i, j}$ represents the routing weight of expert $j$ for agent $i$. Blended logits are masked prior to softmax sampling to prevent invalid transitions:
$$y_i^{\text{masked}} = \text{torch.where}(\text{action\_mask}_i, y_i^{\text{final}}, -10^9)$$

---

## Two-Stage Optimization Workflow

### Stage 1: Expert Distillation (Offline Behavioral Cloning)
Supervised Cross-Entropy distillation is executed to clone policies into the expert heads from recorded grid trajectories:
$$\mathcal{L}_{\text{distill}} = -\sum_{a \in A} \log \pi_{\text{expert}}(a \mid o)$$
During this phase, the gating router parameters are kept frozen (`requires_grad = False`).

### Stage 2: Router Optimization (Online Policy Gradient)
The Gating Router is trained online using Policy Gradients (MAPPO). To enforce correct behavior under communication dropouts, we apply a **Conditional Indicator Mask Penalty**:
$$\mathcal{L}_{\text{penalty}} = \lambda \cdot \mathbb{I}(\text{peer\_count} == 1.0) \cdot (1.0 - g_{\text{fallback}})^2$$
Where $\mathbb{I}$ is the indicator function yielding $1.0$ if the agent is isolated (peer count is exactly $1.0$) and $0.0$ if it is connected to peers. This forces $g_{\text{fallback}} \approx 1.0$ under blackout, while leaving the router free to choose the optimal expert during normal operations.

---

## Docker Quickstart

### Prerequisites
* Docker installed on host.
* NVIDIA Container Toolkit (for VRAM-optimized GPU acceleration).

### 1. Build the Container
```bash
docker build -t marl-moe -f Dockerfile_moe .
```

### 2. Run the Container
```bash
docker run --rm --gpus all marl-moe
```
*(Omit the `--gpus all` flag if running on a CPU-only laptop; the PyTorch backend will automatically fallback to host CPU execution).*
