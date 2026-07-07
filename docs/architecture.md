# Architecture — Multi-Agent Rescue Swarm Simulator

This document describes the current architecture of the Multi-Agent Rescue Teams
simulator: the problem we solve, the design we arrived at (a Neural Mixture of
Experts whose coordination expert is distilled from three trained deep-RL
models), and how a reviewer can test every claim.

---

## The Problem

A team of 4 rescue robots must locate and rescue targets scattered on an
obstacle-filled grid (default 14×14, 15% obstacles). Three properties make this
hard:

1. **Partial observability** — each agent only sees a 7×7 egocentric window
   (3-block sensor range). Nobody sees the whole map, and the map is different
   every episode, so no policy can memorize a layout.
2. **Communication disruption** — agents share information only when linked to
   teammates. An agent that drifts out of range is on its own and must still
   act sensibly.
3. **No single algorithm wins everywhere.** Measured on identical grids, our
   trained methods have complementary strengths: MAPPO finds targets fastest
   when it succeeds, TransfQMix succeeds most often, QMIX sits in between, and
   simple heuristics beat all of them early in an episode when nothing is
   visible yet.

### A sub-problem we hit and solved: unstable deep-RL training

QMIX and TransfQMix initially plateaued (~30% and ~0% success) while MAPPO
trained fine. The root cause was **reward scale**: episode rewards span roughly
−10 to +150 (team-summed), so the mixers regressed raw TD targets spanning
hundreds — measured losses of 50–2700 — which destabilized the monotonic mixing
network. **Our answer**: apply the same `RunningMeanStd` value-target
normalization MAPPO already used (`normalize_value=True` in the settings) to
both value-based methods, so the mixer regresses standardized targets. After
retraining, all three checkpoints reach 60–80% greedy success on unseen 14×14
grids (`scripts/eval_checkpoints.py`).

---

## Our Answer: MoE over specialized experts, with a distilled ensemble coordinator

Because no single policy dominates, the top layer is a **Neural Mixture of
Experts (MoE)**: three specialist heads blended per step, per agent, by a
learned attention gating router.

```text
                               +----------------------------+
                               |     Local Agent Obs o_i    |
                               +--------------+-------------+
                                              |
                     expert_encoder(o_i)             router_encoder(o_i)
                                              |
                       +----------------------+----------------------+
                       |                      |                      |
                       v                      v                      v
                +──────────────+       +──────────────+       +──────────────+
                |   Expert 1   |       |   Expert 2   |       |   Expert 3   |
                | Exploration  |       | Coordination |       |  Fallback    |
                | (distilled   |       | (distilled   |       | (GRU clone + |
                |  APF non-AI  |       |  MAPPO+QMIX+ |       |  LIVE epid.  |
                |  baseline)   |       |  TransfQMix) |       |  hyst. Q)    |
                +──────┬───────+       +──────┬───────+       +──────┬───────+
                       | y_1                  | y_2                  | y_3
                       +----------------------+----------------------+
                                              |
                              g = AttentionGatingRouter(router_encoder(o_i))
                                              |
                                              v
                             y_final = Σ_j g_j · y_j   →  masked argmax
```

### System design paradigm: CTDE

Training is centralized (QMIX/TransfQMix mixers and the MAPPO critic see joint
state; the MoE distills from team trajectories), but execution is fully
decentralized: at runtime each agent samples from
`π_i(a_i | o_i) = Softmax(y_i^masked)` using only its own local observation.

### Dual-Encoder topology

To prevent representational drift between stages, feature extraction is split:

* `expert_encoder` — feeds the three expert heads; frozen after distillation.
* `router_encoder` — feeds the gating router; stays trainable so the gate can
  keep improving without disturbing the experts.

Both are CNN(7×7×4 local window) + MLP(meta features + agent ID) encoders.

### Expert 2 — the core contribution

Expert 2 is **not** cloned from a hand-written teacher (the old design cloned a
heuristic that was nearly identical to E1/E3, so the router had no reason to
use it). Instead, E2 is distilled from **all three trained deep-RL checkpoints
at once** (`src/rescue_sim/MoE/gated_distill.py`):

1. **TeacherBank** loads `checkpoints/{mappo,qmix,transfqmix}.pt` and converts
   each model's output into an action distribution. Q-values and policy logits
   live on different scales, so per-teacher **temperatures are calibrated** to
   a common entropy target — all three teachers "speak the same language."
2. A **state-conditioned gating network** is trained on oracle weights (which
   teacher would have chosen the best action in this state) to decide, per
   situation, how much to trust each teacher. Learned average trust:
   ~45% MAPPO, ~27% QMIX, ~28% TransfQMix.
3. E2 is trained by **reverse-KL distillation** onto the gated mixture of the
   three teacher distributions, giving it the combined knowledge rather than a
   copy of any single model.

This mirrors the standalone `Ensemble/` result: on identical grids the
QMIX+TransfQMix value ensemble (87% success, 30 episodes) beats every single
model (best single: TransfQMix 80%).

### Gating Router — outcome-trained, near winner-take-all

The attention router maps the `router_encoder` embedding to 3 gating weights
via Softmax. It is initialized with rule-based regime penalties, then
**retrained on outcome labels** (`MoE/gated_distill.py::train_outcome_router`):
for every state the MoE itself visits, each expert head proposes its greedy
action, the calibrated teacher mixture judges the proposals, and the router
learns to hand the state to the winning expert (92-98% routing accuracy).
Finally the gate temperature is sharpened (`gate_tau = 0.25`) so routing is
near winner-take-all — the MoE acts like its best expert per state instead of
blending logits from heads with incompatible scales. Under communication
blackout, a **Conditional Indicator Mask Penalty** pushes weight onto the
fallback expert:

$$\mathcal{L}_{\text{penalty}} = \lambda \cdot \mathbb{I}(\text{peer\_count} = 1) \cdot (1 - g_{\text{fallback}})^2$$

so an isolated agent leans on the GRU fallback head, while connected agents
lean on the distilled coordinator.

### Expert 1 — the non-AI baseline, distilled

E1's teacher is the project's real non-learning multi-robot algorithm:
**Artificial Potential Fields** (`Qlearning/baseline.py::APFExplorer`, Khatib
1986). Each agent sums local forces — attraction to the nearest visible
target, separation repulsion from teammates, obstacle repulsion, open-space
attraction — all computable from its own sensor window, so the feed-forward
head clones it faithfully. The same strategy runs standalone as the
**"Non-AI (APF)"** row of the head-to-head panel: the gap between it and the
ML policies quantifies exactly what learning buys. (The centralized CBS
planner was removed from the baselines: a central plan cannot be executed
from local observations, so it fits neither the CTDE setting nor the MoE.)

### Expert 3 — LIVE Epidemic Hysteretic Q-learning during tries

Tabular Q-learning shines when it can keep learning on a persistent map — so
instead of distilling it (tried, and measured at 0-8% MoE success vs ~57%,
because its greedy policy is keyed to a per-grid Q-table the local window
cannot expose), the real `EpidemicHystereticQLearning` fleet runs **live**
inside the dashboard rollout (`visualization/api.py::_run_moe_rollout`):

* it learns from **every** transition (whichever expert drove the agent),
* it gossips Q-table deltas through the comms layer
  (`Qlearning/communications.py`) whenever agents meet,
* its Q-tables **persist across tries** on the fixed competition grid —
  by the later tries it is near-certain about the map,
* whenever the router routes an agent to the fallback expert, the live
  learner takes control (the GRU clone of a learnable sweep policy covers
  try 1, before the tables know anything).

### Persistence: pretrain once, keep improving

The MoE is **pretrained offline** (`scripts/pretrain_moe.py`, ~5 minutes on
CPU) over many freshly seeded grids and saved to `checkpoints/moe.pt`. The
dashboard backend loads it at startup, and every later "Train" press continues
from the saved policy and re-saves it — the policy accumulates improvement
across sessions instead of restarting. A saved policy is keyed by
`(grid_w, grid_h, num_agents, view_radius)`; it generalizes to any unseen grid
of that configuration because it only ever consumes local observations.

---

## Package Directory Structure

```text
src/rescue_sim/
├── MoE/                # Neural Mixture of Experts (top layer)
│   ├── moe.py          # Dual-Encoder policy, attention router, 3 expert heads
│   ├── pipeline.py     # collect → distill experts → optimize router; save/load
│   └── gated_distill.py# TeacherBank + gated reverse-KL distillation for E2
├── Ensemble/           # QMIX+TransfQMix value ensemble + Q-vector Distiller
├── MAPPO/              # Multi-agent PPO (policy-gradient teacher)
├── QMIX/               # Value-factorization QMIX (with value-target norm)
├── TransfQMix/         # Transformer QMIX variant (with value-target norm)
├── Qlearning/          # Tabular hysteretic Q + frontier/CBS baselines
├── config/             # YAML loader and settings dataclasses
├── environment/        # Grid, obstacles, sensors
├── simulation/         # Runners and metric collection
└── visualization/      # FastAPI backend + React dashboard
checkpoints/            # mappo.pt, qmix.pt, transfqmix.pt, moe.pt (pretrained)
scripts/                # train_*, eval_checkpoints.py, pretrain_moe.py
```

---

## How to Test It

Everything below runs on CPU; no GPU required.

**1. Verify the trained checkpoints (no training, ~2 min).**

```bash
python scripts/eval_checkpoints.py --grid 14 --episodes 30
```

Loads the saved MAPPO/QMIX/TransfQMix models, evaluates them greedily on 30
held-out random grids, and prints success/rescued/steps per method plus the
value ensemble. Expected ballpark: MAPPO ~73%, QMIX ~63%, TransfQMix ~80%,
Ensemble ~87%.

**2. Pretrain (or refresh) the MoE (~5 min).**

```bash
python scripts/pretrain_moe.py --minutes 5
```

Watch the per-round log: gated E2 distillation accuracy should climb (54% →
~65–70%) and the teacher-trust weights should stay spread across all three
teachers (evidence E2 really uses all of them).

**3. Test the live dashboard.**

```bash
docker compose up --build viz     # then open http://localhost:8000
```

* On startup the log prints `[MoE] loaded pretrained policy from
  checkpoints/moe.pt` — Evaluate works without training first.
* Pick **Neural MoE** mode and press Train: distillation and router progress
  stream live; afterwards the rollout shows per-agent expert routing shares.
* Press **▶ Compare** in the *Experts vs. MoE — head-to-head* panel: each
  standalone expert head and the blended MoE run greedy rollouts on identical
  seeded grids; winner cards + normalized bar chart summarize success,
  rescues, connectivity, efficiency.

**4. Run the automated test suite.**

```bash
docker compose run --rm test      # or: pytest
```

---

## Attribution

Developed by **TUHH Group 05** (Software Development SS26): Adriana Herrero
Callejo, Cristina Marcos Alonso, Mohammad Mustafa Orfany, Alireza Moazzen.
Supervised by Rainer Marrone. Licensed under Apache 2.0 — see `LICENSE`.
