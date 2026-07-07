# Design: Expert 2 as an Ensemble-Distilled Coordination Policy

**Status:** design only (no MoE code changed yet). Companion to the retrained
deep-RL checkpoints (`checkpoints/{qmix,transfqmix,mappo}.pt`, 14×14).

## Why change E2

In the Neural MoE (`src/rescue_sim/MoE/`) the three expert heads are trained by
behavioral cloning of hand-written teachers in
`src/rescue_sim/MoE/pipeline.py`. Expert 2's teacher, `CoordinationTeacher`, is
labelled *"deep coordination (QMIX/MAPPO style)"* — but its first and dominant
rule is literally the same line the other two teachers use:

```python
target = self._visible_target(env, i)
if target is not None:
    actions[i] = self._step_toward(env, i, target, valid_mask)   # identical in E1, E2, E3
```

So E2 only diverges from E1/E3 in the rare "no visible target + a teammate
drifting to the comm edge" branch. Two consequences, both measured:

1. **E2 behaves ~like E1/E3** — it is not a distinct coordination policy, just a
   differently-labelled copy.
2. **The router underuses E2** — `run_router_optimization` only pushes
   `g_coord → 1` when `has_target AND linked`, a rare joint condition, so E2
   receives little supervision and the gate leans on explore/fallback.

More BC epochs cannot fix this (the network already imitates the teacher well);
the *teacher itself* carries no distinct coordination signal. The fix is to give
E2 a genuinely different teacher: the **trained** cooperative agents, combined.

## What "combined" means (not just one of QMIX/TransfQMix/MAPPO)

We already ship most of the machinery:

- `src/rescue_sim/Ensemble/ensemble.py::ValueEnsemble` — averages **per-agent
  Q-values** of trained QMIX and TransfQMix (weighted by each one's success via
  `performance_weights`) and takes the greedy valid action. Both are value-based
  and output comparable Q's, so they mix cleanly.
- `src/rescue_sim/Ensemble/distill.py::Distiller` — rolls the ensemble as a
  teacher and regresses a single `AgentQNet` student onto the teacher Q-values
  (MSE), i.e. classic policy distillation.

The gap the user asked to close: **MAPPO** is policy-based (action
probabilities, not Q-values), so it does not average into `combined_q` directly.
Design options to fold all three into one ensemble teacher:

- **(A) Q-space fusion + policy prior (recommended).** Keep
  `w_q·Q_qmix + w_t·Q_transf` as the value backbone, then add MAPPO as a soft
  action prior: convert its actor logits to log-probs and add
  `w_m · logits_mappo` to the (mask-filled) combined Q before the argmax. One
  scalar `w_m` (from MAPPO's standalone success via `performance_weights`
  extended to three) controls its pull. Cheap, and MAPPO breaks ties toward the
  on-policy-preferred action.
- **(B) Weighted action vote.** Each method proposes a greedy action; vote with
  performance weights; MAPPO breaks ties. Simpler but throws away value
  magnitude (worse distillation target than (A), which gives the student a dense
  Q-vector to regress on).

Recommendation: **(A)** — it preserves the dense Q target the `Distiller`
already regresses on, and only needs a 3-way `performance_weights` and a MAPPO
logit term added to `ValueEnsemble.combined_q`.

## Integration into E2 — two layers

### Layer 1 — swap E2's teacher (the actual behavior change)
Replace `CoordinationTeacher.act`'s output with the ensemble's greedy action:

- Build the 3-way ensemble once from the retrained checkpoints (load QMIX,
  TransfQMix on an `EntityRescueEnv`; MAPPO on the flat `RescueEnv`; both read
  the *same* stepped env so tokens/flat-obs align, exactly as `ValueEnsemble`
  already does for two).
- In `pipeline.py`, give the coordination teacher an `act()` that queries the
  ensemble instead of the heuristic. Everything downstream is unchanged: the BC
  stage in `run_expert_distillation` clones this new teacher into the E2 head
  just like today.
- **Cost/latency:** the ensemble runs 2–3 network forwards per step *during data
  collection only*. Precompute an offline labelled dataset (obs → ensemble
  action, and optionally ensemble Q for a richer regression target) once, then
  BC on it — keeps live training fast and reproducible. This mirrors
  `Distiller.collect`.

### Layer 2 — let the router actually use E2
With E2 now genuinely distinct, broaden its supervision in
`run_router_optimization`: currently the coord regime is
`has_target AND linked`. Loosen to reward `g_coord` whenever agents are **linked
with teammates** (closer to the original `peer_count == A` signal), so the gate
learns "when connected as a team, trust the trained coordinator." Expect higher
and better-balanced E2 routing share in the dashboard's live MoE panel.

## Distillation target choice
Prefer **Q-vector regression** (MSE onto `combined_q`, as `Distiller` does) over
hard-action cross-entropy: the dense target transfers the ensemble's relative
action preferences, not just its argmax, which is what makes the student behave
*like* the coordinator rather than merely agreeing with it at the greedy action.

## Verification plan (when implemented)
1. Ensemble sanity: `ValueEnsemble.evaluate` (extended to 3-way) on 14×14 beats
   each member alone — confirms the combination helps before distilling.
2. E2-in-isolation: roll only the E2 head; its rescue behavior should now match
   the ensemble far better than the old heuristic (compare avg_rescued).
3. Router share: in the live MoE dashboard, E2's gating share and
   rescues-by-expert should rise vs. today's E1/E3-dominated split.
4. End-to-end: full MoE `evaluate` success rate on 14×14 ≥ current MoE.

## Open questions
- 3-way weight calibration: static `performance_weights` from standalone eval, or
  a small validation search over `(w_q, w_t, w_m)`?
- Offline dataset size vs. coverage: how many ensemble-labelled transitions give
  the E2 head stable behavior without overfitting one grid distribution?
- Whether to also distill a *value* signal into E2 (not just actions) for the
  router to read confidence from.
