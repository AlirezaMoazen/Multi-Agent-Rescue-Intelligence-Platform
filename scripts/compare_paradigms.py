"""Three-paradigm comparison: APF (no ML) vs Epidemic Hysteretic Q (online
tabular) vs Neural MoE (offline-trained generalist).

Protocol (paired design — the strongest fairness guarantee):
  * The SAME N seeded 14x14 grids are used for every method.
  * Every method gets 30 attempts per grid with a 200-step cap (30 episodes
    with epsilon 1.0 and x0.85 decay is the dashboard's own default training
    regime for the fleet, so Epidemic Q is evaluated in its designed setting).
  * Reported conditions:
      - "attempt 1"  : first ever try on the unseen grid (zero-shot).
      - "attempt 30" : 30th episode on the same grid, still learning at the
        epsilon floor (0.05) — the fleet's normal operating regime, matching
        how the dashboard itself reports fleet success.
    Only Epidemic Q learns between attempts (Q-tables persist, epsilon decays
    1.0 -> 0.05, gossip syncs tables when agents meet). APF and the frozen
    MoE policy cannot improve across attempts by design — that contrast IS
    the result. A "frozen_greedy" probe (epsilon=0, learning off) is also
    recorded to JSON: freezing the position-keyed table collapses it into
    loops, which is exactly why the MoE runs its E3 learner LIVE.
  * Epidemic Q runs through the canonical fleet loop (MovementModel +
    CentralSensor + calculate_reward + gossip), mirroring visualization/api.py
    fleet mode, so it is not handicapped.
  * The MoE is the pure saved policy (checkpoints/moe.pt) evaluated greedily,
    exactly like /api/compare_policies — the live-E3 hybrid is excluded so the
    three methods stay independent.

Usage:  python scripts/compare_paradigms.py --grids 30
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from rescue_sim.config.settings import GridSettings
from rescue_sim.environment.grid import Position
from rescue_sim.environment.movement import MovementModel
from rescue_sim.environment.sensors import CentralSensor
from rescue_sim.MoE.pipeline import (
    ExplorationTeacher,
    FixedGridRescueEnv,
    build_peer_matrix,
    load_moe_policy,
)
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning
from rescue_sim.shared import (
    CARDINAL_ACTIONS,
    GossipConfig,
    HystereticConfig,
    RewardEvent,
    SPRINT3_REWARD_CONFIG,
    TargetType,
    calculate_reward,
)

ATTEMPTS = 30  # attempts per grid; attempt 1 and attempt 30 are reported


def make_env(grid_seed: int, args) -> FixedGridRescueEnv:
    settings = GridSettings(
        width=args.grid, height=args.grid,
        obstacle_probability=0.15,
        target_a_count=2, target_b_count=2,
    )
    return FixedGridRescueEnv(
        settings, num_agents=args.agents, max_steps=args.max_steps,
        view_radius=args.view_radius, seed=grid_seed, grid_seed=grid_seed,
    )


# ── Neural MoE: greedy rollout of the saved policy (deterministic) ─────────
def run_moe_episode(policy, env) -> dict:
    obs = env.reset()
    hidden = None
    done = False
    info = {"success": False, "rescued": 0, "steps": 0}
    while not done:
        peer_np = build_peer_matrix(env.positions)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        peer_t = torch.tensor(peer_np, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(env.valid_action_mask(), dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            y_final, _w, hidden = policy(obs_t, peer_t, mask_t, hidden)
        actions = torch.argmax(y_final.squeeze(0), dim=-1).numpy()
        obs, _r, done, info = env.step(actions)
    return {"success": bool(info["success"]), "rescued": int(info["rescued"]),
            "steps": int(info["steps"])}


# ── APF: the non-ML potential-fields baseline ───────────────────────────────
def run_apf_episode(teacher, env) -> dict:
    obs = env.reset()  # noqa: F841 - APF acts from env state, not the obs vector
    teacher.reset(env)
    done = False
    info = {"success": False, "rescued": 0, "steps": 0}
    while not done:
        actions = teacher.act(env, env.valid_action_mask())
        _obs, _r, done, info = env.step(actions)
        teacher.observe(env)
    return {"success": bool(info["success"]), "rescued": int(info["rescued"]),
            "steps": int(info["steps"])}


# ── Epidemic Hysteretic Q: canonical fleet loop (mirrors api.py fleet mode) ─
def run_eq_episode(fleet, grid, starts, sensor_range: int, max_steps: int,
                   learn: bool) -> dict:
    movement = MovementModel()
    sensor = CentralSensor(grid)
    positions = dict(starts)
    fleet.reset_positions(positions)
    active_targets = set(grid.target_a_positions) | set(grid.target_b_positions)
    total_targets = len(active_targets)
    rescued_xy: set[tuple[int, int]] = set()
    visited_by_agent = {aid: {(p.x, p.y)} for aid, p in positions.items()}
    steps = 0

    for step in range(max_steps):
        if not active_targets:
            break
        action_indices = fleet.select_actions()
        rewards: dict[str, float] = {}
        next_positions: dict[str, Position] = {}
        dones: dict[str, bool] = {}
        for agent_id in sorted(positions):
            action = CARDINAL_ACTIONS[action_indices[agent_id]]
            before = positions[agent_id]
            after = movement.apply(grid, before, action.value).end
            positions[agent_id] = after
            next_positions[agent_id] = after
            next_obs = sensor.observe(agent_id, after, sensor_range)
            target_type = grid.target_type_at(after)
            rescued_type = None
            if target_type is not None and (after.x, after.y) not in rescued_xy:
                rescued_xy.add((after.x, after.y))
                active_targets = {t for t in active_targets
                                  if (t.x, t.y) != (after.x, after.y)}
                rescued_type = TargetType(target_type)
            done = not active_targets
            rewards[agent_id] = calculate_reward(
                RewardEvent(
                    moved=(after != before),
                    move=action.value,
                    newly_discovered_cells=len(next_obs.newly_discovered_cells),
                    rescued_target_type=rescued_type,
                    completed_episode=done,
                    repeated_cell=(after.x, after.y) in visited_by_agent[agent_id],
                ),
                SPRINT3_REWARD_CONFIG,
            )
            dones[agent_id] = done
            visited_by_agent[agent_id].add((after.x, after.y))
        if learn:
            fleet.record_transitions(action_indices, rewards, next_positions, dones)
            fleet.gossip()
        else:
            fleet.reset_positions(next_positions)
        steps = step + 1

    return {"success": not active_targets,
            "rescued": total_targets - len(active_targets), "steps": steps}


# ── Aggregation ─────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r["success"])
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "success_rate": k / n,
        "ci95": [round(lo, 3), round(hi, 3)],
        "avg_rescued": sum(r["rescued"] for r in rows) / n,
        "avg_steps": sum(r["steps"] for r in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, default=30)
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--view-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=20260713)
    parser.add_argument("--checkpoint", default="checkpoints/moe.pt")
    parser.add_argument("--out", default="", help="optional JSON output path")
    args = parser.parse_args()

    loaded = load_moe_policy(args.checkpoint)
    if loaded is None:
        raise SystemExit(f"no MoE checkpoint at {args.checkpoint} — run scripts/pretrain_moe.py")
    policy, shape, epochs = loaded
    expected = (args.grid, args.grid, args.agents, args.view_radius)
    if tuple(shape) != expected:
        raise SystemExit(f"checkpoint shape {shape} != requested {expected}")
    policy = policy.to("cpu").eval()
    print(f"MoE checkpoint: {args.checkpoint} (shape={shape}, epochs={epochs})")
    print(f"Protocol: {args.grids} paired grids | {ATTEMPTS} attempts each | "
          f"{args.grid}x{args.grid}, {args.agents} agents, 4 targets, "
          f"cap {args.max_steps} steps\n")

    results = {m: {"attempt1": [], "attempt30": []} for m in ("APF", "EpidemicQ", "MoE")}
    apf_rng = np.random.default_rng(args.seed_base)
    t0 = time.time()

    for g in range(args.grids):
        grid_seed = args.seed_base + g
        env = make_env(grid_seed, args)
        env.reset()  # materialize the grid
        grid = env.grid
        starts = {f"agent-{i}": Position(0, 0) for i in range(args.agents)}

        # MoE: greedy on a fixed grid is deterministic -> one rollout serves
        # both conditions (the policy cannot change between attempts).
        moe_metrics = run_moe_episode(policy, env)
        results["MoE"]["attempt1"].append(moe_metrics)
        results["MoE"]["attempt30"].append(moe_metrics)

        # APF: stateless (stochastic tie-breaking) -> independent samples.
        teacher = ExplorationTeacher(apf_rng)
        results["APF"]["attempt1"].append(run_apf_episode(teacher, env))
        results["APF"]["attempt30"].append(run_apf_episode(teacher, env))

        # Epidemic Q: fresh fleet per grid; Q-tables persist across attempts.
        # Hyper-parameters match the dashboard fleet mode exactly
        # (visualization/api.py: alpha=learning_rate=0.1, beta=min(0.1, alpha),
        # discount=0.9, epsilon 1.0 with x0.85 decay, floor 0.05).
        fleet = EpidemicHystereticQLearning(
            grid,
            HystereticConfig(alpha=0.1, beta=0.1, discount_factor=0.9, epsilon=1.0),
            GossipConfig(comm_radius=float(args.view_radius)),
            max_agents=args.agents,
            seed=grid_seed,
        )
        for aid, start in starts.items():
            fleet.add_agent(aid, start)
        curve = []
        for attempt in range(ATTEMPTS):               # attempts 1..30 learn
            fleet.epsilon = max(0.05, 1.0 * (0.85 ** attempt))
            m = run_eq_episode(fleet, grid, starts, args.view_radius,
                               args.max_steps, learn=True)
            curve.append({"success": m["success"], "rescued": m["rescued"]})
            if attempt == 0:
                results["EpidemicQ"]["attempt1"].append(m)
            if attempt == ATTEMPTS - 1:
                results["EpidemicQ"]["attempt30"].append(m)
        results["EpidemicQ"].setdefault("curves", []).append(curve)
        # Frozen-greedy probe (JSON only): epsilon=0, learning off. Shows the
        # position-keyed table collapsing into loops once it stops adapting.
        fleet.epsilon = 0.0
        results["EpidemicQ"].setdefault("frozen_greedy", []).append(
            run_eq_episode(fleet, grid, starts, args.view_radius,
                           args.max_steps, learn=False)
        )
        print(f"  grid {g + 1:>2}/{args.grids} done ({time.time() - t0:5.1f}s)")

    summary = {
        method: {cond: aggregate(rows) for cond, rows in conds.items()
                 if cond != "curves"}
        for method, conds in results.items()
    }
    # Per-attempt EQ learning curve (mean success / rescued over grids).
    curves = results["EpidemicQ"].get("curves", [])
    if curves:
        n = len(curves)
        summary["EpidemicQ"]["learning_curve"] = {
            "success_rate": [round(sum(c[a]["success"] for c in curves) / n, 3)
                             for a in range(ATTEMPTS)],
            "avg_rescued": [round(sum(c[a]["rescued"] for c in curves) / n, 3)
                            for a in range(ATTEMPTS)],
        }

    print(f"\n=== {args.grids} paired grids, {args.grid}x{args.grid}, "
          f"{args.agents} agents, 4 targets, {args.max_steps}-step cap ===")
    header = (f"{'method':<12}{'condition':<26}{'success':>9}{'95% CI':>14}"
              f"{'rescued/4':>11}{'steps':>8}")
    print(header)
    print("-" * len(header))
    for method in ("APF", "EpidemicQ", "MoE"):
        for cond, label in (("attempt1", "unseen grid (attempt 1)"),
                            ("attempt30", "same grid, attempt 30")):
            s = summary[method][cond]
            ci = f"[{s['ci95'][0] * 100:.0f}-{s['ci95'][1] * 100:.0f}%]"
            print(f"{method:<12}{label:<26}{s['success_rate'] * 100:>8.0f}%"
                  f"{ci:>14}{s['avg_rescued']:>11.2f}{s['avg_steps']:>8.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"protocol": vars(args), "summary": summary}, indent=2))
        print(f"\nsaved JSON -> {args.out}")


if __name__ == "__main__":
    main()
