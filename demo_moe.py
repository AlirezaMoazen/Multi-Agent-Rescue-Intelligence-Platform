# Copyright 2026 Alireza Moazen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Production demonstration of the step-level Neural Mixture of Experts (MoE) policy.

Runs the full pipeline against the real 20x20 ``RescueEnv`` — no synthetic
shortcuts or mock iterations:

    1. Collect expert trajectories on the 20x20 grid from three heuristic
       teachers (frontier exploration, target-greedy coordination, and a
       direction-persistent fallback for isolated agents).
    2. Phase A — full behavioral-cloning training of the expert heads with
       live progress bars (epoch, cross-entropy loss, BC accuracy, grad norm),
       followed by attention-router fine-tuning with communication-blackout
       penalties.
    3. Phase B — live 20x20 ASCII grid render of a policy-driven rollout
       (``.`` empty, ``#`` walls, ``G0-G3`` goals, ``A0-A3`` agents).
    4. Phase C — per-step telemetry table: step index, agent coordinates,
       active peer count, baseline parameters, and the live softmax routing
       vector ``[g_explore, g_coord, g_fallback]`` plus the GRU hidden norm.
    5. Integration gate — automatically executes ``pytest tests/test_moe.py``
       to confirm loss minimization, BC accuracy, and multi-agent dimension
       flows compile without runtime errors.

The training pipeline itself (teachers, dataset collection, behavioral
cloning, router optimization) lives in ``rescue_sim.MoE.pipeline`` and is
shared with the web visualization backend.

Usage:
    python demo_moe.py                       # full production run
    python demo_moe.py --epochs 30 --seed 3  # custom configuration
    python demo_moe.py --skip-tests          # skip the pytest gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent

try:
    from rescue_sim.config.settings import GridSettings
    from rescue_sim.MAPPO import RescueEnv
    from rescue_sim.MoE.moe import NeuralMoEPolicy
    from rescue_sim.MoE.pipeline import (
        collect_expert_dataset,
        make_teachers,
        run_expert_distillation,
        run_router_optimization,
        build_peer_matrix,
        ExplorationMemory,
        ACTION_DIM,
        EXPERT_NAMES,
    )
    from rescue_sim.shared import Position
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from rescue_sim.config.settings import GridSettings
    from rescue_sim.MAPPO import RescueEnv
    from rescue_sim.MoE.moe import NeuralMoEPolicy
    from rescue_sim.MoE.pipeline import (
        collect_expert_dataset,
        make_teachers,
        run_expert_distillation,
        run_router_optimization,
        build_peer_matrix,
        ExplorationMemory,
        ACTION_DIM,
        EXPERT_NAMES,
    )
    from rescue_sim.shared import Position

GRID_SIZE: int = 20
NUM_AGENTS: int = 4
VIEW_RADIUS: int = 3  # 7x7 ego-centric window (3-block blindness constraint)


# ===========================================================================
# Phase A: live training teleprinter
# ===========================================================================

def print_progress_bar(
    label: str,
    epoch: int,
    total_epochs: int,
    loss: float,
    acc: float,
    grad_norm: float,
) -> None:
    """Renders a dynamic ASCII progress bar for live training feedback.

    Args:
        label: Short stage label (e.g. "BC" or "Gate").
        epoch: Current epoch number (1-indexed).
        total_epochs: Total epoch count.
        loss: Current cross-entropy (or penalty) loss value.
        acc: Behavioral-cloning validation accuracy (%).
        grad_norm: L2 gradient norm of the trainable parameters.
    """
    width = 25
    filled = int(width * epoch / total_epochs)
    bar = ("=" * filled + ">" + "." * (width - filled - 1))[:width]
    print(
        f"{label} {epoch:03d}/{total_epochs:03d} [{bar}] "
        f"Loss: {loss:.6f} | BC-Acc: {acc:6.2f}% | Grad Norm: {grad_norm:.4f}"
    )


# ===========================================================================
# Phase B & C: live dashboard rendering
# ===========================================================================

def render_grid_20x20(
    positions: list[Position],
    goals: list[Position],
    rescued: set[Position],
    obstacles: frozenset[Position],
) -> None:
    """Renders the live 20x20 ASCII grid world to the terminal.

    Legend: ``.`` empty | ``#`` wall | ``G0-G3`` goals | ``*`` rescued | ``A0-A3`` agents.

    Args:
        positions: Current agent positions.
        goals: All goal positions in stable index order (G0..G3).
        rescued: Goal positions already rescued (rendered as ``*``).
        obstacles: Static wall positions.
    """
    grid: list[list[str]] = [[" . " for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for p in obstacles:
        grid[p.y][p.x] = " # "
    for idx, g in enumerate(goals):
        grid[g.y][g.x] = " * " if g in rescued else f"G{idx} "
    for idx, a in enumerate(positions):
        grid[a.y][a.x] = f"A{idx} "

    print("\n[PHASE B: 20x20 GRID WORLD VIEW]  ( . empty | # wall | G0-G3 goals | * rescued | A0-A3 agents )")
    print("   " + "".join(f"{i:^3}" for i in range(GRID_SIZE)))
    print("   " + "---" * GRID_SIZE)
    for r_idx, row in enumerate(grid):
        print(f"{r_idx:2d}|" + "".join(row) + "|")
    print("   " + "---" * GRID_SIZE)


def print_telemetry_table(
    step: int,
    positions: list[Position],
    weights: torch.Tensor,
    peer_matrix: torch.Tensor,
    fallback_hidden: torch.Tensor,
) -> None:
    """Prints the parametric evolution telemetry table with GRU state info.

    Args:
        step: Current simulation step index.
        positions: Agent positions at this step.
        weights: [1, A, 3] softmax gating weights.
        peer_matrix: [1, A, A] peer adjacency tensor.
        fallback_hidden: [1, A, H] GRU hidden state tensor.
    """
    print("\n[PHASE C: PARAMETRIC EVOLUTION TELEMETRY]")
    header = (
        f"{'Step':<6}{'Agent':<9}{'Pos':<10}{'Peers':<7}"
        f"{'Baseline Params':<30}"
        f"{'MoE Gating [g_exp, g_coord, g_fall]':<38}"
        f"{'GRU |h|':<9}"
    )
    print(header)
    print("-" * 109)

    peer_count = torch.sum(peer_matrix, dim=-1).squeeze(0)    # [A]
    weights_step = weights.squeeze(0)                          # [A, 3]
    h_norms = torch.norm(fallback_hidden.squeeze(0), dim=-1)   # [A]

    for a, pos in enumerate(positions):
        pos_str = f"({pos.x},{pos.y})"
        if int(peer_count[a].item()) <= 1:
            baseline_str = "Hyst Q: α=0.10, β=0.01 [ISO]"
        else:
            baseline_str = "Frontier Exploration: γ=0.95"
        w_str = (
            f"[{weights_step[a, 0]:.4f}, "
            f"{weights_step[a, 1]:.4f}, "
            f"{weights_step[a, 2]:.4f}]"
        )
        print(
            f"{step:<6}{f'Agent-{a}':<9}{pos_str:<10}"
            f"{int(peer_count[a].item()):<7}{baseline_str:<30}"
            f"{w_str:<38}{h_norms[a].item():<9.4f}"
        )
    print("=" * 109)


def run_live_simulation(
    policy: NeuralMoEPolicy,
    env: RescueEnv,
    sim_steps: int,
    render_every: int,
) -> dict[str, float]:
    """Phase B/C: policy-driven rollout on a fresh 20x20 grid with GRU tracking.

    Actions are sampled from the blended masked logits under
    ``torch.no_grad()``; the GRU hidden state persists across the whole
    episode timeline.

    Args:
        policy: The trained NeuralMoEPolicy.
        env: The 20x20 rescue environment.
        sim_steps: Maximum simulation steps.
        render_every: Grid/telemetry render period (step 1 and the final step
            always render).

    Returns:
        Episode summary: rescued targets, total targets, success flag, steps.
    """
    policy = policy.to("cpu")  # training may run on GPU; serve inference on CPU
    obs_np = env.reset()
    memory = ExplorationMemory(env.num_agents)
    memory.observe(env.positions)
    assert env.grid is not None
    goals: list[Position] = sorted(
        env.grid.target_a_positions | env.grid.target_b_positions,
        key=lambda p: (p.y, p.x),
    )
    rescued: set[Position] = set()
    fallback_hidden: Optional[torch.Tensor] = None
    info: dict[str, float] = {"rescued": 0, "targets": len(goals), "success": False, "steps": 0}

    print("\n" + "=" * 96)
    print(f"   SIMULATION: policy-driven rollout on a fresh {GRID_SIZE}x{GRID_SIZE} grid "
          f"(max {sim_steps} steps, render every {render_every})")
    print("=" * 96)

    for step in range(1, sim_steps + 1):
        peer_np = build_peer_matrix(env.positions)

        obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)      # [1, A, obs_dim]
        peer_t = torch.tensor(peer_np, dtype=torch.float32).unsqueeze(0)    # [1, A, A]
        mask_t = torch.tensor(env.valid_action_mask(), dtype=torch.bool).unsqueeze(0)

        # Tensor shape assertions before the routing step
        assert obs_t.shape == (1, env.num_agents, env.obs_dim), \
            f"obs shape: expected [1, {env.num_agents}, {env.obs_dim}], got {obs_t.shape}"
        assert peer_t.shape == (1, env.num_agents, env.num_agents), \
            f"peer shape: expected [1, {env.num_agents}, {env.num_agents}], got {peer_t.shape}"
        assert mask_t.shape == (1, env.num_agents, policy.action_dim), \
            f"mask shape: expected [1, {env.num_agents}, {policy.action_dim}], got {mask_t.shape}"

        valid_mask = env.valid_action_mask()
        with torch.no_grad():
            y_final, weights, fallback_hidden = policy(obs_t, peer_t, mask_t, fallback_hidden)
            # Anti-revisit bias steers exploration toward unvisited ground so
            # far-away targets are actually reached (replaces the old blind
            # 10% random jitter, which broke loops but also broke rescues).
            scores = memory.bias_logits(env, obs_np, y_final.squeeze(0), valid_mask)
            actions = torch.argmax(scores, dim=-1)

        # Tiny residual epsilon still breaks exact policy ties
        actions = actions.numpy().copy()
        for i in range(env.num_agents):
            if np.random.random() < 0.03:
                valid = np.flatnonzero(valid_mask[i])
                if len(valid):
                    actions[i] = np.random.choice(valid)
        actions = torch.tensor(actions)

        positions_before = list(env.positions)
        if step == 1 or step % render_every == 0:
            render_grid_20x20(positions_before, goals, rescued, env.grid.obstacles)
            print_telemetry_table(step, positions_before, weights, peer_t, fallback_hidden)

        obs_np, _, done, step_info = env.step(actions.numpy())
        memory.observe(env.positions)
        info = step_info
        rescued.update(g for g in goals if g in set(env.positions))

        if done:
            render_grid_20x20(list(env.positions), goals, rescued, env.grid.obstacles)
            print_telemetry_table(step, list(env.positions), weights, peer_t, fallback_hidden)
            break

    print(
        f"\n[SIMULATION SUMMARY] steps: {info['steps']} | "
        f"rescued: {info['rescued']}/{info['targets']} | success: {bool(info['success'])}"
    )
    return dict(info)


# ===========================================================================
# Integration gate: automatic pytest execution
# ===========================================================================

def run_test_suite() -> int:
    """Executes the local pytest suite for the MoE module and reports the result.

    Returns:
        The pytest process exit code (0 = all tests green).
    """
    print("\n" + "=" * 96)
    print("   INTEGRATION GATE: pytest tests/test_moe.py")
    print("=" * 96)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_moe.py", "-q"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    print(output[-2000:] if len(output) > 2000 else output)
    verdict = "ALL TESTS PASSED" if result.returncode == 0 else "TESTS FAILED"
    print(f"\n[INTEGRATION GATE] {verdict} (exit code {result.returncode})")
    return result.returncode


# ===========================================================================
# Main production pipeline
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parses the production run configuration."""
    parser = argparse.ArgumentParser(description="Neural MoE production demonstration (20x20 grid).")
    parser.add_argument("--epochs", type=int, default=20, help="behavioral cloning epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="BC mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (both stages)")
    parser.add_argument("--episodes-per-head", type=int, default=6,
                        help="20x20 collection episodes per expert teacher")
    parser.add_argument("--collect-steps", type=int, default=80,
                        help="step cap per collection episode")
    parser.add_argument("--router-steps", type=int, default=120,
                        help="attention router optimization steps")
    parser.add_argument("--router-batch", type=int, default=16,
                        help="real states sampled per router step")
    # 150 steps: a 20x20 grid with targets far from the start corner needs
    # room to cross (~19 steps) and sweep; 60 steps routinely timed out with
    # distant targets still undiscovered.
    parser.add_argument("--sim-steps", type=int, default=150, help="live simulation step cap")
    parser.add_argument("--render-every", type=int, default=5, help="dashboard render period")
    parser.add_argument("--seed", type=int, default=7, help="global random seed")
    parser.add_argument("--skip-tests", action="store_true", help="skip the pytest gate")
    return parser.parse_args()


def main() -> int:
    """Full production pipeline: collect -> train -> simulate -> test.

    Returns:
        Process exit code (non-zero if the pytest gate fails).
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 96)
    print("   NEURAL MIXTURE OF EXPERTS (MoE) — PRODUCTION DEMONSTRATION")
    print("   Attention-Based Router | GRU Temporal Fallback | Real 20x20 RescueEnv")
    print("=" * 96)

    grid = GridSettings(
        width=GRID_SIZE, height=GRID_SIZE, obstacle_probability=0.15,
        target_a_count=2, target_b_count=2,
    )
    env = RescueEnv(
        grid, num_agents=NUM_AGENTS, max_steps=max(args.collect_steps, args.sim_steps),
        view_radius=VIEW_RADIUS, seed=args.seed,
    )
    policy = NeuralMoEPolicy(env.obs_dim, NUM_AGENTS, VIEW_RADIUS, ACTION_DIM)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nEnvironment: {GRID_SIZE}x{GRID_SIZE} grid | {NUM_AGENTS} agents | "
          f"7x7 ego window (view radius {VIEW_RADIUS}) | obs_dim {env.obs_dim}")
    print(f"Model Parameters: {total_params:,} total")

    # -- Trajectory collection on the real grid -----------------------------
    print(f"\nCollecting expert trajectories "
          f"({args.episodes_per_head} episodes x {args.collect_steps} steps per teacher) ...")
    datasets = [
        collect_expert_dataset(env, teacher, args.episodes_per_head, args.collect_steps)
        for teacher in make_teachers(rng)
    ]

    # -- Phase A: full training loops ----------------------------------------
    print("\n[PHASE A: LIVE TRAINING TELEPRINTER]")
    print("=" * 96)
    print("Stage 1: Expert Policy Distillation (Behavioral Cloning on the 20x20 grid)")
    print("-> Freezing Gating Router weights; training encoder + 3 expert heads.")
    sizes = " | ".join(f"{n}: {len(d)} transitions" for n, d in zip(EXPERT_NAMES, datasets))
    print(f"-> Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | {sizes}")
    print("-" * 96)
    run_expert_distillation(
        policy, datasets, args.epochs, args.batch_size, args.lr, args.seed,
        on_epoch=lambda e, t, loss, acc, g: print_progress_bar("BC  ", e, t, loss, acc, g),
    )

    print("\nStage 2: Attention-Based Gating Router Optimization")
    print("-> Freezing expert heads; fine-tuning scaled dot-product attention router.")
    print(f"-> Steps: {args.router_steps} | Batch: {args.router_batch} "
          f"(x3 connectivity augmentation) | LR: {args.lr}")
    print("-" * 96)
    run_router_optimization(
        policy, datasets, args.router_steps, args.router_batch, args.lr, args.seed,
        on_step=lambda s, t, loss, acc, g: (
            print_progress_bar("Gate", s, t, loss, acc, g)
            if s == 1 or s % 10 == 0 or s == t else None
        ),
    )
    print("=" * 96)

    # -- Phase B & C: live dashboard rollout ---------------------------------
    run_live_simulation(policy, env, args.sim_steps, args.render_every)

    # -- Integration gate -----------------------------------------------------
    exit_code = 0 if args.skip_tests else run_test_suite()
    print("\n[DEMO COMPLETE] All phases executed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
