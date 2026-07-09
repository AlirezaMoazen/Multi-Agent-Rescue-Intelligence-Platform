# Copyright 2026 TUHH Group 05 — A. Herrero Callejo, C. Marcos Alonso,
# M. M. Orfany, A. Moazzen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Training pipeline for the Neural MoE policy on the real rescue environment.

Shared by ``demo_moe.py`` (terminal dashboard) and the visualization API
(live web dashboard): heuristic expert teachers label trajectories on real
grids, the expert heads are behavioral-cloned, and the attention router is
fine-tuned with communication-blackout penalties. Progress is reported
through optional callbacks so each caller renders it its own way.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rescue_sim.MAPPO.environment import RescueEnv
from rescue_sim.MoE.moe import NeuralMoEPolicy
from rescue_sim.shared import CARDINAL_ACTIONS, MOVE_DELTAS, Position

ACTION_DIM: int = 4       # N, S, E, W
COMM_RADIUS: int = 3      # Manhattan distance threshold for peer links
EXPERT_NAMES: tuple[str, str, str] = ("exploration", "coordination", "fallback")

# (dx, dy) per action index, in CARDINAL_ACTIONS order (N, S, E, W).
ACTION_DELTAS: list[tuple[int, int]] = [MOVE_DELTAS[a.value] for a in CARDINAL_ACTIONS]

# One team transition: (obs [A, obs_dim], peer_matrix [A, A], actions [A]).
Transition = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# One collected episode, in step order (the GRU head trains on sequences).
Episode = list[Transition]

# Progress callbacks: (current, total, loss, accuracy_pct, grad_norm)
ProgressCallback = Callable[[int, int, float, float, float], None]


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Selects the compute device: an explicit override, else CUDA when a GPU
    is available, else CPU.

    The models here are small and most of the wall-clock cost is the Python
    environment rollouts (teacher data collection), so a GPU rarely speeds
    training up — but honoring one when present costs nothing and keeps the
    pipeline portable to a CUDA host.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_peer_matrix(positions: list[Position]) -> np.ndarray:
    """Builds the dynamic peer adjacency matrix from agent positions.

    A link exists where the Manhattan distance is below ``COMM_RADIUS``;
    the diagonal (self link) is always 1.

    Args:
        positions: Current agent positions.

    Returns:
        peer_matrix: [A, A] float32 adjacency matrix.
    """
    n = len(positions)
    peer = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            dist = abs(positions[i].x - positions[j].x) + abs(positions[i].y - positions[j].y)
            if dist < COMM_RADIUS:
                peer[i, j] = 1.0
    return peer


class ExplorationMemory:
    """Rollout-time anti-revisit bias (decentralized, per-agent visit counts).

    Why: the policy only sees a small ego window, so a target far from the
    start positions is invisible for most of the episode. The observation has
    no "already visited" channel, so the memoryless feed-forward heads can
    loop over already-searched ground and never reach a distant target. This
    helper gives each agent its own visited-cell counter and penalizes moves
    whose destination was already visited — pushing the search frontier
    outward instead of relying on random jitter.

    The bias only applies while an agent has NO live target in its ego
    window; the moment one is visible the raw policy logits decide, so
    close-range rescue behavior is completely untouched. Each agent uses only
    its own history — the scheme stays honestly decentralized.
    """

    def __init__(self, num_agents: int, beta: float = 0.6) -> None:
        self.beta = beta
        # Keyed by plain (x, y) tuples: the codebase has two distinct Position
        # dataclasses (shared vs environment.grid) that hash alike but never
        # compare equal across classes, which would silently break dict lookups.
        self.visits: list[dict[tuple[int, int], int]] = [{} for _ in range(num_agents)]

    def observe(self, positions: list[Position]) -> None:
        """Records the agents' current cells (call after reset and each step)."""
        for i, pos in enumerate(positions):
            key = (pos.x, pos.y)
            self.visits[i][key] = self.visits[i].get(key, 0) + 1

    def bias_logits(
        self,
        env: RescueEnv,
        obs: np.ndarray,
        logits: torch.Tensor,
        valid_mask: np.ndarray,
    ) -> torch.Tensor:
        """Returns [A, 4] scores: masked log-probs minus the visit penalty.

        Log-probs (not raw logits) give a bounded, comparable scale so one
        ``beta`` works regardless of how large the blended expert logits are.
        """
        win, channels = 2 * env.view_radius + 1, 4  # matches RescueEnv obs layout
        win_dim = win * win * channels
        window = obs[:, :win_dim].reshape(env.num_agents, win, win, channels)
        has_target = window[..., 1:3].max(axis=(1, 2, 3)) > 0.5  # [A]

        mask_t = torch.tensor(valid_mask, dtype=torch.bool)
        scores = torch.log_softmax(logits.masked_fill(~mask_t, -1e9), dim=-1)
        for i, pos in enumerate(env.positions):
            if has_target[i]:
                continue  # target in sight: let the pure policy act
            for a, (dx, dy) in enumerate(ACTION_DELTAS):
                if not valid_mask[i, a]:
                    continue
                nxt = (pos.x + dx, pos.y + dy)
                scores[i, a] = scores[i, a] - self.beta * self.visits[i].get(nxt, 0)
        return scores.masked_fill(~mask_t, -1e9)


class FixedGridRescueEnv(RescueEnv):
    """RescueEnv that regenerates the *same* grid on every reset.

    The base env draws a fresh random grid per episode (right for training
    generalization). The live dashboard instead shows the MoE solving one
    fixed competition grid over repeated tries, so the routing evolution is
    comparable try-to-try.
    """

    def __init__(self, *args: object, grid_seed: int = 0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._grid_seed = grid_seed

    def reset(self) -> np.ndarray:
        self.rng = np.random.default_rng(self._grid_seed)
        return super().reset()


# ===========================================================================
# Heuristic expert teachers (trajectory labellers on the real grid)
#
# Every label is a function of what the agent can actually OBSERVE (the 7x7
# ego window: targets, teammates, walls) — otherwise behavioral cloning has
# nothing to learn from. All three teachers rescue a target the moment one
# enters the window; they differ in what they do when none is visible.
# ===========================================================================

class _RescueTeacher:
    """Shared bookkeeping: live-target tracking and window-visibility helpers."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.remaining: set[Position] = set()

    def reset(self, env: RescueEnv) -> None:
        assert env.grid is not None
        self.remaining = set(env.grid.target_a_positions | env.grid.target_b_positions)
        self._reset_state(env)

    def _reset_state(self, env: RescueEnv) -> None:
        """Per-teacher episode state; overridden where needed."""

    def observe(self, env: RescueEnv) -> None:
        """Removes targets an agent has just rescued (landed on)."""
        for pos in env.positions:
            self.remaining.discard(pos)

    def _visible_target(self, env: RescueEnv, i: int) -> Optional[Position]:
        """Nearest live target inside agent i's ego window, if any."""
        pos = env.positions[i]
        best: Optional[Position] = None
        best_dist = 10 ** 9
        for t in self.remaining:
            if abs(t.x - pos.x) <= env.view_radius and abs(t.y - pos.y) <= env.view_radius:
                dist = abs(t.x - pos.x) + abs(t.y - pos.y)
                if dist < best_dist:
                    best, best_dist = t, dist
        return best

    def _visible_teammates(self, env: RescueEnv, i: int) -> list[Position]:
        """Teammates inside agent i's ego window."""
        pos = env.positions[i]
        return [
            p for j, p in enumerate(env.positions)
            if j != i
            and abs(p.x - pos.x) <= env.view_radius
            and abs(p.y - pos.y) <= env.view_radius
        ]

    def _step_toward(self, env: RescueEnv, i: int, goal: Position, valid_mask: np.ndarray) -> int:
        """Valid action that most reduces the Manhattan distance to goal."""
        pos = env.positions[i]
        best_dist, best_actions = 10 ** 9, []
        for a, (dx, dy) in enumerate(ACTION_DELTAS):
            if not valid_mask[i, a]:
                continue
            dist = abs(goal.x - (pos.x + dx)) + abs(goal.y - (pos.y + dy))
            if dist < best_dist:
                best_dist, best_actions = dist, [a]
            elif dist == best_dist:
                best_actions.append(a)
        if not best_actions:
            return int(self.rng.integers(0, ACTION_DIM))
        return int(self.rng.choice(best_actions))

    def _toward_open_space(self, env: RescueEnv, i: int, valid_mask: np.ndarray) -> int:
        """Valid action whose side of the ego window holds the most free cells.

        The blocked channel is part of the observation, so unlike a random
        label this rule is learnable by behavioral cloning.
        """
        assert env.grid is not None
        pos, r = env.positions[i], env.view_radius
        best_open, best_actions = -1, []
        for a, (dx, dy) in enumerate(ACTION_DELTAS):
            if not valid_mask[i, a]:
                continue
            open_cells = 0
            for cy in range(pos.y - r, pos.y + r + 1):
                for cx in range(pos.x - r, pos.x + r + 1):
                    if (cx - pos.x) * dx + (cy - pos.y) * dy <= 0:
                        continue
                    cell = Position(cx, cy)
                    if env.grid.contains(cell) and not env.grid.is_blocked(cell):
                        open_cells += 1
            if open_cells > best_open:
                best_open, best_actions = open_cells, [a]
            elif open_cells == best_open:
                best_actions.append(a)
        if not best_actions:
            return int(self.rng.integers(0, ACTION_DIM))
        return int(self.rng.choice(best_actions))


class ExplorationTeacher(_RescueTeacher):
    """Expert 1 teacher: the real APF baseline (``Qlearning.baseline.APFExplorer``).

    Thin adapter that builds per-agent ``LearningState``s from the ego window
    and delegates action choice to the Artificial Potential Fields strategy —
    so E1 behaviorally clones the project's actual non-AI multi-robot
    algorithm rather than an ad-hoc rule. Every APF force term is computable
    from the window, which keeps the policy learnable by the feed-forward head.
    """

    def _reset_state(self, env: RescueEnv) -> None:
        from rescue_sim.Qlearning.baseline import APFExplorer

        self._apf = APFExplorer(seed=int(self.rng.integers(0, 2**31 - 1)))

    def act(self, env: RescueEnv, valid_mask: np.ndarray) -> np.ndarray:
        from rescue_sim.shared import LearningState

        assert env.grid is not None
        r = env.view_radius
        actions = np.zeros(env.num_agents, dtype=np.int64)
        for i, pos in enumerate(env.positions):
            window = [
                Position(cx, cy)
                for cy in range(pos.y - r, pos.y + r + 1)
                for cx in range(pos.x - r, pos.x + r + 1)
                if env.grid.contains(Position(cx, cy))
            ]
            live_visible = frozenset(
                t for t in self.remaining
                if abs(t.x - pos.x) <= r and abs(t.y - pos.y) <= r
            )
            state = LearningState(
                agent_id=f"a{i}",
                agent_position=pos,
                visible_cells=frozenset(window),
                visible_obstacles=frozenset(
                    c for c in window if env.grid.is_blocked(c)
                ),
                visible_target_a_positions=live_visible,
            )
            valid = tuple(
                CARDINAL_ACTIONS[a]
                for a in range(len(CARDINAL_ACTIONS))
                if valid_mask[i, a]
            )
            if not valid:
                actions[i] = 0
                continue
            chosen = self._apf.select_action(f"a{i}", state, valid)
            if chosen in CARDINAL_ACTIONS:
                actions[i] = CARDINAL_ACTIONS.index(chosen)
            else:  # WAIT fallback: take any valid move
                actions[i] = int(np.flatnonzero(valid_mask[i])[0])
        return actions


class CoordinationTeacher(_RescueTeacher):
    """Expert 2 (deep coordination style): rescue what you see, stay linked.

    When a teammate is drifting toward the edge of the communication radius,
    close the gap; while comfortably linked, sweep together. Distilled from
    the same cohesive movement pattern the deep CTDE methods (QMIX/MAPPO)
    learn on this task.
    """

    def act(self, env: RescueEnv, valid_mask: np.ndarray) -> np.ndarray:
        actions = np.zeros(env.num_agents, dtype=np.int64)
        for i, pos in enumerate(env.positions):
            target = self._visible_target(env, i)
            if target is not None:
                actions[i] = self._step_toward(env, i, target, valid_mask)
                continue
            teammates = self._visible_teammates(env, i)
            if teammates:
                nearest = min(
                    teammates, key=lambda t: abs(t.x - pos.x) + abs(t.y - pos.y)
                )
                gap = abs(nearest.x - pos.x) + abs(nearest.y - pos.y)
                if gap >= COMM_RADIUS - 1:
                    actions[i] = self._step_toward(env, i, nearest, valid_mask)
                else:
                    actions[i] = self._toward_open_space(env, i, valid_mask)
            else:
                actions[i] = self._toward_open_space(env, i, valid_mask)
        return actions


class FallbackTeacher(_RescueTeacher):
    """Expert 3 BC teacher: a learnable lone-wolf sweep for the GRU head.

    Rescues any target it sees; otherwise keeps its heading and rotates
    clockwise on walls (wall-following). Every decision is inferable from the
    ego window + short memory, so behavioral cloning works. Distilling the
    *real* epidemic hysteretic Q policy was tried and measured at 0-8% MoE
    success (vs ~57%): its greedy policy is keyed to a converged per-grid
    Q-table the local observation cannot expose. The genuine epidemic
    Q-learner instead runs LIVE during dashboard rollouts (see
    visualization/api.py::_run_moe_rollout), where it learns the fixed
    competition grid across tries — tabular learning needs grid persistence,
    not distillation.
    """

    def _reset_state(self, env: RescueEnv) -> None:
        self.headings = self.rng.integers(0, ACTION_DIM, size=env.num_agents)

    def act(self, env: RescueEnv, valid_mask: np.ndarray) -> np.ndarray:
        # Clockwise scan order per heading: N -> E -> S -> W (indices 0, 2, 1, 3).
        clockwise = [0, 2, 1, 3]
        actions = np.zeros(env.num_agents, dtype=np.int64)
        for i in range(env.num_agents):
            target = self._visible_target(env, i)
            if target is not None:
                actions[i] = self._step_toward(env, i, target, valid_mask)
                self.headings[i] = actions[i]
                continue
            heading = int(self.headings[i])
            if valid_mask[i, heading] and self.rng.random() < 0.85:
                actions[i] = heading
                continue
            start = clockwise.index(heading)
            for offset in range(1, ACTION_DIM + 1):
                candidate = clockwise[(start + offset) % ACTION_DIM]
                if valid_mask[i, candidate]:
                    actions[i] = candidate
                    break
            else:
                actions[i] = heading
            self.headings[i] = actions[i]
        return actions


class EnsembleTeacher(_RescueTeacher):
    """Expert 2 teacher driven by the trained QMIX+TransfQMix+MAPPO ensemble.

    Replaces the heuristic ``CoordinationTeacher``: instead of a hand-written
    dispersal/cohesion rule, the coordination label comes from a genuinely
    trained cooperative policy (``PolicyEnsemble``), so the distilled E2 head
    behaves like the deep MARL agents rather than a copy of E1/E3.

    Following the same visible-target convention as the other teachers, an agent
    that can see a target steps greedily toward it; otherwise it takes the
    ensemble's argmax action. Must run on an ``EntityRescueEnv`` so TransfQMix
    gets entity tokens.
    """

    def __init__(self, rng: np.random.Generator, ensemble) -> None:
        super().__init__(rng)
        self.ensemble = ensemble

    def act(self, env: RescueEnv, valid_mask: np.ndarray) -> np.ndarray:
        flat_obs = env._observations()      # (A, obs_dim) — same tensor the MoE clones on
        tokens = env.entity_obs()           # (A, n_tokens, token_dim) — needs EntityRescueEnv
        probs = self.ensemble.action_probs(flat_obs, tokens).numpy()  # (A, n_actions)

        actions = np.zeros(env.num_agents, dtype=np.int64)
        for i in range(env.num_agents):
            target = self._visible_target(env, i)
            if target is not None:
                actions[i] = self._step_toward(env, i, target, valid_mask)
                continue
            masked = np.where(valid_mask[i], probs[i], -np.inf)
            actions[i] = int(np.argmax(masked))
        return actions


Teacher = Union[ExplorationTeacher, CoordinationTeacher, FallbackTeacher, EnsembleTeacher]


def make_teachers(rng: np.random.Generator, ensemble=None) -> list[Teacher]:
    """The three teachers in expert-head order (exploration, coordination, fallback).

    When ``ensemble`` is provided, Expert 2's heuristic ``CoordinationTeacher``
    is replaced by the trained ``EnsembleTeacher``.
    """
    coordination = (
        EnsembleTeacher(rng, ensemble) if ensemble is not None else CoordinationTeacher(rng)
    )
    return [ExplorationTeacher(rng), coordination, FallbackTeacher(rng)]


def collect_expert_dataset(
    env: RescueEnv,
    teacher: Teacher,
    episodes: int,
    max_steps: int,
) -> list[Episode]:
    """Rolls the teacher on the real grid and records labelled episodes.

    Step order is preserved inside each episode so the recurrent fallback
    head can be trained on true sequences.

    Args:
        env: The rescue environment (a fresh random grid per episode).
        teacher: Heuristic expert producing target action labels.
        episodes: Number of full episodes to collect.
        max_steps: Step cap per collection episode.

    Returns:
        Episodes of (obs [A, obs_dim], peer_matrix [A, A], actions [A]) tuples.
    """
    dataset: list[Episode] = []
    for _ in range(episodes):
        obs = env.reset()  # [A, obs_dim]
        teacher.reset(env)
        episode: Episode = []
        for _ in range(max_steps):
            valid_mask = env.valid_action_mask()
            actions = teacher.act(env, valid_mask)
            peer = build_peer_matrix(env.positions)

            episode.append((
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(peer, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
            ))

            obs, _, done, _ = env.step(actions)
            teacher.observe(env)
            if done:
                break
        dataset.append(episode)
    return dataset


def _stack_transitions(transitions: list[Transition]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacks a transition list into batched tensors [N, A, ...]."""
    obs = torch.stack([t[0] for t in transitions])
    peer = torch.stack([t[1] for t in transitions])
    acts = torch.stack([t[2] for t in transitions])
    return obs, peer, acts


# ===========================================================================
# Stage 1: behavioral cloning of the expert heads
# ===========================================================================

def run_expert_distillation(
    policy: NeuralMoEPolicy,
    datasets: list[list[Episode]],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    on_epoch: Optional[ProgressCallback] = None,
    tbptt_steps: int = 10,
) -> dict[str, float]:
    """Offline behavioral cloning of the three expert heads.

    The feed-forward heads (exploration, coordination) train on shuffled
    mini-batches. The recurrent fallback head trains on *ordered episode
    sequences* with truncated backpropagation through time, carrying its GRU
    hidden state across steps — a memoryless single-step regime could never
    teach it heading persistence. The gating router stays frozen; validation
    accuracy is computed under ``torch.no_grad()``.

    Args:
        policy: The NeuralMoEPolicy to train.
        datasets: [exploration, coordination, fallback] episode datasets.
        epochs: Number of full training epochs.
        batch_size: Mini-batch size for the feed-forward heads.
        lr: Adam learning rate.
        seed: Shuffle seed for reproducible batching.
        on_epoch: Called after each epoch with (epoch, epochs, loss, acc, grad_norm).
        tbptt_steps: Chunk length for truncated backprop through the GRU.

    Returns:
        Final metrics: {"loss": ..., "accuracy": ...}.
    """
    for p in policy.router_encoder.parameters():
        p.requires_grad = False
    for p in policy.router.parameters():
        p.requires_grad = False
    trainable = (
        list(policy.expert_encoder.parameters())
        + list(policy.expert_exploration.parameters())
        + list(policy.expert_coordination.parameters())
        + list(policy.expert_fallback.parameters())
    )
    for p in trainable:
        p.requires_grad = True

    optimizer = optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    # 90/10 train/validation split, per head, at episode granularity
    train_sets: list[list[Episode]] = []
    val_sets: list[list[Episode]] = []
    for dataset in datasets:
        n_val = max(1, len(dataset) // 10)
        train_sets.append(dataset[n_val:])
        val_sets.append(dataset[:n_val])

    # Pre-stacked flat tensors for the feed-forward heads
    ff_train = [
        _stack_transitions([t for ep in train_sets[i] for t in ep]) for i in (0, 1)
    ]

    device = next(policy.parameters()).device

    def encode(obs_b: torch.Tensor, peer_b: torch.Tensor) -> torch.Tensor:
        B, A, obs_dim = obs_b.shape
        assert peer_b.shape == (B, A, A), f"peer batch shape {peer_b.shape}"
        obs_b, peer_b = obs_b.to(device), peer_b.to(device)
        peer_count = torch.sum(peer_b, dim=-1, keepdim=True)
        return policy.expert_encoder(obs_b.view(B * A, obs_dim), peer_count.view(B * A, 1))

    avg_loss, val_acc = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        total_loss, num_batches, grad_norm = 0.0, 0, 0.0

        # -- Feed-forward heads: shuffled mini-batches --------------------
        for head_idx, head in ((0, policy.expert_exploration), (1, policy.expert_coordination)):
            obs, peer, acts = ff_train[head_idx]
            perm = torch.randperm(obs.size(0), generator=generator)
            for start in range(0, obs.size(0), batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                z = encode(obs[idx], peer[idx])
                loss = criterion(head(z), acts[idx].reshape(-1).to(device))
                loss.backward()
                grad_norm = float(
                    torch.norm(
                        torch.stack([
                            p.grad.detach().norm(2)
                            for p in policy.expert_encoder.parameters()
                            if p.grad is not None
                        ])
                    )
                )
                optimizer.step()
                total_loss += float(loss.item())
                num_batches += 1

        # -- Recurrent fallback head: ordered sequences with TBPTT --------
        for episode in train_sets[2]:
            hidden: Optional[torch.Tensor] = None
            for start in range(0, len(episode), tbptt_steps):
                chunk = episode[start:start + tbptt_steps]
                optimizer.zero_grad()
                chunk_loss = torch.zeros((), device=device)
                for obs_s, peer_s, act_s in chunk:
                    z = encode(obs_s.unsqueeze(0), peer_s.unsqueeze(0))  # [A, D]
                    logits, hidden = policy.expert_fallback(z, hidden)
                    chunk_loss = chunk_loss + criterion(logits, act_s.to(device))
                (chunk_loss / len(chunk)).backward()
                optimizer.step()
                hidden = hidden.detach()
                total_loss += float(chunk_loss.item()) / len(chunk)
                num_batches += 1

        # -- Validation accuracy (no autograd graph) ----------------------
        correct, total = 0, 0
        with torch.no_grad():
            for head_idx, head in ((0, policy.expert_exploration), (1, policy.expert_coordination)):
                obs_v, peer_v, act_v = _stack_transitions(
                    [t for ep in val_sets[head_idx] for t in ep]
                )
                preds = torch.argmax(head(encode(obs_v, peer_v)), dim=-1)
                correct += int((preds == act_v.reshape(-1).to(device)).sum().item())
                total += act_v.numel()
            for episode in val_sets[2]:
                hidden = None
                for obs_s, peer_s, act_s in episode:
                    z = encode(obs_s.unsqueeze(0), peer_s.unsqueeze(0))
                    logits, hidden = policy.expert_fallback(z, hidden)
                    correct += int((torch.argmax(logits, dim=-1) == act_s.to(device)).sum().item())
                    total += act_s.numel()

        avg_loss = total_loss / max(num_batches, 1)
        val_acc = 100.0 * correct / max(total, 1)
        if on_epoch is not None:
            on_epoch(epoch, epochs, avg_loss, val_acc, grad_norm)

    return {"loss": avg_loss, "accuracy": val_acc}


# ===========================================================================
# Stage 2: attention-router optimization with blackout penalties
# ===========================================================================

def run_router_optimization(
    policy: NeuralMoEPolicy,
    datasets: list[list[Episode]],
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    on_step: Optional[ProgressCallback] = None,
) -> dict[str, float]:
    """Fine-tunes the attention router with Conditional Indicator Mask penalties.

    Samples real grid states and augments each batch with forced blackout
    (identity peer matrix) and forced full-connectivity copies so the
    penalties always have coverage. Every routing regime is observable, so
    all three experts get supervision:

        isolated (peer_count == 1)          -> push g_fallback -> 1
        linked + target in the ego window   -> push g_coord    -> 1
        linked + nothing to coordinate on   -> push g_explore  -> 1

    Args:
        policy: The NeuralMoEPolicy whose router is fine-tuned.
        datasets: The collected expert datasets (real-state pool).
        steps: Number of optimization steps.
        batch_size: States sampled per step (before augmentation).
        lr: Adam learning rate.
        seed: Sampling seed.
        on_step: Called per step with (step, steps, loss, 100.0, grad_norm).

    Returns:
        Final metrics: {"loss": ...}.
    """
    for p in policy.expert_encoder.parameters():
        p.requires_grad = False
    for p in policy.expert_exploration.parameters():
        p.requires_grad = False
    for p in policy.expert_coordination.parameters():
        p.requires_grad = False
    for p in policy.expert_fallback.parameters():
        p.requires_grad = False
    router_params = list(policy.router_encoder.parameters()) + list(policy.router.parameters())
    for p in router_params:
        p.requires_grad = True

    optimizer = optim.Adam(router_params, lr=lr)
    rng = np.random.default_rng(seed)
    device = next(policy.parameters()).device

    pool = [t for dataset in datasets for episode in dataset for t in episode]
    A = policy.num_agents

    loss_value = 0.0
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(pool), size=batch_size)
        obs = torch.stack([pool[i][0] for i in idx]).to(device)        # [B, A, obs_dim]
        peer_real = torch.stack([pool[i][1] for i in idx]).to(device)  # [B, A, A]

        # Connectivity augmentation: real / forced blackout / forced connected
        peer_blackout = torch.eye(A, device=device).expand(batch_size, A, A)
        peer_connected = torch.ones(batch_size, A, A, device=device)
        obs_all = torch.cat([obs, obs, obs], dim=0)
        peer_all = torch.cat([peer_real, peer_blackout, peer_connected], dim=0)
        action_mask = torch.ones(obs_all.size(0), A, policy.action_dim, dtype=torch.bool, device=device)

        assert obs_all.shape[1:] == (A, obs.shape[-1]), f"obs batch shape {obs_all.shape}"
        assert peer_all.shape[1:] == (A, A), f"peer batch shape {peer_all.shape}"

        optimizer.zero_grad()
        _, weights, _ = policy(obs_all, peer_all, action_mask)

        peer_count = torch.sum(peer_all, dim=-1)  # [3B, A]
        is_isolated = (peer_count == 1.0).float()

        # Target visibility from the ego window's target channels (1 and 2)
        win, win_dim, channels = (
            policy.expert_encoder.win,
            policy.expert_encoder.window_dim,
            policy.expert_encoder.channels,
        )
        window = obs_all[:, :, :win_dim].reshape(-1, A, win, win, channels)
        has_target = (torch.amax(window[..., 1:3], dim=(-3, -2, -1)) > 0.5).float()  # [3B, A]

        coord_regime = has_target * (1.0 - is_isolated)
        explore_regime = (1.0 - has_target) * (1.0 - is_isolated)

        g_explore = weights[:, :, 0]
        g_coord = weights[:, :, 1]
        g_fallback = weights[:, :, 2]

        blackout_penalty = (
            (is_isolated * (1.0 - g_fallback) ** 2).sum() / is_isolated.sum().clamp(min=1.0)
        )
        coord_penalty = (
            (coord_regime * (1.0 - g_coord) ** 2).sum() / coord_regime.sum().clamp(min=1.0)
        )
        explore_penalty = (
            (explore_regime * (1.0 - g_explore) ** 2).sum() / explore_regime.sum().clamp(min=1.0)
        )
        loss = 10.0 * (blackout_penalty + coord_penalty + explore_penalty)

        loss.backward()
        grad_norm = float(
            torch.norm(
                torch.stack([
                    p.grad.detach().norm(2) for p in router_params if p.grad is not None
                ])
            )
        )
        optimizer.step()

        loss_value = float(loss.item())
        if on_step is not None:
            on_step(step, steps, loss_value, 100.0, grad_norm)

    return {"loss": loss_value}


def _build_ensemble_env(env: RescueEnv, checkpoint_dir: str, seed: int):
    """Builds an ``EntityRescueEnv`` mirror of ``env`` and the trained ensemble.

    Returns ``(collect_env, ensemble)`` on success, or ``(env, None)`` if the
    checkpoints are missing or their dimensions don't match ``env`` (so the
    caller transparently falls back to the heuristic coordination teacher).
    """
    from pathlib import Path

    try:
        from rescue_sim.Ensemble.ensemble import PolicyEnsemble
        from rescue_sim.TransfQMix.transf_qmix import EntityRescueEnv

        paths = {n: Path(checkpoint_dir) / f"{n}.pt" for n in ("qmix", "transfqmix", "mappo")}
        if not all(p.exists() for p in paths.values()):
            print(f"[MoE] ensemble checkpoints missing in {checkpoint_dir}/; "
                  "using heuristic CoordinationTeacher")
            return env, None

        collect_env = EntityRescueEnv(
            env.grid_settings,
            num_agents=env.num_agents,
            max_steps=env.max_steps,
            view_radius=env.view_radius,
            seed=seed,
        )
        ensemble = PolicyEnsemble.from_checkpoints(
            collect_env,
            qmix_path=str(paths["qmix"]),
            transf_path=str(paths["transfqmix"]),
            mappo_path=str(paths["mappo"]),
            device="cpu",
        )
        return collect_env, ensemble
    except Exception as exc:  # noqa: BLE001 - any load/dim error -> safe fallback
        print(f"[MoE] ensemble teacher unavailable ({exc}); using heuristic CoordinationTeacher")
        return env, None


def train_moe_policy(
    env: RescueEnv,
    episodes_per_head: int = 6,
    collect_steps: int = 80,
    epochs: int = 20,
    batch_size: int = 64,
    router_steps: int = 120,
    router_batch: int = 16,
    lr: float = 1e-3,
    seed: int = 7,
    on_distill_epoch: Optional[ProgressCallback] = None,
    on_router_step: Optional[ProgressCallback] = None,
    policy: Optional[NeuralMoEPolicy] = None,
    device: Optional[str] = None,
    use_ensemble: bool = True,
    checkpoint_dir: str = "checkpoints",
    e2_gated: bool = False,
) -> NeuralMoEPolicy:
    """End-to-end pipeline: collect teacher trajectories, clone experts, tune router.

    Args:
        env: The rescue environment to train against.
        episodes_per_head: Collection episodes per expert teacher.
        collect_steps: Step cap per collection episode.
        epochs: Behavioral cloning epochs.
        batch_size: BC mini-batch size.
        router_steps: Router optimization steps.
        router_batch: Real states sampled per router step.
        lr: Adam learning rate (both stages).
        seed: Global random seed.
        on_distill_epoch: Progress callback for the BC stage.
        on_router_step: Progress callback for the router stage.
        policy: Existing policy to continue training on fresh data; a new
            one is created when omitted.

    Returns:
        The trained NeuralMoEPolicy.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = resolve_device(device)

    if policy is None:
        policy = NeuralMoEPolicy(env.obs_dim, env.num_agents, env.view_radius, ACTION_DIM)
    else:
        # Continue training on a fresh module tree with copied weights. The
        # passed-in policy may be live (serving rollouts elsewhere); training
        # a copy keeps its autograd state clean and swaps in atomically.
        fresh = NeuralMoEPolicy(env.obs_dim, env.num_agents, env.view_radius, ACTION_DIM)
        fresh.load_state_dict(policy.state_dict())
        policy = fresh
    policy = policy.to(dev)

    # Expert 2's teacher is the trained QMIX+TransfQMix+MAPPO ensemble when its
    # checkpoints load cleanly; otherwise fall back to the heuristic teacher so
    # training never crashes (e.g. missing checkpoints or a dim mismatch).
    collect_env, ensemble = env, None
    if use_ensemble:
        collect_env, ensemble = _build_ensemble_env(env, checkpoint_dir, seed)
    datasets = [
        collect_expert_dataset(collect_env, teacher, episodes_per_head, collect_steps)
        for teacher in make_teachers(rng, ensemble)
    ]
    run_expert_distillation(policy, datasets, epochs, batch_size, lr, seed, on_distill_epoch)

    # Optional: refine Expert 2 with State-Conditioned Gated Teacher Selection
    # (per-teacher calibrated temperatures + reverse-KL). Requires the entity
    # env + loaded ensemble; the encoder is already trained by the BC stage.
    if e2_gated and ensemble is not None and hasattr(collect_env, "entity_obs"):
        from rescue_sim.MoE.gated_distill import train_gated_expert2
        metrics = train_gated_expert2(policy, collect_env, checkpoint_dir=checkpoint_dir, seed=seed)
        print(f"[MoE] gated E2 distillation: acc={metrics['acc']:.1f}% "
              f"weights={metrics['teacher_weights']}")

    run_router_optimization(policy, datasets, router_steps, router_batch, lr, seed, on_router_step)

    # Final stage: retrain the gate on outcome labels (which expert's action
    # the trained teachers rate best per visited state) and sharpen it toward
    # winner-take-all. Rule-based optimization above serves as initialization.
    if e2_gated and ensemble is not None and hasattr(collect_env, "entity_obs"):
        try:
            from rescue_sim.MoE.gated_distill import train_outcome_router

            rm = train_outcome_router(policy, collect_env, checkpoint_dir=checkpoint_dir, seed=seed)
            print(f"[MoE] outcome router: acc={rm['acc']:.1f}% tau={rm['gate_tau']} "
                  f"share={rm['expert_share']} (n={rm['n_states']})")
        except Exception as exc:  # noqa: BLE001 - gate refinement must not kill training
            print(f"[MoE] outcome router skipped ({exc})")
    return policy


# ── Persistence: pretrained MoE checkpoint (checkpoints/moe.pt) ─────────────

def save_moe_policy(
    policy: NeuralMoEPolicy,
    path: str,
    obs_dim: int,
    view_radius: int,
    shape: tuple,
    epochs: int,
) -> None:
    """Save the trained MoE with everything needed to rebuild it at load time.

    ``shape`` is the dashboard compatibility key (grid_w, grid_h, num_agents,
    view_radius); a loaded policy is only reused when the requested config
    matches it exactly.
    """
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "num_agents": policy.num_agents,
            "view_radius": view_radius,
            "action_dim": policy.action_dim,
            "latent_dim": policy.latent_dim,
            "shape": tuple(shape),
            "epochs": int(epochs),
            "gate_tau": float(getattr(policy, "gate_tau", 1.0)),
        },
        target,
    )


def load_moe_policy(path: str) -> tuple[NeuralMoEPolicy, tuple, int] | None:
    """Load a pretrained MoE checkpoint; returns (policy, shape, epochs) or None."""
    from pathlib import Path

    source = Path(path)
    if not source.is_file():
        return None
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    policy = NeuralMoEPolicy(
        ckpt["obs_dim"],
        ckpt["num_agents"],
        ckpt["view_radius"],
        ckpt["action_dim"],
        ckpt["latent_dim"],
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.gate_tau = float(ckpt.get("gate_tau", 1.0))
    policy.eval()
    return policy, tuple(ckpt["shape"]), int(ckpt["epochs"])


def evaluate_moe_policy(
    policy: NeuralMoEPolicy,
    env: RescueEnv,
    episodes: int = 15,
) -> dict:
    """Greedy full-MoE rollouts; returns success_rate / avg_rescued / avg_steps.

    Uses the env's own RNG stream, so seeding the env fixes the grid sequence —
    pass a freshly constructed env for a reproducible validation set.
    """
    policy.eval()
    successes, rescued, steps = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        memory = ExplorationMemory(env.num_agents)
        memory.observe(env.positions)
        hidden = None
        done = False
        info = {"success": False, "rescued": 0, "steps": 0}
        while not done:
            valid_mask = env.valid_action_mask()
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            peer_t = torch.tensor(
                build_peer_matrix(env.positions), dtype=torch.float32
            ).unsqueeze(0)
            mask_t = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                y_final, _weights, hidden = policy(obs_t, peer_t, mask_t, hidden)
            scores = memory.bias_logits(env, obs, y_final.squeeze(0), valid_mask)
            actions = torch.argmax(scores, dim=-1)
            obs, _reward, done, info = env.step(actions.numpy())
            memory.observe(env.positions)
        successes.append(bool(info["success"]))
        rescued.append(int(info["rescued"]))
        steps.append(int(info["steps"]))
    n = max(len(successes), 1)
    return {
        "success_rate": sum(successes) / n,
        "avg_rescued": sum(rescued) / n,
        "avg_steps": sum(steps) / n,
    }
