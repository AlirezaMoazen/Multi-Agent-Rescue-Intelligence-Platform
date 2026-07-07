# Copyright 2026 Alireza Moazen (alirezamoazen.com)
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
    """Expert 1 (non-AI explorer): rescue what you see, otherwise spread out.

    Moving away from visible teammates maximizes joint sensor coverage, and
    the teammate positions are in the observation, so the rule is learnable.
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
                # Disperse: pick the valid move that gains the most distance
                best_gain, best_actions = -(10 ** 9), []
                for a, (dx, dy) in enumerate(ACTION_DELTAS):
                    if not valid_mask[i, a]:
                        continue
                    gain = sum(
                        abs(t.x - (pos.x + dx)) + abs(t.y - (pos.y + dy))
                        for t in teammates
                    )
                    if gain > best_gain:
                        best_gain, best_actions = gain, [a]
                    elif gain == best_gain:
                        best_actions.append(a)
                actions[i] = (
                    int(self.rng.choice(best_actions)) if best_actions
                    else self._toward_open_space(env, i, valid_mask)
                )
            else:
                actions[i] = self._toward_open_space(env, i, valid_mask)
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
    """Expert 3 (local hysteretic Q style): a competent lone wolf.

    Rescues any target it sees; otherwise keeps its heading and rotates
    clockwise on walls (wall-following). The persistence is temporal state,
    which is exactly what the GRU fallback head can represent — so an
    isolated agent sweeps the map instead of blind looping.
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


Teacher = Union[ExplorationTeacher, CoordinationTeacher, FallbackTeacher]


def make_teachers(rng: np.random.Generator) -> list[Teacher]:
    """The three teachers in expert-head order (exploration, coordination, fallback)."""
    return [ExplorationTeacher(rng), CoordinationTeacher(rng), FallbackTeacher(rng)]


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
    datasets = [
        collect_expert_dataset(env, teacher, episodes_per_head, collect_steps)
        for teacher in make_teachers(rng)
    ]
    run_expert_distillation(policy, datasets, epochs, batch_size, lr, seed, on_distill_epoch)
    run_router_optimization(policy, datasets, router_steps, router_batch, lr, seed, on_router_step)
    return policy
