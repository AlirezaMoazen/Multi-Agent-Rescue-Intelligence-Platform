"""Shared building blocks for the deep multi-agent RL methods.

MAPPO, QMIX, and TransfQMix all need the same handful of helpers (weight init,
target-network sync, value normalization, a replay buffer).  Keeping them in
one place means each algorithm file stays short and tells one clear story.

Requires `torch` (these are only imported by the deep-RL packages).
"""

from __future__ import annotations

from collections import deque
from random import Random

import numpy as np
import torch
from torch import nn


def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """Orthogonal weights + zero bias -- a standard PPO/DQN stability trick."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy every weight from `source` into `target` (target-network sync)."""
    target.load_state_dict(source.state_dict())


class RunningMeanStd:
    """Running mean/variance used to normalize value targets (MAPPO trick #1).

    Uses Welford-style batched updates so the statistics stay stable as training
    rewards drift over time.
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        batch_mean = float(x.mean())
        batch_var = float(x.var(unbiased=False))
        batch_count = x.numel()
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)


class ReplayBuffer:
    """Fixed-size buffer of single-step team transitions for off-policy methods.

    Each transition is a plain dict of NumPy arrays; ``sample`` stacks a random
    batch into torch tensors with the right dtype per field.  Shared by QMIX and
    TransfQMix.
    """

    # Field name -> torch dtype the batched tensor should have.
    _DTYPES = {
        "obs": torch.float32, "state": torch.float32, "actions": torch.long,
        "avail": torch.bool, "reward": torch.float32, "next_obs": torch.float32,
        "next_state": torch.float32, "next_avail": torch.bool, "done": torch.float32,
    }

    def __init__(self, capacity: int, rng: Random) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = rng

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: dict) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict:
        batch = self.rng.sample(self.buffer, batch_size)
        return {
            key: torch.as_tensor(np.stack([t[key] for t in batch])).to(dtype)
            for key, dtype in self._DTYPES.items()
        }
