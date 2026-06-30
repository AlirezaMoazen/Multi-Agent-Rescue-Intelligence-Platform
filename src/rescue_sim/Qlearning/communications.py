"""Peer-to-peer communication layer for the decentralized rescue fleet.
#Cristina Marcos Alonso

IMPLEMENTATION NOTES:
Two classes are provided so the team can compare them in the report:

  DefaultCommsBus   — wraps fleet.gossip() directly. Every nearby pair that
                      is off cooldown exchanges their full dirty Q-table delta.
                      This is the baseline: perfect, instant, unlimited channel.

  ResilientCommsBus — a more realistic channel on top of the same gossip
                      mechanism. Three optional impairments can be enabled
                      independently or together:

                      1. Probabilistic packet loss (drop_prob):
                         Each candidate sync is dropped with probability
                         drop_prob before it happens, simulating an unreliable
                         wireless link. The pair stays off cooldown so they may
                         retry on the next step they are close.

                      2. Bandwidth cap (max_entries_per_message):
                         Instead of sync_pair (which sends everything dirty),
                         uses export_delta + import_delta and keeps only the
                         top-N entries by |Q-value|. Models a link that can
                         only carry N floats per step.

                      3. Transmission delay (delay_steps):
                         Messages are buffered and only delivered delay_steps
                         steps later. Simulates propagation latency or a
                         store-and-forward network.

Both classes expose the same single method:

    exchange(fleet) -> int
        Called once per simulation step, after all robots have moved and
        learned. Returns the number of pairwise syncs that actually happened.

Comparison idea for the report
-------------------------------
Run the same multi-robot scenario three times:
  a) DefaultCommsBus                      — perfect channel
  b) ResilientCommsBus(drop_prob=0.3)     — 30 % packet loss
  c) ResilientCommsBus(max_entries=20)    — narrow-bandwidth channel
  d) ResilientCommsBus(delay_steps=3)     — 3-step latency

Compare convergence speed (steps to first rescue), total reward, and
CommunicationStats.entries_sent to quantify the bandwidth cost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning, GossipMessage

__all__ = [
    "DefaultCommsBus",
    "ResilientCommsBus",
    "CommunicationStats",
    "GossipMessage",
    "EpidemicHystereticQLearning",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class CommunicationStats:
    """Cumulative statistics collected by ResilientCommsBus.

    These are useful for the project report: they let you quantify the cost
    and effectiveness of the communication channel.
    """

    syncs_performed: int = 0
    """Number of pairwise Q-table merges that actually completed."""

    syncs_dropped: int = 0
    """Candidate syncs that were discarded (packet loss or budget)."""

    entries_sent: int = 0
    """Total Q-value entries transmitted across all messages (bandwidth proxy)."""

    entries_improved: int = 0
    """Total Q-value entries that improved a receiver's table (learning gain)."""

    delayed_messages: int = 0
    """Messages currently waiting in the delay buffer (in-flight)."""

    @property
    def drop_rate(self) -> float:
        """Fraction of candidate syncs that were dropped (0.0 – 1.0)."""
        total = self.syncs_performed + self.syncs_dropped
        return self.syncs_dropped / total if total > 0 else 0.0

    def reset(self) -> None:
        """Zero all counters (call between episodes if needed)."""
        self.syncs_performed = 0
        self.syncs_dropped = 0
        self.entries_sent = 0
        self.entries_improved = 0
        self.delayed_messages = 0


# ---------------------------------------------------------------------------
# DefaultCommsBus — perfect channel, baseline
# ---------------------------------------------------------------------------

class DefaultCommsBus:
    """Minimal communication bus: delegate entirely to fleet.gossip().

    This is the perfect-channel baseline. Every pair of nearby robots that is
    off cooldown exchanges their full dirty Q-table delta via the built-in
    epidemic round. No drops, no bandwidth limit, no delay.

    Use this as the performance ceiling when comparing against ResilientCommsBus.
    """

    def exchange(self, fleet: EpidemicHystereticQLearning) -> int:
        """Run one epidemic gossip round; return the number of synced pairs."""
        return fleet.gossip()


# ---------------------------------------------------------------------------
# ResilientCommsBus — realistic impaired channel
# ---------------------------------------------------------------------------

class ResilientCommsBus:
    """Realistic communication bus with configurable channel impairments.

    Parameters
    ----------
    drop_prob : float
        Probability [0, 1) that a candidate sync is dropped entirely.
        0.0 = perfect delivery (no drops).  0.3 = 30 % packet loss.
    max_entries_per_message : int | None
        If set, each GossipMessage is truncated to the top-N entries by
        absolute Q-value before delivery.  None = unlimited (full delta).
    delay_steps : int
        Number of simulation steps a message waits before being delivered.
        0 = instant delivery (no delay).
    seed : int | None
        RNG seed for reproducible drop/delay behaviour.
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        max_entries_per_message: int | None = None,
        delay_steps: int = 0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        if max_entries_per_message is not None and max_entries_per_message < 1:
            raise ValueError("max_entries_per_message must be >= 1")
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")

        self.drop_prob = drop_prob
        self.max_entries = max_entries_per_message
        self.delay_steps = delay_steps
        self._rng = random.Random(seed)
        self.stats = CommunicationStats()

        # Delay buffer: list of (deliver_at_step, recipient_id, GossipMessage)
        self._buffer: list[tuple[int, str, GossipMessage]] = []
        self._step: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def exchange(self, fleet: EpidemicHystereticQLearning) -> int:
        """Run one impaired communication round; return pairs that synced.

        Step sequence
        -------------
        1. Deliver any buffered messages whose delay has expired.
        2. Find nearby pairs via fleet.neighbors().
        3. For each candidate pair:
           a. Skip if on cooldown (fleet.can_sync() is False).
           b. Drop with probability drop_prob (packet loss).
           c. Export the delta from each side, truncate to max_entries.
           d. Buffer or deliver immediately depending on delay_steps.
        """
        synced = 0
        synced += self._deliver_buffered(fleet)
        synced += self._process_candidates(fleet)
        self._step += 1
        return synced

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deliver_buffered(self, fleet: EpidemicHystereticQLearning) -> int:
        """Deliver messages whose delay has expired; return pairs completed."""
        if not self._buffer:
            return 0

        ready: list[tuple[int, str, GossipMessage]] = []
        waiting: list[tuple[int, str, GossipMessage]] = []
        for deliver_at, recipient, msg in self._buffer:
            if self._step >= deliver_at:
                ready.append((deliver_at, recipient, msg))
            else:
                waiting.append((deliver_at, recipient, msg))
        self._buffer = waiting

        completed = 0
        for _, recipient, msg in ready:
            # The agent might have been removed mid-episode — skip gracefully.
            if recipient not in fleet._id_to_slot:
                continue
            if not fleet.active[fleet._id_to_slot[recipient]]:
                continue
            improved = fleet.import_delta(recipient, msg)
            self.stats.entries_improved += improved
            completed += 1

        self.stats.delayed_messages = len(self._buffer)
        return completed

    def _process_candidates(self, fleet: EpidemicHystereticQLearning) -> int:
        """Evaluate all nearby pairs and decide whether / what to send."""
        pairs = fleet.neighbors()  # [(id_a, id_b, distance), ...]
        synced = 0

        for id_a, id_b, _ in pairs:
            # Cooldown check — the fleet tracks this per pair.
            if not fleet.can_sync(id_a, id_b):
                continue

            # Packet loss: drop this link entirely with probability drop_prob.
            if self.drop_prob > 0.0 and self._rng.random() < self.drop_prob:
                self.stats.syncs_dropped += 1
                continue

            # Export bandwidth-limited deltas from both sides.
            msg_a = self._trim(fleet.export_delta(id_a))
            msg_b = self._trim(fleet.export_delta(id_b))

            self.stats.entries_sent += msg_a.size + msg_b.size

            if self.delay_steps > 0:
                # Buffer both messages for delayed delivery.
                deliver_at = self._step + self.delay_steps
                self._buffer.append((deliver_at, id_b, msg_a))  # a → b
                self._buffer.append((deliver_at, id_a, msg_b))  # b → a
                self.stats.delayed_messages = len(self._buffer)
            else:
                # Instant delivery — apply now.
                self.stats.entries_improved += fleet.import_delta(id_b, msg_a)
                self.stats.entries_improved += fleet.import_delta(id_a, msg_b)

            # Record the sync time so the cooldown is respected.
            slot_a = fleet._id_to_slot[id_a]
            slot_b = fleet._id_to_slot[id_b]
            fleet.last_sync[slot_a, slot_b] = fleet.step_count
            fleet.last_sync[slot_b, slot_a] = fleet.step_count

            self.stats.syncs_performed += 1
            synced += 1

        return synced

    def _trim(self, msg: GossipMessage) -> GossipMessage:
        """Truncate a GossipMessage to the top-N entries by |Q-value|.

        When max_entries is None the message is returned unchanged.
        When max_entries is set, only the entries with the highest absolute
        value are kept — these carry the most information about discovered
        high-reward regions and are worth the bandwidth cost.
        """
        if self.max_entries is None or msg.size <= self.max_entries:
            return msg

        # Rank entries by |Q-value| descending and keep the top-N.
        order = np.argsort(-np.abs(msg.values))[: self.max_entries]
        return GossipMessage(
            sender=msg.sender,
            indices=msg.indices[order],
            values=msg.values[order],
        )
