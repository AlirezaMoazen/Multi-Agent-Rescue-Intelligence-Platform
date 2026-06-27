"""Tests for rescue_sim.Qlearning.communications.
#Cristina Marcos Alonso

EpidemicHystereticQLearning lives in Alireza's branch and will be merged
later. These tests inject a lightweight stub so the comms layer can be
validated independently of that module.
"""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Stub out rescue_sim.Qlearning.q_learning before importing communications
# ---------------------------------------------------------------------------


@dataclass
class GossipMessage:
    """Minimal replica of the real GossipMessage wire format."""

    sender: int
    indices: np.ndarray
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.indices.size)


_fake_mod = ModuleType("rescue_sim.Qlearning.q_learning")
_fake_mod.GossipMessage = GossipMessage          # type: ignore[attr-defined]
_fake_mod.EpidemicHystereticQLearning = MagicMock  # type: ignore[attr-defined]

# Only stub the missing q_learning module; let Python find the real package.
sys.modules["rescue_sim.Qlearning.q_learning"] = _fake_mod

from rescue_sim.Qlearning.communications import (  # noqa: E402
    CommunicationStats,
    DefaultCommsBus,
    ResilientCommsBus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fleet(
    pairs: list[tuple[str, str, float]] | None = None,
    can_sync: bool = True,
    gossip_return: int = 1,
    n_entries: int = 5,
) -> MagicMock:
    """Build a mock EpidemicHystereticQLearning with sensible defaults."""
    fleet = MagicMock()
    fleet.gossip.return_value = gossip_return
    fleet.neighbors.return_value = pairs if pairs is not None else [("r0", "r1", 1.5)]
    fleet.can_sync.return_value = can_sync
    fleet.step_count = 10
    fleet._id_to_slot = {"r0": 0, "r1": 1}
    fleet.last_sync = np.full((4, 4), -(10**9), dtype=np.int64)
    fleet.active = np.ones(4, dtype=bool)

    def _export(agent_id: str) -> GossipMessage:
        rng = np.random.default_rng(abs(hash(agent_id)) % (2**31))
        return GossipMessage(
            sender=fleet._id_to_slot[agent_id],
            indices=np.arange(n_entries, dtype=int),
            values=rng.random(n_entries),
        )

    fleet.export_delta.side_effect = _export
    fleet.import_delta.return_value = 2
    return fleet


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------


class TestCommunicationStats(unittest.TestCase):

    def test_drop_rate_zero_when_no_attempts(self) -> None:
        stats = CommunicationStats()
        self.assertEqual(stats.drop_rate, 0.0)

    def test_drop_rate_correct(self) -> None:
        stats = CommunicationStats(syncs_performed=3, syncs_dropped=1)
        self.assertTrue(math.isclose(stats.drop_rate, 0.25))

    def test_reset_clears_all_fields(self) -> None:
        stats = CommunicationStats(
            syncs_performed=5,
            syncs_dropped=2,
            entries_sent=100,
            entries_improved=40,
            delayed_messages=3,
        )
        stats.reset()
        self.assertEqual(stats.syncs_performed, 0)
        self.assertEqual(stats.syncs_dropped, 0)
        self.assertEqual(stats.entries_sent, 0)
        self.assertEqual(stats.entries_improved, 0)
        self.assertEqual(stats.delayed_messages, 0)


class TestDefaultCommsBus(unittest.TestCase):

    def test_delegates_to_gossip(self) -> None:
        fleet = _make_fleet(gossip_return=3)
        bus = DefaultCommsBus()
        result = bus.exchange(fleet)
        fleet.gossip.assert_called_once()
        self.assertEqual(result, 3)

    def test_returns_zero_when_gossip_returns_zero(self) -> None:
        fleet = _make_fleet(gossip_return=0)
        bus = DefaultCommsBus()
        self.assertEqual(bus.exchange(fleet), 0)


class TestResilientCommsBusValidation(unittest.TestCase):

    def test_drop_prob_equal_to_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResilientCommsBus(drop_prob=1.0)

    def test_negative_drop_prob_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResilientCommsBus(drop_prob=-0.1)

    def test_zero_max_entries_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResilientCommsBus(max_entries_per_message=0)

    def test_negative_delay_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResilientCommsBus(delay_steps=-1)

    def test_default_params_accepted(self) -> None:
        bus = ResilientCommsBus()
        self.assertEqual(bus.drop_prob, 0.0)
        self.assertIsNone(bus.max_entries)
        self.assertEqual(bus.delay_steps, 0)


class TestResilientCommsBusPerfectChannel(unittest.TestCase):

    def test_no_impairments_syncs_pair(self) -> None:
        fleet = _make_fleet()
        bus = ResilientCommsBus(seed=0)
        synced = bus.exchange(fleet)
        self.assertEqual(synced, 1)
        self.assertEqual(bus.stats.syncs_performed, 1)
        self.assertEqual(bus.stats.syncs_dropped, 0)

    def test_skips_pair_on_cooldown(self) -> None:
        fleet = _make_fleet(can_sync=False)
        bus = ResilientCommsBus(seed=0)
        synced = bus.exchange(fleet)
        self.assertEqual(synced, 0)
        self.assertEqual(bus.stats.syncs_performed, 0)

    def test_no_neighbors_means_zero_syncs(self) -> None:
        fleet = _make_fleet(pairs=[])
        bus = ResilientCommsBus(seed=0)
        self.assertEqual(bus.exchange(fleet), 0)


class TestResilientCommsBusPacketLoss(unittest.TestCase):

    def test_high_drop_prob_eventually_drops(self) -> None:
        bus = ResilientCommsBus(drop_prob=0.9999, seed=42)
        for _ in range(20):
            bus.exchange(_make_fleet())
        self.assertGreater(bus.stats.syncs_dropped, 0)

    def test_zero_drop_prob_never_drops(self) -> None:
        bus = ResilientCommsBus(drop_prob=0.0, seed=0)
        for _ in range(5):
            bus.exchange(_make_fleet())
        self.assertEqual(bus.stats.syncs_dropped, 0)
        self.assertEqual(bus.stats.syncs_performed, 5)


class TestResilientCommsBusBandwidthCap(unittest.TestCase):

    def test_trim_keeps_all_when_under_limit(self) -> None:
        bus = ResilientCommsBus(max_entries_per_message=10)
        msg = GossipMessage(sender=0, indices=np.arange(5, dtype=int), values=np.ones(5))
        self.assertEqual(bus._trim(msg).size, 5)

    def test_trim_truncates_to_max_entries(self) -> None:
        bus = ResilientCommsBus(max_entries_per_message=3)
        msg = GossipMessage(
            sender=0, indices=np.arange(10, dtype=int), values=np.arange(10, dtype=float)
        )
        self.assertEqual(bus._trim(msg).size, 3)

    def test_trim_keeps_highest_abs_values(self) -> None:
        bus = ResilientCommsBus(max_entries_per_message=2)
        values = np.array([8.0, 1.0, 2.0, 3.0, 9.0])
        msg = GossipMessage(sender=0, indices=np.arange(5, dtype=int), values=values)
        trimmed = bus._trim(msg)
        self.assertEqual(set(trimmed.indices.tolist()), {0, 4})

    def test_trim_no_limit_returns_unchanged(self) -> None:
        bus = ResilientCommsBus()
        msg = GossipMessage(sender=0, indices=np.arange(50, dtype=int), values=np.ones(50))
        self.assertEqual(bus._trim(msg).size, 50)

    def test_entries_sent_respects_cap(self) -> None:
        fleet = _make_fleet(n_entries=10)
        bus = ResilientCommsBus(max_entries_per_message=3, seed=0)
        bus.exchange(fleet)
        # Both directions: 3 entries each = 6
        self.assertEqual(bus.stats.entries_sent, 6)


class TestResilientCommsBusDelay(unittest.TestCase):

    def test_delay_zero_delivers_immediately(self) -> None:
        fleet = _make_fleet()
        bus = ResilientCommsBus(delay_steps=0, seed=0)
        bus.exchange(fleet)
        # Both directions delivered at once
        self.assertEqual(fleet.import_delta.call_count, 2)

    def test_delay_buffers_then_delivers(self) -> None:
        fleet = _make_fleet()
        bus = ResilientCommsBus(delay_steps=2, seed=0)

        # Step 0: messages buffered, not yet delivered
        fleet.neighbors.return_value = [("r0", "r1", 1.5)]
        bus.exchange(fleet)
        self.assertGreater(bus.stats.delayed_messages, 0)
        initial_calls = fleet.import_delta.call_count

        # Steps 1-2: same fleet, no new pairs — only buffered msgs delivered
        fleet.neighbors.return_value = []
        bus.exchange(fleet)
        bus.exchange(fleet)

        # After delay expires, import_delta should have been called
        self.assertGreater(fleet.import_delta.call_count, initial_calls)

    def test_entries_sent_accumulates(self) -> None:
        fleet = _make_fleet(n_entries=5)
        bus = ResilientCommsBus(seed=0)
        bus.exchange(fleet)
        # 5 entries × 2 directions = 10
        self.assertEqual(bus.stats.entries_sent, 10)


if __name__ == "__main__":
    unittest.main()
