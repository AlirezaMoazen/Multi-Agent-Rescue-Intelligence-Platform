"""Tests for EpidemicHystereticQLearning and the communications layer."""

from __future__ import annotations

import numpy as np
import pytest

from rescue_sim.environment.grid import Grid, Position
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning, GossipMessage
from rescue_sim.shared import GossipConfig, HystereticConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_grid(width: int = 6, height: int = 6) -> Grid:
    return Grid(
        width=width,
        height=height,
        obstacles=frozenset({Position(2, 2)}),
        target_a_positions=frozenset({Position(5, 5)}),
        target_b_positions=frozenset({Position(0, 5)}),
    )


def make_fleet(grid: Grid | None = None, *, num_agents: int = 2, seed: int = 0) -> EpidemicHystereticQLearning:
    g = grid or make_grid()
    cfg = HystereticConfig(alpha=0.5, beta=0.1, discount_factor=0.0)
    gossip = GossipConfig(comm_radius=3.0, cooldown=1, max_links_per_step=4, utility_threshold=0.0)
    fleet = EpidemicHystereticQLearning(g, config=cfg, gossip=gossip, max_agents=20, seed=seed)
    for i in range(num_agents):
        fleet.add_agent(f"r{i}", Position(i, 0))
    return fleet


# ---------------------------------------------------------------------------
# 1. Construction and membership
# ---------------------------------------------------------------------------

def test_add_agent_increases_fleet_size():
    fleet = make_fleet(num_agents=0)
    assert fleet.fleet_size() == 0
    fleet.add_agent("r0", Position(0, 0))
    assert fleet.fleet_size() == 1


def test_remove_agent_decreases_fleet_size():
    fleet = make_fleet(num_agents=2)
    fleet.remove_agent("r0")
    assert fleet.fleet_size() == 1


def test_removed_agent_retains_q_table():
    fleet = make_fleet(num_agents=1)
    slot = fleet._id_to_slot["r0"]
    fleet.q[slot, 0, 0, 0] = 7.0
    fleet.remove_agent("r0")
    assert fleet.q[slot, 0, 0, 0] == pytest.approx(7.0)


def test_rejoin_after_remove_restores_active():
    fleet = make_fleet(num_agents=1)
    fleet.remove_agent("r0")
    fleet.add_agent("r0", Position(1, 0))
    assert fleet.fleet_size() == 1


def test_forget_agent_wipes_q_table():
    fleet = make_fleet(num_agents=1)
    slot = fleet._id_to_slot["r0"]
    fleet.q[slot, 0, 0, 0] = 9.0
    fleet.forget_agent("r0")
    assert fleet.q[slot, 0, 0, 0] == pytest.approx(0.0)


def test_max_agents_capacity():
    wide = Grid(width=25, height=1, obstacles=frozenset(),
                target_a_positions=frozenset(), target_b_positions=frozenset())
    fleet = make_fleet(wide, num_agents=0)
    for i in range(20):
        fleet.add_agent(f"r{i}", Position(i, 0))
    with pytest.raises(RuntimeError, match="capacity"):
        fleet.add_agent("overflow", Position(20, 0))


def test_invalid_start_position_raises():
    fleet = make_fleet(num_agents=0)
    with pytest.raises(ValueError, match="outside the grid"):
        fleet.add_agent("r0", Position(99, 99))


def test_alpha_lt_beta_raises():
    with pytest.raises(ValueError, match="beta"):
        EpidemicHystereticQLearning(make_grid(), config=HystereticConfig(alpha=0.1, beta=0.9))


# ---------------------------------------------------------------------------
# 2. Hysteretic update — the core algorithmic property
# ---------------------------------------------------------------------------

def test_positive_td_uses_alpha():
    """Positive TD error: Q += alpha * delta."""
    fleet = make_fleet(num_agents=1)
    # gamma=0 so target = reward only; Q starts at 0 → delta = reward - 0 = reward
    # Q_new = 0 + alpha * reward = 0.5 * 10.0 = 5.0
    fleet.record_transitions(
        {"r0": 2},          # East
        {"r0": 10.0},
        {"r0": Position(1, 0)},
    )
    assert fleet.q_table("r0")[0, 0, 2] == pytest.approx(5.0)


def test_negative_td_uses_beta():
    """Negative TD error: Q += beta * delta (beta << alpha)."""
    fleet = make_fleet(num_agents=1)
    slot = fleet._id_to_slot["r0"]
    fleet.q[slot, 0, 0, 2] = 5.0   # prime with 5.0
    fleet.pos[slot] = [0, 0]        # reset position
    # reward=0, gamma=0 → delta = 0 - 5.0 = -5.0; Q_new = 5.0 + 0.1*(-5.0) = 4.5
    fleet.record_transitions({"r0": 2}, {"r0": 0.0}, {"r0": Position(1, 0)})
    assert fleet.q_table("r0")[0, 0, 2] == pytest.approx(4.5)


def test_hysteretic_asymmetry(make_fleet=make_fleet):
    """alpha > beta means Q shrinks slower than it grows for same |delta|."""
    fleet_a = make_fleet(num_agents=1)
    fleet_b = make_fleet(num_agents=1)

    fleet_a.record_transitions({"r0": 2}, {"r0": 5.0}, {"r0": Position(1, 0)})
    q_after_positive = float(fleet_a.q_table("r0")[0, 0, 2])

    fleet_b.record_transitions({"r0": 2}, {"r0": -5.0}, {"r0": Position(1, 0)})
    q_after_negative = float(fleet_b.q_table("r0")[0, 0, 2])

    assert abs(q_after_positive) > abs(q_after_negative)


def test_terminal_state_ignores_next_q():
    """When done=True the future value must be zeroed so the target is just r."""
    fleet = make_fleet(num_agents=1)
    slot = fleet._id_to_slot["r0"]
    fleet.q[slot, 1, 0, :] = 100.0  # large future Q at next cell
    # gamma=0 so future is already 0 — but with done=True it should also be 0
    cfg = HystereticConfig(alpha=1.0, beta=1.0, discount_factor=0.9)
    g = make_grid()
    f = EpidemicHystereticQLearning(g, config=cfg, gossip=GossipConfig(), max_agents=20, seed=0)
    f.add_agent("r0", Position(0, 0))
    s = f._id_to_slot["r0"]
    f.q[s, 1, 0, :] = 100.0
    f.record_transitions({"r0": 0}, {"r0": 5.0}, {"r0": Position(0, 1)}, dones={"r0": True})
    # target = 5.0 + 0.9*0 = 5.0; Q = 0 + 1.0*(5.0) = 5.0  (not 5+90)
    assert f.q_table("r0")[0, 0, 0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 3. Dirty mask / bandwidth minimization
# ---------------------------------------------------------------------------

def test_dirty_set_after_update():
    fleet = make_fleet(num_agents=1)
    fleet.record_transitions({"r0": 2}, {"r0": 1.0}, {"r0": Position(1, 0)})
    assert fleet.dirty[fleet._id_to_slot["r0"], 0, 0, 2]


def test_export_delta_carries_dirty_entries_only():
    fleet = make_fleet(num_agents=1)
    fleet.record_transitions({"r0": 2}, {"r0": 1.0}, {"r0": Position(1, 0)})
    delta = fleet.export_delta("r0")
    assert delta.size == 1


def test_export_clears_dirty():
    fleet = make_fleet(num_agents=1)
    fleet.record_transitions({"r0": 2}, {"r0": 1.0}, {"r0": Position(1, 0)})
    fleet.export_delta("r0")
    assert not fleet.dirty[fleet._id_to_slot["r0"], 0, 0, 2]


def test_utility_threshold_filters_low_values():
    g = make_grid()
    cfg = HystereticConfig(alpha=0.01, beta=0.005, discount_factor=0.0)
    gossip = GossipConfig(utility_threshold=5.0, comm_radius=3.0, cooldown=1)
    fleet = EpidemicHystereticQLearning(g, config=cfg, gossip=gossip, max_agents=20)
    fleet.add_agent("r0", Position(0, 0))
    fleet.record_transitions({"r0": 2}, {"r0": 1.0}, {"r0": Position(1, 0)})
    # Q value after update is small (0.01*1.0=0.01), below threshold 5.0
    delta = fleet.export_delta("r0")
    assert delta.size == 0


# ---------------------------------------------------------------------------
# 4. Epidemic max-sync — the cooperative property
# ---------------------------------------------------------------------------

def test_sync_pair_takes_max():
    """After sync, the lower-value agent adopts the peer's higher value."""
    fleet = make_fleet(num_agents=2)
    sa, sb = fleet._id_to_slot["r0"], fleet._id_to_slot["r1"]
    fleet.q[sa, 0, 0, 2] = 1.0
    fleet.q[sb, 0, 0, 2] = 99.0
    fleet.dirty[sb, 0, 0, 2] = True
    fleet.sync_pair("r0", "r1")
    assert fleet.q_table("r0")[0, 0, 2] == pytest.approx(99.0)


def test_sync_does_not_lower_better_value():
    """Existing high value must not be overwritten by a peer's lower value."""
    fleet = make_fleet(num_agents=2)
    sa, sb = fleet._id_to_slot["r0"], fleet._id_to_slot["r1"]
    fleet.q[sa, 0, 0, 2] = 50.0
    fleet.q[sb, 0, 0, 2] = 1.0
    fleet.dirty[sb, 0, 0, 2] = True
    fleet.sync_pair("r0", "r1")
    assert fleet.q_table("r0")[0, 0, 2] == pytest.approx(50.0)


def test_import_marks_improved_entries_dirty():
    """Improved entries must be re-dirtied so knowledge keeps spreading."""
    fleet = make_fleet(num_agents=2)
    sa, sb = fleet._id_to_slot["r0"], fleet._id_to_slot["r1"]
    fleet.q[sb, 0, 0, 2] = 99.0
    fleet.dirty[sb, 0, 0, 2] = True
    msg = fleet.export_delta("r1")
    fleet.import_delta("r0", msg)
    assert fleet.dirty[sa, 0, 0, 2]


def test_gossip_returns_sync_count():
    g = make_grid()
    cfg = HystereticConfig()
    gossip = GossipConfig(comm_radius=10.0, cooldown=0, max_links_per_step=10)
    fleet = EpidemicHystereticQLearning(g, config=cfg, gossip=gossip, max_agents=20, seed=0)
    fleet.add_agent("r0", Position(0, 0))
    fleet.add_agent("r1", Position(1, 0))
    syncs = fleet.gossip()
    assert syncs == 1


def test_gossip_respects_comm_radius():
    """Agents outside comm_radius must not sync."""
    g = make_grid()
    gossip = GossipConfig(comm_radius=1.0, cooldown=0, max_links_per_step=10)
    fleet = EpidemicHystereticQLearning(g, config=HystereticConfig(), gossip=gossip, max_agents=20, seed=0)
    fleet.add_agent("r0", Position(0, 0))
    fleet.add_agent("r1", Position(5, 5))  # far away
    assert fleet.gossip() == 0


def test_gossip_respects_cooldown():
    g = make_grid()
    gossip = GossipConfig(comm_radius=10.0, cooldown=100, max_links_per_step=10)
    fleet = EpidemicHystereticQLearning(g, config=HystereticConfig(), gossip=gossip, max_agents=20, seed=0)
    fleet.add_agent("r0", Position(0, 0))
    fleet.add_agent("r1", Position(1, 0))
    fleet.gossip()           # first sync (step 0)
    fleet.step_count += 1
    assert fleet.gossip() == 0   # still in cooldown


def test_gossip_link_budget():
    """With max_links_per_step=1 only one sync per robot per step."""
    g = make_grid()
    gossip = GossipConfig(comm_radius=10.0, cooldown=0, max_links_per_step=1)
    fleet = EpidemicHystereticQLearning(g, config=HystereticConfig(), gossip=gossip, max_agents=20, seed=0)
    for i in range(4):
        fleet.add_agent(f"r{i}", Position(i, 0))
    syncs = fleet.gossip()
    assert syncs <= 2   # 4 robots, budget=1 each → ≤ 2 pairs can shake hands


# ---------------------------------------------------------------------------
# 5. Dynamic membership during a run
# ---------------------------------------------------------------------------

def test_remove_and_rejoin_mid_run():
    fleet = make_fleet(num_agents=2)
    fleet.remove_agent("r0")
    assert fleet.fleet_size() == 1
    fleet.add_agent("r0", Position(0, 0))
    assert fleet.fleet_size() == 2


def test_multiple_add_same_id_does_not_duplicate():
    fleet = make_fleet(num_agents=1)
    fleet.add_agent("r0", Position(1, 0))   # re-add same id
    assert fleet.fleet_size() == 1


def test_fresh_flag_wipes_q_table():
    fleet = make_fleet(num_agents=1)
    slot = fleet._id_to_slot["r0"]
    fleet.q[slot, 0, 0, 2] = 42.0
    fleet.add_agent("r0", Position(0, 0), fresh=True)
    assert fleet.q_table("r0")[0, 0, 2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Action selection
# ---------------------------------------------------------------------------

def test_greedy_action_selects_highest_q():
    fleet = make_fleet(num_agents=1)
    fleet.epsilon = 0.0   # pure greedy
    slot = fleet._id_to_slot["r0"]
    # Make East (index 2) strongly preferred from (0,0)
    fleet.q[slot, 0, 0, :] = [0.0, 0.0, 99.0, 0.0]
    actions = fleet.select_actions()
    assert actions["r0"] == 2


def test_select_actions_only_valid_moves():
    """Agent in corner (0,0): North (dy=-1) and West (dx=-1) are out-of-bounds."""
    fleet = make_fleet(num_agents=1)
    fleet.epsilon = 0.0
    slot = fleet._id_to_slot["r0"]
    # Make N and W (indices 0, 3) look best — but they are invalid from (0,0)
    fleet.q[slot, 0, 0, :] = [99.0, 0.0, 0.0, 99.0]
    actions = fleet.select_actions()
    # Must choose among valid moves only: S (1) or E (2)
    assert actions["r0"] in (1, 2)


def test_peek_next_reflects_movement_model():
    fleet = make_fleet(num_agents=1)
    # East from (0,0) → (1,0)
    nxt = fleet.peek_next("r0", 2)
    assert nxt == Position(1, 0)


def test_peek_next_blocked_returns_same_cell():
    """Trying to enter an obstacle keeps the agent in place."""
    g = make_grid()  # obstacle at (2,2)
    fleet = EpidemicHystereticQLearning(g, max_agents=20, seed=0)
    fleet.add_agent("r0", Position(2, 1))   # one cell above the obstacle
    # South (index 1) would go to (2,2) which is blocked
    nxt = fleet.peek_next("r0", 1)
    assert nxt == Position(2, 1)


# ---------------------------------------------------------------------------
# 7. Communications: GossipMessage wire format
# ---------------------------------------------------------------------------

def test_gossip_message_size():
    msg = GossipMessage(sender=0, indices=np.array([0, 1, 2]), values=np.array([1.0, 2.0, 3.0]))
    assert msg.size == 3


def test_gossip_message_empty():
    msg = GossipMessage(sender=0, indices=np.array([], dtype=np.int64), values=np.array([]))
    assert msg.size == 0


# ---------------------------------------------------------------------------
# 8. Mean Q and greedy policy shapes
# ---------------------------------------------------------------------------

def test_mean_q_zero_on_fresh_fleet():
    fleet = make_fleet(num_agents=3)
    assert fleet.mean_q() == pytest.approx(0.0)


def test_greedy_policy_shape():
    fleet = make_fleet(num_agents=1)
    g = make_grid()
    policy = fleet.greedy_policy("r0")
    assert policy.shape == (g.height, g.width)
