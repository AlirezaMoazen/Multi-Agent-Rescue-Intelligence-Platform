"""Mixture-of-Experts (MoE) router over every rescue strategy in the project.

This is the project's top-level "use the best method for *this* grid" layer. It
is a **performance-gated Mixture of Experts**, which is exactly the classical
**per-instance algorithm-selection / portfolio** problem (Rice 1976; SATzilla,
Xu et al. 2008) phrased as an MoE with mostly *fixed* experts (Jacobs et al.
1991): a gate scores each expert on the grid in front of it and routes the fleet
to whichever solves it best.

The expert pool is the whole project:

* **Classical (no-AI) experts** -- the 7 baselines (frontier, DFS, prioritized
  planning, CBS, ICBS, ECBS, M*), run as a synchronized team.
* **Deep experts (frozen)** -- QMIX, TransfQMix, MAPPO, and their ValueEnsemble.
* **One adaptive expert** -- the ``EpidemicHystereticQLearning`` fleet, which
  keeps learning *this* grid on every try.

The gate is what makes the combination safe: it only ever serves the
best-scoring expert, so the MoE is **never worse than the strongest single
method**. The twist over a static portfolio: the adaptive expert improves every
try (bootstrapping off-policy from the best deep expert's demonstrations while
it is behind), so on a grid it can master it climbs the leaderboard and takes
over -- and from then on it self-plays.

Everything is comparable because every expert is scored on the same fixed grid
with the same ``solve_score`` (read from the environment's success/rescued/steps,
not the reward). Requires torch for the deep experts: ``pip install -e ".[ensemble]"``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from rescue_sim.config.settings import GridSettings, MoeSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.Qlearning.communications import DefaultCommsBus
from rescue_sim.Qlearning.multi_agent_baseline import (
    DEFAULT_MULTI_AGENT_BASELINES,
    default_start_positions,
    run_multi_agent_baseline,
)
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning
from rescue_sim.shared import GossipConfig, HystereticConfig, RewardConfig
from rescue_sim.TransfQMix.transf_qmix import EntityRescueEnv

# A greedy joint-action policy: given the env and its flat observation, return
# one action index (into CARDINAL_ACTIONS) per agent. This is the uniform
# interface the gate uses to roll any deep expert out on the fixed grid.
Policy = Callable[["FixedGridEntityEnv", np.ndarray], np.ndarray]

# All agents start at the grid origin unless spread starts are supplied.
START = Position(0, 0)

# Reward the adaptive (tabular) expert learns from. Its state is just the grid
# cell (y, x), so it needs a *Markovian* navigation reward: a large sparse bonus
# for reaching a target plus a per-step cost. The deep models' SPRINT3 reward
# adds history-dependent terms ("newly discovered cell", "repeated cell") that
# are not a function of the cell alone, which -- combined with the hysteretic
# max-only update and the epidemic max-merge -- would inflate every Q-value
# instead of forming a gradient toward the targets. This only changes what the
# fleet *learns*: the gate scores every expert from the environment's
# success/rescued/steps info (not the reward), so the comparison stays fair.
ADAPTIVE_REWARD_CONFIG = RewardConfig(
    move=-1.0,
    invalid_move=-5.0,
    wait=-2.0,
    discovered_cell_bonus=0.0,
    repeated_cell=0.0,
    rescued_target_a=150.0,
    rescued_target_b=100.0,
    completed_episode_bonus=50.0,
)


# ---------------------------------------------------------------------------
# Fixed-grid environment
# ---------------------------------------------------------------------------


class FixedGridEntityEnv(EntityRescueEnv):
    """An ``EntityRescueEnv`` pinned to ONE grid (reset reuses it, never regenerates).

    The base environment draws a fresh random grid on every ``reset()``. The MoE
    instead needs every expert to compete on the *same* rescue instance -- that
    is what "solve the grid" means here -- so this subclass freezes the grid and
    only resets agent positions and bookkeeping. All observation, stepping,
    reward, and metric logic is inherited unchanged (no duplication).
    """

    def __init__(
        self,
        grid: Grid,
        num_agents: int,
        max_steps: int,
        view_radius: int,
        reward_config,
        start_positions: list[Position] | None = None,
    ) -> None:
        # Rebuild a GridSettings that matches the fixed grid so the parent's
        # observation/state dimensions are correct; obstacle_probability is
        # irrelevant because reset() never regenerates.
        settings = GridSettings(
            width=grid.width,
            height=grid.height,
            obstacle_probability=0.0,
            target_a_count=len(grid.target_a_positions),
            target_b_count=len(grid.target_b_positions),
        )
        super().__init__(
            settings,
            num_agents=num_agents,
            max_steps=max_steps,
            view_radius=view_radius,
            reward_config=reward_config,
            seed=0,
        )
        self._fixed_grid = grid
        self._start_positions = list(start_positions) if start_positions else [
            START for _ in range(num_agents)
        ]

    def reset(self) -> np.ndarray:
        """Reset agents to their start cells on the *fixed* grid; returns flat obs."""
        self.grid = self._fixed_grid
        self.positions = list(self._start_positions)
        self._rescued = set()
        self._discovered = set()
        self._visited = set(self._start_positions)
        self._steps = 0
        self._build_grid_arrays()
        return self._observations()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def solve_score(info: dict, max_steps: int) -> float:
    """Scalar "how well was the grid solved" score; higher is better.

    Lexicographic by construction:

    * a full success (all targets rescued) always outscores any failure
      (the ``+2.0`` term dwarfs the rest), then
    * more targets rescued is better (``rescued_frac`` in ``[0, 1]``), then
    * fewer steps is better (``-0.5 * step_frac`` breaks ties between two runs
      that rescued the same number).

    One float keeps the gate trivial to compare and the history easy to plot.
    """
    targets = info.get("targets", 0) or 0
    rescued_frac = info["rescued"] / targets if targets else 1.0
    step_frac = info["steps"] / max_steps if max_steps else 0.0
    return (2.0 if info["success"] else 0.0) + rescued_frac - 0.5 * step_frac


def _metrics(info: dict, max_steps: int) -> dict:
    """Serializable per-expert result for one grid (API/frontend friendly)."""
    return {
        "success": bool(info["success"]),
        "rescued": int(info["rescued"]),
        "targets": int(info.get("targets", 0)),
        "steps": int(info["steps"]),
        "score": round(solve_score(info, max_steps), 4),
    }


# ---------------------------------------------------------------------------
# Greedy policy adapters for the deep experts
# ---------------------------------------------------------------------------
# Each returns a uniform ``Policy`` so the gate can roll any model out on the
# fixed grid without knowing its internals.


def qmix_policy(qmix) -> Policy:
    def act(env: FixedGridEntityEnv, flat_obs: np.ndarray) -> np.ndarray:
        return qmix.select_actions(flat_obs, env.valid_action_mask(), greedy=True)

    return act


def transf_policy(transf) -> Policy:
    def act(env: FixedGridEntityEnv, _flat_obs: np.ndarray) -> np.ndarray:
        return transf.select_actions(env.entity_obs(), env.valid_action_mask(), greedy=True)

    return act


def ensemble_policy(ensemble) -> Policy:
    def act(env: FixedGridEntityEnv, flat_obs: np.ndarray) -> np.ndarray:
        return ensemble.select_actions(flat_obs, env.entity_obs(), env.valid_action_mask())

    return act


def mappo_policy(mappo) -> Policy:
    import torch

    def act(env: FixedGridEntityEnv, flat_obs: np.ndarray) -> np.ndarray:
        mask = torch.as_tensor(env.valid_action_mask())
        logits = mappo.net.actor(torch.as_tensor(flat_obs).float()).masked_fill(~mask, -1e8)
        return logits.argmax(dim=-1).numpy()

    return act


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MoeTrial:
    """One try on the fixed grid: how the adaptive fleet did and who led."""

    trial: int
    leader: str          # expert that would be served this try (best score so far)
    fleet: dict          # adaptive-fleet greedy metrics after this try's learning
    epsilon: float       # fleet's current exploration rate
    mean_q: float        # mean Q across the fleet (learning-progress signal)
    syncs: int           # gossip syncs during this try's learning episode

    def to_dict(self) -> dict:
        return {
            "trial": self.trial,
            "leader": self.leader,
            "fleet": self.fleet,
            "epsilon": round(self.epsilon, 4),
            "mean_q": round(self.mean_q, 4),
            "syncs": self.syncs,
        }


@dataclass
class MoeReport:
    """Outcome of an MoE run: the fixed-expert leaderboard plus the fleet's climb."""

    leaderboard: dict = field(default_factory=dict)   # frozen expert name -> metrics
    teacher: str | None = None                        # deep expert that taught the fleet
    history: list[MoeTrial] = field(default_factory=list)
    surpassed_at: int | None = None                   # first try the fleet led overall
    final_leader: str = ""
    best_fleet_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "leaderboard": self.leaderboard,
            "teacher": self.teacher,
            "history": [t.to_dict() for t in self.history],
            "surpassed_at": self.surpassed_at,
            "final_leader": self.final_leader,
            "best_fleet_score": round(self.best_fleet_score, 4),
        }


# ---------------------------------------------------------------------------
# The Mixture-of-Experts router
# ---------------------------------------------------------------------------

FLEET = "EpidemicFleet"


class MixtureOfExperts:
    """Performance-gated router over classical, deep, and adaptive rescue experts.

    Parameters
    ----------
    deep_experts:
        ``{name: greedy Policy}`` for the frozen deep models (see ``from_models``
        and the ``*_policy`` adapters). Used both as gate candidates and as the
        teacher pool for the adaptive fleet.
    ref_env:
        Any ``RescueEnv``/``EntityRescueEnv`` the deep models were built with --
        the fixed grid, agent count, view radius, episode length, and grid
        settings are taken from it so every expert is dimension-compatible.
    settings, grid, comms:
        Mixture hyper-parameters, an optional fixed grid (else generated from
        ``ref_env`` + ``settings.grid_seed``), and the fleet's gossip bus.
    baselines:
        ``{name: factory}`` of classical (no-AI) strategies to include as
        gate candidates; defaults to all seven.
    """

    def __init__(
        self,
        deep_experts: Mapping[str, Policy],
        ref_env,
        settings: MoeSettings = MoeSettings(),
        grid: Grid | None = None,
        comms=None,
        baselines: Mapping[str, Callable] = DEFAULT_MULTI_AGENT_BASELINES,
        adaptive_reward_config: RewardConfig = ADAPTIVE_REWARD_CONFIG,
    ) -> None:
        self.deep_experts = dict(deep_experts)
        self.baselines = dict(baselines)
        self.cfg = settings
        self.comms = comms if comms is not None else DefaultCommsBus()

        self.num_agents = ref_env.num_agents
        self.max_steps = ref_env.max_steps
        self.view_radius = ref_env.view_radius
        self.sensor_range = max(2, ref_env.view_radius)

        if grid is None:
            grid_settings = replace(ref_env.grid_settings, random_seed=settings.grid_seed)
            grid = generate_grid(grid_settings, START)
        self.grid = grid

        # Spread, reachable, non-target starts (reuses the multi-agent baseline's
        # placement). With several targets this lets the memoryless fleet divide
        # the work -- each agent's shared "go to nearest target" policy flows it
        # to a different nearby target.
        start_map = default_start_positions(grid, self.num_agents)
        self.agent_ids = list(start_map)
        self.start_list = [start_map[aid] for aid in self.agent_ids]
        self.starts = dict(start_map)

        # Shared fixed-grid env both the deep experts and the fleet roll out on.
        self.env = FixedGridEntityEnv(
            grid,
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            view_radius=self.view_radius,
            reward_config=adaptive_reward_config,
            start_positions=self.start_list,
        )

        # The adaptive expert: a decentralized Epidemic Hysteretic fleet.
        self.fleet = EpidemicHystereticQLearning(
            grid,
            config=HystereticConfig(
                alpha=settings.alpha,
                beta=settings.beta,
                discount_factor=settings.discount_factor,
                epsilon=settings.epsilon_start,
            ),
            gossip=GossipConfig(
                comm_radius=settings.comm_radius,
                cooldown=settings.gossip_cooldown,
                max_links_per_step=settings.max_links_per_step,
                utility_threshold=settings.utility_threshold,
            ),
            max_agents=max(settings.max_agents, self.num_agents),
            seed=settings.random_seed,
        )
        for aid in self.agent_ids:
            self.fleet.add_agent(aid, self.starts[aid])
        self._rng = np.random.default_rng(settings.random_seed)

    # -- construction from trained models -----------------------------------

    @classmethod
    def from_models(
        cls,
        *,
        qmix=None,
        transf=None,
        mappo=None,
        ensemble=None,
        settings: MoeSettings = MoeSettings(),
        grid: Grid | None = None,
        comms=None,
        baselines: Mapping[str, Callable] = DEFAULT_MULTI_AGENT_BASELINES,
    ) -> "MixtureOfExperts":
        """Build an MoE from any subset of trained deep models (at least one)."""
        deep: dict[str, Policy] = {}
        ref_env = None
        if qmix is not None:
            deep["QMIX"] = qmix_policy(qmix)
            ref_env = qmix.env
        if transf is not None:
            deep["TransfQMix"] = transf_policy(transf)
            ref_env = transf.env
        if mappo is not None:
            deep["MAPPO"] = mappo_policy(mappo)
            ref_env = mappo.env
        if ensemble is not None:
            deep["Ensemble"] = ensemble_policy(ensemble)
            ref_env = ensemble.qmix.env
        if ref_env is None:
            raise ValueError("from_models needs at least one of qmix/transf/mappo/ensemble")
        return cls(deep, ref_env, settings=settings, grid=grid, comms=comms, baselines=baselines)

    # -- rollouts -----------------------------------------------------------

    def _greedy_episode(self, policy: Policy) -> dict:
        """Run one no-learning episode on the fixed grid driven by ``policy``."""
        flat_obs = self.env.reset()
        done = False
        info: dict = {}
        while not done:
            actions = policy(self.env, flat_obs)
            flat_obs, _, done, info = self.env.step(actions)
        return info

    def _adaptive_greedy_episode(self) -> dict:
        """Greedy roll-out of the *current* fleet policy (read-only, no learning)."""
        policies = {aid: self.fleet.greedy_policy(aid) for aid in self.agent_ids}

        def act(env: FixedGridEntityEnv, _flat_obs: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    policies[self.agent_ids[i]][env.positions[i].y, env.positions[i].x]
                    for i in range(env.num_agents)
                ],
                dtype=np.int64,
            )

        return self._greedy_episode(act)

    def _baseline_metrics(self, name: str, factory: Callable) -> dict:
        """Score one classical baseline as a synchronized team on the fixed grid."""
        strategy = factory(self.cfg.random_seed)
        result = run_multi_agent_baseline(
            strategy=strategy,
            grid=self.grid,
            start_positions=self.starts,
            max_steps=self.max_steps,
            sensor_range=self.sensor_range,
            strategy_name=name,
        )
        info = {
            "success": result.success,
            "rescued": result.rescued_targets,
            "targets": result.total_targets,
            "steps": result.steps,
        }
        return _metrics(info, self.max_steps)

    def _teacher_actions(self, flat_obs: np.ndarray, teacher: Policy) -> np.ndarray:
        """Teacher's greedy joint action with a little exploration noise."""
        actions = teacher(self.env, flat_obs)
        avail = self.env.valid_action_mask()
        for i in range(self.num_agents):
            if self._rng.random() < self.fleet.epsilon:
                actions[i] = int(self._rng.choice(np.flatnonzero(avail[i])))
        return actions

    def _learn_episode(self, teacher: Policy | None) -> tuple[dict, int]:
        """One learning episode for the fleet on the fixed grid.

        With a ``teacher`` the fleet learns **off-policy** from that expert's
        demonstrations (how it bootstraps on large/sparse grids); with ``None``
        it explores with its own epsilon-greedy policy. Either way it applies a
        hysteretic TD update and runs a gossip round each step.
        """
        flat_obs = self.env.reset()
        self.fleet.reset_positions(self.starts)
        done = False
        info: dict = {}
        syncs = 0
        while not done:
            if teacher is not None:
                actions = self._teacher_actions(flat_obs, teacher)
            else:
                amap = self.fleet.select_actions()
                actions = np.array(
                    [amap[self.agent_ids[i]] for i in range(self.num_agents)], dtype=np.int64
                )
            action_map = {self.agent_ids[i]: int(actions[i]) for i in range(self.num_agents)}
            flat_obs, team_reward, done, info = self.env.step(actions)
            next_positions = {
                self.agent_ids[i]: self.env.positions[i] for i in range(self.num_agents)
            }
            rewards = {aid: team_reward for aid in self.agent_ids}  # shared cooperative reward
            dones = {aid: done for aid in self.agent_ids}
            self.fleet.record_transitions(action_map, rewards, next_positions, dones)
            syncs += self.comms.exchange(self.fleet)
        self.fleet.decay_epsilon(self.cfg.epsilon_decay, floor=self.cfg.epsilon_floor)
        return info, syncs

    # -- the gate -----------------------------------------------------------

    def run(self, num_trials: int | None = None) -> MoeReport:
        """Score the fixed experts once, then let the fleet climb the leaderboard.

        The deep models and classical baselines are frozen and the grid is
        fixed, so their scores are constant -- evaluated once. Each try the fleet
        learns (taught by the best deep expert until it leads) and is re-scored;
        the overall leader is whoever has the best score that try.
        """
        trials = self.cfg.num_trials if num_trials is None else num_trials

        leaderboard: dict[str, dict] = {}
        for name, policy in self.deep_experts.items():
            leaderboard[name] = _metrics(self._greedy_episode(policy), self.max_steps)
        for name, factory in self.baselines.items():
            leaderboard[name] = self._baseline_metrics(name, factory)

        best_frozen_name = max(leaderboard, key=lambda n: leaderboard[n]["score"])
        best_frozen_score = leaderboard[best_frozen_name]["score"]
        # The fleet learns from the strongest deep expert (baselines can't act on
        # the deep env, so they are gate candidates only, not teachers).
        teacher_name = (
            max(self.deep_experts, key=lambda n: leaderboard[n]["score"])
            if self.deep_experts
            else None
        )
        teacher_policy = self.deep_experts.get(teacher_name) if teacher_name else None

        report = MoeReport(leaderboard=leaderboard, teacher=teacher_name)
        fleet_leads = False
        for t in range(1, trials + 1):
            _, syncs = self._learn_episode(None if fleet_leads else teacher_policy)
            fleet_metrics = _metrics(self._adaptive_greedy_episode(), self.max_steps)
            fleet_score = fleet_metrics["score"]
            report.best_fleet_score = round(max(report.best_fleet_score, fleet_score), 4)

            if fleet_score > best_frozen_score:
                leader = FLEET
                if report.surpassed_at is None:
                    report.surpassed_at = t
                    fleet_leads = True  # self-play from the next try on
            else:
                leader = best_frozen_name

            report.history.append(
                MoeTrial(
                    trial=t,
                    leader=leader,
                    fleet=fleet_metrics,
                    epsilon=self.fleet.epsilon,
                    mean_q=self.fleet.mean_q(),
                    syncs=syncs,
                )
            )

        report.final_leader = report.history[-1].leader if report.history else best_frozen_name
        return report

    # -- serving ------------------------------------------------------------

    def leaderboard(self) -> dict:
        """Score every fixed expert once (for an API/frontend comparison view)."""
        board = {n: self._metrics_for_deep(p) for n, p in self.deep_experts.items()}
        board.update({n: self._baseline_metrics(n, f) for n, f in self.baselines.items()})
        return board

    def _metrics_for_deep(self, policy: Policy) -> dict:
        return _metrics(self._greedy_episode(policy), self.max_steps)
