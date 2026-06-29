"""Mixture-of-Experts (MoE) gate: generalist deep ensemble vs. specialist fleet.

Motivation
----------
The deep value methods (QMIX, TransfQMix, and their ``ValueEnsemble``) are
*generalists*: trained over many random grids, they act well on a grid they have
never seen, but they are frozen at deployment and never adapt to the grid in
front of them. ``EpidemicHystereticQLearning`` is the opposite -- a tabular
*specialist* that knows nothing about a fresh grid but can learn that one grid
very well if it is allowed to try repeatedly.

This module combines the two with a classic Mixture-of-Experts pattern:

* **Experts.** Expert 1 is the frozen deep ensemble. Expert 2 is the adaptive
  tabular fleet.
* **Gate.** A performance gate scores both experts on the *same* fixed grid and
  routes the fleet to whichever currently solves it better.
* **The trick.** The adaptive expert keeps learning on *every* try -- even while
  the deep expert is the one driving -- so its score climbs try after try. On a
  fixed grid it eventually beats the generalist; from the next try on it drives
  the fleet and keeps improving.

Everything is comparable because both experts share the project contract:
``RescueEnv`` actions are indices into ``CARDINAL_ACTIONS`` (N, S, E, W) and both
experts read the same ``Grid``. The MoE never retrains the deep nets; it only
runs them greedily and lets the tabular learner catch up.

Requires torch (the deep expert): ``pip install -e ".[ensemble]"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from rescue_sim.config.settings import GridSettings, MoeSettings
from rescue_sim.environment.generator import generate_grid
from rescue_sim.environment.grid import Grid, Position
from rescue_sim.Ensemble.ensemble import ValueEnsemble
from rescue_sim.Qlearning.communications import DefaultCommsBus
from rescue_sim.Qlearning.multi_agent_baseline import default_start_positions
from rescue_sim.Qlearning.q_learning import EpidemicHystereticQLearning
from rescue_sim.shared import GossipConfig, HystereticConfig, RewardConfig
from rescue_sim.TransfQMix.transf_qmix import EntityRescueEnv

# All agents start at the grid origin, matching RescueEnv.reset().
START = Position(0, 0)

# Reward the adaptive (tabular) expert learns from. Its state is just the grid
# cell (y, x), so it needs a *Markovian* navigation reward: a large sparse bonus
# for reaching a target plus a per-step cost. The deep models' SPRINT3 reward
# adds history-dependent terms ("newly discovered cell", "repeated cell") that
# are not a function of the cell alone, which -- combined with the hysteretic
# max-only update and the epidemic max-merge -- would inflate every Q-value
# instead of forming a gradient toward the targets. This only changes what the
# fleet *learns*: the gate scores both experts from the environment's
# success/rescued/steps info (not the reward), so the comparison stays fair, and
# the frozen deep experts are evaluated greedily (reward-independent).
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


class FixedGridEntityEnv(EntityRescueEnv):
    """An ``EntityRescueEnv`` pinned to ONE grid (reset reuses it, never regenerates).

    The base environment draws a fresh random grid on every ``reset()``. The MoE
    instead needs both experts to compete on the *same* rescue instance -- that
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
        # Spread-out, reachable starts let the memoryless fleet divide several
        # targets (each agent flows to its nearest); defaults to all-origin to
        # match the base RescueEnv when no starts are given.
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


def solve_score(info: dict, max_steps: int) -> float:
    """Scalar "how well was the grid solved" score; higher is better.

    Lexicographic by construction:

    * a full success (all targets rescued) always outscores any failure
      (the ``+2.0`` term dwarfs the rest), then
    * more targets rescued is better (``rescued_frac`` in ``[0, 1]``), then
    * fewer steps is better (``-0.5 * step_frac`` breaks ties between two runs
      that rescued the same number).

    Returning one float keeps the gate trivial to compare and the history easy
    to plot in the API/frontend.
    """
    targets = info.get("targets", 0) or 0
    rescued_frac = info["rescued"] / targets if targets else 1.0
    step_frac = info["steps"] / max_steps if max_steps else 0.0
    return (2.0 if info["success"] else 0.0) + rescued_frac - 0.5 * step_frac


def _metrics(info: dict, max_steps: int) -> dict:
    """Serializable per-expert result for one try (API/frontend friendly)."""
    return {
        "success": bool(info["success"]),
        "rescued": int(info["rescued"]),
        "targets": int(info.get("targets", 0)),
        "steps": int(info["steps"]),
        "score": round(solve_score(info, max_steps), 4),
    }


@dataclass(frozen=True, slots=True)
class MoeTrial:
    """One try on the fixed grid (one row of the MoE history)."""

    trial: int
    leader: str          # expert that drove the fleet THIS try: "deep" | "adaptive"
    deep: dict           # deep-expert metrics on the grid (constant; it is frozen)
    adaptive: dict       # adaptive-expert greedy metrics after this try's learning
    epsilon: float       # adaptive learner's current exploration rate
    mean_q: float        # mean Q-value across the fleet (learning-progress signal)
    syncs: int           # gossip syncs performed during this try's learning episode

    def to_dict(self) -> dict:
        return {
            "trial": self.trial,
            "leader": self.leader,
            "deep": self.deep,
            "adaptive": self.adaptive,
            "epsilon": round(self.epsilon, 4),
            "mean_q": round(self.mean_q, 4),
            "syncs": self.syncs,
        }


@dataclass
class MoeReport:
    """Outcome of an MoE run: per-try history plus a short summary."""

    history: list[MoeTrial] = field(default_factory=list)
    surpassed_at: int | None = None   # first try the adaptive expert beat the deep one
    final_leader: str = "deep"
    deep_score: float = 0.0
    best_adaptive_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "history": [t.to_dict() for t in self.history],
            "surpassed_at": self.surpassed_at,
            "final_leader": self.final_leader,
            "deep_score": round(self.deep_score, 4),
            "best_adaptive_score": round(self.best_adaptive_score, 4),
        }


class MixtureOfExperts:
    """Gate the trained deep ensemble against an adaptive tabular fleet on one grid.

    Parameters
    ----------
    deep_expert:
        A ``ValueEnsemble`` of a trained QMIX + TransfQMix. It is used only to
        *act* (greedily); it is never retrained. The fixed grid the experts
        compete on, the agent count, the view radius, the episode length, and
        the reward config are all taken from this ensemble's models so the two
        experts are guaranteed comparable.
    settings:
        Mixture and adaptive-learner hyper-parameters (see ``MoeSettings``).
    grid:
        The single grid both experts compete on. If ``None``, one is generated
        from the deep models' ``GridSettings`` using ``settings.grid_seed``.
    comms:
        Communication bus for the epidemic gossip round each learning step.
        Defaults to ``DefaultCommsBus`` (perfect channel); pass a
        ``ResilientCommsBus`` to study lossy/limited links.
    """

    def __init__(
        self,
        deep_expert: ValueEnsemble,
        settings: MoeSettings = MoeSettings(),
        grid: Grid | None = None,
        comms=None,
        adaptive_reward_config: RewardConfig = ADAPTIVE_REWARD_CONFIG,
    ) -> None:
        self.deep = deep_expert
        self.cfg = settings
        self.comms = comms if comms is not None else DefaultCommsBus()

        # Inherit the deployment shape from the trained models so dimensions match.
        deep_env = deep_expert.qmix.env
        self.num_agents = deep_env.num_agents
        self.max_steps = deep_env.max_steps
        view_radius = deep_env.view_radius

        if grid is None:
            grid_settings = replace(deep_env.grid_settings, random_seed=settings.grid_seed)
            grid = generate_grid(grid_settings, START)
        self.grid = grid

        # Spread, reachable, non-target start cells (reuses the multi-agent
        # baseline's placement). With several targets this lets the memoryless
        # fleet divide the work -- each agent's shared "go to nearest target"
        # policy flows it to a different nearby target -- so the tabular
        # specialist stays competitive on multi-target grids.
        start_map = default_start_positions(grid, self.num_agents)
        self.agent_ids = list(start_map)
        self.start_list = [start_map[aid] for aid in self.agent_ids]
        self.starts = dict(start_map)

        # The shared, fixed-grid environment both experts roll out on. Its reward
        # config drives only the adaptive learner (deep eval ignores reward, and
        # the gate scores from env info), so it uses the navigation-focused reward.
        self.env = FixedGridEntityEnv(
            grid,
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            view_radius=view_radius,
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

    # -- rollouts -----------------------------------------------------------

    def _greedy_episode(self, act) -> dict:
        """Run one no-learning episode on the fixed grid; ``act`` picks the joint action.

        ``act(env, flat_obs) -> np.ndarray`` returns one action index per agent.
        Reused for both the deep eval and the adaptive greedy eval.
        """
        flat_obs = self.env.reset()
        done = False
        info: dict = {}
        while not done:
            actions = act(self.env, flat_obs)
            flat_obs, _, done, info = self.env.step(actions)
        return info

    def _deep_episode(self) -> dict:
        """Greedy roll-out of the deep ensemble on the fixed grid."""

        def act(env, flat_obs):
            tokens = env.entity_obs()
            avail = env.valid_action_mask()
            return self.deep.select_actions(flat_obs, tokens, avail)

        return self._greedy_episode(act)

    def _adaptive_greedy_episode(self) -> dict:
        """Greedy roll-out of the *current* fleet policy (read-only, no learning)."""
        # greedy_policy(aid) is the best valid action per cell; index it by position.
        policies = {aid: self.fleet.greedy_policy(aid) for aid in self.agent_ids}

        def act(env, _flat_obs):
            return np.array(
                [
                    policies[self.agent_ids[i]][env.positions[i].y, env.positions[i].x]
                    for i in range(env.num_agents)
                ],
                dtype=np.int64,
            )

        return self._greedy_episode(act)

    def _teacher_actions(self, flat_obs: np.ndarray) -> np.ndarray:
        """Deep-expert joint action for the current env state, with epsilon noise.

        Mixing in a little exploration means the fleet sees alternatives to the
        demonstrated path, not just the single trajectory the teacher walks.
        """
        tokens = self.env.entity_obs()
        avail = self.env.valid_action_mask()
        actions = self.deep.select_actions(flat_obs, tokens, avail)
        for i in range(self.num_agents):
            if self._rng.random() < self.fleet.epsilon:
                actions[i] = int(self._rng.choice(np.flatnonzero(avail[i])))
        return actions

    def _learn_episode(self, teacher: str) -> tuple[dict, int]:
        """One learning episode for the fleet on the fixed grid.

        The *behaviour* policy is the current leader's. While the deep expert
        leads it **demonstrates**: the fleet learns off-policy (Q-learning is
        off-policy) from the generalist's trajectory, which is how the tabular
        specialist bootstraps on grids too large to stumble onto targets by
        chance. Once the fleet leads, it explores with its own epsilon-greedy
        policy. Either way the fleet applies a hysteretic TD update and gossips
        each step. Returns ``(episode_info, total_syncs)``.
        """
        flat_obs = self.env.reset()
        self.fleet.reset_positions(self.starts)
        done = False
        info: dict = {}
        syncs = 0
        while not done:
            if teacher == "deep":
                actions = self._teacher_actions(flat_obs)
            else:
                amap = self.fleet.select_actions()  # fleet's own epsilon-greedy policy
                actions = np.array(
                    [amap[self.agent_ids[i]] for i in range(self.num_agents)], dtype=np.int64
                )
            action_map = {self.agent_ids[i]: int(actions[i]) for i in range(self.num_agents)}
            flat_obs, team_reward, done, info = self.env.step(actions)
            # The cooperative team reward is shared by every agent.
            next_positions = {
                self.agent_ids[i]: self.env.positions[i] for i in range(self.num_agents)
            }
            rewards = {aid: team_reward for aid in self.agent_ids}
            dones = {aid: done for aid in self.agent_ids}
            self.fleet.record_transitions(action_map, rewards, next_positions, dones)
            syncs += self.comms.exchange(self.fleet)
        self.fleet.decay_epsilon(self.cfg.epsilon_decay, floor=self.cfg.epsilon_floor)
        return info, syncs

    # -- the gate -----------------------------------------------------------

    def run(self, num_trials: int | None = None) -> MoeReport:
        """Run the MoE for ``num_trials`` tries and return the routing history.

        Each try: the adaptive expert learns one episode, then is scored
        greedily and compared against the (constant) deep score. The expert that
        was *leading at the start of the try* is the one credited with driving
        the fleet that try; once the adaptive expert wins, it leads from the
        next try on -- matching "it runs the agents in the next try".
        """
        trials = self.cfg.num_trials if num_trials is None else num_trials

        # The deep expert is frozen and the grid is fixed, so its greedy score is
        # constant -- evaluate it once and reuse it as the bar to beat.
        deep_info = self._deep_episode()
        deep_score = solve_score(deep_info, self.max_steps)
        deep_metrics = _metrics(deep_info, self.max_steps)

        report = MoeReport(deep_score=round(deep_score, 4), final_leader="deep")
        leader = "deep"
        for t in range(1, trials + 1):
            leader_this_try = leader  # who drives the fleet this try
            # While the deep expert leads it teaches; once the fleet leads it self-plays.
            _, syncs = self._learn_episode(leader_this_try)
            adapt_info = self._adaptive_greedy_episode()
            adapt_score = solve_score(adapt_info, self.max_steps)
            report.best_adaptive_score = round(
                max(report.best_adaptive_score, adapt_score), 4
            )

            report.history.append(
                MoeTrial(
                    trial=t,
                    leader=leader_this_try,
                    deep=deep_metrics,
                    adaptive=_metrics(adapt_info, self.max_steps),
                    epsilon=self.fleet.epsilon,
                    mean_q=self.fleet.mean_q(),
                    syncs=syncs,
                )
            )

            # Hand over to the adaptive expert for the NEXT try once it wins.
            if leader == "deep" and adapt_score > deep_score:
                report.surpassed_at = t
                leader = "adaptive"

        report.final_leader = leader
        return report

    # -- serving ------------------------------------------------------------

    @property
    def current_leader(self) -> str:
        """Cheap convenience: the expert with the better score right now."""
        deep_score = solve_score(self._deep_episode(), self.max_steps)
        adapt_score = solve_score(self._adaptive_greedy_episode(), self.max_steps)
        return "adaptive" if adapt_score > deep_score else "deep"

    def serve_actions(self, positions: list[Position]) -> np.ndarray:
        """Joint action from the better expert, for live stepping by an API/frontend.

        ``positions`` are the agents' current cells (env order). Returns one
        action index (into ``CARDINAL_ACTIONS``) per agent.
        """
        if self.current_leader == "adaptive":
            policies = [self.fleet.greedy_policy(aid) for aid in self.agent_ids]
            return np.array(
                [policies[i][positions[i].y, positions[i].x] for i in range(self.num_agents)],
                dtype=np.int64,
            )
        # Deep expert needs a fresh observation, so roll the fixed env to these
        # positions is overkill; expose the deep path only when it leads.
        self.env.reset()
        self.env.positions = list(positions)
        flat_obs = self.env._observations()
        tokens = self.env.entity_obs()
        avail = self.env.valid_action_mask()
        return self.deep.select_actions(flat_obs, tokens, avail)
