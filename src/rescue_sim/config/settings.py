"""Typed settings for ST02 scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSettings:
    width: int
    height: int
    obstacle_probability: float
    target_a_count: int
    target_b_count: int
    random_seed: int | None = None


@dataclass(frozen=True)
class AgentSettings:
    start_x: int
    start_y: int
    sensor_range: int


@dataclass(frozen=True)
class SimulationSettings:
    max_steps: int


@dataclass(frozen=True)
class FleetSettings:
    """YAML-facing settings for the decentralized Epidemic Hysteretic fleet.

    Maps directly onto ``shared.HystereticConfig`` and ``shared.GossipConfig``;
    see ``rescue_sim.Qlearning.q_learning.EpidemicHystereticQLearning``.
    """

    num_agents: int = 4           # agents active at the start of an episode
    max_agents: int = 20          # pre-allocated capacity (1 <= N <= 20)
    alpha: float = 0.5            # learning rate for positive TD error
    beta: float = 0.1             # muted learning rate for negative TD error (beta << alpha)
    discount_factor: float = 0.95
    epsilon: float = 0.2
    epsilon_decay: float = 0.0    # subtracted from epsilon each episode (floored at 0)
    comm_radius: float = 3.0      # Euclidean distance that opens a peer link
    gossip_cooldown: int = 5      # steps before the same pair may re-sync
    max_links_per_step: int = 2   # per-agent handshake budget (congestion control)
    utility_threshold: float = 0.0
    random_seed: int | None = None


@dataclass(frozen=True)
class MappoSettings:
    """Hyper-parameters for MAPPO (see rescue_sim.MAPPO).

    Defaults follow the recommendations in Yu et al. 2022, "The Surprising
    Effectiveness of PPO in Cooperative Multi-Agent Games".
    """

    num_agents: int = 4
    view_radius: int = 2          # egocentric window radius for the actor
    max_steps: int = 200
    hidden_dim: int = 64          # 64x64 MLP per the MAPPO paper
    learning_rate: float = 7e-4
    gamma: float = 0.99           # discount factor
    gae_lambda: float = 0.95      # GAE bias/variance trade-off
    clip: float = 0.2             # PPO/value clip range
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 10              # update epochs per rollout
    rollout_steps: int = 512      # timesteps collected before each update
    max_grad_norm: float = 0.5
    normalize_value: bool = True  # value-target normalization (MAPPO trick #1)
    random_seed: int | None = None

