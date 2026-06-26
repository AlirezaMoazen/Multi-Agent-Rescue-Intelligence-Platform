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


@dataclass(frozen=True)
class QmixSettings:
    """Hyper-parameters for QMIX (see rescue_sim.QMIX).

    Defaults follow the PyMARL reference for Rashid et al. 2018, "QMIX:
    Monotonic Value Function Factorisation for Deep Multi-Agent RL".
    """

    num_agents: int = 4
    view_radius: int = 2
    max_steps: int = 200
    hidden_dim: int = 64           # agent Q-network width
    mixing_embed_dim: int = 32     # mixing-network hidden width
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0     # exploration annealed linearly to epsilon_end
    epsilon_end: float = 0.05
    epsilon_anneal_episodes: int = 100
    buffer_size: int = 5000        # replay buffer capacity (transitions)
    batch_size: int = 32
    target_update_interval: int = 200  # learn steps between hard target syncs
    max_grad_norm: float = 10.0
    double_q: bool = True          # Double-DQN target (reduces overestimation)
    random_seed: int | None = None


@dataclass(frozen=True)
class TransfQmixSettings:
    """Hyper-parameters for TransfQMix (see rescue_sim.TransfQMix).

    Defaults follow Gallici et al. 2023, "TransfQMix: Transformers for
    Leveraging the Graph Structure of MARL Problems" (AAMAS). Both the agent
    network and the mixer are transformers, so the same parameters transfer to
    any number of agents/entities.
    """

    num_agents: int = 4
    view_radius: int = 2
    max_steps: int = 200
    d_model: int = 32              # transformer embedding width
    n_heads: int = 4               # attention heads (d_model must divide by this)
    n_agent_layers: int = 2        # encoder layers in the agent transformer
    n_mixer_layers: int = 1        # encoder layers in the mixer transformer
    ff_dim: int = 64               # feed-forward width inside the transformer
    mixing_embed_dim: int = 32     # monotonic mixing hidden width
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_anneal_episodes: int = 100
    buffer_size: int = 5000
    batch_size: int = 32
    target_update_interval: int = 200
    max_grad_norm: float = 10.0
    double_q: bool = True
    random_seed: int | None = None

