from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    max_episode_steps: int = 500

    observation_dim: int = 20
    action_dim: int = 6

    success_distance: float = 0.005

    workspace_x = (-0.6, 0.6)
    workspace_y = (-0.6, 0.6)
    workspace_z = (0.0, 0.8)


@dataclass
class RewardConfig:
    distance_weight: float = 1.0
    alignment_weight: float = 2.0
    insertion_weight: float = 5.0

    collision_penalty: float = -20.0
    time_penalty: float = -0.01

    success_bonus: float = 100.0
    