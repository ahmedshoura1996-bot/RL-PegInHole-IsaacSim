"""
Peg-in-Hole Environment

Master's Thesis:
Reinforcement Learning for Peg-in-Hole Assembly
using NVIDIA Isaac Sim
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PegInHoleEnv(gym.Env):
    """
    Reinforcement Learning environment for the Peg-in-Hole task.
    Compatible with Gymnasium and Stable-Baselines3.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Environment Dimensions
        self.observation_dim = 20
        self.action_dim = 6

        # Cartesian Control Action Space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment at the beginning of each episode.
        """
        super().reset(seed=seed)

        observation = np.zeros(self.observation_dim, dtype=np.float32)

        info = {}

        return observation, info

    def step(self, action):
        """
        Execute one environment step.
        """

        # Placeholder observation
        observation = np.zeros(self.observation_dim, dtype=np.float32)

        # Placeholder reward
        reward = 0.0

        # Episode termination flags
        terminated = False
        truncated = False

        # Additional information
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Rendering will be handled later by Isaac Sim.
        """
        pass

    def close(self):
        """
        Clean up environment resources.
        """
        pass