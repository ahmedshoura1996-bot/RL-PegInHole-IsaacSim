import numpy as np


class RewardFunction:
    """
    Reward Function for Peg-in-Hole Assembly
    """

    def __init__(self):

        # Reward weights
        self.distance_weight = 1.0
        self.success_bonus = 100.0
        self.collision_penalty = -20.0

    def compute_reward(
        self,
        distance,
        success=False,
        collision=False,
    ):
        """
        Parameters
        ----------
        distance : float
            Distance between peg and hole.

        success : bool
            True if insertion succeeds.

        collision : bool
            True if unsafe collision occurs.
        """

        reward = 0.0

        # Encourage approaching the hole
        reward -= self.distance_weight * distance

        # Successful insertion
        if success:
            reward += self.success_bonus

        # Penalize collisions
        if collision:
            reward += self.collision_penalty

        return reward