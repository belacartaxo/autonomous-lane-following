import gymnasium as gym
import numpy as np


class LiDARNoiseWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that injects noise into LiDAR observations.

    This wrapper simulates sensor imperfections by:
    - Adding Gaussian noise to all LiDAR rays
    - Randomly dropping (zeroing) some rays

    Other observation components (e.g., camera) remain unchanged.
    """

    def __init__(self, env, noise_std=0.1, dropout_prob=0.1, lidar_max_range=100.0):
        """Initialize noise parameters."""
        super().__init__(env)

        # Standard deviation of Gaussian noise applied to LiDAR readings
        self.noise_std = noise_std

        # Probability of dropping each LiDAR ray (simulating sensor failure)
        self.dropout_prob = dropout_prob

        # Maximum valid LiDAR range (used for clipping values)
        self.lidar_max_range = lidar_max_range

    def observation(self, obs):
        """Modify the observation by applying noise to the LiDAR component"""

        # Create a shallow copy of the observation dictionary
        noisy_obs = dict(obs)

        # Extract LiDAR readings
        lidar = obs["lidar"].copy()

        # Add zero-mean Gaussian noise to each LiDAR ray
        gaussian_noise = np.random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=lidar.shape
        ).astype(np.float32)

        lidar = lidar + gaussian_noise

        # Randomly zero some LiDAR rays with probability dropout_prob
        dropout_mask = np.random.rand(*lidar.shape) < self.dropout_prob
        lidar[dropout_mask] = 0.0

        # Ensure LiDAR values remain within valid physical limits
        lidar = np.clip(lidar, 0.0, self.lidar_max_range).astype(np.float32)

        # Replace LiDAR in observation
        noisy_obs["lidar"] = lidar

        return noisy_obs