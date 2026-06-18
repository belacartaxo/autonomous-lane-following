import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts a continuous action space into a discrete one for DQN.
    """

    def __init__(self, env):
        super().__init__(env)

        # Define discrete actions
        self.actions = [
            np.array([0.0, 1.0]),
            np.array([-0.1, 1.0]),
            np.array([0.1, 1.0]),
            np.array([-0.2, 0.95]),
            np.array([0.2, 0.95]),
            np.array([-0.35, 0.85]),
        ]

        self.action_space = spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]
