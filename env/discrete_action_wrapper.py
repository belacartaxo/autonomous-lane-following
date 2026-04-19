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
            np.array([0.0, 0.6]),  # forward
            np.array([-0.4, 0.6]),  # strong left
            np.array([0.4, 0.6]),  # strong right
            np.array([-0.2, 0.6]),  # slight left
            np.array([0.2, 0.6]),  # slight right
            np.array([0.0, 0.0])  # stop
        ]

        self.action_space = spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]