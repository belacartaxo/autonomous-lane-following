import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts a continuous action space into a discrete one for DQN.

    For C1-C4, include_brake=False keeps the original driving actions.
    For C5-C6, include_brake=True adds explicit stop/brake actions so that
    DQN can learn to stop for critical dynamic obstacles.
    """

    def __init__(self, env, include_brake=False):
        super().__init__(env)

        self.include_brake = include_brake

        # Original driving / steering actions used in C1-C4
        self.actions = [
            np.array([0.0, 1.0], dtype=np.float32),      # 0: forward
            np.array([-0.1, 1.0], dtype=np.float32),     # 1: slight left
            np.array([0.1, 1.0], dtype=np.float32),      # 2: slight right
            np.array([-0.2, 0.95], dtype=np.float32),    # 3: moderate left
            np.array([0.2, 0.95], dtype=np.float32),     # 4: moderate right
            np.array([-0.35, 0.85], dtype=np.float32),   # 5: stronger left / avoidance
        ]

        # Extra actions needed for critical dynamic obstacles in C5-C6
        if include_brake:
            self.actions.extend([
                np.array([0.0, 0.0], dtype=np.float32),     # 6: zero throttle / coast
                np.array([0.0, -0.35], dtype=np.float32),   # 7: soft brake
                np.array([0.0, -1.0], dtype=np.float32),    # 8: full brake
            ])

        self.action_space = spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[int(action)]
