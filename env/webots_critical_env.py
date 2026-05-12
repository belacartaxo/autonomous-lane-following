from env.webots_env import WebotsVehicleEnv
from env.critical_obstacles import CriticalObstacleManager


class WebotsCriticalVehicleEnv(WebotsVehicleEnv):
    """
    Webots environment with critical dynamic obstacles.

    This extends the base WebotsVehicleEnv without modifying it.
    """

    def __init__(self):
        super().__init__()

        self.critical_obstacles = CriticalObstacleManager(
            supervisor=self.robot,
            vehicle_translation_field=self.translation_field
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.critical_obstacles.reset()

        return obs, info

    def step(self, action):
        real_action = self._denormalize_action(action)

        self._apply_action(real_action)

        self.critical_obstacles.step()

        if self.robot.step(self.timestep) == -1:
            return {}, 0.0, True, False, {}

        obs = self._get_observations()

        self._check_if_is_stuck(real_action)

        reward, done = self._compute_reward(obs, real_action)

        return obs, float(reward), done, False, {}