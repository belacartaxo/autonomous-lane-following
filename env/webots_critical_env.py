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

    def _compute_reward(self, obs, action):
        throttle = float(action[1])
        steering = float(action[0])

        dynamic_obstacle_active = (
            self.critical_obstacles.is_any_obstacle_active()
        )

        # Critical dynamic obstacle behavior
        if dynamic_obstacle_active:
            self.lost_line_steps = 0

            reward = 0.0
            done = False

            # Reward stopping for pedestrian/vehicle
            if abs(throttle) < 0.05:
                reward += 10.0

            # Strong penalty for continuing forward
            elif throttle > 0.1:
                reward -= throttle * 15.0

            # Small penalty for unstable behavior
            else:
                reward -= 2.0

            # Discourage unnecessary steering while waiting
            reward -= abs(steering) * 0.5

            return reward, done

        # Normal environment reward
        reward, done, termination_reason = super()._compute_reward(obs, action)

        # Encourage moving again after obstacle disappears
        if throttle > 0.1:
            reward += throttle * 1.0

        return reward, done

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