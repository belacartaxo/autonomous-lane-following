import numpy as np

from env.webots_env import WebotsVehicleEnv
from env.critical_obstacles import CriticalObstacleManager


class WebotsCriticalVehicleEnv(WebotsVehicleEnv):
    """
    Webots environment with critical dynamic obstacles.

    The agent must fully stop when a critical obstacle is active.
    """

    def __init__(self):
        super().__init__()

        self.critical_obstacles = CriticalObstacleManager(
            supervisor=self.robot,
            vehicle_translation_field=self.translation_field
        )

        self.not_stopped_step_count = 0
        self.max_not_stopped_steps = int(5000 / self.timestep)

        self.full_stop_speed_threshold = 0.005
        self.forward_throttle_threshold = 0.1

        self.stop_reward = 10.0
        self.not_stopped_penalty = 3.0
        self.forward_penalty_weight = 15.0
        self.failed_to_stop_penalty = 30.0
        self.critical_collision_penalty = 50.0

        self.previous_critical_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.critical_obstacles.reset()
        self.not_stopped_step_count = 0

        self.previous_critical_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32
        )

        return obs, info

    def _get_vehicle_movement(self):
        current_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32
        )

        movement = np.linalg.norm(current_position - self.previous_critical_position)
        self.previous_critical_position = current_position

        return float(movement)

    def _get_collision_lidar_info(self, lidar):
        lidar_values = np.array(lidar, dtype=np.float32)

        valid_lidar = lidar_values[
            np.isfinite(lidar_values) &
            (lidar_values > 0.10)
        ]

        if len(valid_lidar) == 0:
            return float("inf"), 0

        min_lidar_distance = float(np.min(valid_lidar))
        collision_ray_count = int(np.sum(valid_lidar < cfg.COLLISION_DISTANCE))

        return min_lidar_distance, collision_ray_count

    def _compute_reward(self, obs, action):
        throttle = float(action[1])
        steering = float(action[0])

        dynamic_obstacle_active = self.critical_obstacles.is_any_obstacle_active()

        if dynamic_obstacle_active:
            self.lost_line_steps = 0

            reward = 0.0
            done = False
            termination_reason = None

            vehicle_movement = self._get_vehicle_movement()
            vehicle_fully_stopped = vehicle_movement <= self.full_stop_speed_threshold

            min_lidar_distance, collision_ray_count = self._get_collision_lidar_info(
                obs["lidar"]
            )

            if collision_ray_count >= 3:
                self.collision_step_count += 1
            else:
                self.collision_step_count = 0

            if self.collision_step_count >= self.collision_step_limit:
                reward -= self.critical_collision_penalty
                done = True
                termination_reason = "collision_with_critical_obstacle"
                return reward, done, termination_reason

            if vehicle_fully_stopped:
                reward += self.stop_reward
                self.not_stopped_step_count = 0
            else:
                reward -= self.not_stopped_penalty
                self.not_stopped_step_count += 1

                if throttle > self.forward_throttle_threshold:
                    reward -= throttle * self.forward_penalty_weight

            reward -= abs(steering) * 0.5

            if self.not_stopped_step_count >= self.max_not_stopped_steps:
                reward -= self.failed_to_stop_penalty
                done = True
                termination_reason = "failed_to_fully_stop_for_critical_obstacle"

            if self.stuck_step_count >= self.stuck_step_limit:
                reward -= cfg.STUCK_DONE_PENALTY
                done = True
                termination_reason = "stuck"

            self.current_step += 1

            if self.current_step >= self.max_episode_steps:
                done = True
                termination_reason = "max_episode_steps"

            reward = float(np.clip(reward, -1.0, 1.0))

            return reward, done, termination_reason

        self.not_stopped_step_count = 0

        reward, done, termination_reason = super()._compute_reward(obs, action)

        if throttle > self.forward_throttle_threshold:
            reward += throttle * 1.0

        reward = float(np.clip(reward, -1.0, 1.0))

        return reward, done, termination_reason

    def step(self, action):
        real_action = self._denormalize_action(action)

        self._apply_action(real_action)

        self.critical_obstacles.step()

        if self.robot.step(self.timestep) == -1:
            return {}, 0.0, True, False, {}

        obs = self._get_observations()

        self._check_if_is_stuck(real_action)

        reward, done, termination_reason = self._compute_reward(obs, real_action)

        min_front_lidar, close_front_ray_count = self._get_front_lidar_info(obs["lidar"])
        min_lidar_distance, collision_ray_count = self._get_collision_lidar_info(
            obs["lidar"]
        )

        collision = bool(
            termination_reason in [
                "collision",
                "collision_with_critical_obstacle",
            ]
        )

        return obs, float(reward), done, False, {
            "collision": collision,
            "termination_reason": termination_reason,
            "critical_obstacle_active": self.critical_obstacles.is_any_obstacle_active(),
            "not_stopped_step_count": self.not_stopped_step_count,
            "max_not_stopped_steps": self.max_not_stopped_steps,
            "min_front_lidar": min_front_lidar,
            "close_front_rays": close_front_ray_count,
            "min_lidar_distance": min_lidar_distance,
            "collision_rays": collision_ray_count,
            "lost_line_steps": self.lost_line_steps,
            "stuck_step_count": self.stuck_step_count,
            "collision_step_count": self.collision_step_count,
        }