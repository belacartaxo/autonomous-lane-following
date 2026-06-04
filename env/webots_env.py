# File: webots_vehicle_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

import src.config as cfg

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
WEBOTS_PYTHON_PATH = os.path.join(WEBOTS_HOME, "lib", "controller", "python")

if WEBOTS_PYTHON_PATH not in sys.path:
    sys.path.append(WEBOTS_PYTHON_PATH)

from controller import Supervisor


class WebotsVehicleEnv(gym.Env):
    """
    Gymnasium environment for autonomous vehicle lane following in Webots.
    """

    def __init__(self):
        super(WebotsVehicleEnv, self).__init__()

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.lidar = self.robot.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        self.vehicle_node = self.robot.getSelf()
        self.translation_field = self.vehicle_node.getField("translation")
        self.rotation_field = self.vehicle_node.getField("rotation")

        self.initial_translation = self.translation_field.getSFVec3f()
        self.initial_rotation = self.rotation_field.getSFRotation()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.action_low = np.array(cfg.ACTION_LOW, dtype=np.float32)
        self.action_high = np.array(cfg.ACTION_HIGH, dtype=np.float32)

        camera_height = self.camera.getHeight()
        camera_width = self.camera.getWidth()
        lidar_width = self.lidar.getHorizontalResolution()

        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(
                low=0.0,
                high=100.0,
                shape=(lidar_width,),
                dtype=np.float32
            ),
            "camera": spaces.Box(
                low=0,
                high=255,
                shape=(camera_height, camera_width, 3),
                dtype=np.uint8
            )
        })

        self.left_steering = self.robot.getDevice("left_steer")
        self.right_steering = self.robot.getDevice("right_steer")

        self.wheels = []
        wheel_names = ["left_front_wheel", "right_front_wheel"]

        for name in wheel_names:
            wheel = self.robot.getDevice(name)
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)

        self.stuck_step_count = 0
        self.stuck_step_limit = cfg.STUCK_STEP_LIMIT
        self.stuck_distance_threshold = cfg.STUCK_DISTANCE_THRESHOLD
        self.previous_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32
        )

        self.lost_line_steps = 0
        self.cumulative_lane_deviation = 0.0
        self.lane_deviation_count = 0
        self.max_lost_line_steps = cfg.MAX_LOST_LINE_STEPS

        self.current_step = 0
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS

        self.viewpoint_node = self.robot.getFromDef("VIEWPOINT")

        if self.viewpoint_node is not None:
            self.viewpoint_position_field = self.viewpoint_node.getField("position")
            self.viewpoint_orientation_field = self.viewpoint_node.getField("orientation")

            self.initial_viewpoint_position = self.viewpoint_position_field.getSFVec3f()
            self.initial_viewpoint_orientation = self.viewpoint_orientation_field.getSFRotation()
        else:
            self.viewpoint_node = None
            self.viewpoint_position_field = None
            self.viewpoint_orientation_field = None
            self.initial_viewpoint_position = None
            self.initial_viewpoint_orientation = None

        self.resettable_def_names = (
            [f"BARREL_{i}" for i in range(1, 9)] +
            [f"CONE_{i}" for i in range(1, 20)] +
            ["PEDESTRIAN_1", "VEHICLE_1"]
        )

        self.resettable_nodes = []

        for def_name in self.resettable_def_names:
            node = self.robot.getFromDef(def_name)

            if node is None:
                continue

            translation_field = node.getField("translation")
            rotation_field = node.getField("rotation")

            self.resettable_nodes.append({
                "node": node,
                "translation_field": translation_field,
                "rotation_field": rotation_field,
                "initial_translation": translation_field.getSFVec3f(),
                "initial_rotation": rotation_field.getSFRotation(),
            })

        self.is_avoiding_obstacle = False
        self.last_line_side = None
        self.obstacle_was_near = False

        self.collision_step_count = 0
        self.collision_step_limit = 3

        self.steering_gain = 1.15

    def _denormalize_action(self, action):
        """Convert normalized action in [-1, 1] to real actuator ranges."""
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        return self.action_low + (action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        self.translation_field.setSFVec3f(self.initial_translation)
        self.rotation_field.setSFRotation(self.initial_rotation)
        self.vehicle_node.resetPhysics()

        for item in self.resettable_nodes:
            item["translation_field"].setSFVec3f(item["initial_translation"])
            item["rotation_field"].setSFRotation(item["initial_rotation"])
            item["node"].resetPhysics()

        if self.viewpoint_node is not None:
            self.viewpoint_position_field.setSFVec3f(self.initial_viewpoint_position)
            self.viewpoint_orientation_field.setSFRotation(self.initial_viewpoint_orientation)

        for wheel in self.wheels:
            wheel.setVelocity(0.0)

        self.is_avoiding_obstacle = False
        self.last_line_side = None
        self.obstacle_was_near = False

        self.robot.step(self.timestep)

        for wheel in self.wheels:
            wheel.setVelocity(0.0)

        self.stuck_step_count = 0
        self.previous_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32
        )

        self.lost_line_steps = 0
        self.current_step = 0
        self.cumulative_lane_deviation = 0.0
        self.lane_deviation_count = 0
        self.collision_step_count = 0

        obs = self._get_observations()
        return obs, {}

    def get_camera_image(self):
        """Process raw Webots camera data into an RGB numpy array."""
        image_data = self.camera.getImage()

        img = np.frombuffer(image_data, np.uint8).reshape(
            (self.camera.getHeight(), self.camera.getWidth(), 4)
        )

        img_rgb = img[:, :, [2, 1, 0]]
        return img_rgb

    def _get_observations(self):
        """Collect and clean sensor data for the RL agent."""
        lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_values = np.nan_to_num(
            lidar_values,
            nan=100.0,
            posinf=100.0,
            neginf=0.0
        )

        return {
            "lidar": lidar_values,
            "camera": self.get_camera_image()
        }

    def _apply_action(self, action):
        """Apply steering and throttle values to the vehicle actuators."""
        steer_angle = float(action[0]) * self.steering_gain
        steer_angle = float(np.clip(steer_angle, self.action_low[0], self.action_high[0]))

        velocity = float(action[1]) * cfg.MAX_VELOCITY

        self.left_steering.setPosition(steer_angle)
        self.right_steering.setPosition(steer_angle)

        for wheel in self.wheels:
            wheel.setVelocity(velocity)

    def _extract_yellow_line_features(self, camera_image):
        """Detect the yellow lane line using the camera image."""
        height, width, _ = camera_image.shape

        roi = camera_image[int(height * 0.45):, :, :]

        red = roi[:, :, 0]
        green = roi[:, :, 1]
        blue = roi[:, :, 2]

        yellow_mask = (
            (red > 120) &
            (green > 100) &
            (blue < 100) &
            (red > blue * 1.4) &
            (green > blue * 1.4)
        )

        yellow_pixels = np.argwhere(yellow_mask)
        yellow_ratio = yellow_pixels.shape[0] / yellow_mask.size

        if yellow_pixels.shape[0] == 0:
            return False, 1.0, 0.0

        line_center_x = np.mean(yellow_pixels[:, 1])
        image_center_x = width / 2.0

        lane_error = (line_center_x - image_center_x) / image_center_x
        lane_error = float(np.clip(lane_error, -1.0, 1.0))

        self.cumulative_lane_deviation += abs(lane_error)
        self.lane_deviation_count += 1

        return True, lane_error, float(yellow_ratio)

    def _get_front_lidar_info(self, lidar):
        """Return robust front LiDAR information, ignoring isolated noisy rays."""
        lidar_values = np.array(lidar, dtype=np.float32)

        lidar_size = len(lidar_values)
        center = lidar_size // 2

        front_window = lidar_values[
            max(0, center - 12):min(lidar_size, center + 12)
        ]

        valid_front_lidar = front_window[
            np.isfinite(front_window) &
            (front_window > 0.10)
        ]

        if len(valid_front_lidar) == 0:
            return float("inf"), 0

        min_front_lidar = float(np.min(valid_front_lidar))
        close_ray_count = int(np.sum(valid_front_lidar < cfg.COLLISION_DISTANCE))

        return min_front_lidar, close_ray_count

    def _get_front_lidar_distance(self, lidar):
        """Return only the robust minimum front LiDAR distance."""
        min_front_lidar, _ = self._get_front_lidar_info(lidar)
        return min_front_lidar

    def _compute_reward(self, obs, action):
        """
        Compute reward for three behaviours:
        1. follow the yellow line,
        2. avoid obstacles when needed,
        3. recover back to the line after avoidance.
        """
        lidar = obs["lidar"]
        camera = obs["camera"]

        steering = float(action[0])
        throttle = float(action[1])

        line_visible, lane_error, yellow_ratio = self._extract_yellow_line_features(camera)

        current_line_side = None

        if line_visible:
            if lane_error < -cfg.LINE_SIDE_THRESHOLD:
                current_line_side = "left"
            elif lane_error > cfg.LINE_SIDE_THRESHOLD:
                current_line_side = "right"
            else:
                current_line_side = "center"

        reward = 0.0
        done = False
        termination_reason = None

        lidar_size = len(lidar)
        center = lidar_size // 2

        left_window = lidar[
            max(0, center - 45):max(0, center - 15)
        ]

        right_window = lidar[
            min(lidar_size, center + 15):min(lidar_size, center + 45)
        ]

        front_distance, close_ray_count = self._get_front_lidar_info(lidar)

        left_distance = (
            float(np.mean(left_window))
            if len(left_window) > 0 else 0.0
        )

        right_distance = (
            float(np.mean(right_window))
            if len(right_window) > 0 else 0.0
        )

        obstacle_near = front_distance < cfg.OBSTACLE_NEAR_DISTANCE
        obstacle_close = front_distance < cfg.OBSTACLE_CLOSE_DISTANCE
        obstacle_very_close = front_distance < cfg.OBSTACLE_VERY_CLOSE_DISTANCE

        forward_speed = max(0.0, throttle)

        if obstacle_near:
            if not self.is_avoiding_obstacle:
                if line_visible and current_line_side is not None:
                    self.last_line_side = current_line_side

            self.is_avoiding_obstacle = True
            self.obstacle_was_near = True

            if right_distance > left_distance:
                desired_steering = 1.0
            else:
                desired_steering = -1.0

            steering_alignment = steering * desired_steering

            reward += max(0.0, steering_alignment) * cfg.AVOIDANCE_STEERING_REWARD
            reward -= max(0.0, -steering_alignment) * cfg.AVOIDANCE_WRONG_STEERING_PENALTY

            if abs(steering) < 0.20:
                reward -= cfg.LANE_LOW_STEERING_PENALTY

            if line_visible:
                self.lost_line_steps = 0
                center_reward = max(0.0, 1.0 - abs(lane_error))
                reward += center_reward * cfg.AVOIDANCE_LINE_CENTER_WEIGHT
            else:
                self.lost_line_steps += 1
                reward -= cfg.RECOVERY_LINE_LOST_PENALTY

            if obstacle_close:
                reward -= forward_speed * cfg.OBSTACLE_SPEED_PENALTY
                reward += max(
                    0.0,
                    cfg.OBSTACLE_SLOW_SPEED_TARGET - forward_speed
                ) * cfg.OBSTACLE_SLOW_SPEED_REWARD
            else:
                reward += forward_speed * cfg.AVOIDANCE_FORWARD_REWARD

            if obstacle_very_close:
                reward -= cfg.VERY_CLOSE_OBSTACLE_PENALTY
                reward -= forward_speed * cfg.VERY_CLOSE_SPEED_PENALTY

        elif line_visible:
            self.lost_line_steps = 0

            center_reward = max(0.0, 1.0 - abs(lane_error))
            desired_steering = float(np.clip(lane_error * 2.0, -1.0, 1.0))
            steering_alignment = steering * desired_steering

            reward += center_reward * cfg.CENTER_REWARD_WEIGHT
            reward += forward_speed * cfg.FORWARD_REWARD_WEIGHT
            reward -= abs(lane_error) * cfg.LANE_ERROR_PENALTY
            reward -= abs(steering) * cfg.STEERING_SMOOTHNESS_PENALTY

            reward += max(0.0, steering_alignment) * cfg.LANE_STEERING_REWARD
            reward -= max(0.0, -steering_alignment) * cfg.LANE_WRONG_STEERING_PENALTY

            if abs(lane_error) > 0.25 and abs(steering) < 0.10:
                reward -= cfg.LANE_LOW_STEERING_PENALTY

            if abs(lane_error) > 0.35:
                reward -= forward_speed * cfg.LANE_HIGH_ERROR_SPEED_PENALTY

            if abs(lane_error) < cfg.LINE_CENTER_THRESHOLD:
                reward += cfg.CENTERED_LINE_BONUS

            if self.is_avoiding_obstacle or self.obstacle_was_near:
                recovery_reward = max(0.0, 1.0 - abs(lane_error))
                reward += recovery_reward * cfg.RECOVERY_CENTER_WEIGHT

                if abs(lane_error) < cfg.RECOVERY_CENTER_THRESHOLD:
                    reward += cfg.RECOVERY_CENTER_BONUS
                    self.is_avoiding_obstacle = False
                    self.last_line_side = None
                    self.obstacle_was_near = False
                else:
                    reward += cfg.RECOVERY_PARTIAL_BONUS

        else:
            self.lost_line_steps += 1

            if self.is_avoiding_obstacle and self.last_line_side is not None:
                if self.last_line_side == "left":
                    desired_steering = -1.0
                elif self.last_line_side == "right":
                    desired_steering = 1.0
                else:
                    desired_steering = 0.0

                steering_alignment = steering * desired_steering

                reward -= cfg.RECOVERY_LINE_LOST_PENALTY
                reward += max(0.0, steering_alignment) * cfg.RECOVERY_STEERING_REWARD
                reward -= max(0.0, -steering_alignment) * cfg.RECOVERY_WRONG_STEERING_PENALTY
                reward -= forward_speed * cfg.RECOVERY_SPEED_PENALTY

            elif self.is_avoiding_obstacle:
                reward -= cfg.RECOVERY_LINE_LOST_PENALTY
                reward += abs(steering) * cfg.RECOVERY_NO_SIDE_STEERING_REWARD
                reward -= forward_speed * cfg.RECOVERY_SPEED_PENALTY

            else:
                reward -= cfg.LINE_LOST_PENALTY
                reward -= forward_speed * cfg.LINE_LOST_SPEED_PENALTY

        if throttle < -0.1:
            reward -= abs(throttle) * cfg.REVERSE_PENALTY

        reward -= cfg.STEP_PENALTY

        if self.lost_line_steps >= self.max_lost_line_steps:
            reward -= cfg.LOST_LINE_DONE_PENALTY
            done = True
            termination_reason = "lost_line"

        if self.stuck_step_count >= self.stuck_step_limit:
            reward -= cfg.STUCK_DONE_PENALTY
            done = True
            termination_reason = "stuck"

        if close_ray_count >= 3:
            self.collision_step_count += 1
        else:
            self.collision_step_count = 0

        if self.collision_step_count >= self.collision_step_limit:
            reward -= cfg.COLLISION_PENALTY
            done = True
            termination_reason = "collision"

        self.current_step += 1

        if self.current_step >= self.max_episode_steps:
            done = True
            termination_reason = "max_episode_steps"

        reward = float(np.clip(reward, -1.0, 1.0))

        return reward, done, termination_reason

    def _check_if_is_stuck(self, action):
        """Check whether the vehicle is stuck while trying to move."""
        current_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32
        )

        movement = np.linalg.norm(current_position - self.previous_position)

        dynamic_obstacle_active = (
            hasattr(self, "critical_obstacles")
            and self.critical_obstacles.is_any_obstacle_active()
        )

        if dynamic_obstacle_active:
            self.stuck_step_count = 0
        elif abs(action[1]) > 0.05 and movement < self.stuck_distance_threshold:
            self.stuck_step_count += 1
        else:
            self.stuck_step_count = 0

        self.previous_position = current_position

    def step(self, action):
        """Execute one environment step with the given action."""
        real_action = self._denormalize_action(action)

        self._apply_action(real_action)

        if self.robot.step(self.timestep) == -1:
            return {}, 0.0, True, False, {}

        obs = self._get_observations()

        self._check_if_is_stuck(real_action)

        reward, done, termination_reason = self._compute_reward(obs, real_action)

        min_front_lidar, close_ray_count = self._get_front_lidar_info(obs["lidar"])

        collision = bool(termination_reason == "collision")

        return obs, float(reward), done, False, {
            "collision": collision,
            "termination_reason": termination_reason,
            "close_front_rays": close_ray_count,
            "min_front_lidar": min_front_lidar,
            "lost_line_steps": self.lost_line_steps,
            "stuck_step_count": self.stuck_step_count,
            "collision_step_count": self.collision_step_count,
        }

    def mean_lane_deviation(self):
        """Return the mean lane deviation accumulated during the episode."""
        if self.lane_deviation_count == 0:
            return 1.0

        return self.cumulative_lane_deviation / self.lane_deviation_count


if __name__ == "__main__":
    env = WebotsVehicleEnv()
    print("Gymnasium environment created successfully!")

    obs, _ = env.reset()

    print(f"LiDAR shape: {obs['lidar'].shape}")
    print(f"Camera shape: {obs['camera'].shape}")