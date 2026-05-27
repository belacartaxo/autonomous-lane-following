import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
WEBOTS_PYTHON_PATH = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if WEBOTS_PYTHON_PATH not in sys.path:
    sys.path.append(WEBOTS_PYTHON_PATH)
from controller import Supervisor


class WebotsVehicleEnv(gym.Env):
    """
    Gymnasium environment for autonomous vehicle lane following in Webots.

    Action Space:
        Box(2,): normalized [steering, throttle/brake] in [-1, 1]

    Observation Space:
        Dict with:
        - "lidar": LiDAR distance measurements
        - "camera": RGB image
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

        self.action_low = np.array([-0.5, -0.3], dtype=np.float32)
        self.action_high = np.array([0.5, 0.6], dtype=np.float32)

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
        self.stuck_step_limit = 500
        self.stuck_distance_threshold = 0.001
        self.previous_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32
        )

        self.lost_line_steps = 0
        self.cumulative_lane_deviation = 0.0
        self.lane_deviation_count = 0
        self.max_lost_line_steps = 300

        self.current_step = 0
        self.max_episode_steps = 5000
        
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
        self.last_line_side = None  # "left" or "right"
        self.obstacle_was_near = False

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
        self.last_line_side = None  # "left" or "right"
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
        steer_angle = float(action[0])
        velocity = float(action[1]) * 30.0

        self.left_steering.setPosition(steer_angle)
        self.right_steering.setPosition(steer_angle)

        for wheel in self.wheels:
            wheel.setVelocity(velocity)

    def _extract_yellow_line_features(self, camera_image):
        """
        Detect the yellow lane line using the camera image.

        Returns:
            line_visible: whether the yellow line was detected
            lane_error: normalized horizontal error between line center and image center
            yellow_ratio: percentage of pixels classified as yellow
        """
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

        return True, lane_error, float(yellow_ratio)

    def _compute_reward(self, obs, action):
        """
        Compute reward using:
        - lane following
        - obstacle avoidance
        - smooth recovery back to the lane after avoidance
        """

        lidar = obs["lidar"]
        camera = obs["camera"]

        steering = float(action[0])
        throttle = float(action[1])

        line_visible, lane_error, yellow_ratio = (
            self._extract_yellow_line_features(camera)
        )

        current_line_side = None

        if line_visible:
            if lane_error < -0.15:
                current_line_side = "left"
            elif lane_error > 0.15:
                current_line_side = "right"
            else:
                current_line_side = "center"

        reward = 0.0
        done = False

        # LiDAR processing

        lidar_size = len(lidar)
        center = lidar_size // 2

        # Front obstacle detection
        front_window = lidar[
            max(0, center - 12):min(lidar_size, center + 12)
        ]

        # Left and right obstacle analysis
        left_window = lidar[
            max(0, center - 45):max(0, center - 15)
        ]

        right_window = lidar[
            min(lidar_size, center + 15):min(lidar_size, center + 45)
        ]

        front_distance = float(np.min(front_window))

        left_distance = (
            float(np.mean(left_window))
            if len(left_window) > 0 else 0.0
        )

        right_distance = (
            float(np.mean(right_window))
            if len(right_window) > 0 else 0.0
        )

        # Obstacle proximity states

        obstacle_near = front_distance < 10.0
        obstacle_close = front_distance < 6.0
        obstacle_very_close = front_distance < 3.0

        # Positive forward speed only
        forward_speed = max(0.0, throttle)


        # 1. Obstacle avoidance has priority
        if obstacle_near:
            if obstacle_near:
                # Save the line side only once, at the beginning of the avoidance maneuver.
                if not self.is_avoiding_obstacle:
                    if line_visible and current_line_side is not None:
                        self.last_line_side = current_line_side

            self.is_avoiding_obstacle = True

            if right_distance > left_distance:
                desired_steering = 1.0
            else:
                desired_steering = -1.0

            steering_alignment = steering * desired_steering

            reward += max(0.0, steering_alignment) * 4.0
            reward -= max(0.0, -steering_alignment) * 3.0

            if line_visible:
                self.lost_line_steps = 0
                center_reward = max(0.0, 1.0 - abs(lane_error))
                reward += center_reward * 0.3
            else:
                self.lost_line_steps += 1
                reward -= 0.1

            if obstacle_close:
                reward -= forward_speed * 5.0
                reward += max(0.0, 0.35 - forward_speed) * 3.0
            else:
                reward += forward_speed * 0.2

            if obstacle_very_close:
                reward -= 30.0
                reward -= forward_speed * 10.0

        # 2. No obstacle: follow lane
        elif line_visible:
            self.lost_line_steps = 0

            center_reward = max(0.0, 1.0 - abs(lane_error))

            reward += center_reward * 4.0
            reward += forward_speed * 0.8
            reward -= abs(lane_error) * 2.0
            reward -= abs(steering) * 0.2

            if abs(lane_error) < 0.15:
                reward += 2.0

            if self.is_avoiding_obstacle:
                if self.last_line_side is not None and current_line_side == self.last_line_side:
                    reward += 18.0
                elif abs(lane_error) < 0.25:
                    reward += 12.0
                else:
                    reward += 5.0

                self.is_avoiding_obstacle = False
                self.last_line_side = None

        # 3. No obstacle and no line
        else:
            self.lost_line_steps += 1

            # No obstacle and no line, but the car was avoiding an obstacle.
            # Now it should try to return to the side where the line was last seen.
            if self.is_avoiding_obstacle and self.last_line_side is not None:

                if self.last_line_side == "left":
                    desired_steering = -1.0
                elif self.last_line_side == "right":
                    desired_steering = 1.0
                else:
                    desired_steering = 0.0

                steering_alignment = steering * desired_steering

                # Small penalty because the line is still lost.
                reward -= 0.2

                # Reward steering back toward the last known line side.
                reward += max(0.0, steering_alignment) * 3.0

                # Penalize steering away from the last known line side.
                reward -= max(0.0, -steering_alignment) * 2.0

                # Encourage slow recovery.
                reward -= forward_speed * 0.5

            elif self.is_avoiding_obstacle:
                reward -= 0.2
                reward += abs(steering) * 0.5
                reward -= forward_speed * 0.5

            else:
                reward -= 5.0
                reward -= forward_speed * 1.0

        # Reverse penalty
        if throttle < -0.1:
            reward -= abs(throttle) * 0.5

        # Small step penalty to encourage efficiency
        reward -= 0.01

        # Episode termination conditions

        # Lost line for too long
        if self.lost_line_steps >= self.max_lost_line_steps:
            reward -= 100.0
            done = True

        # Vehicle stuck
        if self.stuck_step_count >= self.stuck_step_limit:
            reward -= 100.0
            done = True

        # Collision detected
        if np.min(lidar) < 2.0:
            reward -= 150.0
            done = True

        # Maximum episode duration
        self.current_step += 1

        if self.current_step >= self.max_episode_steps:
            done = True

        return reward, done

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

        reward, done = self._compute_reward(obs, real_action)

        collision = bool(np.min(obs["lidar"]) < 2.0)
        return obs, float(reward), done, False, {"collision": collision}
    
    def mean_lane_deviation(self):
        if self.lane_deviation_count == 0:
            return 1.0
        return self.cumulative_lane_deviation / self.lane_deviation_count


if __name__ == "__main__":
    env = WebotsVehicleEnv()
    print("Gymnasium environment created successfully!")

    obs, _ = env.reset()

    print(f"LiDAR shape: {obs['lidar'].shape}")
    print(f"Camera shape: {obs['camera'].shape}")