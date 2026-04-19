import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor

class WebotsVehicleEnv(gym.Env):
    """
    Gymnasium environment for autonomous vehicle lane following in Webots.

    This environment simulates a BMW X5 vehicle equipped with LiDAR and camera sensors.
    The agent learns to control steering and throttle to stay centered on the lane while
    avoiding collisions.

    Action Space:
        Box(2,): [steering (-0.5 to 0.5), throttle/brake (-1.0 to 1.0)]

    Observation Space:
        Dict with:
        - "lidar": Box(lidar_resolution,): distance measurements (0-100m)
        - "camera": Box(height, width, 3): RGB image (0-255)
    """

    def __init__(self):
        """Initialize the Webots vehicle environment."""
        super(WebotsVehicleEnv, self).__init__()

        # Initialize Webots Supervisor for robot control
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Enable sensors
        self.lidar = self.robot.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # Get vehicle node and initial pose fields
        self.vehicle_node = self.robot.getSelf()
        self.translation_field = self.vehicle_node.getField("translation")
        self.rotation_field = self.vehicle_node.getField("rotation")

        # Store initial position and rotation for resets
        self.initial_translation = self.translation_field.getSFVec3f()
        self.initial_rotation = self.rotation_field.getSFRotation()

        # Define action space: steering and throttle/brake
        # Normalized action space exposed to the RL agent
        # The agent always outputs values in [-1, 1] for both dimensions:
        #   action[0] → steering
        #   action[1] → throttle/brake
        # This normalization improves training stability (especially for PPO)
        self.action_space = spaces.Box( # TODO - PERGUNTAR PRO PROF SE ERA ESSA A NORMALIZAÇÃO QUE ELE TINHA COMENTADO
            low=-1.0,  # min steering, min throttle (full brake)
            high=1.0,   # max steering, max throttle
            shape=(2,),
            dtype=np.float32
        )

        # Real (physical) actuator limits used by the vehicle in Webots
        # These define how normalized actions are mapped to actual control values
        #   steering ∈ [-0.5, 0.5] radians
        #   throttle/brake ∈ [-1.0, 1.0]
        self.action_low = np.array([-0.5, -1.0], dtype=np.float32)
        self.action_high = np.array([0.5, 1.0], dtype=np.float32)

        # Get sensor dimensions dynamically
        camera_height = self.camera.getHeight()
        camera_width = self.camera.getWidth()
        lidar_width = self.lidar.getHorizontalResolution()

        # Define observation space with actual sensor dimensions
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0.0, high=100.0, shape=(lidar_width,), dtype=np.float32),
            "camera": spaces.Box(low=0, high=255, shape=(camera_height, camera_width, 3), dtype=np.uint8)
        })

        # Get steering actuators
        self.left_steering = self.robot.getDevice("left_steer")
        self.right_steering = self.robot.getDevice("right_steer")

        # Initialize wheel actuators (only front wheels for this setup)
        self.wheels = []
        wheel_names = ["left_front_wheel", "right_front_wheel"]

        for name in wheel_names:
            wheel = self.robot.getDevice(name)
            wheel.setPosition(float("inf"))  # Set to velocity control mode
            wheel.setVelocity(0.0)  # Initialize velocity to 0
            self.wheels.append(wheel)

    def _denormalize_action(self, action):
        """Convert normalized action in [-1, 1] to real actuator ranges"""
        action = np.clip(action, -1.0, 1.0) # Clip to ensure action stays within valid normalized bounds
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)  # Scale to the real range [action_low, action_high]

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state and return initial observations."""
        super().reset(seed=seed)

        # Reset vehicle position and rotation to initial values
        self.translation_field.setSFVec3f(self.initial_translation)
        self.rotation_field.setSFRotation(self.initial_rotation)

        # Reset physics to stop any residual motion
        self.vehicle_node.resetPhysics()
        self.robot.step(self.timestep)  # Advance one timestep to apply reset

        # Get initial observations
        obs = self._get_observations()
        return obs, {}

    def get_camera_image(self):
        """Process raw Webots camera data into an RGB numpy array."""
        image_data = self.camera.getImage()
        # Convert raw bytes to numpy array with BGRA format
        img = np.frombuffer(image_data, np.uint8).reshape(
            (self.camera.getHeight(), self.camera.getWidth(), 4)
        )
        # Convert BGRA to RGB by rearranging channels and removing alpha
        img_rgb = img[:, :, [2, 1, 0]]
        return img_rgb

    def _get_observations(self):
        """Collect and clean sensor data for the RL agent."""
        # Get LiDAR range data and handle infinite values
        lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_values = np.nan_to_num(lidar_values, posinf=100.0)  # Replace inf with 100.0

        return {
            "lidar": lidar_values,
            "camera": self.get_camera_image()
        }

    def _apply_action(self, action):
        """Apply steering and throttle values to the vehicle actuators."""
        steer_angle = float(action[0])
        velocity = float(action[1]) * 50.0  # Scale throttle to target velocity

        # Set steering angle for both left and right steering
        self.left_steering.setPosition(steer_angle)
        self.right_steering.setPosition(steer_angle)

        # Set velocity for all wheels
        for wheel in self.wheels:
            wheel.setVelocity(velocity)

    def _compute_reward(self, obs, action):
        """Compute the dense shaped reward signal based on lane following objectives."""
        lidar = obs["lidar"]
        throttle = action[1]

        # Forward speed reward: encourage moving forward
        reward = max(0.0, float(throttle)) * 0.5

        # Lateral deviation penalty: keep car centered using LiDAR symmetry
        # Compare left and right side distances
        side_diff = abs(np.mean(lidar[:10]) - np.mean(lidar[-10:]))
        reward -= side_diff * 0.3

        # Time penalty: discourage idling/staying still
        reward -= 0.01

        # Collision check: large negative reward for crashes
        done = False
        if np.min(lidar) < 0.45:  # Collision threshold
            reward = -20.0
            done = True

        return reward, done

    def step(self, action):
        """Execute one environment step with the given action."""
        # Convert normalized agent action to real control commands
        real_action = self._denormalize_action(action)

        # Apply the action to the vehicle
        self._apply_action(real_action)

        # Advance the simulation by one timestep
        if self.robot.step(self.timestep) == -1:
            # Simulation ended
            return {}, 0.0, True, False, {}

        # Get new observations
        obs = self._get_observations()

        # Compute reward and check for termination
        reward, done = self._compute_reward(obs, real_action)

        return obs, float(reward), done, False, {}

# Script entry point for environment testing
if __name__ == "__main__":
    # Create and test the environment
    env = WebotsVehicleEnv()
    print("Gymnasium environment created successfully!")
    obs, _ = env.reset()
    print(f"LiDAR shape: {obs['lidar'].shape}")  # Should match lidar resolution
    print(f"Camera shape: {obs['camera'].shape}")  # Should match camera dimensions