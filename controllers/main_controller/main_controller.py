from env.webots_env import WebotsVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper
import numpy as np

env = WebotsVehicleEnv()
noise_wrapper = LiDARNoiseWrapper(env, noise_std=0.1, dropout_prob=0.1)

obs, _ = env.reset()
step_count = 0

while True:
    action = np.array([0.0, 0.5], dtype=np.float32)

    obs, reward, done, truncated, _ = env.step(action)

    # Create a noisy version of the same observation only for comparison
    noisy_obs = noise_wrapper.observation(obs)

    step_count += 1

    if step_count % 20 == 0:
        print(f"Step {step_count}")
        print("Original LiDAR:", obs["lidar"][:10])
        print("Noisy LiDAR:   ", noisy_obs["lidar"][:10])
        print("Original min:", np.min(obs["lidar"]))
        print("Noisy min:   ", np.min(noisy_obs["lidar"]))
        print("Reward:", reward)
        print("-----")

    if done or truncated:
        print("Reset\n")
        obs, _ = env.reset()
        step_count = 0