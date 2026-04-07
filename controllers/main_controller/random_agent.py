from main_controller import WebotsVehicleEnv
import numpy as np

# Initialize the environment
env = WebotsVehicleEnv()

print("Starting Reward Logic Test (Week 2)...")

for episode in range(3):
    # Reset environment at the start of each episode
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    while not (done or truncated or step_count >= 1000):
        # Sample a random action from the action space
        action = env.action_space.sample() 
        
        # Apply action and get environment feedback
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Print status every 10 steps to monitor reward behavior
        if step_count % 10 == 0:
            print(f"Eps {episode} | Step {step_count} | Instant Reward: {reward:.2f} | Total: {total_reward:.2f}")

    print(f"--- Episode {episode} finished | Steps: {step_count} | Final Reward: {total_reward:.2f} ---\n")