from env.webots_env import WebotsVehicleEnv

env = WebotsVehicleEnv()

obs, _ = env.reset()

while True:
    action = [0.0, 0.5] 
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        obs, _ = env.reset() # Reinicia se bater ou acabar o tempo