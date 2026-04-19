import sys, os

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
sys.path.insert(0, os.path.join(WEBOTS_HOME, "Contents", "lib", "controller", "python"))

from stable_baselines3 import DQN
from env.webots_env import WebotsVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper
import numpy as np
import json

MODEL_PATH = "./models/dqn_noise/dqn_noise_final"
N_EPISODES = 10  # depois usa 100

# 🔥 Aqui aplicas noise
base_env = WebotsVehicleEnv()
env = LiDARNoiseWrapper(base_env, noise_std=0.1, dropout_prob=0.1)

model = DQN.load(MODEL_PATH, env=env)

results = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    collisions = 0

    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        total_reward += reward
        steps += 1

        if reward <= -20.0:
            collisions += 1

    success = collisions == 0

    results.append({
        "episode": ep,
        "total_reward": round(total_reward, 2),
        "steps": steps,
        "collisions": collisions,
        "success": success,
    })

    print(f"Ep {ep:2d} | Reward: {total_reward:7.2f} | Steps: {steps} | Collisions: {collisions} | Success: {success}")

# 📊 Resumo
print("\n=== Validation Summary (DQN Noise) ===")
print(f"Success rate:   {np.mean([r['success'] for r in results])*100:.1f}%")
print(f"Avg reward:     {np.mean([r['total_reward'] for r in results]):.2f}")
print(f"Avg steps:      {np.mean([r['steps'] for r in results]):.1f}")
print(f"Avg collisions: {np.mean([r['collisions'] for r in results]):.2f}")

# 💾 Guardar resultados
os.makedirs("./results", exist_ok=True)

with open("./results/c2_noise_dqn.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResultados guardados em ./results/c2_noise_dqn.json")