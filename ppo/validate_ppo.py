import sys, os, ctypes

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
if not WEBOTS_HOME:
    raise EnvironmentError("Variável WEBOTS_HOME não definida.")

sys.path.insert(0, os.path.join(WEBOTS_HOME, "lib", "controller", "python"))
ctypes.CDLL(os.path.join(WEBOTS_HOME, "lib", "controller", "Controller.dll"))

from stable_baselines3 import PPO
from env.webots_env import WebotsVehicleEnv
import numpy as np
import json

MODEL_PATH = "models/old/ppo_baseline/ppo_baseline_final.zip"
N_EPISODES = 100  # usa 100 para avaliação final

env = WebotsVehicleEnv()
model = PPO.load(MODEL_PATH, env=env)

results = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    collisions = 0

    while not done and steps < 5000:
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

# Resumo
print("\n=== Validation Summary (C1 - Baseline) ===")
print(f"Success rate:   {np.mean([r['success'] for r in results])*100:.1f}%")
print(f"Avg reward:     {np.mean([r['total_reward'] for r in results]):.2f}")
print(f"Avg steps:      {np.mean([r['steps'] for r in results]):.1f}")
print(f"Avg collisions: {np.mean([r['collisions'] for r in results]):.2f}")

# Guarda resultados em JSON para usar depois
os.makedirs("./results", exist_ok=True)
with open("./results/c1_baseline_ppo.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResultados guardados em ./results/c1_baseline_ppo.json")