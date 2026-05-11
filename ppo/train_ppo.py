from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env.webots_env import WebotsVehicleEnv
import sys
import os
import ctypes

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")

webots_python_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
webots_controller_dll = os.path.join(WEBOTS_HOME, "lib", "controller", "Controller.dll")

if not os.path.exists(webots_python_path):
    raise FileNotFoundError(f"Webots Python path not found: {webots_python_path}")

if not os.path.exists(webots_controller_dll):
    raise FileNotFoundError(f"Controller.dll not found: {webots_controller_dll}")

if webots_python_path not in sys.path:
    sys.path.append(webots_python_path)

ctypes.CDLL(webots_controller_dll)

TOTAL_TIMESTEPS = 800_000
LOG_DIR = "./logs/ppo_baseline/"
SAVE_DIR = "./models/ppo_baseline/"
MODEL_PATH = os.path.join(SAVE_DIR, "ppo_baseline_final.zip")
# Exemplo alternativo:
# MODEL_PATH = os.path.join(SAVE_DIR, "ppo_lane_100000_steps.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

env = WebotsVehicleEnv()
check_env(env, warn=True)

if os.path.exists(MODEL_PATH):
    print(f"A carregar modelo existente de: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("Modelo existente não encontrado. A iniciar treino do zero.")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="ppo_lane",
)

print("A continuar treino PPO...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_cb,
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(os.path.join(SAVE_DIR, "ppo_baseline_continued"))
print("Treino concluído. Modelo guardado.")