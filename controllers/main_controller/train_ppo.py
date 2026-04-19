import sys, os, ctypes

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
if not WEBOTS_HOME:
    raise EnvironmentError("Variável WEBOTS_HOME não definida. Certifica-te que o Webots está instalado.")

sys.path.insert(0, os.path.join(WEBOTS_HOME, "lib", "controller", "python"))
ctypes.CDLL(os.path.join(WEBOTS_HOME, "lib", "controller", "Controller.dll"))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from main_controller import WebotsVehicleEnv
import os

TOTAL_TIMESTEPS = 200_000
LOG_DIR  = "./logs/ppo_baseline/"
SAVE_DIR = "./models/ppo_baseline/"
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

env = WebotsVehicleEnv()
check_env(env, warn=True)

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

print("A iniciar treino PPO baseline...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_cb,
    progress_bar=True,
)

model.save(os.path.join(SAVE_DIR, "ppo_baseline_final"))
print("Treino concluído. Modelo guardado.")