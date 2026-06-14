# File: train_ppo_noise.py

import os
import sys
import platform

if platform.system() == "Windows":
    WEBOTS_HOME = r"C:\Program Files\Webots"
elif platform.system() == "Darwin":
    WEBOTS_HOME = "/Applications/Webots.app"
else:
    WEBOTS_HOME = "/usr/local/webots"

os.environ["WEBOTS_HOME"] = WEBOTS_HOME

WEBOTS_PYTHON_PATH = os.path.join(
    WEBOTS_HOME,
    "lib",
    "controller",
    "python"
)

if WEBOTS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, WEBOTS_PYTHON_PATH)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit

from env.webots_env import WebotsVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper


TOTAL_TIMESTEPS = 1_600_000

LOG_DIR = "./logs/ppo_baseline_noise/"
SAVE_DIR = "./models/ppo_baseline_noise/"
BEST_MODEL_DIR = "./models/ppo_baseline_noise/ppo_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_noise_final")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

base_env = WebotsVehicleEnv()
env = LiDARNoiseWrapper(
    base_env,
    noise_std=0.1,
    dropout_prob=0.1
)
env = TimeLimit(env, max_episode_steps=5000)

model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"Loading existing PPO noise model: {model_zip_path}")
    model = PPO.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Model loaded. Continuing training.")
else:
    print("No existing PPO noise model found. Creating a new one.")
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

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="ppo_noise_lane",
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=os.path.join(LOG_DIR, "eval"),
    eval_freq=10_000,
    n_eval_episodes=3,
    deterministic=True,
    render=False,
)

print("Starting/continuing PPO training with LiDAR noise...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([checkpoint_callback, eval_callback]),
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(FINAL_MODEL_PATH)

print(f"Training completed. Final model saved at: {FINAL_MODEL_PATH}.zip")
