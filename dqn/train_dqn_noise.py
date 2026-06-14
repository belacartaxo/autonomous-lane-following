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

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import TimeLimit

from env.webots_env import WebotsVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper
from env.discrete_action_wrapper import DiscreteActionWrapper


TOTAL_TIMESTEPS = 200_000

LOG_DIR = "./logs/dqn_noise/"
SAVE_DIR = "./models/dqn_noise/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_noise_final")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

base_env = WebotsVehicleEnv()

env = LiDARNoiseWrapper(
    base_env,
    noise_std=0.1,
    dropout_prob=0.1
)

env = DiscreteActionWrapper(env)
env = TimeLimit(env, max_episode_steps=5000)

model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"Loading existing DQN noise model: {model_zip_path}")
    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Model loaded. Continuing training.")
else:
    print("No existing DQN noise model found. Creating a new one.")
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="dqn_noise",
    save_replay_buffer=False,
)

print("Starting/continuing DQN training with LiDAR noise...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(FINAL_MODEL_PATH)

print(f"Training completed. Final model saved at: {FINAL_MODEL_PATH}.zip")