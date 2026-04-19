import sys, os

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
sys.path.insert(0, os.path.join(WEBOTS_HOME, "Contents", "lib", "controller", "python"))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from env.webots_env import WebotsVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper

TOTAL_TIMESTEPS = 200_000
LOG_DIR  = "./logs/dqn_noise/"
SAVE_DIR = "./models/dqn_noise/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


base_env = WebotsVehicleEnv()
env = LiDARNoiseWrapper(base_env, noise_std=0.1, dropout_prob=0.1)

check_env(env, warn=True)

model = DQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-4,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="dqn_noise",
)

print("A iniciar treino DQN com noise...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb)

model.save(os.path.join(SAVE_DIR, "dqn_noise_final"))
print("Treino DQN com noise concluído.")