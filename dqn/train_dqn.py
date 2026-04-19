import sys, os

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
sys.path.insert(0, os.path.join(WEBOTS_HOME, "Contents", "lib", "controller", "python"))

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper

TOTAL_TIMESTEPS = 200_000
LOG_DIR  = "./logs/dqn_baseline/"
SAVE_DIR = "./models/dqn_baseline/"

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

base_env = WebotsVehicleEnv()
env = DiscreteActionWrapper(base_env)
check_env(env, warn=True)

model = DQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="dqn_lane",
)

print("A iniciar treino DQN baseline...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_cb,
    progress_bar=True,
)

model.save(os.path.join(SAVE_DIR, "dqn_baseline_final"))
print("Treino DQN concluído.")