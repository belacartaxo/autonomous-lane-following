import os
from webots_setup import setup_webots_path
setup_webots_path()

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from gymnasium.wrappers import TimeLimit

from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper


# Training configurations
TOTAL_TIMESTEPS = 1_600_000

# Directories
LOG_DIR = "./logs/dqn_baseline/"
SAVE_DIR = "./models/dqn_baseline/"
BEST_MODEL_DIR = "./models/dqn_baseline/dqn_best/"

FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_baseline_final")
LATEST_REPLAY_BUFFER_PATH = os.path.join(SAVE_DIR, "latest_replay_buffer.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# Environment setup
def make_env():
    base_env = WebotsVehicleEnv()
    wrapped_env = DiscreteActionWrapper(base_env)
    wrapped_env = TimeLimit(wrapped_env, max_episode_steps=5000)
    return wrapped_env


# Custom callback to save the replay buffer periodically
class LatestReplayBufferCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            self.model.save_replay_buffer(self.save_path)

            if self.verbose > 0:
                print(f"Latest replay buffer saved at: {self.save_path}")

        return True


env = make_env()

# Load existing model or create a new one
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"Loading existing DQN model: {model_zip_path}")

    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    if os.path.exists(LATEST_REPLAY_BUFFER_PATH):
        print(f"Loading existing replay buffer: {LATEST_REPLAY_BUFFER_PATH}")
        model.load_replay_buffer(LATEST_REPLAY_BUFFER_PATH)

    print("Model loaded. Training continues from the previous state.")

else:
    print("No existing model found. Creating new DQN model...")

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )


# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="dqn_lane",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

latest_replay_buffer_callback = LatestReplayBufferCallback(
    save_freq=50_000,
    save_path=LATEST_REPLAY_BUFFER_PATH,
    verbose=1,
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

# Training execution
print("Starting/continuing DQN training...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([
        checkpoint_callback,
        latest_replay_buffer_callback,
        eval_callback,
    ]),
    progress_bar=True,
    reset_num_timesteps=False,
)

# Save final model and buffers
model.save(FINAL_MODEL_PATH)
model.save_replay_buffer(LATEST_REPLAY_BUFFER_PATH)

print(f"Training completed. Final model saved at: {FINAL_MODEL_PATH}.zip")
print(f"Latest replay buffer saved at: {LATEST_REPLAY_BUFFER_PATH}")
print(f"Best model saved at: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")

env.close()
