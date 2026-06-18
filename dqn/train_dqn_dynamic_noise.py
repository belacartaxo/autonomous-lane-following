import os
from webots_setup import setup_webots_path
setup_webots_path()


from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from env.webots_critical_env import WebotsCriticalVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper
from env.lidar_noise_wrapper import LiDARNoiseWrapper

# Training Configurations 
TOTAL_TIMESTEPS = 1_600_000  # Adjust as needed

# Directories 
# New exclusive directories for DQN training with noise
LOG_DIR = "./logs/dqn_dynamic_noise/"
SAVE_DIR = "./models/dqn_dynamic_noise/"
BEST_MODEL_DIR = "./models/dqn_dynamic_noise/dqn_dynamic_noise_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_dynamic_noise_final")

# Path pointing exactly to the best clean DQN model
PRETRAINED_MODEL_PATH = "./best_models/best_model_dqn_noise.zip"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ─── Environment C4: Dynamic obstacles WITH noise (DQN) ───────────────────────
def make_env():
    base_env = WebotsCriticalVehicleEnv()

    # Add noise to the sensors
    env = LiDARNoiseWrapper(base_env, noise_std=0.1, dropout_prob=0.1)

    # Convert continuous actions to discrete (Required for DQN)
    env = DiscreteActionWrapper(env)

    # Time limit and monitoring
    env = TimeLimit(env, max_episode_steps=5000)
    env = Monitor(env, LOG_DIR)

    return env


env = make_env()

# Load existing model or create a new one 
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    # Continue interrupted noise training
    print(f"Loading existing DQN-Dynamic-Noise model: {model_zip_path}")
    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Noise model loaded. Training continues from the previous state.")

elif os.path.exists(PRETRAINED_MODEL_PATH):
    # Start fine-tuning from the clean model
    print(f"Starting training with noise from the clean base model: {PRETRAINED_MODEL_PATH}")

    
    custom_objects = {
        "learning_rate": 1e-5,  # 10x slower to avoid forgetting the current policy
        "exploration_initial_eps": 0.15,  # Start exploring at only 15% (instead of 100%)
        "exploration_fraction": 0.1  # Decays rapidly to the final 5%
    }

    model = DQN.load(
        PRETRAINED_MODEL_PATH,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Base model loaded successfully in the new noisy environment!")

else:
    # 3. Fallback (training from scratch)
    print("WARNING: No base model found. Creating new DQN-Dynamic-Noise model...")
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        buffer_size=100_000,
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
    name_prefix="dqn_dynamic_noise_lane",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=os.path.join(LOG_DIR, "eval"),
    eval_freq=10_000,
    n_eval_episodes=3,
    deterministic=True,
    render=False,
    verbose=1,
)

# Training 
print("Starting DQN training with dynamic obstacles and LiDAR noise...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([
        checkpoint_callback,
        eval_callback,
    ]),
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(FINAL_MODEL_PATH)

print(f"Training completed. Final model saved at: {FINAL_MODEL_PATH}.zip")
print(f"Best model saved at: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")

env.close()
