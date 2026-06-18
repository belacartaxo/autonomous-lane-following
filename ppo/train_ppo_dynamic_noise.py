import os
from webots_setup import setup_webots_path
setup_webots_path()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit

from env.webots_critical_env import WebotsCriticalVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper


# Training Configurations 
TOTAL_TIMESTEPS = 1_600_000  

# Directories 
LOG_DIR = "./logs/ppo_dynamic_noise/"
SAVE_DIR = "./models/ppo_dynamic_noise/"
BEST_MODEL_DIR = "./models/ppo_dynamic_noise/ppo_dynamic_noise_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_dynamic_noise_final")

# Corrected path pointing exactly to the base model without creating duplicate folders
PRETRAINED_MODEL_PATH = "./best_models/c5_ppo_dynamic_best.zip"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# Environment C4: Dynamic obstacles WITH noise 
base_env = WebotsCriticalVehicleEnv()

# Wrap the environment with the noise wrapper
env = LiDARNoiseWrapper(
    base_env,
    noise_std=0.1,
    dropout_prob=0.1
)

env = TimeLimit(env, max_episode_steps=5000)


# Load existing model or create a new one 
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    # If noise training had already started and was interrupted, continue from where it left off
    print(f"Loading existing PPO-Dynamic-Noise model: {model_zip_path}")
    model = PPO.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Noise model loaded. Training continues from the previous state.")

elif os.path.exists(PRETRAINED_MODEL_PATH):
    # If it's the first time running with noise, fetch the best_model from the clean training
    print(f"Starting training with noise from the clean base model: {PRETRAINED_MODEL_PATH}")

    custom_objects = {
        "learning_rate": 3e-5,  # 10x smaller than normal to adapt smoothly to the noise
        "clip_range": 0.1       # More restricted to avoid sudden policy changes
    }

    model = PPO.load(
        PRETRAINED_MODEL_PATH,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Base model loaded successfully in the new noisy environment!")

else:
    # Fallback in case the best_model is not found (should not happen)
    print("WARNING: No base model found. Creating model from scratch...")
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


# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="ppo_dynamic_noise_lane",
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


# Training 
print("Starting PPO training with dynamic obstacles and LiDAR noise...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([
        checkpoint_callback,
        eval_callback,
    ]),
    progress_bar=True,
    reset_num_timesteps=False,
)


# Save final model 
model.save(FINAL_MODEL_PATH)

print(f"Training completed. Final model saved at: {FINAL_MODEL_PATH}.zip")
print(f"Best model saved at: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")

env.close()
