import os
from webots_setup import setup_webots_path
setup_webots_path()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit

from env.webots_critical_env import WebotsCriticalVehicleEnv


# Training Configurations 
TOTAL_TIMESTEPS = 1_600_000


# Directories 
LOG_DIR = "./logs/ppo_dynamic/"
SAVE_DIR = "./models/ppo_dynamic/"
BEST_MODEL_DIR = "./models/ppo_dynamic/ppo_dynamic_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_dynamic_final")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# Environment C3: Dynamic obstacles, no noise 
env = WebotsCriticalVehicleEnv()
env = TimeLimit(env, max_episode_steps=5000)


# Load existing model or create a new one 
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"Loading existing PPO-Dynamic model: {model_zip_path}")

    model = PPO.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print("Model loaded. Training continues from the previous state.")

else:
    print("No existing model found. Creating new PPO-Dynamic model...")

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
    name_prefix="ppo_dynamic_lane",
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
print("Starting/continuing PPO training with dynamic obstacles (C3)...")

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
