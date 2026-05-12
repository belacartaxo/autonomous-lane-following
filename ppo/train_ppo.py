from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from env.webots_env import WebotsVehicleEnv
import sys
import os
import ctypes

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")

webots_python_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
webots_controller_dll = os.path.join(WEBOTS_HOME, "lib", "controller", "Controller.dll")

if webots_python_path not in sys.path:
    sys.path.append(webots_python_path)

ctypes.CDLL(webots_controller_dll)

TOTAL_TIMESTEPS = 800_000
LOG_DIR = "./logs/ppo_baseline/"
SAVE_DIR = "./models/ppo_baseline/"
BEST_MODEL_DIR = "./models/ppo_baseline/ppo_best/"

FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_baseline_final")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

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

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="ppo_lane",
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

callbacks = CallbackList([
    checkpoint_callback,
    eval_callback,
])

print("A iniciar treino PPO...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(FINAL_MODEL_PATH)

print("Treino concluído.")
print(f"Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
print(f"Melhor modelo guardado em: {BEST_MODEL_DIR}best_model.zip")