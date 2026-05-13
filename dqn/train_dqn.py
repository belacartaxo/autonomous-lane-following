from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper
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

LOG_DIR = "./logs/dqn_baseline/"
SAVE_DIR = "./models/dqn_baseline/"
BEST_MODEL_DIR = "./models/dqn_baseline/dqn_best/"

FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_baseline_final")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

base_env = WebotsVehicleEnv()
env = DiscreteActionWrapper(base_env)

check_env(env, warn=True)

model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"A carregar modelo existente: {model_zip_path}")

    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print("Modelo carregado. O treino continua com replay buffer novo.")

else:
    print("Nenhum modelo existente encontrado. A criar novo modelo DQN...")

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

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="dqn_lane",
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
)

callbacks = CallbackList([
    checkpoint_callback,
    eval_callback,
])

print("A iniciar/continuar treino DQN...")

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