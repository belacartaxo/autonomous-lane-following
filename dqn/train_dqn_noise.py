import os
import sys

# ─── Paths do projeto ──────────────────────────────────────────────────────────
# Este ficheiro está dentro da pasta dqn/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webots_setup import setup_webots_path
setup_webots_path()


# ─── Imports ───────────────────────────────────────────────────────────────────
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper
from env.lidar_noise_wrapper import LiDARNoiseWrapper


# ─── Configurações de treino ───────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_600_000

NOISE_STD = 0.1
DROPOUT_PROB = 0.1
LIDAR_MAX_RANGE = 100.0

MAX_EPISODE_STEPS = 5000


# ─── Diretórios ────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, "logs", "dqn_noise")
SAVE_DIR = os.path.join(BASE_DIR, "models", "dqn_noise")
BEST_MODEL_DIR = os.path.join(SAVE_DIR, "dqn_noise_best")

FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_noise_final")
REPLAY_BUFFER_PATH = os.path.join(SAVE_DIR, "dqn_noise_replay_buffer")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ─── Ambiente C2: Sensor Noise ─────────────────────────────────────────────────
base_env = WebotsVehicleEnv()

env = LiDARNoiseWrapper(
    base_env,
    noise_std=NOISE_STD,
    dropout_prob=DROPOUT_PROB,
    lidar_max_range=LIDAR_MAX_RANGE,
)

# DQN precisa de ações discretas
env = DiscreteActionWrapper(env)

# Mesmo limite usado no PPO
env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

# Monitor para logging
env = Monitor(env, LOG_DIR)


# ─── Carregar modelo existente ou criar novo ───────────────────────────────────
model_zip_path = FINAL_MODEL_PATH + ".zip"
replay_buffer_path = REPLAY_BUFFER_PATH + ".pkl"

if os.path.exists(model_zip_path):
    print(f"A carregar modelo DQN-Noise existente: {model_zip_path}")

    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    if os.path.exists(replay_buffer_path):
        print(f"A carregar replay buffer existente: {replay_buffer_path}")
        model.load_replay_buffer(REPLAY_BUFFER_PATH)
    else:
        print("Replay buffer não encontrado. O treino continua apenas a partir dos pesos do modelo.")

    print("Modelo carregado. O treino continua a partir do estado anterior.")

else:
    print("Nenhum modelo existente encontrado. A criar novo modelo DQN-Noise...")

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


# ─── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="dqn_noise_lane",
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

callbacks = CallbackList([
    checkpoint_callback,
    eval_callback,
])


# ─── Treino ────────────────────────────────────────────────────────────────────
print("A iniciar/continuar treino DQN com ruído LiDAR (C2)...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=False,
)


# ─── Guardar modelo final e replay buffer ──────────────────────────────────────
model.save(FINAL_MODEL_PATH)
model.save_replay_buffer(REPLAY_BUFFER_PATH)

print("Treino concluído.")
print(f"Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
print(f"Replay buffer guardado em: {REPLAY_BUFFER_PATH}.pkl")
print(f"Melhor modelo guardado em: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")
