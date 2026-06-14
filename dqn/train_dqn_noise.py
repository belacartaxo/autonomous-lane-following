import sys
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')

import os
os.environ['WEBOTS_HOME'] = '/Applications/Webots.app/Contents'

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper
from env.lidar_noise_wrapper import LiDARNoiseWrapper

# ─── Configurações de Treino ───────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_600_000

NOISE_STD       = 0.1
DROPOUT_PROB    = 0.1
LIDAR_MAX_RANGE = 100.0

# ─── Diretórios ────────────────────────────────────────────────────────────────
LOG_DIR          = "./logs/dqn_noise/"
SAVE_DIR         = "./models/dqn_noise/"
BEST_MODEL_DIR   = "./models/dqn_noise/dqn_noise_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_noise_final")

os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(SAVE_DIR,       exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# ─── Ambiente (igual ao ppo_noise mas com DiscreteActionWrapper para DQN) ──────
base_env = WebotsVehicleEnv()

env = LiDARNoiseWrapper(
    base_env,
    noise_std=NOISE_STD,
    dropout_prob=DROPOUT_PROB,
    lidar_max_range=LIDAR_MAX_RANGE,
)
env = DiscreteActionWrapper(env)
env = TimeLimit(env, max_episode_steps=5000)
env = Monitor(env, LOG_DIR)

# ─── Carregar modelo existente ou criar novo ───────────────────────────────────
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"A carregar modelo DQN-Noise existente: {model_zip_path}")
    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
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

# ─── Callbacks ────────────────────────────────────────────────────────────────
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

# ─── Treino ───────────────────────────────────────────────────────────────────
print("A iniciar treino DQN com ruído LiDAR (C2)...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([checkpoint_callback, eval_callback]),
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(FINAL_MODEL_PATH)
print(f"Treino concluído. Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
print(f"Melhor modelo guardado em: {BEST_MODEL_DIR}best_model.zip")
