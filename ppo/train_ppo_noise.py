
# ─── Ajusta este path para o teu Mac ───────────────────────────────────────────
import sys
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')
import os
os.environ['WEBOTS_HOME'] = '/Applications/Webots.app/Contents'
# ────────────────────────────────────────────────────────────────────────────────

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit

from env.webots_env import WebotsVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper

# ─── Configurações de Treino ───────────────────────────────────────────────────
TOTAL_TIMESTEPS = 800_000

# Parâmetros de ruído (mesmos descritos no proposal)
NOISE_STD       = 0.1   # desvio padrão do ruído Gaussiano
DROPOUT_PROB    = 0.1   # probabilidade de dropout por raio LiDAR
LIDAR_MAX_RANGE = 100.0 # alcance máximo do sensor (metros)

# ─── Diretórios ────────────────────────────────────────────────────────────────
LOG_DIR        = "./logs/ppo_noise/"
SAVE_DIR       = "./models/ppo_noise/"
BEST_MODEL_DIR = "./models/ppo_noise/ppo_noise_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_noise_final")

os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(SAVE_DIR,       exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# ─── Ambiente ─────────────────────────────────────────────────────────────────
# Ambiente base (cidade com obstáculos estáticos, sem obstáculos dinâmicos)
base_env = WebotsVehicleEnv()

# Wrapper de ruído LiDAR — simula degradação do sensor (C2)
env = LiDARNoiseWrapper(
    base_env,
    noise_std=NOISE_STD,
    dropout_prob=DROPOUT_PROB,
    lidar_max_range=LIDAR_MAX_RANGE,
)

# Limite de passos por episódio (igual ao baseline C1)
env = TimeLimit(env, max_episode_steps=5000)

# ─── Carregar modelo existente ou criar novo ───────────────────────────────────
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"A carregar modelo PPO-Noise existente: {model_zip_path}")
    model = PPO.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Modelo carregado. O treino continua a partir do estado anterior.")
else:
    print("Nenhum modelo existente encontrado. A criar novo modelo PPO-Noise...")
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

# ─── Callbacks ────────────────────────────────────────────────────────────────
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="ppo_noise_lane",
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

# ─── Treino ───────────────────────────────────────────────────────────────────
print("A iniciar/continuar treino PPO com ruído LiDAR (C2)...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([checkpoint_callback, eval_callback]),
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(FINAL_MODEL_PATH)
print(f"Treino concluído. Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
