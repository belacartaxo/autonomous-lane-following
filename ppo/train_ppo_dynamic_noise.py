import os
from webots_setup import setup_webots_path
setup_webots_path()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit

from env.webots_critical_env import WebotsCriticalVehicleEnv
from env.lidar_noise_wrapper import LiDARNoiseWrapper

# ─── Configurações de treino ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000  # Podes ajustar, como ele já tem base, pode precisar de menos tempo

# ─── Diretórios ────────────────────────────────────────────────────────────────
# Removido o "./ppo/" inicial porque o script já está dentro da pasta ppo!
LOG_DIR = "./logs/ppo_dynamic_noise/"
SAVE_DIR = "./models/ppo_dynamic_noise/"
BEST_MODEL_DIR = "./models/ppo_dynamic_noise/ppo_dynamic_noise_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_dynamic_noise_final")

# Caminho corrigido apontando exatamente para o modelo base sem criar pastas duplicadas
PRETRAINED_MODEL_PATH = "./best_models/c5_ppo_dynamic_best.zip"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# ─── Ambiente C4: dynamic obstacles COM noise ────────────────────────────────
base_env = WebotsCriticalVehicleEnv()

# Envolvemos o ambiente no wrapper de ruído
env = LiDARNoiseWrapper(
    base_env,
    noise_std=0.1,
    dropout_prob=0.1
)

env = TimeLimit(env, max_episode_steps=5000)

# ─── Carregar modelo existente ou criar novo ──────────────────────────────────
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    # 1. Se o treino com ruído já tinha começado e foi interrompido, continua de onde ficou
    print(f"A carregar modelo PPO-Dynamic-Noise existente: {model_zip_path}")
    model = PPO.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Modelo de ruído carregado. O treino continua a partir do estado anterior.")

elif os.path.exists(PRETRAINED_MODEL_PATH):
    # 2. Se é a primeira vez que corres com ruído, vai buscar o best_model do treino limpo
    print(f"A iniciar treino com ruído a partir do modelo base limpo: {PRETRAINED_MODEL_PATH}")

    # PROTEÇÃO MATEMÁTICA PARA O FINE-TUNING (Evita o colapso do approx_kl)
    custom_objects = {
        "learning_rate": 3e-5,  # 10x menor que o normal para adaptar suavemente ao ruído
        "clip_range": 0.1  # Mais restrito para evitar mudanças bruscas na política
    }

    model = PPO.load(
        PRETRAINED_MODEL_PATH,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Modelo base carregado com sucesso no novo ambiente ruidoso!")

else:
    # 3. Fallback caso não encontre o best_model (não deve acontecer)
    print("AVISO: Nenhum modelo base encontrado. A criar modelo do zero...")
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

# ─── Treino ───────────────────────────────────────────────────────────────────
print("A iniciar treino PPO com obstáculos dinâmicos e ruído no LiDAR...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([
        checkpoint_callback,
        eval_callback,
    ]),
    progress_bar=True,
    reset_num_timesteps=False,
)

# ─── Guardar modelo final ─────────────────────────────────────────────────────
model.save(FINAL_MODEL_PATH)

print(f"Treino concluído. Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
print(f"Melhor modelo guardado em: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")

env.close()
