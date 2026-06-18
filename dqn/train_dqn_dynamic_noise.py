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

# ─── Configurações de treino ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_600_000  # Podes ajustar conforme necessário

# ─── Diretórios ────────────────────────────────────────────────────────────────
# Novos diretórios exclusivos para o treino DQN com ruído
LOG_DIR = "./logs/dqn_dynamic_noise/"
SAVE_DIR = "./models/dqn_dynamic_noise/"
BEST_MODEL_DIR = "./models/dqn_dynamic_noise/dqn_dynamic_noise_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_dynamic_noise_final")

# Caminho apontando exatamente para o melhor modelo DQN limpo
PRETRAINED_MODEL_PATH = "./best_models/best_model_dqn_noise.zip"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ─── Ambiente C4: dynamic obstacles COM noise (DQN) ───────────────────────────
def make_env():
    base_env = WebotsCriticalVehicleEnv()

    # 1. Adicionamos o ruído aos sensores
    env = LiDARNoiseWrapper(base_env, noise_std=0.1, dropout_prob=0.1)

    # 2. Convertêmos as ações contínuas para discretas (Obrigatório para DQN)
    env = DiscreteActionWrapper(env)

    # 3. Limite de tempo e monitorização
    env = TimeLimit(env, max_episode_steps=5000)
    env = Monitor(env, LOG_DIR)

    return env


env = make_env()

# ─── Carregar modelo existente ou criar novo ──────────────────────────────────
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    # 1. Continua treino de ruído interrompido
    print(f"A carregar modelo DQN-Dynamic-Noise existente: {model_zip_path}")
    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Modelo de ruído carregado. O treino continua a partir do estado anterior.")

elif os.path.exists(PRETRAINED_MODEL_PATH):
    # 2. Inicia fine-tuning a partir do modelo limpo
    print(f"A iniciar treino com ruído a partir do modelo base limpo: {PRETRAINED_MODEL_PATH}")

    # PROTEÇÃO MATEMÁTICA PARA O FINE-TUNING DO DQN
    custom_objects = {
        "learning_rate": 1e-5,  # 10x mais lento para não esquecer a política atual
        "exploration_initial_eps": 0.15,  # Começa a explorar apenas 15% (em vez de 100%)
        "exploration_fraction": 0.1  # Decai rapidamente para os 5% finais
    }

    model = DQN.load(
        PRETRAINED_MODEL_PATH,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Modelo base carregado com sucesso no novo ambiente ruidoso!")

else:
    # 3. Fallback (treino do zero)
    print("AVISO: Nenhum modelo base encontrado. A criar novo modelo DQN-Dynamic-Noise...")
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

# ─── Treino ───────────────────────────────────────────────────────────────────
print("A iniciar treino DQN com obstáculos dinâmicos e ruído no LiDAR...")

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

print(f"Treino concluído. Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
print(f"Melhor modelo guardado em: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")

env.close()
