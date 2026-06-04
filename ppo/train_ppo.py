import os
import sys
import platform

if platform.system() == "Windows":
    WEBOTS_HOME = r"C:\Program Files\Webots"
elif platform.system() == "Darwin":  # macOS
    WEBOTS_HOME = "/Applications/Webots.app"
else:  # Linux
    WEBOTS_HOME = "/usr/local/webots"

os.environ["WEBOTS_HOME"] = WEBOTS_HOME

WEBOTS_PYTHON_PATH = os.path.join(
    WEBOTS_HOME,
    "lib",
    "controller",
    "python"
)

if WEBOTS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, WEBOTS_PYTHON_PATH)
    
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit
from env.webots_env import WebotsVehicleEnv
from env.webots_critical_env import WebotsCriticalVehicleEnv

# Configurações de Treino
TOTAL_TIMESTEPS = 1_600_000

# Diretórios
LOG_DIR = "./logs/ppo_critical/"
SAVE_DIR = "./models/ppo_critical/"
BEST_MODEL_DIR = "./models/ppo_critical/ppo_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_critical_final")

# Garantir que as pastas existem
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Inicializar Ambiente com Limite de Tempo
env = WebotsVehicleEnv()
#env = WebotsCriticalVehicleEnv()
env = TimeLimit(env, max_episode_steps=5000) # Limite de 5000 passos por episódio

model_zip_path = FINAL_MODEL_PATH + ".zip"

# Carregar modelo existente ou Criar novo
if os.path.exists(model_zip_path):
    print(f"A carregar modelo PPO existente: {model_zip_path}")
    model = PPO.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    print("Modelo carregado. O treino continua a partir do estado anterior.")
else:
    print("Nenhum modelo existente encontrado. A criar novo modelo PPO...")
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

# Callbacks para salvar progresso e o melhor modelo
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

# Iniciar/Continuar Treino
print("A iniciar/continuar treino PPO...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([checkpoint_callback, eval_callback]),
    progress_bar=True,
    reset_num_timesteps=False, # Mantém o histórico do otimizador para retreino
)

# Salvar modelo final
model.save(FINAL_MODEL_PATH)

print(f"Treino concluído. Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
