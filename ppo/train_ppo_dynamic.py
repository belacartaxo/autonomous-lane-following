import os
import sys
import platform

# ─── Webots path ───────────────────────────────────────────────────────────────
if platform.system() == "Windows":
    WEBOTS_HOME = r"C:\Program Files\Webots"
    WEBOTS_PYTHON_PATH = os.path.join(WEBOTS_HOME, "lib", "controller", "python")

elif platform.system() == "Darwin":  # macOS
    WEBOTS_HOME = "/Applications/Webots.app"
    WEBOTS_PYTHON_PATH = os.path.join(
        WEBOTS_HOME,
        "Contents",
        "lib",
        "controller",
        "python"
    )

else:  # Linux
    WEBOTS_HOME = "/usr/local/webots"
    WEBOTS_PYTHON_PATH = os.path.join(WEBOTS_HOME, "lib", "controller", "python")

os.environ["WEBOTS_HOME"] = WEBOTS_HOME

if WEBOTS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, WEBOTS_PYTHON_PATH)


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit

from env.webots_critical_env import WebotsCriticalVehicleEnv


# ─── Configurações de treino ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_600_000


# ─── Diretórios ────────────────────────────────────────────────────────────────
LOG_DIR = "./logs/ppo_dynamic/"
SAVE_DIR = "./models/ppo_dynamic/"
BEST_MODEL_DIR = "./models/ppo_dynamic/ppo_dynamic_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_dynamic_final")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ─── Ambiente C3: dynamic obstacles, sem noise ────────────────────────────────
env = WebotsCriticalVehicleEnv()
env = TimeLimit(env, max_episode_steps=5000)


# ─── Carregar modelo existente ou criar novo ──────────────────────────────────
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"A carregar modelo PPO-Dynamic existente: {model_zip_path}")

    model = PPO.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print("Modelo carregado. O treino continua a partir do estado anterior.")

else:
    print("Nenhum modelo existente encontrado. A criar novo modelo PPO-Dynamic...")

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


# ─── Treino ───────────────────────────────────────────────────────────────────
print("A iniciar/continuar treino PPO com obstáculos dinâmicos (C3)...")

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