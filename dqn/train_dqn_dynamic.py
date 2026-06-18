import os
from webots_setup import setup_webots_path
setup_webots_path()


from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from env.webots_critical_env import WebotsCriticalVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper


# ─── Configurações de treino ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_600_000

# ─── Diretórios ────────────────────────────────────────────────────────────────
LOG_DIR = "./logs/dqn_dynamic/"
SAVE_DIR = "./models/dqn_dynamic/"
BEST_MODEL_DIR = "./models/dqn_dynamic/dqn_dynamic_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_dynamic_final")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ─── Ambiente C3: dynamic obstacles, sem noise ────────────────────────────────
def make_env():
    base_env = WebotsCriticalVehicleEnv()
    env = DiscreteActionWrapper(base_env)
    env = TimeLimit(env, max_episode_steps=5000)
    env = Monitor(env, LOG_DIR)
    return env


env = make_env()


# ─── Carregar modelo existente ou criar novo ──────────────────────────────────
model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"A carregar modelo DQN-Dynamic existente: {model_zip_path}")

    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print("Modelo carregado. O treino continua a partir do estado anterior.")

else:
    print("Nenhum modelo existente encontrado. A criar novo modelo DQN-Dynamic...")

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
    name_prefix="dqn_dynamic_lane",
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
print("A iniciar treino DQN com obstáculos dinâmicos (C3)...")

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
