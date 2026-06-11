import os
import sys
import platform

if platform.system() == "Windows":
    WEBOTS_HOME = r"C:\Program Files\Webots"
elif platform.system() == "Darwin":
    WEBOTS_HOME = "/Applications/Webots.app"
else:
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

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from gymnasium.wrappers import TimeLimit

from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper


TOTAL_TIMESTEPS = 1_600_000

LOG_DIR = "./logs/dqn_baseline/"
SAVE_DIR = "./models/dqn_baseline/"
BEST_MODEL_DIR = "./models/dqn_baseline/dqn_best/"

FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_baseline_final")
LATEST_REPLAY_BUFFER_PATH = os.path.join(SAVE_DIR, "latest_replay_buffer.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


def make_env():
    base_env = WebotsVehicleEnv()
    wrapped_env = DiscreteActionWrapper(base_env)
    wrapped_env = TimeLimit(wrapped_env, max_episode_steps=5000)
    return wrapped_env


class LatestReplayBufferCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            self.model.save_replay_buffer(self.save_path)

            if self.verbose > 0:
                print(f"Latest replay buffer saved at: {self.save_path}")

        return True


env = make_env()

model_zip_path = FINAL_MODEL_PATH + ".zip"

if os.path.exists(model_zip_path):
    print(f"A carregar modelo DQN existente: {model_zip_path}")

    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    if os.path.exists(LATEST_REPLAY_BUFFER_PATH):
        print(f"A carregar replay buffer existente: {LATEST_REPLAY_BUFFER_PATH}")
        model.load_replay_buffer(LATEST_REPLAY_BUFFER_PATH)

    print("Modelo carregado. O treino continua a partir do estado anterior.")

else:
    print("Nenhum modelo existente encontrado. A criar novo modelo DQN...")

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        buffer_size=50_000,
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

latest_replay_buffer_callback = LatestReplayBufferCallback(
    save_freq=50_000,
    save_path=LATEST_REPLAY_BUFFER_PATH,
    verbose=1,
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

print("A iniciar/continuar treino DQN...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([
        checkpoint_callback,
        latest_replay_buffer_callback,
        eval_callback,
    ]),
    progress_bar=True,
    reset_num_timesteps=False,
)

model.save(FINAL_MODEL_PATH)
model.save_replay_buffer(LATEST_REPLAY_BUFFER_PATH)

print(f"Treino concluído. Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
print(f"Replay buffer mais recente guardado em: {LATEST_REPLAY_BUFFER_PATH}")
print(f"Melhor modelo guardado em: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")

env.close()