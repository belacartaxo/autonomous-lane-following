import os

os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium.wrappers import TimeLimit
from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper

# Configurações de Treino
TOTAL_TIMESTEPS = 800_000

# Diretórios
LOG_DIR = "./logs/dqn_baseline/"
SAVE_DIR = "./models/dqn_baseline/"
BEST_MODEL_DIR = "./models/dqn_baseline/dqn_best/"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_baseline_final")

# Garantir que as pastas existem
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Inicializar Ambiente com Wrapper de Ações Discretas e Limite de Tempo
base_env = WebotsVehicleEnv()
env = DiscreteActionWrapper(base_env) # Wrapper de ações discretas para DQN [cite: 30]
env = TimeLimit(env, max_episode_steps=5000) # Limite de 5000 passos por episódio [cite: 64]

model_zip_path = FINAL_MODEL_PATH + ".zip"

# Carregar modelo existente ou Criar novo
if os.path.exists(model_zip_path):
    print(f"A carregar modelo DQN existente: {model_zip_path}")
    model = DQN.load(
        FINAL_MODEL_PATH,
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
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
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

# Callbacks para salvar progresso e o melhor modelo
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix="dqn_lane",
    save_replay_buffer=True,
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

# Iniciar/Continuar Treino
print("A iniciar/continuar treino DQN...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([checkpoint_callback, eval_callback]),
    progress_bar=True,
    reset_num_timesteps=False, # Mantém o histórico do otimizador para retreino
)

# Salvar modelo final
model.save(FINAL_MODEL_PATH)

print(f"Treino concluído. Modelo final guardado em: {FINAL_MODEL_PATH}.zip")
