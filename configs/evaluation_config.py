MODELS_DIR = "./best_models"
RESULTS_DIR = "./results"
SEED = 42
N_EPISODES = 100
NOISE_STD = 0.1
DROPOUT_PROB = 0.1   

SCENARIOS = {
    "c1": {
        "name": "No Obstacles",
        "env_type": "base",
        "use_noise": False,
        "has_dynamic_obstacles": False,
        "models": {
            "ppo": f"{MODELS_DIR}/c1_ppo_default_best.zip",
            "dqn": f"{MODELS_DIR}/c1_dqn_default_best.zip",
        },
    },

    "c2": {
        "name": "No Obstacles + Noise",
        "env_type": "base",
        "use_noise": True,
        "has_dynamic_obstacles": False,
        "models": {
            "ppo": f"{MODELS_DIR}/c2_ppo_default_noise_best.zip",
            "dqn": f"{MODELS_DIR}/c2_dqn_default_noise_best.zip",
        },
    },

    "c3": {
        "name": "Static Obstacles",
        "env_type": "base",
        "use_noise": False,
        "has_dynamic_obstacles": False,
        "models": {
            "ppo": f"{MODELS_DIR}/c3_ppo_static_best.zip",
            "dqn": f"{MODELS_DIR}/c3_dqn_static_best.zip",
        },
    },

    "c4": {
        "name": "Static Obstacles + Noise",
        "env_type": "base",
        "use_noise": True,
        "has_dynamic_obstacles": False,
        "models": {
            "ppo": f"{MODELS_DIR}/c4_ppo_static_noise_best.zip",
            "dqn": f"{MODELS_DIR}/c4_dqn_static_noise_best.zip",
        },
    },

    "c5": {
        "name": "Dynamic Obstacles",
        "env_type": "critical",
        "use_noise": False,
        "has_dynamic_obstacles": True,
        "models": {
            "ppo": f"models/ppo_dynamic/ppo_dynamic_best/best_model.zip",
            "dqn": f"{MODELS_DIR}/c5_dqn_dynamic_best.zip",
        },
    },

    "c6": {
        "name": "Dynamic Obstacles + Noise",
        "env_type": "critical",
        "use_noise": True,
        "has_dynamic_obstacles": True,
        "models": {
            "ppo": f"{MODELS_DIR}/c6_ppo_dynamic_noise_best.zip",
            "dqn": f"{MODELS_DIR}/c6_dqn_dynamic_noise_best.zip",
        },
    },
}