RESULTS_DIR = "./results"

SCENARIOS = {
    "c1": {
        "name": "No Obstacles",
        "plots_dir": "./results/plots_c1",

        "dqn_file": "c1_dqn.json",
        "ppo_file": "c1_ppo.json",

        "dynamic": False,
        "compare_with": None,
    },

    "c2": {
        "name": "No Obstacles + Noise",
        "plots_dir": "./results/plots_c2",

        "dqn_file": "c2_dqn.json",
        "ppo_file": "c2_ppo.json",

        "dynamic": False,
        "compare_with": "c1",
    },

    "c3": {
        "name": "Static Obstacles",
        "plots_dir": "./results/plots_c3",

        "dqn_file": "c3_dqn.json",
        "ppo_file": "c3_ppo.json",

        "dynamic": False,
        "compare_with": None,
    },

    "c4": {
        "name": "Static Obstacles + Noise",
        "plots_dir": "./results/plots_c4",

        "dqn_file": "c4_dqn.json",
        "ppo_file": "c4_ppo.json",

        "dynamic": False,
        "compare_with": "c3",
    },

    "c5": {
        "name": "Dynamic Obstacles",
        "plots_dir": "./results/plots_c5",

        "dqn_file": "c5_dqn.json",
        "ppo_file": "c5_ppo.json",

        "dynamic": True,
        "compare_with": "c3",
    },

    "c6": {
        "name": "Dynamic Obstacles + Noise",
        "plots_dir": "./results/plots_c6",

        "dqn_file": "c6_dqn.json",
        "ppo_file": "c6_ppo.json",

        "dynamic": True,
        "compare_with": "c5",
    },
}