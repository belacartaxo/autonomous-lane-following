# Autonomous Lane Following with Reinforcement Learning

This repository contains the source code, trained models, evaluation scripts and experimental results for an autonomous lane-following project developed in Webots using reinforcement learning.

The objective is to train and evaluate an autonomous vehicle agent capable of following a lane, handling sensor noise, avoiding static obstacles and responding to critical dynamic obstacles. Two reinforcement learning algorithms are compared: Proximal Policy Optimization (PPO) and Deep Q-Network (DQN).

The scientific paper is submitted separately and should not be included inside the source-code `.zip`.

---

## 1. Project Overview

The project uses a simulated BMW X5 vehicle in Webots. The vehicle receives observations from a LiDAR sensor and a camera and outputs control actions for steering and longitudinal movement.

The work evaluates six experimental scenarios:

| Scenario | Description | Webots world | LiDAR noise | Obstacle type |
|---|---|---|---|---|
| C1 | Clean lane following | `worlds/city_default.wbt` | No | None |
| C2 | Lane following with LiDAR noise | `worlds/city_default.wbt` | Yes | None |
| C3 | Static obstacle avoidance | `worlds/city_static_obstacles.wbt` | No | Static |
| C4 | Static obstacle avoidance with LiDAR noise | `worlds/city_static_obstacles.wbt` | Yes | Static |
| C5 | Critical dynamic obstacle response | `worlds/city_dynamic_obstacles.wbt` | No | Dynamic |
| C6 | Critical dynamic obstacle response with LiDAR noise | `worlds/city_dynamic_obstacles.wbt` | Yes | Dynamic |

The main research question is whether PPO and DQN can learn robust autonomous driving behaviour across increasingly difficult simulation conditions, especially when sensor noise and dynamic obstacles are introduced.

---

## 2. Algorithms

### PPO

PPO is used as a continuous-control reinforcement learning algorithm. It directly predicts a two-dimensional action vector:

```text
[steering, throttle/brake]
```

The action values are mapped to the vehicle control range defined in `configs/env_config.py`.

### DQN

DQN requires a discrete action space. Therefore, the project includes a `DiscreteActionWrapper` that converts discrete actions into steering and throttle/brake commands.

For C1-C4, DQN uses six driving actions:

```text
0: forward
1: slight left
2: slight right
3: moderate left
4: moderate right
5: stronger left / avoidance
```

For C5-C6, three additional braking actions are enabled so that the vehicle can attempt to stop for critical dynamic obstacles:

```text
6: zero throttle / coast
7: soft brake
8: full brake
```

---

## 3. Sensor Noise

The noisy scenarios C2, C4 and C6 use the `LiDARNoiseWrapper`.

This wrapper simulates imperfect LiDAR measurements by:

1. adding zero-mean Gaussian noise to LiDAR readings;
2. randomly dropping LiDAR rays by setting them to zero;
3. clipping the resulting values to the valid sensor range.

The default noise parameters are:

```text
noise_std = 0.1
dropout_prob = 0.1
lidar_max_range = 100.0
```

These values are defined in `configs/evaluation_config.py` and in the corresponding training scripts.

---

## 4. Repository Structure

```text
.
├── best_models/
│   ├── c1_dqn_default_best.zip
│   ├── c1_ppo_default_best.zip
│   ├── c2_dqn_default_noise_best.zip
│   ├── c2_ppo_default_noise_best.zip
│   ├── c3_dqn_static_best.zip
│   ├── c3_ppo_static_best.zip
│   ├── c4_dqn_static_noise_best.zip
│   ├── c4_ppo_static_noise_best.zip
│   ├── c5_dqn_dynamic_best.zip
│   ├── c5_ppo_dynamic_best.zip
│   ├── c6_dqn_dynamic_noise_best.zip
│   └── c6_ppo_dynamic_noise_best.zip
│
├── configs/
│   ├── env_config.py
│   ├── evaluation_config.py
│   └── validation_config.py
│
├── dqn/
│   ├── train_dqn.py
│   ├── train_dqn_noise.py
│   ├── train_dqn_dynamic.py
│   └── train_dqn_dynamic_noise.py
│
├── env/
│   ├── webots_env.py
│   ├── webots_critical_env.py
│   ├── critical_obstacles.py
│   ├── discrete_action_wrapper.py
│   └── lidar_noise_wrapper.py
│
├── ppo/
│   ├── train_ppo.py
│   ├── train_ppo_noise.py
│   ├── train_ppo_dynamic.py
│   └── train_ppo_dynamic_noise.py
│
├── results/
│   ├── c1_dqn.json
│   ├── c1_ppo.json
│   ├── c2_dqn.json
│   ├── c2_ppo.json
│   ├── c3_dqn.json
│   ├── c3_ppo.json
│   ├── c4_dqn.json
│   ├── c4_ppo.json
│   ├── c5_dqn.json
│   ├── c5_ppo.json
│   ├── c6_dqn.json
│   ├── c6_ppo.json
│   └── plots_c*/
│
├── worlds/
│   ├── city_default.wbt
│   ├── city_static_obstacles.wbt
│   └── city_dynamic_obstacles.wbt
│
├── evaluate.py
├── validate.py
├── webots_setup.py
├── requirements.txt
└── README.md
```

---

## 5. Prerequisites

### Required Software

- Python 3.10 recommended
- Python 3.8+ should also work if the dependencies are compatible
- Webots installed locally
- Virtual environment support

### Tested Operating Systems

The project is intended to run on:

- Windows 10
- Ubuntu Linux 20.04.6 LTS

The helper file `webots_setup.py` automatically searches for Webots in common Windows and Linux installation paths. If Webots is installed elsewhere, set `WEBOTS_HOME` manually.

---

## 6. Installation

Clone or extract the repository and open a terminal in the project root folder.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Webots is not found automatically, define `WEBOTS_HOME`:

```bash
set WEBOTS_HOME=C:\Program Files\Webots
```

For a user-local Webots installation:

```bash
set WEBOTS_HOME=C:\Users\<USERNAME>\AppData\Local\Programs\Webots
```

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Webots is not found automatically, define `WEBOTS_HOME`:

```bash
export WEBOTS_HOME=/usr/local/webots
```

If needed, also expose the Webots controller library:

```bash
export PYTHONPATH=$PWD:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/controller:$LD_LIBRARY_PATH
```

---

## 7. Checking the Webots-Python Connection

Before running the experiments, check that Python can import the Webots controller module.

### Windows

```bash
python -c "from controller import Supervisor; print('Webots controller OK')"
```

### Linux

```bash
python3 -c "from controller import Supervisor; print('Webots controller OK')"
```

If the command prints:

```text
Webots controller OK
```

then the Python-Webots connection is correctly configured.

---

## 8. Important Webots Setup

Webots must be opened manually before running any training or evaluation script.

The Python scripts use the Webots external controller interface. They do not automatically open the `.wbt` world file.

Before each run:

1. Open Webots.
2. Open the correct world file.
3. Keep Webots running.
4. Run the Python command from the project root folder.

Use the following world for each scenario:

| Scenario | Required world |
|---|---|
| C1, C2 | `worlds/city_default.wbt` |
| C3, C4 | `worlds/city_static_obstacles.wbt` |
| C5, C6 | `worlds/city_dynamic_obstacles.wbt` |

Running a scenario with the wrong Webots world will produce invalid results.

---

## 9. Evaluating the Trained Models

The repository includes selected best models under `best_models/`.

Evaluation is performed with:

```bash
python evaluate.py --algo <ppo|dqn> --scenario <c1|c2|c3|c4|c5|c6>
```

The evaluation configuration is defined in:

```text
configs/evaluation_config.py
```

By default, each evaluation uses:

```text
N_EPISODES = 100
SEED = 42
```

The outputs are saved as JSON files in:

```text
results/
```

---

## 10. Evaluation Commands

### C1: Clean Lane Following

Open:

```text
worlds/city_default.wbt
```

Run:

```bash
python evaluate.py --algo ppo --scenario c1
python evaluate.py --algo dqn --scenario c1
```

### C2: Lane Following with LiDAR Noise

Open:

```text
worlds/city_default.wbt
```

Run:

```bash
python evaluate.py --algo ppo --scenario c2
python evaluate.py --algo dqn --scenario c2
```

### C3: Static Obstacle Avoidance

Open:

```text
worlds/city_static_obstacles.wbt
```

Run:

```bash
python evaluate.py --algo ppo --scenario c3
python evaluate.py --algo dqn --scenario c3
```

### C4: Static Obstacle Avoidance with LiDAR Noise

Open:

```text
worlds/city_static_obstacles.wbt
```

Run:

```bash
python evaluate.py --algo ppo --scenario c4
python evaluate.py --algo dqn --scenario c4
```

### C5: Critical Dynamic Obstacle Response

Open:

```text
worlds/city_dynamic_obstacles.wbt
```

Run:

```bash
python evaluate.py --algo ppo --scenario c5
python evaluate.py --algo dqn --scenario c5
```

### C6: Critical Dynamic Obstacle Response with LiDAR Noise

Open:

```text
worlds/city_dynamic_obstacles.wbt
```

Run:

```bash
python evaluate.py --algo ppo --scenario c6
python evaluate.py --algo dqn --scenario c6
```

### Manual Model Path

If needed, a model path can be passed manually:

```bash
python evaluate.py --algo ppo --scenario c5 --model best_models/c5_ppo_dynamic_best.zip
python evaluate.py --algo dqn --scenario c6 --model best_models/c6_dqn_dynamic_noise_best.zip
```

---

## 11. Generating Tables and Plots

After generating the evaluation JSON files, use:

```bash
python validate.py --scenario <c1|c2|c3|c4|c5|c6>
```

Examples:

```bash
python validate.py --scenario c1
python validate.py --scenario c2
python validate.py --scenario c3
python validate.py --scenario c4
python validate.py --scenario c5
python validate.py --scenario c6
```

If only one algorithm was evaluated for a scenario, specify the algorithm:

```bash
python validate.py --scenario c5 --algo ppo
python validate.py --scenario c5 --algo dqn
```

The script prints summary tables in the terminal and saves plots under the scenario-specific folders defined in:

```text
configs/validation_config.py
```

Example output folders:

```text
results/plots_c1/
results/plots_c2/
results/plots_c3/
results/plots_c4/
results/plots_c5/
results/plots_c6/
```

---

## 12. Training Models from Scratch

Training scripts are provided in the `ppo/` and `dqn/` directories.

All training scripts should be executed from the project root folder using `python -m`.

### C1: Clean Lane Following

Open:

```text
worlds/city_default.wbt
```

Run:

```bash
python -m ppo.train_ppo
python -m dqn.train_dqn
```

### C2: Lane Following with LiDAR Noise

Open:

```text
worlds/city_default.wbt
```

Run:

```bash
python -m ppo.train_ppo_noise
python -m dqn.train_dqn_noise
```

### C3: Static Obstacles

Open:

```text
worlds/city_static_obstacles.wbt
```

Run:

```bash
python -m ppo.train_ppo
python -m dqn.train_dqn
```

### C4: Static Obstacles with LiDAR Noise

Open:

```text
worlds/city_static_obstacles.wbt
```

Run:

```bash
python -m ppo.train_ppo_noise
python -m dqn.train_dqn_noise
```

### C5: Critical Dynamic Obstacles

Open:

```text
worlds/city_dynamic_obstacles.wbt
```

Run:

```bash
python -m ppo.train_ppo_dynamic
python -m dqn.train_dqn_dynamic
```

### C6: Critical Dynamic Obstacles with LiDAR Noise

Open:

```text
worlds/city_dynamic_obstacles.wbt
```

Run:

```bash
python -m ppo.train_ppo_dynamic_noise
python -m dqn.train_dqn_dynamic_noise
```

Training outputs are saved under:

```text
logs/
models/
```

The selected best models should then be copied to `best_models/` and referenced in `configs/evaluation_config.py`.

---

## 13. Main Evaluation Metrics

The evaluation code records scenario-dependent metrics, including:

- average cumulative reward;
- average lane deviation;
- episode length;
- termination reason;
- collision occurrence;
- success rate;
- LiDAR-noise robustness;
- static-obstacle avoidance performance;
- dynamic-obstacle stop behaviour;
- physical stop detection;
- action-based stop detection;
- resume success after the obstacle clears.

For C5 and C6, the most important safety-related metrics are:

```text
stop_success
resume_success
physical_stop
action_stop
```

These metrics evaluate whether the agent reacts correctly to a critical dynamic obstacle and whether it resumes movement afterwards.

---

## 14. Final Experimental Results

The final evaluation compares PPO and DQN across the six scenarios.

### C1-C4: Lane Following, Noise and Static Obstacles

| Scenario | Algorithm | Average lane deviation | Average reward |
|---|---|---:|---:|
| C1 | DQN | 0.014 | 3025 |
| C1 | PPO | 0.014 | 3027 |
| C2 | DQN | 0.106 | 2698 |
| C2 | PPO | 0.013 | 3034 |
| C3 | DQN | 0.090 | 2494 |
| C3 | PPO | 0.160 | 2183 |
| C4 | DQN | 0.051 | 2672 |
| C4 | PPO | 0.379 | 1244 |

### C5-C6: Dynamic Obstacle Response

| Scenario | Algorithm | Stop success | Resume success | Physical stop | Action stop |
|---|---|---:|---:|---:|---:|
| C5 | DQN | 0.0% | 0.0% | 0.0% | 45.9% |
| C5 | PPO | 100.0% | 0.0% | 65.3% | 100.0% |
| C6 | DQN | 0.0% | 0.0% | 0.0% | 1.3% |
| C6 | PPO | 100.0% | 0.0% | 64.9% | 100.0% |

The main observed limitation is that neither algorithm achieved successful resume behaviour after stopping for a dynamic obstacle. PPO learned a stronger stopping policy, while DQN showed limited braking behaviour in the dynamic scenarios, especially under LiDAR noise.

---

## 15. Reproducing the Experimental Pipeline

To reproduce the full evaluation pipeline:

1. Install Webots and Python dependencies.
2. Open the correct Webots world for the target scenario.
3. Run `evaluate.py` for PPO and DQN.
4. Repeat for all scenarios C1-C6.
5. Run `validate.py` for each scenario.
6. Use the JSON files and generated plots in `results/` for analysis.

Recommended order:

```bash
python evaluate.py --algo ppo --scenario c1
python evaluate.py --algo dqn --scenario c1
python validate.py --scenario c1

python evaluate.py --algo ppo --scenario c2
python evaluate.py --algo dqn --scenario c2
python validate.py --scenario c2

python evaluate.py --algo ppo --scenario c3
python evaluate.py --algo dqn --scenario c3
python validate.py --scenario c3

python evaluate.py --algo ppo --scenario c4
python evaluate.py --algo dqn --scenario c4
python validate.py --scenario c4

python evaluate.py --algo ppo --scenario c5
python evaluate.py --algo dqn --scenario c5
python validate.py --scenario c5

python evaluate.py --algo ppo --scenario c6
python evaluate.py --algo dqn --scenario c6
python validate.py --scenario c6
```

Remember to change the Webots world manually before switching between scenario groups.

---

## 16. Troubleshooting

### `ModuleNotFoundError: No module named 'controller'`

Python cannot find the Webots controller module.

Check that:

1. Webots is installed;
2. `WEBOTS_HOME` points to the correct Webots installation folder;
3. the Webots controller Python path is available.

Run:

```bash
python webots_setup.py
```

If Webots is correctly detected, the script should print the detected Webots path and confirm that the controller module can be imported.

### The Evaluation Runs, but the Scenario Seems Wrong

Check whether the correct Webots world is open.

The Python scripts do not automatically open the world file. The world must be selected manually in Webots before running the command.

### The Model Cannot Be Found

Check that the required model exists in `best_models/`.

The expected model paths are configured in:

```text
configs/evaluation_config.py
```

### Results Are Not Being Updated

Check whether the script has permission to write to:

```text
results/
```

Also check whether the terminal is being run from the project root folder.

---

## 17. Notes

- All commands should be executed from the project root folder.
- The trained models in `best_models/` are the models used for the final evaluation.
- The results in `results/` contain the JSON files and plots used to support the experimental analysis.
- The scientific paper is submitted separately through Moodle and is not part of the source-code archive.
- This project was developed for educational and academic purposes.

---

## 18. License

This project is for educational use only.
