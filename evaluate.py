"""
Cenários disponíveis: c1-c6
python evaluate.py --algo ppo --scenario c1
python evaluate.py --algo dqn --scenario c1
"""
import argparse
import json
import os
import sys
import time
import platform
from collections import Counter

import numpy as np

import configs.env_config as cfg
from configs.evaluation_config import (
    SEED,
    N_EPISODES,
    NOISE_STD,
    DROPOUT_PROB,
    RESULTS_DIR,
    SCENARIOS,
)


def setup_webots_path():
    if platform.system() == "Windows":
        webots_home = r"C:\Program Files\Webots"
    elif platform.system() == "Darwin":
        webots_home = "/Applications/Webots.app"
    else:
        webots_home = "/usr/local/webots"

    os.environ["WEBOTS_HOME"] = webots_home

    webots_python_path = os.path.join(
        webots_home,
        "lib",
        "controller",
        "python",
    )

    if webots_python_path not in sys.path:
        sys.path.insert(0, webots_python_path)


def build_env(algo: str, scenario_config: dict):
    from env.webots_env import WebotsVehicleEnv
    from env.webots_critical_env import WebotsCriticalVehicleEnv
    from env.lidar_noise_wrapper import LiDARNoiseWrapper
    from env.discrete_action_wrapper import DiscreteActionWrapper

    if scenario_config["env_type"] == "critical":
        env = WebotsCriticalVehicleEnv()
    else:
        env = WebotsVehicleEnv()

    if scenario_config["use_noise"]:
        env = LiDARNoiseWrapper(
            env,
            noise_std=NOISE_STD,
            dropout_prob=DROPOUT_PROB,
        )

    if algo.lower() == "dqn":
        env = DiscreteActionWrapper(env)

    return env


def get_base_env(env):
    base = env

    while hasattr(base, "env"):
        base = base.env

    return base


def load_model(algo: str, model_path: str, env):
    from stable_baselines3 import DQN, PPO

    model_class = DQN if algo.lower() == "dqn" else PPO
    return model_class.load(model_path, env=env)


def is_dynamic_obstacle_active(base_env):
    if not hasattr(base_env, "critical_obstacles"):
        return False

    manager = base_env.critical_obstacles

    if not hasattr(manager, "is_any_obstacle_active"):
        return False

    return bool(manager.is_any_obstacle_active())


def is_stop_or_brake_action(algo: str, action):
    if algo.lower() == "dqn":
        try:
            return int(action) == 5
        except Exception:
            return False

    action_array = np.array(action).flatten()

    if len(action_array) < 2:
        return False

    throttle = float(action_array[1])

    return throttle <= 0.05


def detect_collision(obs, info):
    if info.get("collision", False):
        return True

    if isinstance(obs, dict) and "lidar" in obs:
        try:
            return bool(np.min(obs["lidar"]) < cfg.COLLISION_DISTANCE)
        except Exception:
            return False

    return False


def get_termination(collision: bool, steps: int):
    if collision:
        return "collision"

    if steps >= cfg.MAX_EPISODE_STEPS:
        return "max_steps"

    return "early_end"


def run(algo: str, scenario: str, model_path: str | None = None):
    setup_webots_path()

    scenario = scenario.lower()
    algo = algo.lower()

    scenario_config = SCENARIOS[scenario]

    if model_path is None:
        model_path = scenario_config["models"][algo]

    np.random.seed(SEED)

    print(f"\n{'=' * 70}")
    print(f"  Evaluation: {scenario.upper()} - {scenario_config['name']}")
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Model     : {model_path}")
    print(f"  Episodes  : {N_EPISODES}")
    print(f"  Max steps : {cfg.MAX_EPISODE_STEPS}")

    if scenario_config["use_noise"]:
        print(f"  Noise std : {NOISE_STD}")
        print(f"  Dropout   : {DROPOUT_PROB}")

    print(f"{'=' * 70}\n")

    env = build_env(algo, scenario_config)
    model = load_model(algo, model_path, env)
    base_env = get_base_env(env)

    results = []
    start_time = time.time()

    for episode in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + episode)

        done = False
        total_reward = 0.0
        steps = 0
        collision = False

        obstacle_active_steps = 0
        stopped_steps = 0
        stopped_when_obstacle = 0

        episode_start = time.time()

        while not done and steps < cfg.MAX_EPISODE_STEPS:
            action, _ = model.predict(obs, deterministic=True)

            obstacle_active = is_dynamic_obstacle_active(base_env)
            stop_or_brake = is_stop_or_brake_action(algo, action)

            obs, reward, done, _, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            if scenario_config["has_dynamic_obstacles"]:
                if obstacle_active:
                    obstacle_active_steps += 1

                if stop_or_brake:
                    stopped_steps += 1

                if obstacle_active and stop_or_brake:
                    stopped_when_obstacle += 1

            if detect_collision(obs, info):
                collision = True

        lane_deviation = (
            base_env.mean_lane_deviation()
            if hasattr(base_env, "mean_lane_deviation")
            else float("nan")
        )

        success = (not collision) and (steps >= cfg.MAX_EPISODE_STEPS)
        non_collision = not collision
        termination = get_termination(collision, steps)

        obstacle_active_rate = (
            obstacle_active_steps / steps * 100
            if steps > 0
            else 0.0
        )

        stopped_rate = (
            stopped_steps / steps * 100
            if steps > 0
            else 0.0
        )

        stopped_when_obstacle_rate = (
            stopped_when_obstacle / obstacle_active_steps * 100
            if obstacle_active_steps > 0
            else 0.0
        )

        episode_time = time.time() - episode_start

        result = {
            "episode": episode,
            "success": success,
            "non_collision": non_collision,
            "collision": collision,
            "termination": termination,
            "steps": steps,
            "total_reward": round(total_reward, 3),
            "lane_deviation": round(float(lane_deviation), 4),
            "episode_time_s": round(episode_time, 2),
        }

        if scenario_config["has_dynamic_obstacles"]:
            result.update(
                {
                    "obstacle_active_steps": obstacle_active_steps,
                    "obstacle_active_rate_pct": round(obstacle_active_rate, 2),
                    "stopped_steps": stopped_steps,
                    "stopped_rate_pct": round(stopped_rate, 2),
                    "stopped_when_obstacle": stopped_when_obstacle,
                    "stopped_when_obstacle_rate_pct": round(
                        stopped_when_obstacle_rate,
                        2,
                    ),
                }
            )

        results.append(result)

        if success:
            status = "✓ SUCCESS"
        elif collision:
            status = "✗ COLLISION"
        else:
            status = "✗ EARLY END"

        print(
            f"Ep {episode:3d} | {status:11s} | "
            f"Steps: {steps:5d} | "
            f"Reward: {total_reward:8.1f} | "
            f"LaneDev: {lane_deviation:.3f}"
        )

    total_time = time.time() - start_time

    success_rate = np.mean([r["success"] for r in results]) * 100
    non_collision_rate = np.mean([r["non_collision"] for r in results]) * 100
    collision_rate = np.mean([r["collision"] for r in results]) * 100
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_lane_deviation = np.nanmean([r["lane_deviation"] for r in results])
    termination_counts = Counter(r["termination"] for r in results)

    successful_steps = [r["steps"] for r in results if r["success"]]
    avg_lap_steps = (
        float(np.mean(successful_steps))
        if successful_steps
        else None
    )

    summary = {
        "algo": algo.upper(),
        "condition": scenario.upper(),
        "condition_name": scenario_config["name"],
        "model_path": model_path,
        "n_episodes": N_EPISODES,
        "max_steps": cfg.MAX_EPISODE_STEPS,
        "seed": SEED,
        "use_noise": scenario_config["use_noise"],
        "noise_std": NOISE_STD if scenario_config["use_noise"] else None,
        "dropout_prob": DROPOUT_PROB if scenario_config["use_noise"] else None,
        "success_rate_pct": round(float(success_rate), 2),
        "non_collision_rate_pct": round(float(non_collision_rate), 2),
        "collision_rate_pct": round(float(collision_rate), 2),
        "avg_steps": round(float(avg_steps), 1),
        "avg_lap_steps": round(avg_lap_steps, 1) if avg_lap_steps else None,
        "avg_reward": round(float(avg_reward), 2),
        "avg_lane_deviation": round(float(avg_lane_deviation), 4),
        "termination_counts": dict(termination_counts),
        "total_time_s": round(float(total_time), 1),
        "episodes": results,
    }

    if scenario_config["has_dynamic_obstacles"]:
        summary.update(
            {
                "avg_obstacle_active_steps": round(
                    float(np.mean([r["obstacle_active_steps"] for r in results])),
                    1,
                ),
                "avg_obstacle_active_rate_pct": round(
                    float(np.mean([r["obstacle_active_rate_pct"] for r in results])),
                    2,
                ),
                "avg_stopped_steps": round(
                    float(np.mean([r["stopped_steps"] for r in results])),
                    1,
                ),
                "avg_stopped_rate_pct": round(
                    float(np.mean([r["stopped_rate_pct"] for r in results])),
                    2,
                ),
                "avg_stopped_when_obstacle": round(
                    float(np.mean([r["stopped_when_obstacle"] for r in results])),
                    1,
                ),
                "avg_stopped_when_obstacle_rate_pct": round(
                    float(
                        np.mean(
                            [
                                r["stopped_when_obstacle_rate_pct"]
                                for r in results
                            ]
                        )
                    ),
                    2,
                ),
            }
        )

    print(f"\n{'─' * 70}")
    print(f"  Success rate        : {success_rate:.1f}%")
    print(f"  Non-collision rate  : {non_collision_rate:.1f}%")
    print(f"  Collision rate      : {collision_rate:.1f}%")
    print(f"  Avg steps/ep        : {avg_steps:.0f}")
    print(f"  Avg lap steps       : {avg_lap_steps if avg_lap_steps else 'N/A'}")
    print(f"  Avg reward/ep       : {avg_reward:.1f}")
    print(f"  Avg lane deviation  : {avg_lane_deviation:.3f}")
    print(f"  Terminations        : {dict(termination_counts)}")
    print(f"  Total time          : {total_time:.0f}s")
    print(f"{'─' * 70}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULTS_DIR,
        f"{scenario_config['result_prefix']}_{algo}.json",
    )

    with open(output_path, "w") as file:
        json.dump(summary, file, indent=2)

    print(f"Resultados guardados → {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified model evaluation")

    parser.add_argument(
        "--algo",
        required=True,
        choices=["dqn", "ppo"],
        help="Algorithm: dqn or ppo",
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS.keys()),
        help="Evaluation scenario: c1, c2, c3, c4, c5 or c6",
    )

    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Optional model path without .zip. If omitted, uses evaluation_config.py",
    )

    args = parser.parse_args()

    run(
        algo=args.algo,
        scenario=args.scenario,
        model_path=args.model,
    )