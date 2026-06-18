"""
Unified evaluation script for C1-C6.


Before running each scenario, open the corresponding Webots world:
    C1/C2: default lane-following world
    C3/C4: static-obstacles world
    C5/C6: city_dynamic_obstacles.wbt

Evaluation commands:
    C1 PPO: python evaluate.py --algo ppo --scenario c1
    C1 DQN: python evaluate.py --algo dqn --scenario c1

    C2 PPO: python evaluate.py --algo ppo --scenario c2
    C2 DQN: python evaluate.py --algo dqn --scenario c2

    C3 PPO: python evaluate.py --algo ppo --scenario c3
    C3 DQN: python evaluate.py --algo dqn --scenario c3

    C4 PPO: python evaluate.py --algo ppo --scenario c4
    C4 DQN: python evaluate.py --algo dqn --scenario c4

    C5 PPO: python evaluate.py --algo ppo --scenario c5
    C5 DQN: python evaluate.py --algo dqn --scenario c5

    C6 PPO: python evaluate.py --algo ppo --scenario c6
    C6 DQN: python evaluate.py --algo dqn --scenario c6


"""

import argparse
import json
import os
import sys
import time
import platform
from collections import Counter

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from webots_setup import setup_webots_path

import configs.env_config as cfg
from configs.evaluation_config import (
    SEED,
    N_EPISODES,
    NOISE_STD,
    DROPOUT_PROB,
    RESULTS_DIR,
    SCENARIOS,
)


class LegacyCriticalObsWrapper(gym.ObservationWrapper):
    

    LEGACY_REMOVED_KEYS = [
        "inactive_stopped_steps_norm",
    ]

    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = dict(env.observation_space.spaces)

            for key in self.LEGACY_REMOVED_KEYS:
                new_spaces.pop(key, None)

            self.observation_space = spaces.Dict(new_spaces)

    def observation(self, observation):
        if isinstance(observation, dict):
            observation = dict(observation)

            for key in self.LEGACY_REMOVED_KEYS:
                observation.pop(key, None)

        return observation


def model_expects_legacy_critical_obs(model) -> bool:
    
    obs_space = getattr(model, "observation_space", None)

    if not isinstance(obs_space, spaces.Dict):
        return False

    keys = set(obs_space.spaces.keys())

    return (
        "critical_obstacle_active" in keys
        and "inactive_stopped_steps_norm" not in keys
    )


def build_env(algo: str, scenario_config: dict, legacy_critical_obs: bool = False):
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

    if legacy_critical_obs and scenario_config["env_type"] == "critical":
        env = LegacyCriticalObsWrapper(env)

    if algo.lower() == "dqn":
        env = DiscreteActionWrapper(env)

    return env


def get_base_env(env):
    base = env

    while hasattr(base, "env"):
        base = base.env

    return base

def get_vehicle_position(base_env):
    """
    Returns the current vehicle position as a numpy array.
    Uses the Webots Supervisor node when available.
    """
    possible_attrs = [
        "vehicle_node",
        "robot_node",
        "car_node",
        "vehicle",
        "robot",
    ]

    for attr in possible_attrs:
        node = getattr(base_env, attr, None)

        if node is not None and hasattr(node, "getPosition"):
            try:
                return np.array(node.getPosition(), dtype=np.float32)
            except Exception:
                pass

    # Fallback: try GPS sensor if available
    gps = getattr(base_env, "gps", None)

    if gps is not None and hasattr(gps, "getValues"):
        try:
            return np.array(gps.getValues(), dtype=np.float32)
        except Exception:
            pass

    return None


def is_vehicle_physically_stopped(prev_pos, current_pos, threshold=0.005):
    """
    Checks whether the vehicle has physically stopped based on displacement.
    """
    if prev_pos is None or current_pos is None:
        return False

    displacement = float(np.linalg.norm(current_pos - prev_pos))
    return displacement < threshold


def is_vehicle_moving(prev_pos, current_pos, threshold=0.02):
    """
    Checks whether the vehicle resumed movement after a stop.
    """
    if prev_pos is None or current_pos is None:
        return False

    displacement = float(np.linalg.norm(current_pos - prev_pos))
    return displacement > threshold

def load_model(algo: str, model_path: str):
    from stable_baselines3 import DQN, PPO

    model_class = DQN if algo.lower() == "dqn" else PPO

   
    custom_objects = {
        "_last_obs": None,
        "_last_original_obs": None,
        "_last_episode_starts": None,
    }

    return model_class.load(
        model_path,
        env=None,
        custom_objects=custom_objects,
    )


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
            return int(action) in [5, 6]
        except Exception:
            return False

    action_array = np.array(action).flatten()

    if len(action_array) < 2:
        return False

    throttle = float(action_array[1])

    return throttle <= 0.05


def detect_collision(obs, info):
    """
    Uses the collision flag reported by the environment.

    We intentionally avoid using the minimum LiDAR value as a collision
    criterion during evaluation, because C2/C4/C6 apply LiDAR dropout and
    may create artificial zero readings. Those zeros can otherwise be
    incorrectly counted as collisions.
    """
    return bool(info.get("collision", False))


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

    model = load_model(algo, model_path)

    legacy_critical_obs = (
        scenario_config["env_type"] == "critical"
        and model_expects_legacy_critical_obs(model)
    )

    if legacy_critical_obs:
        print(
            "  Legacy model detected: removing "
            "'inactive_stopped_steps_norm' from observations."
        )

    env = build_env(
        algo=algo,
        scenario_config=scenario_config,
        legacy_critical_obs=legacy_critical_obs,
    )
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
        
        # Dynamic-obstacle safety metrics
        physically_stopped_steps = 0
        physically_stopped_when_obstacle = 0
        stop_success = False
        obstacle_was_active = False
        obstacle_cleared_after_active = False
        resume_success = False
        resume_check_steps = 0
        MAX_RESUME_CHECK_STEPS = 300

        prev_vehicle_pos = get_vehicle_position(base_env)

        episode_start = time.time()

        while not done and steps < cfg.MAX_EPISODE_STEPS:
            action, _ = model.predict(obs, deterministic=True)

            obstacle_active = is_dynamic_obstacle_active(base_env)
            stop_or_brake = is_stop_or_brake_action(algo, action)

            obs, reward, done, _, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            if scenario_config["has_dynamic_obstacles"]:
                current_vehicle_pos = get_vehicle_position(base_env)

                physically_stopped = is_vehicle_physically_stopped(
                    prev_vehicle_pos,
                    current_vehicle_pos,
                    threshold=0.005,
                )

                physically_moving = is_vehicle_moving(
                    prev_vehicle_pos,
                    current_vehicle_pos,
                    threshold=0.02,
                )

                if obstacle_active:
                    obstacle_active_steps += 1
                    obstacle_was_active = True

                if stop_or_brake:
                    stopped_steps += 1

                if obstacle_active and stop_or_brake:
                    stopped_when_obstacle += 1

                if physically_stopped:
                    physically_stopped_steps += 1

                if obstacle_active and physically_stopped:
                    physically_stopped_when_obstacle += 1
                    stop_success = True

                # Detect transition: obstacle was active before and is now cleared
                if obstacle_was_active and not obstacle_active:
                    obstacle_cleared_after_active = True

                # After the obstacle clears, check whether the vehicle resumes movement
                if obstacle_cleared_after_active and not resume_success:
                    resume_check_steps += 1

                    if physically_moving:
                        resume_success = True

                    if resume_check_steps >= MAX_RESUME_CHECK_STEPS:
                        # Stop checking after a fixed window
                        obstacle_cleared_after_active = False

                prev_vehicle_pos = current_vehicle_pos

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
            physical_stop_when_obstacle_rate = (
                physically_stopped_when_obstacle / obstacle_active_steps * 100
                if obstacle_active_steps > 0
                else 0.0
            )

            result.update(
                {
                    "obstacle_active_steps": obstacle_active_steps,
                    "obstacle_active_rate_pct": round(obstacle_active_rate, 2),

                    # Action-based stopping metrics
                    "stopped_steps_action_based": stopped_steps,
                    "stopped_rate_action_based_pct": round(stopped_rate, 2),
                    "stopped_when_obstacle_action_based": stopped_when_obstacle,
                    "stopped_when_obstacle_action_based_rate_pct": round(
                        stopped_when_obstacle_rate,
                        2,
                    ),

                    # Physical stopping metrics
                    "physically_stopped_steps": physically_stopped_steps,
                    "physically_stopped_when_obstacle": physically_stopped_when_obstacle,
                    "physically_stopped_when_obstacle_rate_pct": round(
                        physical_stop_when_obstacle_rate,
                        2,
                    ),

                    # Episode-level safety metrics
                    "stop_success": stop_success,
                    "resume_success": resume_success,
                    "resume_check_steps": resume_check_steps,
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
        "legacy_critical_obs": legacy_critical_obs,
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

                # Action-based stopping
                "avg_stopped_steps_action_based": round(
                    float(np.mean([r["stopped_steps_action_based"] for r in results])),
                    1,
                ),
                "avg_stopped_when_obstacle_action_based": round(
                    float(np.mean([r["stopped_when_obstacle_action_based"] for r in results])),
                    1,
                ),
                "avg_stopped_when_obstacle_action_based_rate_pct": round(
                    float(
                        np.mean(
                            [
                                r["stopped_when_obstacle_action_based_rate_pct"]
                                for r in results
                            ]
                        )
                    ),
                    2,
                ),

                # Physical stopping
                "avg_physically_stopped_steps": round(
                    float(np.mean([r["physically_stopped_steps"] for r in results])),
                    1,
                ),
                "avg_physically_stopped_when_obstacle": round(
                    float(np.mean([r["physically_stopped_when_obstacle"] for r in results])),
                    1,
                ),
                "avg_physically_stopped_when_obstacle_rate_pct": round(
                    float(
                        np.mean(
                            [
                                r["physically_stopped_when_obstacle_rate_pct"]
                                for r in results
                            ]
                        )
                    ),
                    2,
                ),

                # Main safety metrics
                "stop_success_rate_pct": round(
                    float(np.mean([r["stop_success"] for r in results]) * 100),
                    2,
                ),
                "resume_success_rate_pct": round(
                    float(np.mean([r["resume_success"] for r in results]) * 100),
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
    if scenario_config["has_dynamic_obstacles"]:
        print(
            f"  Stop success rate   : "
            f"{summary['stop_success_rate_pct']:.1f}%"
        )
        print(
            f"  Resume success rate : "
            f"{summary['resume_success_rate_pct']:.1f}%"
        )
        print(
            f"  Physical stop rate  : "
            f"{summary['avg_physically_stopped_when_obstacle_rate_pct']:.1f}%"
        )
        print(
            f"  Action stop rate    : "
            f"{summary['avg_stopped_when_obstacle_action_based_rate_pct']:.1f}%"
        )
    print(f"  Total time          : {total_time:.0f}s")
    print(f"{'─' * 70}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    result_prefix = scenario_config.get("result_prefix", scenario)

    output_path = os.path.join(
        RESULTS_DIR,
        f"{result_prefix}_{algo}.json",
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
