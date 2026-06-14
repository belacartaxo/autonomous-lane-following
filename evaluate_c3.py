"""
evaluate_c3.py — Avaliação da condição C3 (Dynamic Obstacles) para DQN e PPO.

Uso recomendado — avaliar o melhor modelo guardado pelo EvalCallback:
    python evaluate_c3.py --algo ppo --model ./models/ppo_dynamic/ppo_dynamic_best/best_model
    python evaluate_c3.py --algo dqn --model ./models/dqn_dynamic/dqn_dynamic_best/best_model

Alternativa — avaliar o modelo final:
    python evaluate_c3.py --algo ppo --model ./models/ppo_dynamic/ppo_dynamic_final
    python evaluate_c3.py --algo dqn --model ./models/dqn_dynamic/dqn_dynamic_final

Nota:
    O argumento --model deve receber o caminho do modelo sem a extensão .zip.

Condição avaliada:
    C3 introduz obstáculos dinâmicos no ambiente, sem ruído LiDAR.

Métricas recolhidas por episódio:
    - success                 : completa os 5000 steps sem colisão nem terminação antecipada
    - non_collision           : não colide, mesmo que termine cedo
    - collision               : colisão ou situação de risco detetada pelo LiDAR
    - steps                   : total de steps no episódio
    - total_reward            : reward acumulado
    - lane_deviation          : erro lateral médio normalizado (0–1)
    - termination             : razão de terminação (max_steps / collision / early_end)
    - obstacle_active_steps   : steps em que algum obstáculo dinâmico esteve ativo
    - obstacle_active_rate    : percentagem do episódio com obstáculo dinâmico ativo
    - stopped_steps           : steps em que o agente pareceu parar/travar
    - stopped_when_obstacle   : steps em que o agente parou/travou enquanto havia obstáculo ativo

Definição de sucesso:
    An episode is considered successful if the agent reaches the full
    5000-step horizon without collision or premature termination.
    In C3, temporary stopping or braking near a dynamic obstacle is not
    considered a failure. It is only a failure if the episode terminates
    early, if there is a collision-risk event, if the line is lost for too
    long, or if the vehicle becomes stuck.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
import platform

import numpy as np
import src.config as cfg


# ── Webots path ────────────────────────────────────────────────────────────────
if platform.system() == "Windows":
    WEBOTS_HOME = r"C:\Program Files\Webots"
elif platform.system() == "Darwin":  # macOS
    WEBOTS_HOME = "/Applications/Webots.app"
else:  # Linux
    WEBOTS_HOME = "/usr/local/webots"

os.environ["WEBOTS_HOME"] = WEBOTS_HOME

if platform.system() == "Darwin":
    WEBOTS_PYTHON_PATH = os.path.join(
        WEBOTS_HOME,
        "Contents",
        "lib",
        "controller",
        "python"
    )
else:
    WEBOTS_PYTHON_PATH = os.path.join(
        WEBOTS_HOME,
        "lib",
        "controller",
        "python"
    )

if WEBOTS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, WEBOTS_PYTHON_PATH)


SEED = 42
np.random.seed(SEED)

N_EPISODES = 100


def build_env(algo: str):
    from env.webots_critical_env import WebotsCriticalVehicleEnv
    from env.discrete_action_wrapper import DiscreteActionWrapper

    env = WebotsCriticalVehicleEnv()

    if algo.lower() == "dqn":
        env = DiscreteActionWrapper(env)

    return env


def get_base_env(env):
    """
    Traverse wrapper chain to get the base WebotsCriticalVehicleEnv.
    """
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def load_model(algo: str, model_path: str, env):
    from stable_baselines3 import DQN, PPO

    cls = DQN if algo.lower() == "dqn" else PPO
    return cls.load(model_path, env=env)


def is_collision(obs, info):
    """
    Collision/risk detection.

    Prefer info["collision"] when available.
    If the critical environment does not provide it, fallback to LiDAR.
    """
    if info.get("collision", False):
        return True

    if isinstance(obs, dict) and "lidar" in obs:
        lidar = obs["lidar"]
        try:
            return bool(np.min(lidar) < 1.1)
        except Exception:
            return False

    return False


def is_dynamic_obstacle_active(base_env):
    """
    Checks whether any dynamic obstacle is currently active.

    This is written defensively because the exact attribute may vary
    depending on the implementation of WebotsCriticalVehicleEnv.
    """
    if hasattr(base_env, "critical_obstacles"):
        manager = base_env.critical_obstacles

        if hasattr(manager, "is_any_obstacle_active"):
            try:
                return bool(manager.is_any_obstacle_active())
            except Exception:
                return False

    return False


def is_stop_or_brake_action(algo: str, action):
    """
    Approximate stop/brake detection.

    For PPO:
        action is usually continuous [steering, throttle/brake].
        throttle/brake is action[1].
        Negative or near-zero values indicate braking/stopping.

    For DQN:
        action is a discrete index. In our DiscreteActionWrapper,
        action 5 corresponds to [0.0, 0.0], i.e., stop.
    """
    if algo.lower() == "dqn":
        try:
            return int(action) == 5
        except Exception:
            return False

    try:
        a = np.array(action).flatten()
        if len(a) < 2:
            return False

        throttle = a[1]
        return bool(throttle <= 0.05)

    except Exception:
        return False


def run(algo: str, model_path: str):
    print(f"\n{'=' * 65}")
    print("  C3 Dynamic Obstacles Evaluation")
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Model     : {model_path}")
    print(f"  Episodes  : {N_EPISODES}")
    print(f"  Max steps : {cfg.MAX_EPISODE_STEPS}")
    print(f"{'=' * 65}\n")

    env = build_env(algo)
    model = load_model(algo, model_path, env)
    base = get_base_env(env)

    results = []
    start_time = time.time()

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        steps = 0
        collision = False
        obstacle_active_steps = 0
        stopped_steps = 0
        stopped_when_obstacle = 0
        ep_start = time.time()

        while not done and steps < cfg.MAX_EPISODE_STEPS:
            action, _ = model.predict(obs, deterministic=True)

            obstacle_active = is_dynamic_obstacle_active(base)
            stop_or_brake = is_stop_or_brake_action(algo, action)

            obs, reward, done, _, info = env.step(action)

            total_reward += reward
            steps += 1

            if obstacle_active:
                obstacle_active_steps += 1

            if stop_or_brake:
                stopped_steps += 1

            if obstacle_active and stop_or_brake:
                stopped_when_obstacle += 1

            if is_collision(obs, info):
                collision = True

        lane_dev = (
            base.mean_lane_deviation()
            if hasattr(base, "mean_lane_deviation")
            else float("nan")
        )

        success = (not collision) and (steps >= cfg.MAX_EPISODE_STEPS)
        non_collision = not collision

        if collision:
            termination = "collision"
        elif steps >= cfg.MAX_EPISODE_STEPS:
            termination = "max_steps"
        else:
            termination = "early_end"

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

        ep_time = time.time() - ep_start

        result = {
            "episode": ep,
            "success": success,
            "non_collision": non_collision,
            "collision": collision,
            "termination": termination,
            "steps": steps,
            "total_reward": round(total_reward, 3),
            "lane_deviation": round(lane_dev, 4),
            "obstacle_active_steps": obstacle_active_steps,
            "obstacle_active_rate_pct": round(obstacle_active_rate, 2),
            "stopped_steps": stopped_steps,
            "stopped_rate_pct": round(stopped_rate, 2),
            "stopped_when_obstacle": stopped_when_obstacle,
            "stopped_when_obstacle_rate_pct": round(stopped_when_obstacle_rate, 2),
            "episode_time_s": round(ep_time, 2),
        }

        results.append(result)

        if success:
            status = "✓ SUCCESS  "
        elif collision:
            status = "✗ COLLISION"
        else:
            status = "✗ EARLY END"

        print(
            f"Ep {ep:3d} | {status} | "
            f"Steps: {steps:5d} | "
            f"Reward: {total_reward:8.1f} | "
            f"LaneDev: {lane_dev:.3f} | "
            f"ObsActive: {obstacle_active_steps:4d} | "
            f"Stop@Obs: {stopped_when_obstacle:4d}"
        )

    total_time = time.time() - start_time

    success_rate = np.mean([r["success"] for r in results]) * 100
    non_collision_rate = np.mean([r["non_collision"] for r in results]) * 100
    collision_rate = np.mean([r["collision"] for r in results]) * 100
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_lane_dev = np.nanmean([r["lane_deviation"] for r in results])
    avg_obstacle_active_steps = np.mean([r["obstacle_active_steps"] for r in results])
    avg_obstacle_active_rate = np.mean([r["obstacle_active_rate_pct"] for r in results])
    avg_stopped_steps = np.mean([r["stopped_steps"] for r in results])
    avg_stopped_rate = np.mean([r["stopped_rate_pct"] for r in results])
    avg_stopped_when_obstacle = np.mean([r["stopped_when_obstacle"] for r in results])
    avg_stopped_when_obstacle_rate = np.mean([
        r["stopped_when_obstacle_rate_pct"] for r in results
    ])

    termination_counts = Counter(r["termination"] for r in results)

    successful_steps = [r["steps"] for r in results if r["success"]]
    avg_lap_steps = float(np.mean(successful_steps)) if successful_steps else None

    summary = {
        "algo": algo.upper(),
        "condition": "C3",
        "condition_name": "Dynamic Obstacles",
        "model_path": model_path,
        "n_episodes": N_EPISODES,
        "max_steps": cfg.MAX_EPISODE_STEPS,
        "seed": SEED,
        "success_rate_pct": round(float(success_rate), 2),
        "non_collision_rate_pct": round(float(non_collision_rate), 2),
        "collision_rate_pct": round(float(collision_rate), 2),
        "avg_steps": round(float(avg_steps), 1),
        "avg_lap_steps": round(avg_lap_steps, 1) if avg_lap_steps else None,
        "avg_reward": round(float(avg_reward), 2),
        "avg_lane_deviation": round(float(avg_lane_dev), 4),
        "avg_obstacle_active_steps": round(float(avg_obstacle_active_steps), 1),
        "avg_obstacle_active_rate_pct": round(float(avg_obstacle_active_rate), 2),
        "avg_stopped_steps": round(float(avg_stopped_steps), 1),
        "avg_stopped_rate_pct": round(float(avg_stopped_rate), 2),
        "avg_stopped_when_obstacle": round(float(avg_stopped_when_obstacle), 1),
        "avg_stopped_when_obstacle_rate_pct": round(
            float(avg_stopped_when_obstacle_rate),
            2
        ),
        "termination_counts": dict(termination_counts),
        "total_time_s": round(total_time, 1),
        "episodes": results,
    }

    print(f"\n{'─' * 65}")
    print(f"  Success rate                  : {success_rate:.1f}%")
    print(f"  Non-collision rate            : {non_collision_rate:.1f}%")
    print(f"  Collision rate                : {collision_rate:.1f}%")
    print(f"  Avg steps/ep                  : {avg_steps:.0f}")
    print(f"  Avg lap steps                 : {avg_lap_steps if avg_lap_steps else 'N/A (no successes)'}")
    print(f"  Avg reward/ep                 : {avg_reward:.1f}")
    print(f"  Avg lane deviation            : {avg_lane_dev:.3f}")
    print(f"  Avg obstacle active steps     : {avg_obstacle_active_steps:.1f}")
    print(f"  Avg obstacle active rate      : {avg_obstacle_active_rate:.1f}%")
    print(f"  Avg stopped steps             : {avg_stopped_steps:.1f}")
    print(f"  Avg stopped rate              : {avg_stopped_rate:.1f}%")
    print(f"  Avg stopped when obstacle     : {avg_stopped_when_obstacle:.1f}")
    print(f"  Avg stop when obstacle rate   : {avg_stopped_when_obstacle_rate:.1f}%")
    print(f"  Terminations                  : {dict(termination_counts)}")
    print(f"  Total time                    : {total_time:.0f}s")
    print(f"{'─' * 65}\n")

    os.makedirs("./results", exist_ok=True)
    out_path = f"./results/c3_{algo.lower()}.json"

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Resultados guardados → {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação C3 Dynamic Obstacles")
    parser.add_argument(
        "--algo",
        required=True,
        choices=["dqn", "ppo"],
        help="Algoritmo: dqn ou ppo",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Caminho para o modelo, sem .zip",
    )
    args = parser.parse_args()

    run(algo=args.algo, model_path=args.model)