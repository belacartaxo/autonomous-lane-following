"""
evaluate_c1.py — Avaliação do baseline (C1) para DQN e PPO.

Uso:
    python evaluate_c1.py --algo ppo --model ./models/ppo_baseline/ppo_best
    python evaluate_c1.py --algo dqn --model ./models/dqn_baseline/dqn_best

# no uso acrescentar o nome do ficheiro do modelo

Métricas recolhidas por episódio:
    - success        : sobrevive os 5000 steps completos sem colidir
    - non_collision  : não colide (mesmo terminando cedo)
    - collision      : colidiu (via info["collision"])
    - steps          : total de steps no episódio
    - total_reward   : reward acumulado
    - lane_deviation : erro lateral médio normalizado (0–1)
    - termination    : razão de terminação (max_steps / collision / early_end)
"""

#An episode is considered successful if the agent reaches the full 5000-step horizon without collision or premature termination. Premature terminations include loss of lane tracking, collision-risk events, or stuck behaviour.

import argparse
import json
import os
import sys
import time
from collections import Counter
import numpy as np
import platform

if platform.system() == "Windows":
    WEBOTS_HOME = r"C:\Program Files\Webots"
elif platform.system() == "Darwin":  # macOS
    WEBOTS_HOME = "/Applications/Webots.app"
else:  # Linux
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
  
# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

MAX_STEPS = 5000
N_EPISODES = 100


def build_env(algo: str):
    from env.webots_env import WebotsVehicleEnv
    from env.discrete_action_wrapper import DiscreteActionWrapper

    env = WebotsVehicleEnv()

    if algo.lower() == "dqn":
        env = DiscreteActionWrapper(env)

    return env


def get_base_env(env):
    """Traverse wrapper chain to get the base WebotsVehicleEnv."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def load_model(algo: str, model_path: str, env):
    from stable_baselines3 import DQN, PPO
    cls = DQN if algo.lower() == "dqn" else PPO
    return cls.load(model_path, env=env)


def run(algo: str, model_path: str):
    print(f"\n{'=' * 55}")
    print(f"  C1 Baseline Evaluation")
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Model     : {model_path}")
    print(f"  Episodes  : {N_EPISODES}")
    print(f"  Max steps : {MAX_STEPS}")
    print(f"{'=' * 55}\n")

    env   = build_env(algo)
    model = load_model(algo, model_path, env)
    base  = get_base_env(env)

    results    = []
    start_time = time.time()

    for ep in range(N_EPISODES):
        obs, _       = env.reset()
        done         = False
        total_reward = 0.0
        steps        = 0
        collision    = False
        ep_start     = time.time()

        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            total_reward += reward
            steps        += 1

            if info.get("collision", False):
                collision = True

        # ── Metrics ───────────────────────────────────────────────
        lane_dev = (
            base.mean_lane_deviation()
            if hasattr(base, "mean_lane_deviation")
            else float("nan")
        )

        success       = (not collision) and (steps >= MAX_STEPS)
        non_collision = not collision

        if collision:
            termination = "collision"
        elif steps >= MAX_STEPS:
            termination = "max_steps"
        else:
            termination = "early_end"

        ep_time = time.time() - ep_start

        result = {
            "episode"       : ep,
            "success"       : success,
            "non_collision" : non_collision,
            "collision"     : collision,
            "termination"   : termination,
            "steps"         : steps,
            "total_reward"  : round(total_reward, 3),
            "lane_deviation": round(lane_dev, 4),
            "episode_time_s": round(ep_time, 2),
        }
        results.append(result)

        # ── Per-episode print ──────────────────────────────────────
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
            f"LaneDev: {lane_dev:.3f}"
        )

    total_time = time.time() - start_time

    # ── Summary ────────────────────────────────────────────────────
    success_rate       = np.mean([r["success"]       for r in results]) * 100
    non_collision_rate = np.mean([r["non_collision"]  for r in results]) * 100
    collision_rate     = np.mean([r["collision"]      for r in results]) * 100
    avg_steps          = np.mean([r["steps"]          for r in results])
    avg_reward         = np.mean([r["total_reward"]   for r in results])
    avg_lane_dev       = np.nanmean([r["lane_deviation"] for r in results])
    termination_counts = Counter(r["termination"] for r in results)

    successful_steps = [r["steps"] for r in results if r["success"]]
    avg_lap_steps    = float(np.mean(successful_steps)) if successful_steps else None

    summary = {
        "algo"                 : algo.upper(),
        "condition"            : "C1",
        "model_path"           : model_path,
        "n_episodes"           : N_EPISODES,
        "max_steps"            : MAX_STEPS,
        "seed"                 : SEED,
        "success_rate_pct"     : round(float(success_rate),       2),
        "non_collision_rate_pct": round(float(non_collision_rate), 2),
        "collision_rate_pct"   : round(float(collision_rate),      2),
        "avg_steps"            : round(float(avg_steps),           1),
        "avg_lap_steps"        : round(avg_lap_steps, 1) if avg_lap_steps else None,
        "avg_reward"           : round(float(avg_reward),          2),
        "avg_lane_deviation"   : round(float(avg_lane_dev),        4),
        "termination_counts"   : dict(termination_counts),
        "total_time_s"         : round(total_time,                 1),
        "episodes"             : results,
    }

    print(f"\n{'─' * 55}")
    print(f"  Success rate        : {success_rate:.1f}%")
    print(f"  Non-collision rate  : {non_collision_rate:.1f}%")
    print(f"  Collision rate      : {collision_rate:.1f}%")
    print(f"  Avg steps/ep        : {avg_steps:.0f}")
    print(f"  Avg lap steps       : {avg_lap_steps if avg_lap_steps else 'N/A (no successes)'}")
    print(f"  Avg reward/ep       : {avg_reward:.1f}")
    print(f"  Avg lane deviation  : {avg_lane_dev:.3f}")
    print(f"  Terminations        : {dict(termination_counts)}")
    print(f"  Total time          : {total_time:.0f}s")
    print(f"{'─' * 55}\n")

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs("./results", exist_ok=True)
    out_path = f"./results/c1_{algo.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Resultados guardados → {out_path}")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação C1 Baseline")
    parser.add_argument(
        "--algo", required=True, choices=["dqn", "ppo"],
        help="Algoritmo: dqn ou ppo"
    )
    parser.add_argument(
        "--model", required=True,
        help="Caminho para o modelo (sem .zip)"
    )
    args = parser.parse_args()

    run(algo=args.algo, model_path=args.model)
