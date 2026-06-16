"""
Cenários disponíveis: c1-c6
python validate.py --scenario c1
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.validation_config import RESULTS_DIR, SCENARIOS


DQN_COLOR = "#00B4D8"
PPO_COLOR = "#F59E0B"
BG_COLOR = "#161B22"
GRID_COLOR = "#1E3A4C"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": "#94A3B8",
    "axes.labelcolor": "#F1F5F9",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "text.color": "#F1F5F9",
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def get_result_paths(scenario_config):
    dqn_path = os.path.join(RESULTS_DIR, scenario_config["dqn_file"])
    ppo_path = os.path.join(RESULTS_DIR, scenario_config["ppo_file"])
    return dqn_path, ppo_path


def check_files(dqn_path, ppo_path):
    missing = []

    if not os.path.exists(dqn_path):
        missing.append(dqn_path)

    if not os.path.exists(ppo_path):
        missing.append(ppo_path)

    if missing:
        print(f"\nFicheiros em falta: {missing}")
        print("Corre primeiro o evaluate.py para DQN e PPO neste cenário.")
        return False

    return True


def save_plot(fig, plots_dir, filename):
    os.makedirs(plots_dir, exist_ok=True)

    path = os.path.join(plots_dir, filename)

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        facecolor=BG_COLOR,
    )

    plt.close(fig)

    print(f"Guardado: {path}")


def plot_summary(dqn, ppo, scenario, scenario_name, plots_dir):
    metrics = ["Success (%)", "Non-Collision (%)", "Collision (%)"]

    dqn_vals = [
        dqn["success_rate_pct"],
        dqn["non_collision_rate_pct"],
        dqn["collision_rate_pct"],
    ]

    ppo_vals = [
        ppo["success_rate_pct"],
        ppo["non_collision_rate_pct"],
        ppo["collision_rate_pct"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)

    bars_dqn = ax.bar(
        x - width / 2,
        dqn_vals,
        width,
        label="DQN",
        color=DQN_COLOR,
        alpha=0.85,
    )

    bars_ppo = ax.bar(
        x + width / 2,
        ppo_vals,
        width,
        label="PPO",
        color=PPO_COLOR,
        alpha=0.85,
    )

    for bar, val in zip(list(bars_dqn) + list(bars_ppo), dqn_vals + ppo_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#F1F5F9",
        )

    ax.set_title(f"{scenario.upper()} — {scenario_name} Summary", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 110)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8")
    ax.grid(axis="y")
    fig.tight_layout()

    save_plot(fig, plots_dir, f"{scenario}_summary_bar.png")


def plot_lane_deviation(dqn, ppo, scenario, scenario_name, plots_dir):
    dqn_devs = [
        ep["lane_deviation"]
        for ep in dqn["episodes"]
        if ep.get("lane_deviation") is not None
    ]

    ppo_devs = [
        ep["lane_deviation"]
        for ep in ppo["episodes"]
        if ep.get("lane_deviation") is not None
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG_COLOR)

    bp = ax.boxplot(
        [dqn_devs, ppo_devs],
        labels=["DQN", "PPO"],
        patch_artist=True,
        medianprops=dict(color="#F1F5F9", linewidth=2),
        whiskerprops=dict(color="#94A3B8"),
        capprops=dict(color="#94A3B8"),
        flierprops=dict(
            marker="o",
            color="#94A3B8",
            alpha=0.5,
            markersize=4,
        ),
    )

    bp["boxes"][0].set_facecolor(DQN_COLOR)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(PPO_COLOR)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_title(f"{scenario.upper()} — Lane Deviation Distribution", pad=12)
    ax.set_ylabel("Lane deviation (normalised, 0–1)")
    ax.grid(axis="y")
    fig.tight_layout()

    save_plot(fig, plots_dir, f"{scenario}_lane_deviation.png")


def plot_termination(dqn, ppo, scenario, scenario_name, plots_dir):
    reasons = ["max_steps", "early_end", "collision"]

    colors = {
        "max_steps": "#22C55E",
        "early_end": "#F59E0B",
        "collision": "#EF4444",
    }

    labels = {
        "max_steps": "Completed",
        "early_end": "Early End",
        "collision": "Collision",
    }

    def get_percentages(data):
        n = data["n_episodes"]
        termination_counts = data["termination_counts"]
        return {
            reason: termination_counts.get(reason, 0) / n * 100
            for reason in reasons
        }

    dqn_pcts = get_percentages(dqn)
    ppo_pcts = get_percentages(ppo)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_COLOR)

    algos = ["DQN", "PPO"]
    bottoms = [0.0, 0.0]

    for reason in reasons:
        vals = [dqn_pcts[reason], ppo_pcts[reason]]

        ax.bar(
            algos,
            vals,
            bottom=bottoms,
            label=labels[reason],
            color=colors[reason],
            alpha=0.85,
            width=0.5,
        )

        for i, (value, bottom) in enumerate(zip(vals, bottoms)):
            if value > 3:
                ax.text(
                    i,
                    bottom + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )

        bottoms = [
            bottom + value
            for bottom, value in zip(bottoms, vals)
        ]

    ax.set_title(f"{scenario.upper()} — Termination Breakdown", pad=12)
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 105)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8", loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()

    save_plot(fig, plots_dir, f"{scenario}_termination.png")


def plot_reward(dqn, ppo, scenario, scenario_name, plots_dir):
    dqn_rewards = [ep["total_reward"] for ep in dqn["episodes"]]
    ppo_rewards = [ep["total_reward"] for ep in ppo["episodes"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG_COLOR)

    bp = ax.boxplot(
        [dqn_rewards, ppo_rewards],
        labels=["DQN", "PPO"],
        patch_artist=True,
        medianprops=dict(color="#F1F5F9", linewidth=2),
        whiskerprops=dict(color="#94A3B8"),
        capprops=dict(color="#94A3B8"),
        flierprops=dict(
            marker="o",
            color="#94A3B8",
            alpha=0.5,
            markersize=4,
        ),
    )

    bp["boxes"][0].set_facecolor(DQN_COLOR)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(PPO_COLOR)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_title(f"{scenario.upper()} — Total Reward Distribution", pad=12)
    ax.set_ylabel("Total reward per episode")
    ax.grid(axis="y")
    fig.tight_layout()

    save_plot(fig, plots_dir, f"{scenario}_reward_dist.png")


def plot_obstacle_activity(dqn, ppo, scenario, scenario_name, plots_dir):
    metrics = [
        "Obstacle Active\nSteps",
        "Obstacle Active\nRate (%)",
    ]

    dqn_vals = [
        dqn.get("avg_obstacle_active_steps", 0),
        dqn.get("avg_obstacle_active_rate_pct", 0),
    ]

    ppo_vals = [
        ppo.get("avg_obstacle_active_steps", 0),
        ppo.get("avg_obstacle_active_rate_pct", 0),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_COLOR)

    bars_dqn = ax.bar(
        x - width / 2,
        dqn_vals,
        width,
        label="DQN",
        color=DQN_COLOR,
        alpha=0.85,
    )

    bars_ppo = ax.bar(
        x + width / 2,
        ppo_vals,
        width,
        label="PPO",
        color=PPO_COLOR,
        alpha=0.85,
    )

    max_val = max(max(dqn_vals), max(ppo_vals), 1)

    for bar, val in zip(list(bars_dqn) + list(bars_ppo), dqn_vals + ppo_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#F1F5F9",
        )

    ax.set_title(f"{scenario.upper()} — Dynamic Obstacle Activity", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.set_ylim(0, max_val * 1.25)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8")
    ax.grid(axis="y")
    fig.tight_layout()

    save_plot(fig, plots_dir, f"{scenario}_obstacle_activity.png")


def plot_stopping_behavior(dqn, ppo, scenario, scenario_name, plots_dir):
    metrics = [
        "Stopped\nSteps",
        "Stopped\nRate (%)",
        "Stopped When\nObstacle",
        "Stop@Obstacle\nRate (%)",
    ]

    dqn_vals = [
        dqn.get("avg_stopped_steps", 0),
        dqn.get("avg_stopped_rate_pct", 0),
        dqn.get("avg_stopped_when_obstacle", 0),
        dqn.get("avg_stopped_when_obstacle_rate_pct", 0),
    ]

    ppo_vals = [
        ppo.get("avg_stopped_steps", 0),
        ppo.get("avg_stopped_rate_pct", 0),
        ppo.get("avg_stopped_when_obstacle", 0),
        ppo.get("avg_stopped_when_obstacle_rate_pct", 0),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)

    bars_dqn = ax.bar(
        x - width / 2,
        dqn_vals,
        width,
        label="DQN",
        color=DQN_COLOR,
        alpha=0.85,
    )

    bars_ppo = ax.bar(
        x + width / 2,
        ppo_vals,
        width,
        label="PPO",
        color=PPO_COLOR,
        alpha=0.85,
    )

    max_val = max(max(dqn_vals), max(ppo_vals), 1)

    for bar, val in zip(list(bars_dqn) + list(bars_ppo), dqn_vals + ppo_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#F1F5F9",
        )

    ax.set_title(f"{scenario.upper()} — Stopping Behaviour", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.set_ylim(0, max_val * 1.25)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8")
    ax.grid(axis="y")
    fig.tight_layout()

    save_plot(fig, plots_dir, f"{scenario}_stopping_behavior.png")


def plot_comparison(
    reference_dqn,
    reference_ppo,
    current_dqn,
    current_ppo,
    reference_scenario,
    current_scenario,
    plots_dir,
):
    labels = [
        reference_scenario.upper(),
        current_scenario.upper(),
    ]

    x = np.arange(len(labels))
    width = 0.35

    dqn_vals = [
        reference_dqn["success_rate_pct"],
        current_dqn["success_rate_pct"],
    ]

    ppo_vals = [
        reference_ppo["success_rate_pct"],
        current_ppo["success_rate_pct"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG_COLOR)

    bars_dqn = ax.bar(
        x - width / 2,
        dqn_vals,
        width,
        label="DQN",
        color=DQN_COLOR,
        alpha=0.85,
    )

    bars_ppo = ax.bar(
        x + width / 2,
        ppo_vals,
        width,
        label="PPO",
        color=PPO_COLOR,
        alpha=0.85,
    )

    for bar, val in zip(list(bars_dqn) + list(bars_ppo), dqn_vals + ppo_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#F1F5F9",
        )

    ax.set_title(
        f"{reference_scenario.upper()} vs {current_scenario.upper()} — Success Rate",
        pad=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Task success rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8")
    ax.grid(axis="y")
    fig.tight_layout()

    save_plot(
        fig,
        plots_dir,
        f"{current_scenario}_vs_{reference_scenario}_success.png",
    )


def print_table(dqn, ppo, dynamic=False):
    width = 88 if dynamic else 70

    print(f"\n{'─' * width}")
    print(f"{'Metric':<42} {'DQN':>18} {'PPO':>18}")
    print(f"{'─' * width}")

    rows = [
        ("Success Rate (%)", dqn["success_rate_pct"], ppo["success_rate_pct"]),
        ("Non-Collision Rate (%)", dqn["non_collision_rate_pct"], ppo["non_collision_rate_pct"]),
        ("Collision Rate (%)", dqn["collision_rate_pct"], ppo["collision_rate_pct"]),
        ("Avg Steps", dqn["avg_steps"], ppo["avg_steps"]),
        ("Avg Lap Steps", dqn["avg_lap_steps"] or "N/A", ppo["avg_lap_steps"] or "N/A"),
        ("Avg Reward", dqn["avg_reward"], ppo["avg_reward"]),
        ("Avg Lane Deviation", dqn["avg_lane_deviation"], ppo["avg_lane_deviation"]),
    ]

    if dynamic:
        rows.extend([
            ("Avg Obstacle Active Steps", dqn.get("avg_obstacle_active_steps", "N/A"), ppo.get("avg_obstacle_active_steps", "N/A")),
            ("Avg Obstacle Active Rate (%)", dqn.get("avg_obstacle_active_rate_pct", "N/A"), ppo.get("avg_obstacle_active_rate_pct", "N/A")),
            ("Avg Stopped Steps", dqn.get("avg_stopped_steps", "N/A"), ppo.get("avg_stopped_steps", "N/A")),
            ("Avg Stopped Rate (%)", dqn.get("avg_stopped_rate_pct", "N/A"), ppo.get("avg_stopped_rate_pct", "N/A")),
            ("Avg Stopped When Obstacle", dqn.get("avg_stopped_when_obstacle", "N/A"), ppo.get("avg_stopped_when_obstacle", "N/A")),
            ("Avg Stop@Obstacle Rate (%)", dqn.get("avg_stopped_when_obstacle_rate_pct", "N/A"), ppo.get("avg_stopped_when_obstacle_rate_pct", "N/A")),
        ])

    for label, dqn_value, ppo_value in rows:
        print(f"  {label:<40} {str(dqn_value):>18} {str(ppo_value):>18}")

    print(f"{'─' * width}\n")


def print_comparison_delta(
    reference_dqn,
    reference_ppo,
    current_dqn,
    current_ppo,
    reference_scenario,
    current_scenario,
):
    dqn_delta = (
        current_dqn["success_rate_pct"]
        - reference_dqn["success_rate_pct"]
    )

    ppo_delta = (
        current_ppo["success_rate_pct"]
        - reference_ppo["success_rate_pct"]
    )

    print(f"\n=== {reference_scenario.upper()} → {current_scenario.upper()} Impact ===")
    print(f"DQN success change: {dqn_delta:+.1f}%")
    print(f"PPO success change: {ppo_delta:+.1f}%")


def validate_scenario(scenario):
    scenario = scenario.lower()

    scenario_config = SCENARIOS[scenario]
    scenario_name = scenario_config["name"]
    plots_dir = scenario_config["plots_dir"]
    dynamic = scenario_config["dynamic"]

    os.makedirs(plots_dir, exist_ok=True)

    dqn_path, ppo_path = get_result_paths(scenario_config)

    if not check_files(dqn_path, ppo_path):
        return

    dqn = load_json(dqn_path)
    ppo = load_json(ppo_path)

    print(f"\n=== {scenario.upper()} — {scenario_name} Validation Report ===")

    print_table(
        dqn=dqn,
        ppo=ppo,
        dynamic=dynamic,
    )

    print("A gerar plots...")

    plot_summary(
        dqn=dqn,
        ppo=ppo,
        scenario=scenario,
        scenario_name=scenario_name,
        plots_dir=plots_dir,
    )

    plot_lane_deviation(
        dqn=dqn,
        ppo=ppo,
        scenario=scenario,
        scenario_name=scenario_name,
        plots_dir=plots_dir,
    )

    plot_termination(
        dqn=dqn,
        ppo=ppo,
        scenario=scenario,
        scenario_name=scenario_name,
        plots_dir=plots_dir,
    )

    plot_reward(
        dqn=dqn,
        ppo=ppo,
        scenario=scenario,
        scenario_name=scenario_name,
        plots_dir=plots_dir,
    )

    if dynamic:
        plot_obstacle_activity(
            dqn=dqn,
            ppo=ppo,
            scenario=scenario,
            scenario_name=scenario_name,
            plots_dir=plots_dir,
        )

        plot_stopping_behavior(
            dqn=dqn,
            ppo=ppo,
            scenario=scenario,
            scenario_name=scenario_name,
            plots_dir=plots_dir,
        )

    comparison_scenario = scenario_config["compare_with"]

    if comparison_scenario is not None:
        comparison_config = SCENARIOS[comparison_scenario]
        comparison_dqn_path, comparison_ppo_path = get_result_paths(
            comparison_config
        )

        if os.path.exists(comparison_dqn_path) and os.path.exists(comparison_ppo_path):
            comparison_dqn = load_json(comparison_dqn_path)
            comparison_ppo = load_json(comparison_ppo_path)

            print_comparison_delta(
                reference_dqn=comparison_dqn,
                reference_ppo=comparison_ppo,
                current_dqn=dqn,
                current_ppo=ppo,
                reference_scenario=comparison_scenario,
                current_scenario=scenario,
            )

            plot_comparison(
                reference_dqn=comparison_dqn,
                reference_ppo=comparison_ppo,
                current_dqn=dqn,
                current_ppo=ppo,
                reference_scenario=comparison_scenario,
                current_scenario=scenario,
                plots_dir=plots_dir,
            )
        else:
            print(
                f"\nFicheiros de comparação com {comparison_scenario.upper()} "
                "não encontrados. Comparação ignorada."
            )

    print(f"\nTodos os plots guardados em {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified validation script for all evaluation scenarios."
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS.keys()),
        help="Scenario to validate: c1, c2, c3, c4, c5 or c6.",
    )

    args = parser.parse_args()

    validate_scenario(args.scenario)