"""
validate_c2.py — Gera plots e tabela da condição C2 Sensor Noise.

Uso:
    python validate_c2.py

Requer:
    results/c2_dqn.json
    results/c2_ppo.json

Opcional, para comparar C1 → C2:
    results/c1_dqn.json
    results/c1_ppo.json

Gera em results/plots/:
    c2_summary_bar.png
    c2_lane_deviation.png
    c2_termination.png
    c2_reward_dist.png
    c2_vs_c1_success.png
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"
PLOTS_DIR = "./results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DQN_PATH = os.path.join(RESULTS_DIR, "c2_dqn_default_noise.json")
PPO_PATH = os.path.join(RESULTS_DIR, "c2_ppo_default_noise.json")

# C1_DQN_PATH = os.path.join(RESULTS_DIR, "c1_dqn.json")
# C1_PPO_PATH = os.path.join(RESULTS_DIR, "c1_ppo.json")

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


def load(path):
    with open(path) as f:
        return json.load(f)


def check_files():
    missing = []

    if not os.path.exists(DQN_PATH):
        missing.append(DQN_PATH)
    if not os.path.exists(PPO_PATH):
        missing.append(PPO_PATH)

    if missing:
        print(f"\nFicheiros em falta: {missing}")
        print("Corre primeiro evaluate_c2.py para DQN e PPO.")
        return False

    return True


def has_c1_files():
    return os.path.exists(C1_DQN_PATH) and os.path.exists(C1_PPO_PATH)


def plot_summary(dqn, ppo):
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

    ax.set_title("C2 — Sensor Noise Summary", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 110)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8")
    ax.grid(axis="y")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "c2_summary_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Guardado: {path}")


def plot_lane_deviation(dqn, ppo):
    dqn_devs = [
        ep["lane_deviation"]
        for ep in dqn["episodes"]
        if ep["lane_deviation"] is not None
    ]
    ppo_devs = [
        ep["lane_deviation"]
        for ep in ppo["episodes"]
        if ep["lane_deviation"] is not None
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
        flierprops=dict(marker="o", color="#94A3B8", alpha=0.5, markersize=4),
    )

    bp["boxes"][0].set_facecolor(DQN_COLOR)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(PPO_COLOR)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_title("C2 — Lane Deviation Distribution", pad=12)
    ax.set_ylabel("Lane deviation (normalised, 0–1)")
    ax.grid(axis="y")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "c2_lane_deviation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Guardado: {path}")


def plot_termination(dqn, ppo):
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

    def get_pcts(data):
        n = data["n_episodes"]
        tc = data["termination_counts"]
        return {r: tc.get(r, 0) / n * 100 for r in reasons}

    dqn_pcts = get_pcts(dqn)
    ppo_pcts = get_pcts(ppo)

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

        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 3:
                ax.text(
                    i,
                    b + v / 2,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )

        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_title("C2 — Termination Breakdown", pad=12)
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 105)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8", loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "c2_termination.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Guardado: {path}")


def plot_reward(dqn, ppo):
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
        flierprops=dict(marker="o", color="#94A3B8", alpha=0.5, markersize=4),
    )

    bp["boxes"][0].set_facecolor(DQN_COLOR)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(PPO_COLOR)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_title("C2 — Total Reward Distribution", pad=12)
    ax.set_ylabel("Total reward per episode")
    ax.grid(axis="y")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "c2_reward_dist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Guardado: {path}")


def plot_c1_vs_c2(c1_dqn, c1_ppo, c2_dqn, c2_ppo):
    labels = ["C1", "C2"]
    x = np.arange(len(labels))
    width = 0.35

    dqn_vals = [
        c1_dqn["success_rate_pct"],
        c2_dqn["success_rate_pct"],
    ]
    ppo_vals = [
        c1_ppo["success_rate_pct"],
        c2_ppo["success_rate_pct"],
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

    ax.set_title("C1 vs C2 — Sensor Noise Impact", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Task success rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(facecolor=BG_COLOR, edgecolor="#94A3B8")
    ax.grid(axis="y")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "c2_vs_c1_success.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Guardado: {path}")


def print_table(dqn, ppo):
    print(f"\n{'─' * 70}")
    print(f"{'Metric':<30} {'DQN':>16} {'PPO':>16}")
    print(f"{'─' * 70}")

    rows = [
        ("Success Rate (%)", dqn["success_rate_pct"], ppo["success_rate_pct"]),
        ("Non-Collision Rate (%)", dqn["non_collision_rate_pct"], ppo["non_collision_rate_pct"]),
        ("Collision Rate (%)", dqn["collision_rate_pct"], ppo["collision_rate_pct"]),
        ("Avg Steps", dqn["avg_steps"], ppo["avg_steps"]),
        ("Avg Lap Steps", dqn["avg_lap_steps"] or "N/A", ppo["avg_lap_steps"] or "N/A"),
        ("Avg Reward", dqn["avg_reward"], ppo["avg_reward"]),
        ("Avg Lane Deviation", dqn["avg_lane_deviation"], ppo["avg_lane_deviation"]),
    ]

    for label, dv, pv in rows:
        print(f"  {label:<28} {str(dv):>16} {str(pv):>16}")

    print(f"{'─' * 70}\n")


def print_c1_vs_c2_delta(c1_dqn, c1_ppo, c2_dqn, c2_ppo):
    print("\n=== C1 → C2 Impact ===")

    dqn_delta = c2_dqn["success_rate_pct"] - c1_dqn["success_rate_pct"]
    ppo_delta = c2_ppo["success_rate_pct"] - c1_ppo["success_rate_pct"]

    print(f"DQN success change: {dqn_delta:+.1f}%")
    print(f"PPO success change: {ppo_delta:+.1f}%")


if __name__ == "__main__":
    if not check_files():
        exit(1)

    dqn = load(DQN_PATH)
    ppo = load(PPO_PATH)

    print("\n=== C2 Sensor Noise — Validation Report ===")
    print_table(dqn, ppo)

    print("A gerar plots C2...")
    plot_summary(dqn, ppo)
    plot_lane_deviation(dqn, ppo)
    plot_termination(dqn, ppo)
    plot_reward(dqn, ppo)

    if has_c1_files():
        c1_dqn = load(C1_DQN_PATH)
        c1_ppo = load(C1_PPO_PATH)
        print_c1_vs_c2_delta(c1_dqn, c1_ppo, dqn, ppo)
        plot_c1_vs_c2(c1_dqn, c1_ppo, dqn, ppo)
    else:
        print("\nFicheiros C1 não encontrados. A comparação C1 vs C2 foi ignorada.")

    print(f"\nTodos os plots guardados em {PLOTS_DIR}/")