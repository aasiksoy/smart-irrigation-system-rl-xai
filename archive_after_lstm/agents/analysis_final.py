import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from simulator.fao56 import theta_from_w
from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

EPISODES_DIR = "results/final_episodes"
RUN_FILES = ["results/final_runs.csv", "results/final_noise.csv", "results/final_weights.csv"]
CI_CSV = "results/final_ci.csv"
GAP_PLOT = "results/fig_gap_closed.png"
BEHAVIOUR_PLOT = "results/fig_behaviour.png"
CKPT_DIR = "checkpoints/final"

N_BOOT = 10_000
HOLD_MARGIN = 0.005

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

PAIRS = [
    ("fc_start", "stress_only", "blind"),
    ("fc_start", "stress_only", "remember_start"),
    ("fc_start", "stress_only", "rule_probe_hold"),
    ("fc_start", "stress_only", "informed"),
    ("fc_start", "stress_start", "stress_only"),
    ("fc_start", "remember_start", "blind"),
    ("random_start", "stress_only", "blind"),
    ("random_start", "stress_only", "remember_start"),
    ("random_start", "stress_only", "rule_probe_hold"),
    ("random_start", "stress_only", "informed"),
]

SETTING_FLAGS = {"fc_start": {}, "random_start": {"start_range": (0.6, 1.0)}}


def read_rows():
    rows = []
    for path in RUN_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                for key in ["reward", "irrigation", "stressed_days", "waited_rate"]:
                    row[key] = float(row[key])
                row["seed"] = int(row["seed"])
                rows.append(row)
    return rows


def mean_reward(rows, run):
    values = [r["reward"] for r in rows if r["run"] == run]
    return np.mean(values) if values else None


def episode_rewards(run, seeds):
    per_seed = [np.load(f"{EPISODES_DIR}/{run}_seed{s}.npy")[:, 0] for s in seeds]
    return np.mean(per_seed, axis=0)


def probe_and_hold(obs):
    theta = obs[0] * 0.5
    stress_seen = obs[6]
    theta_stress = obs[7] * 0.5
    if stress_seen < 0.5:
        return 0
    if theta < theta_stress + HOLD_MARGIN:
        return 4
    return 0


def rule_episode_rewards(setting, test_soils, test_seasons):
    flags = {**SETTING_FLAGS[setting], "remember_start": True, "stress_memory": True}
    rewards = []
    episode = 0
    for soil in test_soils:
        for season in test_seasons:
            env = IrrigationEnv([soil], [season], **flags)
            obs, info = env.reset(seed=episode)
            episode += 1
            total = 0.0
            done = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(probe_and_hold(obs))
                total += reward
                done = terminated or truncated
            rewards.append(total)
    return np.array(rewards)


def paired_bootstrap(a, b, rng):
    diff = a - b
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    low, high = np.percentile(means, [2.5, 97.5])
    return diff.mean(), low, high, (diff > 0).mean()


def confidence_intervals(rows, test_soils, test_seasons):
    rng = np.random.default_rng(0)
    rule = {s: rule_episode_rewards(s, test_soils, test_seasons) for s in SETTING_FLAGS}
    out = []
    print("\nB1. Paired differences in reward per test season (mean over seeds), 95% bootstrap CI")
    print(f"{'setting':13s} | {'A':16s} vs {'B':16s} | {'A - B':>7s} | {'95% CI':>18s} | A better in")
    for setting, a, b in PAIRS:
        vectors = []
        for agent in (a, b):
            if agent == "rule_probe_hold":
                vectors.append(rule[setting])
            else:
                run = f"{setting}_{agent}"
                seeds = sorted({r["seed"] for r in rows if r["run"] == run})
                vectors.append(episode_rewards(run, seeds))
        mean, low, high, share = paired_bootstrap(vectors[0], vectors[1], rng)
        print(f"{setting:13s} | {a:16s} vs {b:16s} | {mean:7.2f} | [{low:7.2f}, {high:7.2f}] | {100 * share:4.0f}% of seasons")
        out.append({"setting": setting, "a": a, "b": b, "diff": mean, "ci_low": low,
                    "ci_high": high, "a_better_share": share})
    os.makedirs(os.path.dirname(CI_CSV), exist_ok=True)
    with open(CI_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    for setting in SETTING_FLAGS:
        print(f"rule probe_and_hold, {setting}: mean reward {rule[setting].mean():.2f}")


def gap_per_seed(rows, run, setting):
    blind = mean_reward(rows, f"{setting}_blind")
    informed = mean_reward(rows, f"{setting}_informed")
    values = [r["reward"] for r in rows if r["run"] == run]
    return 100 * (np.array(values) - blind) / (informed - blind)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_panel(ax, labels, series, colors, names, title):
    y = np.arange(len(labels))
    height = 0.8 / len(series)
    for i, (values, color, name) in enumerate(zip(series, colors, names)):
        means = [np.mean(v) if len(v) else np.nan for v in values]
        stds = [np.std(v) if len(v) else 0 for v in values]
        pos = y - 0.4 + height * (i + 0.5)
        ax.barh(pos, means, height=height - 0.04, color=color, label=name,
                xerr=stds, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        for p, m, sd in zip(pos, means, stds):
            if not np.isnan(m):
                ax.text(max(m, 0) + sd + 3, p, f"{m:.0f}%", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axvline(100, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlim(right=125)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("% of the gap closed (blind = 0%, informed = 100%)", color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)


def gap_figure(rows):
    agents = [("remember_start", "remember start"), ("stress_only", "stress sensor"),
              ("stress_start", "stress sensor + start")]
    noise_levels = [("", "no noise"), ("_noise_low", "low noise"),
                    ("_noise_mid", "mid noise"), ("_noise_high", "high noise")]
    has_confirm = any(r["run"].startswith("fc_start_stress_only_confirm2") for r in rows)

    panels = 2 if has_confirm or any("noise" in r["run"] for r in rows) else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 4.2), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    fc = [gap_per_seed(rows, f"fc_start_{a}", "fc_start") for a, _ in agents]
    rs = [gap_per_seed(rows, f"random_start_{a}", "random_start") for a, _ in agents]
    bar_panel(axes[0, 0], [label for _, label in agents], [fc, rs], [BLUE, ORANGE],
              ["start at field capacity", "random start"], "How much of the gap each agent closes")

    if panels == 2:
        plain = [gap_per_seed(rows, f"fc_start_stress_only{suffix}", "fc_start") for suffix, _ in noise_levels]
        series, colors, names = [plain], [BLUE], ["memory: any detection"]
        if has_confirm:
            confirmed = [gap_per_seed(rows, f"fc_start_stress_only_confirm2{suffix}", "fc_start")
                         for suffix, _ in noise_levels]
            series.append(confirmed)
            colors.append(AQUA)
            names.append("memory: 2 days confirmed")
        bar_panel(axes[0, 1], [label for _, label in noise_levels], series, colors, names,
                  "Stress sensor under sensor noise")

    fig.tight_layout()
    fig.savefig(GAP_PLOT, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"saved {GAP_PLOT}")


def run_trajectory(model, soil, season, flags, seed):
    env = IrrigationEnv([soil], [season], **flags)
    obs, info = env.reset(seed=seed)
    thetas = [theta_from_w(env.w, env.soil)]
    irrigation = []
    stressed = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        thetas.append(theta_from_w(env.w, env.soil))
        irrigation.append(info["irrigation"])
        stressed.append(info["ks"] < 1.0)
        done = terminated or truncated
    return np.array(thetas), np.array(irrigation), np.array(stressed)


def behaviour_figure(test_soils, test_seasons):
    agents = [("blind", "blind", {}, INK_2),
              ("remember_start", "remember start", {"remember_start": True}, BLUE),
              ("stress_only", "stress sensor", {"stress_memory": True}, AQUA)]
    models = {name: PPO.load(f"{CKPT_DIR}/fc_start_{name}_seed0") for name, _, _, _ in agents}

    chosen = None
    episode = 0
    for soil in test_soils:
        for season in test_seasons:
            _, irrigation, stressed = run_trajectory(models["stress_only"], soil, season,
                                                     {"stress_memory": True}, episode)
            first = np.argmax(stressed) if stressed.any() else None
            if chosen is None and first is not None and 25 <= first <= 50 and irrigation[:first].sum() == 0 and stressed.sum() <= 3:
                chosen = (soil, season, episode)
            episode += 1
    soil, season, seed = chosen

    stress_theta = soil.w_stress / (1000 * soil.root_depth_m)
    fig, axes = plt.subplots(len(agents), 1, figsize=(9, 6.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (name, label, flags, color) in zip(axes, agents):
        thetas, irrigation, stressed = run_trajectory(models[name], soil, season, flags, seed)
        days = np.arange(len(thetas))
        ax.plot(days, thetas, color=color, linewidth=2)
        end = len(thetas) - 1
        ax.hlines(soil.theta_fc, 0, end, color=INK_2, linewidth=1, linestyle=":")
        ax.hlines(stress_theta, 0, end, color=INK_2, linewidth=1, linestyle="--")
        ax.hlines(soil.theta_wp, 0, end, color=INK_2, linewidth=1, linestyle="-.")
        irrigated = np.nonzero(irrigation)[0] + 1
        ax.plot(irrigated, np.full(len(irrigated), soil.theta_wp + 0.005), linestyle="none",
                marker="|", markersize=8, color=color)
        stress_days = np.nonzero(stressed)[0] + 1
        ax.plot(stress_days, thetas[stress_days], linestyle="none", marker="o", markersize=4,
                markerfacecolor="white", markeredgecolor=INK)
        ax.set_title(f"{label}: {irrigation.sum():.0f} mm irrigation, {stressed.sum()} stressed days",
                     loc="left", fontsize=10, color=INK)
        ax.set_ylabel("soil moisture (theta)", color=INK_2, fontsize=9)
        style_axis(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xlim(-2, len(thetas) + 22)
    for ax in axes:
        ax.text(len(thetas) + 2, soil.theta_fc, "field capacity", va="center", fontsize=8, color=INK_2)
        ax.text(len(thetas) + 2, stress_theta, "stress threshold", va="center", fontsize=8, color=INK_2)
        ax.text(len(thetas) + 2, soil.theta_wp, "wilting point", va="center", fontsize=8, color=INK_2)
    axes[-1].set_xlabel("day of season   (| = irrigation day, o = stressed day)", color=INK_2, fontsize=9)
    fig.suptitle(f"One unseen test soil and season (test episode {seed})", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(BEHAVIOUR_PLOT, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"saved {BEHAVIOUR_PLOT} (test episode {seed})")


def weights_table(rows):
    print("\nA3. Reward weight sensitivity: stress_only, % of gap closed")
    for setting, weight in [("stress_w1", 1.0), ("fc_start", 2.0), ("stress_w4", 4.0)]:
        run = f"{setting}_stress_only"
        if mean_reward(rows, run) is None:
            continue
        gaps = gap_per_seed(rows, run, setting)
        print(f"stress weight {weight}: {gaps.mean():5.0f}% +- {gaps.std():.0f}% (n={len(gaps)})")


def main():
    rows = read_rows()
    test_soils = make_soils(20, seed=1000)
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    confidence_intervals(rows, test_soils, test_seasons)
    weights_table(rows)
    gap_figure(rows)
    behaviour_figure(test_soils, test_seasons)


if __name__ == "__main__":
    main()