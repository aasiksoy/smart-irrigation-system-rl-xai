import sys

import numpy as np

from agents.final_experiments import (AGENTS, CKPT_DIR, EPISODES_DIR, NOISE, evaluate_model,
                                      get_model, log, write_csv)
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

NOISE_SEEDS = [0, 1, 2, 3, 4]
WEIGHT_SEEDS = [0, 1, 2]
CONFIRM_DAYS = 2
STRESS_WEIGHTS = {"stress_w1": 1.0, "stress_w4": 4.0}
WEIGHT_AGENTS = ["blind", "stress_only", "informed"]


def build_noise_runs():
    base = {**AGENTS["stress_only"], "confirm_days": CONFIRM_DAYS}
    runs = [{"name": "fc_start_stress_only_confirm2", "setting": "fc_start",
             "agent": "stress_only_confirm2", "flags": base}]
    for noise, noise_flags in NOISE.items():
        runs.append({"name": f"fc_start_stress_only_confirm2_{noise}", "setting": "fc_start",
                     "agent": f"stress_only_confirm2_{noise}", "flags": {**base, **noise_flags}})
    return runs


def build_weight_runs():
    runs = []
    for setting, weight in STRESS_WEIGHTS.items():
        for agent in WEIGHT_AGENTS:
            runs.append({"name": f"{setting}_{agent}", "setting": setting, "agent": agent,
                         "flags": {**AGENTS[agent], "stress_weight": weight}})
    return runs


def run_all(runs, seeds, csv_path, train_soils, train_seasons, test_soils, test_seasons):
    rows = []
    total = len(runs) * len(seeds)
    done = 0
    for seed in seeds:
        for run in runs:
            done += 1
            path = f"{CKPT_DIR}/{run['name']}_seed{seed}"
            model, trained = get_model(path, run["flags"], seed, train_soils, train_seasons)
            rewards, irrigation, stressed, waited = evaluate_model(model, test_soils, test_seasons, run["flags"])
            np.save(f"{EPISODES_DIR}/{run['name']}_seed{seed}.npy",
                    np.column_stack([rewards, irrigation, stressed, waited]))
            rows.append({
                "run": run["name"], "setting": run["setting"], "agent": run["agent"], "seed": seed,
                "reward": float(rewards.mean()), "irrigation": float(irrigation.mean()),
                "stressed_days": float(stressed.mean()), "waited_rate": float(waited.mean()),
            })
            write_csv(csv_path, rows)
            status = "trained" if trained else "loaded"
            log(f"[{done}/{total}] {run['name']} seed {seed} ({status}): reward {rewards.mean():.2f}, "
                f"irrigation {irrigation.mean():.0f} mm, stressed {stressed.mean():.1f}, "
                f"waited {100 * waited.mean():.0f}%")
    return rows


def print_table(rows):
    names = []
    for row in rows:
        if row["run"] not in names:
            names.append(row["run"])
    print(f"\n{'run':45s} | {'reward':>14s} | irrigation | stressed | waited")
    for name in names:
        group = [r for r in rows if r["run"] == name]
        rewards = [r["reward"] for r in group]
        print(f"{name:45s} | {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} "
              f"| {np.mean([r['irrigation'] for r in group]):7.0f} mm "
              f"| {np.mean([r['stressed_days'] for r in group]):8.1f} "
              f"| {100 * np.mean([r['waited_rate'] for r in group]):5.0f}% (n={len(group)})")


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "noise"
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    if part == "noise":
        log(f"sensor noise with confirmation ({CONFIRM_DAYS} consecutive days)")
        rows = run_all(build_noise_runs(), NOISE_SEEDS, "results/final_noise.csv",
                       train_soils, train_seasons, test_soils, test_seasons)
    elif part == "weights":
        log(f"reward weight sensitivity: {STRESS_WEIGHTS}")
        rows = run_all(build_weight_runs(), WEIGHT_SEEDS, "results/final_weights.csv",
                       train_soils, train_seasons, test_soils, test_seasons)
    else:
        raise ValueError("use: python -m agents.final_experiments_2 noise|weights")

    print_table(rows)
    log("done")


if __name__ == "__main__":
    main()