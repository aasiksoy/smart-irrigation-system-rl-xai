import csv
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2, 3, 4]
TIMESTEPS = 300_000
CURVE_EVERY = 50_000

CKPT_DIR = "checkpoints/final"
RESULTS_CSV = "results/final_runs.csv"
CURVES_CSV = "results/final_curves.csv"
EPISODES_DIR = "results/final_episodes"

AGENTS = {
    "blind": {},
    "remember_start": {"remember_start": True},
    "stress_only": {"stress_memory": True},
    "stress_start": {"remember_start": True, "stress_memory": True},
    "informed": {"informed": True},
}

SETTINGS = {
    "fc_start": {},
    "random_start": {"start_range": (0.6, 1.0)},
}

NOISE = {
    "noise_low": {"sensor_miss": 0.1, "sensor_false": 0.01},
    "noise_mid": {"sensor_miss": 0.3, "sensor_false": 0.03},
    "noise_high": {"sensor_miss": 0.5, "sensor_false": 0.05},
}

CURVE_AGENTS = ["blind", "stress_only"]

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def build_runs():
    runs = []
    for setting, setting_flags in SETTINGS.items():
        for agent, agent_flags in AGENTS.items():
            runs.append({"name": f"{setting}_{agent}", "setting": setting, "agent": agent,
                         "flags": {**setting_flags, **agent_flags}})
    for noise, noise_flags in NOISE.items():
        runs.append({"name": f"fc_start_stress_only_{noise}", "setting": "fc_start",
                     "agent": f"stress_only_{noise}",
                     "flags": {**AGENTS["stress_only"], **noise_flags}})
    return runs


def get_model(path, flags, seed, train_soils, train_seasons):
    if os.path.exists(path + ".zip"):
        return PPO.load(path), False
    env = IrrigationEnv(train_soils, train_seasons, **flags)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(path)
    return model, True


def evaluate_model(model, soils, seasons, flags):
    rewards, irrigations, stressed, waited = [], [], [], []
    episode = 0
    for soil in soils:
        for season in seasons:
            env = IrrigationEnv([soil], [season], **flags)
            obs, info = env.reset(seed=episode)
            episode += 1
            total_reward = 0.0
            total_irrigation = 0.0
            stressed_days = 0
            stress_seen = False
            irrigated_before_stress = False
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                total_irrigation += info["irrigation"]
                if info["ks"] < 1.0:
                    stressed_days += 1
                    stress_seen = True
                if not stress_seen and info["irrigation"] > 0:
                    irrigated_before_stress = True
                done = terminated or truncated
            rewards.append(total_reward)
            irrigations.append(total_irrigation)
            stressed.append(stressed_days)
            waited.append(not irrigated_before_stress)
    return np.array(rewards), np.array(irrigations), np.array(stressed), np.array(waited)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    names = []
    for row in rows:
        if row["run"] not in names:
            names.append(row["run"])

    def mean_reward(run):
        return np.mean([r["reward"] for r in rows if r["run"] == run])

    print("\nFinal results, mean +- std over seeds")
    print(f"{'run':40s} | {'reward':>14s} | gap closed | irrigation | stressed | waited")
    for name in names:
        group = [r for r in rows if r["run"] == name]
        setting = group[0]["setting"]
        blind = mean_reward(f"{setting}_blind")
        informed = mean_reward(f"{setting}_informed")
        rewards = [r["reward"] for r in group]
        gap = 100 * (np.mean(rewards) - blind) / (informed - blind)
        irrigation = np.mean([r["irrigation"] for r in group])
        stressed = np.mean([r["stressed_days"] for r in group])
        waited = np.mean([r["waited_rate"] for r in group])
        print(f"{name:40s} | {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} | {gap:9.0f}% "
              f"| {irrigation:7.0f} mm | {stressed:8.1f} | {100 * waited:5.0f}% "
              f"(n={len(group)})")


def learning_curves(train_soils, train_seasons, test_soils, test_seasons):
    rows = []
    for agent in CURVE_AGENTS:
        flags = AGENTS[agent]
        folder = f"{CKPT_DIR}/curves/{agent}"
        last = f"{folder}/{agent}_{TIMESTEPS}_steps.zip"
        if not os.path.exists(last):
            log(f"learning curve: training {agent}")
            env = IrrigationEnv(train_soils, train_seasons, **flags)
            model = PPO("MlpPolicy", env, verbose=0, seed=0)
            saver = CheckpointCallback(save_freq=CURVE_EVERY, save_path=folder, name_prefix=agent)
            model.learn(total_timesteps=TIMESTEPS, callback=saver)
        for steps in range(CURVE_EVERY, TIMESTEPS + 1, CURVE_EVERY):
            model = PPO.load(f"{folder}/{agent}_{steps}_steps")
            rewards, _, _, _ = evaluate_model(model, test_soils, test_seasons, flags)
            rows.append({"agent": agent, "steps": steps, "reward": float(rewards.mean())})
            log(f"curve {agent} {steps} steps: reward {rewards.mean():.2f}")
    write_csv(CURVES_CSV, rows)


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    runs = build_runs()
    for run in runs:
        check_env(IrrigationEnv(train_soils, train_seasons, **run["flags"]))
    log(f"check_env passed for {len(runs)} configurations, {len(runs) * len(SEEDS)} runs in total")

    os.makedirs(EPISODES_DIR, exist_ok=True)
    rows = []
    total = len(runs) * len(SEEDS)
    done = 0
    for seed in SEEDS:
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
            write_csv(RESULTS_CSV, rows)
            status = "trained" if trained else "loaded"
            log(f"[{done}/{total}] {run['name']} seed {seed} ({status}): reward {rewards.mean():.2f}, "
                f"irrigation {irrigation.mean():.0f} mm, stressed {stressed.mean():.1f}, "
                f"waited {100 * waited.mean():.0f}%")

    summarize(rows)
    learning_curves(train_soils, train_seasons, test_soils, test_seasons)
    log("all done")


if __name__ == "__main__":
    main()