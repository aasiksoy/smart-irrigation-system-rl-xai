"""Main pipeline results, 5 seeds: blind, informed, step A, steps A + B.

  synthetic  : synthetic weather (train seasons 0-99, test seasons 100-109)
  openmeteo  : real weather, Leuven / Nicosia / Seville (train 1990-2015, test 2016-2023)

Both use 50 train soils (seed 0) and 20 unseen test soils (seed 1000).
Models that already exist are loaded, not retrained.

Add "random_start" to start each season at a random 60-100% of the plant-available water
instead of at field capacity (tests the main assumption of step A).

Run from the repo root:
    python -m agents.experiment_pipeline synthetic
    python -m agents.experiment_pipeline openmeteo
    python -m agents.experiment_pipeline synthetic random_start
"""
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from simulator.fao56 import CropParams
from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from simulator.weather import load_seasons
from tests.test_fao56 import synthetic_weather

SEEDS = list(range(10))
TIMESTEPS = 300_000
RANDOM_START = (0.6, 1.0)
T_95 = 2.262   # t-value for a 95% confidence interval with 10 seeds (9 degrees of freedom)

AGENTS = {
    "blind": {},
    "informed": {"informed": True},
    "start (A)": {"remember_start": True},
    "start + stress (A+B)": {"remember_start": True, "stress_memory": True},
}

# existing synthetic-weather models, so they are loaded instead of retrained
SYNTHETIC_PATHS = {
    "blind": "checkpoints/soil_experiment/ppo_mlp_many_soils_300k_seed{seed}",
    "informed": "checkpoints/soil_experiment/ppo_mlp_informed_300k_seed{seed}",
    "start (A)": "checkpoints/soil_experiment/ppo_mlp_remember_start_300k_seed{seed}",
    "start + stress (A+B)": "checkpoints/pipeline/ppo_mlp_start_and_stress_300k_seed{seed}",
}
OPENMETEO_PATHS = {
    "blind": "checkpoints/pipeline_openmeteo/ppo_mlp_blind_300k_seed{seed}",
    "informed": "checkpoints/pipeline_openmeteo/ppo_mlp_informed_300k_seed{seed}",
    "start (A)": "checkpoints/pipeline_openmeteo/ppo_mlp_start_300k_seed{seed}",
    "start + stress (A+B)": "checkpoints/pipeline_openmeteo/ppo_mlp_start_and_stress_300k_seed{seed}",
}

PLANTING = {"leuven": "05-01", "nicosia": "04-01", "seville": "04-15"}
TRAIN_YEARS = range(1990, 2016)
TEST_YEARS = range(2016, 2024)

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def threshold_rule(obs, env):
    """Knows the soil: irrigate 20 mm when the water drops below the stress threshold."""
    return 4 if env.w < env.soil.w_stress else 0


def get_model(path, flags, seed, train_soils, train_seasons):
    if os.path.exists(path + ".zip"):
        return PPO.load(path), "loaded"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    env = IrrigationEnv(train_soils, train_seasons, **flags)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(path)
    return model, "trained"


def evaluate(choose_action, flags, soils, seasons):
    """Play every soil x season once; return the averages."""
    rewards, irrigations, stressed, detected = [], [], [], []
    episode = 0
    for soil in soils:
        for season in seasons:
            env = IrrigationEnv([soil], [season], **flags)
            obs, info = env.reset(seed=episode)
            episode += 1
            total_reward, total_irrigation, stressed_days = 0.0, 0.0, 0
            done = False
            while not done:
                action = choose_action(obs, env)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                total_irrigation += info["irrigation"]
                stressed_days += info["ks"] < 1.0
                done = terminated or truncated
            rewards.append(total_reward)
            irrigations.append(total_irrigation)
            stressed.append(stressed_days)
            detected.append(env.stress_seen)
    return {"reward": float(np.mean(rewards)), "irrigation": float(np.mean(irrigations)),
            "stressed_days": float(np.mean(stressed)), "detected": float(np.mean(detected))}


def print_table(title, scores, threshold):
    """scores: {agent: [result per seed]}; gap closed is computed from this run's blind/informed."""
    blind = np.mean([s["reward"] for s in scores["blind"]])
    informed = np.mean([s["reward"] for s in scores["informed"]])
    print(f"\n{title}, mean +- std over {len(SEEDS)} seeds")
    print(f"{'agent':22s} | {'reward':>14s} | {'95% CI':>16s} | irrigation | stressed days | gap closed")
    for name, group in scores.items():
        rewards = [s["reward"] for s in group]
        gap = 100 * (np.mean(rewards) - blind) / (informed - blind)
        half = T_95 * np.std(rewards, ddof=1) / np.sqrt(len(rewards))
        line = (f"{name:22s} | {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} "
                f"| [{np.mean(rewards) - half:6.2f}, {np.mean(rewards) + half:6.2f}] "
                f"| {np.mean([s['irrigation'] for s in group]):7.0f} mm "
                f"| {np.mean([s['stressed_days'] for s in group]):13.1f} | {gap:9.0f}%")
        if AGENTS[name].get("stress_memory"):
            line += f" | stress detected {100 * np.mean([s['detected'] for s in group]):3.0f}%"
        print(line)
    print(f"{'threshold rule':22s} | {threshold['reward']:6.2f}         | {'':16s} "
          f"| {threshold['irrigation']:7.0f} mm | {threshold['stressed_days']:13.1f} | (knows the soil)")


def main():
    weather = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    random_start = "random_start" in sys.argv[2:]
    if weather not in ("synthetic", "openmeteo"):
        raise ValueError("use: python -m agents.experiment_pipeline synthetic|openmeteo [random_start]")
    start_flags = {"start_range": RANDOM_START} if random_start else {}

    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)

    if weather == "synthetic":
        train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
        test_sets = {"synthetic": [synthetic_weather(125, i) for i in range(100, 110)]}
        paths = SYNTHETIC_PATHS
    else:
        length = CropParams().season_length
        train_seasons = [season for site, start in PLANTING.items()
                         for season in load_seasons(site, TRAIN_YEARS, start, length)]
        test_sets = {site: load_seasons(site, TEST_YEARS, start, length)
                     for site, start in PLANTING.items()}
        paths = OPENMETEO_PATHS
    if random_start:
        paths = {name: f"checkpoints/pipeline_random_start/{weather}/" + path.split("/")[-1]
                 for name, path in paths.items()}
        paths = {name: path.replace("many_soils", "blind") for name, path in paths.items()}
    agents = {name: {**flags, **start_flags} for name, flags in AGENTS.items()}
    log(f"{weather} weather, {'random start' if random_start else 'start at field capacity'}: "
        f"{len(train_seasons)} training seasons, "
        + ", ".join(f"{site} {len(seasons)} test seasons" for site, seasons in test_sets.items()))

    for flags in agents.values():
        check_env(IrrigationEnv(train_soils, train_seasons, **flags))
    log("check_env passed")

    scores = {site: {name: [] for name in AGENTS} for site in test_sets}
    for seed in SEEDS:
        for name, flags in agents.items():
            model, status = get_model(paths[name].format(seed=seed), flags, seed,
                                      train_soils, train_seasons)

            def choose_action(obs, env):
                return model.predict(obs, deterministic=True)[0]

            for site, seasons in test_sets.items():
                result = evaluate(choose_action, flags, test_soils, seasons)
                scores[site][name].append(result)
                log(f"{name}, seed {seed} ({status}), {site} | irrigation {result['irrigation']:4.0f} mm "
                    f"| stressed days {result['stressed_days']:4.1f} | reward {result['reward']:7.2f}")

    for site, seasons in test_sets.items():
        threshold = evaluate(threshold_rule, start_flags, test_soils, seasons)
        print_table(f"{site}: 20 unseen test soils x {len(seasons)} test seasons",
                    scores[site], threshold)


if __name__ == "__main__":
    main()