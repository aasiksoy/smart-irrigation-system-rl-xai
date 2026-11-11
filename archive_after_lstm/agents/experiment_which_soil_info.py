import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents.evaluate import evaluate
from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2]
TIMESTEPS = 300_000

AGENTS = {
    "fc_only": ["theta_fc"],
    "wp_only": ["theta_wp"],
    "drain_only": ["drain_rate"],
    "fc_wp": ["theta_wp", "theta_fc"],
    "fc_drain": ["theta_fc", "drain_rate"],
}

REFERENCE = {
    "blind": -8.4,
    "remember_start": -6.0,
    "informed (all three)": -5.3,
}

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def print_row(name, r):
    print(f"{name:25s} | irrigation {r['irrigation']:6.0f} mm | "
          f"stressed days {r['stressed_days']:5.1f} | reward {r['reward']:7.1f}", flush=True)


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    for params in AGENTS.values():
        check_env(IrrigationEnv(train_soils, train_seasons, informed_params=params))
    log("check_env passed")

    scores = {name: [] for name in AGENTS}
    for seed in SEEDS:
        for name, params in AGENTS.items():
            log(f"seed {seed}: training {name}")
            env = IrrigationEnv(train_soils, train_seasons, informed_params=params)
            model = PPO("MlpPolicy", env, verbose=0, seed=seed)
            model.learn(total_timesteps=TIMESTEPS)
            model.save(f"checkpoints/which_soil_info/ppo_mlp_{name}_{TIMESTEPS // 1000}k_seed{seed}")

            strategy = lambda obs, env, model=model: model.predict(obs, deterministic=True)[0]
            result = evaluate(strategy, test_soils, test_seasons, informed_params=params)
            scores[name].append(result)
            print_row(f"{name}, seed {seed}", result)

    blind = REFERENCE["blind"]
    informed = REFERENCE["informed (all three)"]

    print(f"\nWhich soil information matters? Mean +- std over {len(SEEDS)} seeds")
    print(f"{'agent':25s} | {'reward':>14s} | {'irrigation':>10s} | {'stressed':>8s} | gap closed")
    for name in AGENTS:
        rewards = [s["reward"] for s in scores[name]]
        irrigation = np.mean([s["irrigation"] for s in scores[name]])
        stressed = np.mean([s["stressed_days"] for s in scores[name]])
        gap = (np.mean(rewards) - blind) / (informed - blind)
        print(f"{name:25s} | {np.mean(rewards):6.1f} +- {np.std(rewards):4.2f} | "
              f"{irrigation:7.0f} mm | {stressed:8.1f} | {100 * gap:5.0f}%")

    print("\nEarlier results for comparison:")
    for name, reward in REFERENCE.items():
        gap = (reward - blind) / (informed - blind)
        print(f"{name:25s} | {reward:6.1f} | gap closed {100 * gap:5.0f}%")


if __name__ == "__main__":
    main()