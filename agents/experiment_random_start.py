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
START_RANGE = (0.6, 1.0)

AGENTS = {
    "blind": {"informed": False, "remember_start": False},
    "remember_start": {"informed": False, "remember_start": True},
    "informed": {"informed": True, "remember_start": False},
}

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def print_row(name, r):
    print(f"{name:30s} | irrigation {r['irrigation']:6.0f} mm | "
          f"stressed days {r['stressed_days']:5.1f} | reward {r['reward']:7.1f}", flush=True)


def summarize(name, scores):
    for key in ["reward", "irrigation", "stressed_days"]:
        values = [s[key] for s in scores]
        print(f"{name:30s} {key:14s} {np.mean(values):7.1f} +- {np.std(values):.2f}")


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    for flags in AGENTS.values():
        check_env(IrrigationEnv(train_soils, train_seasons, start_range=START_RANGE, **flags))
    log(f"check_env passed, start range {START_RANGE}")

    scores = {name: [] for name in AGENTS}
    for seed in SEEDS:
        for name, flags in AGENTS.items():
            log(f"seed {seed}: training {name}")
            env = IrrigationEnv(train_soils, train_seasons, start_range=START_RANGE, **flags)
            model = PPO("MlpPolicy", env, verbose=0, seed=seed)
            model.learn(total_timesteps=TIMESTEPS)
            model.save(f"checkpoints/random_start/ppo_mlp_{name}_randstart_{TIMESTEPS // 1000}k_seed{seed}")

            strategy = lambda obs, env, model=model: model.predict(obs, deterministic=True)[0]
            result = evaluate(strategy, test_soils, test_seasons, start_range=START_RANGE, **flags)
            scores[name].append(result)
            print_row(f"{name}, seed {seed}", result)

    print(f"\nRandom start {START_RANGE}, mean +- std over {len(SEEDS)} seeds")
    for name in AGENTS:
        summarize(name, scores[name])

    blind = np.mean([s["reward"] for s in scores["blind"]])
    remember = np.mean([s["reward"] for s in scores["remember_start"]])
    informed = np.mean([s["reward"] for s in scores["informed"]])
    gap_closed = (remember - blind) / (informed - blind)
    print(f"\nremember_start closes {100 * gap_closed:.0f}% of the gap "
          f"(blind {blind:.1f} -> informed {informed:.1f}, remember start {remember:.1f})")


if __name__ == "__main__":
    main()