"""Stacking: a blind MLP agent that sees the last N days instead of only today.
Run from the repo root:  python -m agents.memory.experiment_stacking
"""
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents.evaluate import run_season
from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from simulator.memory.window_wrapper import WindowWrapper
from tests.test_fao56 import synthetic_weather

WINDOWS = [14, 30]
SEEDS = [0, 1, 2, 3, 4]
TIMESTEPS = 300_000
CKPT_DIR = "checkpoints/memory"

# earlier results on the same test set, for the gap closed
BLIND = -8.4
INFORMED = -5.3
REMEMBER_START = -6.0

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def make_env(soils, seasons, window):
    return WindowWrapper(IrrigationEnv(soils, seasons), window)


def get_model(path, window, seed, train_soils, train_seasons):
    if os.path.exists(path + ".zip"):
        return PPO.load(path), "loaded"
    env = make_env(train_soils, train_seasons, window)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(path)
    return model, "trained"


def evaluate_window(model, window, soils, seasons):
    def choose_action(obs, env):
        return model.predict(obs, deterministic=True)[0]

    results = []
    for soil in soils:
        for season in seasons:
            env = make_env([soil], [season], window)
            results.append(run_season(env, choose_action, seed=0))
    return {key: float(np.mean([r[key] for r in results]))
            for key in ["reward", "irrigation", "stressed_days"]}


def gap_closed(reward):
    return 100 * (reward - BLIND) / (INFORMED - BLIND)


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    os.makedirs(CKPT_DIR, exist_ok=True)
    for window in WINDOWS:
        check_env(make_env(train_soils, train_seasons, window))
    log("check_env passed")

    scores = {window: [] for window in WINDOWS}
    for seed in SEEDS:
        for window in WINDOWS:
            path = f"{CKPT_DIR}/ppo_mlp_window{window}_300k_seed{seed}"
            log(f"seed {seed}: window {window} days")
            model, status = get_model(path, window, seed, train_soils, train_seasons)
            result = evaluate_window(model, window, test_soils, test_seasons)
            scores[window].append(result)
            log(f"window {window:2d}, seed {seed} ({status}) | irrigation {result['irrigation']:4.0f} mm "
                f"| stressed days {result['stressed_days']:4.1f} | reward {result['reward']:6.2f}")

    print(f"\nStacking, blind agent, 20 unseen test soils, mean +- std over {len(SEEDS)} seeds")
    print(f"references: blind {BLIND}, remember-start {REMEMBER_START}, informed {INFORMED}")
    for window in WINDOWS:
        rewards = [s["reward"] for s in scores[window]]
        irrigation = np.mean([s["irrigation"] for s in scores[window]])
        stressed = np.mean([s["stressed_days"] for s in scores[window]])
        print(f"window {window:2d} days | reward {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} "
              f"| irrigation {irrigation:4.0f} mm | stressed days {stressed:4.1f} "
              f"| gap closed {gap_closed(np.mean(rewards)):4.0f}%")


if __name__ == "__main__":
    main()