"""PPO with the CNN soil estimate as extra input (trained on the estimates, not the true soil).
Also tests the old informed agent with the estimates plugged in, for comparison.

Run from the repo root (after train_soil_cnn):  python -m agents.memory.experiment_estimator
Quick test (1-2 min):                          python -m agents.memory.experiment_estimator quick
"""
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents.evaluate import run_season
from simulator.irrigation_env import IrrigationEnv
from simulator.memory.soil_cnn import EstimateWrapper, load_soil_cnn
from simulator.memory.window_wrapper import WindowWrapper
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"

WINDOW = 30
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]
TIMESTEPS = 2_000 if QUICK else 300_000
CKPT_DIR = "checkpoints/memory_test" if QUICK else "checkpoints/memory"
CNN_PATH = f"{CKPT_DIR}/soil_cnn.pt"
INFORMED_PATH = "checkpoints/soil_experiment/ppo_mlp_informed_300k_seed{seed}"

# earlier results on the same test set
BLIND = -8.4
INFORMED = -5.3
REMEMBER_START = -6.0
STACKING_30 = -6.95

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def make_env(soils, seasons, cnn):
    return EstimateWrapper(WindowWrapper(IrrigationEnv(soils, seasons), WINDOW), *cnn)


def get_model(path, seed, cnn, train_soils, train_seasons):
    if os.path.exists(path + ".zip"):
        return PPO.load(path), "loaded"
    env = make_env(train_soils, train_seasons, cnn)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(path)
    return model, "trained"


def evaluate_model(model, cnn, soils, seasons):
    def choose_action(obs, env):
        return model.predict(obs, deterministic=True)[0]

    results = []
    for soil in soils:
        for season in seasons:
            results.append(run_season(make_env([soil], [season], cnn), choose_action, seed=0))
    return {key: float(np.mean([r[key] for r in results]))
            for key in ["reward", "irrigation", "stressed_days"]}


def gap_closed(reward):
    return 100 * (reward - BLIND) / (INFORMED - BLIND)


def show(name, result):
    log(f"{name} | irrigation {result['irrigation']:4.0f} mm "
        f"| stressed days {result['stressed_days']:4.1f} | reward {result['reward']:6.2f}")


def summary(name, scores):
    rewards = [s["reward"] for s in scores]
    irrigation = np.mean([s["irrigation"] for s in scores])
    stressed = np.mean([s["stressed_days"] for s in scores])
    print(f"{name:32s} (n={len(scores)}) | reward {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} "
          f"| irrigation {irrigation:4.0f} mm | stressed days {stressed:4.1f} "
          f"| gap closed {gap_closed(np.mean(rewards)):4.0f}%")


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    cnn = load_soil_cnn(CNN_PATH)
    check_env(make_env(train_soils, train_seasons, cnn))
    log("check_env passed")

    trained_scores = []
    plugin_scores = []
    for seed in SEEDS:
        path = f"{CKPT_DIR}/ppo_mlp_cnn_estimate_window{WINDOW}_300k_seed{seed}"
        log(f"seed {seed}: PPO trained on CNN estimates")
        model, status = get_model(path, seed, cnn, train_soils, train_seasons)
        result = evaluate_model(model, cnn, test_soils, test_seasons)
        trained_scores.append(result)
        show(f"trained on estimates, seed {seed} ({status})", result)

        informed_path = INFORMED_PATH.format(seed=seed)
        if os.path.exists(informed_path + ".zip"):
            informed = PPO.load(informed_path)
            result = evaluate_model(informed, cnn, test_soils, test_seasons)
            plugin_scores.append(result)
            show(f"informed agent + estimates, seed {seed}", result)

    print(f"\nCNN soil estimate (window {WINDOW}), 20 unseen test soils, "
          f"mean +- std over {len(SEEDS)} seeds")
    print(f"references: blind {BLIND}, stacking 30 days {STACKING_30}, "
          f"remember-start {REMEMBER_START}, informed {INFORMED}")
    summary("PPO trained on estimates", trained_scores)
    summary("informed agent + estimates", plugin_scores)


if __name__ == "__main__":
    main()