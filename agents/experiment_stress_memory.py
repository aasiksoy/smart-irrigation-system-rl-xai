"""Pipeline step B: remember the soil moisture when stress is first detected.

Stress is detected from the soil-moisture sensor alone (see IrrigationEnv._detect_stress).
Compares, on 20 unseen test soils:
  remember_start      step A only: the agent sees the moisture on day 0 (= field capacity)
  start_and_stress    steps A + B: also sees whether stress was detected and the moisture then
Run from the repo root:  python -m agents.experiment_stress_memory
"""
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2, 3, 4]
TIMESTEPS = 300_000
AGENTS = {
    "remember_start": {
        "flags": {"remember_start": True},
        "path": "checkpoints/soil_experiment/ppo_mlp_remember_start_300k_seed{seed}",
    },
    "start_and_stress": {
        "flags": {"remember_start": True, "stress_memory": True},
        "path": "checkpoints/pipeline/ppo_mlp_start_and_stress_300k_seed{seed}",
    },
}

BLIND = -8.4
INFORMED = -5.3

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def get_model(path, flags, seed, train_soils, train_seasons):
    if os.path.exists(path + ".zip"):
        return PPO.load(path), "loaded"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    env = IrrigationEnv(train_soils, train_seasons, **flags)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(path)
    return model, "trained"


def evaluate_model(model, flags, soils, seasons):
    """Play every soil x season; also record when stress was detected and how accurately."""
    rewards, irrigations, stressed, detected, detect_days, detect_errors = [], [], [], [], [], []
    for soil in soils:
        for season in seasons:
            env = IrrigationEnv([soil], [season], **flags)
            obs, info = env.reset(seed=0)
            total_reward, total_irrigation, stressed_days = 0.0, 0.0, 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                total_irrigation += info["irrigation"]
                stressed_days += info["ks"] < 1.0
                done = terminated or truncated
            rewards.append(total_reward)
            irrigations.append(total_irrigation)
            stressed.append(stressed_days)
            detected.append(env.stress_seen)
            if env.stress_seen:
                detect_days.append(env.stress_day)
                true_threshold = soil.w_stress / (1000 * soil.root_depth_m)
                detect_errors.append(env.theta_stress - true_threshold)
    result = {"reward": np.mean(rewards), "irrigation": np.mean(irrigations),
              "stressed_days": np.mean(stressed)}
    if flags.get("stress_memory"):
        result["detected"] = np.mean(detected)
        result["detect_day"] = np.median(detect_days) if detect_days else float("nan")
        result["detect_error"] = np.mean(detect_errors) if detect_errors else float("nan")
    return result


def gap_closed(reward):
    return 100 * (reward - BLIND) / (INFORMED - BLIND)


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    for agent in AGENTS.values():
        check_env(IrrigationEnv(train_soils, train_seasons, **agent["flags"]))
    log("check_env passed")

    scores = {name: [] for name in AGENTS}
    for seed in SEEDS:
        for name, agent in AGENTS.items():
            log(f"seed {seed}: {name}")
            model, status = get_model(agent["path"].format(seed=seed), agent["flags"], seed,
                                      train_soils, train_seasons)
            result = evaluate_model(model, agent["flags"], test_soils, test_seasons)
            scores[name].append(result)
            line = (f"{name}, seed {seed} ({status}) | irrigation {result['irrigation']:4.0f} mm "
                    f"| stressed days {result['stressed_days']:4.1f} | reward {result['reward']:6.2f}")
            if "detected" in result:
                line += (f" | stress detected in {100 * result['detected']:3.0f}% of seasons, "
                         f"median day {result['detect_day']:.0f}")
            log(line)

    print(f"\nPipeline A (+ B), 20 unseen test soils, mean +- std over {len(SEEDS)} seeds")
    print(f"references: blind {BLIND}, informed {INFORMED}")
    for name in AGENTS:
        group = scores[name]
        rewards = [s["reward"] for s in group]
        line = (f"{name:17s} | reward {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} "
                f"| irrigation {np.mean([s['irrigation'] for s in group]):4.0f} mm "
                f"| stressed days {np.mean([s['stressed_days'] for s in group]):4.1f} "
                f"| gap closed {gap_closed(np.mean(rewards)):4.0f}%")
        if "detected" in group[0]:
            line += (f" | detected {100 * np.mean([s['detected'] for s in group]):3.0f}% "
                     f"| theta error {np.nanmean([s['detect_error'] for s in group]):+.4f}")
        print(line)


if __name__ == "__main__":
    main()