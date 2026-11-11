import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2]
TIMESTEPS = 300_000
EARLY_DAYS = 40
HOLD_MARGIN = 0.005

PROBE_FLAGS = {"remember_start": True, "stress_memory": True}
STRESS_ONLY_FLAGS = {"stress_memory": True}

REFERENCE = {
    "blind": -8.4,
    "remember_start": -6.0,
    "stress_memory (PPO)": -5.5,
    "informed": -5.3,
}

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def run_eval(policy, soils, seasons, **flags):
    rewards, irrigations, stressed, first_days = [], [], [], []
    episode = 0
    for soil in soils:
        for season in seasons:
            env = IrrigationEnv([soil], [season], **flags)
            obs, info = env.reset(seed=episode)
            episode += 1
            last_info = None
            total_reward = 0.0
            total_irrigation = 0.0
            stressed_days = 0
            first_stress = None
            done = False
            while not done:
                action = policy(obs, last_info)
                obs, reward, terminated, truncated, info = env.step(int(action))
                last_info = info
                total_reward += reward
                total_irrigation += info["irrigation"]
                if info["ks"] < 1.0:
                    stressed_days += 1
                    if first_stress is None:
                        first_stress = info["day"] - 1
                done = terminated or truncated
            rewards.append(total_reward)
            irrigations.append(total_irrigation)
            stressed.append(stressed_days)
            first_days.append(first_stress)

    early = [d for d in first_days if d is not None and d < EARLY_DAYS]
    return {
        "reward": np.mean(rewards),
        "irrigation": np.mean(irrigations),
        "stressed_days": np.mean(stressed),
        "probe_rate": len(early) / len(first_days),
        "first_stress_day": np.median(early) if early else float("nan"),
    }


def reactive_sensor_rule(obs, last_info):
    if last_info is not None and last_info["ks"] < 1.0:
        return 4
    return 0


def probe_and_hold_rule(obs, last_info):
    theta = obs[0] * 0.5
    stress_seen = obs[6]
    theta_stress = obs[7] * 0.5
    if stress_seen < 0.5:
        return 0
    if theta < theta_stress + HOLD_MARGIN:
        return 4
    return 0


def print_row(name, r):
    print(f"{name:30s} | reward {r['reward']:6.1f} | irrigation {r['irrigation']:4.0f} mm | "
          f"stressed {r['stressed_days']:4.1f} | early stress {100 * r['probe_rate']:3.0f}% "
          f"| first day {r['first_stress_day']:4.0f}", flush=True)


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    check_env(IrrigationEnv(train_soils, train_seasons, **STRESS_ONLY_FLAGS))
    log("check_env passed")

    log("rule baselines (same sensor as the PPO agent)")
    reactive = run_eval(reactive_sensor_rule, test_soils, test_seasons, **PROBE_FLAGS)
    print_row("rule: irrigate after stress", reactive)
    hold = run_eval(probe_and_hold_rule, test_soils, test_seasons, **PROBE_FLAGS)
    print_row("rule: probe, then hold", hold)

    stress_only_scores = []
    for seed in SEEDS:
        log(f"seed {seed}: training stress_only (no remember_start)")
        env = IrrigationEnv(train_soils, train_seasons, **STRESS_ONLY_FLAGS)
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        model.learn(total_timesteps=TIMESTEPS)
        model.save(f"checkpoints/probe/ppo_mlp_stress_only_{TIMESTEPS // 1000}k_seed{seed}")
        policy = lambda obs, last_info, model=model: model.predict(obs, deterministic=True)[0]
        result = run_eval(policy, test_soils, test_seasons, **STRESS_ONLY_FLAGS)
        stress_only_scores.append(result)
        print_row(f"stress_only, seed {seed}", result)

    blind = REFERENCE["blind"]
    informed = REFERENCE["informed"]

    def gap(reward):
        return 100 * (reward - blind) / (informed - blind)

    print("\nChecks for the probing result (start at field capacity)")
    print(f"{'rule: irrigate after stress':30s} | reward {reactive['reward']:6.1f} | gap closed {gap(reactive['reward']):4.0f}% "
          f"| {reactive['irrigation']:4.0f} mm | stressed {reactive['stressed_days']:4.1f}")
    print(f"{'rule: probe, then hold':30s} | reward {hold['reward']:6.1f} | gap closed {gap(hold['reward']):4.0f}% "
          f"| {hold['irrigation']:4.0f} mm | stressed {hold['stressed_days']:4.1f}")
    rewards = [r["reward"] for r in stress_only_scores]
    irrigation = np.mean([r["irrigation"] for r in stress_only_scores])
    stressed = np.mean([r["stressed_days"] for r in stress_only_scores])
    print(f"{'stress_only (PPO)':30s} | reward {np.mean(rewards):6.1f} +- {np.std(rewards):4.2f} "
          f"| gap closed {gap(np.mean(rewards)):4.0f}% | {irrigation:4.0f} mm | stressed {stressed:4.1f}")

    print("\nReferences:")
    for name, reward in REFERENCE.items():
        print(f"{name:30s} | reward {reward:6.1f} | gap closed {gap(reward):4.0f}%")


if __name__ == "__main__":
    main()