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
PROBE_BONUS = 1.0

AGENTS = {
    "stress_memory": {"remember_start": True, "stress_memory": True},
    "stress_memory_bonus": {"remember_start": True, "stress_memory": True, "probe_bonus": PROBE_BONUS},
}

REFERENCE = {
    "blind": -8.4,
    "remember_start": -6.0,
    "fc_wp (knows theta_wp)": -5.4,
    "informed": -5.3,
}

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def run_eval(model, soils, seasons, **flags):
    flags = {k: v for k, v in flags.items() if k != "probe_bonus"}
    rewards, irrigations, stressed, first_days = [], [], [], []
    episode = 0
    for soil in soils:
        for season in seasons:
            env = IrrigationEnv([soil], [season], **flags)
            obs, info = env.reset(seed=episode)
            episode += 1
            total_reward = 0.0
            total_irrigation = 0.0
            stressed_days = 0
            first_stress = None
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
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


def print_row(name, r):
    print(f"{name:30s} | reward {r['reward']:6.1f} | irrigation {r['irrigation']:4.0f} mm | "
          f"stressed {r['stressed_days']:4.1f} | early stress in {100 * r['probe_rate']:3.0f}% of seasons "
          f"| median first day {r['first_stress_day']:4.0f}", flush=True)


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    for flags in AGENTS.values():
        check_env(IrrigationEnv(train_soils, train_seasons, **flags))
    log("check_env passed")

    scores = {"remember_start (loaded)": []}
    scores.update({name: [] for name in AGENTS})

    for seed in SEEDS:
        old = PPO.load(f"checkpoints/soil_experiment/ppo_mlp_remember_start_{TIMESTEPS // 1000}k_seed{seed}")
        result = run_eval(old, test_soils, test_seasons, remember_start=True)
        scores["remember_start (loaded)"].append(result)
        print_row(f"remember_start (loaded), seed {seed}", result)

        for name, flags in AGENTS.items():
            log(f"seed {seed}: training {name}")
            env = IrrigationEnv(train_soils, train_seasons, **flags)
            model = PPO("MlpPolicy", env, verbose=0, seed=seed)
            model.learn(total_timesteps=TIMESTEPS)
            model.save(f"checkpoints/probe/ppo_mlp_{name}_{TIMESTEPS // 1000}k_seed{seed}")
            result = run_eval(model, test_soils, test_seasons, **flags)
            scores[name].append(result)
            print_row(f"{name}, seed {seed}", result)

    blind = REFERENCE["blind"]
    informed = REFERENCE["informed"]

    print(f"\nProbing experiment (start at field capacity), mean over {len(SEEDS)} seeds")
    for name, results in scores.items():
        rewards = [r["reward"] for r in results]
        gap = (np.mean(rewards) - blind) / (informed - blind)
        probe = np.mean([r["probe_rate"] for r in results])
        first = np.nanmean([r["first_stress_day"] for r in results])
        stressed = np.mean([r["stressed_days"] for r in results])
        irrigation = np.mean([r["irrigation"] for r in results])
        print(f"{name:30s} | reward {np.mean(rewards):6.1f} +- {np.std(rewards):4.2f} | gap closed {100 * gap:4.0f}% "
              f"| {irrigation:4.0f} mm | stressed {stressed:4.1f} | early stress {100 * probe:3.0f}% "
              f"| first day {first:4.0f}")

    print("\nReferences:")
    for name, reward in REFERENCE.items():
        gap = (reward - blind) / (informed - blind)
        print(f"{name:30s} | reward {reward:6.1f} | gap closed {100 * gap:4.0f}%")


if __name__ == "__main__":
    main()