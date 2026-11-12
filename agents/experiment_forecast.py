import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2]
TIMESTEPS = 300_000

AGENTS = {
    "basic": {"observe_et0": False, "forecast": "none"},
    "et0": {"observe_et0": True, "forecast": "none"},
    "et0_perfect_forecast": {"observe_et0": True, "forecast": "perfect"},
    "et0_noisy_forecast": {"observe_et0": True, "forecast": "noisy"},
}

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def run_eval(model, soils, seasons, flags):
    rewards, irrigations, stressed = [], [], []
    episode = 0
    for soil in soils:
        for season in seasons:
            env = IrrigationEnv([soil], [season], **flags)
            obs, info = env.reset(seed=episode)
            episode += 1
            total_reward = 0.0
            total_irrigation = 0.0
            stressed_days = 0
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
    return {"reward": np.mean(rewards), "irrigation": np.mean(irrigations),
            "stressed_days": np.mean(stressed)}


def print_row(name, r):
    print(f"{name:40s} | irrigation {r['irrigation']:4.0f} mm | stressed days {r['stressed_days']:4.1f} "
          f"| reward {r['reward']:6.2f}", flush=True)


def summarize(name, results):
    rewards = [r["reward"] for r in results]
    irrigation = np.mean([r["irrigation"] for r in results])
    stressed = np.mean([r["stressed_days"] for r in results])
    print(f"{name:40s} | reward {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} "
          f"| irrigation {irrigation:4.0f} mm | stressed days {stressed:4.1f}")


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    for flags in AGENTS.values():
        check_env(IrrigationEnv(train_soils, train_seasons, **flags))
    log("check_env passed")

    scores = {name: [] for name in AGENTS}
    scores["perfect forecast agent, tested with noisy forecast"] = []

    for seed in SEEDS:
        for name, flags in AGENTS.items():
            log(f"seed {seed}: training {name}")
            env = IrrigationEnv(train_soils, train_seasons, **flags)
            model = PPO("MlpPolicy", env, verbose=0, seed=seed)
            model.learn(total_timesteps=TIMESTEPS)
            model.save(f"checkpoints/forecast/ppo_mlp_{name}_{TIMESTEPS // 1000}k_seed{seed}")

            result = run_eval(model, test_soils, test_seasons, flags)
            scores[name].append(result)
            print_row(f"{name}, seed {seed}", result)

            if name == "et0_perfect_forecast":
                noisy_flags = {**flags, "forecast": "noisy"}
                result = run_eval(model, test_soils, test_seasons, noisy_flags)
                scores["perfect forecast agent, tested with noisy forecast"].append(result)
                print_row(f"  same agent, tested with noisy forecast, seed {seed}", result)

    print(f"\nForecast ablation, blind agent, 20 unseen test soils, mean +- std over {len(SEEDS)} seeds")
    for name, results in scores.items():
        summarize(name, results)


if __name__ == "__main__":
    main()