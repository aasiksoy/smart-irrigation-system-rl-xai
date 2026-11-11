import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents.evaluate import evaluate
from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2]
TIMESTEPS = 300_000


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    check_env(IrrigationEnv(train_soils, train_seasons, remember_start=True))

    scores = []
    for seed in SEEDS:
        env = IrrigationEnv(train_soils, train_seasons, remember_start=True)
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        model.learn(total_timesteps=TIMESTEPS)
        model.save(f"checkpoints/soil_experiment/ppo_mlp_remember_start_{TIMESTEPS // 1000}k_seed{seed}")

        strategy = lambda obs, env: model.predict(obs, deterministic=True)[0]
        result = evaluate(strategy, test_soils, test_seasons, remember_start=True)
        scores.append(result)
        print(f"seed {seed} | irrigation {result['irrigation']:.0f} mm | "
              f"stressed days {result['stressed_days']:.1f} | reward {result['reward']:.1f}", flush=True)

    for key in ["reward", "irrigation", "stressed_days"]:
        values = [s[key] for s in scores]
        print(f"remember start {key}: {np.mean(values):.1f} +- {np.std(values):.2f}")

    blind, informed = -8.4, -5.3
    mean_reward = np.mean([s["reward"] for s in scores])
    print(f"closes {100 * (mean_reward - blind) / (informed - blind):.0f}% of the gap")


if __name__ == "__main__":
    main()