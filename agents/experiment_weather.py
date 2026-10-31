"""Generalization to unseen weather, over several training seeds.
Run from the repo root:  python -m agents.experiment_weather
"""
import numpy as np 
from stable_baselines3 import PPO

from simulator.fao56 import SoilParams
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather
from agents.evaluate import evaluate

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 300_000 


def print_row(name, r):
    print(f"{name:22s} | irrigation {r['irrigation']:6.0f} mm | "
          f"stressed days {r['stressed_days']:5.1f} | reward {r['reward']:7.1f}")


def summarize(name, scores):
    for key in ["reward", "irrigation", "stressed_days"]:
        values = [s[key] for s in scores]
        print(f"{name:22s} {key:14s} {np.mean(values):7.1f} ± {np.std(values):.2f}")


def main():
    soil = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8)

    train_seasons = []
    for i in range(0, 100):
        train_seasons.append(synthetic_weather(125, i))

    test_seasons = []
    for i in range(100, 150):
        test_seasons.append(synthetic_weather(125, i))

    print("Number of training seasons:", len(train_seasons))
    print("Number of test seasons:", len(test_seasons))

    one_season = [train_seasons[0]]

    threshold = lambda obs, env: 4 if env.w < env.soil.w_stress else 0
    fixed_5mm = lambda obs, env: 1

    eval_threshold = evaluate(threshold, [soil], test_seasons)
    eval_fixed = evaluate(fixed_5mm, [soil], test_seasons)

    varied_scores = []
    one_season_scores = []


    for seed in SEEDS:
        varied_path = f"checkpoints/weather_experiment/ppo_mlp_varied_scaled_300k_seed{seed}"
        one_path = f"checkpoints/weather_experiment/ppo_mlp_one_season_scaled_100k_seed{seed}"

        env = IrrigationEnv([soil], train_seasons)
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(varied_path)
        ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]

        env_one = IrrigationEnv([soil], one_season)
        one_model = PPO("MlpPolicy", env_one, verbose=0, seed=seed)
        one_model.learn(total_timesteps=TOTAL_TIMESTEPS)
        one_model.save(one_path)
        one_ppo = lambda obs, env: one_model.predict(obs, deterministic=True)[0]

        eval_ppo = evaluate(ppo, [soil], test_seasons)
        varied_scores.append(eval_ppo)

        eval_one = evaluate(one_ppo, [soil], test_seasons)
        one_season_scores.append(eval_one)

        print(f"seed {seed} done") 
        print_row(f"  varied, seed {seed}", eval_ppo)
        print_row(f"  one season, seed {seed}", eval_one)


    print("\nRule-based (single run)")
    print_row("Threshold oracle", eval_threshold)
    print_row("Fixed 5 mm/day", eval_fixed)

    print(f"\nPPO, mean ± std over {len(SEEDS)} seeds")
    summarize("PPO, varied weather", varied_scores)
    summarize("PPO, one season", one_season_scores)


if __name__ == "__main__":
    main()