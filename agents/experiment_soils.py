"""Generalization to unseen soils, over several training seeds.
Run from the repo root:  python -m agents.experiment_soils
"""
import numpy as np 
from stable_baselines3 import PPO

from simulator.fao56 import SoilParams, theta_from_w
from simulator.soils import make_soils
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather
from agents.evaluate import evaluate

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 300_000 

# SEEDS = [0]
# TOTAL_TIMESTEPS = 5_000


def print_row(name, r):
    print(f"{name:22s} | irrigation {r['irrigation']:6.0f} mm | "
          f"stressed days {r['stressed_days']:5.1f} | reward {r['reward']:7.1f}")


def summarize(name, scores):
    for key in ["reward", "irrigation", "stressed_days"]:
        values = [s[key] for s in scores]
        print(f"{name:22s} {key:14s} {np.mean(values):7.1f} ± {np.std(values):.2f}")


def main():

    sandy = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8)
 
    train_soils = make_soils(50, seed=0)
    trigger_theta = sum(s.w_stress / (1000 * s.root_depth_m) for s in train_soils) / len(train_soils)
    print(f"Fixed trigger: irrigate when theta < {trigger_theta:.3f}")
    
    test_soils = make_soils(20, seed=1000)

    train_seasons = []
    for i in range(0, 100):
        train_seasons.append(synthetic_weather(125, i))

    test_seasons = []
    for i in range(100, 110):
        test_seasons.append(synthetic_weather(125, i))


    threshold = lambda obs, env: 4 if env.w < env.soil.w_stress else 0
    fixed_5mm = lambda obs, env: 1
    fixed_trigger = lambda obs, env: 4 if theta_from_w(env.w, env.soil) < trigger_theta else 0

    eval_threshold = evaluate(threshold, test_soils, test_seasons)
    eval_fixed = evaluate(fixed_5mm, test_soils, test_seasons)
    eval_trigger = evaluate(fixed_trigger, test_soils, test_seasons)

    many_soils_scores = []
    one_soil_scores = []


    for seed in SEEDS:
        many_path = f"checkpoints/soil_experiment/ppo_mlp_many_soils_300k_seed{seed}"
        one_path = f"checkpoints/soil_experiment/ppo_mlp_one_soil_300k_seed{seed}"

        env = IrrigationEnv(train_soils, train_seasons)
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(many_path)
        ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]

        env_one = IrrigationEnv([sandy], train_seasons)
        one_model = PPO("MlpPolicy", env_one, verbose=0, seed=seed)
        one_model.learn(total_timesteps=TOTAL_TIMESTEPS)
        one_model.save(one_path)
        one_ppo = lambda obs, env: one_model.predict(obs, deterministic=True)[0]

        eval_ppo = evaluate(ppo, test_soils, test_seasons)
        many_soils_scores.append(eval_ppo)

        eval_one = evaluate(one_ppo, test_soils, test_seasons)
        one_soil_scores.append(eval_one)

        print(f"seed {seed} done") 
        print_row(f"many soils, seed {seed}", eval_ppo)
        print_row(f"one soil, seed {seed}", eval_one)


    print("\nRule-based (single run)")
    print_row("Threshold oracle", eval_threshold)
    print_row("Fixed 5 mm/day", eval_fixed)
    print_row("Fixed-trigger threshold", eval_trigger)

    print(f"\nPPO, mean ± std over {len(SEEDS)} seeds")
    summarize("PPO, many soils", many_soils_scores)
    summarize("PPO, one soil", one_soil_scores)


if __name__ == "__main__":
    main()