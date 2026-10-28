"""Train a PPO agent on IrrigationEnv on multiple seasons and evaluate on multiple season.
Run from the repo root:  python -m agents.experiment_weather
"""
from stable_baselines3 import PPO
from simulator.fao56 import SoilParams 
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather
from agents.evaluate import run_season


TOTAL_TIMESTEPS = 300_000
SEED = 0
OLD_MODEL_PATH = f"checkpoints/ppo_mlp_sandy_scaled_seed{SEED}"
NEW_MODEL_PATH = f"checkpoints/weather_experiment/ppo_mlp_varied_scaled_300k_seed{SEED}"


def main():

    soil = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8) 
    
    train_seasons= [] 
    for i in range(0,100):
        train_seasons.append(synthetic_weather(125,i))

    test_seasons= []  
    for i in range(100,150):
        test_seasons.append(synthetic_weather(125,i))

    print("Number of the training seasons ", len(train_seasons))
    print("Number of the test seasons ", len(test_seasons))

    env = IrrigationEnv(soil,train_seasons)
    model = PPO("MlpPolicy", env, verbose=1, seed=SEED) 
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(NEW_MODEL_PATH) 
   # model = PPO.load(NEW_MODEL_PATH)

    results = []
    ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]
    for season in test_seasons:
        eval_env = IrrigationEnv(soil, [season])
        result = run_season(eval_env, ppo, seed=0)
        results.append(result)
    
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_irrigation =sum(r["irrigation"] for r in results) / len(results)
    avg_stressed_days = sum(r["stressed_days"] for r in results) / len(results)

    print(f"PPO (varied weather), avg over {len(results)} test seasons,"
          f"irrigation {avg_irrigation:6.0f} mm, "
          f"stressed days {avg_stressed_days:5.1f},  reward {avg_reward:7.1f}")
    eval_env.close()
    #env.close()

    #oracle
    threshold = lambda obs, env: 4 if env.w < env.soil.w_stress else 0

    results = []
    for season in test_seasons:
        eval_env = IrrigationEnv(soil, [season])
        result = run_season(eval_env, threshold, seed=0)
        results.append(result)

    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_irrigation = sum(r["irrigation"] for r in results) / len(results)
    avg_stressed_days = sum(r["stressed_days"] for r in results) / len(results)

    print(f"Threshold oracle, avg over {len(results)} test seasons,  "
          f"irrigation {avg_irrigation:6.0f} mm,  "
          f"stressed days {avg_stressed_days:5.1f},  reward {avg_reward:7.1f}")


    #old model
    old_model = PPO.load(OLD_MODEL_PATH)
    old_ppo = lambda obs, env: old_model.predict(obs, deterministic=True)[0]
    
    results = []
    for season in test_seasons:
        eval_env = IrrigationEnv(soil, [season])
        result = run_season(eval_env, old_ppo, seed=0)
        results.append(result)

    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_irrigation = sum(r["irrigation"] for r in results) / len(results)
    avg_stressed_days = sum(r["stressed_days"] for r in results) / len(results)

    print(f"PPO one season (old), avg over {len(results)} test seasons,  "
          f"irrigation {avg_irrigation:6.0f} mm,"
          f"stressed days {avg_stressed_days:5.1f},  reward {avg_reward:7.1f}")


    #fixed 5 mm every day
    fixed_5mm = lambda obs, env: 1

    results = []
    for season in test_seasons:
        eval_env = IrrigationEnv(soil, [season])
        result = run_season(eval_env, fixed_5mm, seed=0)
        results.append(result)

    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_irrigation = sum(r["irrigation"] for r in results) / len(results)
    avg_stressed_days = sum(r["stressed_days"] for r in results) / len(results)

    print(f"Fixed 5 mm/day, avg over {len(results)} test seasons,  "
          f"irrigation {avg_irrigation:6.0f} mm,  "
          f"stressed days {avg_stressed_days:5.1f},  reward {avg_reward:7.1f}")

if __name__ == "__main__":
    main()