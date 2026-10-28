"""Train a PPO agent on IrrigationEnv on multiple seasons and evaluate on multiple season.
Run from the repo root:  python -m agents.experiment_weather
"""
from stable_baselines3 import PPO
from simulator.fao56 import SoilParams 
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather
from agents.evaluate import evaluate


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

    # env = IrrigationEnv(soil,train_seasons)
    # model = PPO("MlpPolicy", env, verbose=1, seed=SEED) 
    # model.learn(total_timesteps=TOTAL_TIMESTEPS)
    # model.save(NEW_MODEL_PATH) 
    model = PPO.load(NEW_MODEL_PATH)
    
    ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]
    threshold = lambda obs, env: 4 if env.w < env.soil.w_stress else 0
    old_model = PPO.load(OLD_MODEL_PATH)
    old_ppo = lambda obs, env: old_model.predict(obs, deterministic=True)[0]
    fixed_5mm = lambda obs, env: 1

    for name, strategy in [("PPO, varied weather", ppo),
                           ("Threshold oracle", threshold),
                           ("PPO, one season", old_ppo),
                           ("Fixed 5 mm/day", fixed_5mm)]:
        r = evaluate(strategy, soil, test_seasons)
        print(f"{name:20s} | irrigation {r['irrigation']:6.0f} mm | "
              f"stressed days {r['stressed_days']:5.1f} | reward {r['reward']:7.1f}")

if __name__ == "__main__":
    main()


SEEDS = [0, 1, 2]

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

    one_season = [train_seasons[0]]

    # 3. threshold and fixed: evaluate ONCE (no randomness -> no seeds needed), print

    varied_scores = []
    one_season_scores = []

    # 5. for seed in SEEDS:
    #       - train varied agent on train_seasons (seed=seed), save with seed in the name
    #       - train one-season agent on one_season (seed=seed), save with seed in the name
    #       - evaluate both on test_seasons
    #       - append each result dict to its bag

    # 6. after the loop: for each bag, mean and std of reward, irrigation, stressed days
    #    (np.mean / np.std over the per-seed values), print one row per agent