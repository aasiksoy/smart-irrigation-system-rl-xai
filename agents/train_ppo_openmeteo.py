"""Train a PPO agent on IrrigationEnv with OpenMeteo weather data.
Evaluation on held out years (test years).
Run from the repo root:  python -m agents.train_ppo_openmeteo
"""
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from simulator.fao56 import CropParams, SoilParams 
from simulator.irrigation_env import IrrigationEnv
from simulator.weather import load_seasons
from simulator.soils import make_soils

from agents.evaluate import run_season



TOTAL_TIMESTEPS = 300_000
SEED = 0
MODEL_PATH = f"checkpoints/openmeteo_experiment/ppo_mlp_informed_seed{SEED}"

PLANTING = {"leuven": "05-01", "nicosia": "04-01", "seville": "04-15"}
TRAIN_YEARS = range(1990, 2016)
TEST_YEARS = range(2016, 2024)


def main():
    crop = CropParams()
    length = crop.season_length
    
    train_soils = make_soils(50, seed=100)
    test_soils = make_soils(20, seed=200)      # different seed, unseen soils

    train = [s for site, start in PLANTING.items()
             for s in load_seasons(site, TRAIN_YEARS, start, length)]
    test = {site: load_seasons(site, TEST_YEARS, start, length)
            for site, start in PLANTING.items()}

    env = Monitor(IrrigationEnv(train_soils, train, crop, informed=True))
    model = PPO("MlpPolicy", env, verbose=1, seed=SEED)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)

    ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]
    for site, seasons in test.items():
        rows = []
        for season in seasons:
            for soil in test_soils:
                eval_env = IrrigationEnv([soil], [season], crop, informed=True)
                r = run_season(eval_env, ppo, seed=0)
                rows.append((r["irrigation"], r["stressed_days"], r["reward"]))
                eval_env.close()
        irr, stress, rew = np.mean(rows, axis=0)
        print(f"{site:8s} irrigation {irr:6.0f} mm, stressed days {stress:5.1f}, "
              f"reward {rew:7.1f}  ({len(rows)} episodes)")
    env.close()

if __name__ == "__main__":
    main()