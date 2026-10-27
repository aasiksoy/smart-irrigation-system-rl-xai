"""Train a PPO agent on IrrigationEnv and evaluate one season.
Run from the repo root:  python -m agents.train_ppo
"""

from stable_baselines3 import PPO
from simulator.fao56 import CropParams, SoilParams 
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather
from agents.evaluate import run_season

TOTAL_TIMESTEPS = 100_000
SEED = 0
MODEL_PATH = f"checkpoints/ppo_mlp_sandy_seed{SEED}"


def main():
    crop = CropParams()
    weather = synthetic_weather(crop.season_length)
    soil = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8) 

    env = IrrigationEnv(soil, [synthetic_weather(125)])
    model = PPO("MlpPolicy", env, verbose=1, seed=SEED) 
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH) 

    eval_env = IrrigationEnv(soil, [weather])

    ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]
    result = run_season(eval_env, ppo, seed=0)
    print(f"PPO (MLP), seed {SEED}, days {result['days']}, irrigation {result['irrigation']:6.0f} mm," 
        f"stressed days {result['stressed_days']:3d}, total reward {result['reward']:7.1f}")
    eval_env.close()
    env.close()

if __name__ == "__main__":
    main()