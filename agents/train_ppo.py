"""Train a PPO agent on IrrigationEnv and evaluate one season.
Run from the repo root:  python -m agents.train_ppo
"""

from stable_baselines3 import PPO
from simulator.fao56 import CropParams, SoilParams 
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather

TOTAL_TIMESTEPS = 100_000 
MODEL_PATH = "checkpoints/ppo_mlp_sandy"


def main():
    crop = CropParams()
    weather = synthetic_weather(crop.season_length)
    soil = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8) 

    env = IrrigationEnv(soil, [synthetic_weather(125)])
    model = PPO("MlpPolicy", env, verbose=1) 
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH) 

    eval_env = IrrigationEnv(soil, [weather])
    obs, info = eval_env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0
    total_irrigation = 0.0
    stressed = 0
    days = 0

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(int(action)) 
        total_reward += reward
        total_irrigation += info["irrigation"]
        stressed += info["ks"] < 1.0
        days += 1

    print(f"PPO (MLP), days {days}, irrigation {total_irrigation:6.0f} mm, stressed days {stressed:3d}, total reward {total_reward:7.1f}")

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()