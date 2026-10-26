from stable_baselines3.common.env_checker import check_env
from simulator.fao56 import SoilParams
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather

soil = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8)
env = IrrigationEnv(soil, synthetic_weather(125))

# 1. Official check: does the env follow the Gymnasium rules?
check_env(env)
print("check_env passed")

# 2. Run one season per fixed action, and compare
for action in range(5):
    obs, info = env.reset(seed=0)
    total_reward, total_irr, stressed, days = 0.0, 0.0, 0, 0
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_irr += info["irrigation"]
        stressed += info["ks"] < 1.0
        days += 1
    print(f"always action {action} ({env.IRRIGATION_LEVELS[action]:4.0f} mm/day) | "
          f"days {days} | irrigation {total_irr:6.0f} mm | "
          f"stressed days {stressed:3d} | total reward {total_reward:7.1f}")