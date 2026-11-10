import os
import time

import numpy as np
from stable_baselines3 import PPO

from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

MODEL_PATH = "checkpoints/soil_experiment/ppo_mlp_informed_300k_seed0"
OUT_DIR = "data"
N_TRAIN_SOILS = 5000

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def collect_episode(model, env):
    obs, info = env.reset(seed=0)
    obs_list = []
    action_list = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[:5].copy())
        action_list.append(int(action))
        obs, reward, terminated, truncated, info = env.step(int(action))
    return obs_list, action_list


def collect(model, pairs, name):
    all_obs = []
    all_actions = []
    all_soil = []
    total = len(pairs)

    for soil, season in pairs:
        env = IrrigationEnv([soil], [season], informed=True)
        episode_obs, episode_actions = collect_episode(model, env)
        all_obs.append(episode_obs)
        all_actions.append(episode_actions)
        all_soil.append([soil.theta_wp, soil.theta_fc, soil.drain_rate])

        if len(all_obs) % 500 == 0:
            log(f"{name}: {len(all_obs)} / {total} episodes")

    obs = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)
    soil = np.array(all_soil, dtype=np.float32)

    theta_day0 = obs[:, 0, 0] * IrrigationEnv.OBS_SCALE[0]
    assert np.allclose(theta_day0, soil[:, 1], atol=1e-5), "theta on day 0 does not match theta_fc"

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"soil_dataset_{name}.npz")
    np.savez(path, obs=obs, actions=actions, soil=soil)
    log(f"{name}: obs {obs.shape}, actions {actions.shape}, soil {soil.shape} saved to {path}")


def main():
    model = PPO.load(MODEL_PATH)
    log(f"loaded informed agent from {MODEL_PATH}")

    train_soils = make_soils(N_TRAIN_SOILS, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    train_pairs = [(soil, train_seasons[k % 100]) for k, soil in enumerate(train_soils)]
    test_pairs = [(soil, season) for soil in test_soils for season in test_seasons]

    collect(model, train_pairs, "train")
    collect(model, test_pairs, "test")
    log("done")


if __name__ == "__main__":
    main()