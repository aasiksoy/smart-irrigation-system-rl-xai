"""How accurate must the soil knowledge be? (no training, evaluation only)

The informed agent (trained on the true soil) is tested with an error added to theta_wp and
theta_fc, like a soil map that is off. The error is drawn once per season and stays the same.
Error levels are fractions of the CNN's typical error (theta_fc 0.0175, theta_wp 0.031).

Run from the repo root:  python -m agents.memory.noisy_informed
"""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from agents.evaluate import run_season
from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2]
INFORMED_PATH = "checkpoints/soil_experiment/ppo_mlp_informed_300k_seed{seed}"
CNN_ERROR_FC = 0.0175
CNN_ERROR_WP = 0.031
LEVELS = [0.0, 0.25, 0.5, 1.0]
THETA_SCALE = 0.5
WP_INDEX, FC_INDEX = 5, 6

BLIND = -8.4
INFORMED = -5.3


class SoilErrorWrapper(gym.Wrapper):
    """Adds a fixed error (drawn at reset) to the theta_wp and theta_fc the agent sees."""

    def __init__(self, env, error_fc, error_wp, rng):
        super().__init__(env)
        self.error_fc = error_fc
        self.error_wp = error_wp
        self.rng = rng
        self.offset = np.zeros(2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.offset = self.rng.normal(0.0, [self.error_wp, self.error_fc]) / THETA_SCALE
        return self._noisy(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            obs = self._noisy(obs)
        return obs, reward, terminated, truncated, info

    def _noisy(self, obs):
        obs = obs.copy()
        obs[WP_INDEX] = np.clip(obs[WP_INDEX] + self.offset[0], 0.0, 1.0)
        obs[FC_INDEX] = np.clip(obs[FC_INDEX] + self.offset[1], 0.0, 1.0)
        return obs


def evaluate_noisy(model, level, soils, seasons):
    rng = np.random.default_rng(0)

    def choose_action(obs, env):
        return model.predict(obs, deterministic=True)[0]

    results = []
    for soil in soils:
        for season in seasons:
            env = SoilErrorWrapper(IrrigationEnv([soil], [season], informed=True),
                                   level * CNN_ERROR_FC, level * CNN_ERROR_WP, rng)
            results.append(run_season(env, choose_action, seed=0))
    return {key: float(np.mean([r[key] for r in results]))
            for key in ["reward", "irrigation", "stressed_days"]}


def main():
    test_soils = make_soils(20, seed=1000)
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    scores = {level: [] for level in LEVELS}
    for seed in SEEDS:
        model = PPO.load(INFORMED_PATH.format(seed=seed))
        for level in LEVELS:
            result = evaluate_noisy(model, level, test_soils, test_seasons)
            scores[level].append(result)
            print(f"seed {seed}, error {level:4.2f} x CNN | irrigation {result['irrigation']:4.0f} mm "
                  f"| stressed days {result['stressed_days']:4.1f} | reward {result['reward']:6.2f}",
                  flush=True)

    print(f"\nInformed agent with soil error, 20 unseen test soils, mean +- std over {len(SEEDS)} seeds")
    print(f"references: blind {BLIND}, informed {INFORMED}")
    for level in LEVELS:
        rewards = [s["reward"] for s in scores[level]]
        irrigation = np.mean([s["irrigation"] for s in scores[level]])
        stressed = np.mean([s["stressed_days"] for s in scores[level]])
        print(f"error {level:4.2f} x CNN (fc {level * CNN_ERROR_FC:.4f}, wp {level * CNN_ERROR_WP:.4f}) "
              f"| reward {np.mean(rewards):6.2f} +- {np.std(rewards):4.2f} "
              f"| irrigation {irrigation:4.0f} mm | stressed days {stressed:4.1f}")


if __name__ == "__main__":
    main()