import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WindowWrapper(gym.Wrapper):
    """Gives the agent the last `window` days instead of only today.

    Each day in the window holds that day's normal observation plus the irrigation
    given the day before (scaled to 0-1), so the agent can tell rain from its own watering.
    The oldest day comes first, today comes last. At the start of a season the window
    is filled with copies of day 0.
    """

    def __init__(self, env, window=7):
        super().__init__(env)
        self.window = window
        self.max_irrigation = max(env.unwrapped.IRRIGATION_LEVELS)
        low = np.append(env.observation_space.low, 0.0)
        high = np.append(env.observation_space.high, 1.0)
        self.observation_space = spaces.Box(low=np.tile(low, window).astype(np.float32),
                                            high=np.tile(high, window).astype(np.float32),
                                            dtype=np.float32)
        self.history = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        today = np.append(obs, 0.0).astype(np.float32)
        self.history = [today] * self.window
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        today = np.append(obs, info["irrigation"] / self.max_irrigation).astype(np.float32)
        self.history = self.history[1:] + [today]
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self):
        return np.concatenate(self.history).astype(np.float32)