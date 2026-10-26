import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.fao56 import SoilParams, CropParams, step_day, theta_from_w
from typing import Optional

class IrrigationEnv(gym.Env):

    IRRIGATION_LEVELS = [0.0, 5.0, 10.0, 15.0, 20.0]

    def __init__(self, soil:SoilParams, weather, crop: CropParams = None):
        """
        soil: the field (SoilParams)
        weather: list of (rain_mm, et0_mm) per day, one entry per season day
        crop: crop curve; maize defults if not given
        """
        super().__init__()
        self.soil = soil
        self.weather = weather
        self.crop = crop if crop is not None else CropParams()

        #picking one of the irrigation levels
        self.action_space = spaces.Discrete(len(self.IRRIGATION_LEVELS))
        # what the agent can see
        # sensor theta, ET0 today, rain today, rain tomorrow, season length]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 15.0, 200.0, 200.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )


    def _get_obs(self):

        theta = theta_from_w(self.w, self.soil)
        rain_today, et0 = self.weather[self.day]
        if len(self.weather) == self.day + 1 :
            rain_tomorrow = 0.0
        else:
            rain_tomorrow, _ = self.weather[self.day + 1]

        season_length = self.crop.season_length
        season_progress = self.day / season_length

        return np.array([theta, et0, rain_today, rain_tomorrow, season_progress], dtype = np.float32)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        self.w = self.soil.w_fc
        self.day = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info


    def _get_info(self):
        """Extra information for debugging and plots, not seen by the agent."""
        return {"day": self.day, "water_mm": self.w}


    def step(self, action):
#NOT DONE YETTTTTTT
        irrigation = self.IRRIGATION_LEVELS[action]
        rain_today, et0 = self.weather[self.day]

        self.w, day_log = step_day(self.w, rain_today, irrigation, et0, self.day, self.soil, self.crop)
        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        terminated = True if self.season_progress == 1 else False
        truncated = False

        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

