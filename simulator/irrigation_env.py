import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.fao56 import SoilParams, CropParams, step_day, theta_from_w
from typing import Optional

class IrrigationEnv(gym.Env):

    IRRIGATION_LEVELS = [0.0, 5.0, 10.0, 15.0, 20.0]
    OBS_SCALE = np.array([0.5, 10.0, 50.0, 50.0, 1.0], dtype=np.float32)

    def __init__(self, soil_options: list[SoilParams], weather_options, crop: CropParams = None):
        """
        soil_options: list of SoilParams; one is picked at random each episode
        weather_options: list of seasons, each a list of (rain_mm, et0_mm) per day
        crop: crop curve; maize defults if not given
        """
        super().__init__()
        self.soil = None #note to myself: we pick the soil in reset (reason for changing)
        self.soil_options = soil_options
        self.weather_options = weather_options
        self.weather = None
        self.crop = crop if crop is not None else CropParams()

        #picking one of the irrigation levels
        self.action_space = spaces.Discrete(len(self.IRRIGATION_LEVELS))
        # what the agent can see
        # sensor theta, ET0 today, rain today, rain tomorrow, season length]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 15.0, 200.0, 200.0, 1.0], dtype=np.float32)/self.OBS_SCALE,
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

        return np.array([theta, et0, rain_today, rain_tomorrow, season_progress], dtype = np.float32)/self.OBS_SCALE


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        index = self.np_random.integers(0, len(self.weather_options))
        self.weather = self.weather_options[index]
        self.soil = self.soil_options[self.np_random.integers(0,len(self.soil_options))]
        self.w = self.soil.w_fc
        self.day = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info


    def _get_info(self):
        """Extra information for debugging and plots, not seen by the agent."""
        return {
            "day": self.day,
            "water_mm": self.w,
            "theta_wp": self.soil.theta_wp,
            "theta_fc": self.soil.theta_fc,
            "theta_sat": self.soil.theta_sat,
            "drain_rate": self.soil.drain_rate,
        }

    def _compute_reward(self, irrigation, day_log):
        """ All terms are penalties, so the best day scores 0."""
        water_cost = irrigation * 0.02
        stress_cost = 2.0 * (1 - day_log["ks"])
        drainage_cost = day_log["drainage"] * 0.02
        return -(water_cost + stress_cost + drainage_cost)

    def step(self, action):
        irrigation = self.IRRIGATION_LEVELS[action]
        rain_today, et0 = self.weather[self.day]
        self.w, day_log = step_day(self.w, rain_today, irrigation, et0, self.day, self.soil, self.crop)

        reward = self._compute_reward(irrigation, day_log)

        self.day += 1
        
        terminated = True if self.day >= self.crop.season_length else False
        truncated = False

        if terminated:
            observation = np.zeros(5, dtype=np.float32)
        else:
            observation = self._get_obs()

        info = self._get_info()
        info.update(day_log)

        return observation, reward, terminated, truncated, info

