import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.fao56 import SoilParams, CropParams, step_day, theta_from_w
from typing import Optional

class IrrigationEnv(gym.Env):

    IRRIGATION_LEVELS = [0.0, 5.0, 10.0, 15.0, 20.0]
    OBS_SCALE = np.array([0.5, 10.0, 50.0, 50.0, 1.0], dtype=np.float32)

    def __init__(self, soil_options: list[SoilParams], weather_options,
                 crop: CropParams = None, informed: bool = False, remember_start: bool = False,
                 observe_et0: bool = True, forecast: str = "perfect"):
        """
        soil_options: list of SoilParams; one is picked at random each episode
        weather_options: list of seasons, each a list of (rain_mm, et0_mm) per day
        crop: crop curve; maize defaults if not given
        informed: if True, the agent also sees the true soil parameters
        remember_start: if True, the agent also sees the soil moisture of day 0
        observe_et0: if False, the ET0 input is always 0 (ablation)
        forecast: rain forecast for tomorrow: "perfect", "noisy" or "none"
        """
        super().__init__()
        self.soil_options = soil_options
        self.soil = None
        self.weather_options = weather_options
        self.weather = None
        self.crop = crop if crop is not None else CropParams()
        self.informed = informed
        self.remember_start = remember_start
        self.theta_start = 0.0
        if forecast not in ("perfect", "noisy", "none"):
            raise ValueError(f"unknown forecast type: {forecast}")
        self.observe_et0 = observe_et0
        self.forecast = forecast
        self.rain_forecast = None

        self.action_space = spaces.Discrete(len(self.IRRIGATION_LEVELS))

        #what the agents see
        low = np.zeros(5, dtype=np.float32)
        high = np.array([1.0, 15.0, 200.0, 200.0, 1.0], dtype=np.float32) / self.OBS_SCALE

        if self.remember_start:
            low = np.concatenate([low, np.zeros(1, dtype=np.float32)])
            high = np.concatenate([high, np.ones(1, dtype=np.float32)])

        if self.informed:
            low = np.concatenate([low, np.zeros(3, dtype=np.float32)])
            high = np.concatenate([high, np.ones(3, dtype=np.float32)])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def _get_obs(self):

        theta = theta_from_w(self.w, self.soil)
        rain_today, et0 = self.weather[self.day]

        if len(self.weather) == self.day + 1 :
            rain_tomorrow = 0.0
        else:
            rain_tomorrow, _ = self.weather[self.day + 1]

        if self.forecast == "none":
            rain_tomorrow = 0.0
        elif self.forecast == "noisy":
            rain_tomorrow = self.rain_forecast[self.day]

        if not self.observe_et0:
            et0 = 0.0

        season_length = self.crop.season_length
        season_progress = self.day / season_length

        obs = np.array([theta, et0, rain_today, rain_tomorrow, season_progress],
                       dtype=np.float32) / self.OBS_SCALE

        if self.remember_start:
            obs = np.concatenate([obs, np.array([self.theta_start / 0.5], dtype=np.float32)])

        if self.informed:
            soil_info = np.array([self.soil.theta_wp / 0.5,
                                  self.soil.theta_fc / 0.5,
                                  self.soil.drain_rate], dtype=np.float32)
            obs = np.concatenate([obs, soil_info])

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        index = self.np_random.integers(0, len(self.weather_options))
        self.weather = self.weather_options[index]
        self.soil = self.soil_options[self.np_random.integers(0,len(self.soil_options))]
        self.w = self.soil.w_fc
        self.theta_start = theta_from_w(self.w, self.soil)
        if self.forecast == "noisy":
            self.rain_forecast = self._make_noisy_forecast()
        self.day = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info


    def _make_noisy_forecast(self, p_miss=0.2, p_false=0.1, false_mean=3.0, error=0.4):
        """An imperfect forecast of tomorrow's rain for every day of the season.

        A rainy day is missed with probability p_miss, otherwise its amount is off by a
        random factor (log-normal, spread `error`). A dry day gets a false alarm with
        probability p_false. Made once per season in reset, so it stays the same within
        the season.
        """
        forecast = np.zeros(len(self.weather), dtype=np.float32)
        for day in range(len(self.weather) - 1):
            true_rain = self.weather[day + 1][0]
            if true_rain > 0.5:
                if self.np_random.random() >= p_miss:
                    forecast[day] = true_rain * np.exp(self.np_random.normal(0.0, error))
            elif self.np_random.random() < p_false:
                forecast[day] = self.np_random.exponential(false_mean)
        return np.minimum(forecast, 200.0)

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
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_obs()

        info = self._get_info()
        info.update(day_log)

        return observation, reward, terminated, truncated, info