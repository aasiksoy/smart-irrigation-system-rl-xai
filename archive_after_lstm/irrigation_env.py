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
                 start_range=None, informed_params=None,
                 stress_memory: bool = False, probe_bonus: float = 0.0,
                 sensor_miss: float = 0.0, sensor_false: float = 0.0,
                 confirm_days: int = 1, stress_weight: float = 2.0):
        super().__init__()
        self.soil_options = soil_options
        self.soil = None
        self.weather_options = weather_options
        self.weather = None
        self.crop = crop if crop is not None else CropParams()
        if informed_params is None:
            informed_params = ["theta_wp", "theta_fc", "drain_rate"] if informed else []
        self.informed_params = list(informed_params)
        self.informed = len(self.informed_params) > 0
        self.remember_start = remember_start
        self.theta_start = 0.0
        self.start_range = start_range
        self.stress_memory = stress_memory
        self.probe_bonus = probe_bonus
        self.sensor_miss = sensor_miss
        self.sensor_false = sensor_false
        self.confirm_days = confirm_days
        self.stress_weight = stress_weight
        self.detect_streak = 0
        self.streak_theta = 0.0
        self.stress_seen = 0.0
        self.theta_stress_seen = 0.0

        self.action_space = spaces.Discrete(len(self.IRRIGATION_LEVELS))

        #what the agents see
        low = np.zeros(5, dtype=np.float32)
        high = np.array([1.0, 15.0, 200.0, 200.0, 1.0], dtype=np.float32) / self.OBS_SCALE

        if self.remember_start:
            low = np.concatenate([low, np.zeros(1, dtype=np.float32)])
            high = np.concatenate([high, np.ones(1, dtype=np.float32)])

        if self.stress_memory:
            low = np.concatenate([low, np.zeros(2, dtype=np.float32)])
            high = np.concatenate([high, np.ones(2, dtype=np.float32)])

        if self.informed:
            n = len(self.informed_params)
            low = np.concatenate([low, np.zeros(n, dtype=np.float32)])
            high = np.concatenate([high, np.ones(n, dtype=np.float32)])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def _get_obs(self):

        theta = theta_from_w(self.w, self.soil)
        rain_today, et0 = self.weather[self.day]

        if len(self.weather) == self.day + 1 :
            rain_tomorrow = 0.0
        else:
            rain_tomorrow, _ = self.weather[self.day + 1]

        season_length = self.crop.season_length
        season_progress = self.day / season_length

        obs = np.array([theta, et0, rain_today, rain_tomorrow, season_progress],
                       dtype=np.float32) / self.OBS_SCALE

        if self.remember_start:
            obs = np.concatenate([obs, np.array([self.theta_start / 0.5], dtype=np.float32)])

        if self.stress_memory:
            memory = np.array([self.stress_seen, self.theta_stress_seen / 0.5], dtype=np.float32)
            obs = np.concatenate([obs, memory])

        if self.informed:
            values = {"theta_wp": self.soil.theta_wp / 0.5,
                      "theta_fc": self.soil.theta_fc / 0.5,
                      "drain_rate": self.soil.drain_rate}
            soil_info = np.array([values[p] for p in self.informed_params], dtype=np.float32)
            obs = np.concatenate([obs, soil_info])

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        index = self.np_random.integers(0, len(self.weather_options))
        self.weather = self.weather_options[index]
        self.soil = self.soil_options[self.np_random.integers(0,len(self.soil_options))]
        if self.start_range is None:
            self.w = self.soil.w_fc
        else:
            fraction = self.np_random.uniform(self.start_range[0], self.start_range[1])
            self.w = self.soil.w_wp + fraction * (self.soil.w_fc - self.soil.w_wp)
        self.theta_start = theta_from_w(self.w, self.soil)
        self.stress_seen = 0.0
        self.theta_stress_seen = 0.0
        self.detect_streak = 0
        self.streak_theta = 0.0
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
        stress_cost = self.stress_weight * (1 - day_log["ks"])
        drainage_cost = day_log["drainage"] * 0.02
        return -(water_cost + stress_cost + drainage_cost)

    def step(self, action):
        irrigation = self.IRRIGATION_LEVELS[action]
        rain_today, et0 = self.weather[self.day]
        self.w, day_log = step_day(self.w, rain_today, irrigation, et0, self.day, self.soil, self.crop)

        reward = self._compute_reward(irrigation, day_log)

        detected = day_log["ks"] < 1.0
        if self.sensor_miss > 0.0 or self.sensor_false > 0.0:
            if detected:
                detected = self.np_random.random() >= self.sensor_miss
            else:
                detected = self.np_random.random() < self.sensor_false

        if detected:
            theta = theta_from_w(self.w, self.soil)
            if self.detect_streak == 0:
                self.streak_theta = theta
            else:
                self.streak_theta = max(self.streak_theta, theta)
            self.detect_streak += 1
        else:
            self.detect_streak = 0

        if self.detect_streak >= self.confirm_days:
            if self.stress_seen == 0.0:
                reward += self.probe_bonus
            self.stress_seen = 1.0
            self.theta_stress_seen = max(self.theta_stress_seen, self.streak_theta)

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