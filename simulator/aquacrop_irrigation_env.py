import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from aquacrop import AquaCropModel, Crop, InitialWaterContent, Soil, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

# AquaCrop requires a DataFrame with: ['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET', 'Date']

class AquacropIrrigationEnv(gym.Env):
    # Gymnasium environment wrapping for AquaCrop-OSPy
    # Declare rendering capabilites of environment to prevent compatibility warnings
   
    metadata = {"render_modes": ["human"]} # "human": Environment continuosly rendered in display or terminal
    IRRIGATION_LEVELS = [0.0, 5.0, 10.0, 15.0, 20.0]

    # render_mode=None instead of "human" in order to avoid rendering during training.
    def __init__(self, weather_df:pd.DataFrame, soil:Soil, crop:Crop, sim_start: str="2025/05/01", water_cost=0.02, render_mode=None):
        super().__init__()

        self.weather_df = weather_df.reset_index(drop=True) # index unwanted as a collumn in the aquacrop structure
        self.soil = soil
        self.crop = crop
        self.sim_start = pd.to_datetime(self.weather_df["Date"].iloc[0]).strftime("%Y/%m/%d")
        self.sim_end = pd.to_datetime(self.weather_df["Date"].iloc[-1]).strftime("%Y/%m/%d")
        self.total_days = len(self.weather_df) #implemented to normalize days and to check out of bound days
        self.water_cost = water_cost
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(self.IRRIGATION_LEVELS))

        # [sensor_theta, ET0_today, rain_today, rain_tomorrow, season_progress]
        self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 15.0, 200.0, 200.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.model = None
        self.day = 0
        self.total_irrigation = 0.0

    # options is an optional dictionary that lets the caller pass extra settings to a single reset, without changing the constructor.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.model = AquaCropModel(
            self.sim_start, self.sim_end, self.weather_df, self.soil, self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=5), # initialization set to constant depth irrigation (e.g. 5mm)
            initial_water_content=InitialWaterContent(value=["FC"]), # initialization set to field capacity
        )
        self.model.run_model()  # initialized without starting simulating
        self.day, self.total_irrigation = 0, 0.0 # resets the 2 counters: 1. the day of season the agent is on, 2. total mm of applied water
        return self._get_obs(), {}
    

    def step(self, action):
        depth = self.IRRIGATION_LEVELS[int(action)]
        self.total_irrigation += depth
        # set today's irrigation depth on the model, then advance one day
        self.model._param_struct.IrrMngt.depth = depth
        self.model.run_model(initialize_model=False, num_steps=1)
        self.day += 1

        terminated = bool(self.model._clock_struct.model_is_finished)
        truncated = self.day >= self.total_days - 1
        reward = -self.water_cost * depth            # per-step water penalty
        if terminated:
            reward += float(self.model._outputs.final_stats["Dry yield (tonne/ha)"].iloc[-1]) # there are 2 types of yields: 1. Dry (when all moisture has been collected), 2. Wet (with all moisture in plant)
        return self._get_obs(), reward, terminated, truncated, {"irrigation_total": self.total_irrigation}

    # build the 5 numbers the agent has
    def _get_obs(self):
        d = min(self.day, self.total_days - 1) # today's position in the weather table (min takes the last row)
        row = self.weather_df.iloc[d]
        rain_tmrw = self.weather_df["Precipitation"].iloc[min(d + 1, self.total_days - 1)]
        theta = self._sensor_theta()
        return np.array([theta, row["ReferenceET"], row["Precipitation"],
                            rain_tmrw, d / (self.total_days - 1)], dtype=np.float32)

    def _sensor_theta(self):
        # root-zone water content from model._init_cond.th, normalized to [0, 1]
        th = np.asarray(self.model._init_cond.th)
        return float(np.clip(th.mean() / self.soil.profile["th_s"].max(), 0, 1))

    def render(self):
        if self.render_mode == "human":
            print(f"day={self.day} irrigation_total={self.total_irrigation:.1f} mm")

    def close(self):
        self.model = None