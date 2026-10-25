import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from aquacrop import AquaCropModel, Crop, InitialWaterContent, Soil
from aquacrop.utils import prepare_weather, get_filepath

# AquaCrop requires a DataFrame with: ['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET', 'Date']

class AquacropIrrigationEnv(gym.Env):
    # Gymnasium environment wrapping for AquaCrop-OSPy
    # Declare rendering capabilites of environment to prevent compatibility warnings
   
    metadata = {"render_modes": ["human"]} # "human": Environment continuosly rendered in display or terminal
    IRRIGATION_LEVELS = [0.0, 5.0, 10.0, 15.0, 20.0]

    def __init__(self, weather_df:pd.DataFrame, soil:Soil, crop:Crop, sim_start: str="2025/05/01"):
        super().__init__()

    self.weather_df = weather_df.reset_index(drop=True) # index unwanted as a collumn in the aquacrop structure
    self.soil = soil
    self.crop = crop
    self.sim_start = sim_start
    self.total_days = len(self.weather_df) #implemented to normalize days and to check out of bound days

    self.action_space = spaces.Discrete(len(self.IRRIGATION_LEVELS))

    # [sensor_theta, ET0_today, rain_today, rain_tomorrow, season_progress]
    self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 15.0, 200.0, 200.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    self.model = None