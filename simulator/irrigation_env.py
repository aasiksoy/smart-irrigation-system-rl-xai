import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.fao56 import SoilParams, CropParams, step_day, theta_from_w

class IrrigationEnv(gym.Env):

    IRRIGATION_LEVELS = [0.0, 5.0, 10.0, 15.0, 20.0]

    def __init__(self, soil:SoilParams, weather, crop: CropParams = None):
        """
        soil: the field (SoilParams)
        weather: list of (rain_mm, et0_mm) per day, one entry per season day
        crop: crop curve; maze defults if not given
        """
        super().__init__()
        self.soil = soil
        self.weather = weather
        self.crop = crop if crop is not None else CropParams()

        #picking one of the irrigation levels
        self.action_space = spaces.Discrete(len(self.IRRIGATION_LEVELS))
        # what the agent can see
        # sensor theta, ET0 today, rain today, rain tomorrow, say or season length]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 15.0, 200.0, 200.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )



