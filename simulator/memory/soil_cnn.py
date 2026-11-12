import numpy as np
import torch
from torch import nn
import gymnasium as gym
from gymnasium import spaces

THETA_SCALE = 0.5


class SoilCNN(nn.Module):
    """Reads a window of days (days x 6 values) and predicts theta_wp, theta_fc, drain_rate."""

    def __init__(self, n_inputs=6, channels=32, n_outputs=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_inputs, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(2 * channels, 64), nn.ReLU(), nn.Linear(64, n_outputs))

    def forward(self, x):
        # x: (batch, days, values); Conv1d wants (batch, values, days)
        h = self.conv(x.transpose(1, 2))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(h)


def load_soil_cnn(path):
    checkpoint = torch.load(path, weights_only=False)
    model = SoilCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["std"]


def estimate_soil(model, mean, std, windows):
    """windows: (batch, days, 6) -> (batch, 3) soil guess in real units."""
    with torch.no_grad():
        out = model(torch.as_tensor(windows, dtype=torch.float32)).numpy()
    return out * std + mean


class EstimateWrapper(gym.Wrapper):
    """The agent sees today's normal observation plus the CNN's guess of the soil.

    Wraps a WindowWrapper: the CNN reads the window, the agent only gets today and the guess,
    in the same format as the informed agent (theta_wp / 0.5, theta_fc / 0.5, drain_rate).
    """

    def __init__(self, env, model, mean, std):
        super().__init__(env)
        self.model = model
        self.mean = mean
        self.std = std
        self.window = env.window
        base = env.unwrapped.observation_space
        low = np.concatenate([base.low, np.zeros(3)]).astype(np.float32)
        high = np.concatenate([base.high, np.ones(3)]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        stacked, info = self.env.reset(**kwargs)
        return self._observe(stacked), info

    def step(self, action):
        stacked, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._observe(stacked)
        return obs, reward, terminated, truncated, info

    def _observe(self, stacked):
        days = stacked.reshape(self.window, -1)
        guess = estimate_soil(self.model, self.mean, self.std, days[None])[0]
        soil_info = np.clip([guess[0] / THETA_SCALE, guess[1] / THETA_SCALE, guess[2]], 0.0, 1.0)
        return np.concatenate([days[-1, :-1], soil_info]).astype(np.float32)