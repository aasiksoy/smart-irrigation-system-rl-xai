"""Train a 1D-CNN that predicts the soil from the last 30 days.

1. Collect windows from seasons played by the blind PPO agent (with 20% random actions,
   so the data also contains less typical irrigation).
2. Train the CNN on the true soil (supervised).
3. Report R2 on the unseen test soils, per part of the season.

Run from the repo root:   python -m agents.memory.train_soil_cnn
Quick test (1-2 min):     python -m agents.memory.train_soil_cnn quick
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO

from simulator.irrigation_env import IrrigationEnv
from simulator.memory.soil_cnn import SoilCNN, estimate_soil
from simulator.memory.window_wrapper import WindowWrapper
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"

WINDOW = 30
N_SOILS = 30 if QUICK else 1500
SEASONS_PER_SOIL = 2
SAVE_EVERY = 5
EPSILON = 0.2
EPOCHS = 2 if QUICK else 20
BATCH_SIZE = 256
LR = 1e-3
BLIND_PATH = "checkpoints/soil_experiment/ppo_mlp_many_soils_300k_seed0"
DATA_PATH = "data/soil_windows_quick.npz" if QUICK else "data/soil_windows.npz"
MODEL_PATH = "checkpoints/memory_test/soil_cnn.pt" if QUICK else "checkpoints/memory/soil_cnn.pt"
TARGETS = ["theta_wp", "theta_fc", "drain_rate"]

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def collect(policy, soils, seasons, seasons_per_soil, save_every, rng):
    """Play seasons and save the 30-day window, the true soil and the day."""
    windows, targets, days = [], [], []
    for soil in soils:
        picks = rng.choice(len(seasons), size=seasons_per_soil, replace=False)
        for pick in picks:
            env = WindowWrapper(IrrigationEnv([soil], [seasons[pick]]), WINDOW)
            stacked, info = env.reset(seed=int(rng.integers(1_000_000)))
            day = 0
            done = False
            while not done:
                window = stacked.reshape(WINDOW, -1)
                if day % save_every == 0:
                    windows.append(window)
                    targets.append([soil.theta_wp, soil.theta_fc, soil.drain_rate])
                    days.append(day)
                if rng.random() < EPSILON:
                    action = int(rng.integers(5))
                else:
                    action = int(policy.predict(window[-1, :-1], deterministic=False)[0])
                stacked, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                day += 1
    return (np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32),
            np.array(days))


def train(windows, targets):
    torch.manual_seed(0)
    mean, std = targets.mean(axis=0), targets.std(axis=0)
    x = torch.as_tensor(windows)
    y = torch.as_tensor((targets - mean) / std)
    model = SoilCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        log(f"epoch {epoch + 1}/{EPOCHS}: loss {total / len(x):.3f}")
    model.eval()
    return model, mean, std


def report(model, mean, std, windows, targets, days):
    guess = estimate_soil(model, mean, std, windows)
    print(f"\nCNN soil estimate, R2 on unseen test soils (1 = perfect, 0 = no better than the mean)")
    print(f"{'part of season':14s} | " + " | ".join(f"{t:>10s}" for t in TARGETS))
    for name, low, high in [("days 0-9", 0, 10), ("days 10-29", 10, 30), ("days 30-59", 30, 60),
                            ("days 60-124", 60, 125), ("all days", 0, 125)]:
        mask = (days >= low) & (days < high)
        error = ((targets[mask] - guess[mask]) ** 2).sum(axis=0)
        spread = ((targets[mask] - targets[mask].mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1 - error / spread
        print(f"{name:14s} | " + " | ".join(f"{v:10.2f}" for v in r2))


def main():
    rng = np.random.default_rng(0)
    train_soils = make_soils(N_SOILS, seed=500)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 100)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]
    policy = PPO.load(BLIND_PATH)

    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH)
        windows, targets = data["windows"], data["targets"]
        log(f"loaded {len(windows):,} training windows from {DATA_PATH}")
    else:
        log(f"collecting training windows: {N_SOILS} soils x {SEASONS_PER_SOIL} seasons")
        windows, targets, _ = collect(policy, train_soils, train_seasons, SEASONS_PER_SOIL,
                                      SAVE_EVERY, rng)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        np.savez(DATA_PATH, windows=windows, targets=targets)
        log(f"saved {len(windows):,} training windows to {DATA_PATH}")

    log("collecting test windows: 20 test soils x 10 test seasons, every day")
    test_windows, test_targets, test_days = collect(policy, test_soils, test_seasons,
                                                    len(test_seasons), 1, rng)

    model, mean, std = train(windows, targets)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "window": WINDOW},
               MODEL_PATH)
    log(f"saved CNN to {MODEL_PATH}")
    report(model, mean, std, test_windows, test_targets, test_days)


if __name__ == "__main__":
    main()