"""Probe: does the LSTM's memory contain the soil?
Run from the repo root:  python -m agents.probe_lstm
"""
import numpy as np
from sb3_contrib import RecurrentPPO

from simulator.irrigation_env import IrrigationEnv
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

SEEDS = [0, 1, 2]
PROBE_DAYS = [0, 1, 5, 10, 30, 60, 100]
TARGETS = ["theta_fc", "theta_wp", "drain_rate"]


def collect(model, soils, seasons):
    """Play seasons with the LSTM and save its memory, its observation and the true soil."""
    memory = {day: [] for day in PROBE_DAYS}
    observations = {day: [] for day in PROBE_DAYS}
    targets = []
    for soil in soils:
        for season in seasons:
            env = IrrigationEnv([soil], [season])
            obs, info = env.reset(seed=0)
            state = None
            episode_start = True
            for day in range(max(PROBE_DAYS) + 1):
                action, state = model.predict(obs, state=state,
                                              episode_start=np.array([episode_start]),
                                              deterministic=True)
                episode_start = False
                if day in PROBE_DAYS:
                    memory[day].append(np.concatenate([state[0].ravel(), state[1].ravel()]))
                    observations[day].append(obs.copy())
                obs, reward, terminated, truncated, info = env.step(int(action))
            targets.append([soil.theta_fc, soil.theta_wp, soil.drain_rate])
    memory = {day: np.array(values) for day, values in memory.items()}
    observations = {day: np.array(values) for day, values in observations.items()}
    return memory, observations, np.array(targets)


def probe_r2(x_train, y_train, x_test, y_test, alpha=10.0):
    """Fit a linear (ridge) regression on the training soils, return R2 on the test soils."""
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    y_mean = y_train.mean(axis=0)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    n_features = x_train.shape[1]
    weights = np.linalg.solve(x_train.T @ x_train + alpha * np.eye(n_features),
                              x_train.T @ (y_train - y_mean))
    prediction = x_test @ weights + y_mean
    error = ((y_test - prediction) ** 2).sum(axis=0)
    spread = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - error / spread


def main():
    train_soils = make_soils(50, seed=0)
    test_soils = make_soils(20, seed=1000)
    train_seasons = [synthetic_weather(125, i) for i in range(0, 20)]
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    for seed in SEEDS:
        path = f"checkpoints/soil_experiment/ppo_lstm_many_soils_n2048_300k_seed{seed}"
        model = RecurrentPPO.load(path)
        mem_train, obs_train, y_train = collect(model, train_soils, train_seasons)
        mem_test, obs_test, y_test = collect(model, test_soils, test_seasons)

        print(f"\nSeed {seed}: R2 on unseen test soils (1 = perfect, 0 = no better than the mean)")
        print(f"{'day':>4s} | {'input':6s} | " + " | ".join(f"{t:>10s}" for t in TARGETS))
        for day in PROBE_DAYS:
            for name, x_train, x_test in [("memory", mem_train[day], mem_test[day]),
                                          ("obs", obs_train[day], obs_test[day])]:
                scores = probe_r2(x_train, y_train, x_test, y_test)
                print(f"{day:4d} | {name:6s} | " + " | ".join(f"{s:10.2f}" for s in scores),
                      flush=True)


if __name__ == "__main__":
    main()