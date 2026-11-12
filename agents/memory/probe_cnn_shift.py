"""Does the CNN still guess the soil well when the agent that USES it is irrigating?

The CNN was trained on seasons played by the blind agent. Here we measure its R2 on seasons
played by (a) that same blind agent and (b) the PPO agents trained on the CNN estimates.
If (b) is much worse, the agent's own behaviour confuses the CNN (distribution shift).

Run from the repo root:  python -m agents.memory.probe_cnn_shift
"""
import numpy as np
from stable_baselines3 import PPO

from simulator.irrigation_env import IrrigationEnv
from simulator.memory.soil_cnn import EstimateWrapper, estimate_soil, load_soil_cnn
from simulator.memory.window_wrapper import WindowWrapper
from simulator.soils import make_soils
from tests.test_fao56 import synthetic_weather

WINDOW = 30
CNN_PATH = "checkpoints/memory/soil_cnn.pt"
BLIND_PATH = "checkpoints/soil_experiment/ppo_mlp_many_soils_300k_seed0"
ESTIMATE_PATH = "checkpoints/memory/ppo_mlp_cnn_estimate_window30_300k_seed{seed}"
ESTIMATE_SEEDS = [0, 1, 2, 3]
TARGETS = ["theta_wp", "theta_fc", "drain_rate"]
PARTS = [("days 0-29", 0, 30), ("days 30-59", 30, 60), ("days 60-124", 60, 125), ("all days", 0, 125)]


def play(choose_action, cnn, soils, seasons):
    """Play every soil x season; save the CNN's guess and the true soil every day."""
    guesses, targets, days = [], [], []
    for soil in soils:
        for season in seasons:
            window_env = WindowWrapper(IrrigationEnv([soil], [season]), WINDOW)
            env = EstimateWrapper(window_env, *cnn)
            obs, info = env.reset(seed=0)
            done = False
            day = 0
            while not done:
                window = np.stack(window_env.history)
                guesses.append(estimate_soil(*cnn, window[None])[0])
                targets.append([soil.theta_wp, soil.theta_fc, soil.drain_rate])
                days.append(day)
                action = choose_action(obs, window)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                day += 1
    return np.array(guesses), np.array(targets), np.array(days)


def print_r2(name, guesses, targets, days):
    print(f"\n{name}")
    print(f"{'part of season':14s} | " + " | ".join(f"{t:>10s}" for t in TARGETS))
    for part, low, high in PARTS:
        mask = (days >= low) & (days < high)
        error = ((targets[mask] - guesses[mask]) ** 2).sum(axis=0)
        spread = ((targets[mask] - targets[mask].mean(axis=0)) ** 2).sum(axis=0)
        print(f"{part:14s} | " + " | ".join(f"{v:10.2f}" for v in 1 - error / spread), flush=True)


def main():
    cnn = load_soil_cnn(CNN_PATH)
    test_soils = make_soils(20, seed=1000)
    test_seasons = [synthetic_weather(125, i) for i in range(100, 110)]

    blind = PPO.load(BLIND_PATH)
    result = play(lambda obs, window: blind.predict(window[-1, :-1], deterministic=True)[0],
                  cnn, test_soils, test_seasons)
    print_r2("CNN R2, seasons played by the blind agent (the CNN's training behaviour)", *result)

    for seed in ESTIMATE_SEEDS:
        model = PPO.load(ESTIMATE_PATH.format(seed=seed))
        result = play(lambda obs, window: model.predict(obs, deterministic=True)[0],
                      cnn, test_soils, test_seasons)
        print_r2(f"CNN R2, seasons played by the PPO agent trained on estimates, seed {seed}", *result)


if __name__ == "__main__":
    main()