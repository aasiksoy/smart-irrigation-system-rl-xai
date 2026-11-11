"""Benchmark rule-based and agronomic baselines against trained PPO on OpenMeteo test set.
Run from repo root:  python -m agents.baselines_openmeteo
"""
import numpy as np
from stable_baselines3 import PPO

from simulator.fao56 import CropParams
from simulator.irrigation_env import IrrigationEnv
from simulator.weather import load_seasons
from simulator.soils import make_soils
from agents.evaluate import run_season

SEED = 0
MODEL_PATH = f"checkpoints/openmeteo_experiment/ppo_mlp_informed_seed{SEED}"

PLANTING = {"leuven": "05-01", "nicosia": "04-01", "seville": "04-15"}
TEST_YEARS = range(2016, 2024)

# Baseline Policies
def zero_irrigation(obs, env):
    """Lower bound: Never irrigate (action 0)."""
    return 0


class FixedIntervalBaseline:
    """Timer baseline: Irrigates fixed mm every `interval_days`."""
    def __init__(self, interval_days=3, action=1):
        self.interval_days = interval_days
        self.action = action
        self.day = 0

    def reset(self):
        self.day = 0

    def __call__(self, obs, env):
        self.day += 1
        return self.action if (self.day % self.interval_days == 0) else 0


def fao56_stress_threshold(obs, env):
    """Agronomic benchmark: Irrigates when soil moisture drops below stress threshold (Ks < 1.0)."""
    unwrapped = getattr(env, "unwrapped", env)
    # Uses env state matching the threshold rule from your experiment files
    if hasattr(unwrapped, "w") and hasattr(unwrapped, "soil"):
        if unwrapped.w < unwrapped.soil.w_stress:
            return 4  # or your standard replenishment dose action
    return 0

# Evaluate
def evaluate_policy(policy, test_dict, test_soils, crop, informed=True):
    """Evaluates a policy across all test sites, seasons, and unseen soils."""
    site_metrics = {}
    for site, seasons in test_dict.items():
        rows = []
        for season in seasons:
            for soil in test_soils:
                eval_env = IrrigationEnv([soil], [season], crop, informed=informed)
                r = run_season(eval_env, policy, seed=0)
                rows.append((r["irrigation"], r["stressed_days"], r["reward"]))
                eval_env.close()

        irr, stress, rew = np.mean(rows, axis=0)
        site_metrics[site] = (irr, stress, rew, len(rows))
    return site_metrics


def print_comparison_table(results_by_policy):
    """Prints a clean comparison table per site."""
    sites = list(next(iter(results_by_policy.values())).keys())
    
    for site in sites:
        print(f"\n{'='*25} {site.upper()} (Test 2016-2023) {'='*25}")
        print(f"{'Policy':<25} | {'Irrig (mm)':<12} | {'Stress Days':<12} | {'Reward':<10}")
        print("-" * 68)
        for policy_name, site_data in results_by_policy.items():
            irr, stress, rew, _ = site_data[site]
            print(f"{policy_name:<25} | {irr:10.1f} mm | {stress:10.1f}   | {rew:10.1f}")

# Run
def main():
    crop = CropParams()
    length = crop.season_length
    test_soils = make_soils(20, seed=200)

    test = {site: load_seasons(site, TEST_YEARS, start, length)
            for site, start in PLANTING.items()}

    policies = {
        "Zero Irrigation": zero_irrigation,
        "Fixed (Every 3 Days)": FixedIntervalBaseline(interval_days=3, action=1),
        "FAO-56 Threshold": fao56_stress_threshold,
    }

    try:
        print(f"Loading trained PPO model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH)
        policies["PPO (MLP Informed)"] = lambda obs, env: model.predict(obs, deterministic=True)[0]
    except FileNotFoundError:
        print(f"Checkpoint not found at {MODEL_PATH}. Running rule-based baselines only.")

    results = {}
    for name, policy in policies.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_policy(policy, test, test_soils, crop, informed=True)

    print_comparison_table(results)


if __name__ == "__main__":
    main()