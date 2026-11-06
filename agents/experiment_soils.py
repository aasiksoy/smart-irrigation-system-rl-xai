"""Generalization to unseen soils, over several training seeds.
Run from the repo root:  python -m agents.experiment_soils
"""
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from agents.evaluate import evaluate, LstmStrategy
from simulator.fao56 import SoilParams, theta_from_w
from simulator.soils import make_soils
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather
import time
from stable_baselines3.common.callbacks import BaseCallback

START = time.time()

def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:6.1f} min] {msg}", flush=True)


class ProgressLog(BaseCallback):
    """Prints a line every `every` training steps."""
    def __init__(self, every=10_000):
        super().__init__()
        self.every = every

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            log(f"  training: {self.num_timesteps:,} / {LSTM_TIMESTEPS:,} steps")
        return True   # False would stop training


BASELINE_STEPS = "300k"
SEEDS = [1,2]
LSTM_TIMESTEPS = 300_000

#quick test:
#SEEDS = [0]
#LSTM_TIMESTEPS = 5_000


def print_row(name, r):
    print(f"{name:25s} | irrigation {r['irrigation']:6.0f} mm | "
          f"stressed days {r['stressed_days']:5.1f} | reward {r['reward']:7.1f}")


def summarize(name, scores):
    for key in ["reward", "irrigation", "stressed_days"]:
        values = [s[key] for s in scores]
        print(f"{name:25s} {key:14s} {np.mean(values):7.1f} ± {np.std(values):.2f}")


def main():

    sandy = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8)

    train_soils = make_soils(50, seed=0)
    trigger_theta = sum(s.w_stress / (1000 * s.root_depth_m) for s in train_soils) / len(train_soils)
    log(f"Fixed trigger: irrigate when theta < {trigger_theta:.3f}")

    test_soils = make_soils(20, seed=1000)

    train_seasons = []
    for i in range(0, 100):
        train_seasons.append(synthetic_weather(125, i))

    test_seasons = []
    for i in range(100, 110):
        test_seasons.append(synthetic_weather(125, i))


    threshold = lambda obs, env: 4 if env.w < env.soil.w_stress else 0
    fixed_5mm = lambda obs, env: 1
    fixed_trigger = lambda obs, env: 4 if theta_from_w(env.w, env.soil) < trigger_theta else 0

    log("Evaluating rule-based baselines...")
    eval_threshold = evaluate(threshold, test_soils, test_seasons)
    eval_fixed = evaluate(fixed_5mm, test_soils, test_seasons)
    eval_trigger = evaluate(fixed_trigger, test_soils, test_seasons)

    many_soils_scores = []
    one_soil_scores = []
    informed_scores = []
    lstm_scores = []


    for seed in SEEDS:
        log(f"Seed {seed}: loading PPO agents")
        ##Training
        # steps = f"{TOTAL_TIMESTEPS // 1000}k"
        # many_path = f"checkpoints/soil_experiment/ppo_mlp_many_soils_{steps}_seed{seed}"
        # one_path = f"checkpoints/soil_experiment/ppo_mlp_one_soil_{steps}_seed{seed}"
        # informed_path = f"checkpoints/soil_experiment/ppo_mlp_informed_{steps}_seed{seed}"

        # env = IrrigationEnv(train_soils, train_seasons)
        # model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        # model.learn(total_timesteps=TOTAL_TIMESTEPS)
        # model.save(many_path)
        # ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]

        # env_one = IrrigationEnv([sandy], train_seasons)
        # one_model = PPO("MlpPolicy", env_one, verbose=0, seed=seed)
        # one_model.learn(total_timesteps=TOTAL_TIMESTEPS)
        # one_model.save(one_path)
        # one_ppo = lambda obs, env: one_model.predict(obs, deterministic=True)[0]

        # env_inf = IrrigationEnv(train_soils, train_seasons, informed=True)
        # inf_model = PPO("MlpPolicy", env_inf, verbose=0, seed=seed)
        # inf_model.learn(total_timesteps=TOTAL_TIMESTEPS)
        # inf_model.save(informed_path)
        # inf_ppo = lambda obs, env: inf_model.predict(obs, deterministic=True)[0]
        # load the three PPO agents trained earlier (no retraining)

        many_path = f"checkpoints/soil_experiment/ppo_mlp_many_soils_{BASELINE_STEPS}_seed{seed}"
        one_path = f"checkpoints/soil_experiment/ppo_mlp_one_soil_{BASELINE_STEPS}_seed{seed}"
        informed_path = f"checkpoints/soil_experiment/ppo_mlp_informed_{BASELINE_STEPS}_seed{seed}"

        model = PPO.load(many_path)
        ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]

        one_model = PPO.load(one_path)
        one_ppo = lambda obs, env: one_model.predict(obs, deterministic=True)[0]

        inf_model = PPO.load(informed_path)
        inf_ppo = lambda obs, env: inf_model.predict(obs, deterministic=True)[0]

        # train the LSTM agent
        lstm_steps = f"{LSTM_TIMESTEPS // 1000}k"
        lstm_path = f"checkpoints/soil_experiment/ppo_lstm_many_soils_n2048_{lstm_steps}_seed{seed}"

        env = IrrigationEnv(train_soils, train_seasons)
        lstm_model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, seed=seed, n_steps = 2048)
        log(f"Seed {seed}: training LSTM for {LSTM_TIMESTEPS:,} steps")
        lstm_model.learn(total_timesteps=LSTM_TIMESTEPS, callback=ProgressLog())
        lstm_model.save(lstm_path)
        lstm_strategy = LstmStrategy(lstm_model)

        log(f"Seed {seed}: evaluating")
        eval_ppo = evaluate(ppo, test_soils, test_seasons)
        many_soils_scores.append(eval_ppo)

        eval_one = evaluate(one_ppo, test_soils, test_seasons)
        one_soil_scores.append(eval_one)

        eval_inf = evaluate(inf_ppo, test_soils, test_seasons, informed=True)
        informed_scores.append(eval_inf)

        eval_lstm = evaluate(lstm_strategy, test_soils, test_seasons)
        lstm_scores.append(eval_lstm)

        log(f"Seed {seed} done")
        print_row(f"many soils, seed {seed}", eval_ppo)
        print_row(f"one soil, seed {seed}", eval_one)
        print_row(f"informed, seed {seed}", eval_inf)
        print_row(f"LSTM, seed {seed}", eval_lstm)


    print("\nRule-based (single run)")
    print_row("Threshold oracle", eval_threshold)
    print_row("Fixed 5 mm/day", eval_fixed)
    print_row("Fixed-trigger threshold", eval_trigger)

    print(f"\nPPO, mean ± std over {len(SEEDS)} seeds")
    summarize("PPO, many soils", many_soils_scores)
    summarize("PPO, one soil", one_soil_scores)
    summarize("PPO, informed", informed_scores)
    summarize("PPO LSTM, many soils", lstm_scores)

    many_mean = np.mean([s["reward"] for s in many_soils_scores])
    inf_mean = np.mean([s["reward"] for s in informed_scores])
    lstm_mean = np.mean([s["reward"] for s in lstm_scores])
    gap_closed = (lstm_mean - many_mean) / (inf_mean - many_mean)
    print(f"\nLSTM closes {100 * gap_closed:.0f}% of the gap "
          f"(uninformed {many_mean:.1f} -> informed {inf_mean:.1f}, LSTM {lstm_mean:.1f})")


if __name__ == "__main__":
    main()