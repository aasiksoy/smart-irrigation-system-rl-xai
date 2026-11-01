"""Tests for LstmStrategy.  Run: pytest tests/test_lstm_strategy.py -v"""

from sb3_contrib import RecurrentPPO

from simulator.soils import make_soils
from simulator.irrigation_env import IrrigationEnv
from tests.test_fao56 import synthetic_weather
from agents.evaluate import run_season, LstmStrategy


def make_env():
    """One unseen soil, one season"""
    soil = make_soils(1, seed=1000)[0]
    return IrrigationEnv([soil], [synthetic_weather(125, 100)])


def test_lstm_strategy_runs_a_full_season():
    env = make_env()
    model = RecurrentPPO("MlpLstmPolicy", env, seed=0) 
    
    strategy = LstmStrategy(model)
    result = run_season(env, strategy, seed=0)

    assert result["days"] == 125, f"season lasted {result['days']} days, expected 125"


def test_reset_gives_identical_results():
    # same model, same season, memory wiped before each run , results must match
    env = make_env()
    model = RecurrentPPO("MlpLstmPolicy", env, seed=0)
    strategy = LstmStrategy(model)

    first = run_season(env, strategy, seed=0)

    second = run_season(env, strategy, seed=0)

    assert first == second, f"results differ after reset:\n{first}\n{second}"