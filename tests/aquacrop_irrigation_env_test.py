"""
Tests for AquacropIrrigationEnv.

Run with:  python -m pytest tests/aquacrop_irrigation_env_test.py -v

Assumes your env class lives in aquacrop_env.py. Change the import below if not.
"""
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from aquacrop import Crop, Soil
from simulator.aquacrop_irrigation_env import AquacropIrrigationEnv


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def weather_df():
    """Synthetic but realistic-looking weather: 1 May to 30 Sep 2025."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-05-01", "2025-09-30", freq="D")
    n = len(dates)
    min_t = 10 + 6 * np.sin(np.linspace(0, np.pi, n)) + rng.normal(0, 1, n)
    max_t = min_t + 10 + rng.normal(0, 1.5, n)
    rain = np.where(rng.random(n) < 0.25, rng.gamma(2.0, 3.0, n), 0.0)
    et0 = np.clip(3.5 + 1.5 * np.sin(np.linspace(0, np.pi, n)) + rng.normal(0, 0.3, n), 0.5, 9)
    return pd.DataFrame({
        "MinTemp": min_t,
        "MaxTemp": max_t,
        "Precipitation": rain,
        "ReferenceET": et0,
        "Date": dates,
    })


@pytest.fixture
def env(weather_df):
    e = AquacropIrrigationEnv(
        weather_df,
        Soil("SandyLoam"),
        Crop("Maize", planting_date="05/01"),
    )
    yield e
    e.close()


def run_episode(env, action, max_steps=400):
    """Run one full episode with a constant action. Returns (obs_list, rewards, ended)."""
    obs, _ = env.reset(seed=0)
    observations, rewards = [obs], []
    for _ in range(max_steps):
        obs, r, terminated, truncated, info = env.step(action)
        observations.append(obs)
        rewards.append(r)
        if terminated or truncated:
            return observations, rewards, True
    return observations, rewards, False


# ------------------------------------------------------------------- tests
def test_gymnasium_api_compliance(env):
    """
    WHAT: the env follows the Gymnasium API (signatures, return types,
          observation/action space consistency, seeding behaviour).
    HOW:  Gymnasium's own check_env runs reset/step with random actions and
          raises or warns on anything non-standard.
    WHY:  RL libraries (Stable-Baselines3 etc.) assume this contract.
    """
    check_env(env, skip_render_check=True)


def test_action_space_matches_irrigation_levels(env):
    """
    WHAT: one discrete action exists per irrigation depth.
    HOW:  compare action_space.n with len(IRRIGATION_LEVELS).
    """
    assert env.action_space.n == len(env.IRRIGATION_LEVELS)


def test_dates_derived_from_weather(env, weather_df):
    """
    WHAT: sim_start/sim_end and total_days agree with the weather table.
    HOW:  the number of days between the two dates (inclusive) must equal
          the number of weather rows.
    WHY:  if these disagree, self.day indexes the wrong weather row and the
          agent's observations drift from what the model experienced.
    """
    start = pd.to_datetime(env.sim_start, format="%Y/%m/%d")
    end = pd.to_datetime(env.sim_end, format="%Y/%m/%d")
    assert (end - start).days + 1 == env.total_days == len(weather_df)


def test_reset_returns_valid_observation(env):
    """
    WHAT: reset() returns (obs, info) with obs inside the observation space.
    HOW:  check tuple structure, shape (5,), dtype float32, and
          observation_space.contains(obs).
    """
    result = env.reset(seed=0)
    assert isinstance(result, tuple) and len(result) == 2
    obs, info = result
    assert isinstance(info, dict)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_reset_is_repeatable(env):
    """
    WHAT: two resets with the same seed give the same first observation,
          and counters are zeroed (no leftovers from the previous episode).
    HOW:  run a few steps, reset again, compare observations and counters.
    """
    obs1, _ = env.reset(seed=123)
    for _ in range(5):
        env.step(4)
    obs2, _ = env.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2)
    assert env.day == 0
    assert env.total_irrigation == 0.0


def test_step_return_signature(env):
    """
    WHAT: step() returns (obs, reward, terminated, truncated, info) with
          the right types.
    HOW:  take one step and check each element.
    """
    env.reset(seed=0)
    out = env.step(0)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "irrigation_total" in info


def test_episode_ends(env):
    """
    WHAT: an episode always ends (terminated or truncated) and never crashes
          by reading past the end of the weather data.
    HOW:  run with no irrigation and with maximum irrigation; the loop must
          hit terminated/truncated within the length of the weather file.
    """
    for action in (0, env.action_space.n - 1):
        _, _, ended = run_episode(env, action, max_steps=env.total_days + 5)
        assert ended, f"episode with action {action} never ended"


def test_observations_stay_in_bounds_all_episode(env):
    """
    WHAT: every observation of a whole season lies inside observation_space.
    HOW:  run a full episode and check contains() on each observation.
    WHY:  catches theta > 1, NaNs, or weather values above the declared bounds.
    """
    observations, _, _ = run_episode(env, 2)
    for i, o in enumerate(observations):
        assert not np.isnan(o).any(), f"NaN in obs at step {i}"
        assert env.observation_space.contains(o), f"out of bounds at step {i}: {o}"


def test_season_progress_is_monotonic(env):
    """
    WHAT: observation[4] (season progress) never decreases and starts at 0.
    HOW:  collect it over an episode and check the differences are >= 0.
    """
    observations, _, _ = run_episode(env, 0)
    progress = np.array([o[4] for o in observations])
    assert progress[0] == 0.0
    assert np.all(np.diff(progress) >= 0)


def test_irrigation_total_is_tracked(env):
    """
    WHAT: total_irrigation sums the depths of the chosen actions.
    HOW:  apply known actions and compare with the expected sum.
    """
    env.reset(seed=0)
    actions = [1, 2, 0, 4]
    expected = sum(env.IRRIGATION_LEVELS[a] for a in actions)
    for a in actions:
        _, _, _, _, info = env.step(a)
    assert info["irrigation_total"] == pytest.approx(expected)


def test_water_penalty_in_reward(env):
    """
    WHAT: on a non-final step the reward equals -water_cost * depth.
    HOW:  step with action 0 (no water -> reward 0) and with the largest
          action (reward = -water_cost * 20) on the first day.
    """
    env.reset(seed=0)
    _, r0, term, _, _ = env.step(0)
    assert not term
    assert r0 == pytest.approx(0.0)

    env.reset(seed=0)
    _, r_max, term, _, _ = env.step(env.action_space.n - 1)
    assert not term
    assert r_max == pytest.approx(-env.water_cost * env.IRRIGATION_LEVELS[-1])


def test_irrigation_actually_changes_soil_moisture(env):
    """
    WHAT: the chosen irrigation depth really reaches the AquaCrop model.
          This is the key integration test for the version-dependent line
          that sets the depth on the model before each step.
    HOW:  run the same days twice from the same seed, once with no water and
          once with maximum irrigation. Soil moisture (obs[0]) after 20 days
          must be higher in the irrigated run.
    FAILS IF: the depth attribute you set is ignored by your aquacrop
          version. Both runs would then be identical and this test fails,
          which is exactly the silent bug you want to catch.
    """
    def theta_after(action, n=20):
        env.reset(seed=0)
        obs = None
        for _ in range(n):
            obs, *_ = env.step(action)
        return obs[0]

    dry = theta_after(0)
    wet = theta_after(env.action_space.n - 1)
    assert wet > dry, f"irrigation had no effect (dry={dry:.4f}, wet={wet:.4f})"


def test_final_reward_includes_yield(env):
    """
    WHAT: when the season finishes (terminated), the reward includes the
          yield, so the last reward is larger than the water penalty alone.
    HOW:  run an episode with no irrigation (penalty 0). If it terminated,
          the last reward must be > 0 (yield in t/ha). Skipped if the
          episode was truncated instead, since no final yield exists then.
    """
    env.reset(seed=0)
    last_reward, terminated = 0.0, False
    for _ in range(env.total_days + 5):
        _, last_reward, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    if not terminated:
        pytest.skip("episode was truncated, not terminated; no final yield to check")
    assert last_reward > 0