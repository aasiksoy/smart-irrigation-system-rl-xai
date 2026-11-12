# Project log

## 2026-10-24 — FAO-56 soil-water model
**Did:** Implemented SoilParams, CropParams (maize Kc curve), stress coefficient Ks,
daily water balance (step_day). Partial drainage (drain_rate) instead of textbook
same-day drainage, so soils differ in how fast moisture drops.
**Results:** Season test passes; water balance closes. Sandy vs clay behave differently
(threshold controller: sandy 280 mm irrigation / 73 mm drainage, clay 220 / 13).

## 2026-10-26 — Gymnasium environment
**Did:** IrrigationEnv: 5 discrete actions (0–20 mm), observation [θ, ET0, rain today,
rain tomorrow, season progress], reward = −(0.02·irrigation + 2·(1−Ks) + 0.02·drainage).
**Decided:** First reward weights (0.05/1/0.05) made "never irrigate" the best strategy →
raised stress weight to 2.0 and lowered water/drainage to 0.02. Now threshold oracle
(−7.1) beats fixed schedules and never-irrigate (−76.2).

## 2026-10-27 — First PPO agent
**Did:** PPO (MLP), 100k steps, one synthetic season, sandy soil. Added seeds.
**Results:** seed 0: −6.2 (255 mm, 1 stressed day) vs oracle −7.1 — on its training season.
Run-to-run variation without seeds: −8.1 vs −9.3 → need multiple seeds.

## 2026-10-28 — Generalization to unseen weather + observation scaling
**Did:** Train on 100 synthetic seasons, test on 50 unseen. Refactored run_season / evaluate.
**Results (unscaled obs, 300k):** varied −9.1, one-season −9.8, oracle −6.4, fixed 5 mm −20.3.
**Decided:** Scaled observations (OBS_SCALE = [0.5, 10, 50, 50, 1], chosen from training
data) — raw θ (~0.1) was swamped by rain (up to ~70).
**Results (scaled, 300k, 3 seeds):** varied −5.7 ± 0.0, one-season −5.6 ± 0.0,
oracle −6.4, fixed −20.3 → both PPO agents beat the oracle on unseen weather.
**Learned:** Earlier "overfitting" of the one-season agent was mostly the scaling issue.
With a fixed soil, weather variety is not a hard generalization test.

## 2026-10-31 — Randomized soils
**Did:** Env picks a random soil per episode (soil_options). make_soils() with pytest
tests. Soil experiment: 50 train soils, 20 unseen test soils, 10 test seasons.
Added fixed-trigger baseline (one θ trigger = mean training stress threshold, 0.195).
**Results (rule-based):** oracle −5.6, fixed 5 mm −19.9, fixed trigger −32.7
(404 mm AND 23.8 stressed days) → not knowing the soil is very costly for a simple rule.
**Next:** Full run (3 seeds, 300k): one-soil vs many-soils PPO on unseen soils.

## 2026-10-31 — Real weather (Open-Meteo)
**Did:** Download script for Leuven, Nicosia, Seville (1990–2023, daily rain, ET0, tmax, tmin).
Fixed an off-by-one-day bug (UTC vs local time). load_seasons() cuts each year into a
125-day season from 1 May. 5 pytest tests pass.
**Data:** July averages — Leuven 22.7 °C / ET0 3.7 / rain 2.4; Nicosia 36.2 / 7.3 / 0.1;
Seville 35.7 / 7.2 / 0.1 (mm/day).
**Next:** Check OBS_SCALE against real training-year extremes, then train on real weather.

**OBS_SCALE check (real weather, training years 1990–2015):** rain 99% ≤ 16 mm (max 59.9),
ET0 99% ≤ 8.8 mm (max 10.6) → scales [0.5, 10, 50, 50, 1] kept unchanged.

## 2026-10-31 — Informed agent: the price of not knowing the soil
Settings: 50 train soils (seed 0), 20 test soils (seed 1000), 100 train / 10 test synthetic
seasons, PPO MLP 300k steps, seeds 0–2.
Results: informed −5.3 ± 0.02 (239 mm, 1.7 stressed days), many soils −8.4 ± 0.37
(267 mm, 7.1), oracle −5.6, one soil −72.5. Many/one-soil runs reproduce earlier numbers exactly.
Takeaway: the entire gap to the oracle is caused by not knowing the soil.
Price of not knowing: 3.1 reward points, +29 mm water (+12%), +5.4 stressed days.
Next: LSTM (RecurrentPPO) — can memory infer the soil and close the gap?

## 2026-11-01 — Start of LSTM 

On a field with an unknown soil, today’s θ alone isn’t enough. θ = 0.2 can mean “wet” (sandy soil) or “dry” (heavy soil). You can only tell by watching how θ changes over several days: how fast it drops after rain.

An LSTM agent has memory, it stores states in between days, and resets these states when a new season starts.

**Did:** RecurrentPPO (sb3-contrib, MlpLstmPolicy), same setup as the informed experiment
(50 train soils, 20 test soils, 300k steps, seeds 0–2). Added progress logging every 10k steps.
**Results (n_steps = 128, default):** −59.4 ± 35.3 (109 mm, 50.1 stressed days) → collapsed:
the agent learned to barely irrigate.
**Decided:** n_steps = 2048 (same as the MLP agents), so each update sees ~16 full seasons
instead of ~1.
**Results (n_steps = 2048):** −8.5 ± 0.86 (251 mm, 11.5 stressed days) → no better than the
blind MLP (−8.4).
**Learned:** Memory alone does not close the gap; the LSTM did not learn to infer the soil.
Recurrent RL is sensitive to hyperparameters (Ni et al. 2022). Speed: ~1.7 min per 10k steps.

## Remember-start: one remembered value
**Did:** remember_start switch: the agent also sees θ on day 0. Every season starts at field
capacity, so this value equals θ_fc of the field.
**Results (3 seeds):** −5.9, −6.2, −5.8 → −6.0 ± 0.15 (254 mm, 1.5 stressed days).
Closes 79% of the gap between blind (−8.4) and informed (−5.3).
**Learned:** A single remembered value recovers most of the price of not knowing.
**Assumption:** The field starts near field capacity (pre-irrigation or wet winter).

## 2026-11-12 - Forecast ablation
**Did:** Env switches observe_et0 (True/False) and forecast ("perfect", "noisy", "none").
Noisy forecast: a rainy day is missed with p = 0.2, otherwise the amount is off by a
log-normal factor (σ = 0.4); a dry day gets a false alarm with p = 0.1 (exponential,
mean 3 mm). Made once per season in reset. Blind MLP agents, same splits, 3 seeds, 300k.
**Results:**
basic (no ET0, no forecast) −8.99 ± 0.37 (293 mm, 5.0 stressed days)
+ ET0                         −8.80 ± 0.16 (292 mm, 5.2)
+ ET0 + perfect forecast      −8.35 ± 0.37 (267 mm, 7.1)
+ ET0 + noisy forecast        −8.42 ± 0.32 (274 mm, 6.4)
perfect-forecast agent tested with noisy forecast −8.54 ± 0.42 (268 mm, 7.4)
**Learned:** ET0 gives no clear gain. The forecast helps in all 3 seeds (≈ +0.5) and saves
~25 mm (9%) by skipping irrigation before rain. A noisy forecast is almost as good as a
perfect one. The forecast is worth ~0.5; knowing the soil is worth ~3.1.
**Caveat:** 3 seeds, small differences → trends; add seeds 3–4 for the poster.
**Next:** Revisit why the LSTM failed, then try frame stacking (MLP with the last N days).

## 2026-11-12 — Stacking (MLP with last N days)
**Did:** simulator/memory/window_wrapper.py: last N days of obs + yesterday's irrigation.
agents/memory/experiment_stacking.py, blind PPO MLP, 300k.
**Results (3 seeds):** window 1 −8.83 (≈ blind), 3 −10.84, 7 −9.02, 14 −6.82.
**Results (5 seeds):** window 14 −7.42 ± 0.97 (32%), window 30 −6.95 ± 0.69 (47%, better than
blind in all seeds). Seeds split: some reach −6.1 (≈ remember-start), others ≈ −7.5.
**Learned:** Long history helps, short history hurts. Learning soil from reward alone is
unreliable across seeds; 3 seeds were too few (14-day looked like 51%).
**Next:** CNN soil estimator trained explicitly on the true soil, used as PPO input.