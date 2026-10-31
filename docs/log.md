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