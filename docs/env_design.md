What skills should our agent learn?
Resource allocation under uncertainty

What info does it need?
Sensor θ (soil moisture) , ET₀ today, Rain today, Rain forecast for tomorrow, Season progress (0 to 1, for kc)

What actions can the agent take?
One discrete ( maybe later updated to continuous ) irrigation level choice per day.

When should episodes end?
One episode = one growing season, ending at harvest.


How do we measure success? (reward)
reward = −(0.02 × irrigation_mm + 2.0 × (1 − Ks) + 0.02 × drainage_mm)
Stress is weighted heavily: with the first weights (0.05 / 1 / 0.05), never irrigating
scored best, which is agronomically wrong.

What does the agent NOT see?
Water in mm, field capacity, wilting point, drain rate. Partial observability on purpose:
a real sensor only gives θ; the agent must infer the soil.

Observation scaling
Each input is divided by OBS_SCALE = [0.5, 10, 50, 50, 1] (θ, ET₀, rain, rain tomorrow,
progress), chosen from the training data (max ET₀ 8.7, max rain 68.7; θ ≤ θ_sat ≤ 0.5).
One fixed scale for all soils, so it carries no soil information.

Soils
Random soils: θ_wp 0.05–0.20, θ_fc = θ_wp + 0.08–0.20, θ_sat = θ_fc + 0.10–0.50,
drain_rate 0.2–0.9. Train and test soils come from different seeds.

Baselines
- Threshold oracle: knows each field's true stress threshold (upper reference)
- Fixed-trigger threshold: one θ trigger for every field (sensor, no soil analysis)
- Fixed 5 mm/day: naive schedule