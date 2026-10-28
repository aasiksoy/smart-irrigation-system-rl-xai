What skills should our agent learn?
Resource allocation under uncertainty

What info does it need?
Sensor θ (soil moisture) , ET₀ today, Rain today, Rain forecast for tomorrow, Season progress (0 to 1, for kc)

What actions can the agent take?
One discrete ( maybe later updated to continuous ) irrigation level choice per day.

When should episodes end?
One episode = one growing season, ending at harvest.


seed 0 done
  varied, seed 0       | irrigation    249 mm | stressed days   1.3 | reward    -5.7
  one season, seed 0   | irrigation    248 mm | stressed days   0.5 | reward    -5.6
seed 1 done
  varied, seed 1       | irrigation    246 mm | stressed days   1.5 | reward    -5.7
  one season, seed 1   | irrigation    247 mm | stressed days   0.8 | reward    -5.6
seed 2 done
  varied, seed 2       | irrigation    249 mm | stressed days   1.2 | reward    -5.7
  one season, seed 2   | irrigation    248 mm | stressed days   0.4 | reward    -5.6
\Rule-based (single run)
Threshold oracle       | irrigation    270 mm | stressed days   0.0 | reward    -6.4
Fixed 5 mm/day         | irrigation    625 mm | stressed days   0.0 | reward   -20.3

PPO, mean ± std over 3 seeds
PPO, varied weather    reward            -5.7 ± 0.0
PPO, varied weather    irrigation       248.2 ± 1.3
PPO, varied weather    stressed_days      1.3 ± 0.1
PPO, one season        reward            -5.6 ± 0.0
PPO, one season        irrigation       247.9 ± 0.5
PPO, one season        stressed_days      0.6 ± 0.2