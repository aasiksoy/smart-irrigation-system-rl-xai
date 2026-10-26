What skills should our agent learn?
Resource allocation under uncertainty

What info does it need?
Sensor θ (soil moisture) , ET₀ today, Rain today, Rain forecast for tomorrow, Season progress (0 to 1, for kc)

What actions can the agent take?
One discrete ( maybe later updated to continuous ) irrigation level choice per day.

When should episodes end?
One episode = one growing season, ending at harvest.
