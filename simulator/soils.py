from simulator.fao56 import SoilParams
import numpy as np

def random_soil(rng) -> SoilParams:
    """One realistic random soil, drawn with the given random generator."""

    theta_wp = rng.uniform(0.05, 0.2) 
    theta_fc = rng.uniform(theta_wp + 0.08, theta_wp + 0.2) 
    theta_sat = rng.uniform(theta_fc + 0.10, 0.5) 
    drain_rate = rng.uniform(0.2, 0.9) 
    soil = SoilParams(theta_wp=theta_wp, theta_fc=theta_fc, theta_sat=theta_sat, drain_rate=drain_rate)
    return soil 

def make_soils(n, seed) -> list[SoilParams]:
    """n random soils; the same seed always gives the same list."""
    rng = np.random.default_rng(seed=seed)
    random_soils = []
    for _ in range (n):
        soil = random_soil(rng)
        random_soils.append(soil)

    return random_soils