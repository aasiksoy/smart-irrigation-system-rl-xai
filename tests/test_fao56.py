import random
from simulator.fao56 import CropParams, SoilParams, step_day, theta_from_w


def synthetic_weather(n_days: int, seed: int = 0):
    """Plausible summer weather, only for testing the physics."""
    rng = random.Random(seed)
    weather = []
    for _ in range(n_days):
        rain = rng.expovariate(1 / 8) if rng.random() < 0.25 else 0.0
        et0 = max(0.5, rng.gauss(4.5, 1.2)) 
        weather.append((rain, et0))
    return weather


def run_season(soil, crop, policy, weather):
    """Run one season. policy(day, w, soil) returns irrigation in mm."""
    w = soil.w_fc            # start the season at field capacity
    w_start = w
    totals = {k: 0.0 for k in ["rain", "irrigation", "runoff", "drainage", "et_actual"]}
    stressed_days = 0
    for day, (rain, et0) in enumerate(weather):
        irr = policy(day, w, soil)
        w, log = step_day(w, rain, irr, et0, day, soil, crop)
        for k in totals:
            totals[k] += log[k]
        stressed_days += log["ks"] < 1.0
    return w_start, w, totals, stressed_days


def check_water_balance(w_start, w_end, t):
    """Water in minus water out must equal the change in storage."""
    inflow = t["rain"] + t["irrigation"]
    outflow = t["runoff"] + t["drainage"] + t["et_actual"]
    error = (w_start + inflow - outflow) - w_end
    assert abs(error) < 1e-6, f"water balance does not close: {error:.6f} mm"


if __name__ == "__main__":
    crop = CropParams()
    weather = synthetic_weather(crop.season_length)

    sandy = SoilParams(theta_wp=0.06, theta_fc=0.16, theta_sat=0.40, drain_rate=0.8)
    clay = SoilParams(theta_wp=0.22, theta_fc=0.38, theta_sat=0.50, drain_rate=0.2)

    no_irrigation = lambda day, w, soil: 0.0
    threshold = lambda day, w, soil: 20.0 if w < soil.w_stress else 0.0

    for name, soil in [("sandy", sandy), ("clay", clay)]:
        print(f"{name}: TAW = {soil.taw:.0f} mm, RAW = {soil.raw:.0f} mm")
        for pname, policy in [("no irrigation", no_irrigation), ("threshold", threshold)]:
            w0, w1, t, stressed = run_season(soil, crop, policy, weather)
            check_water_balance(w0, w1, t)
            print(f"  {pname:13s} | irrigation {t['irrigation']:5.0f} mm | "
                  f"drainage {t['drainage']:5.0f} mm | ET {t['et_actual']:5.0f} mm | "
                  f"stressed days {stressed:3d}/{crop.season_length} | "
                  f"final theta {theta_from_w(w1, soil):.3f}")

    print("\nAll checks passed.")