def run_season(env, choose_action, seed=None):
    """Play one full season and return its totals."""

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    total_reward = 0.0
    total_irrigation = 0.0
    stressed = 0
    days = 0

    while not (terminated or truncated):
        action = choose_action(obs, env)
        obs, reward, terminated, truncated, info = env.step(int(action)) 
        total_reward += reward
        total_irrigation += info["irrigation"]
        stressed += info["ks"] < 1.0
        days += 1

    return {
        "reward": total_reward,
        "irrigation": total_irrigation,
        "stressed_days": stressed,
        "days": days,
    }

