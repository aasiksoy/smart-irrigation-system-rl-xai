from simulator.irrigation_env import IrrigationEnv

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

def evaluate(choose_action, soils, test_seasons): 
    """Play every (soil, season) pair once with this strategy; return the averages."""
    results = []
    for soil in soils: 
        for season in test_seasons:
            eval_env = IrrigationEnv([soil], [season])
            result = run_season(eval_env, choose_action, seed=0)
            result["soil"] = soil 
            results.append(result)
    
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_irrigation =sum(r["irrigation"] for r in results) / len(results)
    avg_stressed_days = sum(r["stressed_days"] for r in results) / len(results)
    
    return {
        "reward": avg_reward,
        "irrigation": avg_irrigation,
        "stressed_days": avg_stressed_days,
        "per_episode": results, 
    }


