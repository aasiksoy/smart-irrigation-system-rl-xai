import gymnasium as gym

from stable_baselines3 import PPO

env = IrrigationEnv(soil, weather)
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10_000)

eval_env = IrrigationEnv(soil, weather)
obs, info = eval_env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    actions, _states = model.predict(obs, deterministic=True) 
    #deterministic=True tells PPO to choose its deterministic action instead of randomly sampling from its learned action distribution.
    
    observation, reward, terminated, truncated, info = eval_env.step(actions)
    total_reward += reward
    total_irrigation += info["irrigation"]
    eval_env.close()
    env.close()

if __name__ == "__main__":
    main()