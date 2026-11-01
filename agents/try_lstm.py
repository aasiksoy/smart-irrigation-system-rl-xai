from simulator.soils import make_soils
from sb3_contrib import RecurrentPPO
from tests.test_fao56 import synthetic_weather
from simulator.irrigation_env import IrrigationEnv

train_soils= make_soils(50, seed=0)
train_seasons = [synthetic_weather(125, i) for i in range(100)]
env = IrrigationEnv(train_soils, train_seasons)

model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, seed=0)
model.learn(5000)


model.save("checkpoints/soil_experiment/ppo_lstm_many_soils_5k_seed0")
model = RecurrentPPO.load("checkpoints/soil_experiment/ppo_lstm_many_soils_5k_seed0")

print("done: LSTM trained, saved and loaded")