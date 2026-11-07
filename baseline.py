# initialize open ai gym and other tools
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import FlightEnv

env = FlightEnv() # initialize environment
model = PPO("MlpPolicy", env, verbose=1) # initialize PPO model

model.learn(total_timesteps=100000) # train the model for 100,000 timesteps
deterministic = True # set deterministic to True for evaluation, prevents exploration

model.save("ppo_flight_model") # save the trained model
del model

model = PPO.load("ppo_flight_model", env=env) # load the trained model

obs, info = env.reset() # reset the environment and get the initial observation