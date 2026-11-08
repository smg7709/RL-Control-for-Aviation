# initialize open ai gym and other tools
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import FlightEnv

# set up PPO model and environment
env = FlightEnv() # initialize environment
model = PPO("MlpPolicy", env, verbose=1) # initialize PPO model

model.learn(total_timesteps=1000) # train the model for 1000 timesteps
model.save("ppo_flight_model") # save the trained model
del model

model = PPO.load("ppo_flight_model", env=env) # load the trained model

# evaluation of trained model
obs, info = env.reset() # reset the environment and get the initial observation

fuel_history = [] # assign fuel to an empty array
speed_history = [] # assign cruise speed to an empty array
target_speed_history = [] # assign target cruise speed to an empty array
alt_history = [] # assign altitude to an empty array
target_alt_history = [] # assign target altitude to an empty array
total_reward = 0 # initialize total reward

for _ in range(1000):
  action, _ = model.predict(obs, deterministic=True)  # trained policy w/ no exploration
  obs, reward, terminated, truncated, info = env.step(action)
  
  weight, wind_speed, cruise_speed, target_cruise_speed, altitude, target_altitude, remaining_fuel, remaining_distance, thrust =  obs # assign variables to observations
  fuel_history.append(float(obs[6])) # get list of fuel over episode
  speed_history.append(float(obs[2])) # get list of cruise speed over episode
  alt_history.append(float(obs[4])) # get list of altitude over episode
  target_speed_history.append(float(obs[3])) # get list of target cruise speed over episode
  target_alt_history.append(float(obs[5])) # get list of target altitude over episode

  if terminated or truncated:
    break
env.close() # close the environment

fig, axs = plt.subplots(1, 3, figsize=(6, 12))
axs[0].plot(target_speed_history, label="target")
axs[0].plot(speed_history, label="speed")
axs[0].legend()
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Speed Over Episode")

axs[1].plot(fuel_history, marker='x')
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Fuel Over Episode")

axs[2].plot(target_alt_history, label="target")
axs[2].plot(alt_history, label="altitude")
axs[2].legend()
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Altitude Over Episode")

plt.suptitle('Baseline PPO Performance', fontsize=16)
plt.tight_layout()
plt.show()
print("Total reward:", total_reward)
env.close()