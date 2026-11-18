# initialize open ai gym and other tools
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from environment import FlightEnv

env=FlightEnv()

# reset environment
obs, info = env.reset()

# initialize tracking arrays
fuel_history = [] # assign fuel to an empty array
speed_history = [] # assign cruise speed to an empty array
target_speed_history = [] # assign target cruise speed to an empty array
alt_history = [] # assign altitude to an empty array
target_alt_history = [] # assign target altitude to an empty array
total_reward = 0 # initialize total reward

  # random steps
for _ in range(1000):
  action = env.action_space.sample()  # random action
  obs, reward, terminated, truncated, info = env.step(action)

  s = env.unwrapped.state
  fuel_history.append(float(s[6]))
  speed_history.append(float(s[2]))
  target_speed_history.append(float(s[3]))
  alt_history.append(float(s[4]))
  target_alt_history.append(float(s[5]))

min_target_speed = min(target_speed_history)
max_target_speed = max(target_speed_history)
min_target_alt = min(target_alt_history)
max_target_alt = max(target_alt_history)

fig, axs = plt.subplots(1, 3, figsize=(6, 12))
axs[0].fill_between(range(len(target_speed_history)), min_target_speed, max_target_speed, color='orange', alpha=0.2, label="Target Range")
axs[0].plot(speed_history, label="Cruise Speed")
axs[0].legend()
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Cruise Speed Over Episode")

axs[1].plot(fuel_history, marker='x')
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Fuel Over Episode")

axs[2].fill_between(range(len(target_alt_history)), min_target_alt, max_target_alt, color='orange', alpha=0.2, label="Target Range")
axs[2].plot(alt_history, label="Altitude")
axs[2].legend()
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Altitude Over Episode")

plt.suptitle('Random Action', fontsize=16)
plt.tight_layout()
plt.show()
print("Total reward:", total_reward)
env.close()