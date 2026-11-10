# initialize open ai gym and other tools
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from environment import FlightEnv
from gymnasium.wrappers import TimeLimit

# set up PPO model and environment, begin learning
env = TimeLimit(FlightEnv(), max_episode_steps=1000)
model = PPO("MlpPolicy", env, gamma=0.99, seed=0, verbose=1) # initialize PPO model
model.learn(total_timesteps=100000) # train the model for 100000 timesteps

mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=50, deterministic=True) # evaluate the trained model
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}") # print the mean and standard deviation of the reward

print('Training Done')
model.save("ppo_flight_model") # save the trained model
model = PPO.load("ppo_flight_model", env=env) # load the trained model
print('Saved, starting evaluation')

# evaluation of trained model
obs, info = env.reset() # reset the environment and get the initial observation

fuel_history = [] # assign fuel to an empty array
speed_history = [] # assign cruise speed to an empty array
target_speed_history = [] # assign target cruise speed to an empty array
alt_history = [] # assign altitude to an empty array
target_alt_history = [] # assign target altitude to an empty array
reward_history = [] # initialize reward history

for _ in range(1000):
  action, _ = model.predict(obs, deterministic=True)  # trained policy w/ no exploration
  obs, reward, terminated, truncated, info = env.step(action)
  reward_history.append(reward)

  fuel_history.append(float(obs[6])) # get list of fuel over episode
  speed_history.append(float(obs[2])) # get list of cruise speed over episode
  alt_history.append(float(obs[4])) # get list of altitude over episode
  target_speed_history.append(float(obs[3])) # get list of target cruise speed over episode
  target_alt_history.append(float(obs[5])) # get list of target altitude over episode

  if terminated or truncated:
    break
env.close() # close the environment

# find min and max target values
min_target_speed = min(target_speed_history)
max_target_speed = max(target_speed_history)
min_target_alt = min(target_alt_history)
max_target_alt = max(target_alt_history)

# figure for plotting
fig, axs = plt.subplots(1, 3, figsize=(6, 12))
axs[0].plot(target_speed_history, color='orange')
axs[0].fill_between(range(len(target_speed_history)), min_target_speed, max_target_speed, color='orange', alpha=0.2, label="target range")
axs[0].plot(speed_history, label="speed")
axs[0].legend()
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Speed Over Episode")

axs[1].plot(target_alt_history, color='orange')
axs[1].fill_between(range(len(target_alt_history)), min_target_alt, max_target_alt, color='orange', alpha=0.2, label="target range")
axs[1].plot(alt_history, label="altitude")
axs[1].legend()
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Altitude Over Episode")

axs[2].plot(reward_history, marker='x')
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Reward Over Episode")

plt.suptitle('Baseline PPO Performance', fontsize=16)
plt.tight_layout()
plt.show()
print('Evaluation Done')
