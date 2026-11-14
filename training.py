# initialize open ai gym and other tools
from turtle import done
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from environment import FlightEnv
from gymnasium.wrappers import TimeLimit
  
# set up PPO model and environment, begin learning
def make_env():
  return TimeLimit(FlightEnv(), max_episode_steps=20000) # limit the number of steps per episode to 20000

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0) # normalize observations, states, and rewards to improve training stability

logger = configure("./logs/ppo", ["stdout", "csv"]) # set up logger to log to stdout and csv file
model = PPO("MlpPolicy", env, gamma=0.99, seed=1, verbose=1) # initialize PPO model1
model.set_logger(logger) # set the logger for the model
model.learn(total_timesteps=50000) # train the model for 250000 timesteps

model.save("ppo_flight_model") # save the trained model
env.save("vecnormalize.pkl") # save the VecNormalize statistics for future use

# begin evaluation of the trained model

# evaluate policy command
eval_env = DummyVecEnv([lambda: Monitor(TimeLimit(FlightEnv(), max_episode_steps=1000))]) # create a new evaluation environment
eval_env = VecNormalize.load("vecnormalize.pkl", eval_env) # load the saved VecNormalize statistics
eval_env.training = False # set the evaluation environment to evaluation mode
eval_env.norm_reward = False # disable reward normalization for evaluation

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True) # evaluate the trained model
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}") # print the mean and standard deviation of the reward

# graphical evaluation of the trained model
eval_env = DummyVecEnv([make_env]) # create a new evaluation environment
eval_env = VecNormalize.load("vecnormalize.pkl", eval_env) # load the saved VecNormalize statistics
eval_env.training = False # set the evaluation environment to evaluation mode
eval_env.norm_reward = False # disable reward normalization for evaluation

model = PPO.load("ppo_flight_model", env=eval_env) # load the trained model
obs = eval_env.reset() # reset the evaluation environment and get the initial observation

fuel_history = [] # assign fuel to an empty array
speed_history = [] # assign cruise speed to an empty array
target_speed_history = [] # assign target cruise speed to an empty array
alt_history = [] # assign altitude to an empty array
target_alt_history = [] # assign target altitude to an empty array
reward_history = [] # initialize reward history

for _ in range(20000):
  action, _ = model.predict(obs, deterministic=True)  # trained policy w/ no exploration
  obs, reward, done, info = eval_env.step(action)
  
  if done: # check if the episode is done
    break

  s = eval_env.envs[0].unwrapped.state # get the non-normalized state from the environment
  reward_history.append(float(reward[0])) # get list of rewards over episode
  fuel_history.append(float(s[6])) # get list of fuel over episode
  speed_history.append(float(s[2])) # get list of cruise speed over episode
  alt_history.append(float(s[4])) # get list of altitude over episode
  target_speed_history.append(float(s[3])) # get list of target cruise speed over episode
  target_alt_history.append(float(s[5])) # get list of target altitude over episode

eval_env.close() # close the evaluation environment

# find min and max target values
min_target_speed = min(target_speed_history)
max_target_speed = max(target_speed_history)
min_target_alt = min(target_alt_history)
max_target_alt = max(target_alt_history)

# figure for plotting per step values
fig, axs = plt.subplots(1, 3, figsize=(6, 12))
axs[0].plot(target_speed_history, color='orange')
axs[0].fill_between(range(len(target_speed_history)), min_target_speed, max_target_speed, color='orange', alpha=0.2, label="Target Range")
axs[0].plot(speed_history, label="Cruise Speed")
axs[0].legend()
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Speed Over Episode")

axs[1].plot(target_alt_history, color='orange')
axs[1].fill_between(range(len(target_alt_history)), min_target_alt, max_target_alt, color='orange', alpha=0.2, label="Target Range")
axs[1].plot(alt_history, label="Altitude")
axs[1].legend()
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Altitude Over Episode")

axs[2].plot(reward_history, marker='x')
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Reward Over Episode")

plt.suptitle('PPO Performance', fontsize=16)
plt.tight_layout()
plt.show()

# figure for plotting entropy and value loss of PPO model
log_data = pd.read_csv('./logs/ppo/progress.csv') # read the log data

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(log_data["time/total_timesteps"], -log_data["train/entropy_loss"]) # plot negative entropy loss to show entropy
axs[0].set_xlabel("Total Timesteps")
axs[0].set_ylabel("Entropy")
axs[0].set_title("PPO Entropy")

axs[1].plot(log_data["time/total_timesteps"], log_data["train/value_loss"]) # plot value loss
axs[1].set_xlabel("Total Timesteps")
axs[1].set_ylabel("Value Loss")
axs[1].set_title("PPO Value Loss")

plt.tight_layout()
plt.show()
