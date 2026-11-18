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
  return TimeLimit(FlightEnv(), max_episode_steps=50000) # limit the number of steps per episode to 50000

train_seeds = [10, 11, 12, 13, 14] # list of seeds for training environments
eval_seeds = list(range(1000, 1050)) # list of seeds for evaluation environments
all_returns = {} # location to store returns for each seed

# training loop
for seed in train_seeds:
  env = DummyVecEnv([make_env])
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0) # normalize observations, states, and rewards to improve training stability

  logger = configure(f"./logs/ppo_{seed}", ["stdout", "csv"]) # set up logger to log to stdout and csv file
  model = PPO("MlpPolicy", env, gamma=0.99, seed=3, verbose=1) # initialize PPO model
  model.set_logger(logger) # set the logger for the model
  print(f"Training PPO model with seed {seed}...") # print the current seed being trained
  model.learn(total_timesteps=10000) # train the model for 10000 timesteps

  model.save(f"ppo_flight_model_{seed}") # save the trained model
  env.save(f"vecnormalize.pkl_{seed}") # save the VecNormalize statistics for future use

  # evaluate policy command (check)
  eval_env = DummyVecEnv([lambda: Monitor(TimeLimit(FlightEnv(), max_episode_steps=50000))]) # create a new evaluation environment
  eval_env = VecNormalize.load(f"vecnormalize.pkl_{seed}", eval_env) # load the saved VecNormalize statistics
  eval_env.training = False # set the evaluation environment to evaluation mode
  eval_env.norm_reward = False # disable reward normalization for evaluation

  mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True) # evaluate the trained model
  print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}") # print the mean and standard deviation of the reward
  returns = [] # initialize a list to store returns for the current seed
  all_returns[seed] = returns # store the returns for the current seed


# graphical evaluation of the trained model

eval_env = DummyVecEnv([make_env]) # create a new evaluation environment
eval_env = VecNormalize.load(f"vecnormalize.pkl_{seed}", eval_env) # load the saved VecNormalize statistics
eval_env.training = False # set the evaluation environment to evaluation mode
eval_env.norm_reward = False # disable reward normalization for evaluation

model = PPO.load(f"ppo_flight_model_{seed}", env=eval_env) # load the trained model
obs = eval_env.reset() # reset the evaluation environment and get the initial observation

fuel_history = [] # assign fuel to an empty array
speed_history = [] # assign cruise speed to an empty array
target_speed_history = [] # assign target cruise speed to an empty array
alt_history = [] # assign altitude to an empty array
target_alt_history = [] # assign target altitude to an empty array
reward_history = [] # initialize reward history

for _ in range(50000):
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

# figure for plotting per step values
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(target_speed_history, color='orange')
axs[0].plot(speed_history, label="Cruise Speed")
axs[0].legend()
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Speed Over Episode")

axs[1].plot(target_alt_history, color='orange')
axs[1].plot(alt_history, label="Altitude")
axs[1].legend()
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Altitude Over Episode")

axs[2].plot(reward_history, marker='x')
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Reward Over Episode")

plt.suptitle(f'Seed {seed} Training Results', fontsize=16)
plt.tight_layout()
plt.show()

# figure for plotting entropy and value loss of PPO model
log_data = pd.read_csv(f'./logs/ppo_{seed}/progress.csv') # read the log data

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(log_data["time/total_timesteps"], -log_data["train/entropy_loss"]) # plot negative entropy loss to show entropy
axs[0].set_xlabel("Total Timesteps")
axs[0].set_ylabel("Entropy")
axs[0].set_title("PPO Entropy")

axs[1].plot(log_data["time/total_timesteps"], log_data["train/value_loss"]) # plot value loss
axs[1].set_xlabel("Total Timesteps")
axs[1].set_ylabel("Value Loss")
axs[1].set_title("PPO Value Loss")

plt.suptitle(f'Seed {seed} PPO Training Metrics', fontsize=16)
plt.tight_layout()
plt.show()
