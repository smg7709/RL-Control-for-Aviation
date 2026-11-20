# initialize open ai gym and other tools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from Environment import FlightEnv
from gymnasium.wrappers import TimeLimit
  
# create a function to make the environment with a time limit
def make_env():
  return TimeLimit(FlightEnv(), max_episode_steps=1000) # limit the number of steps per episode to 1000 (~2.75 hrs of flight)

# initialize training and evaluation seeds
train_seeds = [10, 11, 12, 13, 14]
eval_seeds = list(range(1000, 1020))
all_returns = {} # location to store returns for each seed

# training loop
for seed in train_seeds:
  env = DummyVecEnv([make_env])
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0) # normalize observations, states, and rewards to improve training stability

  logger = configure(f"./logs/ppo_{seed}", ["stdout", "csv"]) # set up logger to log to stdout and csv file
  model = PPO("MlpPolicy", env, gamma=0.99, seed=seed, verbose=0) # initialize PPO model
  model.set_logger(logger) # set the logger for the model
  print(f"Training PPO model with seed {seed}...") # print the current seed being trained
  model.learn(total_timesteps=500000) # train the model for 500k timesteps

  model.save(f"ppo_flight_model_{seed}") # save the trained model
  env.save(f"ppo_vecnormalize.pkl_{seed}") # save the VecNormalize statistics for future use

  # evaluate policy using fixed seeds
  eval_env = DummyVecEnv([lambda: Monitor(TimeLimit(FlightEnv(), max_episode_steps=50000))]) # create a new evaluation environment
  eval_env = VecNormalize.load(f"ppo_vecnormalize.pkl_{seed}", eval_env) # load the saved VecNormalize statistics
  eval_env.training = False # set the evaluation environment to evaluation mode
  eval_env.norm_reward = False # disable reward normalization
  mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True) # evaluate the trained model
  print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}") # print the mean and standard deviation of the reward

  # initialize empty list for returns
  returns = []

  # evaluate seeds
  for es in eval_seeds:
    eval_env.seed(es)
    obs = eval_env.reset()
    total, done = 0, False
    
    while not done:
      action, _ = model.predict(obs, deterministic = True)
      obs, reward, done, info = eval_env.step(action)
      total += float(reward[0])
      done = bool(done[0])
    returns.append(total)

  print(f"Train Seed {seed}: Mean {np.mean(returns): .1f} +/- {np.std(returns): .1f}")
  all_returns[seed] = returns # store the returns for the current seed





# %%





# define a function to plot cruise speed, altitude, and reward over the course of an episode
def plot_trajectories(seed):

  eval_env = DummyVecEnv([make_env]) # create a new evaluation environment
  eval_env = VecNormalize.load(f"ppo_vecnormalize.pkl_{seed}", eval_env) # load the saved VecNormalize statistics
  eval_env.training = False # set the evaluation environment to evaluation mode
  eval_env.norm_reward = False # disable reward normalization

  model = PPO.load(f"ppo_flight_model_{seed}", env=eval_env) # load the trained model
  obs = eval_env.reset() # reset the evaluation environment and get the initial observation

  # assign empty lists to store trajectories
  fuel_history = []
  speed_history = []
  target_speed_history = []
  alt_history = []
  target_alt_history = []
  reward_history = []

  for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)  # trained policy w/ no exploration
    obs, reward, done, info = eval_env.step(action)
    
    if done[0]: # check if the episode is done
      break

    s = eval_env.envs[0].unwrapped.state # get the non-normalized state from the environment

    # append trajectories to lists
    reward_history.append(float(reward[0]))
    fuel_history.append(float(s[6]))
    speed_history.append(float(s[2]))
    alt_history.append(float(s[4]))
    target_speed_history.append(float(s[3]))
    target_alt_history.append(float(s[5]))

  eval_env.close() # close the evaluation environment

  # initialize subplots and plot trajectories
  fig, axs = plt.subplots(1, 3, figsize=(15, 4))

  axs[0].plot(target_speed_history, color='orange', label="Target")
  axs[0].plot(speed_history, label="Cruise Speed")
  axs[1].plot(target_alt_history, color='orange', label="Target")
  axs[1].plot(alt_history, label="Altitude")
  axs[2].plot(reward_history)

  # set axis labels and figure titles
  for a, y1 in zip(axs, ["Speed [knots]", "Altitude [ft]", "Reward"]):
    a.set_xlabel("Step")
    a.set_ylabel(y1)

  axs[0].set_title("Cruise Speed")
  axs[1].set_title("Altitude")
  axs[2].set_title("Reward")
  axs[0].legend()
  axs[1].legend()

  plt.suptitle(f'PPO Training Performance (Seed {seed})', fontsize=16)
  plt.tight_layout()
  plt.savefig(f'ppo_trajectory_{seed}.png') # save the figure
  plt.close()

for sd in train_seeds:
  plot_trajectories(sd)





# %%




# define function for plotting PPO model metrics (entropy, value loss)
def plot_metrics(seeds):

  # initialize subplots for metrics
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  # plot metrics for each seed
  for sd in seeds:
    log_data = pd.read_csv(f'./logs/ppo_{sd}/progress.csv') # read the log data
    axs[0].plot(log_data["time/total_timesteps"], -log_data["train/entropy_loss"], label=f"Seed {sd}") # plot negative entropy loss to show entropy
    axs[1].plot(log_data["time/total_timesteps"], log_data["train/value_loss"], label=f"Seed {sd}") # plot value loss

  # set axis labels and figure titles
  for a in axs:
    a.set_xlabel("Total Timesteps")
    a.legend()
  
  axs[0].set_title("PPO Entropy")
  axs[1].set_title("PPO Value Loss")

  plt.suptitle(f'PPO Training Metrics (All Seeds)', fontsize=16)
  plt.tight_layout()
  plt.savefig(f'ppo_metrics_all_seeds.png') # save the figure
  plt.show()

plot_metrics(train_seeds)
