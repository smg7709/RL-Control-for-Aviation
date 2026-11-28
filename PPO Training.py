# initialize open ai gym and other tools
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from Environment import FlightEnv
from gymnasium.wrappers import TimeLimit
  
# create a function to make the environment with a time limit
def make_env():
  return TimeLimit(FlightEnv(), max_episode_steps=1000) # limit the number of steps per episode to 1000 (~2.75 hrs of flight)

# initialize training seeds
#train_seeds = [10, 11, 12, 13, 14]
train_seeds = 10

# training loop
#for seed in train_seeds:
  #env = DummyVecEnv([make_env])
  #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0) # normalize observations, states, and rewards to improve training stability

  #logger = configure(f"./logs/ppo_{seed}", ["stdout", "csv"]) # set up logger to log to stdout and csv file
  #model = PPO("MlpPolicy", env, gamma=0.999, learning_rate=3e-4, target_kl=0.02, seed=seed, verbose=0) # initialize PPO model
  #model.set_logger(logger) # set the logger for the model
  #print(f"Training PPO model with seed {seed}...") # print the current seed being trained
  #model.learn(total_timesteps=400000) # train the model for 400k timesteps

  #model.save(f"ppo_flight_model_{seed}") # save the trained model
  #env.save(f"ppo_vecnormalize.pkl_{seed}") # save the VecNormalize statistics for future use

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0) # normalize observations, states, and rewards to improve training stability

logger = configure(f"./logs/ppo_{train_seeds}", ["stdout", "csv"]) # set up logger to log to stdout and csv file
model = PPO("MlpPolicy", env, gamma=0.999, learning_rate=3e-4, target_kl=0.02, seed=train_seeds, verbose=0) # initialize PPO model
model.set_logger(logger) # set the logger for the model
model.learn(total_timesteps=400000) # train the model for 400k timesteps

model.save(f"ppo_flight_model_{train_seeds}") # save the trained model
env.save(f"ppo_vecnormalize.pkl_{train_seeds}") # save the VecNormalize statistics for future use
