import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit
from Environment import FlightEnv

def make_env():
    return TimeLimit(FlightEnv(), max_episode_steps=1000)

# initialize training and evaluation seeds
train_seeds = [10, 11, 12, 13, 14]
eval_seeds = list(range(1000, 1020))

# plotting functions ------------------------------------------------------------------------------

# define a function to plot cruise speed, altitude, and reward over the course of an episode
def plot_trajectories(seed, eval_seed=1000):

  eval_env = DummyVecEnv([make_env]) # create a new evaluation environment
  eval_env = VecNormalize.load(f"ppo_vecnormalize.pkl_{seed}", eval_env) # load the saved VecNormalize statistics
  eval_env.training = False # set the evaluation environment to evaluation mode
  eval_env.norm_reward = False # disable reward normalization

  model = PPO.load(f"ppo_flight_model_{seed}", env=eval_env) # load the trained model
  eval_env.seed(eval_seed)
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

  plt.suptitle(f'PPO Performance (Seed {eval_seed})', fontsize=16)
  plt.tight_layout()
  plt.savefig(f'ppo_trajectory_{seed}.png') # save the figure
  plt.close()

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

  plt.suptitle(f'PPO Metrics (All Seeds)', fontsize=16)
  plt.tight_layout()
  plt.savefig(f'ppo_metrics_all_seeds.png') # save the figure
  plt.close()

def plot_fuel(train_seed, eval_seeds):

    # create new instance of environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize.load(f"ppo_vecnormalize.pkl_{train_seed}", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model = PPO.load(f"ppo_flight_model_{train_seed}", env=eval_env)

    fuel_consumed = [] # initialize empty list for total fuel consumption

    for es in eval_seeds:
        eval_env.seed(es)
        obs = eval_env.reset()
        fuel_step = [] # initialize empty list for per seed consumption

        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

            # termination condition
            if done[0]:
                break

            s = eval_env.envs[0].unwrapped.state # get observations
            fuel_step.append(abs(s[8]*0.5*1/360)) # append fuel used per timestep

        fuel_consumed.append(fuel_step) # append total fuel consumption
    
    eval_env.close()

    # Plot all 20 individual runs faintly + the mean error curve in bold
    plt.figure(figsize=(10, 4))
    for sd in fuel_consumed:
        plt.plot(sd, alpha=0.2) # plot fuel consumption for all 20 evaluation runs

    # Pad/truncate to uniform length if episodes vary, or use np.mean if all are 1000 steps
    mean_consumption = np.mean(fuel_consumed, axis=0)
    plt.plot(mean_consumption, color="navy", linewidth=2, label="Mean Fuel Consumption")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Fuel Burned [lbf]")
    plt.title(f"PPO Fuel Consumed Across All 20 Eval Seeds (Train Seed {train_seed})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ppo_all_fuel_consumed{train_seed}.png")
    plt.close()

# PPO evaluation function ---------------------------------------------------------------------

def evaluate_ppo():
    records = []

    for train_seed in train_seeds:
        # build evaluation env and load frozen normalization stats for each trained seed
        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize.load(f"ppo_vecnormalize.pkl_{train_seed}", eval_env)
        eval_env.training = False # set the evaluation environment to evaluation mode
        eval_env.norm_reward = False # disable reward normalization

        model = PPO.load(f"ppo_flight_model_{train_seed}", env=eval_env) # load trained model

        for es in eval_seeds:
            eval_env.seed(es)
            obs = eval_env.reset()

            # initialize reward, termination condition, and speed error squared
            total_reward = 0.0
            sq_speed_errors = []
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, info = eval_env.step(action)

                total_reward += float(reward[0]) # append reward
                done = bool(dones[0]) # redefine termination condition

                s = eval_env.envs[0].unwrapped.state # get states
                speed_err = float(s[3] - s[2])  # target speed - current speed
                sq_speed_errors.append(speed_err ** 2) # append squared speed error

            # append training and evaluation information
            records.append({
                "controller": "PPO",
                "train_seed": train_seed,
                "eval_seed": es,
                "return": total_reward,
                "speed_rmse_knots": float(np.sqrt(np.mean(sq_speed_errors))),
                "final_fuel": float(eval_env.envs[0].unwrapped.state[6]),
            })

        eval_env.close()

        plot_trajectories(train_seed, eval_seed=1000) # plots cruise speed, altitude, and reward over seed

    plot_metrics(train_seeds) # plots PPO metrics over all seeds

    df = pd.DataFrame(records)
    df.to_csv("ppo_eval_results.csv", index=False)

    # summary per training seed (across all 20 eval seeds)
    print("\n--- Per-Training-Seed Summary (20 Eval Seeds Each) ---")
    print(df.groupby("train_seed")[["return", "speed_rmse_knots", "final_fuel"]].agg(["mean", "std"]))

    # overall summary across all 5 train seeds x 20 eval seeds = 100 runs
    print(
        f"\nOverall PPO Return:     {df['return'].mean():.2f} +/- {df['return'].std():.2f}\n"
        f"Overall Speed RMSE:     {df['speed_rmse_knots'].mean():.2f} +/- {df['speed_rmse_knots'].std():.2f} knots"
    )
    return df


if __name__ == "__main__":
    evaluate_ppo()

