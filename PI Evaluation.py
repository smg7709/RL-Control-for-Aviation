import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Environment import FlightEnv
from PIController import PIController
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


train_seeds = [10, 11, 12, 13, 14]
eval_seeds = list(range(1000, 1020))

def make_env():
    return TimeLimit(FlightEnv(), max_episode_steps=1000) # limit the number of steps per episode to 1000 (~2.75 hrs of flight)

def reset_env(env, seed: int):
    if isinstance(env, VecEnv):
        env.seed(seed)
        return env.reset()
    obs, _ = env.reset(seed=seed)
    return obs

def step_env(env, action):
    if isinstance(env, VecEnv):
        obs, reward, done, info = env.step(action)
        return obs, float(reward[0]), bool(done[0]), info[0]
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, float(reward), bool(terminated or truncated), info

def plot_trajectories(eval_seeds):

  eval_env = DummyVecEnv([make_env]) # create a new evaluation environment
  unwrapped_env = eval_env.envs[0].unwrapped

  controller = PIController(
      speed_delta=unwrapped_env.speed_delta,
      altitude_delta=unwrapped_env.altitude_delta,
      pitch_delta=unwrapped_env.pitch_delta)

  for es in eval_seeds:
    eval_env.seed(es)
    eval_env.reset()
    controller.reset()

    # assign empty lists to store trajectories
    fuel_history = []
    speed_history = []
    target_speed_history = []
    alt_history = []
    target_alt_history = []
    reward_history = []

    for _ in range(1000):
        s = unwrapped_env.state.copy()
        action, _ = controller.predict(s, deterministic=True)
        _, reward, done, _ = eval_env.step(action[None, :])
        
        if done[0]: # check if the episode is done
            break

        # append trajectories to lists
        s_n = unwrapped_env.state
        reward_history.append(float(reward[0]))
        fuel_history.append(float(s_n[6]))
        speed_history.append(float(s_n[2]))
        alt_history.append(float(s_n[4]))
        target_speed_history.append(float(s_n[3]))
        target_alt_history.append(float(s_n[5]))

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

    plt.suptitle(f'PI Controller Performance (Evaluation Seed {es})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'pi_trajectory_e{es}.png') # save the figure
    plt.close()

    eval_env.close() # close the evaluation environment


def evaluate_pi(eval_seeds):
    env = make_env()
    pi_contoller = PIController(
        speed_delta = env.unwrapped.speed_delta,
        altitude_delta = env.unwrapped.altitude_delta,
        pitch_delta = env.unwrapped.pitch_delta,
        kps=0.5,
        kis=0.005,
        kpa=0.5,
        kia=0,
        kpp=0.5,
        kip=0,
        dt=1/360,
    )
    records = []

    for es in eval_seeds:
        obs = env.reset(seed=es)

        controller = PIController(speed_delta = env.unwrapped.speed_delta, altitude_delta = env.unwrapped.altitude_delta, pitch_delta = env.unwrapped.pitch_delta)
        controller.reset()

        total_reward = 0
        fuel_burned = []
        done = False

        while not done:
            s = env.unwrapped.state

            action, _ = controller.predict(s)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            done = terminated or truncated

            sn = env.unwrapped.state
            alt_err = float(sn[5]-sn[4]) # altitude error
            speed_err = float(sn[3]-sn[2]) # speed error
            pitch_err = float(3-sn[9]) # pitch error
            fuel_step = abs(sn[8]*0.5*1/360) # fuel burned per step
            fuel_burned.append(fuel_step) # append fuel burned

        # summarize fuel burned into an average per step
        fuel_burned_per_step = float(np.mean(fuel_burned))

        # append metrics to file
        records.append({
            "controller": "PI",
            "eval_seed": es,
            "return": total_reward,
            "altitude error": alt_err,
            "speed error": speed_err,
            "pitch error": pitch_err,
            "fuel_burned_per_step": fuel_burned_per_step
        })
    
    env.close()

    # save data frame to csv
    df = pd.DataFrame(records)
    out_csv = "pi_eval_results.csv"
    df.to_csv(out_csv, index=False)

    summary = df.groupby("eval_seed", as_index=False).agg(
        mean_reward=("return", "mean"),
        std_reward=("return", "std"),
        mean_fuel=("fuel_burned_per_step", "mean"),
    )
    print(f"Saved {len(df)} PI evaluation rows to {out_csv}")
    print(summary.to_string(index=False))

    print(
            f"\nOverall PI Return:     {df['return'].mean():.2f} +/- {df['return'].std():.2f}\n"
            f"Overall Fuel Used:     {df['fuel_burned_per_step'].mean():.2f} +/- {df['fuel_burned_per_step'].std():.2f} lbs\n"
        )
    return df

if __name__ == "__main__":
    print("Running PI evaluation...")
    evaluate_pi(eval_seeds)
    plot_trajectories(eval_seeds)