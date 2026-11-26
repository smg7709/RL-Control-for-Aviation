import numpy as np
import pandas as pd
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

def evaluate_pi(eval_seeds):
    env = make_env()
    pi_contoller = PIController(
        speed_delta = env.unwrapped.speed_delta,
        altitude_delta = env.unwrapped.altitude_delta,
        kps=0.5,
        kis=0.005,
        kpa=0.5,
        kia=0,
        dt=1/360,
    )
    records = []

    for es in eval_seeds:
        obs = env.reset(seed=es)

        controller = PIController(speed_delta = env.unwrapped.speed_delta, altitude_delta = env.unwrapped.altitude_delta)
        controller.reset()

        total_reward = 0
        sq_speed_errors = []
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
            f"Overall Speed RMSE:     {df['speed_rmse_knots'].mean():.2f} +/- {df['speed_rmse_knots'].std():.2f} knots\n"
            f"Overall Fuel Used:     {df['fuel_burned_per_step'].mean():.2f} +/- {df['fuel_burned_per_step'].std():.2f} lbs\n"
        )
    return df

if __name__ == "__main__":
    print("Running PI evaluation...")
    evaluate_pi(eval_seeds)