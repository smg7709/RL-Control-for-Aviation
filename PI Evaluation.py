from pathlib import Path
import numpy as np
import pandas as pd
from Environment import FlightEnv
from PIController import PIController
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


train_seeds = [10, 11, 12, 13, 14]
eval_seeds = list(range(1000, 1020))

def make_env():
    return TimeLimit(FlightEnv(), max_episode_steps=1000) # limit the number of steps per episode to 1000 (~2.75 hrs of flight)

def reset_env(env, seed: int):
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result

def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, float(reward), bool(done), info

def evaluate_pi():
    records = []

    for train_seed in train_seeds:
        eval_env = DummyVecEnv([make_env])
        eval_env.training = False
        eval_env.norm_reward = False

        for es in eval_seeds:
            eval_env.seed(es)
            obs = eval_env.reset()

            controller = PIController()
            controller.reset()

            total_reward = 0
            sq_speed_errors = []
            fuel_burned = []
            done = False

            while not done:
                single_action, _ = controller.predict(obs[0], deterministic=True)

                batched_action = np.asarray(single_action, dtype=np.float32).reshape(1, -1)
                obs, reward, dones, info = eval_env.step(batched_action)

                total_reward += float(reward[0])
                done = bool(dones[0])

                s = eval_env.envs[0].unwrapped.state
                speed_err = float(s[3]-s[2])
                sq_speed_errors.append(speed_err**2)
                fuel_step = abs(s[8]*0.5*1/360) # fuel burned per step
                fuel_burned.append(fuel_step) # append fuel burned

            # summarize fuel burned into an average per step
            fuel_burned_per_step = float(np.mean(fuel_burned))

            records.append({
                "controller": "PI",
                "train_seed": train_seed,
                "eval_seed": es,
                "return": total_reward,
                "speed_rmse_knots": float(np.sqrt(np.mean(sq_speed_errors))),
                "fuel_burned_per_step": fuel_burned_per_step
            })
        eval_env.close()

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
    evaluate_pi()