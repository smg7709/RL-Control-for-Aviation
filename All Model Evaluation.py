import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

eval_seeds = list(range(1000, 1020))
algorithms = ["PPO", "SAC", "PI"]
COLORS = {"PPO": "tab:orange", "SAC": "tab:blue", "PI": "tab:green"}


def prepare_controllers(ppo_df: pd.DataFrame, sac_df: pd.DataFrame, pi_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([ppo_df, sac_df, pi_df], ignore_index=True)

    df["controller"] = df["controller"].astype(str).str.strip().str.upper()
    df["eval_seed"] = pd.to_numeric(df["eval_seed"], errors="coerce").astype("Int64")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    df["fuel_burned_per_step"] = pd.to_numeric(df["fuel_burned_per_step"], errors="coerce")
    df["speed_rmse_knots"] = pd.to_numeric(df["speed_rmse_knots"], errors="coerce")

    return df

def plot_controller_metric(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    save_path: str | None = None,):

    fig, ax = plt.subplots(figsize=(11, 5))

    for ctrl in algorithms:
        sub = df[df["controller"] == ctrl]
        vals = np.array([
            sub[sub["eval_seed"] == es][metric_col].mean() for es in eval_seeds
        ])

        ax.plot(
            eval_seeds,
            vals,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=ctrl,
            color=COLORS[ctrl],
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("Evaluation Seed")
    ax.set_ylabel(ylabel)
    ax.set_xticks(eval_seeds)
    ax.set_xticklabels([str(es) for es in eval_seeds], rotation=45, ha="right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Controller")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(ppo_df: pd.DataFrame, sac_df: pd.DataFrame, pi_df: pd.DataFrame) -> pd.DataFrame:
    all_df = prepare_controllers(ppo_df, sac_df, pi_df)

    # Graph 1: Episode Reward across evaluation seeds
    plot_controller_metric(
        df=all_df,
        metric_col="return",
        title="Mean Episode Reward Across Evaluation Seeds",
        ylabel="Mean Return",
        save_path="reward_by_eval_seed.png",
    )

    # Graph 2: Fuel Burned per Step across evaluation seeds
    plot_controller_metric(
        df=all_df,
        metric_col="fuel_burned_per_step",
        title="Mean Fuel Burned per Step Across Evaluation Seeds",
        ylabel="Fuel Burned per Step",
        save_path="fuel_per_step_by_eval_seed.png",
    )

    return all_df

# get dataframes from csv's
pi_df = pd.read_csv("pi_eval_results.csv")
ppo_df = pd.read_csv("ppo_eval_results.csv")
sac_df = pd.read_csv("sac_eval_results.csv")

# plot results
all_df = plot_comparison(pi_df, ppo_df, sac_df)
