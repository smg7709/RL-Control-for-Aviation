import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_seeds = [10, 11, 12, 13, 14]
algorithms = ["PPO", "SAC", "PI"]
COLORS = {"PPO": "tab:orange", "SAC": "tab:blue", "PI": "tab:green"}


def prepare_controllers(ppo_df: pd.DataFrame, sac_df: pd.DataFrame, pi_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([ppo_df, sac_df, pi_df], ignore_index=True)

    df["controller"] = df["controller"].astype(str).str.strip().str.upper()
    df["train_seed"] = pd.to_numeric(df["train_seed"], errors="coerce").astype("Int64")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    df["fuel_burned_per_step"] = pd.to_numeric(df["fuel_burned_per_step"], errors="coerce")
    df["speed_rmse_knots"] = pd.to_numeric(df["speed_rmse_knots"], errors="coerce")

    return df

def plot_controller_metric(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    fmt: str,
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(train_seeds))
    width = 0.25
    offsets = np.linspace(-(len(algorithms) - 1) / 2, (len(algorithms) - 1) / 2, len(algorithms)) * width

    for offset, ctrl in zip(offsets, algorithms):
        sub = df[df["controller"] == ctrl]
        means = np.array([
            sub[sub["train_seed"] == sd][metric_col].mean() for sd in train_seeds
        ])
        stds = np.array([
            sub[sub["train_seed"] == sd][metric_col].std(ddof=1) for sd in train_seeds
        ])

        bars = ax.bar(
            x + offset,
            means,
            width,
            label=ctrl,
            color=COLORS[ctrl],
            alpha=0.9,
        )
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=7)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {sd}" for sd in train_seeds])
    ax.legend(title="Controller")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(ppo_df: pd.DataFrame, sac_df: pd.DataFrame, pi_df: pd.DataFrame):
    all_df = prepare_controllers(ppo_df, sac_df, pi_df)

    # Graph 1: Mean Episode Return across evaluation seeds
    plot_controller_metric(
        df=all_df,
        metric_col="return",
        title="Mean Episode Reward Across 20 Evaluation Seeds",
        ylabel="Mean Return",
        fmt="%.0f",
        save_path="mean_reward_all_controllers.png",
    )

    # Graph 2: Mean Fuel Used per Step across evaluation seeds
    plot_controller_metric(
        df=all_df,
        metric_col="fuel_burned_per_step",
        title="Mean Fuel Burned per Step Across 20 Evaluation Seeds",
        ylabel="Fuel Burned [lb]",
        fmt="%.2f",
        save_path="fuel_per_step_all_controllers.png",
    )

    return all_df

# get dataframes from csv's
pi_df = pd.read_csv("pi_eval_results.csv")
ppo_df = pd.read_csv("ppo_eval_results.csv")
sac_df = pd.read_csv("sac_eval_results.csv")

# plot results
all_df = plot_comparison(pi_df, ppo_df, sac_df)
