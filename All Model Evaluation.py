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
    df["speed error"] = pd.to_numeric(df["speed error"], errors="coerce")
    df["altitude error"] = pd.to_numeric(df["altitude error"], errors="coerce")
    df["pitch error"] = pd.to_numeric(df["pitch error"], errors="coerce")

    return df

def plot_single(
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

def plot_triple(
    df: pd.DataFrame,
    metrics: list[tuple[str, str]],
    title: str,
    ylabel: str,
    save_path: str | None = None,):

    fig, axes = plt.subplots(3, 1, sharex = True, figsize=(11,8))

    for ax, (metric_col, ylabel) in zip(axes, metrics):
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

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_title(title)
    axes[0].legend(title="Controller")
    axes[-1].set_xlabel("Evaluation Seed")
    axes[-1].set_xticks(eval_seeds)
    axes[-1].set_xticklabels([str(es) for es in eval_seeds], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)






def plot_comparison(ppo_df: pd.DataFrame, sac_df: pd.DataFrame, pi_df: pd.DataFrame) -> pd.DataFrame:
    all_df = prepare_controllers(ppo_df, sac_df, pi_df)

    # plot 1: episode reward across evaluation seeds
    plot_single(
        df=all_df,
        metric_col="return",
        title="Mean Episode Reward Across Evaluation Seeds",
        ylabel="Mean Return",
        save_path="reward_by_eval_seed.png",
    )

    # plot 2: fuel burned per step across evaluation seeds
    plot_single(
        df=all_df,
        metric_col="fuel_burned_per_step",
        title="Mean Fuel Burned per Step Across Evaluation Seeds",
        ylabel="Fuel Burned per Step",
        save_path="fuel_per_step_by_eval_seed.png",
    )

    # plot 3: speed, altitude, and pitch errors across evaluation seeds
    plot_triple(
        df=all_df,
        metrics=[("speed_error", "Speed Error [knots]"),
                 ("altitude_error", "Altitude Error [ft]"),
                 ("pitch_error", "Pitch Error [deg]"),],
        title="Speed, Altitude, and Pitch Errors Across Evaluation Seeds",
        save_path="tracking_errors_by_eval_seed.png",
    )

    return all_df

# get dataframes from csv's
pi_df = pd.read_csv("pi_eval_results.csv")
ppo_df = pd.read_csv("ppo_eval_results.csv")
sac_df = pd.read_csv("sac_eval_results.csv")

# plot results
all_df = plot_comparison(pi_df, ppo_df, sac_df)
