import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import configs


def _load_trials(experiment_dir):
    files = sorted(glob.glob(os.path.join(experiment_dir, "diam_trial_*.csv")))
    if not files:
        raise FileNotFoundError(f"No trial files found in: {experiment_dir}")

    diam_series_list = []
    landmark_series_list = []
    max_i = 0

    for fn in files:
        df = pd.read_csv(fn)
        s_diam = pd.Series(df["diameter"].values, index=df["i"].values)
        s_land = pd.Series(df["landmark_diameter"].values, index=df["i"].values)

        max_i = max(max_i, int(s_diam.index.max()))
        diam_series_list.append(s_diam)
        landmark_series_list.append(s_land)

    return diam_series_list, landmark_series_list, max_i


def _series_stats(series_list, index, fill_value=None):
    kwargs = dict(method="ffill")
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    df_all = pd.DataFrame({t: s.reindex(index, **kwargs) for t, s in enumerate(series_list)})

    mean = df_all.mean(axis=1)
    std = df_all.std(axis=1)
    ymin = df_all.min(axis=1)
    ymax = df_all.max(axis=1)

    lower = np.maximum(mean - std, ymin)
    upper = np.minimum(mean + std, ymax)
    return mean, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="test",
                        help="Folder under results/ containing diam_trial_*.csv")
    parser.add_argument("--max_index", type=int, default=1000,
                        help="Maximum iteration index to plot")
    args = parser.parse_args()

    experiment_dir = os.path.join(ROOT_DIR, "results", args.experiment_name)
    diam_series_list, landmark_series_list, max_i = _load_trials(experiment_dir)

    max_i = min(max_i, args.max_index)
    index = np.arange(0, max_i + 1)
    initial_diam = np.hypot(configs.red_width, configs.red_height)

    mean_diam, lower_diam, upper_diam = _series_stats(diam_series_list, index, fill_value=initial_diam)
    mean_land, lower_land, upper_land = _series_stats(landmark_series_list, index)

    # State estimate diameter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(index, mean_diam, color="tab:blue", linewidth=2, label="Mean")
    ax.fill_between(index, lower_diam, upper_diam, color="tab:blue", alpha=0.2, label="Mean ± std (clipped)")
    ax.set_xlabel("Iterations (k)", fontsize=20)
    ax.set_ylabel(r"Initial State Estimate Diameter $\hat{X}_0[k]$", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "state_estimation_error.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Landmark estimate diameter plot
    fig_land, ax_land = plt.subplots(figsize=(10, 6))
    ax_land.plot(index, mean_land, color="tab:green", linewidth=2, label="Mean")
    ax_land.fill_between(index, lower_land, upper_land, color="tab:green", alpha=0.2, label="Mean ± std (clipped)")
    ax_land.set_xlabel("Iterations (k)", fontsize=20)
    ax_land.set_ylabel(r"Landmark Estimate Diameter $\hat{M}[k]$", fontsize=20)
    ax_land.grid(True, alpha=0.3)
    ax_land.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "landmark_estimation_error.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_land)

    print(f"Saved plots for '{args.experiment_name}' in {ROOT_DIR}.")


if __name__ == "__main__":
    main()
