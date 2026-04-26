import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="test",
                        help="Folder under results/ containing diam_trial_*.csv")
    parser.add_argument("--max_index", type=int, default=1000,
                        help="Maximum iteration index to plot")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT_DIR, "results", args.experiment_name, "diam_trial_*.csv")))
    if not files:
        raise FileNotFoundError(f"No trial files found for experiment '{args.experiment_name}'.")

    landmark_series_list = []
    max_i = 0
    for fn in files:
        df = pd.read_csv(fn)
        s = pd.Series(df["landmark_diameter"].values, index=df["i"].values)
        max_i = max(max_i, int(s.index.max()))
        landmark_series_list.append(s)

    max_i = min(max_i, args.max_index)
    index = np.arange(0, max_i + 1)

    df_all = pd.DataFrame({t: s.reindex(index, method="ffill") for t, s in enumerate(landmark_series_list)})
    mean_landmark = df_all.mean(axis=1)
    y_min_landmark = df_all.min(axis=1)
    y_max_landmark = df_all.max(axis=1)

    fig_landmark, ax_landmark = plt.subplots(figsize=(10, 6))
    ax_landmark.plot(index, mean_landmark, color="tab:green", linewidth=2, label="Mean Landmark Diameter")
    ax_landmark.fill_between(index, y_min_landmark, y_max_landmark, color="tab:green", alpha=0.2, label="Min/Max envelope")
    ax_landmark.set_xlabel("Iterations (k)", fontsize=20)
    ax_landmark.set_ylabel(r"Landmark Estimate Diameter $\hat{M}[k]$", fontsize=20)
    ax_landmark.grid(True, alpha=0.3)
    ax_landmark.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "landmark_estimation_error.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_landmark)
    print(f"Saved landmark_estimation_error.png for '{args.experiment_name}' in {ROOT_DIR}.")


if __name__ == "__main__":
    main()
