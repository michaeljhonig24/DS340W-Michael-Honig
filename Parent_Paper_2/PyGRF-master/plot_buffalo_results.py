import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    folder = args.results

    predictions = pd.read_csv(folder / "held_out_predictions.csv")
    summary = pd.read_csv(folder / "summary_metrics.csv").iloc[0]
    importance = pd.read_csv(folder / "global_feature_importance.csv")

    actual = predictions["311_requests"]
    predicted = predictions["prediction"]

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(
        actual, predicted, color="#0072b2", alpha=0.65,
        s=35, edgecolors="white", linewidths=0.4,
    )

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    padding = max((upper - lower) * 0.05, 1)
    limits = [lower - padding, upper + padding]

    ax.plot(limits, limits, "--", color="#d55e00", label="Perfect prediction")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference 311-request target (dataset units)")
    ax.set_ylabel("Held-out prediction (dataset units)")
    ax.set_title("Buffalo PyGRF: 10-fold cross-validation")
    ax.text(
        0.04, 0.96,
        f"N = {len(predictions)}\n"
        f"Pooled RMSE = {summary['pooled_RMSE']:.4f}\n"
        f"Pooled R² = {summary['pooled_R2']:.4f}",
        transform=ax.transAxes, va="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8"),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.savefig(folder / "held_out_predictions.png", dpi=200)
    plt.close(fig)

    top = (
        importance.nlargest(10, "mean_global_importance")
        .sort_values("mean_global_importance")
    )
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.barh(
        top["feature"], top["mean_global_importance"],
        color="#009e73",
    )
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_xlim(0, top["mean_global_importance"].max() * 1.18)
    ax.set_xlabel("Mean impurity-based importance across 10 global forests")
    ax.set_title("Buffalo PyGRF: top 10 global predictors")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(folder / "global_feature_importance.png", dpi=200)
    plt.close(fig)

    print(f"Figures saved to: {folder.resolve()}")

if __name__ == "__main__":
    main()
