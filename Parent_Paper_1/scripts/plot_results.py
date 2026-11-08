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
    metrics = pd.read_csv(folder / "metrics_by_city.csv")
    cities = metrics["city"].tolist()

    # Show the same initial sample window for every city.
    fig, axes = plt.subplots(
        3, 2, figsize=(13, 10), constrained_layout=True
    )

    for ax, city in zip(axes.flat, cities):
        data = pd.read_csv(folder / f"{city}_predictions.csv").head(80)

        ax.plot(
            data["sample_index"],
            data["T_reference"] - 273.15,
            label="Simulation target",
            color="#243746",
            linewidth=1.8,
        )
        ax.plot(
            data["sample_index"],
            data["T_prediction"] - 273.15,
            label="UCFormer prediction",
            color="#d55e00",
            linestyle="--",
            linewidth=1.5,
        )
        ax.set_title(city)
        ax.set_xlabel("Test sample index (first 80 samples)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.25)

    axes.flat[0].legend(fontsize=9)
    fig.suptitle(
        "UCFormer temperature predictions against supplied simulation targets",
        fontsize=14,
    )
    fig.savefig(folder / "temperature_predictions.png", dpi=200)
    plt.close(fig)

    # Separate panels because temperature and humidity have different units.
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    panels = [
        ("T_RMSE", "Temperature RMSE (K)", "#0072b2"),
        ("q_RMSE_g_per_kg", "Specific humidity RMSE (g/kg)", "#009e73"),
    ]

    for ax, (column, title, color) in zip(axes, panels):
        bars = ax.bar(cities, metrics[column], color=color)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_title(title)
        ax.set_ylabel("RMSE — lower is better")
        ax.set_ylim(0, metrics[column].max() * 1.2)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("UCFormer evaluation across the full test set", fontsize=14)
    fig.savefig(folder / "rmse_by_city.png", dpi=200)
    plt.close(fig)

    print(f"Figures saved to: {folder.resolve()}")
    print("  temperature_predictions.png")
    print("  rmse_by_city.png")

if __name__ == "__main__":
    main()
