from pathlib import Path
from datetime import datetime
import json
import time
import platform
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import PyGRF

ROOT = Path(__file__).resolve().parent

def main():
    output = ROOT / "results" / datetime.now().strftime("buffalo_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(ROOT / "Data" / "311Request.csv")
    excluded = ["CBG ID", "Lon", "Lat", "311_requests"]
    features = [c for c in data.columns if c not in excluded]

    settings = {
        "n_estimators": 60,
        "max_features": 1 / 3,
        "band_width": 131,
        "train_weighted": True,
        "predict_weighted": True,
        "bootstrap": False,
        "resampled": True,
        "random_state": 42,
    }
    local_weight = 0.0444

    # Preserve the notebook's fixed parameters, rather than rerunning ISA tuning.
    metadata = {
        "model_settings": settings,
        "local_weight": local_weight,
        "folds": 10,
        "shuffle": True,
        "split_seed": 42,
        "features": features,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn",
                         "libpysal", "esda"]
        },
    }
    (output / "run_settings.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    predictions = np.full(len(data), np.nan)
    fold_ids = np.zeros(len(data), dtype=int)
    fold_results = []
    global_importances = []
    folds = KFold(n_splits=10, shuffle=True, random_state=42)
    start = time.perf_counter()

    for fold, (train_idx, test_idx) in enumerate(folds.split(data), start=1):
        print(f"Starting fold {fold}/10...", flush=True)
        fold_start = time.perf_counter()

        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        # Fit preprocessing using only this fold's training data.
        stats = train[features].describe().transpose()
        if (stats["std"].isna() | stats["std"].eq(0)).any():
            raise ValueError("A predictor has undefined or zero training standard deviation.")

        X_train = (train[features] - stats["mean"]) / stats["std"]
        X_test = (test[features] - stats["mean"]) / stats["std"]

        model = PyGRF.PyGRFBuilder(**settings)
        model.fit(
            X_train,
            train["311_requests"],
            train[["Lon", "Lat"]],
        )
        combined, _, _ = model.predict(
            X_test,
            test[["Lon", "Lat"]],
            local_weight=local_weight,
        )

        pred = np.asarray(combined, dtype=float).reshape(-1)
        if len(pred) != len(test_idx) or not np.isfinite(pred).all():
            raise ValueError("Invalid predictions returned by the model.")

        predictions[test_idx] = pred
        fold_ids[test_idx] = fold

        rmse = float(np.sqrt(mean_squared_error(test["311_requests"], pred)))
        r2 = float(r2_score(test["311_requests"], pred))
        elapsed = time.perf_counter() - fold_start

        fold_results.append({
            "fold": fold, "RMSE": rmse, "R2": r2, "seconds": elapsed
        })
        global_importances.append(model.global_model.feature_importances_)

        # Save local-model importance separately for each fold.
        local_fi = model.get_local_feature_importance()
        pd.DataFrame(local_fi).to_csv(
            output / f"local_feature_importance_fold_{fold:02d}.csv",
            index=True,
        )
        pd.DataFrame(fold_results).to_csv(output / "fold_metrics.csv", index=False)
        print(
            f"Fold {fold}: RMSE={rmse:.4f}, R2={r2:.4f}, "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    if not np.isfinite(predictions).all() or (fold_ids == 0).any():
        raise ValueError("Some rows did not receive held-out predictions.")

    # Match the notebook: calculate metrics across all held-out predictions.
    pooled_rmse = float(np.sqrt(
        mean_squared_error(data["311_requests"], predictions)
    ))
    pooled_r2 = float(r2_score(data["311_requests"], predictions))

    summary = pd.DataFrame([{
        "rows": len(data),
        "predictors": len(features),
        "folds": 10,
        "pooled_RMSE": pooled_rmse,
        "pooled_R2": pooled_r2,
        "total_seconds": time.perf_counter() - start,
    }])
    summary.to_csv(output / "summary_metrics.csv", index=False)

    result = data[["CBG ID", "Lon", "Lat", "311_requests"]].copy()
    result.insert(0, "source_row", np.arange(len(data)))
    result["fold"] = fold_ids
    result["prediction"] = predictions
    result.to_csv(output / "held_out_predictions.csv", index=False)

    pd.DataFrame({
        "feature": features,
        "mean_global_importance": np.mean(global_importances, axis=0),
    }).sort_values(
        "mean_global_importance", ascending=False
    ).to_csv(output / "global_feature_importance.csv", index=False)

    print("\nFINAL RESULTS — pooled held-out predictions")
    print(f"RMSE: {pooled_rmse:.4f}")
    print(f"R2: {pooled_r2:.4f}")
    print(f"Results saved to: {output}")

if __name__ == "__main__":
    main()
