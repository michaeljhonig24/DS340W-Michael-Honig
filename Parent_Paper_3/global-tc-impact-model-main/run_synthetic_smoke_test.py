"""Synthetic execution test of the authors' model; not paper reproduction."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.models.two_stage_xgb import TwoStageXGBoost

SEED = 42
rng = np.random.default_rng(SEED)
target = "perc_affected_pop_grid_region"
features = ["synthetic_wind", "synthetic_rain", "synthetic_vulnerability"]

# Artificial relationships, deliberately unrelated to the paper's accuracy.
n = 1200
wind = rng.uniform(0, 1, n)
rain = rng.uniform(0, 1, n)
vulnerability = rng.uniform(0, 1, n)
affected = wind + 0.35 * rain > 0.65
magnitude = np.clip(
    2 + 12 * wind + 35 * vulnerability + rng.normal(0, 2, n),
    1, 100
)

data = pd.DataFrame({
    features[0]: wind,
    features[1]: rain,
    features[2]: vulnerability,
    target: np.where(affected, magnitude, 0),
})
train, test = train_test_split(
    data, test_size=0.25, random_state=SEED,
    stratify=(data[target] > 0)
)
train_before = train.copy(deep=True)
test_before = test.copy(deep=True)

# Ensure the author's balancing routine receives both impact categories.
assert (train[target] == 0).any()
assert ((train[target] > 0) & (train[target] < 15)).any()
assert (train[target] >= 15).any()

model = TwoStageXGBoost(
    features=features,
    target_name=target,
    use_tuned=False,
)
started = time.perf_counter()
model.fit(train)
result = model.predict(test)
elapsed = time.perf_counter() - started

checks = {}

def check(name, condition):
    checks[name] = bool(condition)
    print(f"{'PASS' if condition else 'FAIL'}: {name}")

check("Both stages fitted",
      model.classifier.get_booster().num_boosted_rounds() > 0
      and model.regressor.get_booster().num_boosted_rounds() > 0)
check("One prediction per held-out row",
      len(result) == len(test)
      and result.index.equals(test.sort_index().index))
check("Finite predictions and probabilities",
      np.isfinite(result[["prediction_perc", "predicted_proba"]].to_numpy()).all())
check("Probabilities between zero and one",
      result["predicted_proba"].between(0, 1).all())
check("Default classification follows threshold",
      np.array_equal(
          result["predicted_bin"].to_numpy(),
          (result["predicted_proba"] >= model.clf_threshold).astype(int).to_numpy()
      ))
check("Negative classifications produce zero impact",
      (result.loc[result["predicted_bin"] == 0, "prediction_perc"] == 0).all())

# Exercise both gating branches explicitly.
all_negative = model.predict(test, clf_threshold=1.1)
check("Forced negative gate produces all zeros",
      (all_negative["predicted_bin"] == 0).all()
      and (all_negative["prediction_perc"] == 0).all())

all_positive = model.predict(test, clf_threshold=0.0)
expected = model.regressor.predict(test.sort_index()[features])
check("Forced positive gate returns regressor predictions",
      (all_positive["predicted_bin"] == 1).all()
      and np.allclose(all_positive["prediction_perc"], expected))

# predict() requires targets for its reporting column, but must not use
# those labels to determine model predictions.
changed_labels = test.copy()
changed_labels[target] = 0.0
changed_result = model.predict(changed_labels)
check("Predictions independent of supplied test labels",
      np.allclose(result["prediction_perc"], changed_result["prediction_perc"])
      and np.allclose(result["predicted_proba"], changed_result["predicted_proba"]))

check("Input frames unchanged",
      train.equals(train_before) and test.equals(test_before))

rmse = float(np.sqrt(mean_squared_error(result[target], result["prediction_perc"])))
r2 = float(r2_score(result[target], result["prediction_perc"]))

out = Path("results") / datetime.now().strftime("synthetic_smoke_%Y%m%d_%H%M%S")
out.mkdir(parents=True, exist_ok=True)
result.to_csv(out / "synthetic_predictions.csv", index_label="sample_id")
pd.DataFrame([{
    "test_type": "synthetic_smoke_test",
    "train_rows": len(train),
    "test_rows": len(test),
    "synthetic_RMSE": rmse,
    "synthetic_R2": r2,
    "seconds": elapsed,
    "all_checks_passed": all(checks.values()),
}]).to_csv(out / "summary_metrics.csv", index=False)

source = Path("src/models/two_stage_xgb.py")
report = {
    "scope": "Synthetic execution test, NOT reproduction of paper performance",
    "seed": SEED,
    "features": features,
    "use_tuned": False,
    "clf_params": model.clf_params,
    "reg_params": model.reg_params,
    "u1": model.u1,
    "u2": model.u2,
    "clf_threshold": model.clf_threshold,
    "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    },
    "checks": checks,
}
(out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(out / "README.txt").write_text(
    "Synthetic smoke test of the authors' TwoStageXGBoost class.\n"
    "Uses built-in defaults, not the tuned research configuration.\n"
    "Metrics describe artificial data only and must not be compared "
    "with the paper's reported results.\n"
    "Does not test data acquisition, geographic processing, or "
    "the paper's full evaluation pipeline.\n",
    encoding="utf-8"
)

print(f"\nSynthetic-only RMSE: {rmse:.4f}")
print(f"Synthetic-only R2: {r2:.4f}")
print(f"Results saved to: {out.resolve()}")

if not all(checks.values()):
    raise SystemExit("SMOKE TEST FAILED: inspect test_report.json")
print("SMOKE TEST PASSED")

