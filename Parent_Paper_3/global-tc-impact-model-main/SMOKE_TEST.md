# Tropical Cyclone Model: Implementation Smoke Test

## Scope
Executed the authors' TwoStageXGBoost class using controlled synthetic
data and built-in model defaults. This is an execution test, not a
reproduction of the paper's reported performance.

## Run
python -u run_synthetic_smoke_test.py

## Results
- Training rows: 900
- Held-out rows: 300
- All 10 implementation checks passed.
- Synthetic-only RMSE: 5.8839
- Synthetic-only R2: 0.8743

Checks covered fitting both stages, output completeness, finite values,
probability bounds, classification gating, agreement with direct
regressor predictions, independence from test labels, and unchanged inputs.

## Limitations
The prepared research dataset was not present in the downloaded repository.
This test does not verify the geographic data pipeline, the paper's
evaluation protocol, or performance on real tropical cyclone events.
The authors' model source was not modified for this test.

## Evidence
results/synthetic_smoke_20261108_183745/
Dependency versions are recorded in requirements-smoke.txt.
