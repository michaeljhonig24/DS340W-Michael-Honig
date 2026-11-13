# Houston heat-health: spatial and two-stage impact experiment

## What this implements
A runnable extension of the Houston PyGRF experiment using observed seasonal
weather and ACS predictors. It adapts the classification-then-magnitude idea
from Parent Paper 3 and combines its count predictions with Parent Paper 2's
PyGRF predictions. It does not yet integrate UCformer outputs.

The target is published residential ZIP-year heat-related illness visit counts.
It is NOT a direct measure of assistance required, individual danger, unique
patients, or causal effects. A future assistance tool would require suitable
labels and independent operational validation.

## Execute in WSL
Extract Houston_Integrated_Experiment into the existing repository's
Modified_Implementation directory. Keep the earlier Houston_Heat_Health folder.

```bash
conda activate houston-health
cd /mnt/c/Users/micha/OneDrive/Documents/GitHub/DS340W-Michael-Honig/Modified_Implementation/Houston_Integrated_Experiment
python -m pip install -r requirements.txt
python -u run_experiment.py
python verify_outputs.py
```

This regenerates the six original approaches under results/ and the integrated
comparison under integrated_results/. It does not change files in the earlier
implementation or train UCformer. Named outputs are overwritten on rerun.
The single-threaded geographic forests can take several minutes.

## Design and execution order
1. run_baseline.py reads source.xlsx, validates and prepares rows, tunes the
   original RF/PyGRF settings on 2024, and exports their predictions.
2. run_experiment.py loads that freshly generated modeling table.
3. Training uses 2023 (121 rows), validation 2024 (121 rows). Final refit uses
   both years (242 rows) and evaluation uses 2025 (114 published rows).
4. All published counts are positive (minimum 5). We cannot identify unaffected
   ZIPs from absent/suppressed rows. Instead define higher burden as >= the
   2023 count upper quartile: 31 visits. Keep this threshold fixed during refit.
   This is an analytical count threshold, not a public-health decision threshold.
5. A binary XGBoost gate predicts the higher-burden probability. Two Poisson
   XGBoost regressors learn counts within lower/higher training groups.
6. Predicted count = (1-p)*lower_count + p*higher_count. This soft mixture is
   our adaptation; it is not a reproduction of the cyclone paper's outcome or
   exact implementation. Both count regressors predict every evaluation row.
7. Independently fit a single-stage Poisson XGBoost comparator on all counts.
8. Blend = w*two_stage + (1-w)*PyGRF. Choose w from .25/.5/.75 using 2024
   count RMSE. This combines predictions without feeding fitted training
   predictions back into a second learner; no stacking claims are made.
9. Select XGBoost depth from 2/3 using validation count RMSE; all other settings
   are fixed (80 trees, learning rate .05, minimum child weight 5, L2 5, seed 42).
   Model-family selection also uses validation count RMSE. Write locked config
   before final XGBoost fitting/scoring. Test ranking does not retune settings.
10. Export nine-model count comparison, per-ZIP predictions, gate diagnostics,
    component predictions, figure, audit, and fitted XGBoost model files.

Features: log population, age 65+ %, poverty %, renter %, four temperature/
precipitation summaries, and three humidity/heat-index summaries. Coordinates
are used by PyGRF only. ZIP, year, observed HRI, and prior counts are not XGBoost
predictors. Prior counts appear only in the separate prior-year baseline.

## Files to show in a demonstration
- run_experiment.py: fit_two and predict_two explain the two stages.
- integrated_results/locked_configuration.json: threshold and validation choices.
- integrated_results/validation_candidates.csv: every new candidate tried.
- integrated_results/test_2025_comparison.csv and comparison.png: final comparison.
- integrated_results/two_stage_details.csv: probability, branch counts, spatial
  prediction, and combined prediction for each ZIP.
- integrated_results/classification_metrics.json: discrimination and Brier score,
  plus a training-prevalence probability baseline. Probabilities are uncalibrated.
- integrated_results/audit.json: versions, counts, provenance, and feature list.

## Limits that belong in the assignment
2025 baseline results had already been viewed before designing this extension.
Thus this is an exploratory chronological comparison, not an untouched test
confirmation. A single split/seed does not establish statistical significance.
Missing outcomes are unknown, not zero; scores apply only to published records.
HRI reporting-window alignment and screenshot transcription need original-source
verification. ZIP/ZCTA correspondence and ACS population are proxies. Repeated
ZIPs and shared weather stations induce dependence; this is not new-geography
validation. The 2025 count distribution differs from earlier years. Observed
seasonal weather supports retrospective modeling, not a preseason forecast.
Count-based higher burden favors larger populations; secondary rate RMSE does
not make the classifier a population-adjusted risk classifier.

UCformer reproduction succeeded on original cities, but the approximate Houston
transfer failed. It is excluded from these predictors. CLMU pilot outputs are
not assigned to all ZIPs or mislabeled as observed exposure.

## Source attribution
Parent 2: supplied PyGRFBuilder is retained under vendor with the original source
and GPLv3 license. See BASELINE_METHOD.md for original extraction details.
Parent 3: new implementation inspired by the two-stage architecture, not copied
parent code and not a cyclone benchmark reproduction:
https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1996/
https://github.com/fedemoss/global-tc-impact-model
XGBoost objective definitions: https://xgboost.readthedocs.io/en/stable/parameter.html
Workbook includes source notes; its checksum is in the audit.

## Suggested GitHub commits (from repository root)
Inspect git status and staged changes before committing. Do not stage unrelated
changes. Use GitHub Desktop Push origin if WSL HTTPS authentication is unavailable.

```bash
git add Modified_Implementation/Houston_Integrated_Experiment/run_experiment.py Modified_Implementation/Houston_Integrated_Experiment/run_baseline.py Modified_Implementation/Houston_Integrated_Experiment/verify_outputs.py Modified_Implementation/Houston_Integrated_Experiment/requirements.txt Modified_Implementation/Houston_Integrated_Experiment/source.xlsx Modified_Implementation/Houston_Integrated_Experiment/vendor Modified_Implementation/Houston_Integrated_Experiment/.gitignore
git diff --cached --stat
git commit -m "Add Houston two-stage XGBoost and PyGRF blend experiment"
git add Modified_Implementation/Houston_Integrated_Experiment/results Modified_Implementation/Houston_Integrated_Experiment/integrated_results Modified_Implementation/Houston_Integrated_Experiment/README.md Modified_Implementation/Houston_Integrated_Experiment/BASELINE_METHOD.md Modified_Implementation/Houston_Integrated_Experiment/EXPLAIN_THE_RUN.md
git diff --cached --stat
git commit -m "Document and record exploratory Houston impact comparisons"
```
