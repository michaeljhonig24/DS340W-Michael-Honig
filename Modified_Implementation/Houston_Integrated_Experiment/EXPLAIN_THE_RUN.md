# Demonstration walkthrough

## 1. State the research question
Can observed heat/humidity and socioeconomic features, combined with spatial
modeling and an adapted two-stage model, improve estimates of published Houston
heat-related illness visit counts?

Longer-term vision: modeled climate exposure could replace or augment observed
weather after validation. This run uses observed weather. Assistance need is
not measured by the available labels.

## 2. Show the input and chronological split
Open source.xlsx and results/modeling_table.csv. Each row is one ZIP/year.
2023 trains, 2024 selects settings, then 2023–2024 refit predicts 2025. Missing
published counts are excluded rather than labeled zero. Explain that 2025 had
already been inspected during baseline development, making this exploratory.

## 3. Explain what is modified
Parent 2 supplies geographic random forests. We adapt the task and predictors
to Houston. Parent 3 motivates separating impact classification and magnitude.
Our counts have no zeros, so we define higher burden using the 2023 upper
quartile (31 visits), not affected/unaffected. This is a new task-specific
adaptation, not an exact reproduction of the cyclone model.

## 4. Show fit_two and predict_two
A classifier estimates the probability of at least 31 visits. Two regressors
learn lower- and higher-burden count patterns. We combine their predicted
counts using the classifier probability. True 2025 counts are used only to
score, never to choose the branch during prediction.

## 5. Show the connection to PyGRF
The combined output is w times the two-stage count plus (1-w) times the PyGRF
count. Validation selected w=0.75. This actually combines predictions; it is
not just placing two independent models on a slide. A learned stacking model
would need appropriately held-out training predictions; this experiment uses
simple validation-selected blending instead.

## 6. Execute
Run python -u run_experiment.py in the houston-health environment. The baseline
runs first, then validation choices are locked and final models are refit.
Run python verify_outputs.py afterward. Show integrated_results/comparison.png
and test_2025_comparison.csv. The code exports all the numbers used in the figure.

## 7. Explain the actual result
2025 count RMSE: PyGRF 10.507; single-stage XGBoost 10.773; adapted two-stage
11.710; combined prediction 11.303. Lower is better. The new stages did not
improve count error over PyGRF. Every model's count R-squared is negative.
Across all nine methods, 2024 validation selected the prior-year baseline;
this differs from the earlier selection restricted to learned RF/PyGRF models.
Do not select the production model retrospectively using the 2025 ranking.

The classifier's 2025 ROC AUC is 0.893, but there are only eight positive ZIPs.
At a fixed 0.5 probability threshold it finds six of eight, with 18 false
positives: precision 25%, recall 75%. Brier score 0.094 is worse than the
training-prevalence baseline's 0.090. Ranking performance does not establish
reliable probabilities or readiness for assistance allocation.

## 8. Describe the contribution without overclaiming
We implemented and tested a spatial/two-stage heat-health extension, including
controlled feature comparisons and an actual prediction blend. The test did
not show a count-error improvement. Negative results guide the next revision.
Original HRI reporting periods and transcription still need verification.
The unsuccessful UCformer transfer is a separate attempted extension, not a
source of valid predictors in this experiment.

## 9. Show GitHub
Show source scripts, the input workbook, configuration, results and meaningful
commits. The commit history records implementation and reproduced results,
not a reconstructed or backdated history of experiments.
