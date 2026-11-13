# Houston integrated experiment

| model | count_RMSE | count_MAE | count_R2 | selected_on_validation |
| --- | --- | --- | --- | --- |
| Population-rate baseline | 14.752 | 11.407 | -2.108 | False |
| Prior-year count baseline | 11.907 | 9.249 | -1.025 | True |
| RF socioeconomic | 11.197 | 9.525 | -0.79 | False |
| RF + temperature | 11.124 | 9.466 | -0.767 | False |
| RF + heat index | 10.724 | 9.369 | -0.642 | False |
| PyGRF + heat index | 10.507 | 9.177 | -0.577 | False |
| XGB single-stage | 10.773 | 9.761 | -0.657 | False |
| XGB two-stage | 11.71 | 10.195 | -0.958 | False |
| PyGRF + two-stage blend | 11.303 | 9.901 | -0.824 | False |

Selected using 2024: Prior-year count baseline.

High burden is >= 31.0 published visits, not a clinical or assistance threshold. Probability estimates are uncalibrated.

2025 has already been inspected in earlier baseline work; this extension is exploratory. No UCformer predictions are used. See README.md for design and limitations.
