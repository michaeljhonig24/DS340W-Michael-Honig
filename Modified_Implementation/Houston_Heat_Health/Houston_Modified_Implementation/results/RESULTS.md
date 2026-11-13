# Preliminary Houston modified implementation

Settings selected on 2024; all models refit on 2023–2024 and evaluated on the same 114 published 2025 ZIP counts.

| model | count_RMSE | count_MAE | count_R2 | rate_RMSE_per_100k |
| --- | --- | --- | --- | --- |
| Population-rate baseline | 14.752 | 11.407 | -2.108 | 33.795 |
| Prior-year count baseline | 11.907 | 9.249 | -1.025 | 32.938 |
| RF socioeconomic | 11.197 | 9.525 | -0.79 | 31.689 |
| RF + temperature | 11.124 | 9.466 | -0.767 | 32.085 |
| RF + heat index | 10.724 | 9.369 | -0.642 | 30.499 |
| PyGRF + heat index | 10.507 | 9.177 | -0.577 | 29.933 |

Validation selected: RF socioeconomic. Its final count RMSE is 11.197. Do not select or retune using the final test ranking.

This adapts Parent Paper 2 PyGRF to Houston, with temperature and humidity/heat-index feature ablations and chronological model selection. It does not integrate UCformer or CLMU outputs, and is not a reproduction of the Buffalo results. Report the actual comparison, including negative results.

HRI reporting windows must be confirmed against the original dashboard before final scientific conclusions. Counts are screenshot-transcribed and unlisted counts are unknown, not zero. Evaluation is conditional on published counts, not all ZIPs. Observed seasonal weather makes this retrospective, not a preseason forecast. Shared stations, repeated ZIPs and only three years limit independent evidence. No causal or statistical-significance claim is supported.

## What the completed test shows

All six test R-squared values are negative: none beats the hindsight constant prediction equal to the 2025 test mean in squared error. That test mean is not a deployable preseason baseline. Every model overpredicts the 2025 total. PyGRF plus heat-index features has the lowest test RMSE, but 2024 validation selected socioeconomic RF. These are mixed preliminary results, not consistent improvement. Confirm outcome reporting periods before attributing the lower 2025 counts to weather.
