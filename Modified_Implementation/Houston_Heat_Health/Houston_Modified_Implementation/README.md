# Houston modified implementation: socioeconomic and weather features with PyGRF

This is an executed, preliminary adaptation of Parent Paper 2's geographic random forest to Houston heat-related illness (HRI) counts. It includes matched feature ablations, ordinary random forests, and the parent PyGRF model. It does not use CLMU or UCformer predictions and is not a claim to reproduce or improve the parent paper's Buffalo benchmark.

## First action

Open `results/comparison.png` and `results/test_2025_comparison.csv` to see the completed results. CSV files can be opened in Excel or imported to Google Sheets. Read `results/RESULTS.md` for interpretation.

Copy this complete folder to `Modified_Implementation/Houston_Heat_Health` inside your project repository. No original parent files need to be overwritten. Keep the `vendor` folder and its license with the implementation. Commit the source code, README and results; decide with your team whether the source workbook belongs in Git or in your project data storage.

## Reproduce

Use a separate environment to avoid disrupting the ongoing weather download:

```bash
conda create -n houston-health python=3.12 -y
conda activate houston-health
python -m pip install -r requirements.txt
python run_houston.py
```

Run commands from this extracted folder. The default input is the included, unchanged `source.xlsx`. To choose another workbook with the same schema, use `python run_houston.py --workbook /path/to/workbook.xlsx`. Results overwrite only this folder's `results` directory. Runtime depends on hardware; the spatial method fits one local forest per training row. The code uses one CPU thread per forest.

## What changed from the parent implementation

1. Replace the original Buffalo task with published Houston residential ZIP-year HRI counts and same-code ACS ZCTA proxies.
2. Add gridded seasonal temperature and station-derived humidity/heat-index exposure blocks. Evaluate each block incrementally with ordinary RF.
3. Apply the parent geographic RF to the combined predictors. Select neighborhood fraction and local/global blending on an earlier validation year, rather than using test-year performance or the parent Moran search.
4. Add outcome-availability accounting, population-normalized reporting, matched temporal evaluation, and reproducible exported predictions.

The PyGRFBuilder class is extracted unchanged from the locally available `Parent_Paper_2/PyGRF-master/PyGRF.py`. Its original source and GPLv3 license are retained under `vendor`. The class's optional dependency-heavy bandwidth-search functions are not imported. This is a parent-code adaptation, not a newly invented spatial algorithm. Spatial model resampling and distance weights retain the parent class behavior. OOB outputs are not used for selection or reported as generalization evidence.

## Experimental design

- Train on 121 published 2023 rows and select settings on 121 published 2024 rows.
- Lock configurations, refit on 242 rows from 2023–2024, and evaluate all six approaches on the same 114 published 2025 rows.
- This evaluates a later year in largely familiar locations. It is not a held-out-geography experiment.
- Target: visit count, not unique patients. Models predict unrounded nonnegative counts. Population-normalized rates are secondary reporting metrics.
- RF: 60 trees, maximum depth 6, all predictors considered per split, seed 42. Select minimum leaf size from 4 and 8 on validation count RMSE.
- PyGRF: 60 trees, depth 6, leaf size 4, adaptive neighborhoods, training/prediction distance weights, parent resampling enabled. Select neighborhood fraction from 0.35 and 0.65 and local weight from 0.25, 0.5 and 0.75 on validation count RMSE. Neighborhood fraction is converted to a number of training records; repeated years at a ZIP count as separate records. Local coordinates use a fixed equirectangular approximation in km around 29.76 N, -95.37 E.
- The population-rate baseline estimates a pooled training visit rate and applies it to each test population. The prior-year baseline uses each available 2024 ZIP count and falls back to the pooled rate when no published prior count exists; fallback counts appear in audit.json.
- Primary metrics: count RMSE, MAE and R-squared. Secondary metric: RMSE of visits per 100,000. Each ZIP has equal metric weight. These are not directly comparable to the parent paper's different dataset and target.
- Candidate settings were specified before the final test run. The final test ranking is not used to retune models. This is one chronological split and one seed, not repeated validation or evidence of statistical significance.

## Feature blocks

- Socioeconomic: log population, age 65+ percentage, poverty percentage, renter-occupied percentage.
- Temperature: mean daily maximum, mean daily minimum, number of days with maximum at least 35 C, and total precipitation.
- Humidity/heat index: mean relative humidity, mean daily maximum heat index, percentage of usable days with heat index at least 100 F. The latter uses valid-day denominators rather than treating missing days as cool days.

Income is excluded because the workbook keeps dollars in each ACS vintage's prices. ZIP identifiers, year, outcome-derived quantities and prior counts are not predictors for RF/PyGRF. Coordinates determine spatial weights only, not RF covariates. No income adjustment, missing-outcome imputation or target-based feature selection is performed.

## Required interpretation limits

- **Confirm original HRI reporting dates before final conclusions.** The workbook labels April–October weather but does not independently establish that every dashboard HRI count covers exactly that period. The execution is valid for the supplied table; substantive alignment is provisional.
- HRI data were transcribed from screenshots. Verify transcription, reporting coverage, and suppression rules against the source. The full 372-key weather panel preserves 16 absent outcome rows as unknown. They are not zero and are excluded from fitting/scoring. Consequently the results describe published-count ZIPs, not an unbiased full-population sample.
- ZIP/ZCTA matching is approximate. ACS five-year estimates overlap; margins of error are retained but not propagated into this experiment. Rates use ACS population proxies.
- Only five stations supply humidity/heat index. Nearby ZIPs share station observations and often gridded weather. Spatial samples are correlated, and three years do not supply strong evidence about interannual weather effects.
- Weather is observed over the outcome season. This is retrospective prediction/explanation, not a forecast made before the season.
- No causal claim, clinical recommendation, or claim of statistically significant improvement is warranted. Negative R-squared is possible and should be reported honestly.
- The Houston CLMU pilot and six-city UCformer reproduction remain separate. Do not say that this pipeline uses simulated urban exposure. Adding that requires validated exposure features for each modeled location and reporting period.

## Paper draft: Contribution subsection

We adapted the parent geographic random forest implementation to retrospective Houston ZIP-year heat-related illness counts. Our extension combines ACS demographic and housing characteristics with gridded temperature and station-derived humidity/heat-index features. We compare socioeconomic-only, temperature-enhanced, and humidity-enhanced random forests, then evaluate a spatially weighted local/global forest using the combined predictors. Model settings are selected on 2024 observations following 2023 training; final evaluation uses 2025 after refitting on earlier years. The contribution is the task-specific integration and controlled comparison, rather than a new random-forest algorithm. The current evaluation is conditional on published HRI counts and provisional reporting-period alignment.

Adapt this text to your team's actual scope; do not claim that the instructor has approved its novelty or that an improvement occurred without citing the results.

## Package contents

`run_houston.py`: complete experiment; `source.xlsx`: unchanged uploaded input; `requirements.txt`: versions used; `vendor`: parent implementation and license; `results/modeling_table.csv`: joined, audited rows; `results/outcome_availability.csv`: full outcome availability; `results/validation_comparison.csv`: every candidate setting; `results/locked_configuration.json`: selected settings; `results/test_2025_comparison.csv`: final comparison; `results/test_2025_predictions.csv`: per-model, per-ZIP predictions; `results/audit.json`: data checks and versions; `results/comparison.png`: presentation figure; `results/RESULTS.md`: numerical result summary.
