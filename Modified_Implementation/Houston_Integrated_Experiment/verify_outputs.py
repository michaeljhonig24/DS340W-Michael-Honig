"""Independent checks of saved experiment outputs; no model refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'integrated_results'
d=pd.read_csv(R/'results/modeling_table.csv',dtype={'zcta':str})
p=pd.read_csv(O/'test_2025_predictions.csv',dtype={'zcta':str})
s=pd.read_csv(O/'test_2025_comparison.csv')
c=json.loads((O/'locked_configuration.json').read_text())
t=pd.read_csv(O/'two_stage_details.csv',dtype={'zcta':str})
assert c['threshold_count']==d.loc[d.year==2023,'hri_visit_count'].quantile(.75)
assert p.model.nunique()==9
expected=set(d.loc[d.year==2025,'zcta'])
for name,g in p.groupby('model'):
 assert len(g)==114 and set(g.zcta)==expected and not g.zcta.duplicated().any()
 assert (g.year==2025).all() and np.isfinite(g.predicted_count).all() and (g.predicted_count>=0).all()
 truth=d[d.year==2025].set_index('zcta').hri_visit_count
 assert np.array_equal(g.hri_visit_count.to_numpy(),g.zcta.map(truth).to_numpy())
 error=g.predicted_count-g.hri_visit_count
 row=s.set_index('model').loc[name]
 assert np.isclose(np.sqrt(np.mean(error**2)),row.count_RMSE)
 assert np.isclose(np.mean(np.abs(error)),row.count_MAE)
 assert np.isclose(g.predicted_count.sum(),row.predicted_total)
prob=t.predicted_higher_burden_probability
assert prob.between(0,1).all()
assert np.allclose(t.two_stage_predicted_count,(1-prob)*t.lower_group_predicted_count+prob*t.higher_group_predicted_count)
w=c['two_stage_blend_weight']
assert np.allclose(t.blended_predicted_count,w*t.two_stage_predicted_count+(1-w)*t.pygrf_predicted_count)
v=pd.read_csv(O/'validation_family_comparison.csv')
assert c['selected_family_on_2024']==v.loc[v.count_RMSE.idxmin(),'model']
assert np.array_equal(t.observed_higher_burden,(t.hri_visit_count>=c['threshold_count']).astype(int))
print('PASS: shared 114-row cohort, nine models, true-label alignment, metrics, probability mixture, spatial blend, training threshold, validation selection.')
