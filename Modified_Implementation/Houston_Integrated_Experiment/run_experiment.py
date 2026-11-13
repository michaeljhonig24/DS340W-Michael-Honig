"""Houston HRI: PyGRF, two-stage XGBoost, and validation-selected blending."""
from pathlib import Path
import json, sys, subprocess, platform
import numpy as np
import pandas as pd
import sklearn, xgboost
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_baseline as base
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'integrated_results'
FEATURES=base.SOC+base.TEMP+base.HUM
DEPTHS=[2,3]
BLEND_WEIGHTS=[.25,.5,.75] # weight on two-stage, remaining weight on PyGRF

def params(depth):
    return dict(n_estimators=80,max_depth=depth,learning_rate=.05,min_child_weight=5,
                subsample=1.,colsample_bytree=1.,reg_lambda=5.,random_state=42,n_jobs=1,tree_method='hist')
def reg(depth):
    return XGBRegressor(objective='count:poisson',**params(depth))
def fit_two(train,threshold,depth):
    y=train.hri_visit_count.to_numpy(); high=y>=threshold
    assert min(high.sum(),(~high).sum())>=10, 'Insufficient observations in a burden group'
    gate=XGBClassifier(objective='binary:logistic',eval_metric='logloss',**params(depth))
    gate.fit(train[FEATURES],high.astype(int))
    low_model=reg(depth).fit(train.loc[~high,FEATURES],y[~high])
    high_model=reg(depth).fit(train.loc[high,FEATURES],y[high])
    return gate,low_model,high_model

def predict_two(models,data):
    gate,lo,hi=models
    p=gate.predict_proba(data[FEATURES])[:,1]
    low=lo.predict(data[FEATURES]); high=hi.predict(data[FEATURES])
    # Soft combination avoids discontinuous routing; no true evaluation labels used.
    return (1-p)*low+p*high,p,low,high

def scores(data,p):
    return base.metrics(data.hri_visit_count.to_numpy(),p,data.population.to_numpy())

def class_scores(y,p):
    hard=p>=.5
    return {'n':len(y),'positives':int(y.sum()),'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'average_precision':float(average_precision_score(y,p)) if y.sum() else None,
            'brier':float(brier_score_loss(y,p)),'precision_at_05':float(precision_score(y,hard,zero_division=0)),
            'recall_at_05':float(recall_score(y,hard,zero_division=0)),'f1_at_05':float(f1_score(y,hard,zero_division=0)),
            'confusion_matrix_tn_fp_fn_tp':confusion_matrix(y,hard,labels=[0,1]).ravel().tolist()}

def main():
    OUT.mkdir(exist_ok=True)
    # Fresh baseline generation prevents mixing predictions from different workbooks/runs.
    subprocess.run([sys.executable,str(ROOT/'run_baseline.py')],check=True)
    d=pd.read_csv(ROOT/'results/modeling_table.csv',dtype={'zcta':str,'residential_zip':str})
    assert not d.duplicated(['year','zcta']).any()
    assert np.isfinite(d[FEATURES+['hri_visit_count','population']].to_numpy()).all()
    assert (d.population>0).all() and (d.hri_visit_count>=0).all()
    train=d[d.year==2023].reset_index(drop=True);val=d[d.year==2024].reset_index(drop=True)
    fit=d[d.year.isin([2023,2024])].reset_index(drop=True);test=d[d.year==2025].reset_index(drop=True)
    threshold=float(train.hri_visit_count.quantile(.75))
    assert threshold>0
    config=json.loads((ROOT/'results/locked_configuration.json').read_text())
    spatial_cfg=config['configs']['PyGRF + heat index']
    model=base.spatial(spatial_cfg['neighbor_fraction'],len(train))
    model.fit(train[FEATURES],train.hri_visit_count,train[['x_km','y_km']])
    spatial_val=np.array(model.predict(val[FEATURES],val[['x_km','y_km']],spatial_cfg['local_weight'])[0])
    candidates=[];best={}; val_preds={}
    for name in ['XGB single-stage','XGB two-stage']:
        options=[]
        for depth in DEPTHS:
            if name=='XGB single-stage':
                m=reg(depth).fit(train[FEATURES],train.hri_visit_count);p=m.predict(val[FEATURES])
            else:
                m=fit_two(train,threshold,depth);p=predict_two(m,val)[0]
            row={'model':name,'max_depth':depth,**scores(val,p)};options.append((row,p));candidates.append(row)
        row,p=min(options,key=lambda pair:pair[0]['count_RMSE'])
        best[name]=int(row['max_depth']);val_preds[name]=p
    blend_options=[]
    for weight in BLEND_WEIGHTS:
        p=weight*val_preds['XGB two-stage']+(1-weight)*spatial_val
        row={'model':'PyGRF + two-stage blend','two_stage_weight':weight,**scores(val,p)}
        candidates.append(row);blend_options.append((row,p))
    blend_row,blend_val=min(blend_options,key=lambda pair:pair[0]['count_RMSE'])
    weight=blend_row['two_stage_weight']
    # Compare selected settings across all model families using validation only.
    validation_final=[]
    pooled=train.hri_visit_count.sum()/train.population.sum()
    prior=train.set_index('zcta').hri_visit_count
    base_val={'Population-rate baseline':val.population.to_numpy()*pooled,
              'Prior-year count baseline':val.zcta.map(prior).fillna(pd.Series(val.population.to_numpy()*pooled)).to_numpy(),
              'PyGRF + heat index':spatial_val}
    for name,cols in base.GROUPS.items():
        m=base.rf(config['configs'][name]['min_samples_leaf']).fit(train[cols],train.hri_visit_count)
        base_val[name]=m.predict(val[cols])
    base_val.update(val_preds);base_val['PyGRF + two-stage blend']=blend_val
    for name,p in base_val.items():validation_final.append({'model':name,**scores(val,p)})
    selected=min(validation_final,key=lambda row:row['count_RMSE'])['model']
    locked={'threshold_count':threshold,'threshold_definition':'2023 75th percentile; high = count >= threshold; held fixed during refit',
            'depths':best,'two_stage_blend_weight':weight,'selected_family_on_2024':selected,
            'candidate_depths':DEPTHS,'candidate_blend_weights':BLEND_WEIGHTS,
            'test_history':'2025 baseline performance was already inspected before this extension; exploratory comparison, not untouched confirmatory test.'}
    (OUT/'locked_configuration.json').write_text(json.dumps(locked,indent=2))
    pd.DataFrame(candidates).to_csv(OUT/'validation_candidates.csv',index=False)
    pd.DataFrame(validation_final).to_csv(OUT/'validation_family_comparison.csv',index=False)
    print('Configuration locked using 2024. Final refit and 2025 scoring.',flush=True)
    single=reg(best['XGB single-stage']).fit(fit[FEATURES],fit.hri_visit_count)
    models=fit_two(fit,threshold,best['XGB two-stage'])
    two,prob,lo,hi=predict_two(models,test)
    original=pd.read_csv(ROOT/'results/test_2025_predictions.csv',dtype={'zcta':str})
    spatial_test=test[['zcta']].merge(original[original.model=='PyGRF + heat index'][['zcta','predicted_count']],on='zcta',validate='one_to_one').predicted_count.to_numpy()
    new={'XGB single-stage':single.predict(test[FEATURES]),'XGB two-stage':two,
         'PyGRF + two-stage blend':weight*two+(1-weight)*spatial_test}
    summary=pd.read_csv(ROOT/'results/test_2025_comparison.csv').drop(columns=['selected_on_validation'])
    summary=pd.concat([summary,pd.DataFrame([{'model':name,'n_test':len(test),**scores(test,p)} for name,p in new.items()])],ignore_index=True)
    summary['selected_on_validation']=summary.model==selected
    summary.to_csv(OUT/'test_2025_comparison.csv',index=False)
    extra=[]
    for name,p in new.items():
        assert np.isfinite(p).all() and (p>=0).all()
        rows=test[['year','zcta','hri_visit_count','population','hri_visits_per_100k','INTPTLAT','INTPTLONG']].copy()
        rows['model']=name;rows['predicted_count']=p;rows['predicted_rate_per_100k']=p/test.population*1e5;extra.append(rows)
    pd.concat([original,*extra],ignore_index=True).to_csv(OUT/'test_2025_predictions.csv',index=False)
    details=test[['year','zcta','hri_visit_count','population']].copy()
    details['higher_burden_threshold']=threshold;details['observed_higher_burden']=(test.hri_visit_count>=threshold).astype(int)
    details['predicted_higher_burden_probability']=prob;details['lower_group_predicted_count']=lo;details['higher_group_predicted_count']=hi
    details['two_stage_predicted_count']=two;details['pygrf_predicted_count']=spatial_test;details['blended_predicted_count']=new['PyGRF + two-stage blend']
    details.to_csv(OUT/'two_stage_details.csv',index=False)
    gate_val=predict_two(fit_two(train,threshold,best['XGB two-stage']),val)[1]
    cls={'validation':class_scores((val.hri_visit_count>=threshold).to_numpy(),gate_val),
         'test':class_scores((test.hri_visit_count>=threshold).to_numpy(),prob),
         'test_training_prevalence_baseline':class_scores((test.hri_visit_count>=threshold).to_numpy(),np.full(len(test),np.mean(fit.hri_visit_count>=threshold)))}
    (OUT/'classification_metrics.json').write_text(json.dumps(cls,indent=2))
    audit={'python':platform.python_version(),'xgboost':xgboost.__version__,'sklearn':sklearn.__version__,
           'rows_by_year':d.groupby('year').size().to_dict(),'zeros_by_year':d.assign(zero=d.hri_visit_count==0).groupby('year').zero.sum().to_dict(),
           'higher_burden_rows_by_year':d.assign(high=d.hri_visit_count>=threshold).groupby('year').high.sum().to_dict(),
           'source_audit':json.loads((ROOT/'results/audit.json').read_text()),'features':FEATURES,'seed':42,
           'interpretation':'Published ZIP-year HRI visits, not unique patients, danger, assistance demand or causal risk. Observed seasonal weather; retrospective.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    # Export reusable fitted XGBoost models; PyGRF remains reproducible via run_baseline.py.
    single.save_model(OUT/'single_stage.ubj')
    for name,m in zip(['gate','lower_count','higher_count'],models):m.save_model(OUT/(name+'.ubj'))
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(summary.model,summary.count_RMSE,color=['#607d8b']*6+['#3585a6','#f0a44b','#ad625c']);ax.invert_yaxis()
    for j,v in enumerate(summary.count_RMSE):ax.text(v+.08,j,f'{v:.2f}',va='center',fontsize=9)
    ax.set_xlim(0,summary.count_RMSE.max()*1.16);ax.set_xlabel('2025 count RMSE (lower is better)');ax.set_title('Houston HRI: exploratory comparison, 114 published ZIP counts')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    table=base.markdown_table(summary[['model','count_RMSE','count_MAE','count_R2','selected_on_validation']].round(3))
    (OUT/'RESULTS.md').write_text('# Houston integrated experiment\n\n'+table+'\n\nSelected using 2024: '+selected+'.\n\nHigh burden is >= '+str(threshold)+' published visits, not a clinical or assistance threshold. Probability estimates are uncalibrated.\n\n2025 has already been inspected in earlier baseline work; this extension is exploratory. No UCformer predictions are used. See README.md for design and limitations.\n')
    print(summary[['model','count_RMSE','count_R2','selected_on_validation']].to_string(index=False))
    print('Saved:',OUT)

if __name__=='__main__': main()
