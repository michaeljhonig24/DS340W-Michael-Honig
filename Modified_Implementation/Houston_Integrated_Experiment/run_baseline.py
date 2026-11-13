"""Retrospective Houston HRI ablations. Run: python run_houston.py"""
from pathlib import Path
import argparse,json,hashlib,platform,warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from vendor.pygrf_core import PyGRFBuilder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SEED=42
SOC=['log_population','age_65_plus_pct','poverty_pct','renter_occupied_pct']
TEMP=['mean_daily_max_c','mean_daily_min_c','days_max_ge_35c','total_precipitation_mm']
HUM=['mean_relative_humidity_pct','mean_daily_max_heat_index_c','pct_valid_days_hi_ge_100f']
GROUPS={'RF socioeconomic':SOC,'RF + temperature':SOC+TEMP,'RF + heat index':SOC+TEMP+HUM}
def markdown_table(frame):
    headers=list(frame.columns)
    rows=[' | '.join(headers),' | '.join(['---']*len(headers))]
    rows.extend(' | '.join(str(v) for v in row) for row in frame.itertuples(index=False,name=None))
    return '\n'.join('| '+r+' |' for r in rows)
def metrics(y,p,pop):
    return {'count_RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'count_MAE':float(mean_absolute_error(y,p)),'count_R2':float(r2_score(y,p)),
            'rate_RMSE_per_100k':float(np.sqrt(mean_squared_error(np.asarray(y)/pop*1e5,np.asarray(p)/pop*1e5))),
            'observed_total':float(np.sum(y)),'predicted_total':float(np.sum(p))}
def rf(leaf):
    return RandomForestRegressor(n_estimators=60,max_depth=6,min_samples_leaf=leaf,max_features=1.0,random_state=SEED,n_jobs=1)
def spatial(frac,n):
    return PyGRFBuilder(band_width=min(n-1,max(5,round(frac*n))),n_estimators=60,max_depth=6,min_samples_leaf=4,
        max_features=1.0,kernel='adaptive',train_weighted=True,predict_weighted=True,resampled=True,
        random_state=SEED,n_jobs=1,bootstrap=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workbook',type=Path,default=ROOT/'source.xlsx');args=parser.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    data=pd.read_excel(args.workbook,sheet_name='Combined Data')
    weather=pd.read_excel(args.workbook,sheet_name='Seasonal Weather')
    loc=pd.read_excel(args.workbook,sheet_name='ZIP Locations')
    for frame,col in [(data,'residential_zip'),(data,'zcta'),(weather,'zcta'),(loc,'GEOID')]:frame[col]=frame[col].astype(str).str.zfill(5)
    assert not data.duplicated(['year','residential_zip']).any()
    assert not weather.duplicated(['year','zcta']).any()
    assert (data.population>0).all()
    # Verify the existing weather join before using it.
    compare=data[['year','zcta']+TEMP+HUM].merge(weather[['year','zcta']+TEMP+HUM],on=['year','zcta'],validate='many_to_one',suffixes=('_combined','_weather'))
    for c in TEMP+HUM:assert np.allclose(compare[c+'_combined'],compare[c+'_weather'])
    data=data.merge(loc[['GEOID','INTPTLAT','INTPTLONG']],left_on='zcta',right_on='GEOID',validate='many_to_one')
    data['log_population']=np.log(data.population)
    data['hri_visits_per_100k']=data.hri_visit_count/data.population*1e5
    # Fixed local equirectangular approximation, kilometres; not raw angular distance.
    data['x_km']=6371*np.cos(np.deg2rad(29.76))*np.deg2rad(data.INTPTLONG+95.37)
    data['y_km']=6371*np.deg2rad(data.INTPTLAT-29.76)
    assert data[SOC+TEMP+HUM+['x_km','y_km']].notna().all().all()
    assert np.isfinite(data[SOC+TEMP+HUM+['x_km','y_km']].to_numpy()).all()
    data['split']=data.year.map({2023:'training',2024:'validation',2025:'test'})
    data.sort_values(['year','residential_zip']).to_csv(out/'modeling_table.csv',index=False)
    # Preserve unavailable outcomes in the complete weather-key panel; never zero-fill.
    panel=weather[['year','zcta']].merge(data[['year','zcta','hri_visit_count']],on=['year','zcta'],how='left',validate='one_to_one')
    panel['outcome_status']=np.where(panel.hri_visit_count.notna(),'published','unlisted_or_suppressed_unknown')
    panel.to_csv(out/'outcome_availability.csv',index=False)
    train=data[data.year==2023].reset_index(drop=True);val=data[data.year==2024].reset_index(drop=True)
    final_train=data[data.year.isin([2023,2024])].reset_index(drop=True);test=data[data.year==2025].reset_index(drop=True)
    coords=['x_km','y_km'];target='hri_visit_count';tuning=[];configs={}
    # Lock configuration choices using 2024 only; final 2025 outcomes are evaluated below.
    for name,cols in GROUPS.items():
        candidates=[]
        for leaf in [4,8]:
            model=rf(leaf).fit(train[cols],train[target]);pred=model.predict(val[cols])
            row={'model':name,'min_samples_leaf':leaf,**metrics(val[target],pred,val.population.to_numpy())}
            tuning.append(row);candidates.append(row)
        configs[name]={'min_samples_leaf':min(candidates,key=lambda r:r['count_RMSE'])['min_samples_leaf']}
    print('Global model validation finished.',flush=True)
    cols=SOC+TEMP+HUM;sp_candidates=[]
    for frac in [.35,.65]:
        m=spatial(frac,len(train));m.fit(train[cols],train[target],train[coords])
        _,global_p,local_p=m.predict(val[cols],val[coords],.5)
        for weight in [.25,.5,.75]:
            pred=(1-weight)*np.array(global_p)+weight*np.array(local_p)
            row={'model':'PyGRF + heat index','neighbor_fraction':frac,'local_weight':weight,**metrics(val[target],pred,val.population.to_numpy())}
            tuning.append(row);sp_candidates.append(row)
        print('Spatial validation finished:',frac,flush=True)
    best=min(sp_candidates,key=lambda r:r['count_RMSE'])
    configs['PyGRF + heat index']={k:best[k] for k in ['neighbor_fraction','local_weight']}
    pd.DataFrame(tuning).to_csv(out/'validation_comparison.csv',index=False)
    selected=min(tuning,key=lambda r:r['count_RMSE'])['model']
    (out/'locked_configuration.json').write_text(json.dumps({'configs':configs,'selected_on_2024':selected,'feature_groups':GROUPS,'seed':SEED},indent=2))
    predictions={}
    predictions['Population-rate baseline']=test.population.to_numpy()*final_train[target].sum()/final_train.population.sum()
    prior=val.set_index('zcta')[target]
    predictions['Prior-year count baseline']=test.zcta.map(prior).fillna(pd.Series(predictions['Population-rate baseline'],index=test.index)).to_numpy()
    for name,cols in GROUPS.items():
        m=rf(configs[name]['min_samples_leaf']).fit(final_train[cols],final_train[target])
        predictions[name]=m.predict(test[cols])
    name='PyGRF + heat index';c=configs[name];cols=SOC+TEMP+HUM
    m=spatial(c['neighbor_fraction'],len(final_train));m.fit(final_train[cols],final_train[target],final_train[coords])
    predictions[name]=np.array(m.predict(test[cols],test[coords],c['local_weight'])[0])
    results=[];pred_rows=[]
    for name,pred in predictions.items():
        assert np.isfinite(pred).all() and (pred>=0).all()
        results.append({'model':name,'n_test':len(test),'selected_on_validation':name==selected,**metrics(test[target],pred,test.population.to_numpy())})
        p=test[['year','zcta',target,'population','hri_visits_per_100k','INTPTLAT','INTPTLONG']].copy()
        p['model']=name;p['predicted_count']=pred;p['predicted_rate_per_100k']=pred/test.population.to_numpy()*1e5
        pred_rows.append(p)
    result=pd.DataFrame(results);result.to_csv(out/'test_2025_comparison.csv',index=False)
    pd.concat(pred_rows).to_csv(out/'test_2025_predictions.csv',index=False)
    audit={'rows_by_year':data.groupby('year').size().to_dict(),'unlisted_outcomes':int(panel.hri_visit_count.isna().sum()),
           'unique_zctas':int(data.zcta.nunique()),'stations':int(data.humidity_station_id.nunique()),
           'all_test_zctas_seen_previously':bool(set(test.zcta)<=set(final_train.zcta)),
           'prior_year_fallback_rows':int(test.zcta.map(prior).isna().sum()),
           'workbook_sha256':hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
           'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,
           'income_excluded_reason':'ACS vintage dollars are not inflation-adjusted',
           'hri_reporting_window':'Workbook assigns April-October; original dashboard dates not independently verified here.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5.5));colors=['#8493a1']*2+['#28678e']*3+['#b56a24']
    ax.barh(result.model,result.count_RMSE,color=colors);ax.invert_yaxis()
    for i,v in enumerate(result.count_RMSE):ax.text(v+.12,i,f'{v:.2f}',va='center')
    ax.set_xlim(0,result.count_RMSE.max()*1.17);ax.set_xlabel('2025 count RMSE (visits per ZIP; lower is better)')
    ax.set_title('Houston HRI | Held-out 2025 comparison',loc='left',fontweight='bold',pad=18)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.text(.02,.018,'Preliminary retrospective analysis • 114 published ZIP counts • settings selected on 2024, refit on 2023–2024',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    r=result[result.model==selected].iloc[0]
    (out/'RESULTS.md').write_text('# Preliminary Houston modified implementation\n\n'
      +'Settings selected on 2024; all models refit on 2023–2024 and evaluated on the same 114 published 2025 ZIP counts.\n\n'
      +markdown_table(result[['model','count_RMSE','count_MAE','count_R2','rate_RMSE_per_100k']].round(3))
      +'\n\nValidation selected: '+selected+'. Its final count RMSE is '+f'{r.count_RMSE:.3f}'+'. Do not select or retune using the final test ranking.\n\n'
      +'This adapts Parent Paper 2 PyGRF to Houston, with temperature and humidity/heat-index feature ablations and chronological model selection. It does not integrate UCformer or CLMU outputs, and is not a reproduction of the Buffalo results. Report the actual comparison, including negative results.\n\n'
      +'HRI reporting windows must be confirmed against the original dashboard before final scientific conclusions. Counts are screenshot-transcribed and unlisted counts are unknown, not zero. Evaluation is conditional on published counts, not all ZIPs. Observed seasonal weather makes this retrospective, not a preseason forecast. Shared stations, repeated ZIPs and only three years limit independent evidence. No causal or statistical-significance claim is supported.\n')
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
