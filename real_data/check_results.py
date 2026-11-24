#!/usr/bin/env python3
"""Independent assessment arithmetic and roster checks; no model fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
p=np.load(ROOT/'predictions.npz'); r=pd.read_csv(ROOT/'results.csv'); y=pd.read_csv(ROOT/'yearly.csv'); v=pd.read_csv(ROOT/'validation_scores.csv'); d=pd.read_csv(ROOT/'diagnostics.csv'); m=json.loads((ROOT/'metadata.json').read_text())
assert m['data_sha256']==hashlib.sha256((ROOT/'co2.csv').read_bytes()).hexdigest()
assert list(r.id)==list(p['id'])==list(range(212)); assert p['predicted_change'].shape==(212,614,3)
# Independently construct eligible origins from dated rows and compare archived labels.
raw=pd.read_csv(ROOT/'co2.csv'); series=pd.Series(raw.co2.values,index=pd.to_datetime(raw.date.astype(str),format='%Y%m%d')); assert series.index.is_unique
H=(1,4,12); elig=[]; target=[]; origins=[]
for dt in series.index:
    offsets=list(range(-12,1))+[-52]+[-52+h for h in H]+list(H)
    vals=series.reindex([dt+pd.Timedelta(weeks=j) for j in offsets])
    if vals.isna().any(): continue
    last=dt+pd.Timedelta(weeks=12)
    if last<pd.Timestamp('1980-01-01'): split=0
    elif dt>=pd.Timestamp('1980-01-01') and last<pd.Timestamp('1990-01-01'): split=1
    elif dt>=pd.Timestamp('1990-01-01'): split=2
    else: continue
    elig.append(split); target.append([series.loc[dt+pd.Timedelta(weeks=h)]-series.loc[dt] for h in H]); origins.append(dt.strftime('%Y-%m-%d'))
elig=np.array(elig); target=np.array(target); origins=np.array(origins); assert [sum(elig==i) for i in range(3)]==[789,454,614]
assert np.array_equal(origins[elig==2],p['origin']); assert np.array_equal(target[elig==2],p['target_change']); assert np.allclose(target[elig==0].var(0),p['train_variance'],rtol=1e-14,atol=1e-14)
assert np.array_equal(series.loc[pd.to_datetime(p['origin'])].values,p['origin_level'])
e=(p['predicted_change']-p['target_change'][None,:,:])**2; mse=e.mean(1); rec=np.column_stack(((mse/p['train_variance']).mean(1),np.sqrt(mse))); cols=['smse','rmse1','rmse4','rmse12']; err=float(np.max(abs(rec-r[cols].values))); assert err<1e-12
maxyear=0.
for yr in y.year.unique():
    use=np.array([int(x[:4])==yr for x in p['origin']]); yy=y[y.year==yr].sort_values('id'); mm=e[:,use].mean(1); a=np.column_stack(((mm/p['train_variance']).mean(1),np.sqrt(mm))); maxyear=max(maxyear,float(np.max(abs(a-yy[cols].values)))); assert (yy.n==use.sum()).all()
assert maxyear<1e-12
for _,rr in r[r.selection=='validation'].iterrows():
    vs=v[(v.seed==rr.seed)&(v.method==rr.method)&(v.omega==rr.omega)&(v.encoding==rr.encoding)]; assert len(vs)==(6 if rr.method=='hgb' else 9); best=vs.loc[vs.val_smse.idxmin()]; assert abs(best.val_smse-rr.val_smse)<1e-12
    if rr.method=='hgb': assert rr.setting==best.setting
    else: assert float(best.setting)==rr.penalty
assert np.allclose(r[r.selection=='matched_df'].df,8,rtol=0,atol=1e-8)
assert len(d)==23 and (d.stacked_rank==15).all() and (d.fallback==0).all()
w=r[(r.omega==16)&r.encoding.isin(['local','sep'])].pivot(index=['seed','method','selection'],columns='encoding',values='smse'); assert len(w)==40
report=dict(status='PASS',candidate_records=len(r),assessment_prediction_values=int(e.size),year_records=len(y),validation_selected_records=int((r.selection=='validation').sum()),matched_df_records=int((r.selection=='matched_df').sum()),maximum_assessment_summary_error=err,maximum_year_summary_error=maxyear,weighted_higher_validation=int((w.xs('validation',level='selection').local>w.xs('validation',level='selection').sep).sum()),weighted_higher_matched=int((w.xs('matched_df',level='selection').local>w.xs('matched_df',level='selection').sep).sum()))
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
