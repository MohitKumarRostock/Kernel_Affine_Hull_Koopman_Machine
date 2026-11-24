#!/usr/bin/env python3
"""Chronological CO2 study; no earlier simulation results used."""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[v]='1'
import sys,csv,json,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.optimize import brentq
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'ablation'))
from run_ablation import references,fit_models,get_features,gates,encodings,PEN
H=np.array([1,4,12]); SEEDS=list(range(2026112406,2026112411)); MAIN=SEEDS[0]
def write_csv(name,rows,out):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def dataset():
    df=pd.read_csv(ROOT/'co2.csv'); date=pd.to_datetime(df.date.astype(str),format='%Y%m%d'); y=df.co2.to_numpy(float)
    assert np.all(np.diff(date.to_numpy()).astype('timedelta64[D]').astype(int)==7)
    xx=[]; yy=[]; tt=[]; base=[]; split=[]; prev=[]
    for t in range(52,len(y)-12):
        recent=y[t-12:t+1]; season=y[t-52+H]-y[t-52]; target=y[t+H]-y[t]
        if not np.all(np.isfinite(np.r_[recent,season,target])): continue
        d=date.iloc[t]; last=date.iloc[t+12]
        if last<pd.Timestamp('1980-01-01'): s=0
        elif d>=pd.Timestamp('1980-01-01') and last<pd.Timestamp('1990-01-01'): s=1
        elif d>=pd.Timestamp('1990-01-01'): s=2
        else: continue
        xx.append(np.r_[np.diff(recent),season]); yy.append(target); tt.append(d.strftime('%Y-%m-%d')); base.append(y[t]); split.append(s); prev.append(season)
    return np.array(xx),np.array(yy),np.array(tt),np.array(base),np.array(split),np.array(prev),len(y),int(np.isnan(y).sum())
def main(out):
    out.mkdir(exist_ok=True,parents=True)
    X,Y,dates,base,s,season,nslots,nmissing=dataset(); a=int((s==0).sum()); b=int((s<2).sum()); assert np.all(np.diff(s)>=0)
    Z=(X-X[:a].mean(0))/X[:a].std(0); var=Y[:a].var(0); assert np.min(var)>0
    rows=[]; valrows=[]; preds=[]; years=[]; diag=[]; ids=[]
    def metrics(idx,pred):
        mse=((Y[idx]-pred)**2).mean(0); return dict(smse=float(np.mean(mse/var)),rmse1=float(np.sqrt(mse[0])),rmse4=float(np.sqrt(mse[1])),rmse12=float(np.sqrt(mse[2])))
    def record(seed,method,omega,enc,selection,penalty,rank,width,df,p,validation,setting=''):
        assert np.isfinite(p).all()
        key=len(ids); ids.append(key); preds.append(p.copy()); r=dict(id=key,seed=seed,method=method,omega=omega,encoding=enc,selection=selection,penalty=penalty,rank=rank,width=width,df=df,setting=setting,val_smse=validation)
        rows.append(dict(**r,**metrics(slice(b,None),p)))
        for yr in sorted(set(dates[b:].astype('U4'))):
            j=dates[b:].astype('U4')==yr; mse=((Y[b:][j]-p[j])**2).mean(0); years.append(dict(id=key,year=yr,n=int(j.sum()),smse=float(np.mean(mse/var)),rmse1=float(np.sqrt(mse[0])),rmse4=float(np.sqrt(mse[1])),rmse12=float(np.sqrt(mse[2]))))
    def ridge(F,seed,method,omega,enc):
        mu=F[:a].mean(0); sd=F[:a].std(0); sd[sd<1e-8]=1.; FF=(F-mu)/sd; FF-=FF[:a].mean(0)
        U,S,V=np.linalg.svd(FF[:a],full_matrices=False); lam=S*S/a; rank=int(np.sum(lam>max(lam[0]*1e-12,1e-15))); ym=Y[:a].mean(0); pr=U.T@(Y[:a]-ym); fval=FF[a:b]@V.T; ftest=FF[b:]@V.T
        def predict(pen,t): return ym+(t*(S/(S*S+a*pen)))@pr
        vals=[]
        for pen in PEN:
            v=metrics(slice(a,b),predict(pen,fval))['smse']; vals.append(v); valrows.append(dict(seed=seed,method=method,omega=omega,encoding=enc,setting=f'{pen:.12g}',val_smse=v))
        settings=[('validation',float(PEN[np.argmin(vals)]))]
        if rank>8: settings.append(('matched_df',float(np.exp(brentq(lambda x:np.sum(lam/(lam+np.exp(x)))-8,-35,35,xtol=1e-12)))))
        for selection,pen in settings:
            edf=float(np.sum(lam/(lam+pen))); assert selection!='matched_df' or abs(edf-8)<1e-8
            record(seed,method,omega,enc,selection,pen,rank,F.shape[1],edf,predict(pen,ftest),metrics(slice(a,b),predict(pen,fval))['smse'])
    ridge(Z,0,'raw',0,'raw')
    for method,P in [('persistence',np.zeros_like(Y)),('seasonal_drift',season)]: record(0,method,0,method,'fixed','',0,0,'',P[b:],metrics(slice(a,b),P[a:b])['smse'])
    hp=[]
    for leaves in (7,15,31):
        for it in (100,300):
            ps=[]
            for j in range(3):
                mod=HistGradientBoostingRegressor(max_leaf_nodes=leaves,max_iter=it,learning_rate=.05,min_samples_leaf=20,early_stopping=False,random_state=MAIN).fit(Z[:a],Y[:a,j]); ps.append(mod.predict(Z[a:]))
            p=np.column_stack(ps); v=metrics(slice(a,b),p[:b-a])['smse']; setting=f'leaves={leaves};iter={it}'; hp.append((v,setting,p[b-a:])); valrows.append(dict(seed=0,method='hgb',omega=0,encoding='raw',setting=setting,val_smse=v))
    v,st,p=min(hp,key=lambda x:x[0]); record(0,'hgb',0,'raw','validation','',15,15,'',p,v,st)
    for seed in SEEDS:
        refs,parts,_=references(Z[:a],(2,4),np.random.default_rng(seed)); models=fit_models(refs,4); scores,pos,fb=get_features(Z,models); stack=np.concatenate([m[2].T for m in models]); sr=int(np.linalg.matrix_rank(stack,tol=1e-10))
        for method in ('K','R','A','center'):
            for omega in ((1,4,16,64) if seed==MAIN and method=='K' else (16,)):
                pp=gates(scores[method],parts,omega); A=np.concatenate([pp[b:,j,None,None]*m[2].T[None,:,:] for j,m in enumerate(models)],axis=1); sv=np.linalg.svd(A,compute_uv=False)[:,-1]
                diag.append(dict(seed=seed,method=method,omega=omega,stacked_rank=sr,local_ranks=';'.join(str(m[2].shape[1]) for m in models),fallback=fb,min_membership=float(pp[b:].min()),median_weighted_sigma=float(np.median(sv)),min_weighted_sigma=float(sv.min())))
                for enc,F in encodings(Z,pp,pos).items(): ridge(F,seed,method,omega,enc)
        print('finished seed',seed,flush=True)
    write_csv('results.csv',rows,out); write_csv('yearly.csv',years,out); write_csv('validation_scores.csv',valrows,out); write_csv('diagnostics.csv',diag,out)
    np.savez_compressed(out/'predictions.npz',id=np.array(ids),predicted_change=np.array(preds),origin=dates[b:],target_change=Y[b:],origin_level=base[b:],train_variance=var)
    metadata=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform(),data_sha256=hashlib.sha256((ROOT/'co2.csv').read_bytes()).hexdigest(),weekly_slots=nslots,missing=nmissing,splits=[dict(name=k,n=int((s==j).sum()),first_origin=str(dates[s==j][0]),last_origin=str(dates[s==j][-1])) for j,k in enumerate(['train','validation','test'])],train_target_change_variances=var.tolist(),candidate_records=len(rows),year_records=len(years),diagnostic_records=len(diag),validation_records=len(valrows),seeds=SEEDS)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n'); print(json.dumps(metadata,indent=2))
if __name__=='__main__': main(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT)
