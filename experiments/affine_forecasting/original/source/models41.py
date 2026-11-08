"""Frozen same-information classical baseline panel; no data access."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
from fractions import Fraction as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor
SEED=3901
def baseline_configs():
    rows=[{'token':'persistence','kind':'persistence','parameters':{}},{'token':'training_mean','kind':'mean','parameters':{}}]
    for token,lam in [('1e6','1/1000000'),('1e4','1/10000'),('1e2','1/100'),('1','1'),('100','100')]:
        rows.append({'token':'ridge_'+token,'kind':'ridge','ridge_lambda':lam,'parameters':{'fit_intercept':True,'solver':'cholesky'}})
    for leaves,leaf,l2 in [(a,b,1) for a in (7,15,31) for b in (5,20)]+[(15,b,10) for b in (5,20)]:
        model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=leaves,min_samples_leaf=leaf,l2_regularization=l2,early_stopping=False,random_state=SEED)
        rows.append({'token':f'hgb_nodes{leaves}_leaf{leaf}_l2{l2}','kind':'hgb','parameters':model.get_params()})
    svrs=[(C,e,'scale') for C in (1,10,100) for e in (.01,.1)]+[(C,.05,g) for C in (1,10) for g in (.01,.1)]
    for i,(C,e,g) in enumerate(svrs):
        rows.append({'token':f'svr_{i:02d}','kind':'svr','parameters':SVR(C=C,epsilon=e,gamma=g,kernel='rbf',tol=1e-5,cache_size=256,max_iter=100000).get_params()})
    for i,(a,g) in enumerate((a,g) for a in (.01,1,100) for g in (.01,.1,1)):
        rows.append({'token':f'krr_{i:02d}','kind':'krr','parameters':KernelRidge(alpha=a,gamma=g,kernel='rbf').get_params(),'target_centering':True})
    for kind,cls,leaves in [('extra_trees',ExtraTreesRegressor,(1,5,20)),('random_forest',RandomForestRegressor,(1,5))]:
        for leaf in leaves:
            for feat in (1.,.5):
                model=cls(n_estimators=256,min_samples_leaf=leaf,max_features=feat,n_jobs=1,random_state=SEED)
                rows.append({'token':f'{kind}_leaf{leaf}_features{feat}','kind':kind,'parameters':model.get_params()})
    result=[{**r,'candidate_id':'baseline::'+r['token']} for r in rows]
    assert len(result)==44 and len({r['candidate_id'] for r in result})==44
    return result
def fit_baseline(cfg,X,y,normalization):
    kind=cfg['kind'];ys=list(map(F,y));yf=np.array([float(x) for x in ys]);normal={'target_min':normalization['target_min'],'span':normalization['span']}
    if kind=='persistence':return {'kind':kind,'normalization':normal,'rule':'last_observed_physical_own_target'}
    if kind=='mean':return {'kind':kind,'normalization':normal,'value':str(F(normal['target_min'])+F(normal['span'])*sum(ys,F(0))/len(ys))}
    scaler=StandardScaler().fit(X);sx=scaler.transform(X);params=dict(cfg['parameters']);offset=0.
    if kind=='ridge':params['alpha']=len(ys)*float(F(cfg['ridge_lambda']));model=Ridge(**params)
    else:
        cls={'hgb':HistGradientBoostingRegressor,'svr':SVR,'krr':KernelRidge,'extra_trees':ExtraTreesRegressor,'random_forest':RandomForestRegressor}[kind];model=cls(**params)
    if kind=='krr':offset=float(yf.mean())
    model.fit(sx,yf-offset)
    if kind=='svr' and model.fit_status_!=0:raise RuntimeError('Frozen SVR budget exhausted')
    return {'kind':kind,'model':model,'scaler':scaler,'offset':offset,'normalization':normal,'fit_n':len(ys)}
def predict_baseline(payload,X):
    kind=payload['kind']
    if kind=='persistence':values=np.asarray(X)[:,-1]
    elif kind=='mean':values=np.full(len(X),float(F(payload['value'])))
    else:
        normal=payload['normalization'];values=float(F(normal['target_min']))+float(F(normal['span']))*(payload['model'].predict(payload['scaler'].transform(X))+payload['offset'])
    if not np.isfinite(values).all():raise ValueError('Nonfinite baseline predictions')
    return np.asarray(values,dtype=np.float64)
