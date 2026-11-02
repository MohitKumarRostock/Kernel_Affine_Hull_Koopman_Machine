#!/usr/bin/env python3
"""Exploratory development only: no calibration stream used."""
import sys, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from certified_forecasting.features import NestedSoftFeatures
from certified_forecasting.models import BlockBoundedRidge
from certified_forecasting.certificates import pair_witness_scores

def data(seed,n,jitter=.2,noise=.05):
 r=np.random.default_rng(seed)
 signs=r.choice([-1.,1.],size=(n,2))
 x=signs*(.8+r.uniform(-jitter,jitter,size=(n,2)))
 y=.8*np.prod(signs,axis=1)+r.uniform(-noise,noise,size=n)
 return x,y
out=[]
for seed in range(97001,97006):
 x,y=data(seed,2048)
 v,z=data(seed+100000,8192)
 bank=NestedSoftFeatures(cluster_counts=(2,4,8),kind='kahm',seed=seed,input_scaling='minmax',max_fit=2048,max_train_per_cluster=48,subspace_dim=2,omega=16).fit(x)
 f=bank.transform(x); h=bank.transform(v)
 rows=[]
 for p in range(1,4):
  k=bank.block_slices_[p-1].stop; blocks=bank.block_slices_[:p]
  model=BlockBoundedRidge(alpha=1e-5,kappa=2,bounds=(-.85,.85),block_slices=blocks,tol=1e-9).fit(f[:,:k],y)
  pred=model.predict(h[:,:k])[:,0]
  w=pair_witness_scores(h[:,None,:k],z[:,None,None],2,blocks)[:,0]
  rows.append({'prefix':p,'cost':k,'mse':float(np.mean((z-pred)**2)),'witness':float(w.mean()),'norm':model.norm_sum_,'gap':model.optimization_gap_})
 out.append({'seed':seed,'rows':rows})
 print(json.dumps(out[-1]),flush=True)
Path(__file__).with_name('pilot_results.json').write_text(json.dumps({'scope':'exploratory development-only; independent calibration not accessed','rows':out},indent=2)+'\n')
