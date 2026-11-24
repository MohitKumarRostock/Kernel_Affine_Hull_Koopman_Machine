#!/usr/bin/env python3
"""New fixed-geometry ablation. No archived forecasting outputs used."""
import os
for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var]='1'
import csv, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import t
from sklearn.cluster import KMeans
import sklearn
ROOT=Path(__file__).resolve().parent
LAWS=['linear_ar','nonlinear_ar','van_der_pol','duffing']
METHODS=['K','R','A','center']
PEN=np.logspace(-6,2,9)
NREP=32
SEED=2026112404

def rng(law,rep,stream): return np.random.default_rng(np.random.SeedSequence([SEED,law,rep,stream]))

def episodes(law,n,g):
    if law<2:
        burn=256 if law==0 else 80
        u=np.zeros((n,burn+72))
        if law==1: u[:,:2]=g.uniform(-.8,.8,(n,2))
        noise=g.normal(0,.1,(n,burn+70)) if law==0 else g.uniform(-.1,.1,(n,burn+70))
        for i in range(2,u.shape[1]):
            x,y=u[:,i-1],u[:,i-2]
            u[:,i]=.65*x-.15*y+noise[:,i-2] if law==0 else np.tanh(1.25*x-.65*y+.25*np.sin(3*x)+noise[:,i-2])
        return u[:,burn:]
    a=g.uniform(0,1,(n,75)); state=4*a[:,:2]-2; phase=2*np.pi*a[:,2]; noise=.04*a[:,3:]-.02
    def rhs(s,x):
        q,v=x.T
        force=(1-q*q)*v-q if law==2 else -.2*v+q-q**3+.3*np.cos(1.2*s+phase)
        return np.column_stack((v,force))
    out=np.empty((n,72)); dt=.025
    for i in range(688):
        s=i*dt; k1=rhs(s,state); k2=rhs(s+dt/2,state+dt*k1/2); k3=rhs(s+dt/2,state+dt*k2/2); k4=rhs(s+dt,state+dt*k3)
        state+=dt*(k1+2*k2+2*k3+k4)/6
        if i+1>=404 and (i+1-404)%4==0: out[:,(i+1-404)//4]=.95*np.tanh(state[:,0])
    return out+noise

def windows(u,g,n=None):
    origins=np.arange(11,60,4)
    x=u[:,origins[:,None]+np.arange(-11,1)].reshape(-1,12)
    y=u[:,origins[:,None]+np.array([1,4,12])].reshape(-1,3)
    ids=np.arange(len(x)) if n is None else np.sort(g.choice(len(x),n,replace=False))
    return x[ids],y[ids]

def references(x,groups,g,features=None):
    refs=[]; partitions=[]; labels=[]
    for k in groups:
        km=KMeans(n_clusters=k,n_init=5,max_iter=150,random_state=int(g.integers(2**31-1))).fit(x if features is None else features)
        assert len(set(km.labels_))==k
        part=[]
        for j in range(k):
            ix=np.flatnonzero(km.labels_==j); sel=g.choice(ix,min(24,len(ix)),replace=False)
            part.append(len(refs)); refs.append(x[sel].astype(np.float32).astype(np.float64))
        partitions.append(part); labels.append(km.labels_)
    return refs,partitions,labels

def fit_models(refs,cap):
    models=[]
    for q in refs:
        n,d=q.shape; mu=q.mean(axis=0); c=q-mu
        if n<=1 or not np.any(c): p=np.empty((d,0)); eig=np.empty(0)
        else:
            eig,p=np.linalg.eigh(c.T@c/(n-1)); eig=eig[::-1]; p=p[:,::-1]
            r=min(cap,d,n-1,int(np.sum(eig>np.finfo(float).eps)))
            while r>1 and np.min(np.ptp(q@p[:,:r],axis=0))<1e-3: r-=1
            eig=eig[:r]; p=p[:,:r]
        r=p.shape[1]
        if r:
            ref=(q@p)/np.sqrt(eig*r)
            dist=np.sum((ref[:,None,:]-ref[None,:,:])**2,axis=2)
            G=np.exp(-.5*dist); nu=np.mean(q*q); lam=2.5*nu
            for _ in range(100):
                C=np.linalg.solve(G+lam*np.eye(n),np.eye(n)); new=2*nu+np.mean((q-G@C@q)**2)
                if abs(new-lam)<=1e-3: break
                lam=new
            models.append((q,mu,p,eig,ref,C))
        else: models.append((q,mu,p,eig,None,None))
    return models

def score(z,a):
    norm=np.linalg.norm(z,axis=1)*np.linalg.norm(a,axis=1)
    cos=np.divide(np.sum(z*a,axis=1),norm,out=np.zeros(len(z)),where=norm>0)
    return np.hypot(-np.expm1(-np.linalg.norm(z-a,axis=1)),np.arccos(np.clip(cos,-1,1))/np.pi)/np.sqrt(2)

def get_features(z,models):
    scores={m:[] for m in METHODS}; pos=[]; fallback=0
    for q,mu,p,eig,ref,C in models:
        r=p.shape[1]; u=(z-mu)@p; pos.append(u)
        a=z.copy() if r==z.shape[1] else mu+u@p.T
        scores['A'].append(np.where(np.linalg.norm(z,axis=1)>0,0.,1/(2*np.sqrt(2))) if r==z.shape[1] else score(z,a)); scores['center'].append(score(z,np.broadcast_to(mu,z.shape)))
        if not r:
            scores['R'].append(scores['A'][-1]); scores['K'].append(scores['A'][-1]); continue
        query=z@p/np.sqrt(eig*r)
        logk=-.5*np.sum((query[:,None,:]-ref[None,:,:])**2,axis=2)
        k=np.exp(logk-logk.max(axis=1,keepdims=True)); ar=(k@q)/k.sum(axis=1,keepdims=True)
        w=k@C.T; den=w.sum(axis=1); norm=np.sum(abs(w),axis=1)
        good=(norm>0)&(abs(den)>=1e-10*norm)&(den!=0)&np.isfinite(w).all(axis=1)
        ak=a.copy(); ak[good]=(w[good]@q)/den[good,None]
        good2=np.isfinite(ak).all(axis=1); ak[~good2]=a[~good2]; fallback+=int(np.sum(~good|~good2))
        scores['R'].append(score(z,ar)); scores['K'].append(score(z,ak))
    return {m:np.column_stack(v) for m,v in scores.items()},pos,fallback

def gates(scores,parts,omega):
    a=omega*np.log(1-scores+1e-6); p=np.empty_like(a)
    for js in parts:
        h=a[:,js]; w=np.exp(h-h.max(axis=1,keepdims=True)); p[:,js]=w/w.sum(axis=1,keepdims=True)
    return p

def encodings(z,p,pos):
    return dict(mem=p,local=np.column_stack([p]+[p[:,j,None]*u for j,u in enumerate(pos)]),
                sep=np.column_stack([p]+pos),rawmem=np.column_stack([p,z]),
                gatedraw=np.column_stack([p]+[p[:,j,None]*z for j in range(p.shape[1])]))

def fit_eval(f,ya,yb,yd,splits,variance,target_df,do_df=True):
    a,b=splits; x=f[:a]; mu=x.mean(0); sd=x.std(0); sd[sd<1e-8]=1
    x=(f-mu)/sd; X=x[:a]; yc=ya.mean(0); Y=ya-yc
    xm=X.mean(0); x-=xm; X=x[:a]
    U,s,V=np.linalg.svd(X,full_matrices=False); lam=s*s/a
    keep=lam>max(lam[0]*1e-12,1e-15); rank=int(keep.sum())
    proj=U.T@Y
    def predict(alpha,arr): return yc+(arr@V.T*(s/(s*s+a*alpha)))@proj
    def mse(y,p): return np.mean((y-p)**2/variance,axis=0)
    val=np.array([mse(yb,predict(alpha,x[a:b])).mean() for alpha in PEN]); j=int(val.argmin())
    def record(alpha,sel):
        v=mse(yd,predict(alpha,x[b:])); return dict(selection=sel,penalty=float(alpha),df=float(np.sum(lam/(lam+alpha))),rank=rank,width=f.shape[1],loss=float(v.mean()),h1=float(v[0]),h4=float(v[1]) if len(v)>1 else None,h12=float(v[2]) if len(v)>2 else None)
    out=[record(PEN[j],'validation')]
    if do_df and rank>target_df:
        alpha=math.exp(brentq(lambda t0: np.sum(lam/(lam+math.exp(t0)))-target_df,-35,35,xtol=1e-12))
        out.append(record(alpha,'matched_df'))
        assert abs(out[-1]['df']-target_df)<1e-8
    return out

def forecast(law,rep):
    xy=[windows(episodes(law,n,rng(law,rep,2*i)),rng(law,rep,2*i+1),size) for i,(n,size) in enumerate(((64,512),(32,256),(128,None)))]; (xa,ya),(xb,yb),(xd,yd)=xy
    mean=xa.mean(0); sd=xa.std(0); sd[sd<=1e-12]=1; z=(np.concatenate((xa,xb,xd))-mean)/sd
    var=np.maximum(ya.var(0),1e-12); refs,parts,_=references(z[:len(xa)],(2,4),rng(law,rep,7)); rows=[]; diag=[]
    def add(f,cap,m,enc):
        for r in fit_eval(f,ya,yb,yd,(512,768),var,8,enc!='mem'):
            rows.append(dict(law=LAWS[law],rep=rep,cap=cap,method=m,encoding=enc,**r))
    add(z,0,'raw','raw')
    for cap in (4,12):
        models=fit_models(refs,cap); scores,pos,fb=get_features(z,models)
        stack=np.concatenate([x[2].T for x in models]); rank=int(np.linalg.matrix_rank(stack,tol=1e-10)); diag.append(dict(law=LAWS[law],rep=rep,cap=cap,stacked_rank=rank,fallback=fb,local_ranks=[x[2].shape[1] for x in models]))
        for m in METHODS:
            p=gates(scores[m],parts,16)
            for enc,f in encodings(z,p,pos).items(): add(f,cap,m,enc)
    return rows,diag

def angular(n,g):
    s=g.integers(0,2,n); radius=g.uniform(1,2,n)/math.sqrt(7/6); angle=g.uniform(.12,.5,n)
    signs=g.choice([-1,1],(n,2)); V=signs[:,0]*radius*np.cos(angle); U=signs[:,1]*radius*np.sin(angle)
    z=np.column_stack((np.where(s==0,V,U),np.where(s==0,U,V)))
    return z,s,np.column_stack((V,U,V+.3*U))

def controlled(rep):
    x,s,_=angular(512,rng(4,rep,0)); w=x/np.linalg.norm(x,axis=1,keepdims=True); feat=np.column_stack((w[:,0]**2-w[:,1]**2,2*w[:,0]*w[:,1])); refs,parts,labels=references(x,(2,),rng(4,rep,1),feat)
    label=labels[0]; correct=np.array([int(np.mean(s[label==j])>=.5) for j in range(2)]); accuracy=np.mean(correct[label]==s)
    data=[angular(n,rng(4,rep,i+2)) for i,n in enumerate((512,256,8192))]; z=np.concatenate([d[0] for d in data]); ss=np.concatenate([d[1] for d in data]); targets=np.concatenate([d[2] for d in data]); noise=rng(4,rep,6).normal(0,.05,(768,1)); ya=targets[:512]+noise[:512]; yb=targets[512:768]+noise[512:]; yd=targets[768:]
    models=fit_models(refs,1); scores,pos,fb=get_features(z,models); rows=[]; diag=[]
    def add(f,m,enc,omega):
        for j,task in enumerate(('aligned','minor','partial')):
            for rec in fit_eval(f,ya[:,j:j+1],yb[:,j:j+1],yd[:,j:j+1],(512,768),np.ones(1),2.85,enc in ('local','sep','rawmem')):
                rows.append(dict(rep=rep,method=m,encoding=enc,omega=omega,task=task,**rec))
    add(z,'raw','raw',0)
    for m in METHODS:
        if len(set(correct))==2:
            sj=np.where(ss==correct[0],0,1); margins=scores[m][np.arange(len(z)),1-sj]-scores[m][np.arange(len(z)),sj]
            margin_pos=float(np.mean(margins[768:]>0)); med=float(np.median(margins[768:]))
        else: margin_pos=None; med=None
        for omega in (1,4,16,64):
            p=gates(scores[m],parts,omega); A=np.stack([p[:,j,None]*v[2][:,0] for j,v in enumerate(models)],axis=1); sv=np.linalg.svd(A[768:],compute_uv=False)[:,-1]
            diag.append(dict(rep=rep,method=m,omega=omega,cluster_accuracy=float(accuracy),positive_margin_fraction=margin_pos,median_margin=med,median_weighted_sigma=float(np.median(sv)),fallback=fb))
            for enc,f in encodings(z,p,pos).items(): add(f,m,enc,omega)
    return rows,diag

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def main():
    import concurrent.futures, multiprocessing
    start=time.time(); fr=[]; fd=[]; cr=[]; cd=[]
    jobs=[(l,r) for l in range(4) for r in range(NREP)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i,(rr,dd) in enumerate(pool.map(forecast,*zip(*jobs))):
            fr.extend(rr); fd.extend(dd)
            if (i+1)%32==0: print('forecast',i+1,flush=True)
        for i,(rr,dd) in enumerate(pool.map(controlled,range(NREP))):
            cr.extend(rr); cd.extend(dd)
            if (i+1)%8==0: print('controlled',i+1,flush=True)
    write_csv(ROOT/'forecast_losses.csv',fr); write_csv(ROOT/'forecast_diagnostics.csv',fd); write_csv(ROOT/'controlled_losses.csv',cr); write_csv(ROOT/'controlled_diagnostics.csv',cd)
    lookup={(r['law'],r['rep'],r['cap'],r['method'],r['encoding'],r['selection']):r['loss'] for r in fr}
    comp=[]; tq=t.ppf(1-.05/(2*12),NREP-1)
    for law in LAWS:
        for m in ('K','R','A'):
            a=np.array([lookup[law,r,4,m,'local','validation'] for r in range(NREP)]); b=np.array([lookup[law,r,4,m,'sep','validation'] for r in range(NREP)]); d=a-b; w=tq*d.std(ddof=1)/np.sqrt(NREP)
            comp.append(dict(law=law,method=m,local=float(a.mean()),sep=float(b.mean()),difference=float(d.mean()),lower=float(d.mean()-w),upper=float(d.mean()+w),wins=int(np.sum(d<0)),n=NREP))
    write_csv(ROOT/'primary_comparisons.csv',comp)
    env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform(),seed=SEED,repetitions=NREP,forecast_records=len(fr),controlled_records=len(cr),forecast_diagnostic_records=len(fd),controlled_diagnostic_records=len(cd),t_quantile=float(tq),runtime_seconds=time.time()-start)
    (ROOT/'run_metadata.json').write_text(json.dumps(env,indent=2)+'\n'); print(json.dumps(env,indent=2))
if __name__=='__main__': main()
