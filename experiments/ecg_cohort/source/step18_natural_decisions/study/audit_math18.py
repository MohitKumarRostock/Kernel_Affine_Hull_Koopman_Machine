"""Independent arithmetic for Step18, without importing a producer or solver.

These float64 checks are a replay audit, not a formal interval proof. Distances
are formed coordinate-by-coordinate and dual divergence with dense incidence;
KL endpoints use scipy's independent Brent root solver.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy,xlog1py


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, atol=2e-11, rtol=2e-11):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape, f'shape mismatch {a.shape} vs {b.shape}')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite comparison')
    if a.dtype==np.bool_ or b.dtype==np.bool_:
        require(np.array_equal(a,b),'Boolean array differs')
        return 0.
    err=float(np.max(np.abs(a-b))) if a.size else 0.
    require(np.all(np.abs(a-b)<=atol+rtol*np.abs(b)),f'numeric discrepancy {err}')
    return err


def binary_kl(q,p):
    if p==0:return 0. if q==0 else math.inf
    if p==1:return 0. if q==1 else math.inf
    return float(xlogy(q,q)-xlogy(q,p)+xlogy(1-q,1-q)-xlog1py(1-q,-p))


def kl_lower(mean,n,bound,alpha):
    require(n>0 and 0<=mean<=bound and 0<alpha<1,'KL domain')
    q=mean/bound;c=-math.log(alpha)/n
    if q==0:return 0.
    if q==1:return bound*math.exp(-c)
    tiny=np.nextafter(0.,1.)
    if binary_kl(q,tiny)<=c:return 0.
    p=brentq(lambda p:binary_kl(q,p)-c,tiny,q,xtol=5e-324,
             rtol=8*np.finfo(float).eps,maxiter=1000)
    return float(bound*p)


def block_radii(F,lo,hi,slices):
    F,lo,hi=map(np.asarray,(F,lo,hi))
    require(F.shape==lo.shape==hi.shape,'feature interval shape')
    require(np.isfinite(F).all() and np.isfinite(lo).all() and np.isfinite(hi).all(),
            'nonfinite feature interval')
    require(np.all(lo<=F) and np.all(F<=hi),'nominal feature outside interval')
    rad=np.maximum(F-lo,hi-F)
    return np.column_stack([np.sqrt(np.sum(rad[:,a:b]**2,axis=1)) for a,b in slices])


def head_values(F,y,head,slices):
    F=np.asarray(F);y=np.asarray(y)
    w=np.asarray(head['coef']).reshape(-1);a=float(head['intercept'])
    require(w.shape==(F.shape[1],) and np.isfinite(w).all() and np.isfinite(a),'head')
    norms=np.array([np.linalg.norm(w[l:r]) for l,r in slices])
    k=float(head['kappa']);ridge=float(head.get('alpha',head.get('ridge',0.)))
    require(norms.sum()<=k+2e-12,'head norm exceeds frozen budget')
    residual=F@w+a-y
    objective=float(np.mean(residual**2)+ridge*np.sum(w*w))
    grad=2*(F-F.mean(0)).T@residual/len(y)+2*ridge*w
    gap=max(0.,float(grad@w+k*max(np.linalg.norm(grad[l:r]) for l,r in slices)))
    return dict(raw_prediction=F@w+a,prediction=np.clip(F@w+a,-1,1),
                norms=norms,objective=objective,gap=gap,norm_sum=float(norms.sum()),
                expected_intercept=float(y.mean()-F.mean(0)@w))


def prediction_interval(F,radii,head,slices):
    w=np.asarray(head['coef']);center=np.asarray(F)@w+head['intercept']
    norms=np.array([np.linalg.norm(w[l:r]) for l,r in slices])
    radius=np.asarray(radii)@norms
    return np.clip(center,-1,1),np.clip(center-radius,-1,1),np.clip(center+radius,-1,1)


def independent_partition(F,b,method):
    """Reconstruct exact index membership; all squared distance ties use index."""
    F=np.asarray(F);n=len(F)
    require(n%b==0,'this protocol uses complete partitions')
    if method=='consecutive':return np.arange(n).reshape(-1,b)
    require(method in ('nearest','greedy_nearest','feature_nearest'),'unknown partition')
    unused=set(range(n));groups=[]
    while unused:
        seed=min(unused);unused.remove(seed)
        # Python sorted with tuple key supplies an independently coded stable tie rule.
        remaining=np.array(sorted(unused),dtype=int)
        difference=F[remaining]-F[seed]
        # The frozen rule uses this float64 reduction. A different summation
        # order can reorder one-ulp near ties; only the membership replay uses
        # this common kernel. Dual arithmetic below remains separately coded.
        sq=np.einsum('ij,ij->i',difference,difference)
        order=np.lexsort((remaining,sq))
        chosen=remaining[order[:b-1]].tolist()
        groups.append([seed]+chosen)
        unused.difference_update(chosen)
    return np.asarray(groups,dtype=int)


def dual_values(F,Y,indices,radii,slices,kappa,evidence):
    """Re-evaluate a saved signed dual with conservative interval distances."""
    idx=np.asarray(indices);n=len(Y);b=idx.shape[1]
    require(idx.ndim==2 and np.issubdtype(idx.dtype,np.integer),'partition dtype')
    require(np.array_equal(np.sort(idx.ravel()),np.arange(n)),'partition not disjoint/full')
    x=np.asarray(F)[idx];y=np.asarray(Y)[idx];r=np.asarray(radii)[idx]
    K=len(idx);ii,jj=np.triu_indices(b,1)
    a=np.asarray(evidence['signed_dual']);v=np.asarray(evidence['primal'])
    require(a.shape==(K,len(ii)) and v.shape==(K,b),'dual/primal dimensions')
    require(np.isfinite(a).all() and np.isfinite(v).all(),'dual/primal finite')
    distance=np.zeros((K,b,b))
    for s,(lo,hi) in enumerate(slices):
        sq=np.zeros_like(distance)
        for q in range(lo,hi):
            diff=x[:,:,q,None]-x[:,None,:,q]
            sq+=diff*diff
        distance=np.maximum(distance,np.sqrt(sq)+r[:,:,s,None]+r[:,None,:,s])
    edge=float(kappa)*distance[:,ii,jj]
    incidence=np.zeros((b,len(ii)))
    incidence[ii,np.arange(len(ii))]=1.;incidence[jj,np.arange(len(ii))]=-1.
    divergence=a@incidence.T
    linear=np.einsum('ij,ij->i',a,y[:,ii]-y[:,jj])
    penalty=np.einsum('ij,ij->i',np.abs(a),edge)
    quadratic=.25*np.einsum('ij,ij->i',divergence,divergence)
    raw=linear-penalty-quadratic
    guard=1e-11*(1+np.abs(linear)+np.abs(penalty)+np.abs(quadratic))
    lower=np.maximum(0.,raw-guard)
    if 'nominal_sse_lower' in evidence:
        lower=np.minimum(np.asarray(evidence['nominal_sse_lower']),lower)
    sse=np.sum((y-v)**2,axis=1);upper=sse+1e-11*(1+np.abs(sse))
    violation=np.maximum(0.,np.max(np.abs(v[:,ii]-v[:,jj])-edge,axis=1))
    require(np.max(violation)<=2e-10,'relaxed primal feasibility')
    require(np.all(lower<=upper+2e-10),'weak duality')
    require(np.min(v)>=-1-2e-12 and np.max(v)<=1+2e-12,'primal range')
    return dict(sse_lower=lower,sse_upper=upper,dual_raw=raw,dual_guard=guard,
                gap=(upper-lower)/(b-1),feasibility_residual=violation,
                edge_penalty=penalty,linear=linear,quadratic=quadratic)


def baseline_predict(model,X):
    """Independent exported-baseline replay, including manual numeric tree walk."""
    X=np.asarray(X,dtype=float);kind=model['kind']
    if kind=='mean':p=np.repeat(model['constant'],len(X))
    elif kind=='persistence':p=X[:,-1]/model['target_scale_mV']
    elif kind=='hgb':
        p=np.repeat(model['baseline'],len(X)).astype(float)
        for tree in model['trees']:
            nodes=tree.nodes;at=np.zeros(len(X),dtype=np.int64)
            active=np.arange(len(X))
            while len(active):
                row=nodes[at[active]];leaf=row['is_leaf'].astype(bool)
                done=active[leaf];p[done]+=row['value'][leaf]
                active=active[~leaf];row=row[~leaf]
                if len(active):
                    require(not np.any(row['is_categorical']),'unexpected categorical tree')
                    left=X[active,row['feature_idx']]<=row['num_threshold']
                    at[active]=np.where(left,row['left'],row['right'])
    else:
        sc=model['scaler'];Z=(X-sc['offset'])/sc['divisor']
        Z[:,sc['constant_mask']]=0
        if kind=='quadratic':
            # Preserve the published upper-triangle feature order explicitly.
            terms=[Z[:,i]*Z[:,j] for i in range(Z.shape[1]) for j in range(i,Z.shape[1])]
            Z=np.column_stack([Z]+terms)
        elif kind in ('compact_rbf','nystrom'):
            sq=np.zeros((len(Z),len(model['centers'])))
            for j in range(Z.shape[1]):
                sq+=(Z[:,j,None]-model['centers'][None,:,j])**2
            logits=-model['gamma']*sq
            if kind=='compact_rbf':
                logits-=logits.max(1)[:,None]
                weights=np.exp(logits);Z=weights/weights.sum(1)[:,None]
            else:Z=np.exp(logits)
        if kind=='mlp':
            H=np.tanh(Z@model['hidden_weights']+model['hidden_bias'])
            p=(H@model['output_weights']+model['output_bias']).reshape(-1)
        else:p=Z@model['coef']+model['intercept']
    return np.clip(p,-1,1)


def numeric_bytes(model):
    """Independent recursive census of actually exported numeric parameters."""
    if isinstance(model,np.ndarray):return int(model.nbytes)
    if isinstance(model,(bool,np.bool_)):return 1
    if isinstance(model,(float,int,np.number)):return 8
    if isinstance(model,dict):return sum(numeric_bytes(x) for x in model.values())
    if isinstance(model,(tuple,list)):return sum(numeric_bytes(x) for x in model)
    if hasattr(model,'nodes'):
        return sum(numeric_bytes(getattr(model,k)) for k in
                   ('nodes','raw_left_cat_bitsets','binned_left_cat_bitsets'))
    if model is None or isinstance(model,str):return 0
    raise AssertionError('Uninventoried model type '+str(type(model)))
