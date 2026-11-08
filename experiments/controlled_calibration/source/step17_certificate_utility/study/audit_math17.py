"""Independent arithmetic for Step 17; imports no producer or certificate module.

Saved signed-edge duals are re-evaluated directly, and KL endpoints use a
separate root solver. Floating arithmetic and guards are not interval proofs.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy, xlog1py


def require(test, message):
    if not test:
        raise AssertionError(message)


def near(actual, expected, atol=2e-11, rtol=2e-11):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f'shape mismatch: {a.shape} versus {b.shape}')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite comparison')
    err = float(np.max(np.abs(a-b))) if a.size else 0.
    require(np.all(np.abs(a-b) <= atol+rtol*np.abs(b)), f'numeric discrepancy {err}')
    return err


def kl(q, p):
    q, p = float(q), float(p)
    require(0 <= q <= 1 and 0 <= p <= 1, 'invalid Bernoulli parameters')
    if p == 0:
        return 0. if q == 0 else math.inf
    if p == 1:
        return 0. if q == 1 else math.inf
    return float(xlogy(q, q)-xlogy(q, p)
                 +xlogy(1-q, 1-q)-xlog1py(1-q, -p))


def kl_endpoint(mean, count, bound, alpha, direction):
    require(type(count) is int and count > 0, 'invalid independent sample count')
    require(np.isfinite(mean) and 0 <= mean <= bound and bound > 0,
            'invalid bounded mean')
    require(0 < alpha < 1 and direction in ('lower', 'upper'), 'invalid calibration')
    q, c = mean/bound, -math.log(alpha)/count
    tiny = np.nextafter(0., 1.)
    almost = np.nextafter(1., 0.)
    if direction == 'lower':
        if q == 0: return 0.
        if q == 1: return bound*math.exp(-c)
        if kl(q, tiny) <= c: return 0.
        p = brentq(lambda p: kl(q, p)-c, tiny, q, xtol=5e-324,
                   rtol=8*np.finfo(float).eps, maxiter=1000)
    else:
        if q == 1: return bound
        if q == 0: return bound*(-math.expm1(-c))
        if kl(q, almost) <= c: return bound
        p = brentq(lambda p: kl(q, p)-c, q, almost, xtol=5e-324,
                   rtol=8*np.finfo(float).eps, maxiter=1000)
    return float(bound*p)


def block_distances(features, slices):
    x = np.asarray(features, dtype=float)
    require(x.ndim == 3 and np.isfinite(x).all(), 'invalid feature block array')
    covered = [j for lo,hi in slices for j in range(lo,hi)]
    require(covered == list(range(x.shape[2])), 'feature blocks do not partition coordinates')
    d = np.zeros(x.shape[:2]+(x.shape[1],))
    for lo,hi in slices:
        # Accumulate one coordinate at a time, independently of producer norm calls.
        sq = np.zeros_like(d)
        for j in range(lo,hi):
            diff = x[:,:,j,None]-x[:,None,:,j]
            sq += diff*diff
        d = np.maximum(d, np.sqrt(sq))
    return d


def audit_duals(features, targets, slices, kappa, record):
    x, y = np.asarray(features), np.asarray(targets)
    require(x.ndim == 3 and y.shape == x.shape[:2], 'dual sample shape')
    require(np.isfinite(y).all(), 'nonfinite targets')
    count,b = y.shape
    require(record['block_size'] == b, 'record block size differs')
    ii,jj = np.triu_indices(b, 1)
    a = np.asarray(record['signed_dual'])
    v = np.asarray(record['primal'])
    require(a.shape == (count,len(ii)) and v.shape == y.shape, 'invalid dual/primal dimensions')
    require(np.isfinite(a).all() and np.isfinite(v).all(), 'nonfinite optimization evidence')
    edge_bounds = kappa*block_distances(x,slices)[:,ii,jj]
    divergence = np.zeros_like(y, dtype=float)
    # Dense incidence multiplication independently rechecks the sparse solver.
    incidence = np.zeros((b,len(ii)))
    incidence[ii,np.arange(len(ii))] = 1.
    incidence[jj,np.arange(len(ii))] = -1.
    divergence[:] = a @ incidence.T
    linear = np.einsum('ij,ij->i', a, y[:,ii]-y[:,jj])
    penalty = np.einsum('ij,ij->i', np.abs(a), edge_bounds)
    quadratic = .25*np.einsum('ij,ij->i', divergence, divergence)
    raw = linear-penalty-quadratic
    guard = 1e-11*(1+np.abs(linear)+np.abs(penalty)+np.abs(quadratic))
    lower = np.maximum(0.,raw-guard)
    primal_sse = np.einsum('ij,ij->i', y-v,y-v)
    upper = primal_sse+1e-11*(1+np.abs(primal_sse))
    violation = np.maximum(0.,np.max(np.abs(v[:,ii]-v[:,jj])-edge_bounds,axis=1))
    maximum = 0.
    for key, expected in (('dual_raw',raw),('dual_guard',guard),
                           ('sse_lower',lower),('sse_upper',upper),
                           ('lower',lower/(b-1)),('upper',upper/(b-1)),
                           ('gap',(upper-lower)/(b-1)),
                           ('feasibility_residual',violation)):
        maximum=max(maximum,near(record[key],expected,2e-10,2e-11))
    require(np.max(violation) <= 2e-10, 'primal is materially infeasible')
    require(np.all(lower <= upper+2e-10), 'weak duality fails')
    require(np.min(v) >= np.min(y)-2e-12 and np.max(v) <= np.max(y)+2e-12,
            'primal target-range condition fails')
    return {'lower_sse':lower,'upper_sse':upper,'max_replay_error':maximum,
            'max_feasibility_residual':float(np.max(violation)),
            'max_gap':float(np.max((upper-lower)/(b-1))),'programs':count}


def score_endpoints(targets, predictions, duals, sizes, alpha, bounds):
    """Reconstruct one fixed class's endpoints from independently checked SSEs."""
    y,p = np.asarray(targets),np.asarray(predictions)
    require(y.shape == p.shape and y.ndim == 1, 'scalar loss shape')
    lo,hi = bounds
    require(np.isfinite(y).all() and np.isfinite(p).all(), 'nonfinite loss data')
    require(np.all((lo <= y)&(y <= hi)) and np.all((lo <= p)&(p <= hi)),
            'target/prediction bound failed')
    B=(hi-lo)**2
    out={}
    for n in sizes:
        losses=(y[:n]-p[:n])**2
        upper=kl_endpoint(float(losses.mean()),int(n),B,alpha,'upper')
        blocks={}
        pooled={}
        for b,values in duals.items():
            count=n//b
            s=np.asarray(values[:count])
            require(len(s)==count and np.isfinite(s).all() and np.min(s)>=0,
                    'invalid block lower values')
            block_bound=B*(b*b//4)/(b*(b-1))
            mean=float(np.mean(s)/(b-1))
            blocks[b]=kl_endpoint(mean,int(count),block_bound,alpha,'lower')
            empirical_lower=float(s.sum()/n)
            pooled[b]=dict(empirical_lower=empirical_lower,
                lower=kl_endpoint(empirical_lower,int(n),B,alpha,'lower'))
        best_block=max(blocks,key=lambda b:(blocks[b],-b))
        best_pool=max(pooled,key=lambda b:(pooled[b]['empirical_lower'],-b))
        out[n]=dict(mean_loss=float(losses.mean()),upper=upper,blocks=blocks,
            pooled=pooled,block_best=blocks[best_block],block_best_b=best_block,
            pool_best=pooled[best_pool]['lower'],pool_best_b=best_pool)
    return out


def head_values(features, targets, model, slices, ridge, kappa=2.):
    """Independent clipped predictions, un-clipped ridge objective and FW gap."""
    x,y=np.asarray(features),np.asarray(targets).reshape(-1)
    w=np.asarray(model['coef']).reshape(-1)
    a=float(np.asarray(model['intercept']).reshape(-1)[0])
    require(w.shape==(x.shape[1],) and np.isfinite(w).all() and np.isfinite(a),
            'invalid head coefficients')
    norms=np.array([np.linalg.norm(w[lo:hi]) for lo,hi in slices])
    require(float(norms.sum()) <= kappa+2e-12, 'decoder norm budget violated')
    residual=x@w+a-y
    objective=float(np.mean(residual**2)+ridge*np.sum(w*w))
    gradient=2*(x-x.mean(0)).T@residual/len(y)+2*ridge*w
    gap=max(0.,float(gradient@w+kappa*max(np.linalg.norm(gradient[lo:hi]) for lo,hi in slices)))
    return dict(raw_prediction=x@w+a,objective=objective,gap=gap,
                norm_sum=float(norms.sum()),block_norms=norms,
                intercept_expected=float(y.mean()-x.mean(0)@w))
