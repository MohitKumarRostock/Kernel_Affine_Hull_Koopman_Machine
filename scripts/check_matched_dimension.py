#!/usr/bin/env python3
"""Deterministic check of Proposition 4 using full reference matrices.

This is a new finite-law population calculation, not a replay of the forecasting
experiment. Uses float64 arithmetic and an independently assembled least-squares
problem, including the redundant membership and intercept columns.
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np


def score(z, a):
    nr = np.linalg.norm(z) * np.linalg.norm(a)
    cos = np.clip(z @ a / nr, -1.0, 1.0) if nr > 0 else 0.0
    return math.hypot(-math.expm1(-np.linalg.norm(z-a)), math.acos(cos)/math.pi)/math.sqrt(2)


def reference_geometry(ratio):
    v = math.sqrt(2/(1+ratio**2))
    u = ratio*v
    q = np.array([[s*u, t*v] for s,t in product((-1,1), repeat=2)])
    return u, v, (q, q[:, ::-1].copy())


def build_scores(z, refs, method, lam):
    scores = np.empty((len(z),2))
    min_guard = 1.0
    max_formula_error = 0.0
    for j,q in enumerate(refs):
        # PCA is fit from the actual four rows, not prescribed by the solver.
        mu = q.mean(axis=0)
        evals, evecs = np.linalg.eigh(np.cov(q, rowvar=False, ddof=1))
        p = evecs[:, -1]
        assert np.allclose(p*p, np.eye(2)[j], atol=1e-13, rtol=0)
        xref, x = q @ p, z @ p
        var = np.var(xref, ddof=1)
        gram = np.exp(-(xref[:,None]-xref[None,:])**2/(2*var))
        gamma = (2*(1+math.exp(-1.5))+lam)/(2*(1-math.exp(-1.5))+lam)
        u = np.max(np.abs(xref))
        for i,zi in enumerate(z):
            k = np.exp(-(xref-x[i])**2/(2*var))
            if method == 'A':
                a = mu+p*(p @ (zi-mu))
                expected = p*x[i]
            else:
                w = np.linalg.solve(gram+lam*np.eye(len(q)),k) if method == 'K' else k
                assert w.sum()>0
                guard = abs(w.sum())/np.sum(abs(w))
                min_guard = min(min_guard, float(guard))
                assert guard > 1e-10
                a = q.T @ w/w.sum()
                expected = p*u*math.tanh(3*x[i]/(4*u))*(gamma if method=='K' else 1)
            max_formula_error = max(max_formula_error, float(np.max(np.abs(a-expected))))
            scores[i,j] = score(zi,a)
    return scores, min_guard, max_formula_error


def risk(design, y):
    # Direct solve in the supplied coordinate space, with intercept.
    x = np.column_stack((np.ones(len(y)),design))
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    assert rank == 4, rank
    return float(np.mean((y-x@coef)**2))


def main():
    max_risk_error = max_score_variation = max_recon_error = 0.0
    min_margin = float('inf')
    min_guard = 1.0
    cases = optimum_checks = 0
    illustrative = []
    for ratio, lam, tau, method in product((1.1,2.,4.),(.01,.1,1.),(1e-6,.05),('A','R','K')):
        u,v,refs = reference_geometry(ratio)
        z = np.concatenate(refs)
        s = np.repeat([0,1],4)
        idx = np.arange(len(z))
        scores, guard, err = build_scores(z,refs,method,lam)
        min_guard = min(min_guard,guard)
        max_recon_error = max(max_recon_error,err)
        good,bad = scores[idx,s],scores[idx,1-s]
        max_score_variation = max(max_score_variation,float(np.ptp(good)),float(np.ptp(bad)))
        margin = float(np.min(bad-good))
        min_margin = min(min_margin,margin)
        assert margin > 0
        logratio = math.log1p(tau-float(bad[0]))-math.log1p(tau-float(good[0]))
        optimum = math.log(.3)/logratio
        for omega in (.5,4.,16.,64.,optimum):
            a = omega*np.log(1-scores+tau)
            p = np.exp(a-a.max(axis=1,keepdims=True))
            p /= p.sum(axis=1,keepdims=True)
            rho = math.exp(omega*logratio)
            local = np.column_stack((p,p*z))
            baseline = np.column_stack((p,z))
            # Direct recovery from the actual positive gates.
            assert np.allclose(local[:,2:]/p,z,atol=1e-13,rtol=1e-13)
            for alpha,beta in ((1.,0.),(0.,1.),(1.,.3),(1.,1.),(-.8,.4)):
                y = np.where(s==0,alpha*z[:,0]+beta*z[:,1],beta*z[:,0]+alpha*z[:,1])
                actual_l,actual_b = risk(local,y),risk(baseline,y)
                exp_l = u*u*v*v*(beta-rho*alpha)**2/(u*u+rho*rho*v*v)
                exp_b = u*u*v*v*(alpha-beta)**2/(u*u+v*v)
                max_risk_error = max(max_risk_error,abs(actual_l-exp_l),abs(actual_b-exp_b))
                if (alpha,beta)==(1.,0.): assert actual_l < actual_b
                if (alpha,beta)==(0.,1.): assert actual_l > actual_b
                if (alpha,beta)==(1.,.3) and omega==optimum:
                    assert actual_l < 1e-25
                    optimum_checks += 1
                if ratio==2. and lam==.1 and tau==1e-6 and omega==16. and (alpha,beta) in ((1.,0.),(0.,1.)):
                    illustrative.append(dict(method=method,alpha=alpha,beta=beta,rho=rho,local_excess_mse=actual_l,baseline_excess_mse=actual_b))
                cases+=1
    assert max_risk_error<1e-12 and max_recon_error<1e-12 and max_score_variation<1e-12
    result = dict(status='PASS', scope='Float64 finite-law population check; no forecasting replay, no interval arithmetic',
                  numpy=np.__version__, task_condition_cases=cases, optimal_contrast_checks=optimum_checks,
                  max_risk_absolute_discrepancy=max_risk_error,
                  max_reconstruction_coordinate_absolute_discrepancy=max_recon_error,
                  max_within_regime_score_variation=max_score_variation,minimum_score_margin=min_margin,
                  minimum_cancellation_guard=min_guard, illustrative_results=illustrative)
    print(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    main()
