#!/usr/bin/env python3
"""Deterministic float64 checks of weighted recovery and the eight-point trade-off.

New controlled checks, not a replay of the forecasting study. Uses the
matched-dimension check's actual reconstruction-score implementation.
"""
import json
import math
from itertools import product
from pathlib import Path
import numpy as np
from check_matched_dimension import build_scores, reference_geometry, risk


def main():
    rng = np.random.default_rng(20261124)
    d = 4
    counts = (2, 4)
    max_recovery_error = 0.
    max_bound_ratio = 0.
    max_spectral_error = 0.
    min_full_rank_s = float('inf')
    gains = []
    cases = 0
    for ranks in ((4,)*6, (2,)*6, (0,2,1,3,1,2)):
        for trial in range(100):
            bases = [np.linalg.qr(rng.normal(size=(d,d)))[0][:,:r] for r in ranks]
            P = np.concatenate([p.T for p in bases])
            assert np.linalg.matrix_rank(P)==d
            mu = rng.normal(size=(6,d))
            z = rng.normal(size=d)
            phi = np.concatenate([np.exp(x-x.max())/np.exp(x-x.max()).sum()
                        for x in [rng.uniform(-8,0,size=q) for q in counts]])
            if ranks==(4,)*6 and trial%2==0:
                phi = np.array([1.,1e-44,1.,1e-46,1e-40,1e-42])
                # Within-partition normalization (rounded in float64).
                phi[:2] /= phi[:2].sum(); phi[2:] /= phi[2:].sum()
            def mat(p):
                return np.concatenate([p[j]*Pj.T for j,Pj in enumerate(bases)])
            def rhs(p,w):
                return w+np.concatenate([p[j]*Pj.T@mu[j] for j,Pj in enumerate(bases)])
            A=mat(phi)
            local=[Pj.T@(z-mu[j]) for j,Pj in enumerate(bases)]
            v=np.concatenate([phi[j]*w for j,w in enumerate(local)])
            M=max(np.linalg.norm(w) for w in local)
            s=np.linalg.svd(A,compute_uv=False)[-1]
            unweighted=np.linalg.svd(P,compute_uv=False)[-1]
            floor=phi.min()*unweighted
            assert s>=floor-1e-13
            gains.append(s/floor)
            if ranks==(4,)*6:
                min_full_rank_s=min(min_full_rank_s,s)
                max_spectral_error=max(max_spectral_error,abs(s*s-phi@phi))
                assert s>=math.sqrt(3/4)-1e-12
            sol=np.linalg.lstsq(A,rhs(phi,v),rcond=None)[0]
            max_recovery_error=max(max_recovery_error,float(np.linalg.norm(sol-z)))
            dp=rng.normal(size=6); dp*=.05*s/np.linalg.norm(dp)
            dv=rng.normal(size=len(v)); dv*=.03*s/np.linalg.norm(dv)
            phat=phi+dp; vhat=v+dv
            shat=np.linalg.svd(mat(phat),compute_uv=False)[-1]
            assert shat>=s-np.linalg.norm(dp)-1e-13>0
            zhat=np.linalg.lstsq(mat(phat),rhs(phat,vhat),rcond=None)[0]
            bound=(np.linalg.norm(dv)+M*np.linalg.norm(dp))/shat
            ratio=float(np.linalg.norm(zhat-z)/bound)
            assert ratio<=1+1e-11
            max_bound_ratio=max(max_bound_ratio,ratio)
            cases+=1
    assert max_recovery_error<1e-9
    assert max_spectral_error<1e-12
    # The known eight-point support: use actual reconstruction scores to
    # derive the constant within-regime gate, as in the analytical proof.
    tradeoff_cases=0; max_risk_error=0.; max_inverse_error=0.; max_product_error=0.
    examples=[]
    for ratio,lam,tau,method in product((1.1,2.,4.),(.01,.1,1.),(1e-6,.05),('A','R','K')):
        u,v,refs=reference_geometry(ratio)
        z=np.concatenate(refs); S=np.repeat([0,1],4)
        scores,_,_=build_scores(z,refs,method,lam)
        good=scores[np.arange(8),S]; bad=scores[np.arange(8),1-S]
        assert np.ptp(good)<1e-12 and np.ptp(bad)<1e-12
        logratio=math.log1p(tau-bad[0])-math.log1p(tau-good[0]); assert logratio<0
        for rho0 in (.8,.3,.1,.01,.001):
            omega=math.log(rho0)/logratio
            rho=math.exp(omega*logratio)
            p=np.where(np.arange(2)[None,:]==S[:,None],1/(1+rho),rho/(1+rho))
            enc=np.column_stack((p,p*z))
            c=max(np.linalg.norm(z[i]-z[j])/np.linalg.norm(enc[i]-enc[j]) for i in range(8) for j in range(i))
            lower=(1+rho)/rho
            assert c>=lower*(1-1e-12)
            if rho<=.01:
                assert abs(c/lower-1)<1e-12
                max_inverse_error=max(max_inverse_error,abs(c/lower-1))
            m=np.where(S==0,z[:,0],z[:,1])
            loss=risk(enc,m)
            exp_loss=u*u*v*v*rho*rho/(u*u+rho*rho*v*v)
            assert abs(loss-exp_loss)<1e-12
            max_risk_error=max(max_risk_error,abs(loss-exp_loss))
            if rho<=.01:
                expected=u*u*v*v*(1+rho)**2/(u*u+rho*rho*v*v)
                max_product_error=max(max_product_error,abs(loss*c*c-expected))
                assert abs(loss*c*c-expected)<1e-9
            if ratio==2 and lam==.1 and tau==1e-6 and method=='K':
                examples.append(dict(rho=rho,omega=omega,smallest_weighted_singular_value=rho/(1+rho),inverse_lipschitz=c,aligned_excess_mse=loss,product=loss*c*c))
            tradeoff_cases+=1
    x=json.loads((Path(__file__).resolve().parents[1]/'reproduction/score_enrichment/evidence/numerical_diagnostics.json').read_text())
    rows=[r for r in x['basis_diagnostics'] if r['cap']==12]
    assert len(rows)==256
    num=sum(r['all_local_models_full_rank'] for r in rows)
    assert num==253
    print(json.dumps(dict(status='PASS',scope='Float64 controlled tests; no interval arithmetic or forecasting replay',numpy=np.__version__,weighted_recovery_cases=cases,max_clean_recovery_error=max_recovery_error,max_perturbed_error_to_bound_ratio=max_bound_ratio,max_full_rank_gram_error=max_spectral_error,min_full_rank_sigma=min_full_rank_s,median_weighted_to_floor_sigma_ratio=float(np.median(gains)),max_weighted_to_floor_sigma_ratio=float(max(gains)),eight_point_tradeoff_cases=tradeoff_cases,max_matched_risk_error=max_risk_error,max_small_rho_inverse_relative_error=max_inverse_error,max_product_absolute_error=max_product_error,recorded_cap12_all_local_full_rank=num,recorded_cap12_collections=len(rows),illustrative_KAHM_results=examples),indent=2))

if __name__=='__main__':
    main()
