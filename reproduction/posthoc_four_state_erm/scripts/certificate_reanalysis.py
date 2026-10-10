#!/usr/bin/env python3
"""Reanalysis of frozen KAHKM evaluation pairs. No simulations, no model training.

Four-state case needs only per-state counts and the p and dynamics design.
Coordinate case uses PRE-EXISTING evaluation coordinates and fixed labels.

The ERM comparator uses a computable Frank-Wolfe / first-order lower bound
on the *empirical optimum*, never a bare numerical minimization estimate.
In exact arithmetic this ensures optimization error cannot inflate a risk
lower certificate. Float rounding is addressed by a configurable safety margin.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np

SOURCE_COMMIT = '0bbb2bafa107892e32e221e882326890b14fe36e'
FIELDS = ('dataset_id','dynamics','p','n_a','n_b','n_d','n_e')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):
            h.update(buf)
    return h.hexdigest()

def positive(x): return max(0.0,float(x))

def calculate_certificates(M,C,s,f,t,q,kappa,delta,erm_lower=None):
    """Pointwise 1-delta confidence formulas from the revised manuscript.

    V & F are jointly calibrated internally with r=ln(2/delta)/M.
    `q` is half-squared distance to an independently fixed class prototype.
    `erm_lower` is a proven lower bound on the optimum empirical MSE.
    """
    assert M>=1 and C>=1 and 0<delta<1 and kappa>=0
    r=math.log(2.0/delta)/M
    V=positive(s-math.sqrt(2*r))
    F=min(1.0, f+r+math.sqrt(r*r+2*r*f))
    T=min(1.0, t+math.sqrt(2*r)+C/M)
    row={
        'M':int(M), 'C':int(C),'kappa':float(kappa),'delta_pointwise':float(delta),
        's_hat':float(s),'f_hat':float(f),'t_hat':float(t),
        'V_delta':V,'F_delta':F,'T_delta':T,
        'L_label':positive(math.sqrt(V)-kappa*math.sqrt(2*F)),
        'L_centered':positive(math.sqrt(V)-kappa*math.sqrt(T)),
    }
    if q is not None:
        Q=min(1.0, q+r+math.sqrt(r*r+2*r*q))
        row['q_hat_fixed_prototype']=float(q)
        row['Q_delta']=Q
        row['L_prototype']=positive(math.sqrt(V)-kappa*math.sqrt(2*Q))
    if erm_lower is not None:
        ep=(1+kappa)**2*math.sqrt(math.log(1/delta)/(2*M))
        row['erm_empirical_opt_MSE_lower']=float(erm_lower)
        row['erm_concentration_penalty_MSE']=ep
        row['L_ERM_generic']=math.sqrt(positive(erm_lower-ep))
    return row

def four_state_row(row,delta,kappa=math.sqrt(2)):
    """Exact four-state formulas using only observed counts (no solver needed).

    For the conflicting map, a feasible matrix with output map
    A=t*I+(1-t)*u*1^T, t=n_a/(n_a+n_b), fits empirical class means;
    ||A||_2 <= sqrt(2). Hence empirical minimum MSE equals s_hat.
    """
    n=[int(row[k]) for k in ('n_a','n_b','n_d','n_e')]
    if min(n)<0: raise ValueError('negative state count')
    M=sum(n)
    if M==0: raise ValueError('empty replicate')
    if 'M' in row and row['M'] not in ('',None) and int(row['M'])!=M:
        raise ValueError('M does not equal the sum of state counts')
    p=float(Fraction(row['p']))
    if not .5<=p<=1.0: raise ValueError('p outside [1/2,1]')
    dyn=row['dynamics'].strip().lower()
    if dyn in ('conflict','conflicting','conflicting_successors'):
        # Only a,b share input coordinates but differ in next coordinates.
        na,nb,nd,ne=n
        s=0.0 if na+nb==0 else 2*(2*p-1)**2*(na*nb)/((na+nb)*M)
        exact_population_opt=p-.5  # because source law is uniform over states
        dyn='conflict'
    elif dyn in ('identity','id'):
        s=0.0
        exact_population_opt=0.0
        dyn='identity'
    else:
        raise ValueError('unrecognized dynamics: '+str(row['dynamics']))
    f=(1-p)**2  # exact since every current point has correct-label coordinate p
    t=0.0 # every coordinate vector is constant within class
    q=0.0 # fixed prototypes h^1=(p,1-p), h^2=(1-p,p), independent of samples
    if abs(kappa-math.sqrt(2))>1e-12:
        raise ValueError('exact empirical optimum equality is established for kappa=sqrt(2) only')
    out=calculate_certificates(M,2,s,f,t,q,kappa,delta,erm_lower=s)
    out.update(dataset_id=row['dataset_id'], dynamics=dyn, p=row['p'],
               n_a=n[0],n_b=n[1],n_d=n[2],n_e=n[3],
               exact_population_min_RMSE=exact_population_opt,
               exact_empirical_min_RMSE=math.sqrt(s),
               analytical_within_class_dispersion_zero=True,
               L_class_exact=math.sqrt(out['V_delta']),
               numerical_origin='exact_reanalysis_of_specified_counts')
    if 'replicate' in row: out['replicate']=row['replicate']
    if 'case' in row: out['case']=row['case']
    return out

def project_spectral_ball(A,kappa):
    U,d,Vt=np.linalg.svd(A,full_matrices=False)
    return (U*np.minimum(d,kappa))@Vt

def certified_empirical_erm_from_moments(S,T,z2,kappa,iteration_limit=2000,
                                         rtol=1e-8,numeric_margin=1e-10):
    """FISTA primal solver + global first-order convex lower bounds.

    Loss f(A)=mean||Z-XA||^2. For any A, convexity gives
      min_{||B||_2<=k} f(B) >= f(A)-<G,A>-k||G||_*.
    Best obtained global lower bound is conservative even if solver stalls.
    Uses float64, records first-order gap and subtracts a rounding margin.
    """
    S=np.asarray(S,dtype=np.float64);T=np.asarray(T,dtype=np.float64)
    if S.ndim!=2 or S.shape[0]!=S.shape[1] or T.shape!=S.shape:
        raise ValueError('S and T must be matching CxC matrices')
    z2=float(z2)
    emax=float(np.linalg.eigvalsh(S)[-1]); L=2*emax
    def objective(A): return positive(z2-2*np.sum(A*T)+np.sum(A*(S@A)))
    def gradient(A): return 2*(S@A-T)
    if L <= 1e-15:
        return {'lower_opt_MSE':z2,'primal_MSE':z2,'optimization_gap_MSE':0,
                'iterations':0,'converged':True,'numeric_margin_MSE':0}
    # Warm start: projected unconstrained LS fit; safe if rank deficient.
    A=project_spectral_ball(np.linalg.lstsq(S,T,rcond=None)[0],kappa)
    Y=A.copy(); best_primal=objective(A)
    lower=0.0; m=1.0; converged=False; used=0
    for it in range(1,iteration_limit+1):
        # One projected accelerated step.
        A1=project_spectral_ball(Y-gradient(Y)/L,kappa)
        f1=objective(A1); G=gradient(A1)
        dual=float(f1-np.sum(G*A1)-kappa*np.linalg.svd(G,compute_uv=False).sum())
        lower=max(lower,dual,0.0)
        best_primal=min(best_primal,f1)
        used=it
        if best_primal-lower <= max(1e-10,rtol*max(1.0,best_primal)):
            converged=True
            break
        m1=(1+math.sqrt(1+4*m*m))/2
        Y=A1+((m-1)/m1)*(A1-A)
        A,m=A1,m1
    safety=numeric_margin*(1+abs(lower))
    lower=positive(lower-safety)
    if best_primal+1e-8<lower: raise AssertionError('invalid lower-vs-upper optimizer bracket')
    return {'lower_opt_MSE':lower,'primal_MSE':best_primal,
            'optimization_gap_MSE':best_primal-lower,'iterations':used,
            'converged':converged,'numeric_margin_MSE':safety}

def certified_empirical_erm(X,Z,kappa,iteration_limit=2000,rtol=1e-8,numeric_margin=1e-10):
    M,C=X.shape
    return certified_empirical_erm_from_moments(X.T@X/M, X.T@Z/M,
                   float(np.sum(Z*Z)/M),kappa,iteration_limit,rtol,numeric_margin)

def coordinates_row(X,Z,classes,kappa,delta,prototypes=None,solver_iters=2000,solver_rtol=1e-8):
    M,C=X.shape
    if Z.shape!=(M,C) or classes.shape!=(M,): raise ValueError('coordinate shapes mismatch')
    if np.any(classes<0) or np.any(classes>=C): raise ValueError('classes must be in 0..C-1')
    for U in (X,Z):
        if not np.all(np.isfinite(U)): raise ValueError('nonfinite coordinate values')
        if np.any(U < -1e-8) or np.max(np.abs(U.sum(axis=1)-1))>1e-7:
            raise ValueError('coordinates must lie in the simplex')
    counts=np.bincount(classes,minlength=C).astype(int)
    # Avoid large 3-D arrays, use just norms, sums, and group totals.
    sums_X=np.zeros((C,C)); sums_Z=np.zeros((C,C))
    np.add.at(sums_X,classes,X);np.add.at(sums_Z,classes,Z)
    nonempty=counts>0
    sx=float(np.sum(X*X)/M-np.sum(sums_X[nonempty]**2/counts[nonempty,None])/M)
    sz=float(np.sum(Z*Z)/M-np.sum(sums_Z[nonempty]**2/counts[nonempty,None])/M)
    # Roundoff can make exactly zero dispersion ~1e-16 negative.
    t=positive(sx); s=positive(sz)
    f=float(np.mean((1-X[np.arange(M),classes])**2))
    q=None
    if prototypes is not None:
        if prototypes.shape!=(C,C):raise ValueError('fixed prototypes must be CxC')
        if np.max(np.abs(prototypes.sum(axis=1)-1))>1e-8 or np.any(prototypes<-1e-8):
            raise ValueError('fixed prototypes must lie in simplex')
        q=float(np.mean(np.sum((X-prototypes[classes])**2,axis=1))/2)
    optim=certified_empirical_erm(X,Z,kappa,iteration_limit=solver_iters,rtol=solver_rtol)
    results=calculate_certificates(M,C,s,f,t,q,kappa,delta,erm_lower=optim['lower_opt_MSE'])
    results.update({'erm_solver_'+key:val for key,val in optim.items()})
    results['numerical_origin']='postprocessing_preexisting_coordinate_arrays'
    return results

def write_run(outdir,results,mode,source_file,args,source_extra=None):
    outdir.mkdir(parents=True,exist_ok=True)
    if not results:raise ValueError('no result records')
    cols=list(dict.fromkeys(k for r in results for k in r))
    tab=outdir/'comparison.csv'
    with tab.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(results)
    source_status='manuscript_reference_not_external_repo_verified'
    if source_file.is_dir():
        source_members=[{'file':str(x.relative_to(source_file)),'sha256':sha256(x)}
                for x in sorted(source_file.rglob('*.npz')) if x.is_file()]
        source_hash=hashlib.sha256(json.dumps(source_members,sort_keys=True).encode()).hexdigest()
        source_extra=dict(source_extra or {},source_members=source_members)
    else:
        source_hash=sha256(source_file)
    manifest={
      'analysis_type':'post_hoc_no_new_simulations',
      'mode':mode,'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'historical_repository_url':'https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine',
      'historical_commit_from_manuscript':SOURCE_COMMIT,
      'historical_commit_verification_status':source_status,
      'source_data_path':str(source_file.resolve()),'source_data_sha256':source_hash,
      'script_sha256':sha256(Path(__file__)),'comparison_csv_sha256':sha256(tab),
      'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
      'command_parameters':vars(args),'rows':len(results),'source_extra':source_extra or {},
      'confidence_scope':'pointwise per certificate; do not select maximum at 95% without allocating delta'
    }
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    print(f'Wrote {len(results)} row(s) to {tab}\nManifest: {outdir/"manifest.json"}')

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest='mode',required=True)
    fs=sub.add_parser('four-state',help='existing four-state COUNT CSV only; no fitting or simulation')
    fs.add_argument('--counts',type=Path,required=True,help='CSV with '+', '.join(FIELDS))
    fs.add_argument('--out',type=Path,required=True)
    fs.add_argument('--delta',type=float,default=.05)
    xy=sub.add_parser('coordinates',help='existing evaluation pairs as numeric NPZ arrays')
    xy.add_argument('--npz',type=Path,required=True,help='arrays phi,z,classes; classes 0..C-1')
    xy.add_argument('--prototype-npz',type=Path,help='independently frozen array h [C,C]')
    xy.add_argument('--kappas',default='1.0',help='comma-separated, pre-specified norm budgets')
    xy.add_argument('--out',type=Path,required=True)
    xy.add_argument('--delta',type=float,default=.05)
    xy.add_argument('--iters',type=int,default=2000)
    xy.add_argument('--solver-rtol',type=float,default=1e-8)
    cap=sub.add_parser('capsules',help='compact sufficient-statistics NPZ capsules: no raw array transfer')
    cap.add_argument('--capsules',type=Path,required=True,help='directory of existing evaluation capsules (*.npz)')
    cap.add_argument('--kappas',required=True,help='comma-separated pre-specified norm budgets')
    cap.add_argument('--out',type=Path,required=True)
    cap.add_argument('--delta',type=float,default=.05)
    cap.add_argument('--iters',type=int,default=2000)
    cap.add_argument('--solver-rtol',type=float,default=1e-8)
    args=ap.parse_args()
    if not 0<args.delta<1:ap.error('--delta must be between 0 and 1')
    if args.mode=='four-state':
        with args.counts.open(newline='') as f:
            rdr=csv.DictReader(f)
            missing=set(FIELDS)-set(rdr.fieldnames or [])
            if missing:ap.error('missing count fields: '+', '.join(sorted(missing)))
            results=[four_state_row(row,args.delta) for row in rdr]
        write_run(args.out,results,args.mode,args.counts,args,{'prototypes':'known before sampling: h1=(p,1-p), h2=(1-p,p)'})
    elif args.mode=='capsules':
        kappas=[float(x) for x in args.kappas.split(',')]
        files=sorted(args.capsules.glob('*.npz'))
        if not files:ap.error('no .npz capsules in directory')
        results=[]
        for file in files:
            with np.load(file,allow_pickle=False) as data:
                needed=('S','T','z2','M','C','s_hat','t_hat','f_hat','dataset_id')
                for key in needed:
                    if key not in data: ap.error(f'capsule {file.name} missing {key}')
                S=np.asarray(data['S'],dtype=np.float64)
                T=np.asarray(data['T'],dtype=np.float64)
                z2=float(data['z2'])
                M=int(data['M']); C=int(data['C'])
                s=float(data['s_hat']); t=float(data['t_hat']); f=float(data['f_hat'])
                q=float(data['q_hat']) if 'q_hat' in data else None
                dataset_id=str(data['dataset_id'].item())
            if S.shape!=(C,C) or T.shape!=(C,C):ap.error(f'invalid matrix size in {file.name}')
            for k in kappas:
                opt=certified_empirical_erm_from_moments(S,T,z2,k,
                                iteration_limit=args.iters,rtol=args.solver_rtol)
                row=calculate_certificates(M,C,s,f,t,q,k,args.delta,opt['lower_opt_MSE'])
                row.update({'erm_solver_'+key:val for key,val in opt.items()})
                row.update(dataset_id=dataset_id,capsule_file=file.name,
                           numerical_origin='postprocessing_frozen_sufficient_statistics')
                results.append(row)
        write_run(args.out,results,args.mode,args.capsules,args,
            {'note':'Each capsule was created from already sampled evaluation coordinates',
             'kappa_scope':'pointwise baseline, multiple budgets require joint error allocation'})
    else:
        with np.load(args.npz,allow_pickle=False) as data:
            for k in ('phi','z','classes'):
                if k not in data:ap.error('missing array '+k)
            X=np.asarray(data['phi'],dtype=np.float64)
            Z=np.asarray(data['z'],dtype=np.float64)
            classes=np.asarray(data['classes'],dtype=np.int64)
        if X.ndim!=2: ap.error('phi must be an MxC matrix')
        prototypes=None
        if args.prototype_npz:
            with np.load(args.prototype_npz,allow_pickle=False) as hfile:
                prototypes=np.asarray(hfile['h'],dtype=np.float64)
        kappas=[float(x) for x in args.kappas.split(',')]
        results=[coordinates_row(X,Z,classes,k,args.delta,prototypes,
                                 args.iters,args.solver_rtol) for k in kappas]
        for row in results:row['dataset_id']=args.npz.stem
        write_run(args.out,results,args.mode,args.npz,args,
                  {'prototype_input_sha256':sha256(args.prototype_npz) if args.prototype_npz else None,
                   'kappa_scope':'prespecified pointwise per kappa; budget adjustments needed for joint ERM coverage'})
if __name__=='__main__':main()
