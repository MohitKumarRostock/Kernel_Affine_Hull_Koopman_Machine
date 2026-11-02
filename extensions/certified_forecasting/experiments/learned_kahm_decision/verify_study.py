#!/usr/bin/env python3
"""Independent arithmetic and portable replay audit for learned KAHM decisions.

Does not import the study runner or its certificate helpers. Reconstructs
training, validation and calibration streams and all features from frozen JSON
banks (no pickle, no refitting); independently recomputes risk scores, pair
scores, empirical-Bernstein endpoints, mass LPs and selection decisions.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from certified_forecasting.features import NestedSoftFeatures, _KAHMBlock


def read_json_gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array_digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def restore(v):
    if isinstance(v,list):return [restore(x) for x in v]
    if not isinstance(v,dict):return v
    if '__array__' in v:
        a=np.array(v['data'],dtype=v['dtype']).reshape(v['shape'])
        if a.dtype.kind not in 'ifub':raise AssertionError('Non-numeric serialized array')
        return a
    if '__slice__' in v:return slice(*v['__slice__'])
    if '__kahm_block__' in v:
        b=_KAHMBlock.__new__(_KAHMBlock);b.__dict__.update(restore(v['__kahm_block__']));return b
    return {k:restore(x) for k,x in v.items()}


def stream(seed,n,regime):
    rng=np.random.default_rng(seed)
    x=rng.choice(np.array([-1.,1.]),size=(n,2))*(.9+rng.uniform(-.1,.1,size=(n,2)))
    if regime=='repairable':source=x
    else:source=rng.choice(np.array([-1.,1.]),size=(n,2))*(.9+rng.uniform(-.1,.1,size=(n,2)))
    y=.8*np.prod(source,axis=1)+rng.uniform(-.05,.05,size=n)
    return x,y


def interval(a,bound,delta):
    mean=float(np.mean(a));var=float(np.var(a,ddof=1));n=len(a)
    radius=math.sqrt(2*var*math.log(2/delta)/n)+7*bound*math.log(2/delta)/(3*(n-1))
    return {'mean':mean,'variance':var,'radius':radius,'lower':max(0.,mean-radius),'upper':min(bound,mean+radius),
        'n_independent':n,'bound':bound,'delta_per_tail':delta}


def pair_scores(f,y,blocks,kappa):
    n=(len(y)//2)*2
    d=np.maximum.reduce([np.linalg.norm(f[:n:2,b]-f[1:n:2,b],axis=1) for b in blocks])
    return np.maximum(np.abs(y[:n:2]-y[1:n:2])-kappa*d,0.)**2/2


class Audit:
    def __init__(self):self.max_difference=0.;self.n_comparisons=0
    def close(self,a,b,tol=2e-11):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if a.shape!=b.shape:raise AssertionError(f'Shape mismatch {a.shape} {b.shape}')
        diff=float(np.max(np.abs(a-b),initial=0));self.max_difference=max(self.max_difference,diff);self.n_comparisons+=a.size
        if diff>tol:raise AssertionError(f'Numeric discrepancy {diff} > {tol}')
    def scalar_details(self,a,b):
        for k in ['mean','variance','radius','lower','upper','n_independent','bound','delta_per_tail']:self.close(a[k],b[k])


def minimum(costs,flags):
    passing=[c for c,f in zip(costs,flags) if f];return min(passing) if passing else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Audit output already exists')
    folder=args.results;c=json.loads((folder/'protocol.json').read_text());r=json.loads((folder/'results.json').read_text())
    f=json.loads((folder/'protocol_freeze.json').read_text());d=json.loads((folder/'development_freeze.json').read_text())
    for key,name in [('protocol_sha256','protocol.json'),('source_snapshot_sha256','source_snapshot.tar.gz'),('source_manifest_sha256','source_snapshot.json')]:assert digest(folder/name)==f[key]
    for key,name in [('frozen_models_sha256','frozen_models.json.gz'),('development_scores_sha256','development_scores.csv'),('protocol_freeze_sha256','protocol_freeze.json')]:assert digest(folder/name)==d[key]
    assert f['calibration_generated'] is False and d['calibration_generated'] is False
    started=json.loads((folder/'calibration_started.json').read_text())
    assert started['development_freeze_sha256']==digest(folder/'development_freeze.json')
    assert f['frozen_at_utc']<=d['frozen_at_utc']<started['started_at_utc']<r['completed_at_utc']
    for name,h in r['hashes'].items():
        if name.endswith('.npz'):continue
        assert digest(folder/name)==h,(name,'hash mismatch')
    sm=json.loads((folder/'source_snapshot.json').read_text())
    with tarfile.open(folder/'source_snapshot.tar.gz','r:gz') as tar:
        for file in sm['files']:
            assert hashlib.sha256(tar.extractfile(file['path']).read()).hexdigest()==file['sha256']
    frozen=read_json_gz(folder/'frozen_models.json.gz');details=read_json_gz(folder/'interval_details.json.gz')
    lookup={(z['record']['regime'],z['record']['repetition'],z['record']['n_calibration'],z['record']['prefix']):z for z in details}
    records={(z['regime'],z['repetition']):z for z in json.loads((folder/'calibration_records.json').read_text())}
    with open(folder/'decisions.csv',newline='') as g:decisions={(z['regime'],int(z['repetition']),int(z['n_calibration'])):z for z in csv.DictReader(g)}
    count=len(c['regimes'])*c['n_repetitions_per_regime']*len(c['calibration_sizes'])*len(c['retained_feature_costs'])
    assert count==c['n_class_size_records']==320
    du=c['upper_budget']/count;dl=c['unconditional_lower_budget']/count;eta=c['stratified_lower_budget']/(count*len(c['partition_names']))
    a=Audit();a.close(count*(du+dl+2*eta),.05)
    bound=(c['target_bounds'][1]-c['target_bounds'][0])**2
    n_candidates=0;n_selected=0;n_mass_lps=0;seeds=[];decision_summary={};recomputed_endpoints=[]
    max_gap=0.;max_norm=0.;n_full_loss_scores=0;n_upper=0;n_pair=0;n_not_converged=0;n_selected_not_converged=0
    for bank_record in frozen:
        regime=bank_record['regime'];rep=bank_record['repetition'];ri=c['regimes'].index(regime)
        assert bank_record['bank_seed']==99100000+1000*ri+rep
        expected={name:98100000+1000*ri+10*rep+offset for name,offset in [('training',1),('validation',2),('calibration',3)]}
        assert bank_record['data_seeds']==expected;seeds.extend(expected.values())
        bank=NestedSoftFeatures();bank.__dict__.update(restore(bank_record['bank']))
        tx,ty=stream(expected['training'],c['n_train'],regime);vx,vy=stream(expected['validation'],c['n_validation'],regime)
        a.close(bank.input_min_,tx.min(axis=0));a.close(bank.input_max_,tx.max(axis=0))
        assert bank.input_scaling_=='minmax' and bank.cluster_counts==c['cluster_counts']
        assert np.array_equal(bank.fit_indices_,np.arange(c['n_train']))
        tf=bank.transform(tx);vf=bank.transform(vx)
        for name,v in [('train_x',tx),('train_y',ty),('valid_x',vx),('valid_y',vy),('train_features',tf),('valid_features',vf)]:
            assert array_digest(v)==bank_record['development_hashes'][name],(regime,rep,name)
        for pi,selection in enumerate(bank_record['models']):
            width=c['retained_feature_costs'][pi];blocks=bank.block_slices_[:pi+1]
            validation_scores=[]
            for cand in selection['candidates']:
                weights=np.array(cand['coefficients']);bias=cand['intercept']
                pred=np.clip(vf[:,:width]@weights+bias,*c['target_bounds'])
                mse=float(np.mean((vy-pred)**2));a.close(mse,cand['validation_mse']);validation_scores.append(mse)
                norm=sum(float(np.linalg.norm(weights[b])) for b in blocks);a.close(norm,cand['norm_sum'])
                assert norm<=c['kappa']+1e-10
                X=tf[:,:width]-tf[:,:width].mean(axis=0);Y=ty-ty.mean()
                gram=X.T@X/len(X)+cand['alpha']*np.eye(width);rhs=X.T@Y/len(X)
                grad=2*(gram@weights-rhs);gap=max(0.,float(weights@grad+c['kappa']*max(np.linalg.norm(grad[b]) for b in blocks)))
                a.close(gap,cand['optimization_gap']);max_gap=max(max_gap,gap);max_norm=max(max_norm,norm)
                objective=float(np.mean((ty-(tf[:,:width]@weights+bias))**2)+cand['alpha']*np.sum(weights**2))
                assert cand['converged']==(cand['optimization_gap']<=c['decoder_tolerance']*max(1.,abs(objective)))
                n_not_converged+=int(not cand['converged']);n_candidates+=1
            best=int(np.argmin(validation_scores));assert selection['selected']==selection['candidates'][best];n_selected+=1
            n_selected_not_converged+=int(not selection['selected']['converged'])
        x,y=stream(expected['calibration'],max(c['calibration_sizes']),regime);ftr=bank.transform(x)
        assert np.all((y>=c['target_bounds'][0])&(y<=c['target_bounds'][1]))
        preds=np.column_stack([np.clip(ftr[:,:s['selected']['cost']]@np.array(s['selected']['coefficients'])+s['selected']['intercept'],*c['target_bounds']) for s in bank_record['models']])
        for name,value in [('x',x),('y',y),('features',ftr),('predictions',preds)]:assert array_digest(value)==records[regime,rep]['hashes'][name],(regime,rep,name)
        labels={name:np.argmax(ftr[:,block],axis=1) for name,block in zip(c['partition_names'],bank.block_slices_)}
        n_full_loss_scores+=preds.size
        for n in c['calibration_sizes']:
            verified=[]
            for pi,selection in enumerate(bank_record['models']):
                width=selection['selected']['cost'];blocks=bank.block_slices_[:pi+1]
                stored=lookup[regime,rep,n,pi+1]
                losses=(y[:n]-preds[:n,pi])**2;upper=interval(losses,bound,du);a.scalar_details(upper,stored['upper']);n_upper+=1
                w=pair_scores(ftr[:n,:width],y[:n],blocks,c['kappa']);unc=interval(w,bound/2,dl);a.scalar_details(unc,stored['unconditional']);n_pair+=1
                strat=[]
                for name,m in zip(c['partition_names'],c['cluster_counts']):
                    lab=labels[name][:n];counts=np.bincount(lab,minlength=m)
                    radius=math.sqrt(math.log(4*m/eta)/(2*n));lo=np.maximum(0,counts/n-radius);hi=np.minimum(1,counts/n+radius)
                    st=stored['stratified'][name];a.close(counts,st['counts']);a.close(radius,st['mass_radius']);a.close(lo,st['mass_lower']);a.close(hi,st['mass_upper'])
                    conditional=[]
                    for s in range(m):
                        chosen=np.flatnonzero(lab==s);cs=st['conditional_intervals'][s]
                        assert cs['count']==len(chosen)
                        if len(chosen)<4:lower=0.;assert cs['insufficient_pairs']
                        else:
                            ws=pair_scores(ftr[chosen,:width],y[chosen],blocks,c['kappa']);ci=interval(ws,bound/2,eta/(2*m))
                            a.scalar_details(ci,cs);lower=ci['lower'];assert not cs['insufficient_pairs'];n_pair+=1
                        conditional.append(lower)
                    # Independent general LP, rather than the runner's sorted greedy algorithm.
                    lp=linprog(conditional,A_eq=np.ones((1,m)),b_eq=[1.],bounds=list(zip(lo,hi)),method='highs')
                    assert lp.success;a.close(lp.fun,st['lower']);n_mass_lps+=1;strat.append(lp.fun)
                    recorded=np.array(st['minimizing_mass']);a.close(recorded.sum(),1.)
                    assert np.all(recorded>=lo-1e-12) and np.all(recorded<=hi+1e-12);a.close(recorded@conditional,lp.fun)
                primary=max(unc['lower'],*strat);row=stored['record']
                for k,v in [('mse',upper['mean']),('upper',upper['upper']),('unconditional_witness',unc['mean']),('unconditional_lower',unc['lower']),('coarse_stratified_lower',strat[0]),('fine_stratified_lower',strat[1]),('primary_lower',primary)]:a.close(v,row[k])
                assert row['accepted']==(upper['upper']<=c['mse_tolerance']);assert row['excluded']==(primary>c['mse_tolerance'])
                verified.append(row);recomputed_endpoints.append(row)
            costs=[z['cost'] for z in verified];up=minimum(costs,[z['accepted'] for z in verified]);low=minimum(costs,[not z['excluded'] for z in verified])
            unc_low=minimum(costs,[z['unconditional_lower']<=c['mse_tolerance'] for z in verified])
            st_low=minimum(costs,[max(z['coarse_stratified_lower'],z['fine_stratified_lower'])<=c['mse_tolerance'] for z in verified])
            dr=decisions[regime,rep,n]
            for k,v in [('upper_choice',up),('cost_upper',up),('cost_lower',low),('unconditional_cost_lower',unc_low),('stratified_cost_lower',st_low)]:assert dr[k]==('' if v is None else str(v))
            bools={'minimum_cost_certified':up is not None and up==low,'all_classes_excluded':low is None,
                'unconditional_minimum_cost_certified':up is not None and up==unc_low,'unconditional_all_excluded':unc_low is None,
                'stratified_minimum_cost_certified':up is not None and up==st_low,'stratified_all_excluded':st_low is None}
            for k,v in bools.items():assert dr[k]==str(v)
            key=(regime,n)
            if key not in decision_summary:decision_summary[key]={k:0 for k in bools}|{'upper_acceptances':0,'n_repetitions':0}
            for k,v in bools.items():decision_summary[key][k]+=int(v)
            decision_summary[key]['upper_acceptances']+=int(up is not None);decision_summary[key]['n_repetitions']+=1
        print(f'Verified {regime} {rep+1}/20',flush=True)
    assert len(seeds)==len(set(seeds))==120
    rename={'minimum_cost_certificates':'minimum_cost_certified','joint_infeasibility_certificates':'all_classes_excluded',
        'unconditional_minimum_cost_certificates':'unconditional_minimum_cost_certified','unconditional_joint_infeasibility_certificates':'unconditional_all_excluded',
        'stratified_minimum_cost_certificates':'stratified_minimum_cost_certified','stratified_joint_infeasibility_certificates':'stratified_all_excluded',
        'upper_acceptances':'upper_acceptances','n_repetitions':'n_repetitions'}
    for row in r['summary']:
        checked=decision_summary[row['regime'],row['n_calibration']]
        for key,dest in rename.items():assert row[key]==checked[dest]
    report={'passed':True,'scope':'Portable no-refit feature/prediction replay and independent arithmetic; source hashes pinned; runner not imported; general LP used for each mass minimization',
        'n_banks':len(frozen),'n_candidate_fits_verified':n_candidates,'n_selected_models_verified':n_selected,'n_distinct_data_streams':len(seeds),
        'n_calibration_episode_model_losses_reconstructed':n_full_loss_scores,'n_upper_endpoints':n_upper,'n_unconditional_and_conditional_mean_intervals':n_pair,
        'n_mass_linear_programs':n_mass_lps,'n_decisions':len(decisions),'n_numeric_scalar_comparisons':a.n_comparisons,
        'max_numeric_difference':a.max_difference,'max_objective_gap':max_gap,'max_block_norm':max_norm,
        'candidate_optimizer_tolerance_not_reached':n_not_converged,'selected_optimizer_tolerance_not_reached':n_selected_not_converged,
        'confidence_allocation_total':count*(du+dl+2*eta),'results_sha256':digest(folder/'results.json')}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
