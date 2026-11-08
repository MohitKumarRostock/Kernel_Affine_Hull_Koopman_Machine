#!/usr/bin/env python3
"""Two-stage, frozen learned-KAHM certificate study.

Run ``develop`` before ``calibrate``. The latter verifies hashes of the
protocol, frozen selected models and exact execution source snapshot before it
generates any calibration sample. It never tunes or refits a predictor.
"""
from __future__ import annotations
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import tarfile
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from certified_forecasting.features import NestedSoftFeatures, _KAHMBlock
from certified_forecasting.models import BlockBoundedRidge
from certified_forecasting.certificates import mean_interval, pair_witness_scores


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_gzip_json(path, value):
    with open(path,'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as z:
            z.write((json.dumps(value, separators=(',',':'), allow_nan=False)+'\n').encode())


def read_gzip_json(path):
    with gzip.open(path,'rt') as stream:
        return json.load(stream)


def write_csv(path, rows):
    with open(path,'w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def encode(value):
    if isinstance(value,np.ndarray):
        return {'__array__':True,'dtype':value.dtype.str,'shape':list(value.shape),'data':value.ravel().tolist()}
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,slice): return {'__slice__':[value.start,value.stop,value.step]}
    if isinstance(value,dict): return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [encode(v) for v in value]
    if isinstance(value,_KAHMBlock): return {'__kahm_block__':encode(value.__dict__)}
    return value


def decode(value):
    if isinstance(value,dict):
        if '__array__' in value:return np.asarray(value['data'],dtype=value['dtype']).reshape(value['shape'])
        if '__slice__' in value:return slice(*value['__slice__'])
        if '__kahm_block__' in value:
            b=_KAHMBlock.__new__(_KAHMBlock); b.__dict__.update(decode(value['__kahm_block__'])); return b
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list): return [decode(v) for v in value]
    return value


def restore_bank(record):
    b=NestedSoftFeatures(); b.__dict__.update(decode(record['bank'])); return b


def draw(seed,n,regime):
    rng=np.random.default_rng(seed)
    signs=rng.choice(np.array([-1.,1.]),size=(n,2))
    x=signs*(.9+rng.uniform(-.1,.1,size=(n,2)))
    if regime=='repairable': driver=x
    elif regime=='irreducible':
        hidden_signs=rng.choice(np.array([-1.,1.]),size=(n,2))
        driver=hidden_signs*(.9+rng.uniform(-.1,.1,size=(n,2)))
    else: raise ValueError('Unknown regime')
    y=.8*np.prod(driver,axis=1)+rng.uniform(-.05,.05,size=n)
    return x,y


def source_paths():
    # Freeze the inherited KAHM's local import closure, rather than unrelated
    # experiments that can be edited concurrently. Numerical dependencies of
    # this study are explicit below; later audit and asset scripts do not
    # influence model selection or calibration.
    paths=set()
    def add_local_imports(path):
        if path in paths:return
        paths.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=([n.name for n in node.names] if isinstance(node,ast.Import)
                   else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                candidate=REPOSITORY/(name.split('.')[0]+'.py')
                if candidate.is_file():add_local_imports(candidate)
    add_local_imports(REPOSITORY/'kernel_affine_hull_koopman_machines.py')
    for name in ['__init__.py','features.py','models.py','certificates.py']:
        paths.add(ROOT/'src/certified_forecasting'/name)
    experiment=ROOT/'experiments/learned_kahm_decision'
    for name in ['run_study.py','pilot.py','pilot_sign.py','protocol.json','protocol.md']:
        paths.add(experiment/name)
    paths.update(experiment.glob('pilot*results.json'))
    return sorted(paths)


def snapshot(output):
    members=[]
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as tar:
        for path in source_paths():
            arc=str(path.relative_to(REPOSITORY)); contents=path.read_bytes()
            info=tarfile.TarInfo(arc); info.size=len(contents); info.mtime=0; info.mode=0o644
            tar.addfile(info,io.BytesIO(contents))
            members.append({'path':arc,'sha256':hashlib.sha256(contents).hexdigest(),'bytes':len(contents)})
    packed=gzip.compress(buffer.getvalue(),mtime=0)
    expected=hashlib.sha256(packed).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as tar:
        if len(tar.getmembers())!=len(members):raise RuntimeError('Incomplete in-memory source archive')
    temporary=output/'source_snapshot.tar.gz.tmp'
    temporary.write_bytes(packed)
    if file_hash(temporary)!=expected:raise RuntimeError('Source archive write failed validation')
    temporary.replace(output/'source_snapshot.tar.gz')
    if file_hash(output/'source_snapshot.tar.gz')!=expected:raise RuntimeError('Source archive replacement failed validation')
    write_json(output/'source_snapshot.json',{'archive_sha256':expected,'files':members})
    return members


def develop(config, output):
    if output.exists(): raise FileExistsError('Use a new output directory; development results are immutable')
    output.mkdir(parents=True)
    write_json(output/'protocol.json',config)
    members=snapshot(output)
    write_json(output/'protocol_freeze.json',{'frozen_at_utc':timestamp(),'calibration_generated':False,
        'protocol_sha256':file_hash(output/'protocol.json'),'source_snapshot_sha256':file_hash(output/'source_snapshot.tar.gz'),
        'source_manifest_sha256':file_hash(output/'source_snapshot.json'),'python':platform.python_version(),'numpy':np.__version__})
    records=[]; all_arrays={}; candidate_rows=[]
    for ri,regime in enumerate(config['regimes']):
        for rep in range(config['n_repetitions_per_regime']):
            base=98100000+1000*ri+10*rep
            train_x,train_y=draw(base+1,config['n_train'],regime)
            valid_x,valid_y=draw(base+2,config['n_validation'],regime)
            seed=99100000+1000*ri+rep
            bank=NestedSoftFeatures(cluster_counts=config['cluster_counts'],kind='kahm',seed=seed,
                input_scaling='minmax',max_fit=config['max_fit'],max_train_per_cluster=config['max_train_per_cluster'],
                subspace_dim=config['subspace_dim'],omega=config['omega'],tau=config['tau']).fit(train_x)
            tf=bank.transform(train_x); vf=bank.transform(valid_x)
            selections=[]
            for prefix in (1,2):
                blocks=bank.block_slices_[:prefix]; width=blocks[-1].stop
                candidates=[]
                for alpha in config['ridge_grid']:
                    m=BlockBoundedRidge(alpha=alpha,kappa=config['kappa'],bounds=config['target_bounds'],
                        block_slices=blocks,tol=config['decoder_tolerance'],max_iter=config['decoder_max_iter']).fit(tf[:,:width],train_y)
                    validation_mse=float(np.mean((valid_y-m.predict(vf[:,:width])[:,0])**2))
                    record={'regime':regime,'repetition':rep,'prefix':prefix,'cost':width,'alpha':alpha,
                        'validation_mse':validation_mse,'norm_sum':m.norm_sum_,'optimization_gap':m.optimization_gap_,
                        'iterations':m.n_iter_,'converged':bool(m.converged_),'coefficients':m.coef_[:,0].tolist(),
                        'intercept':float(m.intercept_[0])}
                    candidates.append(record); candidate_rows.append(record)
                selected=min(candidates,key=lambda z:z['validation_mse'])
                selections.append({'selected':selected,'candidates':candidates})
            record={'regime':regime,'repetition':rep,'data_seeds':{'training':base+1,'validation':base+2,'calibration':base+3},
                'bank_seed':seed,'bank':encode(bank.__dict__),'scaling':bank.input_scaling_metadata(),
                'development_hashes':{key:array_hash(a) for key,a in [('train_x',train_x),('train_y',train_y),('valid_x',valid_x),('valid_y',valid_y),('train_features',tf),('valid_features',vf)]},
                'models':selections}
            records.append(record)
            for key,a in [('train_x',train_x),('train_y',train_y),('valid_x',valid_x),('valid_y',valid_y),('train_features',tf),('valid_features',vf)]:
                all_arrays[f'{regime}_{rep}_{key}']=a
            print(f'Frozen development {regime} {rep+1}/20; MSEs {[r["selected"]["validation_mse"] for r in selections]}',flush=True)
    write_gzip_json(output/'frozen_models.json.gz',records)
    np.savez_compressed(output/'development_arrays.npz',**all_arrays)
    write_csv(output/'development_scores.csv',[{k:v for k,v in r.items() if k!='coefficients'} for r in candidate_rows])
    manifest={'frozen_at_utc':timestamp(),'calibration_generated':False,'n_banks':len(records),'n_fits':len(candidate_rows),
        'n_selected_models':sum(len(r['models']) for r in records),'all_converged':all(r['converged'] for r in candidate_rows),
        'maximum_objective_gap':max(r['optimization_gap'] for r in candidate_rows),'frozen_models_sha256':file_hash(output/'frozen_models.json.gz'),
        'development_scores_sha256':file_hash(output/'development_scores.csv'),'development_arrays_sha256':file_hash(output/'development_arrays.npz'),
        'protocol_freeze_sha256':file_hash(output/'protocol_freeze.json')}
    write_json(output/'development_freeze.json',manifest)
    print(json.dumps(manifest),flush=True)


def mass_minimum(lower,upper,values):
    masses=lower.copy(); remaining=1-float(masses.sum())
    for i in np.argsort(values,kind='stable'):
        extra=min(float(upper[i]-masses[i]),max(0.,remaining));masses[i]+=extra;remaining-=extra
    if abs(remaining)>1e-9: raise RuntimeError('Invalid mass interval polytope')
    return float(masses@values),masses


def stratified(features,y,labels,m,kappa,blocks,pair_bound,eta):
    n=len(y);counts=np.bincount(labels,minlength=m)
    radius=math.sqrt(math.log(4*m/eta)/(2*n))
    lower=np.maximum(0,counts/n-radius);upper=np.minimum(1,counts/n+radius)
    details=[]
    for s in range(m):
        chosen=np.flatnonzero(labels==s)
        if len(chosen)<4:
            d={'lower':0.,'n_independent':len(chosen)//2,'insufficient_pairs':True}
        else:
            w=pair_witness_scores(features[chosen,None,:],y[chosen,None,None],kappa,blocks)[:,0]
            d=mean_interval(w,pair_bound,eta/(2*m));d['insufficient_pairs']=False
        d.update(stratum=s,count=len(chosen));details.append(d)
    value,minimizer=mass_minimum(lower,upper,np.array([d['lower'] for d in details]))
    return {'lower':value,'eta':eta,'mass_radius':radius,'counts':counts.tolist(),'mass_lower':lower.tolist(),
        'mass_upper':upper.tolist(),'minimizing_mass':minimizer.tolist(),'conditional_intervals':details}


def min_cost(costs,flags):
    passing=[c for c,f in zip(costs,flags) if f]
    return min(passing) if passing else None


def calibrate(output):
    if (output/'results.json').exists() or (output/'calibration_started.json').exists():
        raise FileExistsError('Refusing to overwrite or resume an already opened calibration study')
    config=json.loads((output/'protocol.json').read_text())
    freeze=json.loads((output/'protocol_freeze.json').read_text()); dev=json.loads((output/'development_freeze.json').read_text())
    for key,name in [('protocol_sha256','protocol.json'),('source_snapshot_sha256','source_snapshot.tar.gz'),('source_manifest_sha256','source_snapshot.json')]:
        if file_hash(output/name)!=freeze[key]:raise RuntimeError(f'Frozen {name} hash mismatch')
    for key,name in [('frozen_models_sha256','frozen_models.json.gz'),('development_scores_sha256','development_scores.csv'),('protocol_freeze_sha256','protocol_freeze.json')]:
        if file_hash(output/name)!=dev[key]:raise RuntimeError(f'Development {name} hash mismatch')
    source_manifest=json.loads((output/'source_snapshot.json').read_text())
    for item in source_manifest['files']:
        if file_hash(REPOSITORY/item['path'])!=item['sha256']:raise RuntimeError(f'Execution source changed: {item["path"]}')
    frozen=read_gzip_json(output/'frozen_models.json.gz')
    write_json(output/'calibration_started.json',{'started_at_utc':timestamp(),'development_freeze_sha256':file_hash(output/'development_freeze.json'),
        'calibration_generated_before_this_point':False})
    diameter=config['target_bounds'][1]-config['target_bounds'][0];bound=diameter**2
    count=config['n_class_size_records'];du=config['upper_budget']/count;dl=config['unconditional_lower_budget']/count
    eta=config['stratified_lower_budget']/(count*len(config['partition_names']))
    endpoint_rows=[];decision_rows=[];all_details=[];arrays={};cal_records=[]
    for record in frozen:
        regime,rep=record['regime'],record['repetition'];seed=record['data_seeds']['calibration']
        x,y=draw(seed,max(config['calibration_sizes']),regime)
        bank=restore_bank(record);f=bank.transform(x)
        labels={name:np.argmax(f[:,block],axis=1) for name,block in zip(config['partition_names'],bank.block_slices_)}
        predictions=[]
        for model in record['models']:
            selected=model['selected'];width=selected['cost']
            prediction=np.clip(f[:,:width]@np.array(selected['coefficients'])+selected['intercept'],*config['target_bounds'])
            predictions.append(prediction)
        predictions=np.column_stack(predictions)
        for key,a in [('x',x),('y',y),('features',f),('predictions',predictions),*[(name,l) for name,l in labels.items()]]:
            arrays[f'{regime}_{rep}_{key}']=a
        cal_records.append({'regime':regime,'repetition':rep,'seed':seed,
            'hashes':{key:array_hash(a) for key,a in [('x',x),('y',y),('features',f),('predictions',predictions)]},
            'outside_training_range_fraction':float(np.mean((x<bank.input_min_)|(x>bank.input_max_))),
            'scaled_minimum':float(bank._scale_inputs(x).min()),'scaled_maximum':float(bank._scale_inputs(x).max())})
        for n in config['calibration_sizes']:
            point=[]
            for pi,selection in enumerate(record['models']):
                model=selection['selected'];width=model['cost'];blocks=bank.block_slices_[:pi+1]
                losses=(y[:n]-predictions[:n,pi])**2
                upper=mean_interval(losses,bound,du)
                witness=pair_witness_scores(f[:n,None,:width],y[:n,None,None],config['kappa'],blocks)[:,0]
                unconditional=mean_interval(witness,bound/2,dl)
                strat={name:stratified(f[:n,:width],y[:n],labels[name][:n],bank.blocks_[si].n_features_out_,config['kappa'],blocks,bound/2,eta) for si,name in enumerate(config['partition_names'])}
                lower=max(unconditional['lower'],*(r['lower'] for r in strat.values()))
                row={'regime':regime,'repetition':rep,'n_calibration':n,'prefix':pi+1,'cost':width,
                    'mse':upper['mean'],'upper':upper['upper'],'unconditional_witness':unconditional['mean'],
                    'unconditional_lower':unconditional['lower'],'coarse_stratified_lower':strat['learned_coarse']['lower'],
                    'fine_stratified_lower':strat['learned_fine']['lower'],'primary_lower':lower,
                    'accepted':upper['upper']<=config['mse_tolerance'],'excluded':lower>config['mse_tolerance']}
                endpoint_rows.append(row);point.append(row)
                all_details.append({'record':row,'upper':upper,'unconditional':unconditional,'stratified':strat})
            costs=[p['cost'] for p in point]
            c_up=min_cost(costs,[p['accepted'] for p in point]); c_low=min_cost(costs,[not p['excluded'] for p in point])
            c_low_uncond=min_cost(costs,[p['unconditional_lower']<=config['mse_tolerance'] for p in point])
            c_low_strat=min_cost(costs,[max(p['coarse_stratified_lower'],p['fine_stratified_lower'])<=config['mse_tolerance'] for p in point])
            vchoice=min_cost(costs,[r['selected']['validation_mse']<=config['mse_tolerance'] for r in record['models']])
            decision_rows.append({'regime':regime,'repetition':rep,'n_calibration':n,'validation_choice':vchoice,
                'upper_choice':c_up,'cost_lower':c_low,'cost_upper':c_up,'minimum_cost_certified':c_up is not None and c_up==c_low,
                'all_classes_excluded':c_low is None,'unconditional_cost_lower':c_low_uncond,
                'unconditional_minimum_cost_certified':c_up is not None and c_up==c_low_uncond,
                'unconditional_all_excluded':c_low_uncond is None,'stratified_cost_lower':c_low_strat,
                'stratified_minimum_cost_certified':c_up is not None and c_up==c_low_strat,
                'stratified_all_excluded':c_low_strat is None})
        print(f'Calibrated {regime} {rep+1}/20',flush=True)
    write_csv(output/'endpoints.csv',endpoint_rows);write_csv(output/'decisions.csv',decision_rows)
    write_gzip_json(output/'interval_details.json.gz',all_details)
    write_json(output/'calibration_records.json',cal_records)
    np.savez_compressed(output/'calibration_arrays.npz',**arrays)
    summaries=[]
    for regime in config['regimes']:
        for n in config['calibration_sizes']:
            rows=[r for r in decision_rows if r['regime']==regime and r['n_calibration']==n]
            summaries.append({'regime':regime,'n_calibration':n,'n_repetitions':len(rows),
                'upper_acceptances':sum(r['upper_choice'] is not None for r in rows),
                'minimum_cost_certificates':sum(r['minimum_cost_certified'] for r in rows),
                'joint_infeasibility_certificates':sum(r['all_classes_excluded'] for r in rows),
                'unconditional_minimum_cost_certificates':sum(r['unconditional_minimum_cost_certified'] for r in rows),
                'unconditional_joint_infeasibility_certificates':sum(r['unconditional_all_excluded'] for r in rows),
                'stratified_minimum_cost_certificates':sum(r['stratified_minimum_cost_certified'] for r in rows),
                'stratified_joint_infeasibility_certificates':sum(r['stratified_all_excluded'] for r in rows)})
    write_csv(output/'summary.csv',summaries)
    results={'completed_at_utc':timestamp(),'scope':config['scope'],'n_banks':len(frozen),'n_selected_models':80,'n_endpoints':len(endpoint_rows),
        'n_decisions':len(decision_rows),'joint_failure_probability':config['confidence_delta'],'mse_tolerance':config['mse_tolerance'],
        'confidence_allocation':{'upper_per_endpoint':du,'unconditional_per_endpoint':dl,'stratified_per_partition':eta,
            'allocated_total':count*(du+dl+2*eta)},'summary':summaries,
        'known_bayes_risk':{'repairable':.05**2/3,'irreducible':.8**2*(.9**2+.1**2/3)**2+.05**2/3},
        'hashes':{name:file_hash(output/name) for name in ['endpoints.csv','decisions.csv','interval_details.json.gz','calibration_records.json','calibration_arrays.npz','summary.csv','calibration_started.json','development_freeze.json']}}
    write_json(output/'results.json',results)
    print(json.dumps(results,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['develop','calibrate']);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--protocol',type=Path,default=Path(__file__).with_name('protocol.json'))
    args=parser.parse_args()
    if args.stage=='develop': develop(json.loads(args.protocol.read_text()),args.output)
    else:calibrate(args.output)

if __name__=='__main__':main()
