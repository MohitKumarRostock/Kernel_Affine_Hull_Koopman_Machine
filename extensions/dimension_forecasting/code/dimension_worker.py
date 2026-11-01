#!/usr/bin/env python3
"""Staged fit and target-free prediction worker for the frozen dimension study."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import resource
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from focused_dimension_model import (
    METHODS, GEOMETRY_METHODS, FULL_METHODS, SHARED_METHODS, NONKAHM_METHODS,
    array_hash, fit_shared, fit_candidate, calibrate_shared,
    select_cap, calibrate, state_bytes, state_hash, runtime_provenance,
    load_runtime, select_context_policies, prediction_views,
)
from study_contract import verify_freeze,verify_abc_manifest,verify_fit_lock

CANDIDATES = (('historical2',2,'legacy_normal_departure'),
              ('native1',1,'native'), ('native2',2,'native'),
              ('native4',4,'native'), ('native8',8,'native'),
              ('native12',12,'native'), ('native20',20,'native'))
NATIVE_ORDER = tuple(name for name,_,policy in CANDIDATES if policy == 'native')


def deployment_state_sizes(model):
    """Apply the principal source's per-procedure state-accounting convention."""
    m=load_runtime();candidate=model.candidate;output={}
    for method in METHODS:
        state={'input_scaler':candidate.input_scaler,'baseline':candidate.baseline}
        if method!='ridge':
            kind=method.split('_')[0]
            excluded={'reference_distance','factor_error'}
            if kind=='common': excluded|={'q','inverse','whitening','uq','lambda_original'}
            elif kind=='rbf': excluded|={'inverse','lambda_original'}
            bank={**candidate.geometry.bank,'models':[
                {key:value for key,value in local.items() if key not in excluded}
                for local in candidate.geometry.bank['models']]}
            state.update(gamma=model.gammas[method],bank=bank,head=candidate.heads[method],
                scalers={key:candidate.geometry.scalers[key] for key in ('common',kind)})
        output[method]=m._measure_state(state)
    return output


def shared_state_sizes(model):
    m=load_runtime();shared=model.shared;base=shared.baseline_fit;result={}
    for method in SHARED_METHODS:
        state={'input_scaler':base.input_scaler,'baseline':base.head}
        if method!='ridge':state['gamma']=model.gammas[method]
        if method=='raw_rbf':state['model']=shared.raw_rbf
        elif method=='raw_hgb':state['model']=shared.raw_hgb
        result[method]=m._measure_state(state)
    return result


def procedure_resources(shared,candidates,selections,contexts,selection_costs,policy_costs):
    """Original dependency charging extended to the entire B cap search.

    Nodes are identified globally, so a policy pays the union of required work.
    Candidate baseline query nodes are excluded because the shared baseline
    already supplies that complete primitive. Actual joint costs are separate.
    """
    m=load_runtime();registry={};context_resources={}
    def add(prefix,nodes,method,*,without_baseline=False):
        keys=[]
        for key in m._dependencies(nodes,method):
            if key not in nodes:continue
            if without_baseline and (key.startswith('baseline_') or key=='input_scaler_A'):continue
            identity=prefix+'/'+key;registry[identity]=nodes[key];keys.append(identity)
        return set(keys)
    def total(ids):
        return {'wall_seconds':sum(registry[key]['wall_seconds'] for key in ids),
                'cpu_seconds':sum(registry[key]['cpu_seconds'] for key in ids),
                'node_ids':sorted(ids)}
    for context_name,context in contexts.items():
        ids={};states={}
        for method in FULL_METHODS[:-1]:
            base_method=method if method in SHARED_METHODS else 'ridge'
            selected=add('shared/A_B',shared['fit_diagnostics']['fit_nodes'],base_method)
            selected|=add('shared/C',shared['calibration']['nodes'],base_method)
            if method in GEOMETRY_METHODS:
                chosen=context['method_to_candidate'][method]
                search=NATIVE_ORDER if context_name=='selected' else (chosen,)
                for candidate_id in search:
                    selected|=add(candidate_id+'/A_B',candidates[candidate_id]['fit_diagnostics']['fit_nodes'],
                                  method,without_baseline=True)
                selected|=add(chosen+'/C',candidates[chosen]['calibration']['nodes'],method,without_baseline=True)
                if context_name=='selected':
                    key='B_cap_selection/'+method;registry[key]=selection_costs[method];selected.add(key)
                states[method]=candidates[chosen]['resources']['per_method_deployment_state'][method]
            else:states[method]=shared['resources']['per_method_deployment_state'][method]
            ids[method]=selected
        policy=set().union(*(ids[method] for method in NONKAHM_METHODS))
        key=context_name+'/policy_select_C';registry[key]=policy_costs[context_name];policy.add(key)
        ids['nonkahm_policy']=policy
        states['nonkahm_policy']=states[context['selected_policy']]
        context_resources[context_name]={method:{'fit_and_selection':total(ids[method]),
            'state':states[method]} for method in FULL_METHODS}
    return {'per_context':context_resources,'node_registry':registry,
        'scope':'Source dependency convention; native selected methods pay all six cap searches; C policy pays unique node union; actual joint task costs reported separately'}


def query_procedure_resources(resources,contexts,view_nodes):
    m=load_runtime();registry={};result={}
    def add(name,method,*,without_baseline=False):
        nodes=resources[name]['nodes'];ids=set()
        for key in m._dependencies(nodes,method,prediction=True):
            if key not in nodes:continue
            if without_baseline and (key.startswith('baseline_') or key=='input_scaler_A'):continue
            identity=name+'/'+key;registry[identity]=nodes[key];ids.add(identity)
        return ids
    for context_name,context in contexts.items():
        ids={}
        for method in FULL_METHODS[:-1]:
            selected=add('shared',method if method in SHARED_METHODS else 'ridge')
            if method in GEOMETRY_METHODS:
                selected|=add(context['method_to_candidate'][method],method,without_baseline=True)
            ids[method]=selected
        policy=ids[context['selected_policy']].copy()
        key=context_name+'/calibrated/policy_forward';registry[key]=view_nodes[key];policy.add(key)
        ids['nonkahm_policy']=policy
        result[context_name]={method:{'wall_seconds':sum(registry[key]['wall_seconds'] for key in chosen),
            'cpu_seconds':sum(registry[key]['cpu_seconds'] for key in chosen),'node_ids':sorted(chosen)}
            for method,chosen in ids.items()}
    return {'per_context':result,'node_registry':registry,
            'scope':'Calibrated procedure query using selected frozen state; policy pays selected procedure plus forward copy; no cap search at query time'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return value


def write_bytes(path,body):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(body);stream.flush();os.fsync(stream.fileno())
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=hashlib.sha256(body).hexdigest():
        raise IOError('Write verification failed: '+str(path))
    temporary.replace(path)


def write_json(path,value):
    write_bytes(path,(json.dumps(json_safe(value),indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def write_npz(path,arrays):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    with np.load(temporary,allow_pickle=False) as saved:
        if set(saved.files)!=set(arrays): raise IOError('Array roster changed during write')
        for name,value in arrays.items():
            if array_hash(saved[name])!=array_hash(value): raise IOError('Array write mismatch: '+name)
    temporary.replace(path)


def _identity_control(candidates,left='historical2',right='native2'):
    left_id,right_id=left,right
    old,new=candidates[left],candidates[right]
    checked=0
    for method in METHODS:
        np.testing.assert_array_equal(old.b_predictions[method],new.b_predictions[method]);checked+=1
    if len(old.geometry.bank['models'])!=len(new.geometry.bank['models']):
        raise AssertionError('Cap2 local model counts differ')
    for left,right in zip(old.geometry.bank['models'],new.geometry.bank['models']):
        for key in ('q','p','center','radius','rank','whitening','uq','inverse','lambda_original'):
            if (key in left)!=(key in right): raise AssertionError('Cap2 local state keys differ')
            if key in left:
                np.testing.assert_array_equal(left[key],right[key]);checked+=1
    for method in GEOMETRY_METHODS:
        for key in ('coef','alpha_by_horizon','mean','ymean'):
            np.testing.assert_array_equal(old.heads[method][key],new.heads[method][key]);checked+=1
        if old.diagnostics['B'][method]!=new.diagnostics['B'][method]:
            raise AssertionError('Cap2 B alpha selection records differ')
    return {'status':'passed','exact_numeric_checks':checked,
            'separately_fitted':True,'metadata_policy_difference_intentional':True,
            'compared_candidates':[left_id,right_id],
            'reason':('At history>=3, both initial requests equal min(2,N_local-1)'
                      if left_id=='historical2' else 'At history12, native12/native20 initial requests both equal min(12,N_local-1)')}


def _reference_identity(candidates):
    reference=candidates['historical2'].geometry.bank;checks=0
    for name,candidate in candidates.items():
        bank=candidate.geometry.bank
        if bank['block_slices']!=reference['block_slices'] or len(bank['models'])!=len(reference['models']):
            raise AssertionError('Candidate partition roster differs')
        for key in ('mean','scale'):
            np.testing.assert_array_equal(bank[key],reference[key]);checks+=1
        for local,original in zip(bank['models'],reference['models']):
            for key in ('q','center','radius','block'):
                np.testing.assert_array_equal(local[key],original[key]);checks+=1
    return {'status':'passed','exact_numeric_checks':checks,
            'scope':'Identical A-scaling, partitions, ordered reference rows, centers and radii across all7 caps/policies'}


def fit_task(input_path,task,output,*,model_config=None,freeze_sha256=None):
    """A/B construction, persisted cap selection, then C calibration only."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=utcnow();wall=time.perf_counter();cpu=time.process_time()
    summary={'schema':'kahm-dimension-fit-v1','task':task,'started_utc':started,
             'freeze_sha256':freeze_sha256,'status':'running'}
    try:
        with np.load(input_path,allow_pickle=False) as source:
            if any(key.startswith('D_') for key in source.files):
                raise ValueError('Fitting input archive must not contain D arrays')
            ax,ay,bx,by=[source[key] for key in ('A_X','A_Y','B_X','B_Y')]
        if ax.shape[1] != int(task['history']): raise ValueError('Task history differs from input width')
        base_wall,base_cpu=time.perf_counter(),time.process_time()
        shared=fit_shared(ax,ay,bx,by,seed=int(task['fit_seed']),config=model_config)
        baseline=shared.baseline_fit
        baseline_cost={'wall_seconds':time.perf_counter()-base_wall,'cpu_seconds':time.process_time()-base_cpu}
        candidates={};candidate_cost={};validation={'A_target_variance':baseline.loss_scale}
        validation.update({f'shared__{method}':value for method,value in shared.b_predictions.items()})
        basis_arrays={}
        for name,cap,policy in CANDIDATES:
            step_wall,step_cpu=time.perf_counter(),time.process_time()
            candidate=fit_candidate(ax,ay,bx,by,candidate_id=name,rank_cap=cap,
                dimension_policy=policy,seed=int(task['fit_seed']),config=model_config,baseline_fit=baseline)
            candidates[name]=candidate
            candidate_cost[name]={'wall_seconds':time.perf_counter()-step_wall,
                                   'cpu_seconds':time.process_time()-step_cpu}
            validation.update({f'{name}__{method}':value for method,value in candidate.b_predictions.items()})
            for j,local in enumerate(candidate.geometry.bank['models']):
                for key in ('q','p'):
                    basis_arrays[f'{name}__model{j}__{key}']=local[key]
        control=_identity_control(candidates)
        reference_identity=_reference_identity(candidates)
        ambient_control=(_identity_control(candidates,'native12','native20') if task['history']==12 else
                         {'status':'not_applicable','reason':'History exceeds12'})
        native={key:candidates[key] for key in NATIVE_ORDER}
        selection_wall,selection_cpu=time.perf_counter(),time.process_time()
        selections={};selection_costs={}
        for method in GEOMETRY_METHODS:
            sw,sc=time.perf_counter(),time.process_time()
            selections[method]=select_cap(native,by,cap_order=NATIVE_ORDER,method=method)[1]
            selection_costs[method]={'wall_seconds':time.perf_counter()-sw,'cpu_seconds':time.process_time()-sc}
        selection_cost={'wall_seconds':time.perf_counter()-selection_wall,
                        'cpu_seconds':time.process_time()-selection_cpu}
        pre_c={'schema':'kahm-dimension-pre-c-v1','task':task,'completed_utc':utcnow(),
            'selections':selections,'candidate_order':[c[0] for c in CANDIDATES],
            'native_cap_order':list(NATIVE_ORDER),
            'input_hashes':baseline.input_hashes,
            'candidate_state_sha256':{name:state_hash(value) for name,value in candidates.items()},
            'shared_state_sha256':state_hash(shared),
            'A_target_variance':baseline.loss_scale,
            'historical_native2_identity':control,
            'reference_identity_across_candidates':reference_identity,
            'native12_native20_identity':ambient_control,
            'matched_control_cap_source':'kahm_departure B-selected native cap',
            'C_arrays_loaded':False,'D_generated_or_loaded':False}
        write_npz(output/'validation_predictions.npz',validation)
        write_npz(output/'basis_diagnostics.npz',basis_arrays)
        # This durable selection record is written before opening C arrays.
        write_json(output/'pre_c_selection.json',pre_c)
        with np.load(input_path,allow_pickle=False) as source:
            cx,cy=source['C_X'],source['C_Y']
        calibration_arrays={};states={};candidate_records={};c_results={};c_values={}
        if state_hash(shared)!=pre_c['shared_state_sha256']:
            raise AssertionError('Shared A/B state changed before C')
        sw,sc=time.perf_counter(),time.process_time()
        shared_frozen,shared_raw,shared_calibrated=calibrate_shared(shared,cx,cy)
        shared_calibration_cost={'wall_seconds':time.perf_counter()-sw,'cpu_seconds':time.process_time()-sc}
        shared_payload=state_bytes(shared_frozen);shared_compressed=gzip.compress(shared_payload,compresslevel=6,mtime=0)
        write_bytes(output/'states/shared.pkl.gz',shared_compressed)
        shared_state={'path':'states/shared.pkl.gz','sha256':sha256_file(output/'states/shared.pkl.gz'),
            'uncompressed_state_sha256':hashlib.sha256(shared_payload).hexdigest(),
            'compressed_bytes':len(shared_compressed),'pickle_bytes':len(shared_payload)}
        states['shared']=shared_state
        shared_record={'fit_diagnostics':shared.diagnostics,'calibration':shared_frozen.calibration,
            'state':shared_state,'resources':{'A_B_fit_and_selection':baseline_cost,
                'C_calibration_including_queries':shared_calibration_cost,
                'per_method_deployment_state':shared_state_sizes(shared_frozen)}}
        shared_result={'predictions':shared_calibrated,'uncalibrated_predictions':shared_raw}
        for name,_,_ in CANDIDATES:
            candidate=candidates[name]
            if state_hash(candidate)!=pre_c['candidate_state_sha256'][name]:
                raise AssertionError('A/B candidate changed after cap selection')
            step_wall,step_cpu=time.perf_counter(),time.process_time()
            frozen,raw,calibrated=calibrate(candidate,cx,cy)
            calibration_cost={'wall_seconds':time.perf_counter()-step_wall,
                              'cpu_seconds':time.process_time()-step_cpu}
            for mode,values in (('gamma1',raw),('calibrated',calibrated)):
                calibration_arrays.update({f'{name}__{mode}__{method}':value for method,value in values.items()})
            c_results[name]={'predictions':calibrated,'uncalibrated_predictions':raw}
            c_values[name]=calibrated
            payload=state_bytes(frozen)
            compressed=gzip.compress(payload,compresslevel=6,mtime=0)
            relative=f'states/{name}.pkl.gz'
            write_bytes(output/relative,compressed)
            states[name]={'path':relative,'sha256':sha256_file(output/relative),
                          'uncompressed_state_sha256':hashlib.sha256(payload).hexdigest(),
                          'compressed_bytes':len(compressed),'pickle_bytes':len(payload)}
            candidate_records[name]={
                'candidate_id':name,'rank_cap':candidate.rank_cap,'dimension_policy':candidate.dimension_policy,
                'fit_diagnostics':candidate.diagnostics,'calibration':frozen.calibration,
                'bounds':frozen.bounds,'state':states[name],
                'feature_widths':{
                    'common_geometry':len(candidate.geometry.scalers['common']['mean']),
                    'kahm_departure_geometry':len(candidate.geometry.scalers['kahm']['mean']),
                    'rbf_departure_geometry':len(candidate.geometry.scalers['rbf']['mean']),
                    'full_readout':{method:int(candidate.heads[method]['feature_width']) for method in GEOMETRY_METHODS}},
                'resources':{'fit_excluding_shared_ridge':candidate_cost[name],
                             'C_calibration_including_feature_queries':calibration_cost,
                             'per_method_deployment_state':deployment_state_sizes(frozen)}}
        contexts=select_context_policies(selections,c_values,shared_calibrated,cy,baseline.loss_scale)
        policy_costs={name:record['selection_cost'] for name,record in contexts.items()}
        matched=selections['kahm_departure']['selected_candidate_id']
        calibration_arrays=prediction_views(c_results,shared_result,contexts,matched)
        write_npz(output/'calibration_predictions.npz',calibration_arrays)
        files={name:sha256_file(output/name) for name in
               ('validation_predictions.npz','basis_diagnostics.npz','pre_c_selection.json','calibration_predictions.npz')}
        files.update({state['path']:state['sha256'] for state in states.values()})
        summary.update(status='completed',completed_utc=utcnow(),
            runtime_provenance=runtime_provenance(),input_sha256=sha256_file(input_path),
            A_B_C_rows=[len(ax),len(bx),len(cx)],A_target_variance=baseline.loss_scale,
            selections=selections,historical_native2_identity=control,candidates=candidate_records,
            shared=shared_record,contexts=contexts,matched_candidate_id=matched,
            procedure_resources=procedure_resources(shared_record,candidate_records,selections,contexts,
                                                    selection_costs,policy_costs),
            native12_native20_identity=ambient_control,
            reference_identity_across_candidates=reference_identity,
            resources={'shared_competitors_A_B':baseline_cost,'B_cap_selection':selection_cost,
                'B_cap_selection_per_method':selection_costs,'C_policy_selection_per_context':policy_costs,
                'task_fit_wall_seconds':time.perf_counter()-wall,'task_fit_cpu_seconds':time.process_time()-cpu,
                'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'scope':'All7 banks with five geometry heads each, shared original raw competitors once, C calibration and policy; diagnostic/serialization overhead retained'},
            files=files)
        write_json(output/'fit_summary.json',summary)
        return summary
    except Exception as error:
        summary.update(status='failed',exception=repr(error),traceback=traceback.format_exc(),completed_utc=utcnow())
        write_json(output/'fit_failure.json',summary)
        raise


def predict_task(fit_dir,query_path,output,*,fit_lock_sha256):
    """Replay frozen states on D_X only; no target file or target argument."""
    # Fresh query processes must register the verified source module classes
    # before deserializing the frozen ReconstructionFeatures object.
    load_runtime()
    fit_dir,output=Path(fit_dir),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    fit=json.loads((fit_dir/'fit_summary.json').read_text())
    if fit['status']!='completed': raise ValueError('Cannot predict an incomplete fit')
    for relative,expected in fit['files'].items():
        if sha256_file(fit_dir/relative)!=expected: raise ValueError('Fitted artifact changed: '+relative)
    with np.load(query_path,allow_pickle=False) as archive:
        if any(key in archive.files for key in ('D_Y','Y','targets')):
            raise ValueError('Query archive contains reserved targets')
        x=archive['D_X']
    predictions={};diagnostics={};resources={};state_identity={};candidate_results={}
    wall,cpu=time.perf_counter(),time.process_time()
    for name in ('shared',)+tuple(c[0] for c in CANDIDATES):
        state=(fit['shared'] if name=='shared' else fit['candidates'][name])['state']
        payload=gzip.decompress((fit_dir/state['path']).read_bytes())
        if hashlib.sha256(payload).hexdigest()!=state['uncompressed_state_sha256']:
            raise ValueError('Uncompressed fitted state changed')
        model=pickle.loads(payload)
        before=state_hash(model)
        started_wall,started_cpu=time.perf_counter(),time.process_time()
        result=model.predict(x)
        resources[name]={'wall_seconds':time.perf_counter()-started_wall,
                         'cpu_seconds':time.process_time()-started_cpu,
                         'nodes':result['nodes']}
        after=state_hash(model)
        if before!=after: raise AssertionError('Prediction mutated fitted state')
        state_identity[name]={'before_sha256':before,'after_sha256':after,'unchanged':True}
        if name=='shared':shared_result=result
        else:candidate_results[name]=result
        diagnostics[name]=result['diagnostics']
    view_nodes={}
    predictions=prediction_views(candidate_results,shared_result,fit['contexts'],fit['matched_candidate_id'],nodes=view_nodes)
    for value in predictions.values():
        if value.shape!=(len(x),3) or not np.isfinite(value).all():
            raise FloatingPointError('Incomplete finite prediction roster')
    for left,right in (('historical2','native2'),('native12','native20')):
        for mode in ('calibrated','gamma1'):
            for method in METHODS:
                np.testing.assert_array_equal(predictions[f'{left}__{mode}__{method}'],
                                              predictions[f'{right}__{mode}__{method}'])
    write_npz(output/'predictions.npz',predictions)
    summary={'schema':'kahm-dimension-predict-v1','status':'completed','task':fit['task'],
        'completed_utc':utcnow(),'fit_lock_sha256':fit_lock_sha256,
        'fit_summary_sha256':sha256_file(fit_dir/'fit_summary.json'),
        'query_file_sha256':sha256_file(query_path),'D_X_sha256':array_hash(x),
        'D_rows':len(x),'horizon_count':3,'prediction_array_count':len(predictions),
        'prediction_array_sha256':{key:array_hash(value) for key,value in predictions.items()},
        'prediction_file_sha256':sha256_file(output/'predictions.npz'),
        'state_identity':state_identity,'diagnostics':diagnostics,
        'procedure_resources':query_procedure_resources(resources,fit['contexts'],view_nodes),
        'resources':{'per_candidate':resources,'total_wall_seconds':time.perf_counter()-wall,
                     'total_cpu_seconds':time.process_time()-cpu,
                     'scope':'All7 fixed banks/five geometry heads and shared raw competitors plus views/diagnostic/state identity; no fits or selection'},
        'D_target_file_opened':False}
    write_json(output/'predict_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('fit','predict'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--freeze-file',type=Path)
    parser.add_argument('--fit-lock',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0): raise ValueError('Requested CPU is not allowed')
        os.sched_setaffinity(0,{args.cpu})
    config=json.loads((root/'config.json').read_text())
    declared=tuple((r['candidate_id'],r['rank_cap'],r['dimension_policy']) for r in config['candidates'])
    if declared!=CANDIDATES or tuple(config['methods'])!=FULL_METHODS or tuple(config['native_candidate_order'])!=NATIVE_ORDER:
        raise ValueError('Worker method/candidate roster differs from frozen configuration')
    if args.freeze_file is None:raise ValueError('Every production stage requires the source freeze')
    verify_freeze(root,args.freeze_file)
    with threadpool_limits(limits=1):
        if args.mode=='fit':
            if args.freeze_file is None or not args.freeze_file.is_file():
                raise ValueError('Production fitting requires the root source/config freeze')
            manifest=verify_abc_manifest(root,args.freeze_file,hash_inputs=False)
            matches=[t for t in manifest['tasks'] if t['task_id']==args.task_id]
            if len(matches)!=1: raise ValueError('Task is missing or duplicated')
            task=matches[0];input_path=root/task['input']
            if sha256_file(input_path)!=task['sha256']: raise ValueError('ABC input identity mismatch')
            fit_task(input_path,task,root/'results'/args.task_id,model_config=config.get('model'),
                     freeze_sha256=sha256_file(args.freeze_file))
        else:
            if args.fit_lock is None or not args.fit_lock.is_file():
                raise ValueError('D prediction requires the complete fitted-state lock')
            verify_fit_lock(root,args.fit_lock,args.freeze_file,full_hash=False,task_id=args.task_id)
            manifest=json.loads((root/'prepared/d_manifest.json').read_text())
            if manifest['fit_lock_sha256']!=sha256_file(args.fit_lock):
                raise ValueError('D input provenance differs from fitted-state lock')
            matches=[t for t in manifest['tasks'] if t['task_id']==args.task_id]
            if len(matches)!=1: raise ValueError('D query task is missing or duplicated')
            task=matches[0];query_path=root/task['query']
            if sha256_file(query_path)!=task['query_sha256']: raise ValueError('D query identity mismatch')
            predict_task(root/'results'/args.task_id,query_path,root/'predictions'/args.task_id,
                         fit_lock_sha256=sha256_file(args.fit_lock))
    print(json.dumps({'status':'completed','mode':args.mode,'task_id':args.task_id}),flush=True)


if __name__=='__main__':main()
