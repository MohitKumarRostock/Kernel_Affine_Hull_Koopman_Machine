#!/usr/bin/env python3
"""Generate ABC first; D is inaccessible until the complete fitted-state lock.

All numerical observation-law primitives are imported from unmodified pinned
sources. New code supplies declared paired streams, width prefixes and staging.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import argparse
from datetime import datetime,timezone
import importlib
import json
from pathlib import Path
import numpy as np

from study_contract import seed_records,task_roster,verify_freeze,verify_fit_lock,verify_abc_manifest,digest
from dimension_worker import write_npz,write_json
from focused_dimension_model import array_hash

SOURCE_ROOT=Path(__file__).resolve().parents[1]


def generator_module():
    original=SOURCE_ROOT/'vendor_original'
    manifest=json.loads((original/'GENERATOR_SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        if digest(original/row['path'])!=row['sha256']:raise ValueError('Pinned generator identity changed')
    sys.path.insert(0,str(original))
    module=importlib.import_module('extensions.reconstruction_mechanism.data')
    if Path(module.__file__).resolve()!=original/'extensions/reconstruction_mechanism/data.py':
        raise ImportError('Generator and forecasting runtime namespaces must use separate processes')
    return module


def _streams(records,repetition,law,split):
    return {r['role']:r for r in records if r.get('repetition')==repetition
            and r.get('law')==law and r.get('split')==split}


def _base(config,law,split,streams,module):
    n=config['split_episodes'][split];length=config['max_episode_length']
    seed=streams['generation']['seed']
    episodes=(module.linear_episodes(n,length,seed,config['linear_ar']) if law=='linear_ar'
              else module.generate_episodes(law,n,length,seed)[:,:,0])
    noise=np.random.default_rng(streams['unit_noise']['seed']).standard_normal(episodes.shape)
    if not np.isfinite(episodes).all() or not np.isfinite(noise).all():raise FloatingPointError('Nonfinite generated episode')
    return episodes,noise


def _windows(config,base,unit,history,level,amplitude,split,streams,module):
    length=history+config['history_tail_length']
    augmented=base[:,:length]+level*amplitude*unit[:,:length]
    windows=module.make_windows(augmented,history,config['horizons'],config['stride'])
    if len(windows['X'])!=config['split_episodes'][split]*config['origins_per_episode']:
        raise AssertionError('Origin count differs from frozen design')
    cap=config['split_caps'][split];n=len(windows['X'])
    retained=(np.arange(n,dtype=np.int64) if cap is None or cap>=n else
              np.sort(np.random.default_rng(streams['rows']['seed']).choice(n,cap,replace=False)))
    return {key:np.ascontiguousarray(value[retained]) for key,value in windows.items()},retained


def prepare_abc(root,freeze_file):
    root=Path(root).resolve();verify_freeze(root,freeze_file)
    if (root/'prepared/abc_manifest.json').exists():raise FileExistsError('ABC manifest already exists')
    if (root/'prepared/d_manifest.json').exists() or (root/'prepared/targets').exists():
        raise ValueError('D materialization already exists before ABC preparation')
    config=json.loads((root/'config.json').read_text());module=generator_module()
    records=seed_records(config);tasks=task_roster(config);prepared=[];base_files={}
    write_json(root/'prepared/seed_manifest.json',{'distinct_seed_count':len(records),'records':records,
        'collision_check':'passed','D_streams_checked_but_not_generated':True})
    for repetition in range(config['repetitions']):
        for law in config['laws']:
            bases={};units={};streams={};saved={}
            for split in ('A','B','C'):
                streams[split]=_streams(records,repetition,law,split)
                bases[split],units[split]=_base(config,law,split,streams[split],module)
                saved[split+'_base_episodes']=bases[split];saved[split+'_unit_noise']=units[split]
            relative=f'prepared/base_abc/{law}__rep{repetition:02d}.npz'
            write_npz(root/relative,saved);base_files[relative]=digest(root/relative)
            for task in tasks:
                if task['repetition']!=repetition or task['law']!=law:continue
                history=task['history'];level=task['noise_level']
                amplitude=float(np.std(bases['A'][:,:history+config['history_tail_length']],ddof=0))
                arrays={'offsets':np.asarray(config['horizons'],dtype=np.int64),
                        'A_base_episode_std':np.asarray(amplitude),'noise_std':np.asarray(level*amplitude)}
                split_records={}
                for split in ('A','B','C'):
                    windows,retained=_windows(config,bases[split],units[split],history,level,amplitude,
                                              split,streams[split],module)
                    arrays.update({f'{split}_{key}':value for key,value in windows.items()})
                    split_records[split]={'streams':streams[split],
                        'retained_row_indices_sha256':array_hash(retained),
                        'base_episode_sha256':array_hash(bases[split]),
                        'unit_noise_sha256':array_hash(units[split]),
                        'episodes':config['split_episodes'][split],'retained_rows':len(retained)}
                arrays['loss_scale']=np.maximum(np.var(arrays['A_Y'],axis=0,ddof=0),
                                                 config['model']['target_variance_floor'])
                relative_input=f"prepared/abc/{task['task_id']}.npz"
                write_npz(root/relative_input,arrays)
                prepared.append({**task,'input':relative_input,'sha256':digest(root/relative_input),
                    'base_file':relative,'base_file_sha256':base_files[relative],
                    'A_base_episode_std':amplitude,'noise_std':level*amplitude,
                    'loss_scale':arrays['loss_scale'].tolist(),'splits':split_records,
                    'array_sha256':{key:array_hash(value) for key,value in arrays.items()}})
        print(json.dumps({'stage':'prepare_abc','repetition':repetition,'completed_tasks':len(prepared)}),flush=True)
    manifest={'schema':'kahm-dimension-abc-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'config_sha256':digest(root/'config.json'),'freeze_sha256':digest(freeze_file),
        'task_count':len(prepared),'tasks':prepared,'base_files':base_files,
        'seed_manifest_sha256':digest(root/'prepared/seed_manifest.json'),
        'D_episodes_generated':False,'D_targets_materialized':False}
    write_json(root/'prepared/abc_manifest.json',manifest)
    return manifest


def prepare_d(root,freeze_file,fit_lock):
    root=Path(root).resolve();verify_freeze(root,freeze_file)
    verify_fit_lock(root,fit_lock,freeze_file,full_hash=True)
    abc=verify_abc_manifest(root,freeze_file,hash_inputs=True)
    if (root/'prepared/d_manifest.json').exists():raise FileExistsError('D manifest already exists')
    config=json.loads((root/'config.json').read_text());module=generator_module();records=seed_records(config)
    prepared=[];base_files={}
    for repetition in range(config['repetitions']):
        for law in config['laws']:
            streams=_streams(records,repetition,law,'D')
            base,unit=_base(config,law,'D',streams,module)
            relative_base=f'prepared/withheld_d_base/{law}__rep{repetition:02d}.npz'
            write_npz(root/relative_base,{'D_base_episodes':base,'D_unit_noise':unit})
            base_files[relative_base]=digest(root/relative_base)
            for task in abc['tasks']:
                if task['repetition']!=repetition or task['law']!=law:continue
                windows,retained=_windows(config,base,unit,task['history'],task['noise_level'],
                    task['A_base_episode_std'],'D',streams,module)
                query=f"prepared/queries/{task['task_id']}.npz"
                target=f"prepared/targets/{task['task_id']}.npz"
                shared={f'D_{key}':value for key,value in windows.items() if key not in ('X','Y')}
                write_npz(root/query,{'D_X':windows['X'],**shared})
                write_npz(root/target,{'D_Y':windows['Y'],**shared})
                prepared.append({key:task[key] for key in ('task_id','law','law_index','history','seed',
                    'repetition','noise_index','noise_level','fit_seed')} | {
                    'query':query,'query_sha256':digest(root/query),
                    'target':target,'target_sha256':digest(root/target),
                    'D_X_sha256':array_hash(windows['X']),'D_Y_sha256':array_hash(windows['Y']),
                    'D_groups_sha256':array_hash(windows['groups']),
                    'base_file':relative_base,'base_file_sha256':base_files[relative_base],
                    'streams':streams,'retained_row_indices_sha256':array_hash(retained),
                    'A_base_episode_std':task['A_base_episode_std'],'rows':len(retained)})
        print(json.dumps({'stage':'prepare_d','repetition':repetition,'completed_tasks':len(prepared)}),flush=True)
    manifest={'schema':'kahm-dimension-d-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'config_sha256':digest(root/'config.json'),'freeze_sha256':digest(freeze_file),
        'fit_lock_sha256':digest(fit_lock),'ABC_manifest_sha256':digest(root/'prepared/abc_manifest.json'),
        'tasks':prepared,'task_count':len(prepared),'base_files':base_files,
        'generation_after_complete_fit_lock':True,'targets_reserved_until_global_prediction_lock':True}
    write_json(root/'prepared/d_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('abc','d'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--freeze-file',type=Path,required=True)
    parser.add_argument('--fit-lock',type=Path)
    args=parser.parse_args()
    if args.stage=='abc':prepare_abc(args.root,args.freeze_file)
    else:
        if args.fit_lock is None:raise ValueError('D requires complete fit lock')
        prepare_d(args.root,args.freeze_file,args.fit_lock)


if __name__=='__main__':main()
