"""Write-once Step19 follow-up using only sealed Step18 evidence.

No waveform archive access, new predictions, fitting, or selection is available.
Official decisions are made by the separately frozen exact postverifier.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS',
              'BLIS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT'):
    os.environ[_name]='1'
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import scipy
import core19 as K

HERE=Path(__file__).resolve().parent
PROJ=HERE.parent
BASE=PROJ.parent
PARENT=BASE/'step18_natural_decisions/study/run'
RUN=HERE/'run'
TESTS=PROJ/'tests'
PREFIXES=(2,6,14)
POPULATION=8192


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def dumpj(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False,default=default);f.write('\n')


def npz(path,**arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f,**arrays)


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def configuration():
    return dict(schema='step19-producer-configuration-v1',prefixes=PREFIXES,
        population=POPULATION,row_order='Step18 cal then terminal; each existing order unchanged',
        kappa_by_prefix={str(p):1. for p in PREFIXES},base_block_size=K.BASE_BLOCK,
        families=K.FAMILIES,block_sizes=K.LEVELS,chunk_blocks=K.CHUNK_BLOCKS,
        solver=K.SOLVER,configurations=18,total_new_block_programs=2688,
        initialization='zero for every new optimizer; embedded children evaluated separately',
        larger_blocks='concatenate adjacent base groups; never rerun greedy at larger sizes',
        guarded_choice='best corrected SSE per block, new optimizer on ties',
        exact_choice='official postverifier recursively selects max of zero,new,exact-best embedded children',
        guard_scope='diagnostic only; retain explicit previous-level valid empirical lower',
        cross_family='maximum, never sum overlapping partitions',
        target_bounds=(-1.,1.),statistical_confidence_events=0,
        no_waveform_archive_access=True,no_model_fit_or_selection=True)


def producer_source_files():
    return [Path(__file__).resolve(),HERE/'check19.py',*K.source_files()]


def source_files():
    return producer_source_files()+[PROJ/'PROTOCOL.json',HERE/'exact19.py',HERE/'audit19.py',
        HERE/'check_exact19.py',TESTS/'preflight19.json',TESTS/'exact_preflight19.json',
        HERE/'check_theory19.py',TESTS/'theory_preflight19.json',
        PROJ/'manuscript/step19_theory.tex']


def input_files():
    paths=[PARENT/'freeze.json',PARENT/'fit_seal.json',PARENT/'cal_seal.json',
        PARENT/'terminal_seal.json',PARENT/'census_seal.json',
        PARENT/'fit_validation.json',PARENT/'cal_validation.json',PARENT/'verification.json',
        PARENT/'fit/selected.json',PARENT/'fit/selected_models.pkl',PARENT/'census/results.json']
    for stage in ('cal','terminal'):
        paths.extend(PARENT/stage/name for name in ('observations.npz','features.npz','predictions.npz'))
    return paths


def hash_map(paths):
    return {str(Path(p).resolve().relative_to(BASE)):sha(p) for p in sorted(set(paths))}


def verify_protocol():
    p=read(PROJ/'PROTOCOL.json');schedule=p['constraint_schedule'];opt=p['optimization']
    if p['population']['size']!=POPULATION or p['frozen_models']['state_coordinate_counts']!=list(PREFIXES):
        raise ArithmeticError('Protocol population/classes differ from producer')
    expected=dict(base_block_size=K.BASE_BLOCK,base_families=list(K.FAMILIES),
        block_sizes=list(K.LEVELS),number_configurations=18,number_new_block_programs=2688)
    if any(schedule[k]!=v for k,v in expected.items()) or schedule['run_all_configurations'] is not True:
        raise ArithmeticError('Protocol constraint schedule differs from producer')
    if any(opt[k]!=v for k,v in K.SOLVER.items()) or opt['chunk_blocks']!=K.CHUNK_BLOCKS:
        raise ArithmeticError('Protocol optimizer controls differ from producer')
    if p['exact_postverification']['coordinate_grid_power']!=28:
        raise ArithmeticError('Unexpected frozen exact coordinate grid')
    return p


def verify_parent_metadata():
    protocol=verify_protocol();parent_freeze=sha(PARENT/'freeze.json')
    if parent_freeze!=protocol['parent_step18_freeze_sha256']:
        raise ArithmeticError('Parent freeze provenance mismatch')
    if sha(PARENT/'fit/selected_models.pkl')!=protocol['parent_step18_selected_models_sha256']:
        raise ArithmeticError('Frozen selected parent model artifact changed')
    for gate in ('fit_validation.json','cal_validation.json','verification.json'):
        g=read(PARENT/gate)
        if g['passed'] is not True or g['freeze_sha256']!=parent_freeze:
            raise ArithmeticError('Parent independent audit failed or mismatched')
    wanted={str(p.relative_to(PARENT)):p for p in input_files() if len(p.relative_to(PARENT).parts)>1}
    for stage in ('fit','cal','terminal','census'):
        seal=read(PARENT/(stage+'_seal.json'))
        if seal['freeze_sha256']!=parent_freeze:
            raise ArithmeticError('Parent stage seal freeze mismatch')
        for name,p in wanted.items():
            if name.startswith(stage+'/') and seal['files'].get(name)!=sha(p):
                raise ArithmeticError('A required parent input disagrees with its sealed hash')
    parent_sources=read(PARENT/'freeze.json')['sources']
    for p in (Path(K.Q.__file__).resolve(),Path(K.S.__file__).resolve()):
        if parent_sources.get(str(p.relative_to(BASE)))!=sha(p):
            raise ArithmeticError('Reused Step18 scientific routine changed')
    return protocol


def validate_preflight(path,required_sources):
    receipt=read(path)
    if receipt.get('passed') is not True:
        raise ArithmeticError('Required manufactured preflight did not pass')
    source_map=receipt.get('source_sha256',receipt.get('producer_sha256'))
    if not isinstance(source_map,dict):
        raise ArithmeticError('Preflight must identify its tested source hashes')
    for p in required_sources:
        name=str(Path(p).resolve().relative_to(BASE))
        if source_map.get(name)!=sha(p):
            raise ArithmeticError('Preflight producer/source hash mismatch: '+name)
    return receipt


def freeze():
    verify_parent_metadata()
    validate_preflight(TESTS/'preflight19.json',producer_source_files())
    validate_preflight(TESTS/'exact_preflight19.json',[HERE/'exact19.py',HERE/'audit19.py',HERE/'check_exact19.py'])
    theory=read(TESTS/'theory_preflight19.json')
    if not (theory.get('passed') is True
            and theory['test_source_sha256']==sha(HERE/'check_theory19.py')
            and theory['theory_source_sha256']==sha(PROJ/'manuscript/step19_theory.tex')):
        raise ArithmeticError('Theory manufactured preflight/source hash mismatch')
    RUN.mkdir()
    dumpj(RUN/'freeze.json',dict(schema='step19-freeze-v1',frozen=utc(),
        sources=hash_map(source_files()),inputs=hash_map(input_files()),configuration=configuration(),
        protocol_sha256=sha(PROJ/'PROTOCOL.json'),parent_step18_freeze_sha256=sha(PARENT/'freeze.json'),
        parent_gate_sha256={n:sha(PARENT/n) for n in ('fit_validation.json','cal_validation.json','verification.json')},
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),
        no_natural_duals_computed_before_freeze=True))


def check_freeze():
    frozen=read(RUN/'freeze.json')
    if frozen['sources']!=hash_map(source_files()) or frozen['inputs']!=hash_map(input_files()):
        raise ArithmeticError('Frozen source/input closure changed')
    if frozen['configuration']!=json.loads(json.dumps(configuration())):
        raise ArithmeticError('Frozen producer configuration changed')
    verify_parent_metadata()
    return frozen


def load_parent_inputs():
    """The sole scientific-array loader; invoked only by the frozen run stage."""
    observations={s:load_npz(PARENT/s/'observations.npz') for s in ('cal','terminal')}
    features={s:load_npz(PARENT/s/'features.npz') for s in ('cal','terminal')}
    predictions={s:load_npz(PARENT/s/'predictions.npz') for s in ('cal','terminal')}
    selected=read(PARENT/'fit/selected.json');census=read(PARENT/'census/results.json')
    arrays={}
    for key in ('Y','patient_id','ecg_id','origin'):
        arrays[key]=np.concatenate([observations[s][key] for s in ('cal','terminal')])
    if arrays['Y'].shape!=(POPULATION,) or len(set(arrays['patient_id'].tolist()))!=POPULATION:
        raise ArithmeticError('Parent cohort does not have one row per unique patient')
    if not np.isfinite(arrays['Y']).all() or np.any(abs(arrays['Y'])>1):
        raise ArithmeticError('Stored target contract failed')
    kappas={}
    for p in map(str,PREFIXES):
        kappas[p]=float(selected['kahm'][p]['kappa'])
        if kappas[p]!=1.:
            raise ArithmeticError('Selected regularity budget changed')
        for field in ('F','F_lo','F_hi','block_radius'):
            key=field+'_'+p
            arrays[key]=np.concatenate([features[s][key] for s in ('cal','terminal')])
        for field in ('pred','prediction_lo','prediction_hi','loss_lower','loss_upper'):
            key=field+'_'+p
            arrays[key]=np.concatenate([predictions[s][key] for s in ('cal','terminal')])
        f,lo,hi=arrays['F_'+p],arrays['F_lo_'+p],arrays['F_hi_'+p]
        if f.shape!=(POPULATION,int(p)) or not np.isfinite(f).all() or np.any(lo>f) or np.any(f>hi):
            raise ArithmeticError('Inherited canonical feature box mismatch')
        rad=arrays['block_radius_'+p]
        if rad.shape!=(POPULATION,len(K.feature_slices(int(p)))) or not np.isfinite(rad).all() or np.any(rad<0):
            raise ArithmeticError('Inherited feature radius mismatch')
    reference_id=selected['reference']['id'];reference_slot=selected['reference']['selection_slot']
    reference=[]
    for stage in ('cal','terminal'):
        names=predictions[stage]['baseline_candidate_ids'].tolist()
        i=names.index(reference_id)
        if predictions[stage]['baseline_slots'][i]!=reference_slot:
            raise ArithmeticError('Frozen reference column mismatch')
        reference.append(predictions[stage]['baseline_predictions'][:,i])
    arrays['reference_pred']=np.concatenate(reference)
    metadata=dict(population_size=POPULATION,row_order=configuration()['row_order'],
        selected_kappa=kappas,selected_candidate_ids={p:selected['kahm'][p]['id'] for p in kappas},
        feature_blocks={p:[(s.start,s.stop) for s in K.feature_slices(int(p))] for p in kappas},
        selected_geometry={p:selected['kahm'][p]['geometry'] for p in kappas},
        reference_id=reference_id,reference_slot=reference_slot,
        reference_mse=census['reference']['mse'],reference_mse_lower=census['reference']['mse_lower'],
        reference_mse_upper=census['reference']['mse_upper'],
        threshold=census['threshold'],threshold_lower=census['threshold_lower'],threshold_upper=census['threshold_upper'],
        fitted_canonical_mse_upper={str(r['prefix']):r['mse_upper'] for r in census['rows'] if r['family']=='kahm'},
        no_model_change=True,no_new_prediction=True,no_reserved_patient_access=True,
        input_file_sha256=hash_map(input_files()))
    return arrays,metadata


def peak_rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(process_peak_rss_raw=int(value),raw_units='KiB' if sys.platform.startswith('linux') else 'platform-dependent',
                process_peak_rss_bytes=int(value*1024) if sys.platform.startswith('linux') else None,
                scope='process lifetime high-water mark, not isolated stage allocation')


def run_level(prefix,family,block_size,indices,mapping,arrays,metadata,previous=None):
    out=RUN/'results'/f'p{prefix}'/family/f'b{block_size}';out.mkdir()
    npz(out/'nesting.npz',indices=indices,base_group_ids=mapping['base_group_ids'],child_block_ids=mapping['child_block_ids'])
    count=len(indices);slices=K.feature_slices(prefix);p=str(prefix)
    chunk_records=[];all_duals=[];all_lowers=[]
    started=time.perf_counter();rss_before=peak_rss()
    for chunk,start in enumerate(range(0,count,K.CHUNK_BLOCKS)):
        stop=min(count,start+K.CHUNK_BLOCKS);local=indices[start:stop]
        children=None
        if previous is not None:
            child_ids=mapping['child_block_ids'][start:stop]
            children=previous['chosen_signed_dual'][child_ids]
        evidence,stats=K.compute_chunk(arrays['F_'+p],arrays['Y'],arrays['block_radius_'+p],local,
            metadata['selected_kappa'][p],slices,children=children,child_size=mapping['child_size'],
            predictions=arrays['pred_'+p])
        evidence.update(block_ids=np.arange(start,stop,dtype=np.int64),
            base_group_ids=mapping['base_group_ids'][start:stop],
            child_block_ids=mapping['child_block_ids'][start:stop])
        if previous is not None:
            evidence['child_guarded_sse_lower']=previous['chosen_sse_lower'][child_ids]
            evidence['child_guarded_sse_sum']=np.nextafter(np.asarray(np.sum(
                evidence['child_guarded_sse_lower'].astype(np.longdouble),axis=1),dtype=float),0.)
            evidence['embedded_guarded_minus_child_sum']=evidence['embedded_sse_lower']-evidence['child_guarded_sse_sum']
        path=out/f'chunk{chunk:03d}.npz'
        before=time.perf_counter();npz(path,**evidence);saving=time.perf_counter()-before
        stats.update(prefix=prefix,family=family,chunk=chunk,start_block=start,stop_block=stop,
            kappa=metadata['selected_kappa'][p],child_size=mapping['child_size'],
            child_level=None if previous is None else f'b{mapping["child_size"]}',
            saved_array_bytes=sum(int(a.nbytes) for a in evidence.values()),
            artifact_bytes=path.stat().st_size,artifact_sha256=sha(path),saving_seconds=saving,
            field_names=sorted(evidence),peak_memory=peak_rss())
        dumpj(out/f'chunk{chunk:03d}.json',stats)
        chunk_records.append(stats)
        all_duals.append(evidence['chosen_signed_dual']);all_lowers.append(evidence['chosen_sse_lower'])
        print(f'prefix={prefix} family={family} block={block_size} chunk={chunk+1}/{(count+K.CHUNK_BLOCKS-1)//K.CHUNK_BLOCKS}',flush=True)
    chosen=np.concatenate(all_duals);lower=np.concatenate(all_lowers)
    current=K.guarded_empirical_lower(lower,POPULATION)
    previous_lower=0. if previous is None else previous['summary']['retained_guarded_empirical_lower']
    retained=max(current,previous_lower)
    summary=dict(prefix=prefix,family=family,block_size=block_size,blocks=count,rows=POPULATION,
        chunk_blocks=K.CHUNK_BLOCKS,chunks=len(chunk_records),kappa=metadata['selected_kappa'][p],
        current_chosen_witness_empirical_lower=current,retained_previous_level_empirical_lower=previous_lower,
        retained_guarded_empirical_lower=retained,retained_previous_value_active=previous_lower>current,
        guarded_comparison_to_frozen_threshold=retained>metadata['threshold_upper'],
        official_decision_pending_exact_postverification=True,
        new_block_programs=count,complete_undirected_edges=count*block_size*(block_size-1)//2,
        one_sided_inequalities=count*block_size*(block_size-1),
        selected_embedded_blocks=sum(r['chosen_embedded_count'] for r in chunk_records),
        new_solver_iteration_limit_blocks=sum(r['new_solver_iteration_limits'] for r in chunk_records),
        chosen_gap_limit_blocks=sum(r['chosen_iteration_gap_limits'] for r in chunk_records),
        maximum_chosen_gap=max(r['maximum_chosen_gap'] for r in chunk_records),
        new_optimization_seconds=sum(r['new_optimization_seconds'] for r in chunk_records),
        new_radius_correction_seconds=sum(r['new_radius_correction_seconds'] for r in chunk_records),
        embedded_radius_correction_seconds=sum(r['embedded_radius_correction_seconds'] for r in chunk_records),
        child_embedding_seconds=sum(r['child_embedding_seconds'] for r in chunk_records),
        saved_signed_witness_bytes=sum(r['new_signed_witness_bytes']+r['embedded_signed_witness_bytes']+r['chosen_signed_witness_bytes'] for r in chunk_records),
        saved_array_bytes=sum(r['saved_array_bytes'] for r in chunk_records),
        compressed_chunk_bytes=sum(r['artifact_bytes'] for r in chunk_records),
        wall_seconds=time.perf_counter()-started,memory_before=rss_before,memory_after=peak_rss(),
        nesting_sha256=sha(out/'nesting.npz'),
        chunks_metadata=[dict(chunk=r['chunk'],start_block=r['start_block'],stop_block=r['stop_block'],artifact_sha256=r['artifact_sha256']) for r in chunk_records],
        gap_scope='relaxed block program; not a gap to the full canonical class',
        witness_scope='guarded selection diagnostic; exact recursive choices may differ')
    dumpj(out/'summary.json',summary)
    return dict(chosen_signed_dual=chosen,chosen_sse_lower=lower,summary=summary)


def producer_files():
    required=[RUN/'inputs.npz',RUN/'input_manifest.json',RUN/'summary.json']
    required.extend(p for p in (RUN/'results').rglob('*') if p.is_file())
    return sorted(required)


def seal_producer():
    dumpj(RUN/'producer_seal.json',dict(freeze_sha256=sha(RUN/'freeze.json'),sealed=utc(),
        files={str(p.relative_to(RUN)):sha(p) for p in producer_files()},
        scope='producer inputs, partitions, witnesses and summary only; exact outputs are separately sealed'))


def check_producer_seal():
    record=read(RUN/'producer_seal.json')
    if record['freeze_sha256']!=sha(RUN/'freeze.json') or record['files']!={str(p.relative_to(RUN)):sha(p) for p in producer_files()}:
        raise ArithmeticError('Producer seal mismatch')
    return record


def run():
    check_freeze()
    if (RUN/'results').exists() or (RUN/'inputs.npz').exists():
        raise FileExistsError('Scientific run is write-once')
    started=utc();before=time.perf_counter()
    arrays,metadata=load_parent_inputs()
    npz(RUN/'inputs.npz',**arrays)
    metadata.update(input_arrays_sha256=sha(RUN/'inputs.npz'),freeze_sha256=sha(RUN/'freeze.json'),
                    array_shapes={k:list(v.shape) for k,v in arrays.items()})
    dumpj(RUN/'input_manifest.json',metadata)
    (RUN/'results').mkdir();summaries=[];partitions=[]
    for prefix in PREFIXES:
        root=RUN/'results'/f'p{prefix}';root.mkdir()
        for family in K.FAMILIES:
            folder=root/family;folder.mkdir()
            partition,maps,elapsed=K.nested_partitions(arrays['F_'+str(prefix)],family)
            npz(folder/'base_partition.npz',indices=partition[K.BASE_BLOCK])
            part=dict(prefix=prefix,family=family,base_block_size=K.BASE_BLOCK,rows=POPULATION,
                base_partition_seconds=elapsed,base_partition_sha256=sha(folder/'base_partition.npz'),
                larger_partitions_are_adjacent_group_merges=True)
            dumpj(folder/'partition.json',part);partitions.append(part)
            previous=None
            for b in K.LEVELS:
                previous=run_level(prefix,family,b,partition[b],maps[b],arrays,metadata,previous)
                summaries.append(previous['summary'])
    if len(summaries)!=18 or sum(r['new_block_programs'] for r in summaries)!=2688:
        raise ArithmeticError('Not every frozen configuration/program was run')
    results=dict(schema='step19-producer-results-v1',started=started,finished=utc(),
        seconds=time.perf_counter()-before,configuration=configuration(),configurations=summaries,
        partitions=partitions,total_new_block_programs=sum(r['new_block_programs'] for r in summaries),
        total_undirected_edges_with_repetition=sum(r['complete_undirected_edges'] for r in summaries),
        total_one_sided_inequalities_with_repetition=sum(r['one_sided_inequalities'] for r in summaries),
        total_saved_signed_witness_bytes=sum(r['saved_signed_witness_bytes'] for r in summaries),
        total_base_partition_seconds=sum(r['base_partition_seconds'] for r in partitions),
        total_new_optimization_seconds=sum(r['new_optimization_seconds'] for r in summaries),
        total_radius_correction_seconds=sum(r['new_radius_correction_seconds']+r['embedded_radius_correction_seconds'] for r in summaries),
        total_child_embedding_seconds=sum(r['child_embedding_seconds'] for r in summaries),
        new_solver_iteration_limit_blocks=sum(r['new_solver_iteration_limit_blocks'] for r in summaries),
        memory=peak_rss(),official_decisions_pending_exact_postverification=True,
        no_sampling_conversion_used_for_decisions=True,no_model_change=True,no_new_patients=True)
    dumpj(RUN/'summary.json',results)
    seal_producer()
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('preflight','freeze','run'))
    stage=parser.parse_args().stage
    if stage=='preflight':
        import check19
        check19.main()
    elif stage=='freeze':freeze()
    else:run()


if __name__=='__main__':main()

