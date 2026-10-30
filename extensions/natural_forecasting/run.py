"""Reproducible staged runner with frozen rosters and isolated cost measurement.

Run from this directory with a single BLAS thread. `prepare` never evaluates a
model. `run --role confirmation` refuses to proceed without an intact final
freeze. Every method executes in a fresh child, sequentially; its complete
search/refit/forecast cost and process peak RSS are recorded.
"""
from __future__ import annotations

import time
_IMPORT_STARTED = time.perf_counter()
_IMPORT_CPU_STARTED = time.process_time()
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from contextlib import contextmanager
from dataclasses import replace
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import platform
import resource
import sys
import time
import traceback
import warnings

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

if __package__:
    from . import data, models, baselines
else:
    import data, models, baselines

_IMPORT_WALL_SECONDS = time.perf_counter()-_IMPORT_STARTED
_IMPORT_CPU_SECONDS = time.process_time()-_IMPORT_CPU_STARTED

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / 'runs' / 'natural_v1'
CODE_FILES = ('run.py', 'data.py', 'models.py', 'baselines.py', 'analysis.py',
              'protocol.json', 'data_sources.json', 'requirements.txt')
GEOMETRIC = {
    'raw_ridge': ('raw', 'ridge', False),
    'raw_rbf': ('raw', 'rbf', False),
    'affine_rbf': ('affine_local', 'rbf', False),
    'locality_rbf': ('locality_local', 'rbf', False),
    'kahm_rbf': ('kahm_local', 'rbf', False),
    'raw_affine_rbf': ('affine_local', 'rbf', True),
    'raw_locality_rbf': ('locality_local', 'rbf', True),
    'raw_kahm_rbf': ('kahm_local', 'rbf', True),
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def code_hashes():
    result = {name: digest_file(ROOT/name) for name in CODE_FILES}
    historical = ROOT.parent/'predictive_geometry'/'study'
    for name in ('geometry.py', 'heads.py'):
        result['../predictive_geometry/study/'+name] = digest_file(historical/name)
    for path in sorted((historical/'vendor'/'upstream').glob('*.py')):
        result['../predictive_geometry/study/vendor/upstream/'+path.name] = digest_file(path)
    return result


def environment():
    packages = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'statsmodels', 'pandas',
                 'psutil', 'threadpoolctl'):
        packages[name] = importlib.metadata.version(name)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': packages, 'visible_cpus': os.cpu_count(), 'cpu_affinity': affinity,
            'threadpools': threadpool_info(), 'benchmark_threads': 1,
            'shared_library_import_wall_seconds': _IMPORT_WALL_SECONDS,
            'shared_library_import_cpu_seconds': _IMPORT_CPU_SECONDS,
            'import_scope': 'all declared method libraries loaded once before benchmarking; reported separately from warm-library scientific pipeline costs',
            'method_execution': 'one fresh fork child at a time on Linux; spawn elsewhere',
            'memory_measure': 'child process ru_maxrss, includes interpreter and inherited input arrays',
            'timing_measure': 'perf_counter wall seconds and process_time CPU seconds; imports/setup separated'}


def load_protocol():
    return json.loads((ROOT/'protocol.json').read_text())


def run_identifier(base_task_id, budget, seed):
    return f'{base_task_id}__{budget}__seed{seed}'


def cache_name(base_task_id, budget):
    return hashlib.sha256(f'{base_task_id}|{budget}'.encode()).hexdigest()[:24]


def prepare(run_root=DEFAULT_RUN):
    """Fix source/series/task identity and arrays before comparative outcomes."""
    run_root = Path(run_root)
    if (run_root/'development').exists() or (run_root/'confirmation').exists():
        raise RuntimeError('Cannot prepare/replace a roster after model execution')
    protocol = load_protocol()
    manifest = data.load_manifest()
    roster, preparation = [], []
    for budget, cap in protocol['capacity_regimes'].items():
        for source in manifest['sources']:
            started, cpu = time.perf_counter(), time.process_time()
            tasks = data.load_tasks(role=source['role'], source_ids=[source['source_id']],
                max_train_windows=cap['max_train_windows'],
                max_eval_windows=protocol['data']['max_eval_origins'],
                max_series=protocol['data']['max_series_per_source'])
            duration, process_cpu = time.perf_counter()-started, time.process_time()-cpu
            preparation.append({'source_id': source['source_id'], 'budget_id': budget,
                'seconds': duration, 'cpu_seconds': process_cpu,
                'base_task_count': len(tasks),
                'scope': 'source bytes/checksums/parsing, training-only series selection and window construction'})
            for task in tasks:
                cache_id = cache_name(task.task_id, budget)
                cache_path = run_root/'prepared'/f'{cache_id}.npz'
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Refit uses every eligible pretest window before deterministic
                # cap selection, including windows straddling the old train/val
                # cut, because their targets are now legitimately historical.
                started_refit, cpu_refit = time.perf_counter(), time.process_time()
                refit = data.make_windows(task.values, task.context, task.horizon,
                    0, task.validation_end, cap['max_train_windows'])
                refit_duration = time.perf_counter()-started_refit
                refit_cpu = time.process_time()-cpu_refit
                np.savez_compressed(cache_path, values=task.values,
                    X_train=task.X_train, Y_train=task.Y_train, origin_train=task.origin_train,
                    X_validation=task.X_validation, Y_validation=task.Y_validation,
                    origin_validation=task.origin_validation,
                    X_test=task.X_test, Y_test=task.Y_test, origin_test=task.origin_test,
                    X_refit=refit.X, Y_refit=refit.Y, origin_refit=refit.origins)
                base = task.to_dict()
                base['base_task_id'] = base.pop('task_id')
                base['cache_path'] = f'prepared/{cache_id}.npz'
                base['cache_sha256'] = digest_file(cache_path)
                base['budget_id'] = budget
                base['capacity_regime'] = cap
                base['refit_bounds'] = refit.to_dict()
                base['data_preparation_seconds'] = duration/max(1,len(tasks))+refit_duration
                base['data_preparation_cpu_seconds'] = process_cpu/max(1,len(tasks))+refit_cpu
                base['data_preparation_allocation'] = 'source preparation equally allocated across its base tasks; same allocation charged to all alternatives'
                for seed in protocol['seeds']:
                    entry = dict(base)
                    entry['seed'] = seed
                    entry['task_id'] = run_identifier(base['base_task_id'], budget, seed)
                    roster.append(entry)
    result = {'created_utc': now(), 'protocol_sha256': digest_file(ROOT/'protocol.json'),
              'source_manifest_sha256': digest_file(ROOT/'data_sources.json'),
              'tasks': roster, 'data_preparation': preparation,
              'information_used': 'source metadata, historical prefix eligibility and availability masks; no comparative model outcomes'}
    write_json(run_root/'task_roster.json', result)
    print(json.dumps({'event': 'prepared', 'tasks': len(roster),
        'development': sum(t['role']=='development' for t in roster),
        'confirmation': sum(t['role']=='confirmation' for t in roster),
        'supported': sum(t['supported'] for t in roster)}), flush=True)
    return result


def analysis_configuration(roster):
    return {'reserved_source_families': sorted({t['family_id'] for t in roster['tasks'] if t['role']=='confirmation'}),
            'expected_confirmation_task_ids': sorted(t['task_id'] for t in roster['tasks'] if t['role']=='confirmation'),
            'expected_development_task_ids': sorted(t['task_id'] for t in roster['tasks'] if t['role']=='development')}


def initial_freeze(run_root=DEFAULT_RUN):
    run_root = Path(run_root)
    if any((run_root/role).exists() for role in ('development','confirmation')):
        raise RuntimeError('Initial freeze must precede all comparative execution')
    if __package__:
        from . import analysis
    else:
        import analysis
    roster = json.loads((run_root/'task_roster.json').read_text())
    if roster['protocol_sha256'] != digest_file(ROOT/'protocol.json'):
        raise RuntimeError('Protocol changed after preparing the data roster')
    if roster['source_manifest_sha256'] != digest_file(ROOT/'data_sources.json'):
        raise RuntimeError('Source manifest changed after preparing the data roster')
    spec = analysis.prespecified_spec(analysis_configuration(roster))
    value = {'created_utc': now(), 'code_sha256': code_hashes(), 'environment': environment(),
             'roster_sha256': digest_file(run_root/'task_roster.json'),
             'analysis_specification': spec, 'confirmation_results_exist': False,
             'status': 'initial protocol frozen before development outcomes'}
    write_json(run_root/'initial_freeze.json', value)
    return value


def check_freeze(run_root, filename):
    freeze = json.loads((Path(run_root)/filename).read_text())
    if freeze['code_sha256'] != code_hashes():
        changed = [k for k,v in code_hashes().items() if freeze['code_sha256'].get(k)!=v]
        raise RuntimeError(f'Frozen scientific code/protocol changed: {changed}')
    if freeze['roster_sha256'] != digest_file(Path(run_root)/'task_roster.json'):
        raise RuntimeError('Frozen task roster changed')
    current_environment = environment()
    if (current_environment['packages'] != freeze['environment']['packages']
            or current_environment['python'] != freeze['environment']['python']):
        raise RuntimeError('Frozen Python or dependency versions changed')
    if filename == 'confirmation_freeze.json':
        if freeze['diagnostic_sha256'] != digest_file(Path(run_root)/'frozen_diagnostic.json'):
            raise RuntimeError('Frozen diagnostic changed')
        for name, digest in freeze['development_result_sha256'].items():
            if digest_file(Path(run_root)/'development'/name) != digest:
                raise RuntimeError('Development outcomes changed after diagnostic freeze')
    return freeze


def load_arrays(run_root, task):
    path = Path(run_root)/task['cache_path']
    if digest_file(path) != task['cache_sha256']:
        raise RuntimeError('Prepared array checksum mismatch')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@contextmanager
def measured(nodes, identifier, kind, **metadata):
    start, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        nodes.append({'id': identifier, 'kind': kind,
                      'seconds': time.perf_counter()-start,
                      'process_cpu_seconds': time.process_time()-cpu,
                      'peak_rss_bytes': peak_rss_bytes(),
                      'rss_scope': 'cumulative high-water mark of this isolated method process', **metadata})


def candidate_configs(method, protocol):
    if method in GEOMETRIC:
        return [{'alpha':a} for a in protocol['head_search']['alphas']]
    if method == 'holt':
        return [{'damped': b} for b in protocol['practical_search']['holt_damped']]
    if method in ('hist_gradient','mlp'):
        return [{'regularization': a} for a in protocol['practical_search']['regularization']]
    if method == 'seasonal_naive':
        return [{}]
    raise ValueError(method)


def construct(method, config, task, protocol, *, probe=False):
    cap = task['capacity_regime']
    if method in GEOMETRIC:
        representation, head, skip = GEOMETRIC[method]
        geometry = dict(protocol['geometry'])
        landmarks = cap['n_landmarks']
        if probe:
            geometry['max_fit'] = protocol['probe']['max_fit']
            landmarks = protocol['probe']['n_landmarks']
        return models.ForecastRegressor(representation=representation, head=head,
            raw_skip=skip, alpha=config['alpha'], n_landmarks=landmarks,
            bandwidth_multiplier=1.0, seed=task['seed'], head_seed=task['seed']+1000,
            geometry_parameters=geometry, query_mode=protocol['query_mode'])
    if method in ('hist_gradient','mlp'):
        return baselines.TabularBaseline(method, seed=task['seed'],
            width=cap['tree_leaves'] if method=='hist_gradient' else cap['mlp_width'],
            regularization=config['regularization'],
            iterations=cap['tree_iterations'] if method=='hist_gradient' else cap['mlp_iterations'])
    if method == 'holt':
        return baselines.HoltBaseline(damped=config['damped'])
    return baselines.SeasonalNaive(task['seasonal_period'])


def fit_pipeline(model, method, arrays, task, split):
    if method in ('holt','seasonal_naive'):
        cutoff = task['train_end'] if split=='train' else task['validation_end']
        model.fit_history(arrays['values'], cutoff,
            max_history=task['capacity_regime']['max_train_windows']+task['context'])
    else:
        model.fit(arrays['X_'+split], arrays['Y_'+split])
    return model


def predict_pipeline(model, method, arrays, task, split):
    if method in ('holt','seasonal_naive'):
        # Data origin is last observed; stateful controls take first target.
        return model.predict_history(arrays['values'], arrays['origin_'+split]+1, task['horizon'])
    return model.predict(arrays['X_'+split])


def finite_loss(prediction, target):
    prediction = np.asarray(prediction, float)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise FloatingPointError('Forecast has wrong shape or nonfinite values')
    loss = float(np.mean((prediction-target)**2))
    if not np.isfinite(loss):
        raise FloatingPointError('Forecast loss is nonfinite')
    return loss


def run_method(task, arrays, method, protocol):
    nodes, trials = [], []
    selection_started, selection_cpu = time.perf_counter(), time.process_time()
    winner, best = None, float('inf')
    all_warnings = []
    if method in GEOMETRIC:
        node_id=f'{method}:shared_alpha_search'
        try:
            with measured(nodes,node_id,'geometry_features_and_shared_alpha_search',
                          alphas=protocol['head_search']['alphas']):
                model=construct(method,{'alpha':protocol['head_search']['alphas'][0]},task,protocol)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    path=model.fit_select(arrays['X_train'],arrays['Y_train'],
                        arrays['X_validation'],arrays['Y_validation'],
                        alphas=protocol['head_search']['alphas'])
                all_warnings.extend(str(w.message) for w in caught)
                winner={'alpha':float(model.alpha)}
                best=min(float(trial['validation_mse']) for trial in path)
                trials=[{'parameters':{'alpha':trial['alpha']},
                         'validation_loss':trial['validation_mse'],'status':'ok',
                         'cost_node_id':node_id} for trial in path]
            del model
        except Exception as error:
            trials=[{'status':'failed','error':repr(error),'cost_node_id':node_id}]
        gc.collect()
    configs=[] if method in GEOMETRIC else candidate_configs(method,protocol)
    for index, config in enumerate(configs):
        node_id = f'{method}:search:{index}'
        try:
            with measured(nodes, node_id, 'search_fit_and_validation', parameters=config):
                model = construct(method, config, task, protocol)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fit_pipeline(model, method, arrays, task, 'train')
                    prediction = predict_pipeline(model, method, arrays, task, 'validation')
                loss = finite_loss(prediction, arrays['Y_validation'])
                all_warnings.extend(str(w.message) for w in caught)
                internal_warnings=list(getattr(model,'warnings_',[]))
                all_warnings.extend(internal_warnings)
            trials.append({'parameters':config, 'validation_loss':loss, 'status':'ok',
                           'cost_node_id':node_id,'warnings':internal_warnings+[str(w.message) for w in caught]})
            if loss < best:
                winner, best = config, loss
            del model, prediction
        except Exception as error:
            trials.append({'parameters':config, 'status':'failed', 'error':repr(error), 'cost_node_id':node_id})
        gc.collect()
    nodes.append({'id':f'{method}:selection_overhead','kind':'selection_bookkeeping_and_cleanup',
        'seconds':max(0.0,time.perf_counter()-selection_started-sum(n['seconds'] for n in nodes)),
        'process_cpu_seconds':max(0.0,time.process_time()-selection_cpu-sum(n['process_cpu_seconds'] for n in nodes)),
        'peak_rss_bytes':peak_rss_bytes()})
    selection_ids = [n['id'] for n in nodes]
    if winner is None:
        return {'candidate_id': method, 'status':'failed', 'trials':trials,
                'selection_cost_node_ids':selection_ids, 'final_cost_node_ids':[],
                'cost_node_ids':selection_ids, 'cost_nodes':nodes}
    with measured(nodes, f'{method}:refit', 'selected_final_refit'):
        model = construct(method, winner, task, protocol)
        fit_pipeline(model, method, arrays, task, 'refit')
    with measured(nodes, f'{method}:forecast', 'forecast_batch', n_origins=len(arrays['X_test'])):
        prediction = predict_pipeline(model, method, arrays, task, 'test')
    # Timing predictions and scoring are separate. The parent stores and
    # independently rechecks every origin loss from the prediction artifact.
    loss = finite_loss(prediction, arrays['Y_test'])
    with measured(nodes, f'{method}:serialization', 'model_serialization_and_metadata'):
        metadata = model.metadata()
        serialized = pickle.dumps(model, protocol=5)
    final_ids = [f'{method}:refit', f'{method}:forecast', f'{method}:serialization']
    candidate = {'candidate_id': method, 'status':'ok', 'selected_parameters':winner,
        'validation_loss':best, 'test_loss':loss, 'trials':trials,
        'selection_cost_node_ids':selection_ids, 'final_cost_node_ids':final_ids,
        'cost_node_ids':selection_ids+final_ids, 'cost_nodes':nodes,
        'serialized_model_bytes':len(serialized), 'model_sha256':hashlib.sha256(serialized).hexdigest(),
        'model_metadata':metadata, 'warnings':all_warnings,
        'tuning_fit_max_target_index':int((arrays['origin_train']+task['horizon']).max()),
        'selection_max_target_index':int((arrays['origin_validation']+task['horizon']).max()),
        'refit_max_target_index':int((arrays['origin_refit']+task['horizon']).max()),
        'test_min_target_index':int((arrays['origin_test']+1).min()),
        'test_max_target_index':int((arrays['origin_test']+task['horizon']).max())}
    return {'candidate':candidate, 'prediction':prediction,
            'origin_losses':np.mean((prediction-arrays['Y_test'])**2, axis=1)}


def run_probes(task, arrays, protocol):
    nodes, losses, metadata = [], {}, {}
    with measured(nodes, 'probe:prepare', 'historical_probe_subsample'):
        count = min(protocol['probe']['max_training_windows'], len(arrays['X_train']))
        indices = np.arange(count)*(len(arrays['X_train'])-1)//max(1,count-1)
        X, Y = arrays['X_train'][indices], arrays['Y_train'][indices]
    for method in ('raw_rbf','raw_kahm_rbf'):
        with measured(nodes, f'probe:{method}:fit', 'probe_fit', training_windows=count):
            model = construct(method, {'alpha':protocol['probe']['alpha']}, task, protocol, probe=True)
            model.fit(X,Y)
            if method=='raw_kahm_rbf':
                metadata = model.metadata()
        with measured(nodes, f'probe:{method}:validation', 'probe_validation'):
            pred = model.predict(arrays['X_validation'])
            losses[method] = finite_loss(pred, arrays['Y_validation'])
    return {'cost_nodes':nodes, 'metadata':metadata,
        'probe':{'raw_validation_loss':losses['raw_rbf'],
                 'kahm_validation_loss':losses['raw_kahm_rbf'],
                 'cost_node_ids':[n['id'] for n in nodes],
                 'tuning_fit_max_target_index':int((arrays['origin_train'][indices]+task['horizon']).max()),
                 'selection_max_target_index':int((arrays['origin_validation']+task['horizon']).max())},
        'geometry_cost_node_ids':['probe:prepare','probe:raw_kahm_rbf:fit']}


def child_entry(connection, task, arrays, method, protocol):
    started, cpu = time.perf_counter(), time.process_time()
    try:
        with threadpool_limits(limits=1):
            result = run_probes(task, arrays, protocol) if method=='__probe__' else run_method(task, arrays, method, protocol)
        result['child_wall_seconds'] = time.perf_counter()-started
        result['child_cpu_seconds'] = time.process_time()-cpu
        result['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        connection.send({'ok':True, 'result':result})
    except BaseException as error:
        connection.send({'ok':False,'error':repr(error),'traceback':traceback.format_exc(),
                         'child_wall_seconds':time.perf_counter()-started,
                         'child_cpu_seconds':time.process_time()-cpu,
                         'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    finally:
        connection.close()


def isolated(task, arrays, method, protocol, timeout_seconds=None):
    context = mp.get_context('fork' if sys.platform=='linux' else 'spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=child_entry, args=(child,task,arrays,method,protocol))
    started = time.perf_counter()
    process.start(); child.close()
    deadline = protocol['resources']['per_method_timeout_seconds']
    if timeout_seconds is not None:
        deadline=min(deadline,max(0.0,timeout_seconds))
    if parent.poll(deadline):
        try:
            result = parent.recv()
        except EOFError:
            result = {'ok':False,'error':'worker exited without result'}
        process.join(5)
    else:
        process.terminate(); process.join(5)
        result = {'ok':False,'error':f'timeout after {deadline} seconds','timeout':True}
    if process.is_alive():
        process.kill(); process.join(5)
    result['launch_to_result_seconds'] = time.perf_counter()-started
    result['exit_code'] = process.exitcode
    parent.close()
    return result


def run_task(task, run_root, protocol):
    task_started=time.perf_counter()
    started, cpu = time.perf_counter(), time.process_time()
    arrays = load_arrays(run_root, task)
    load_seconds, load_cpu = time.perf_counter()-started, time.process_time()-cpu
    metadata_started, metadata_cpu = time.perf_counter(), time.process_time()
    prefix = arrays['values'][:task['train_end']]
    mask = np.isfinite(prefix[:-1]) & np.isfinite(prefix[1:])
    if mask.sum()>2 and np.std(prefix[:-1][mask])>0 and np.std(prefix[1:][mask])>0:
        lag1 = float(np.corrcoef(prefix[:-1][mask],prefix[1:][mask])[0,1])
    else:
        lag1 = 0.0
    nodes = [{'id':'data:prepare','kind':'shared_data_preparation',
              'seconds':task['data_preparation_seconds'],
              'process_cpu_seconds':task['data_preparation_cpu_seconds']},
             {'id':'data:load','kind':'prepared_arrays_load_and_hash',
              'seconds':load_seconds,'process_cpu_seconds':load_cpu},
             {'id':'metadata:compute','kind':'historical_metadata_features',
              'seconds':time.perf_counter()-metadata_started,
              'process_cpu_seconds':time.process_time()-metadata_cpu}]
    for node in nodes:
        node['peak_rss_bytes']=peak_rss_bytes()
        node['rss_scope']='parent process high-water mark including prepared input arrays'
    record = {'task_id':task['task_id'], 'phase':task['role'],
        'source_family':task['family_id'],'dataset_id':task['source_id'],
        'series_id':task['series_id'],'horizon':task['horizon'],
        'history_length':task['context'],'seed':task['seed'],'budget_id':task['budget_id'],
        'loss_scale':task['loss_scale'],'scale_method':task['scale_method'],
        'split':{'train_end_exclusive':task['train_end'], 'validation_start':task['train_end'],
                 'validation_end_exclusive':task['validation_end'], 'test_start':task['validation_end'],
                 'test_end_exclusive':task['test_end']},
        'features':{'n_train_windows':len(arrays['X_train']), 'lag1_autocorrelation':lag1},
        'feature_provenance':{'metadata_max_observation_index':task['train_end']-1,
                              'geometry_max_observation_index':task['train_end']-1},
        'feature_cost_node_ids':{'metadata':['data:prepare','data:load','metadata:compute'], 'geometry':[]},
        'cost_nodes':nodes,'candidates':[], 'failures':[], 'created_utc':now(),
        'cache_sha256':task['cache_sha256'], 'source_provenance':task['provenance']}
    if not task['supported']:
        record['status']='unsupported'
        record['failures'].append({'reason':task['unsupported_reason']})
        return record, {}
    # Fixed pseudorandom ordering prevents systematic position from being
    # confounded with method. One active measured child; no parallel methods.
    order = list(protocol['methods'])
    entropy = int(hashlib.sha256(task['task_id'].encode()).hexdigest()[:8],16)
    np.random.default_rng(entropy).shuffle(order)
    record['execution_order'] = order
    predictions = {'target':arrays['Y_test'], 'origin':arrays['origin_test'],
                   'loss_scale':np.array(task['loss_scale'])}
    for method in ['__probe__']+order:
        remaining=protocol['resources'].get('per_task_timeout_seconds',1800)-(time.perf_counter()-task_started)
        if remaining <= 0:
            record['failures'].append({'candidate_id':method,'error':'total task timeout',
                                       'elapsed_seconds':time.perf_counter()-task_started})
            break
        result = isolated(task,arrays,method,protocol,timeout_seconds=remaining)
        if not result['ok']:
            record['failures'].append({'candidate_id':method,**result})
            # Failures remain visible and are never silently replaced or
            # discarded. Analysis refuses incomplete primary comparisons.
            continue
        payload = result['result']
        if method=='__probe__':
            record['cost_nodes'].extend(payload['cost_nodes'])
            record['probe']=payload['probe']
            record['probe']['process_peak_rss_kib']=payload['process_peak_rss_kib']
            diag=payload['metadata']['diagnostic_features']
            record['features'].update(kahm_gate_entropy=diag['gate_entropy_fraction'],
                affine_projection_residual=diag['gated_projection_residual_mse'])
            record['feature_cost_node_ids']['geometry']=payload['geometry_cost_node_ids']
            record['probe_model_metadata']=payload['metadata']
            continue
        if 'candidate' not in payload:
            record['failures'].append(payload)
            record['cost_nodes'].extend(payload.get('cost_nodes',[]))
            continue
        candidate=payload['candidate']
        record['cost_nodes'].extend(candidate.pop('cost_nodes'))
        for key in ('selection_cost_node_ids','cost_node_ids'):
            candidate[key]=['data:prepare','data:load']+candidate[key]
        candidate.update(process_peak_rss_kib=payload['process_peak_rss_kib'],
            child_wall_seconds=payload['child_wall_seconds'],child_cpu_seconds=payload['child_cpu_seconds'],
            launch_to_result_seconds=result['launch_to_result_seconds'])
        record['candidates'].append(candidate)
        predictions[method]=payload['prediction']
        predictions[method+'__origin_losses']=payload['origin_losses']
    record['status']='complete' if not record['failures'] else 'incomplete'
    return record,predictions


def run_phase(role, run_root=DEFAULT_RUN, limit=None):
    run_root=Path(run_root)
    check_freeze(run_root,'confirmation_freeze.json' if role=='confirmation' else 'initial_freeze.json')
    protocol=load_protocol()
    roster=json.loads((run_root/'task_roster.json').read_text())
    tasks=[t for t in roster['tasks'] if t['role']==role]
    if role=='confirmation' and limit is not None:
        raise ValueError('Confirmation execution cannot select a subset; resumption skips complete hashed files only')
    if limit is not None:
        tasks=tasks[:limit]
    directory=run_root/role
    directory.mkdir(parents=True,exist_ok=True)
    for i,task in enumerate(tasks):
        name=hashlib.sha256(task['task_id'].encode()).hexdigest()[:24]
        path=directory/(name+'.json')
        if path.exists():
            old=json.loads(path.read_text())
            artifact=directory/(name+'.npz')
            if old.get('status')=='complete' and old.get('predictions_sha256')==digest_file(artifact):
                continue
            raise RuntimeError(f'Existing incomplete/mismatched result requires explicit audit, not overwrite: {path}')
        started=time.perf_counter()
        record,predictions=run_task(task,run_root,protocol)
        artifact=directory/(name+'.npz')
        np.savez_compressed(artifact,**predictions)
        record['predictions_file']=artifact.name
        record['predictions_sha256']=digest_file(artifact)
        record['task_elapsed_seconds']=time.perf_counter()-started
        write_json(path,record)
        print(json.dumps({'event':'task_complete','phase':role,'index':i+1,'total':len(tasks),
            'task_id':task['task_id'],'status':record['status'],'seconds':record['task_elapsed_seconds'],
            'failures':len(record['failures'])}),flush=True)
        if record['status']!='complete':
            raise RuntimeError(f'Incomplete task retained for audit: {path}')
    return load_results(run_root,role)


def load_results(run_root,role):
    return [json.loads(p.read_text()) for p in sorted((Path(run_root)/role).glob('*.json'))]


def freeze_confirmation(run_root=DEFAULT_RUN):
    run_root=Path(run_root)
    check_freeze(run_root,'initial_freeze.json')
    if (run_root/'confirmation').exists():
        raise RuntimeError('Cannot freeze after confirmation execution')
    if __package__:
        from . import analysis
    else:
        import analysis
    roster=json.loads((run_root/'task_roster.json').read_text())
    config=analysis_configuration(roster)
    tasks=load_results(run_root,'development')
    development_audit=audit_results(run_root,roles=('development',))
    report=analysis.analyze_development(tasks,config=config)
    write_json(run_root/'development_analysis.json',report)
    analysis.freeze_diagnostic(tasks,path=run_root/'frozen_diagnostic.json',config=config)
    value={'created_utc':now(),'code_sha256':code_hashes(),'environment':environment(),
           'roster_sha256':digest_file(run_root/'task_roster.json'),
           'diagnostic_sha256':digest_file(run_root/'frozen_diagnostic.json'),
           'development_result_sha256':{p.name:digest_file(p) for p in sorted((run_root/'development').glob('*.json'))},
           'development_prediction_audit':development_audit,
           'confirmation_results_exist':False,'status':'frozen before confirmation execution'}
    write_json(run_root/'confirmation_freeze.json',value)
    return value


def audit_results(run_root,roles=('development','confirmation')):
    run_root=Path(run_root)
    if __package__:
        from . import analysis
    else:
        import analysis
    roster=json.loads((run_root/'task_roster.json').read_text())
    task_by_id={t['task_id']:t for t in roster['tasks']}
    checked=0
    for role in roles:
        for row in load_results(run_root,role):
            file=run_root/role/row['predictions_file']
            if digest_file(file)!=row['predictions_sha256']:
                raise ValueError('Prediction artifact checksum mismatch')
            original=load_arrays(run_root,task_by_id[row['task_id']])
            with np.load(file,allow_pickle=False) as pred:
                if not np.array_equal(pred['target'],original['Y_test']):
                    raise ValueError('Prediction artifact targets differ from frozen source windows')
                if not np.array_equal(pred['origin'],original['origin_test']):
                    raise ValueError('Prediction artifact origins differ from frozen task roster')
                for candidate in row['candidates']:
                    method=candidate['candidate_id']
                    analysis.audit_prediction_arrays(pred['target'],pred[method],candidate['test_loss'],
                        reported_origin_losses=pred[method+'__origin_losses'])
                    checked+=1
    return {'predictions_recomputed':checked,'source_targets_and_origins_equal':True}


def analyze(run_root=DEFAULT_RUN):
    run_root=Path(run_root)
    check_freeze(run_root,'confirmation_freeze.json')
    if __package__:
        from . import analysis
    else:
        import analysis
    frozen=json.loads((run_root/'frozen_diagnostic.json').read_text())
    tasks=load_results(run_root,'confirmation')
    report=analysis.evaluate_frozen(tasks,frozen,allow_confirmation=True)
    write_json(run_root/'confirmation_analysis.json',report)
    audit=audit_results(run_root)
    checkpoint=json.loads((ROOT/'CHECKPOINT.json').read_text())
    repository=ROOT.parent.parent
    changed=[p for p,h in checkpoint['tracked_file_sha256'].items() if digest_file(repository/p)!=h]
    if changed:
        raise ValueError(f'Historical checkpoint changed: {changed}')
    validation={'created_utc':now(),**audit,
        'historical_tracked_files_unchanged':len(checkpoint['tracked_file_sha256']),
        'frozen_protocol_intact':True,'status':'passed'}
    write_json(run_root/'VALIDATION.json',validation)
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','initial-freeze','run','freeze-confirmation','analyze'])
    parser.add_argument('--run-root',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--role',choices=['development','confirmation'],default='development')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.run_root)
    elif args.command=='initial-freeze': print(json.dumps(initial_freeze(args.run_root),indent=2))
    elif args.command=='run': run_phase(args.role,args.run_root,args.limit)
    elif args.command=='freeze-confirmation': print(json.dumps(freeze_confirmation(args.run_root),indent=2))
    else: print(json.dumps(analyze(args.run_root),indent=2))


if __name__=='__main__':
    main()
