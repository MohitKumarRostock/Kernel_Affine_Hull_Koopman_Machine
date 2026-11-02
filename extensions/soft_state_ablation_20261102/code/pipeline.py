#!/usr/bin/env python3
"""Append-only soft-state ablation: freeze, ABC, fit seal, D, forecast seal.

Every fitting or forecasting worker is a fresh process with one numerical
thread. Production preparation requires a complete source/configuration freeze.
Targets are physically present after D preparation but are not passed to any
fitting or forecast interface; the seals document order, not cryptographic
isolation. Terminal scoring belongs to the separately frozen analyzer.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/kahm-softstate-mpl'

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from queue import Queue
import resource
import re
import subprocess
import sys
from threading import Event
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'dimension_forecasting'
NUMERICAL = ROOT.parent / 'policy_augmentation_20261101'
sys.path.insert(0, str(OLD / 'code'))
from study_contract import seed_records, task_roster
from artifacts import (array_hash, digest, load_portable_state, now, safe,
                       state_fingerprint, write_json, write_npz, write_portable_state)

ARMS = ('G', 'P', 'K', 'R')
VIEWS = ('shared', 'independent')
FIT_FILES = ('state.json', 'state.npz', 'pre_c_selection.json',
             'validation_predictions.npz', 'diagnostic_predictions.npz',
             'C_diagnostics.json', 'numerical_solver_events.json', 'fit_summary.json')
PREDICTION_FILES = ('predictions.npz', 'query_diagnostics.json', 'prediction_receipt.json')


def expected_keys():
    return {f'{view}__{arm}' for view in VIEWS for arm in ARMS}


def config(root):
    c = json.loads((Path(root) / 'config.json').read_text())
    if c.get('seed_derivation', {}).get('excluded_seed_manifest'):
        raise ValueError('This study checks its own readiness/prior_seed_exclusions.json')
    if c['execution']['numerical_threads'] != 1 or c['execution']['workers'] > 3:
        raise ValueError('One numerical thread and at most three workers are required')
    if c['execution'].get('output_array_count_per_task', 8) != 8:
        raise ValueError('The study produces exactly eight logical forecasts')
    caps = c['native_cap_order']
    if not caps or len(set(caps)) != len(caps) or any(type(v) is not int or v < 1 for v in caps):
        raise ValueError('Distinct positive native caps are required')
    if any(k in c['model'] for k in ('rank', 'rank_cap', 'dimension_policy')):
        raise ValueError('Cap policy must be specified by native_cap_order')
    if c.get('input_scaling', 'standard') != 'standard':
        raise ValueError('Only the declared standard input scaling is implemented')
    if c.get('arms', list(ARMS)) != list(ARMS) or c.get('contexts', list(VIEWS)) != list(VIEWS):
        raise ValueError('Declared arm/context order differs from implementation')
    transform = c.get('block_transforms', {})
    if transform.get('scale_floor', c['model']['feature_scale_floor']) != c['model']['feature_scale_floor']:
        raise ValueError('Block and model scale floors differ')
    decoder = c.get('decoder', {})
    if any(decoder.get(key, False) for key in ('raw_history_baseline', 'residual_target', 'calibration')):
        raise ValueError('This representation-only study has no retained baseline or calibration')
    task_roster(c)
    return c


def verify_map(root, files):
    root = Path(root).resolve()
    if not isinstance(files, dict):
        raise TypeError('A file digest mapping is required')
    for name, sha in files.items():
        relative = Path(name)
        path = (root / relative).resolve()
        if (relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root)
                or not path.is_file() or digest(path) != sha):
            raise ValueError('Frozen file missing or changed: ' + name)


def _source_files(root):
    root = Path(root)
    paths = [root / 'config.json', root / 'PROTOCOL.md',
             root / 'readiness/prior_seed_exclusions.json',
             root / 'readiness/manufactured_preflight.json']
    for directory in ('code', 'analysis_code', 'audit'):
        paths.extend(p for p in (root / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.md', '.json')
                     and '__pycache__' not in p.parts)
    for name in ('requirements.txt',):
        if (root / name).is_file():
            paths.append(root / name)
    return sorted(set(paths))


def freeze(root):
    root = Path(root).resolve()
    if (root / 'prepared').exists() or (root / 'results').exists():
        raise ValueError('Source freeze must precede all study data and fitting')
    c = config(root)
    readiness = json.loads((root / 'readiness/manufactured_preflight.json').read_text())
    if readiness.get('status') != 'passed' or readiness.get('production_data_used') is not False:
        raise ValueError('Passed artificial-only preflight receipt required')
    prior = json.loads((root / 'readiness/prior_seed_exclusions.json').read_text())
    records = seed_records(c)
    collision = {r['seed'] for r in records} & {int(r['seed']) for r in prior['records']}
    if collision:
        raise ValueError('Seed collision against historical studies: ' + repr(sorted(collision)))
    source = _source_files(root)
    inherited = []
    for directory in ('code', 'runtime', 'vendor_original'):
        inherited.extend(p for p in (OLD / directory).rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts)
    value = {'schema': 'kahm-soft-state-freeze-v1', 'status': 'frozen', 'created_utc': now(),
             'task_count': len(task_roster(c)), 'config_sha256': digest(root / 'config.json'),
             'files': {str(p.relative_to(root)): digest(p) for p in source},
             'inherited_files': {str(p.relative_to(OLD)): digest(p) for p in sorted(inherited)},
             'numerical_solver_files': {'code/numerical_solver.py': digest(NUMERICAL / 'code/numerical_solver.py')},
             'source_publication_commit': c.get('source', {}).get('publication_commit'),
             'source_parent_commit': c.get('source', {}).get('parent_commit'),
             'seed_count': len(records), 'prior_seed_count': len(prior['records']),
             'seed_collision_check': 'passed', 'data_generated': False, 'D_generated': False,
             'manufactured_fixture': bool(c.get('manufactured_fixture')),
             'public_preregistration': False,
             'environment': {'python': sys.version, 'platform': platform.platform(),
                             'packages': {k: importlib.metadata.version(k) for k in
                                          ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}}}
    write_json(root / 'freeze.json', value)
    verify_freeze(root)
    return {'status': 'frozen', 'source_files': len(value['files']),
            'inherited_files': len(value['inherited_files']), 'tasks': value['task_count']}


def verify_freeze(root, *, full_hash=True):
    root = Path(root).resolve()
    value = json.loads((root / 'freeze.json').read_text())
    c = config(root)
    if (value['status'] != 'frozen' or value['config_sha256'] != digest(root / 'config.json')
            or value['task_count'] != len(task_roster(c))):
        raise ValueError('Unfrozen or changed study configuration')
    # Fixtures run the reviewed implementation with copied source files. Bind
    # those copies to the actual executing code as well as to their fixture root.
    executing = {name: sha for name, sha in value['files'].items() if name.startswith('code/')}
    verify_map(ROOT, executing)
    if full_hash:
        verify_map(root, value['files'])
        verify_map(OLD, value['inherited_files'])
        verify_map(NUMERICAL, value['numerical_solver_files'])
    return value


def verify_publication(root):
    """Check the recorded publication gate; this receipt is procedural evidence.

    The publishing client verifies the remote commit/tree and file identities.
    This offline stage checks that receipt and its binding to the local freeze;
    it neither performs a remote lookup nor provides cryptographic oversight.
    """
    root = Path(root).resolve()
    c = config(root)
    if c.get('manufactured_fixture'):
        return None
    receipt = json.loads((root / 'PUBLICATION.json').read_text())
    if (receipt.get('schema') != 'kahm-soft-state-publication-v1'
            or receipt.get('status') != 'published'
            or receipt.get('source_files_verified') is not True):
        raise ValueError('A verified source publication receipt is required before production generation')
    for field in ('source_commit', 'source_tree_sha'):
        if not isinstance(receipt.get(field), str) or not re.fullmatch(r'[0-9a-f]{40}', receipt[field]):
            raise ValueError('Invalid publication identity: ' + field)
    if receipt.get('repository') != c['source']['repository']:
        raise ValueError('Published repository differs from the declared source')
    if not isinstance(receipt.get('branch'), str) or not receipt['branch'].strip():
        raise ValueError('Published source branch is missing')
    if receipt.get('freeze_sha256') != digest(root / 'freeze.json'):
        raise ValueError('Publication is not bound to this exact frozen source')
    frozen = json.loads((root / 'freeze.json').read_text())
    published = datetime.fromisoformat(receipt['published_utc'].replace('Z', '+00:00'))
    frozen_time = datetime.fromisoformat(frozen['created_utc'].replace('Z', '+00:00'))
    if published.tzinfo is None or frozen_time.tzinfo is None:
        raise ValueError('Publication/freeze times must include timezone offsets')
    if published < frozen_time or published > datetime.now(timezone.utc):
        raise ValueError('Publication must follow the frozen source and precede production preparation')
    return receipt


def verify_manifest(root, name, *, hash_inputs=True):
    root = Path(root).resolve()
    manifest = json.loads((root / 'prepared' / name).read_text())
    if (manifest['config_sha256'] != digest(root / 'config.json')
            or manifest['freeze_sha256'] != digest(root / 'freeze.json')):
        raise ValueError('Prepared source/configuration identity changed')
    publication = verify_publication(root)
    if publication is not None:
        if manifest.get('publication_sha256') != digest(root / 'PUBLICATION.json'):
            raise ValueError('Prepared data publication identity changed')
        publication_time = datetime.fromisoformat(publication['published_utc'].replace('Z', '+00:00'))
        prepared_time = datetime.fromisoformat(manifest['created_utc'].replace('Z', '+00:00'))
        if publication_time >= prepared_time:
            raise ValueError('Production data must be prepared after source publication')
    expected = {task['task_id']: task for task in task_roster(config(root))}
    actual = {task['task_id']: task for task in manifest['tasks']}
    if (set(actual) != set(expected) or len(actual) != len(manifest['tasks'])
            or manifest['task_count'] != len(actual)):
        raise ValueError('Incomplete or duplicated prepared task roster')
    for tid, task in actual.items():
        if any(task[k] != v for k, v in expected[tid].items()):
            raise ValueError('Prepared task metadata differs: ' + tid)
        if hash_inputs:
            for field, sha in (('input', 'sha256'), ('query', 'query_sha256'), ('target', 'target_sha256')):
                if field in task:
                    verify_map(root, {task[field]: task[sha]})
    if hash_inputs:
        verify_map(root, manifest['base_files'])
    if name == 'd_manifest.json':
        if manifest['fit_lock_sha256'] != digest(root / 'FIT_LOCK.json'):
            raise ValueError('D generation is not bound to the complete fit seal')
        if manifest['ABC_manifest_sha256'] != digest(root / 'prepared/abc_manifest.json'):
            raise ValueError('D/ABC preparation identity differs')
    return manifest


def _manufactured_base(c, split, streams):
    """Artificial integration fixture only; no inherited scientific law runs."""
    n, length = c['split_episodes'][split], c['max_episode_length']
    rng = np.random.default_rng(streams['generation']['seed'])
    base = rng.normal(size=(n, length))
    base[:, 1:] += .2 * base[:, :-1]
    unit = np.random.default_rng(streams['unit_noise']['seed']).normal(size=base.shape)
    return base, unit


def prepare(root, stage):
    root = Path(root).resolve()
    verify_freeze(root)
    c = config(root)
    publication = verify_publication(root)
    if stage not in ('abc', 'd'):
        raise ValueError('Unknown preparation stage')
    if stage == 'abc':
        if (root / 'prepared').exists():
            raise FileExistsError('Preparation has already started')
        prior = json.loads((root / 'readiness/prior_seed_exclusions.json').read_text())
        if {r['seed'] for r in seed_records(c)} & {int(r['seed']) for r in prior['records']}:
            raise ValueError('Historical seed collision')
    else:
        verify_seal(root, 'fits')
        abc = verify_manifest(root, 'abc_manifest.json')
        if (root / 'prepared/d_manifest.json').exists() or (root / 'prepared/targets').exists():
            raise FileExistsError('D generation has already started')
    # The generator and the geometry package use separate namespace trees;
    # this command is deliberately a separate process from all fitting workers.
    import prepare_dimension_data as original
    module = original.generator_module()
    records, tasks = seed_records(c), task_roster(c)
    if stage == 'abc':
        write_json(root / 'prepared/seed_manifest.json',
                   {'records': records, 'seed_count': len(records),
                    'D_streams_reserved_but_not_generated': True})
    result, base_files = [], {}
    for rep in range(c['repetitions']):
        for law in c['laws']:
            splits = ('A', 'B', 'C') if stage == 'abc' else ('D',)
            bases, units, streams, saved = {}, {}, {}, {}
            for split in splits:
                streams[split] = original._streams(records, rep, law, split)
                bases[split], units[split] = (
                    _manufactured_base(c, split, streams[split]) if c.get('manufactured_fixture')
                    else original._base(c, law, split, streams[split], module))
                saved[split + '_base_episodes'] = bases[split]
                saved[split + '_unit_noise'] = units[split]
            folder = 'base_abc' if stage == 'abc' else 'withheld_d_base'
            base_path = f'prepared/{folder}/{law}__rep{rep:02d}.npz'
            write_npz(root / base_path, saved)
            base_files[base_path] = digest(root / base_path)
            for task in (tasks if stage == 'abc' else abc['tasks']):
                if task['repetition'] != rep or task['law'] != law:
                    continue
                amplitude = (float(np.std(bases['A'][:, :task['history'] + c['history_tail_length']], ddof=0))
                             if stage == 'abc' else task['A_base_episode_std'])
                arrays = {'offsets': np.asarray(c['horizons'], dtype=np.int64),
                          'A_base_episode_std': np.asarray(amplitude),
                          'noise_std': np.asarray(task['noise_level'] * amplitude)}
                split_records = {}
                for split in splits:
                    windows, retained = original._windows(
                        c, bases[split], units[split], task['history'], task['noise_level'],
                        amplitude, split, streams[split], module)
                    arrays.update({split + '_' + key: value for key, value in windows.items()})
                    split_records[split] = {
                        'streams': streams[split], 'episodes': c['split_episodes'][split],
                        'retained_rows': len(retained), 'retained_indices_sha256': array_hash(retained),
                        'base_sha256': array_hash(bases[split]), 'unit_noise_sha256': array_hash(units[split])}
                if stage == 'abc':
                    arrays['loss_scale'] = np.maximum(np.var(arrays['A_Y'], axis=0, ddof=0),
                                                       c['model']['target_variance_floor'])
                    path = f'prepared/abc/{task["task_id"]}.npz'
                    write_npz(root / path, arrays)
                    result.append({**task, 'input': path, 'sha256': digest(root / path),
                                   'base_file': base_path, 'A_base_episode_std': amplitude,
                                   'noise_std': task['noise_level'] * amplitude,
                                   'loss_scale': arrays['loss_scale'].tolist(), 'splits': split_records,
                                   'array_sha256': {k: array_hash(v) for k, v in arrays.items()}})
                else:
                    query = f'prepared/queries/{task["task_id"]}.npz'
                    target = f'prepared/targets/{task["task_id"]}.npz'
                    metadata = {key: value for key, value in arrays.items() if key not in ('D_X', 'D_Y')}
                    write_npz(root / query, {'D_X': arrays['D_X'], **metadata})
                    write_npz(root / target, {'D_Y': arrays['D_Y'], **metadata})
                    result.append({**{key: task[key] for key in tasks[0]},
                                   'query': query, 'query_sha256': digest(root / query),
                                   'target': target, 'target_sha256': digest(root / target),
                                   'rows': len(arrays['D_X']), 'base_file': base_path,
                                   'D_X_sha256': array_hash(arrays['D_X']),
                                   'D_Y_sha256': array_hash(arrays['D_Y']),
                                   'A_base_episode_std': amplitude, 'splits': split_records})
        print(json.dumps({'phase': stage, 'repetitions_prepared': rep + 1, 'tasks': len(result)}), flush=True)
    manifest = {'schema': 'kahm-soft-state-' + stage + '-v1', 'created_utc': now(),
                'config_sha256': digest(root / 'config.json'), 'freeze_sha256': digest(root / 'freeze.json'),
                'tasks': result, 'task_count': len(result), 'base_files': base_files,
                'seed_manifest_sha256': digest(root / 'prepared/seed_manifest.json'),
                'D_generated': stage == 'd', 'manufactured_fixture': bool(c.get('manufactured_fixture'))}
    if publication is not None:
        manifest['publication_sha256'] = digest(root / 'PUBLICATION.json')
    if stage == 'd':
        manifest.update(fit_lock_sha256=digest(root / 'FIT_LOCK.json'),
                        ABC_manifest_sha256=digest(root / 'prepared/abc_manifest.json'),
                        generation_after_complete_fit_lock=True,
                        targets_reserved_until_global_prediction_lock=True)
    name = 'abc_manifest.json' if stage == 'abc' else 'd_manifest.json'
    write_json(root / 'prepared' / name, manifest)
    verify_manifest(root, name)
    return {'tasks': len(result), 'phase': stage, 'manufactured_fixture': bool(c.get('manufactured_fixture'))}


def worker_fit(root, task):
    from model import fit_suite, numerical_solver
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve()
    verify_freeze(root, full_hash=False)
    if (root / 'prepared/d_manifest.json').exists() or (root / 'prepared/targets').exists():
        raise ValueError('D exists before the global fit seal')
    verify_map(root, {task['input']: task['sha256']})
    output = root / 'results' / task['task_id']
    output.mkdir(parents=True, exist_ok=False)
    c = config(root)
    started, wall, cpu = now(), time.perf_counter(), time.process_time()
    solver = numerical_solver()
    solver.clear_events()
    try:
        with np.load(root / task['input'], allow_pickle=False) as z:
            ax, ay, bx, by, loss_scale = [z[k] for k in ('A_X', 'A_Y', 'B_X', 'B_Y', 'loss_scale')]
        with threadpool_limits(limits=1):
            fit_wall, fit_cpu = time.perf_counter(), time.process_time()
            suite, details, validation = fit_suite(ax, ay, bx, by,
                cap_order=c['native_cap_order'], seed=task['fit_seed'], config=c['model'])
            fit_resources = {'wall_seconds': time.perf_counter() - fit_wall,
                             'cpu_seconds': time.process_time() - fit_cpu}
            np.testing.assert_array_equal(suite.loss_scale, loss_scale)
            frozen_state = state_fingerprint(suite.to_state())
            write_json(output / 'pre_c_selection.json',
                       {'schema': 'kahm-soft-state-selection-v1', 'saved_utc': now(),
                        'selections': suite.selections, 'state_fingerprint': frozen_state,
                        'C_loaded': False, 'selection_complete': True})
            write_npz(output / 'validation_predictions.npz', validation)
            state_resources = write_portable_state(output, suite.to_state())
            # C is first accessed only after every state and selection is saved.
            # No C target is passed into fit_suite or suite.predict.
            with np.load(root / task['input'], allow_pickle=False) as z:
                cx, cy = z['C_X'], z['C_Y']
            cwall, ccpu = time.perf_counter(), time.process_time()
            cp, cd = suite.predict(cx, tag='C_diagnostic')
            c_resources = {'wall_seconds': time.perf_counter() - cwall,
                           'cpu_seconds': time.process_time() - ccpu}
            if state_fingerprint(suite.to_state()) != frozen_state:
                raise AssertionError('Diagnostic C prediction changed the fitted state')
            write_npz(output / 'diagnostic_predictions.npz', cp)
            c_losses = {key: {'mse_by_horizon': np.mean((cy - value) ** 2, axis=0),
                              'smse_by_horizon': np.mean((cy - value) ** 2, axis=0) / loss_scale}
                        for key, value in cp.items()}
            write_json(output / 'C_diagnostics.json',
                       {'schema': 'kahm-soft-state-C-diagnostic-v1', 'role': 'diagnostic only',
                        'predictions': cd, 'losses': c_losses, 'selection_performed': False,
                        'state_unchanged': True, 'state_fingerprint': frozen_state})
        events = solver.events()
        write_json(output / 'numerical_solver_events.json',
                   {'installation': details['numerical_solver_installation'], 'events': events,
                    'fallback_count': len(events),
                    'recovered_count': sum(e['status'] == 'recovered' for e in events),
                    'failed_count': sum(e['status'] != 'recovered' for e in events)})
        summary = {**details, 'schema': 'kahm-soft-state-task-fit-v1', 'status': 'completed',
                   'task': task, 'started_utc': started, 'completed_utc': now(),
                   'freeze_sha256': digest(root / 'freeze.json'), 'input_sha256': task['sha256'],
                   'C_diagnostic_only': True, 'C_state_unchanged': True,
                   'state_fingerprint': frozen_state,
                   'resources': {'fit_all_caps_and_four_arms': fit_resources,
                                 'C_diagnostic_predictions': c_resources,
                                 'total_wall_seconds': time.perf_counter() - wall,
                                 'total_cpu_seconds': time.process_time() - cpu,
                                 'peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 **state_resources,
                                 'scope': 'Whole task, all rejected cap/alpha candidates and shared computations; not isolated deployed-model cost'},
                   'files': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
        write_json(output / 'fit_summary.json', summary)
        return {'task': task['task_id'], 'status': 'completed',
                'seconds': summary['resources']['total_wall_seconds']}
    except Exception:
        if not (output / 'numerical_solver_events.json').exists():
            write_json(output / 'numerical_solver_events.json', {'events': solver.events(), 'fit_completed': False})
        write_json(output / 'failure.json',
                   {'status': 'failed', 'task': task, 'utc': now(), 'traceback': traceback.format_exc()})
        raise


def worker_predict(root, task):
    from model import SoftStateSuite
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve()
    verify_freeze(root, full_hash=False)
    output = root / 'predictions' / task['task_id']
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((root / 'FIT_LOCK.json').read_text())
    state_dir = root / 'results' / task['task_id']
    for name in ('state.json', 'state.npz'):
        key = str((state_dir / name).relative_to(root))
        if key not in lock['files'] or digest(state_dir / name) != lock['files'][key]:
            raise ValueError('Fitted state changed: ' + key)
    verify_map(root, {task['query']: task['query_sha256']})
    suite = SoftStateSuite.from_state(load_portable_state(state_dir))
    frozen = state_fingerprint(suite.to_state())
    started, wall, cpu = now(), time.perf_counter(), time.process_time()
    try:
        with np.load(root / task['query'], allow_pickle=False) as values:
            if any(key == 'D_Y' or key.endswith('_Y') for key in values.files):
                raise ValueError('Query archive contains targets')
            x = values['D_X']
        with threadpool_limits(limits=1):
            arrays, diagnostics = suite.predict(x, tag='D')
        if state_fingerprint(suite.to_state()) != frozen:
            raise AssertionError('D query mutated the fitted state')
        if set(arrays) != expected_keys():
            raise AssertionError('Terminal forecast roster differs')
        write_npz(output / 'predictions.npz', arrays)
        write_json(output / 'query_diagnostics.json', diagnostics)
        receipt = {'schema': 'kahm-soft-state-prediction-v1', 'status': 'completed',
                   'task': task, 'started_utc': started, 'completed_utc': now(),
                   'state_json_sha256': digest(state_dir / 'state.json'),
                   'state_npz_sha256': digest(state_dir / 'state.npz'),
                   'state_fingerprint': frozen, 'state_unchanged': True,
                   'query_sha256': task['query_sha256'], 'targets_loaded': False,
                   'array_count': len(arrays), 'array_keys': sorted(arrays),
                   'prediction_sha256': digest(output / 'predictions.npz'),
                   'wall_seconds': time.perf_counter() - wall,
                   'cpu_seconds': time.process_time() - cpu,
                   'peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'files': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
        write_json(output / 'prediction_receipt.json', receipt)
        return {'task': task['task_id'], 'status': 'completed', 'seconds': receipt['wall_seconds']}
    except Exception:
        write_json(output / 'failure.json',
                   {'status': 'failed', 'task': task, 'utc': now(), 'traceback': traceback.format_exc()})
        raise


def run_stage(root, phase, workers):
    root = Path(root).resolve()
    verify_freeze(root)
    c = config(root)
    if workers < 1 or workers > min(3, c['execution']['workers']):
        raise ValueError('Worker count exceeds the frozen budget')
    if phase == 'fit':
        if (root / 'prepared/d_manifest.json').exists():
            raise ValueError('Fitting cannot begin after D generation')
        manifest = verify_manifest(root, 'abc_manifest.json')
    elif phase == 'predict':
        verify_seal(root, 'fits')
        manifest = verify_manifest(root, 'd_manifest.json')
    else:
        raise ValueError('Unknown worker phase')
    tasks = manifest['tasks']
    directory = root / 'execution' / phase
    directory.mkdir(parents=True, exist_ok=False)
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    slots = Queue()
    for i in range(workers):
        slots.put(cores[i % len(cores)] if cores else None)
    failed = Event()
    started, wall = now(), time.perf_counter()

    def run(task):
        if failed.is_set():
            raise RuntimeError('Stage stopped after a prior task failure')
        core = slots.get()
        try:
            if failed.is_set():
                raise RuntimeError('Stage stopped before worker launch')
            command = [sys.executable, str(Path(__file__).resolve()), '_worker',
                       '--root', str(root), '--phase', phase, '--task', task['task_id']]
            if core is not None:
                command += ['--cpu', str(core)]
            with (directory / (task['task_id'] + '.log')).open('x') as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                failed.set()
                raise RuntimeError(f'{task["task_id"]}: worker exited {process.returncode}')
            return task['task_id']
        finally:
            slots.put(core)

    completed = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    completed.append(future.result())
                except Exception:
                    failed.set()
                    for pending in futures:
                        pending.cancel()
                    raise
                if len(completed) % 8 == 0 or len(completed) == len(tasks):
                    print(json.dumps({'phase': phase, 'completed': len(completed),
                                      'total': len(tasks), 'elapsed_seconds': time.perf_counter() - wall}), flush=True)
    finally:
        write_json(directory / 'stage_receipt.json',
                   {'phase': phase, 'started_utc': started, 'completed_utc': now(),
                    'completed_tasks': sorted(completed), 'total_expected': len(tasks),
                    'status': 'completed' if len(completed) == len(tasks) else 'failed',
                    'workers': workers, 'one_numerical_thread_per_worker': True,
                    'wall_seconds': time.perf_counter() - wall})
    return {'phase': phase, 'status': 'completed', 'tasks': len(completed)}


def check_task_receipt(root, phase, expected, value):
    root = Path(root)
    tid = expected['task_id']
    if value['status'] != 'completed' or value['task'] != expected:
        raise ValueError('Receipt differs from complete prepared task: ' + tid)
    if phase == 'fits':
        if (value['freeze_sha256'] != digest(root / 'freeze.json')
                or value['input_sha256'] != expected['sha256']
                or value['C_diagnostic_only'] is not True or value['C_state_unchanged'] is not True):
            raise ValueError('Fitted task source, data or C-role identity differs: ' + tid)
        if set(value['candidates']) != {f'cap{cap}' for cap in config(root)['native_cap_order']}:
            raise ValueError('Candidate roster is incomplete: ' + tid)
        if set(value['selections']) != set(VIEWS):
            raise ValueError('Selected views are incomplete: ' + tid)
        for view in VIEWS:
            if set(value['selections'][view]) != set(ARMS):
                raise ValueError('Selected arms are incomplete: ' + tid)
    else:
        fit = json.loads((root / 'FIT_LOCK.json').read_text())
        for name, field in (('state.json', 'state_json_sha256'), ('state.npz', 'state_npz_sha256')):
            if value[field] != fit['files'][f'results/{tid}/{name}']:
                raise ValueError('Prediction state identity differs: ' + tid)
        if (value['query_sha256'] != expected['query_sha256'] or value['targets_loaded'] is not False
                or value['state_unchanged'] is not True or value['array_count'] != 8
                or set(value['array_keys']) != expected_keys()):
            raise ValueError('Invalid forecast receipt: ' + tid)
        if value['prediction_sha256'] != digest(root / 'predictions' / tid / 'predictions.npz'):
            raise ValueError('Forecast archive changed: ' + tid)


def seal(root, phase):
    root = Path(root).resolve()
    verify_freeze(root)
    if phase not in ('fits', 'predictions'):
        raise ValueError('Unknown seal phase')
    if phase == 'predictions':
        verify_seal(root, 'fits')
    else:
        if (root / 'prepared/d_manifest.json').exists():
            raise ValueError('Global fit seal must precede D generation')
    data_name = 'abc_manifest.json' if phase == 'fits' else 'd_manifest.json'
    manifest = verify_manifest(root, data_name)
    folder = 'results' if phase == 'fits' else 'predictions'
    receipt_name = 'fit_summary.json' if phase == 'fits' else 'prediction_receipt.json'
    required = FIT_FILES if phase == 'fits' else PREDICTION_FILES
    stage = root / 'execution' / ('fit' if phase == 'fits' else 'predict') / 'stage_receipt.json'
    stage_value = json.loads(stage.read_text())
    expected_ids = {task['task_id'] for task in manifest['tasks']}
    if (stage_value['status'] != 'completed' or set(stage_value['completed_tasks']) != expected_ids
            or len(stage_value['completed_tasks']) != len(expected_ids)):
        raise ValueError('Cannot seal an incomplete execution stage')
    files = {str(stage.relative_to(root)): digest(stage)}
    for task in manifest['tasks']:
        directory = root / folder / task['task_id']
        if (directory / 'failure.json').exists() or any(not (directory / name).is_file() for name in required):
            raise ValueError('Failed or incomplete task cannot be sealed: ' + task['task_id'])
        value = json.loads((directory / receipt_name).read_text())
        check_task_receipt(root, phase, task, value)
        verify_map(directory, value['files'])
        files.update({str(p.relative_to(root)): digest(p) for p in directory.iterdir() if p.is_file()})
    value = {'schema': 'kahm-soft-state-seal-v1', 'status': 'sealed', 'phase': phase,
             'created_utc': now(), 'task_count': len(expected_ids),
             'task_ids': [task['task_id'] for task in manifest['tasks']], 'files': files,
             'config_sha256': digest(root / 'config.json'), 'freeze_sha256': digest(root / 'freeze.json'),
             'manifest_sha256': digest(root / 'prepared' / data_name)}
    write_json(root / ('FIT_LOCK.json' if phase == 'fits' else 'PREDICTION_LOCK.json'), value)
    return {'phase': phase, 'tasks': len(expected_ids), 'files': len(files)}


def verify_seal(root, phase):
    root = Path(root).resolve()
    value = json.loads((root / ('FIT_LOCK.json' if phase == 'fits' else 'PREDICTION_LOCK.json')).read_text())
    if (value['status'] != 'sealed' or value['phase'] != phase
            or value['config_sha256'] != digest(root / 'config.json')
            or value['freeze_sha256'] != digest(root / 'freeze.json')):
        raise ValueError('Seal provenance differs')
    data_name = 'abc_manifest.json' if phase == 'fits' else 'd_manifest.json'
    manifest = verify_manifest(root, data_name)
    expected = {task['task_id'] for task in manifest['tasks']}
    if (set(value['task_ids']) != expected or len(value['task_ids']) != len(expected)
            or value['task_count'] != len(expected)
            or value['manifest_sha256'] != digest(root / 'prepared' / data_name)):
        raise ValueError('Seal data/task roster differs')
    folder = 'results' if phase == 'fits' else 'predictions'
    required = FIT_FILES if phase == 'fits' else PREDICTION_FILES
    for task in manifest['tasks']:
        tid = task['task_id']
        if any(f'{folder}/{tid}/{name}' not in value['files'] for name in required):
            raise ValueError('Seal is missing a canonical task artifact: ' + tid)
        if (root / folder / tid / 'failure.json').exists():
            raise ValueError('A sealed task has a failure record')
        receipt = 'fit_summary.json' if phase == 'fits' else 'prediction_receipt.json'
        check_task_receipt(root, phase, task, json.loads((root / folder / tid / receipt).read_text()))
    verify_map(root, value['files'])
    return value


def release(root):
    root = Path(root).resolve()
    verify_freeze(root)
    verify_seal(root, 'fits')
    verify_seal(root, 'predictions')
    manifest = verify_manifest(root, 'd_manifest.json')
    write_json(root / 'TARGET_RELEASE.json',
               {'schema': 'kahm-soft-state-target-release-v1', 'status': 'released',
                'created_utc': now(), 'fit_lock_sha256': digest(root / 'FIT_LOCK.json'),
                'prediction_lock_sha256': digest(root / 'PREDICTION_LOCK.json'),
                'd_manifest_sha256': digest(root / 'prepared/d_manifest.json'),
                'task_count': len(manifest['tasks'])})
    return {'status': 'released', 'tasks': len(manifest['tasks'])}


def verify_release(root):
    root = Path(root).resolve()
    verify_freeze(root)
    verify_seal(root, 'fits')
    verify_seal(root, 'predictions')
    value = json.loads((root / 'TARGET_RELEASE.json').read_text())
    if value['status'] != 'released':
        raise ValueError('Terminal targets are not released for scoring')
    for key, name in (('fit_lock_sha256', 'FIT_LOCK.json'),
                      ('prediction_lock_sha256', 'PREDICTION_LOCK.json'),
                      ('d_manifest_sha256', 'prepared/d_manifest.json')):
        if value[key] != digest(root / name):
            raise ValueError('Target release provenance differs')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'abc', 'fit', 'seal-fits', 'd',
                        'predict', 'seal-predictions', 'release', 'verify', '_worker'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--phase', choices=('fit', 'predict'))
    parser.add_argument('--task')
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == '_worker':
        if args.cpu is not None:
            os.sched_setaffinity(0, {args.cpu})
        name = 'abc_manifest.json' if args.phase == 'fit' else 'd_manifest.json'
        manifest = verify_manifest(root, name, hash_inputs=False)
        matches = [task for task in manifest['tasks'] if task['task_id'] == args.task]
        if len(matches) != 1:
            raise ValueError('Worker task is outside the prepared roster')
        value = worker_fit(root, matches[0]) if args.phase == 'fit' else worker_predict(root, matches[0])
    elif args.command == 'freeze':
        value = freeze(root)
    elif args.command in ('abc', 'd'):
        value = prepare(root, args.command)
    elif args.command in ('fit', 'predict'):
        value = run_stage(root, args.command, args.workers)
    elif args.command.startswith('seal-'):
        value = seal(root, args.command[5:])
    elif args.command == 'release':
        value = release(root)
    else:
        value = verify_release(root)
    print(json.dumps(safe(value), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
