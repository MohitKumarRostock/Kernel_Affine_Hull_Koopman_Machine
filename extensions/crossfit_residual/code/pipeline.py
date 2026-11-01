#!/usr/bin/env python3
"""Cross-fitted residual study: frozen source through sealed evaluation.

Workers are fresh processes, with one numerical thread each. No production
candidate or failed attempt is silently discarded or overwritten.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
from queue import Queue
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'dimension_forecasting'
PREVIOUS = ROOT.parent / 'preprocessing_dimension'
sys.path.insert(0, str(OLD / 'code'))
sys.path.insert(0, str(PREVIOUS / 'code'))
from study_contract import seed_records, task_roster
from focused_dimension_model import array_hash


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [safe(v) for v in value]
    if isinstance(value, Path): return str(value)
    return value


def write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as f:
        f.write(value); f.flush(); os.fsync(f.fileno())
    if digest(temp) != hashlib.sha256(value).hexdigest():
        raise IOError('Atomic output digest mismatch')
    temp.replace(path)


def write_json(path, value):
    write_bytes(path, (json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def write_npz(path, values):
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    payload = stream.getvalue()
    with np.load(io.BytesIO(payload), allow_pickle=False) as check:
        if set(check.files) != set(values): raise IOError('Array roster changed')
        for k, a in values.items():
            if array_hash(a) != array_hash(check[k]): raise IOError('Array roundtrip changed: ' + k)
    write_bytes(path, payload)


def config(root):
    return json.loads((Path(root) / 'config.json').read_text())


def verify_map(root, files):
    root = Path(root).resolve()
    for name, sha in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != sha:
            raise ValueError('Frozen file missing or changed: ' + name)


def freeze(root):
    root = Path(root).resolve()
    if (root / 'prepared').exists(): raise ValueError('Freeze must precede data preparation')
    source = []
    for directory in ('code', 'analysis_code', 'audit'):
        source.extend(p for p in (root / directory).rglob('*') if p.is_file() and p.suffix in ('.py', '.md'))
    source += [root / 'config.json', root / 'PROTOCOL.md', root / 'README.md', root / 'requirements.txt']
    inherited = []
    for directory in ('code', 'runtime', 'vendor_original'):
        inherited.extend(p for p in (OLD / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    previous = [p for p in (PREVIOUS / 'code').rglob('*.py') if '__pycache__' not in p.parts]
    c = config(root)
    records = seed_records(c)
    prior = json.loads((root / 'readiness/prior_seed_exclusions.json').read_text())
    collision = {r['seed'] for r in records} & {r['seed'] for r in prior['records']}
    if collision: raise ValueError('Seed collision against earlier studies')
    source.append(root / 'readiness/prior_seed_exclusions.json')
    value = {'status': 'frozen', 'created_utc': now(), 'task_count': len(task_roster(c)),
             'config_sha256': digest(root / 'config.json'), 'public_preregistration': False,
             'source_commit': c['source']['publication_commit'],
             'files': {str(p.relative_to(root)): digest(p) for p in sorted(set(source))},
             'inherited_files': {str(p.relative_to(OLD)): digest(p) for p in sorted(set(inherited))},
             'previous_extension_files': {str(p.relative_to(PREVIOUS)): digest(p) for p in sorted(previous)},
             'seed_count': len(records), 'prior_seed_count': len(prior['records']),
             'seed_collision_check': 'passed', 'data_generated': False,
             'environment': {'python': sys.version, 'platform': platform.platform(),
                 'packages': {k: importlib.metadata.version(k) for k in
                     ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl', 'pandas')}}}
    write_json(root / 'freeze.json', value)
    return value


def verify_freeze(root):
    root = Path(root).resolve()
    f = json.loads((root / 'freeze.json').read_text())
    if f['status'] != 'frozen' or f['config_sha256'] != digest(root / 'config.json'):
        raise ValueError('Unfrozen or changed configuration')
    if f['task_count'] != len(task_roster(config(root))): raise ValueError('Task count differs')
    verify_map(root, f['files']); verify_map(OLD, f['inherited_files'])
    verify_map(PREVIOUS, f['previous_extension_files'])
    return f


def verify_manifest(root, name):
    root = Path(root)
    manifest = json.loads((root / 'prepared' / name).read_text())
    if manifest['config_sha256'] != digest(root / 'config.json') or manifest['freeze_sha256'] != digest(root / 'freeze.json'):
        raise ValueError('Prepared data configuration/source identity changed')
    expected = {t['task_id']: t for t in task_roster(config(root))}
    actual = {t['task_id']: t for t in manifest['tasks']}
    if set(actual) != set(expected) or len(actual) != len(manifest['tasks']):
        raise ValueError('Incomplete or duplicated task roster')
    for tid, task in actual.items():
        for key, value in expected[tid].items():
            if task[key] != value: raise ValueError('Task metadata differs: ' + tid)
        for field, sha in (('input', 'sha256'), ('query', 'query_sha256'), ('target', 'target_sha256')):
            if field in task and digest(root / task[field]) != task[sha]:
                raise ValueError('Prepared task bytes changed: ' + tid + '/' + field)
    verify_map(root, manifest['base_files'])
    return manifest


def prepare(root, stage):
    root = Path(root).resolve(); verify_freeze(root)
    c = config(root)
    if stage == 'abc':
        if (root / 'prepared').exists(): raise FileExistsError('Preparation already started')
        prior = json.loads((root / 'readiness/prior_seed_exclusions.json').read_text())
        if {r['seed'] for r in seed_records(c)} & {r['seed'] for r in prior['records']}:
            raise ValueError('Prior seed collision')
    else:
        verify_seal(root, 'fits')
        abc = verify_manifest(root, 'abc_manifest.json')
        if (root / 'prepared/d_manifest.json').exists(): raise FileExistsError('D already generated')
    # Generator and forecasting extension packages are intentionally loaded in
    # distinct CLI processes, preserving both verified source identities.
    import prepare_dimension_data as original
    module = original.generator_module()
    records, tasks = seed_records(c), task_roster(c)
    if stage == 'abc':
        write_json(root / 'prepared/seed_manifest.json', {'records': records, 'seed_count': len(records),
                     'D_streams_reserved_but_not_generated': True})
    result, base_files = [], {}
    for rep in range(c['repetitions']):
        for law in c['laws']:
            splits = ('A', 'B', 'C') if stage == 'abc' else ('D',)
            bases, units, streams, saved = {}, {}, {}, {}
            for split in splits:
                streams[split] = original._streams(records, rep, law, split)
                bases[split], units[split] = original._base(c, law, split, streams[split], module)
                saved[split + '_base_episodes'] = bases[split]
                saved[split + '_unit_noise'] = units[split]
            relative_base = f'prepared/{"base_abc" if stage == "abc" else "withheld_d_base"}/{law}__rep{rep:02d}.npz'
            write_npz(root / relative_base, saved)
            base_files[relative_base] = digest(root / relative_base)
            roster = tasks if stage == 'abc' else abc['tasks']
            for task in roster:
                if task['repetition'] != rep or task['law'] != law: continue
                amplitude = (float(np.std(bases['A'][:, :task['history'] + c['history_tail_length']], ddof=0))
                             if stage == 'abc' else task['A_base_episode_std'])
                arrays, split_records = {}, {}
                for split in splits:
                    windows, retained = original._windows(c, bases[split], units[split], task['history'],
                        task['noise_level'], amplitude, split, streams[split], module)
                    arrays.update({split + '_' + k: v for k, v in windows.items()})
                    split_records[split] = {'streams': streams[split], 'retained_rows': len(retained),
                        'retained_indices_sha256': array_hash(retained),
                        'base_sha256': array_hash(bases[split]), 'unit_noise_sha256': array_hash(units[split])}
                if stage == 'abc':
                    arrays['loss_scale'] = np.maximum(np.var(arrays['A_Y'], axis=0, ddof=0), c['model']['target_variance_floor'])
                    rel = f'prepared/abc/{task["task_id"]}.npz'; write_npz(root / rel, arrays)
                    result.append({**task, 'input': rel, 'sha256': digest(root / rel),
                        'loss_scale': arrays['loss_scale'].tolist(), 'A_base_episode_std': amplitude,
                        'splits': split_records, 'base_file': relative_base,
                        'array_sha256': {k: array_hash(v) for k, v in arrays.items()}})
                else:
                    query = f'prepared/queries/{task["task_id"]}.npz'
                    target = f'prepared/targets/{task["task_id"]}.npz'
                    metadata = {k: v for k, v in arrays.items() if k not in ('D_X', 'D_Y')}
                    write_npz(root / query, {'D_X': arrays['D_X'], **metadata})
                    write_npz(root / target, {'D_Y': arrays['D_Y'], **metadata})
                    result.append({**{k: task[k] for k in tasks[0]}, 'query': query, 'target': target,
                        'query_sha256': digest(root / query), 'target_sha256': digest(root / target),
                        'rows': len(arrays['D_X']), 'D_X_sha256': array_hash(arrays['D_X']),
                        'D_Y_sha256': array_hash(arrays['D_Y']), 'splits': split_records,
                        'A_base_episode_std': amplitude, 'base_file': relative_base})
        print(json.dumps({'phase': stage, 'repetitions_prepared': rep + 1, 'tasks': len(result)}), flush=True)
    manifest = {'schema': 'crossfit-residual-' + stage, 'created_utc': now(),
                'config_sha256': digest(root / 'config.json'), 'freeze_sha256': digest(root / 'freeze.json'),
                'tasks': result, 'task_count': len(result), 'base_files': base_files,
                'D_generated': stage == 'd'}
    if stage == 'd': manifest['fit_lock_sha256'] = digest(root / 'FIT_LOCK.json')
    write_json(root / 'prepared' / ('abc_manifest.json' if stage == 'abc' else 'd_manifest.json'), manifest)
    return {'tasks': len(result), 'phase': stage}


def primary_summary(primary):
    return {'selections': primary.selections, 'b_scores': primary.b_scores,
        'raw_rbf_selected_scaler': primary.raw_rbf_selected_scaler,
        'diagnostics': primary.diagnostics,
        'calibration': primary.gammas,
        'context_policies': {k: v['selected_method'] for k, v in primary.policies.items()},
        'calibration_records': primary.calibration,
        'banks': {k: v.diagnostics for k, v in primary.banks.items()},
        'candidates': {k: v.diagnostics for k, v in primary.candidates.items()}}


def compact_legacy(arrays):
    prefix = 'primary__selected__'
    out = {'legacy__' + key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
    if len(out) != 20:
        raise AssertionError('Legacy selected-context roster must contain twenty arrays')
    return out


def policy_key(identity, mode):
    family, method = identity.split('__', 1)
    actual_mode = ('gamma1' if family == 'legacy' else 'eta1') if mode == 'eta1' else 'calibrated'
    return f'{family}__{actual_mode}__{method}'


def choose_policy(arrays, cy, loss_scale, order):
    scores = [{'identity': identity, 'score': float(np.mean(np.mean((cy - arrays[policy_key(identity, 'calibrated')]) ** 2, axis=0) / loss_scale)),
               'mse_by_horizon': np.mean((cy - arrays[policy_key(identity, 'calibrated')]) ** 2, axis=0).tolist(),
               'C_prediction_sha256': array_hash(arrays[policy_key(identity, 'calibrated')])}
              for identity in order]
    index = int(np.argmin([row['score'] for row in scores]))
    return {'selected_identity': order[index], 'candidates': scores, 'selection_split': 'C',
            'loss_scale': np.asarray(loss_scale).tolist(), 'tie_rule': 'First minimum in declared order',
            'eta1_selection': 'Same calibrated C-selected identity; no reselection',
            'C_Y_sha256': array_hash(cy)}


def add_policy(arrays, policy):
    for mode in ('calibrated', 'eta1'):
        arrays[f'policy__{mode}__expanded_nonkahm'] = arrays[policy_key(policy['selected_identity'], mode)].copy()
    return arrays


def worker_fit(root, task):
    from preprocessing_model import fit_primary
    from crossfit_model import fit_crossfit
    from threadpoolctl import threadpool_limits
    root = Path(root); output = root / 'results' / task['task_id']
    output.mkdir(parents=True, exist_ok=False)
    if (root / 'prepared/d_manifest.json').exists(): raise ValueError('D exists before fits')
    c = config(root)
    started, wall, cpu = now(), time.perf_counter(), time.process_time()
    try:
        with np.load(root / task['input'], allow_pickle=False) as z:
            ax, ay, bx, by, groups = [z[k] for k in ('A_X', 'A_Y', 'B_X', 'B_Y', 'A_groups')]
            loss_scale = z['loss_scale']
        with threadpool_limits(limits=1):
            t, p = time.perf_counter(), time.process_time()
            legacy = fit_primary(ax, ay, bx, by, seed=task['fit_seed'], config=c['model'])
            ltime = {'wall_seconds': time.perf_counter() - t, 'cpu_seconds': time.process_time() - p}
            t, p = time.perf_counter(), time.process_time()
            crossfit = fit_crossfit(ax, ay, groups, bx, by, seed=task['fit_seed'], config=c['model'], reference_primary=legacy)
            xtime = {'wall_seconds': time.perf_counter() - t, 'cpu_seconds': time.process_time() - p}
            write_json(output / 'pre_c_selection.json', {'legacy': {'selections': legacy.selections,
                'b_scores': legacy.b_scores, 'raw_rbf_selected_scaler': legacy.raw_rbf_selected_scaler},
                'crossfit': crossfit.selections, 'crossfit_frozen_state_sha256': crossfit.frozen_hash(),
                'C_loaded': False, 'saved_utc': now()})
            validation = {f'legacy__{key}__{method}': arr for key, candidate in legacy.candidates.items()
                          for method, arr in candidate.b_predictions.items()}
            validation.update({'crossfit__' + key: value for key, value in crossfit.validation_predictions.items()})
            write_npz(output / 'validation_predictions.npz', validation)
            write_npz(output / 'training_arrays.npz', crossfit.training_arrays)
            with np.load(root / task['input'], allow_pickle=False) as z:
                cx, cy = z['C_X'], z['C_Y']
            t, p = time.perf_counter(), time.process_time()
            legacy_c, _ = legacy.calibrate(cx, cy)
            cp = compact_legacy(legacy_c)
            crossfit_c = crossfit.calibrate(cx, cy)
            for mode, key in (('calibrated', 'predictions'), ('eta1', 'eta1_predictions')):
                cp.update({f'crossfit__{mode}__{name}': value for name, value in crossfit_c[key].items()})
            policy = choose_policy(cp, cy, loss_scale, c['policy_order'])
            add_policy(cp, policy)
            if set(cp) != expected_keys(): raise AssertionError('Calibration roster differs')
            ctime = {'wall_seconds': time.perf_counter() - t, 'cpu_seconds': time.process_time() - p}
            write_npz(output / 'calibration_predictions.npz', cp)
            basis = {}
            for key, bank in legacy.banks.items():
                for i, local in enumerate(bank.bank['models']):
                    for name in ('q', 'p', 'center'):
                        basis[f'legacy__{key}__local{i}__{name}'] = local[name]
            basis.update({'crossfit__' + key: value for key, value in crossfit.basis_arrays.items()})
            write_npz(output / 'basis_diagnostics.npz', basis)
            payload = pickle.dumps({'legacy': legacy, 'crossfit': crossfit, 'policy': policy}, protocol=5)
            write_bytes(output / 'state.pkl.gz', gzip.compress(payload, compresslevel=5, mtime=0))
        summary = {'status': 'completed', 'task': task, 'started_utc': started, 'completed_utc': now(),
            'freeze_sha256': digest(root / 'freeze.json'), 'input_sha256': digest(root / task['input']),
            'legacy': primary_summary(legacy),
            'crossfit': {'selections': crossfit.selections, 'diagnostics': crossfit.diagnostics,
                         'calibration': crossfit_c['records']}, 'policy': policy,
            'resources': {'legacy_fit_selection': ltime, 'crossfit_fit_selection': xtime,
                'calibration_and_policy': ctime,
                'total_wall_seconds': time.perf_counter() - wall, 'total_cpu_seconds': time.process_time() - cpu,
                'peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'state_pickle_bytes': len(payload), 'state_gzip_bytes': (output / 'state.pkl.gz').stat().st_size},
            'files': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
        write_json(output / 'fit_summary.json', summary)
        return {'task': task['task_id'], 'status': 'completed', 'seconds': summary['resources']['total_wall_seconds']}
    except Exception:
        write_json(output / 'failure.json', {'status': 'failed', 'task': task, 'utc': now(), 'traceback': traceback.format_exc()})
        raise


def worker_predict(root, task):
    from preprocessing_model import fd
    import crossfit_model
    from threadpoolctl import threadpool_limits
    root = Path(root); output = root / 'predictions' / task['task_id']
    output.mkdir(parents=True, exist_ok=False)
    state_path = root / 'results' / task['task_id'] / 'state.pkl.gz'
    lock = json.loads((root / 'FIT_LOCK.json').read_text())
    if digest(state_path) != lock['files'][str(state_path.relative_to(root))]: raise ValueError('Fitted state changed')
    fd.load_runtime()
    suite = pickle.loads(gzip.decompress(state_path.read_bytes()))
    loaded_state_hash = fd.state_hash(suite)
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        with np.load(root / task['query'], allow_pickle=False) as z:
            if any('Y' in k for k in z.files): raise ValueError('Query archive contains targets')
            x = z['D_X']
        with threadpool_limits(limits=1):
            legacy, dl = suite['legacy'].predict(x)
            crossfit = suite['crossfit'].predict(x)
        arrays = compact_legacy(legacy)
        for mode, key in (('calibrated', 'predictions'), ('eta1', 'eta1_predictions')):
            arrays.update({f'crossfit__{mode}__{name}': value for name, value in crossfit[key].items()})
        add_policy(arrays, suite['policy'])
        if set(arrays) != expected_keys(): raise AssertionError('Prediction array roster differs')
        if any(a.shape != (len(x), 3) or not np.isfinite(a).all() for a in arrays.values()):
            raise FloatingPointError('Incomplete prediction matrix')
        if fd.state_hash(suite) != loaded_state_hash:
            raise AssertionError('Query mutated whole fitted state')
        write_npz(output / 'predictions.npz', arrays)
        write_json(output / 'query_diagnostics.json', {'legacy': dl, 'crossfit': crossfit['diagnostics']})
        receipt = {'status': 'completed', 'task': task, 'utc': now(), 'array_count': len(arrays),
            'state_sha256': digest(state_path), 'state_unchanged': True, 'query_sha256': digest(root / task['query']),
            'prediction_sha256': digest(output / 'predictions.npz'), 'targets_loaded': False,
            'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
            'files': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
        write_json(output / 'prediction_receipt.json', receipt)
        return {'task': task['task_id'], 'status': 'completed', 'seconds': receipt['wall_seconds']}
    except Exception:
        write_json(output / 'failure.json', {'status': 'failed', 'task': task, 'utc': now(), 'traceback': traceback.format_exc()})
        raise


def expected_keys():
    from preprocessing_model import METHODS
    from crossfit_model import METHODS as CROSSFIT_METHODS
    keys = {f'legacy__{mode}__{method}' for mode in ('calibrated', 'gamma1') for method in METHODS}
    keys |= {f'crossfit__{mode}__{method}' for mode in ('calibrated', 'eta1') for method in CROSSFIT_METHODS}
    keys |= {f'policy__{mode}__expanded_nonkahm' for mode in ('calibrated', 'eta1')}
    if len(keys) != 42: raise AssertionError('The frozen study requires exactly42 logical arrays')
    return keys


def run_stage(root, phase, workers):
    root = Path(root).resolve(); verify_freeze(root)
    if workers < 1 or workers > 3: raise ValueError('Frozen budget is one to three workers')
    if phase == 'predict': verify_seal(root, 'fits')
    manifest = verify_manifest(root, 'abc_manifest.json' if phase == 'fit' else 'd_manifest.json')
    tasks = manifest['tasks']
    directory = root / 'execution' / phase
    directory.mkdir(parents=True, exist_ok=False)
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    slots = Queue()
    for i in range(workers): slots.put(cores[i] if cores else None)
    started, start = now(), time.perf_counter()
    def run(task, index):
        cmd = [sys.executable, str(Path(__file__).resolve()), '_worker', '--root', str(root),
               '--phase', phase, '--task', task['task_id']]
        core = slots.get()
        try:
            if core is not None: cmd += ['--cpu', str(core)]
            with (directory / (task['task_id'] + '.log')).open('x') as stream:
                proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        finally:
            slots.put(core)
        if proc.returncode: raise RuntimeError(f'{task["task_id"]}: worker exited {proc.returncode}')
        return task['task_id']
    completed = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, task, i): task for i, task in enumerate(tasks)}
            for future in as_completed(futures):
                try: completed.append(future.result())
                except Exception:
                    for pending in futures: pending.cancel()
                    raise
                if len(completed) % 8 == 0 or len(completed) == len(tasks):
                    print(json.dumps({'phase': phase, 'completed': len(completed), 'total': len(tasks),
                                      'elapsed_seconds': time.perf_counter() - start}), flush=True)
    finally:
        write_json(directory / 'stage_receipt.json', {'phase': phase, 'started_utc': started,
            'completed_utc': now(), 'completed_tasks': sorted(completed), 'total_expected': len(tasks),
            'status': 'completed' if len(completed) == len(tasks) else 'failed',
            'workers': workers, 'one_numerical_thread_per_worker': True,
            'wall_seconds': time.perf_counter() - start})


def check_task_receipt(root, phase, expected, data):
    root = Path(root)
    tid = expected['task_id']
    if data['status'] != 'completed' or data['task'] != expected:
        raise ValueError('Task receipt differs from current prepared manifest: ' + tid)
    if phase == 'fits':
        if data['freeze_sha256'] != digest(root / 'freeze.json') or data['input_sha256'] != expected['sha256']:
            raise ValueError('Fitted task provenance differs: ' + tid)
    else:
        fit_lock = json.loads((root / 'FIT_LOCK.json').read_text())
        state_key = f'results/{tid}/state.pkl.gz'
        if data['state_sha256'] != fit_lock['files'][state_key] or data['query_sha256'] != expected['query_sha256']:
            raise ValueError('Prediction state/query provenance differs: ' + tid)
        if (data['array_count'] != len(expected_keys()) or data['targets_loaded'] is not False
                or data['state_unchanged'] is not True):
            raise ValueError('Invalid query phase receipt: ' + tid)
        if data['prediction_sha256'] != digest(root / 'predictions' / tid / 'predictions.npz'):
            raise ValueError('Prediction receipt digest differs: ' + tid)


def seal(root, phase):
    root = Path(root).resolve(); verify_freeze(root)
    manifest = verify_manifest(root, 'abc_manifest.json' if phase == 'fits' else 'd_manifest.json')
    tasks = manifest['tasks']; files = {}
    if phase == 'predictions': verify_seal(root, 'fits')
    for task in tasks:
        folder = root / ('results' if phase == 'fits' else 'predictions') / task['task_id']
        receipt = folder / ('fit_summary.json' if phase == 'fits' else 'prediction_receipt.json')
        data = json.loads(receipt.read_text())
        check_task_receipt(root, phase, task, data)
        if (folder / 'failure.json').exists():
            raise ValueError('Incomplete/failed task cannot be sealed')
        verify_map(folder, data['files'])
        files.update({str(p.relative_to(root)): digest(p) for p in folder.iterdir() if p.is_file()})
    value = {'status': 'sealed', 'phase': phase, 'created_utc': now(), 'task_count': len(tasks),
        'task_ids': [t['task_id'] for t in tasks], 'files': files,
        'config_sha256': digest(root / 'config.json'), 'freeze_sha256': digest(root / 'freeze.json')}
    write_json(root / ('FIT_LOCK.json' if phase == 'fits' else 'PREDICTION_LOCK.json'), value)
    return {'phase': phase, 'files': len(files), 'tasks': len(tasks)}


def verify_seal(root, phase):
    root = Path(root)
    value = json.loads((root / ('FIT_LOCK.json' if phase == 'fits' else 'PREDICTION_LOCK.json')).read_text())
    if value['status'] != 'sealed' or value['phase'] != phase: raise ValueError('Incorrect seal phase')
    if value['config_sha256'] != digest(root / 'config.json') or value['freeze_sha256'] != digest(root / 'freeze.json'):
        raise ValueError('Seal provenance changed')
    expected = {t['task_id'] for t in task_roster(config(root))}
    if set(value['task_ids']) != expected or len(value['task_ids']) != len(expected) or value['task_count'] != len(expected):
        raise ValueError('Seal task roster differs')
    required = (('fit_summary.json', 'state.pkl.gz', 'pre_c_selection.json', 'validation_predictions.npz',
                 'calibration_predictions.npz', 'basis_diagnostics.npz', 'training_arrays.npz') if phase == 'fits' else
                ('predictions.npz', 'prediction_receipt.json', 'query_diagnostics.json'))
    folder = 'results' if phase == 'fits' else 'predictions'
    for tid in expected:
        if any(f'{folder}/{tid}/{name}' not in value['files'] for name in required):
            raise ValueError('Seal is missing canonical task artifacts: ' + tid)
    verify_map(root, value['files'])
    manifest = verify_manifest(root, 'abc_manifest.json' if phase == 'fits' else 'd_manifest.json')
    receipt_name = 'fit_summary.json' if phase == 'fits' else 'prediction_receipt.json'
    for task in manifest['tasks']:
        data = json.loads((root / folder / task['task_id'] / receipt_name).read_text())
        check_task_receipt(root, phase, task, data)
    return value


def release(root):
    root = Path(root); verify_freeze(root); verify_seal(root, 'fits'); verify_seal(root, 'predictions')
    manifest = verify_manifest(root, 'd_manifest.json')
    write_json(root / 'TARGET_RELEASE.json', {'status': 'released', 'created_utc': now(),
        'fit_lock_sha256': digest(root / 'FIT_LOCK.json'),
        'prediction_lock_sha256': digest(root / 'PREDICTION_LOCK.json'),
        'd_manifest_sha256': digest(root / 'prepared/d_manifest.json'),
        'task_count': len(manifest['tasks'])})
    return {'status': 'released', 'tasks': len(manifest['tasks'])}


def verify_release(root):
    root = Path(root).resolve(); verify_freeze(root); verify_seal(root, 'fits'); verify_seal(root, 'predictions')
    r = json.loads((root / 'TARGET_RELEASE.json').read_text())
    for field, name in (('fit_lock_sha256', 'FIT_LOCK.json'), ('prediction_lock_sha256', 'PREDICTION_LOCK.json'),
                        ('d_manifest_sha256', 'prepared/d_manifest.json')):
        if r[field] != digest(root / name): raise ValueError('Release provenance changed')
    verify_manifest(root, 'abc_manifest.json'); verify_manifest(root, 'd_manifest.json')
    if r['status'] != 'released': raise ValueError('Targets not released')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'abc', 'fit', 'seal-fits', 'd', 'predict',
                   'seal-predictions', 'release', 'verify', '_worker'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--phase', choices=('fit', 'predict'))
    p.add_argument('--task'); p.add_argument('--cpu', type=int)
    a = p.parse_args(); root = a.root.resolve()
    if a.command == '_worker':
        if a.cpu is not None: os.sched_setaffinity(0, {a.cpu})
        manifest = json.loads((root / 'prepared' / ('abc_manifest.json' if a.phase == 'fit' else 'd_manifest.json')).read_text())
        task = next(t for t in manifest['tasks'] if t['task_id'] == a.task)
        out = worker_fit(root, task) if a.phase == 'fit' else worker_predict(root, task)
    elif a.command == 'freeze': out = freeze(root)
    elif a.command in ('abc', 'd'): out = prepare(root, a.command)
    elif a.command in ('fit', 'predict'): out = run_stage(root, a.command, a.workers)
    elif a.command.startswith('seal-'): out = seal(root, a.command[5:])
    elif a.command == 'release': out = release(root)
    else: out = verify_release(root)
    if out is not None:
        print(json.dumps(safe(out) if a.command != 'freeze' else {'status': 'frozen', 'files': len(out['files']),
              'inherited_files': len(out['inherited_files']), 'tasks': out['task_count']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
