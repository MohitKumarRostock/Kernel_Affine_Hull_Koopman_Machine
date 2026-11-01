#!/usr/bin/env python3
"""Frozen, staged natural-series policy augmentation evaluation."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT.parent / 'policy_augmentation_20261101'
sys.path.insert(0, str(SYNTHETIC / 'code'))
import pipeline as shared
from insample_model import fit_insample, METHODS, SLOPES, inherited, fd, previous, secondary
from extra_rbf import fit_extra_rbf
from preprocessing_model import fit_primary
import numerical_solver
write_json, write_npz, write_bytes = shared.write_json, shared.write_npz, shared.write_bytes
digest, now, safe, array_hash = shared.digest, shared.now, shared.safe, shared.array_hash


def configuration(root):
    return json.loads((Path(root) / 'config.json').read_text())


def roster(root):
    r = json.loads((Path(root) / 'roster.json').read_text())
    tasks = []
    for freq in ('Weekly', 'Hourly'):
        for row in r['frequencies'][freq]['selected']:
            tid = freq.lower() + '__' + row['series_id']
            seed = int.from_bytes(hashlib.sha256(('natural-policy-fit-v1|' + tid).encode()).digest()[:4], 'little')
            tasks.append({**row, 'task_id': tid, 'frequency': freq, 'fit_seed': seed})
    return tasks


def indices(n, c):
    """Origins are last-observed indices; twelve intervening origins are purged."""
    hmax, d, gap = max(c['horizons']), c['history'], c['purge_origins']
    last = n - hmax - 1
    result = {}
    for split in ('D', 'C', 'B'):
        count = c['rows'][split]
        result[split] = np.arange(last - count + 1, last + 1, dtype=np.int64)
        last = int(result[split][0]) - gap - 1
    result['A'] = np.arange(max(d - 1, last - c['rows']['A_max'] + 1), last + 1, dtype=np.int64)
    if len(result['A']) < c['rows']['A_min']:
        raise ValueError('Insufficient length for the frozen natural split')
    for before, after in zip(('A', 'B', 'C'), ('B', 'C', 'D')):
        if int(result[before][-1]) + hmax >= int(result[after][0]):
            raise AssertionError('Previous targets are not mature at the next block origin')
    if any(np.intersect1d(result[a][:, None] + c['horizons'], result[b][:, None] + c['horizons']).size
           for a, b in (('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D'))):
        raise AssertionError('Target observations overlap between chronological blocks')
    return result


def source_rows(root, tasks):
    """Yield unconverted CSV strings; callers decide which prefix may be parsed."""
    wanted = {t['series_id']: t for t in tasks}
    found = set()
    for freq in ('Weekly', 'Hourly'):
        path = Path(root) / 'input_sources' / (freq + '-train.csv')
        with path.open(newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                sid = row[0]
                if sid not in wanted:
                    continue
                values = list(row[1:])
                while values and not values[-1].strip():
                    values.pop()
                if any(not v.strip() for v in values):
                    raise ValueError('Interior missing observation: chronology cannot be compressed')
                task = wanted[sid]
                if task['frequency'] != freq or len(values) != task['n_observations']:
                    raise ValueError('Length-only source roster changed')
                found.add(sid)
                yield task, values
    if found != set(wanted):
        raise ValueError('The declared natural roster is incomplete in the Train sources')


def freeze(root):
    root = Path(root).resolve()
    if (root / 'prepared').exists():
        raise ValueError('Source freeze must precede parsing study values')
    c = configuration(root)
    tasks = roster(root)
    if len(tasks) != 48 or len({t['series_id'] for t in tasks}) != 48:
        raise ValueError('Natural roster must contain forty-eight unique series')
    prior = {x for ids in json.loads((root / 'roster.json').read_text())['prior_m4_ids'].values() for x in ids}
    if prior & {t['series_id'] for t in tasks}:
        raise ValueError('Natural roster reuses an exposed M4 series')
    for t in tasks:
        ix = indices(t['n_observations'], c)
        if len(ix['A']) != t['A_rows']:
            raise ValueError('A counts disagree with length-only eligibility')
    own = [root / n for n in ('config.json', 'roster.json', 'PROTOCOL.md', 'README.md', 'requirements.txt')]
    own += list((root / 'code').glob('*.py')) + list((root / 'audit').glob('*.py'))
    deps = []
    for folder in ('dimension_forecasting/code', 'dimension_forecasting/runtime',
                   'preprocessing_dimension/code'):
        deps += [p for p in (root.parent / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    deps += [SYNTHETIC / 'code' / f for f in ('pipeline.py', 'crossfit_model.py', 'numerical_solver.py', 'extra_rbf.py')]
    payload = {'status': 'frozen', 'utc': now(), 'source_commit': c['source']['publication_commit'],
        'task_count': 48, 'new_values_parsed': False, 'test_outcomes_scored': False,
        'official_M4_Test_files_downloaded': False, 'public_preregistration': False,
        'files': {str(p.relative_to(root)): digest(p) for p in sorted(set(own))},
        'dependencies': {str(p.relative_to(root.parent)): digest(p) for p in sorted(set(deps))},
        'input_files': {str(p.relative_to(root)): digest(p) for p in sorted((root / 'input_sources').glob('*.csv'))},
        'environment': {'python': sys.version, 'packages': {p: importlib.metadata.version(p)
            for p in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}}}
    write_json(root / 'freeze.json', payload)
    return payload


def verify_freeze(root):
    root = Path(root).resolve()
    f = json.loads((root / 'freeze.json').read_text())
    if f['status'] != 'frozen' or f['task_count'] != len(roster(root)):
        raise ValueError('Source freeze is invalid')
    for base, mapping in ((root, f['files']), (root.parent, f['dependencies']), (root, f['input_files'])):
        for name, expected in mapping.items():
            if digest(base / name) != expected:
                raise ValueError('Frozen input/source changed: ' + name)
    return f


def windows(values, origins, c):
    d = c['history']
    return np.stack([values[t-d+1:t+1] for t in origins]), values[origins[:, None] + c['horizons']]


def direct_predictions(values, origins, period, start, horizons):
    """Seasonal naive and drift query only observations no later than t."""
    h = np.asarray(horizons)
    seasonal_indices = origins[:, None] + h - ((h + period - 1) // period) * period
    if np.any(seasonal_indices > origins[:, None]) or np.any(seasonal_indices < 0):
        raise AssertionError('Seasonal naive accesses an unavailable observation')
    seasonal = values[seasonal_indices]
    denom = origins - start
    if np.any(denom <= 0):
        raise AssertionError('Drift has no historical interval')
    drift = values[origins, None] + h * ((values[origins] - values[start]) / denom)[:, None]
    return {'seasonal_naive': seasonal, 'drift': drift}


def manifest(root, stage):
    m = json.loads((Path(root) / 'prepared' / (stage + '_manifest.json')).read_text())
    if m['freeze_sha256'] != digest(Path(root) / 'freeze.json'):
        raise ValueError('Natural input manifest has a different source freeze')
    expected = {t['task_id'] for t in roster(root)}
    if len(m['tasks']) != 48 or {t['task_id'] for t in m['tasks']} != expected:
        raise ValueError('Natural input manifest has an incomplete roster')
    for t in m['tasks']:
        if digest(Path(root) / t['path']) != t['sha256']:
            raise ValueError('Natural prepared input changed: ' + t['task_id'])
    return m


def prepare(root, stage):
    root = Path(root).resolve()
    verify_freeze(root)
    c = configuration(root)
    if stage == 'abc' and (root / 'prepared').exists():
        raise FileExistsError('Natural ABC already prepared')
    if stage == 'query':
        verify_seal(root, 'fit')
    if stage == 'target':
        verify_seal(root, 'prediction')
    records = []
    for task, strings in source_rows(root, roster(root)):
        ix = indices(task['n_observations'], c)
        start = int(ix['A'][0]) - c['history'] + 1
        if stage == 'abc':
            # Only A/B/C values are converted to numerical arrays at this stage.
            end = int(ix['C'][-1]) + max(c['horizons']) + 1
            values = np.asarray(strings[:end], dtype=np.float64)
            arrays = {}
            for split in ('A', 'B', 'C'):
                arrays[split + '_X'], arrays[split + '_Y'] = windows(values, ix[split], c)
                arrays[split + '_origins'] = ix[split]
            arrays['loss_scale'] = np.maximum(np.var(arrays['A_Y'], axis=0), c['model']['target_variance_floor'])
            train_end = int(ix['A'][-1]) + max(c['horizons']) + 1
            training = values[start:train_end]
            seasonal = int(task['seasonal_period'])
            scale = float(np.mean(np.abs(training[seasonal:] - training[:-seasonal])))
            arrays['mase_scale'] = np.asarray(max(scale, 1e-12))
            arrays['mase_scale_floor_active'] = np.asarray(scale < 1e-12)
            conventional_period = 1 if task['frequency'] == 'Weekly' else 24
            conventional_scale = float(np.mean(np.abs(training[conventional_period:] - training[:-conventional_period])))
            arrays['frequency_mase_scale'] = np.asarray(max(conventional_scale, 1e-12))
            arrays['frequency_mase_scale_floor_active'] = np.asarray(conventional_scale < 1e-12)
            for split in ('B', 'C'):
                arrays.update({split + '_' + k: v for k, v in direct_predictions(
                    values, ix[split], seasonal, start, c['horizons']).items()})
        elif stage == 'query':
            # Earlier D outcomes may enter later observed histories. No D target
            # array or scoring is loaded; fitted models remain fixed throughout.
            values = np.asarray(strings[:int(ix['D'][-1]) + 1], dtype=np.float64)
            arrays = {'D_X': np.stack([values[t-c['history']+1:t+1] for t in ix['D']]), 'D_origins': ix['D']}
            arrays.update({'D_' + k: v for k, v in direct_predictions(values, ix['D'],
                int(task['seasonal_period']), start, c['horizons']).items()})
        elif stage == 'target':
            values = np.asarray(strings, dtype=np.float64)
            arrays = {'D_Y': values[ix['D'][:, None] + c['horizons']], 'D_origins': ix['D']}
        else:
            raise ValueError(stage)
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise FloatingPointError('Nonfinite natural data; no outcome-dependent replacement')
        path = root / 'prepared' / stage / (task['task_id'] + '.npz')
        write_npz(path, arrays)
        records.append({**task, 'path': str(path.relative_to(root)), 'sha256': digest(path),
            'origins': {k: v.tolist() for k, v in ix.items()}, 'history_window_start': start,
            'array_sha256': {k: array_hash(v) for k, v in arrays.items()}, 'stage': stage})
    result = {'stage': stage, 'utc': now(), 'freeze_sha256': digest(root / 'freeze.json'),
        'tasks': records, 'official_M4_Test_files_downloaded': False}
    write_json(root / 'prepared' / (stage + '_manifest.json'), result)
    return {'stage': stage, 'tasks': len(records), 'status': 'prepared'}


def expected_keys():
    from preprocessing_model import METHODS as LEGACY
    keys = {f'legacy__{mode}__{method}' for mode in ('calibrated', 'gamma1') for method in LEGACY}
    keys |= {f'residual__{mode}__{method}' for mode in ('calibrated', 'eta1') for method in METHODS}
    keys |= {f'extra__{mode}__extra_rbf' for mode in ('calibrated', 'eta1')}
    keys |= {f'direct__{mode}__{method}' for mode in ('calibrated', 'eta1') for method in ('seasonal_naive', 'drift')}
    keys |= {f'policy__{mode}__{name}' for mode in ('calibrated', 'eta1') for name in ('P0', 'P1', 'P0plus')}
    return keys


def assemble(legacy, residual, extra, direct):
    arrays = shared.compact_legacy(legacy)
    for mode, key in (('calibrated', 'predictions'), ('eta1', 'eta1_predictions')):
        arrays.update({f'residual__{mode}__{name}': p for name, p in residual[key].items()})
        arrays.update({f'extra__{mode}__{name}': p for name, p in extra[key].items()})
        arrays.update({f'direct__{mode}__{name}': p for name, p in direct.items()})
    return arrays


def worker_fit(root, task):
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve()
    out = root / 'results' / task['task_id']
    out.mkdir(parents=True, exist_ok=False)
    if (root / 'prepared/query_manifest.json').exists():
        raise ValueError('Natural query preparation preceded complete fitting')
    c, wall, cpu = configuration(root), time.perf_counter(), time.process_time()
    numerical_solver.clear_events()
    installation = numerical_solver.install(fd.load_runtime())
    try:
        with np.load(root / task['path'], allow_pickle=False) as z:
            ax, ay, bx, by, loss_scale = [z[k] for k in ('A_X', 'A_Y', 'B_X', 'B_Y', 'loss_scale')]
        with threadpool_limits(limits=1):
            t = time.perf_counter()
            legacy = fit_primary(ax, ay, bx, by, seed=task['fit_seed'], config=c['model'])
            legacy_time = time.perf_counter() - t
            residual = fit_insample(ax, ay, bx, by, seed=task['fit_seed'], config=c['model'], reference_primary=legacy)
            extra = fit_extra_rbf(ax, ay, bx, by, crossfit_suite=residual, seed=task['fit_seed'], config=c['model'])
            write_json(out / 'pre_c_selection.json', {'legacy': {'selections': legacy.selections, 'b_scores': legacy.b_scores},
                'residual': residual.selections, 'extra': extra.selections,
                'residual_frozen_hash': residual.frozen_hash(), 'extra_frozen_hash': extra.frozen_hash(),
                'C_loaded': False, 'utc': now()})
            validation = {'residual__' + k: v for k, v in residual.validation_predictions.items()}
            validation.update({'extra__' + k: v for k, v in extra.validation_predictions.items()})
            validation.update({f'legacy__{key}__{method}': p for key, candidate in legacy.candidates.items()
                for method, p in candidate.b_predictions.items()})
            write_npz(out / 'validation_predictions.npz', validation)
            write_npz(out / 'training_arrays.npz', residual.training_arrays)
            with np.load(root / task['path'], allow_pickle=False) as z:
                cx, cy = z['C_X'], z['C_Y']
                direct = {k: z['C_' + k] for k in ('seasonal_naive', 'drift')}
            lc, _ = legacy.calibrate(cx, cy)
            rc, ec = residual.calibrate(cx, cy), extra.calibrate(cx, cy)
            cp = assemble(lc, rc, ec, direct)
            policies = {name: shared.choose_policy(cp, cy, loss_scale, order)
                for name, order in c['policy_orders'].items()}
            shared.add_policies(cp, policies)
            if set(cp) != expected_keys():
                raise AssertionError('Natural C prediction roster differs')
            write_npz(out / 'calibration_predictions.npz', cp)
            payload = pickle.dumps({'legacy': legacy, 'residual': residual, 'extra': extra, 'policies': policies}, protocol=5)
            write_bytes(out / 'state.pkl.gz', gzip.compress(payload, compresslevel=5, mtime=0))
        events = numerical_solver.events()
        summary = {'status': 'completed', 'task': task, 'utc': now(), 'freeze_sha256': digest(root / 'freeze.json'),
            'input_sha256': digest(root / task['path']), 'legacy': shared.primary_summary(legacy),
            'residual': {'selections': residual.selections, 'diagnostics': residual.diagnostics, 'calibration': rc['records']},
            'extra': {'selections': extra.selections, 'diagnostics': extra.diagnostics, 'calibration': ec['records']},
            'policies': policies, 'numerical_solver': {'installation': installation, 'events': events},
            'resources': {'legacy_fit_wall_seconds': legacy_time, 'total_wall_seconds': time.perf_counter() - wall,
                'total_cpu_seconds': time.process_time() - cpu, 'peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'state_gzip_bytes': (out / 'state.pkl.gz').stat().st_size,
                'scope': 'Complete research suite; includes diagnostic and rejected procedures, not isolated deployment latency'},
            'files': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
        write_json(out / 'fit_summary.json', summary)
        return {'task': task['task_id'], 'status': 'completed', 'seconds': summary['resources']['total_wall_seconds']}
    except Exception:
        write_json(out / 'failure.json', {'utc': now(), 'task': task, 'traceback': traceback.format_exc(),
            'numerical_solver_events': numerical_solver.events()})
        raise


def worker_predict(root, task):
    root = Path(root).resolve()
    out = root / 'predictions' / task['task_id']
    out.mkdir(parents=True, exist_ok=False)
    state = root / 'results' / task['task_id'] / 'state.pkl.gz'
    lock = json.loads((root / 'FIT_LOCK.json').read_text())
    if digest(state) != lock['files'][str(state.relative_to(root))]:
        raise ValueError('Natural fitted state differs from complete fit seal')
    fd.load_runtime()
    suite = pickle.loads(gzip.decompress(state.read_bytes()))
    before = fd.state_hash(suite)
    wall = time.perf_counter()
    with np.load(root / task['path'], allow_pickle=False) as z:
        if 'D_Y' in z:
            raise ValueError('Natural query archive contains terminal targets')
        x = z['D_X']
        direct = {k: z['D_' + k] for k in ('seasonal_naive', 'drift')}
    lp, ld = suite['legacy'].predict(x)
    rp, ep = suite['residual'].predict(x), suite['extra'].predict(x)
    predictions = assemble(lp, rp, ep, direct)
    shared.add_policies(predictions, suite['policies'])
    if set(predictions) != expected_keys() or any(p.shape != (32, 3) or not np.isfinite(p).all() for p in predictions.values()):
        raise AssertionError('Natural terminal predictions are incomplete')
    if fd.state_hash(suite) != before:
        raise AssertionError('Natural terminal query changed fitted state')
    write_npz(out / 'predictions.npz', predictions)
    write_json(out / 'query_diagnostics.json', {'legacy': ld, 'residual': rp['diagnostics'], 'extra': ep['diagnostics']})
    receipt = {'status': 'completed', 'task': task, 'utc': now(), 'state_sha256': digest(state),
        'state_unchanged': True, 'query_sha256': digest(root / task['path']), 'targets_loaded': False,
        'array_count': len(predictions), 'wall_seconds': time.perf_counter() - wall,
        'files': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    write_json(out / 'prediction_receipt.json', receipt)
    return {'task': task['task_id'], 'status': 'completed', 'seconds': receipt['wall_seconds']}


def run(root, phase, workers):
    root = Path(root).resolve()
    verify_freeze(root)
    if phase == 'predict':
        verify_seal(root, 'fit')
    tasks = manifest(root, 'abc' if phase == 'fit' else 'query')['tasks']
    logdir = root / 'execution' / phase
    logdir.mkdir(parents=True, exist_ok=False)
    def launch(task):
        log = logdir / (task['task_id'] + '.log')
        with log.open('x') as stream:
            command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                'worker-' + phase, '--task', task['task_id']]
            # The natural campaign uses a disjoint CPU range while the synthetic
            # campaign is active. Numerical libraries still use one thread.
            allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
            offset = int(configuration(root)['execution']['cpu_offset'])
            cpus = allowed[offset:offset+workers]
            if len(cpus) == workers:
                command = ['taskset', '-c', ','.join(map(str, cpus)), *command]
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Natural task failed; see ' + str(log))
        return task['task_id']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(launch, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            print(json.dumps({'phase': phase, 'completed': i, 'of': len(tasks), 'task': future.result()}), flush=True)
    return seal(root, 'fit' if phase == 'fit' else 'prediction')


def seal(root, phase):
    root = Path(root).resolve()
    verify_freeze(root)
    dirname, receipt = ('results', 'fit_summary.json') if phase == 'fit' else ('predictions', 'prediction_receipt.json')
    expected = {t['task_id'] for t in roster(root)}
    directory = root / dirname
    if {p.name for p in directory.iterdir() if p.is_dir()} != expected:
        raise ValueError('Natural stage has an incomplete or unexpected task roster')
    files = {}
    for tid in sorted(expected):
        summary = json.loads((directory / tid / receipt).read_text())
        if summary['status'] != 'completed' or (directory / tid / 'failure.json').exists():
            raise ValueError('Natural stage contains an incomplete fit')
        for name, sha in summary['files'].items():
            if digest(directory / tid / name) != sha:
                raise ValueError('Natural receipt identity failed')
        files.update({str(p.relative_to(root)): digest(p) for p in (directory / tid).iterdir() if p.is_file()})
    payload = {'status': 'sealed', 'phase': phase, 'utc': now(), 'task_count': 48,
        'freeze_sha256': digest(root / 'freeze.json'), 'files': files}
    if phase == 'prediction':
        verify_seal(root, 'fit')
        payload['fit_lock_sha256'] = digest(root / 'FIT_LOCK.json')
    write_json(root / ('FIT_LOCK.json' if phase == 'fit' else 'PREDICTION_LOCK.json'), payload)
    return {'status': 'sealed', 'phase': phase, 'tasks': 48, 'files': len(files)}


def verify_seal(root, phase):
    root = Path(root).resolve()
    lock = json.loads((root / ('FIT_LOCK.json' if phase == 'fit' else 'PREDICTION_LOCK.json')).read_text())
    if lock['status'] != 'sealed' or lock['task_count'] != 48 or lock['freeze_sha256'] != digest(root / 'freeze.json'):
        raise ValueError('Natural stage seal mismatch')
    for name, sha in lock['files'].items():
        if digest(root / name) != sha:
            raise ValueError('Natural sealed file changed: ' + name)
    return lock


def manufactured_fixture(root):
    """No study values: validate split maturity and residual/selection algebra."""
    c = configuration(root)
    for n in (475, 510, 700, 731, 960, 2597):
        ix = indices(n, c)
        assert len(ix['A']) == min(512, n - 219)
        for split in ('B', 'C', 'D'):
            assert len(ix[split]) == c['rows'][split]
        v = np.arange(n, dtype=float) * 2 + 3
        direct = direct_predictions(v, ix['D'], 24, int(ix['A'][0])-11, c['horizons'])
        np.testing.assert_allclose(direct['drift'], v[ix['D'], None] + 2*np.asarray(c['horizons']))
        np.testing.assert_array_equal(direct['seasonal_naive'], v[ix['D'][:, None] + np.asarray(c['horizons']) - 24])
    rng = np.random.default_rng(9371963)
    ax, bx, cx, dx = [rng.normal(size=(n, 12)) for n in (96, 48, 48, 32)]
    def targets(x):
        return np.column_stack((0.5*x[:, -1] + 0.1*np.sin(x[:, 0]), x[:, 3]*x[:, 2], np.tanh(x[:, 6])))
    ay, by, cy = targets(ax), targets(bx), targets(cx)
    numerical_solver.install(fd.load_runtime())
    legacy = fit_primary(ax, ay, bx, by, seed=7541, config=c['model'])
    residual = fit_insample(ax, ay, bx, by, seed=7541, config=c['model'], reference_primary=legacy)
    extra = fit_extra_rbf(ax, ay, bx, by, crossfit_suite=residual, seed=7541, config=c['model'])
    before = residual.frozen_hash()
    rc = residual.calibrate(cx, cy)
    extra.calibrate(cx, cy)
    legacy.calibrate(cx, cy)
    rp = residual.predict(dx)
    for method in SLOPES:
        state = residual.departures[method]
        raw, _ = inherited._checked_raw(state.bank, ax, 'fixture_verify')
        features = secondary._bounded(fd.load_runtime(), raw[state.kind], state.feature_scaler)
        z = features - features.mean(axis=0)
        y = residual.training_arrays['A_insample_residual']
        y = y - y.mean(axis=0)
        for h, alpha in enumerate(state.head['alpha_by_horizon']):
            oracle = np.linalg.solve(z.T@z + len(z)*alpha*np.eye(z.shape[1]), z.T@y[:, h])
            np.testing.assert_allclose(oracle, state.head['coef'][:, h], atol=2e-8, rtol=2e-7)
    permutation = rng.permutation(len(dx))
    other = residual.predict(dx[permutation])
    for method in METHODS:
        np.testing.assert_allclose(other['predictions'][method], rp['predictions'][method][permutation], atol=1e-12, rtol=1e-12)
    for method in SLOPES:
        residual.etas[method] = np.zeros(3)
    zero = residual.predict(dx)
    for method in SLOPES:
        np.testing.assert_array_equal(zero['predictions'][method], zero['predictions']['shared_bias'])
    assert residual.frozen_hash() == before
    return {'status': 'passed', 'manufactured_values_only': True, 'study_values_parsed': False,
        'split_maturity_and_disjoint_target_checks': True, 'direct_forecast_causality': True,
        'selected_ridge_normal_equations': True, 'zero_correction_recovers_shared_bias': True,
        'query_order_invariance': True, 'A_B_state_unchanged': True,
        'candidate_configurations_per_kind': 15, 'policies': list(c['policy_orders'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('action', choices=('fixture', 'freeze', 'prepare-abc', 'prepare-query', 'release-targets',
        'fit', 'predict', 'worker-fit', 'worker-predict', 'analyze', 'verify'))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--task')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'fixture':
        result = manufactured_fixture(root)
        write_json(root / 'readiness/manufactured_fixture.json', result)
    elif args.action == 'freeze':
        result = freeze(root)
        result = {'status': result['status'], 'tasks': result['task_count']}
    elif args.action in ('prepare-abc', 'prepare-query', 'release-targets'):
        result = prepare(root, {'prepare-abc': 'abc', 'prepare-query': 'query', 'release-targets': 'target'}[args.action])
    elif args.action in ('fit', 'predict'):
        result = run(root, args.action, args.workers)
    elif args.action.startswith('worker-'):
        verify_freeze(root)
        stage = 'abc' if args.action == 'worker-fit' else 'query'
        task = next(t for t in manifest(root, stage)['tasks'] if t['task_id'] == args.task)
        result = (worker_fit if stage == 'abc' else worker_predict)(root, task)
    elif args.action in ('analyze', 'verify'):
        sys.path.insert(0, str(root / 'audit'))
        from analyze_verify import analyze, verify
        result = (analyze if args.action == 'analyze' else verify)(root)
    print(json.dumps(safe(result), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
