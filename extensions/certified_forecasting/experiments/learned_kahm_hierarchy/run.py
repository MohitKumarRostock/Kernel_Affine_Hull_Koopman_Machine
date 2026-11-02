#!/usr/bin/env python3
"""Freeze, then calibrate the block hierarchy on archived learned KAHM banks."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import tarfile
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT.parents[1]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/learned_kahm_decision'))
sys.path.insert(0, str(ROOT / 'experiments/witness_comparison'))
from certified_forecasting.block_witness import block_score_bound
from certified_forecasting.sparse_block_witness import lipschitz_block_brackets
from certified_forecasting.certificates import mean_interval
from run_study import draw, read_gzip_json, restore_bank, source_paths as prior_source_paths
from core import erm_kl_lower


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(native(value), indent=2, allow_nan=False)+'\n')


def write_record(stream, value):
    stream.write(json.dumps(native(value), separators=(',', ':'), allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_paths():
    paths = set(prior_source_paths())
    paths.update([ROOT/'src/certified_forecasting/block_witness.py',
                  ROOT/'src/certified_forecasting/sparse_block_witness.py',
                  ROOT/'experiments/witness_comparison/core.py',
                  ROOT/'tests/test_sparse_block_witness.py'])
    paths.update(EXPERIMENT/name for name in ['run.py', 'protocol.json', 'protocol.md',
                 'development_benchmark.py', 'development_benchmark.json',
                 'sparse_development_benchmark.py', 'sparse_development_benchmark.json'])
    return sorted(paths)


def freeze(output):
    if output.exists():
        raise FileExistsError('Freeze requires a new output directory')
    config = json.loads((EXPERIMENT/'protocol.json').read_text())
    models_path = ROOT/config['frozen_models']
    archived = json.loads((models_path.parent/'development_freeze.json').read_text())
    if digest(models_path) != archived['frozen_models_sha256']:
        raise RuntimeError('Archived feature-bank hash mismatch')
    models = read_gzip_json(models_path)
    expected = [(regime, rep) for regime in config['regimes'] for rep in range(config['repetitions_per_regime'])]
    if [(r['regime'], r['repetition']) for r in models] != expected:
        raise RuntimeError('Unexpected archived bank population or order')
    for record in models:
        if [m['selected']['cost'] for m in record['models']] != config['retained_costs']:
            raise RuntimeError('Unexpected model costs')
    output.mkdir(parents=True)
    write_json(output/'protocol.json', config)
    (output/'protocol.md').write_text((EXPERIMENT/'protocol.md').read_text())
    members = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for path in source_paths():
            contents = path.read_bytes()
            relative = str(path.relative_to(REPOSITORY))
            info = tarfile.TarInfo(relative)
            info.size, info.mtime, info.mode = len(contents), 0, 0o644
            archive.addfile(info, io.BytesIO(contents))
            members.append({'path': relative, 'sha256': hashlib.sha256(contents).hexdigest(), 'bytes': len(contents)})
    (output/'source_snapshot.tar.gz').write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    write_json(output/'source_manifest.json', {'files': members})
    record = {'frozen_at_utc': timestamp(), 'calibration_generated': False,
              'protocol_sha256': digest(output/'protocol.json'),
              'source_snapshot_sha256': digest(output/'source_snapshot.tar.gz'),
              'source_manifest_sha256': digest(output/'source_manifest.json'),
              'models_path': str(models_path.relative_to(ROOT)), 'models_sha256': digest(models_path),
              'archived_development_freeze_sha256': digest(models_path.parent/'development_freeze.json'),
              'banks': expected, 'python': platform.python_version(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'numerical_threads': 1}
    write_json(output/'protocol_freeze.json', record)
    print(json.dumps(record, indent=2), flush=True)


def pair_scores(features, targets, blocks, kappa):
    end = 2*(len(targets)//2)
    distance = np.maximum.reduce([np.linalg.norm(features[:end:2, s]-features[1:end:2, s], axis=1) for s in blocks])
    return .5*np.maximum(np.abs(targets[:end:2]-targets[1:end:2])-kappa*distance, 0.)**2


def stratified(features, targets, labels, classes, blocks, config, eta):
    n = len(targets)
    counts = np.bincount(labels, minlength=classes)
    radius = math.sqrt(math.log(4*classes/eta)/(2*n))
    mass_lower = np.maximum(0., counts/n-radius)
    mass_upper = np.minimum(1., counts/n+radius)
    score_bound = (config['target_bounds'][1]-config['target_bounds'][0])**2/2
    details = []
    for label in range(classes):
        selected = np.flatnonzero(labels == label)
        scores = pair_scores(features[selected], targets[selected], blocks, config['kappa'])
        lower = 0. if len(scores) < 2 else erm_kl_lower(float(scores.mean()), len(scores), eta/(2*classes), score_bound)
        details.append({'stratum': label, 'count': len(selected), 'pairs': len(scores),
                        'mean': float(scores.mean()) if len(scores) else 0., 'lower': lower,
                        'eta': eta/(2*classes), 'bound': score_bound})
    values = np.array([d['lower'] for d in details])
    mass = mass_lower.copy()
    remaining = 1.-float(mass.sum())
    for q in np.argsort(values, kind='stable'):
        add = min(max(remaining, 0.), float(mass_upper[q]-mass[q]))
        mass[q] += add
        remaining -= add
    if abs(remaining) > 1e-9:
        raise ArithmeticError('Mass interval polytope infeasible')
    return {'lower': max(0., float(mass@values)-1e-12), 'counts': counts,
            'mass_radius': radius, 'mass_lower': mass_lower, 'mass_upper': mass_upper,
            'minimizing_mass': mass, 'conditional': details, 'eta': eta}


def solve_batch(features, targets, blocks, config):
    solver = config['solver']
    try:
        result = lipschitz_block_brackets(features, targets, config['kappa'], blocks,
            max_iter=solver['max_iter'], tolerance=solver['tolerance'], check_every=solver['check_every'])
        result['arithmetic_fallback'] = None
        return result
    except (ArithmeticError, FloatingPointError, ValueError) as error:
        count, b = targets.shape
        primal = np.repeat(targets.mean(axis=1)[:, None], b, axis=1)
        sse = np.sum((targets-primal)**2, axis=1)
        high = sse+1e-11*(1.+sse)
        return {'block_size': b, 'lower': np.zeros(count), 'upper': high/(b-1),
                'gap': high/(b-1), 'sse_lower': np.zeros(count), 'sse_upper': high,
                'dual_raw': np.zeros(count), 'dual_guard': np.full(count, 1e-11),
                'signed_dual': np.zeros((count, b*(b-1)//2)), 'primal': primal,
                'iterations': np.zeros(count, dtype=int), 'converged': np.zeros(count, dtype=bool),
                'feasibility_residual': np.zeros(count),
                'arithmetic_fallback': type(error).__name__+': '+str(error)}


def minimum_cost(costs, flags):
    passing = [cost for cost, flag in zip(costs, flags) if flag]
    return min(passing) if passing else None


def summarize_decisions(endpoints, config):
    rows = []
    for regime in config['regimes']:
        for repetition in range(config['repetitions_per_regime']):
            for n in config['calibration_sizes']:
                local = [r for r in endpoints if (r['regime'], r['repetition'], r['n']) == (regime, repetition, n)]
                lookup = {(r['prefix'], r['method']): r for r in local}
                upper = [lookup[p, 'fitted_upper']['upper'] for p in config['prefixes']]
                cost_up = minimum_cost(config['retained_costs'], [u <= config['mse_tolerance'] for u in upper])
                methods = {f'block_{b}': [f'block_{b}'] for b in config['block_sizes']}
                methods['best_block'] = [f'block_{b}' for b in config['block_sizes']]
                methods['stratified_pair'] = [f'stratified_{s}' for s in config['partitions']]
                if n in config['full_erm_sizes']:
                    methods['full_erm'] = ['full_erm']
                methods['all_available'] = list(dict.fromkeys(item for value in methods.values() for item in value))
                for method, sources in methods.items():
                    lowers = [max(lookup[p, m]['lower'] for m in sources) for p in config['prefixes']]
                    cost_low = minimum_cost(config['retained_costs'], [v <= config['mse_tolerance'] for v in lowers])
                    rows.append({'regime': regime, 'repetition': repetition, 'n': n, 'method': method,
                                 'upper_choice': cost_up, 'cost_lower': cost_low,
                                 'lower_cost2': lowers[0], 'lower_cost6': lowers[1],
                                 'minimum_cost_certified': cost_up is not None and cost_up == cost_low,
                                 'all_classes_excluded': cost_low is None,
                                 'upper_accepted': cost_up is not None})
    return rows


def calibrate(output):
    if (output/'calibration_started.json').exists():
        raise FileExistsError('Refusing to overwrite or resume opened calibration')
    config = json.loads((output/'protocol.json').read_text())
    frozen = json.loads((output/'protocol_freeze.json').read_text())
    for field, name in [('protocol_sha256', 'protocol.json'), ('source_snapshot_sha256', 'source_snapshot.tar.gz'),
                        ('source_manifest_sha256', 'source_manifest.json')]:
        if digest(output/name) != frozen[field]:
            raise RuntimeError('Frozen hash mismatch: '+name)
    if digest(ROOT/frozen['models_path']) != frozen['models_sha256']:
        raise RuntimeError('Archived models changed')
    sources = json.loads((output/'source_manifest.json').read_text())['files']
    for item in sources:
        if digest(REPOSITORY/item['path']) != item['sha256']:
            raise RuntimeError('Execution source changed: '+item['path'])
    records = read_gzip_json(ROOT/frozen['models_path'])
    n_max = max(config['calibration_sizes'])
    eta = config['confidence_delta']/config['endpoint_counts']['total']
    loss_bound = (config['target_bounds'][1]-config['target_bounds'][0])**2
    write_json(output/'calibration_started.json', {'started_at_utc': timestamp(),
        'protocol_freeze_sha256': digest(output/'protocol_freeze.json'), 'calibration_generated_before_this_point': False})
    endpoints, timings = [], []
    statistics = {'qp_count': 0, 'unconverged_qp_count': 0, 'arithmetic_fallback_batches': 0,
                  'maximum_normalized_gap': 0., 'maximum_feasibility_residual': 0.}
    started = time.perf_counter()
    with threadpool_limits(limits=1), gzip.open(output/'samples.jsonl.gz', 'wt') as sample_file, \
            gzip.open(output/'qp_certificates.jsonl.gz', 'wt') as certificate_file, \
            gzip.open(output/'stratified_details.jsonl.gz', 'wt') as stratified_file:
        for record in records:
            regime, rep = record['regime'], record['repetition']
            ri = config['regimes'].index(regime)
            seed = config['data_seed_base']+100*ri+rep
            x, y = draw(seed, n_max, regime)
            if np.any(y < config['target_bounds'][0]) or np.any(y > config['target_bounds'][1]):
                raise ArithmeticError('Sample outside declared target bounds')
            bank = restore_bank(record)
            feature_start = time.perf_counter()
            features = bank.transform(x)
            feature_seconds = time.perf_counter()-feature_start
            predictions = np.column_stack([np.clip(features[:, :s['selected']['cost']]@
                np.array(s['selected']['coefficients'])+s['selected']['intercept'], *config['target_bounds']) for s in record['models']])
            labels = {name: np.argmax(features[:, block], axis=1) for name, block in zip(config['partitions'], bank.block_slices_)}
            write_record(sample_file, {'regime': regime, 'repetition': rep, 'seed': seed, 'x': x, 'y': y,
                'features': features, 'predictions': predictions, 'labels': labels,
                'hashes': {key: array_hash(a) for key, a in [('x', x), ('y', y), ('features', features), ('predictions', predictions)]}})
            timings.append({'regime': regime, 'repetition': rep, 'prefix': 0, 'method': 'features', 'n': n_max,
                            'seconds': feature_seconds, 'optimization_programs': 0})
            for pi, prefix in enumerate(config['prefixes']):
                width = config['retained_costs'][pi]
                f = features[:, :width]
                blocks = bank.block_slices_[:prefix]
                base = {'regime': regime, 'repetition': rep, 'prefix': prefix, 'cost': width, 'eta': eta}
                score_sets = {}
                for b in config['block_sizes']:
                    now = time.perf_counter()
                    if b == 2:
                        score_sets[b] = {'lower': pair_scores(f, y, blocks, config['kappa']),
                                         'gap': np.zeros(n_max//2), 'iterations': np.zeros(n_max//2, dtype=int),
                                         'converged': np.ones(n_max//2, dtype=bool), 'feasibility_residual': np.zeros(n_max//2)}
                        seconds = time.perf_counter()-now
                    else:
                        bracket = solve_batch(f.reshape(-1, b, width), y.reshape(-1, b), blocks, config)
                        seconds = time.perf_counter()-now
                        score_sets[b] = bracket
                        write_record(certificate_file, {**base, 'kind': 'block', 'start': 0, 'n': n_max, **bracket})
                        statistics['qp_count'] += len(bracket['lower'])
                        statistics['unconverged_qp_count'] += int((~bracket['converged']).sum())
                        statistics['arithmetic_fallback_batches'] += int(bracket['arithmetic_fallback'] is not None)
                        statistics['maximum_normalized_gap'] = max(statistics['maximum_normalized_gap'], float(bracket['gap'].max()))
                        statistics['maximum_feasibility_residual'] = max(statistics['maximum_feasibility_residual'], float(bracket['feasibility_residual'].max()))
                    timings.append({**base, 'method': f'block_{b}', 'n': n_max, 'seconds': seconds,
                                    'optimization_programs': n_max//b if b > 2 else 0})
                full = {}
                for n in config['full_erm_sizes']:
                    now = time.perf_counter()
                    bracket = solve_batch(f[None, :n], y[None, :n], blocks, config)
                    seconds = time.perf_counter()-now
                    full[n] = bracket
                    write_record(certificate_file, {**base, 'kind': 'full_erm', 'start': 0, 'n': n, **bracket})
                    statistics['qp_count'] += 1
                    statistics['unconverged_qp_count'] += int((~bracket['converged']).sum())
                    statistics['arithmetic_fallback_batches'] += int(bracket['arithmetic_fallback'] is not None)
                    statistics['maximum_normalized_gap'] = max(statistics['maximum_normalized_gap'], float(bracket['gap'].max()))
                    statistics['maximum_feasibility_residual'] = max(statistics['maximum_feasibility_residual'], float(bracket['feasibility_residual'].max()))
                    timings.append({**base, 'method': 'full_erm', 'n': n, 'seconds': seconds, 'optimization_programs': 1})
                for n in config['calibration_sizes']:
                    losses = (y[:n]-predictions[:n, pi])**2
                    upper = mean_interval(losses, loss_bound, eta)
                    endpoints.append({**base, 'n': n, 'method': 'fitted_upper', 'block_size': 1,
                                      'n_independent': n, 'score_mean': upper['mean'], 'lower': '', 'upper': upper['upper'],
                                      'score_bound': loss_bound, 'maximum_gap': 0., 'unconverged_programs': 0})
                    for b in config['block_sizes']:
                        result = score_sets[b]
                        scores = result['lower'][:n//b]
                        bound = block_score_bound(b, math.sqrt(loss_bound))
                        if np.any(scores < 0) or np.any(scores > bound+1e-9):
                            raise ArithmeticError('Block lower score outside declared range')
                        endpoints.append({**base, 'n': n, 'method': f'block_{b}', 'block_size': b,
                            'n_independent': len(scores), 'score_mean': float(scores.mean()),
                            'lower': erm_kl_lower(float(scores.mean()), len(scores), eta, bound), 'upper': '',
                            'score_bound': bound, 'maximum_gap': float(result['gap'][:n//b].max()),
                            'unconverged_programs': int((~result['converged'][:n//b]).sum())})
                    for si, name in enumerate(config['partitions']):
                        detail = stratified(f[:n], y[:n], labels[name][:n], [2, 4][si], blocks, config, eta)
                        write_record(stratified_file, {**base, 'n': n, 'partition': name, **detail})
                        endpoints.append({**base, 'n': n, 'method': 'stratified_'+name, 'block_size': 2,
                            'n_independent': sum(d['pairs'] for d in detail['conditional']), 'score_mean': '',
                            'lower': detail['lower'], 'upper': '', 'score_bound': loss_bound/2,
                            'maximum_gap': 0., 'unconverged_programs': 0})
                    if n in full:
                        bracket = full[n]
                        dual = float(bracket['sse_lower'][0])/n
                        endpoints.append({**base, 'n': n, 'method': 'full_erm', 'block_size': n,
                            'n_independent': n, 'score_mean': dual,
                            'lower': erm_kl_lower(dual, n, eta, loss_bound), 'upper': '', 'score_bound': loss_bound,
                            'maximum_gap': float(bracket['gap'][0])*(n-1)/n,
                            'unconverged_programs': int(not bracket['converged'][0])})
            print(f'Completed {regime} bank {rep+1}/20; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    if len(endpoints) != config['endpoint_counts']['total']:
        raise ArithmeticError('Unexpected endpoint count')
    decisions = summarize_decisions(endpoints, config)
    write_csv(output/'endpoints.csv', endpoints)
    write_csv(output/'decisions.csv', decisions)
    write_csv(output/'timings.csv', timings)
    for item in sources:
        if digest(REPOSITORY/item['path']) != item['sha256']:
            raise RuntimeError('Execution source changed during calibration: '+item['path'])
    summary = {'completed_at_utc': timestamp(), 'scope': config['scope'], 'n_banks': len(records),
               'n_selected_models': 2*len(records), 'n_endpoints': len(endpoints), 'n_decisions': len(decisions),
               'eta_per_endpoint': eta, 'total_allocated_delta': eta*len(endpoints),
               'elapsed_seconds': time.perf_counter()-started, 'source_unchanged': True, **statistics,
               'files_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'run_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'calibrate'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.output)
    else:
        calibrate(args.output)
