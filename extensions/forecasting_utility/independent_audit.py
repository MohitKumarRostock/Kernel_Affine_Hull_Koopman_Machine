"""Independently audit the recorded forecasting-utility extension, without fits.

This check reads frozen inputs and raw saved NPZ/JSON records directly. It does
not import the study analysis or reporting implementation. It recomputes losses,
paired-interval arithmetic and archive weighting, checks paired split nesting,
and preserves the eight model failures and four dependent policy failures.
"""
from pathlib import Path
import argparse
import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import t

root = Path(__file__).resolve().parents[2]
study = root / 'extensions/forecasting_utility'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, default=study / 'results/utility_v1')
parser.add_argument('--output', type=Path, help='Optional JSON audit record; default prints to stdout only')
args = parser.parse_args()
run = args.run_dir.resolve()
freeze = json.loads((study / 'freeze.json').read_text())
for name, checksum in freeze['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum, name
execution = json.loads((run / 'execution.json').read_text())
cfg = json.loads((study / 'config.json').read_text())
report = json.loads((run / 'report/summary.json').read_text())
assert execution['freeze'] == freeze
assert execution['freeze_sha256'] == hashlib.sha256((study / 'freeze.json').read_bytes()).hexdigest()
assert len(execution['expected_tasks']) == 168
rows, records, targets = [], [], {}
failures, policy_failures, successes, gamma_zero = [], [], 0, 0
max_mse_error = 0.0

def ah(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()

for task in execution['expected_tasks']:
    directory = run / 'tasks' / task['task_id']
    record = json.loads((directory / 'result.json').read_text())
    assert record['task'] == task
    assert [m['method'] for m in record['methods']] == cfg['methods']
    assert set(record['policies']) == set(cfg['policies'])
    with np.load(directory / 'targets.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for key, a in arrays.items():
        assert ah(a) == record['data_metadata']['arrays_sha256'][key], (task['task_id'], key)
    y = arrays['Y_test']
    predictions = {}
    by_method = {}
    for m in record['methods']:
        if m['status'] == 'failed':
            assert m['error']
            failures.append((task['task_id'], m['method'], m['error']))
            continue
        assert m['status'] == 'ok'
        with np.load(directory / m['prediction_file'], allow_pickle=False) as saved:
            pred = saved['prediction']
        assert ah(pred) == m['prediction_sha256']
        assert pred.shape == y.shape and np.isfinite(pred).all()
        mse = float(np.mean((y - pred)**2))
        max_mse_error = max(max_mse_error, abs(mse - m['test_mse']))
        np.testing.assert_allclose(mse, m['test_mse'], rtol=1e-13, atol=1e-15)
        predictions[m['method']] = pred
        by_method[m['method']] = {'task_id': task['task_id'], 'phase': task['phase'], 'law': task.get('law', ''),
            'source_id': task.get('source_id', ''), 'series_id': task.get('series_id', ''),
            'seed': task['seed'], 'training_episodes': task.get('training_episodes', 0), 'landmarks': task['landmarks'],
            'method': m['method'], 'test_mse': mse, 'scaled_mse': mse / task.get('loss_scale', 1.0)}
        rows.append(by_method[m['method']])
        if m['method'].startswith('add_') and m['gamma'] == 0:
            assert np.array_equal(pred, predictions['raw_rbf'])
            gamma_zero += 1
        successes += 1
    for name, p in record['policies'].items():
        if p['status'] == 'failed':
            assert p['error']
            policy_failures.append((task['task_id'], name, p['error']))
        else:
            assert p['status'] == 'ok'
            rows.append({**by_method[p['selected_method']], 'method': name})
    targets[task['task_id']] = arrays
    records.append(record)

assert successes == 1336 and len(failures) == 8 and len(policy_failures) == 4
paired = defaultdict(list)
for record in records:
    task = record['task']
    if task['phase'] == 'synthetic':
        paired[(task['law'], task['seed'])].append(record)
for key, group in paired.items():
    assert len(group) == 4
    for split in ['validation', 'test']:
        for prefix in ['X', 'Y', 'groups', 'origins']:
            assert len({r['data_metadata']['arrays_sha256'][f'{prefix}_{split}'] for r in group}) == 1
    for n in [32, 128]:
        size = [r for r in group if r['task']['training_episodes'] == n]
        assert len(size) == 2
        assert size[0]['data_metadata']['arrays_sha256'] == size[1]['data_metadata']['arrays_sha256']
    small = targets[next(r['task']['task_id'] for r in group if r['task']['training_episodes'] == 32)]
    large = targets[next(r['task']['task_id'] for r in group if r['task']['training_episodes'] == 128)]
    keep = large['groups_train'] < 32
    for prefix in ['Y', 'groups', 'origins']:
        np.testing.assert_array_equal(small[f'{prefix}_train'], large[f'{prefix}_train'][keep])
    assert len(np.unique(large['groups_test'])) == 128
    assert set(np.unique(large['groups_test'], return_counts=True)[1]) == {13}

frame = pd.DataFrame(rows)
saved = pd.read_csv(run / 'report/task_methods.csv')
index = ['task_id', 'method']
a = frame.set_index(index).sort_index()
b = saved.set_index(index).sort_index()
assert a.index.equals(b.index)
np.testing.assert_allclose(a[['test_mse', 'scaled_mse']], b[['test_mse', 'scaled_mse']], rtol=1e-12, atol=1e-13)

contrasts = pd.read_csv(run / 'report/synthetic_primary_contrasts.csv')
assert len(contrasts) == 24
for _, row in contrasts.iterrows():
    cell = frame[(frame.phase == 'synthetic') & (frame.law == row.law) &
                 (frame.training_episodes == row.training_episodes) & (frame.landmarks == row.landmarks)]
    table = cell.pivot(index='seed', columns='method', values='test_mse')
    assert list(table.index) == cfg['synthetic']['repetition_seeds']
    d = (table[row.control] - table.add_kahm).to_numpy()
    half = float(t.ppf(1 - .05 / 48, 11) * d.std(ddof=1) / np.sqrt(12))
    np.testing.assert_allclose([d.mean(), d.mean()-half, d.mean()+half],
                              [row.mean_control_minus_additive, row.lower, row.upper], rtol=1e-10, atol=1e-13)

common_ids = {r['task']['task_id'] for r in records if r['task']['phase'] == 'natural' and r['complete']}
assert len(common_ids) == 22
archive = frame[frame.phase == 'natural']
common = archive[archive.task_id.isin(common_ids)]
series = common.groupby(['source_id', 'series_id', 'method'])[['test_mse', 'scaled_mse']].mean()
sources = series.groupby(['source_id', 'method']).mean()
means = sources.groupby('method').mean().sort_index()
saved_common = pd.read_csv(run / 'report/archive_common_support_methods.csv').set_index('method').sort_index()
np.testing.assert_allclose(means, saved_common[['test_mse', 'scaled_mse']], rtol=1e-12, atol=1e-13)
assert report['archive_full_roster_kahm_mean'] is None
assert report['all_methods_succeeded'] is False and report['complete'] is False
assert report['verification_passed'] and report['all_tasks_attempted']
assert report['archive_common_support_tasks'] == 22
summary = {'schema_version': 1, 'verification_passed': True,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'execution_freeze_sha256': hashlib.sha256((study / 'freeze.json').read_bytes()).hexdigest(), 'frozen_files': len(freeze['files']), 'attempted_tasks': len(records),
           'successful_final_forecasts': successes, 'method_failures': len(failures), 'policy_failures': len(policy_failures),
           'paired_law_repetitions_checked': len(paired), 'primary_intervals_independently_recomputed': len(contrasts),
           'common_support_tasks': len(common_ids), 'gamma_zero_exact_equalities': gamma_zero,
           'max_recomputed_mse_difference': max_mse_error, 'forecasting_fits': 0,
           'scope': 'Independent direct NPZ/JSON recomputation; frozen arrays and hashes, paired split nesting, saved losses, simultaneous intervals, source/series/capacity aggregation, and failure counts.'}
content = json.dumps(summary, indent=2, sort_keys=True) + '\n'
if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != content:
        raise FileExistsError('Existing different audit output is preserved')
    args.output.write_text(content)
print(content, end='')
