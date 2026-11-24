#!/usr/bin/env python3
"""Verify content, recompute numerical checks, or regenerate executable studies."""
from __future__ import annotations
import argparse
import csv
import math
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    files = manifest['files']
    # Generated local environments, run outputs, and Python caches are not distributed.
    allowed_generated = {'.git', '.venv', 'runs', '__pycache__', '.pytest_cache'}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and not any(part in allowed_generated for part in p.relative_to(ROOT).parts)}
    expected = set(files) | {'MANIFEST.json'}
    require(actual == expected, f'Distribution inventory mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    # A Git checkout must not track additional files, including generated paths.
    if (ROOT / '.git').exists():
        tracked = set(subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().rstrip('\0').split('\0'))
        require(tracked == expected, f'Tracked inventory mismatch: missing={sorted(expected-tracked)}, extra={sorted(tracked-expected)}')
    for name, record in files.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), f'Invalid manifest path: {name}')
        require(path.is_file(), f'Missing file: {name}')
        require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'],
                f'Content identity mismatch: {name}')
    return {'status': 'PASS', 'manifest_entries': len(files), 'scope': 'Complete distribution inventory and content identity'}


def copy_stage(destination):
    shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns('.git', '.venv', 'runs', '__pycache__', '.pytest_cache'))


def command(stage, *args, log=None):
    cmd = [sys.executable, '-B', *map(str, args)]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    p = subprocess.run(cmd, cwd=stage, env=env, capture_output=True, text=True)
    if log is not None:
        log.write_text(p.stdout + '\n' + p.stderr)
    require(p.returncode == 0, f'Command failed: {args}\n{p.stdout}\n{p.stderr}')
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {'status': 'PASS', 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}


def checks(stage):
    result = {}
    for name, path in [('matched_dimension', 'scripts/check_matched_dimension.py'),
                       ('weighted_recovery', 'scripts/check_weighted_recovery.py'),
                       ('archived_forecast', 'reproduction/score_enrichment/check_saved_losses.py'),
                       ('capacity_summary', 'scripts/check_capacity_summary.py'),
                       ('ablation', 'ablation/summarize.py'),
                       ('observed_series', 'real_data/check_results.py')]:
        result[name] = command(stage, path)
        require(result[name].get('status') == 'PASS', f'Failed {name}')
    result['archive_tests'] = command(stage, '-m', 'unittest', 'discover', '-s', 'reproduction/score_enrichment/tests', '-v')
    require(result['matched_dimension']['task_condition_cases'] == 1350, 'Finite-law roster differs')
    require(result['matched_dimension']['optimal_contrast_checks'] == 54, 'Contrast roster differs')
    require(result['weighted_recovery']['weighted_recovery_cases'] == 300, 'Recovery roster differs')
    require(result['weighted_recovery']['eight_point_tradeoff_cases'] == 270, 'Inverse roster differs')
    require(result['archived_forecast']['primary_comparisons_rebuilt'] == 40, 'Archive comparison count differs')
    require(result['ablation']['primary_directions'] == {'lower': 4, 'higher': 6, 'unresolved': 2}, 'Interval decisions differ')
    require(result['observed_series']['weighted_higher_validation'] == result['observed_series']['weighted_higher_matched'] == 20,
            'Observed-series sensitivity counts differ')
    return result


# Identifiers, selections, ranks, counts, and categorical fields must match exactly.
# Only declared numerical fields receive a floating-point tolerance.
FLOAT_FIELDS = {'penalty', 'df', 'loss', 'h1', 'h4', 'h12', 'local', 'sep', 'difference',
                'lower', 'upper', 'cluster_accuracy', 'positive_margin_fraction', 'median_margin',
                'median_weighted_sigma', 'min_weighted_sigma', 'min_membership', 'val_smse',
                'smse', 'rmse1', 'rmse4', 'rmse12'}
RTOL, ATOL = 1e-10, 1e-12


def numerical_agreement(reference, generated):
    if reference.suffix == '.csv':
        with reference.open(newline='') as f, generated.open(newline='') as g:
            a, b = csv.DictReader(f), csv.DictReader(g)
            if a.fieldnames != b.fieldnames:
                return False, None
            a, b = list(a), list(b)
        if len(a) != len(b):
            return False, None
        err = 0.0
        for x, y in zip(a, b):
            for key, val in x.items():
                other = y[key]
                if val == other:
                    continue
                if key not in FLOAT_FIELDS or not val or not other:
                    return False, None
                u, v = float(val), float(other)
                err = max(err, abs(u-v))
                if not math.isclose(u, v, rel_tol=RTOL, abs_tol=ATOL):
                    return False, err
        return True, err
    if reference.suffix == '.npz':
        import numpy as np
        err = 0.0
        with np.load(reference, allow_pickle=False) as a, np.load(generated, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                return False, None
            for key in a.files:
                x, y = a[key], b[key]
                if x.shape != y.shape or x.dtype != y.dtype:
                    return False, None
                if key == 'predicted_change':
                    err = max(err, float(np.max(np.abs(x-y))))
                    if not np.allclose(x, y, rtol=RTOL, atol=ATOL):
                        return False, err
                elif not np.array_equal(x, y):
                    return False, None
        return True, err
    return False, None


def compare_replay(stage, reports):
    expected = {'ablation': json.loads((ROOT / 'ablation/replay_verification.json').read_text())['files'],
                'real_data': json.loads((ROOT / 'real_data/replay_verification.json').read_text())}
    comparisons = {}
    for group, records in expected.items():
        for name, record in records.items():
            path = f'{group}/{name}'
            actual = digest(stage / path)
            identical = actual == record['sha256']
            agrees, error = (True, 0.0) if identical else numerical_agreement(ROOT/path, stage/path)
            comparisons[path] = {'expected_sha256': record['sha256'], 'actual_sha256': actual,
                                 'identical': identical, 'numerically_agrees': agrees,
                                 'maximum_absolute_difference': error}
    required = {k: v for k, v in comparisons.items() if k != 'real_data/metadata.json'}
    passed = all(v['numerically_agrees'] for v in required.values())
    return {'status': 'PASS' if passed else 'FAIL', 'integrity': verify(), 'checks': reports,
            'replay_comparisons': comparisons, 'rtol': RTOL, 'atol': ATOL,
            'scope': 'Executable studies only; metadata equality reported separately'}


def replay(out):
    out = out.resolve()
    require(not out.exists(), 'Choose a new output directory.')
    require(not ROOT.is_relative_to(out), 'Output cannot be an ancestor of the distribution.')
    # Avoid copying a source tree into itself; only runs/ is excluded by copy_stage.
    if out.is_relative_to(ROOT):
        require(out.relative_to(ROOT).parts[0] == 'runs', 'Use runs/ or an external output directory.')
    out.mkdir(parents=True)
    stage = out / 'study'
    copy_stage(stage)
    command(stage, 'ablation/run_ablation.py', log=out / 'synthetic.log')
    command(stage, 'real_data/run_real_data.py', log=out / 'observed_series.log')
    reports = checks(stage)
    result = compare_replay(stage, reports)
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    require(result['status'] == 'PASS', f'Replay exceeds declared tolerance; see {out / "verification.json"}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('verify', 'check', 'replay'))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if (a.mode == 'replay') != (a.output is not None):
        p.error('--output is required only for replay')
    try:
        integrity = verify()
        if a.mode == 'verify':
            result = integrity
        elif a.mode == 'check':
            with tempfile.TemporaryDirectory(prefix='kahm-check-') as temp:
                stage = Path(temp) / 'study'
                copy_stage(stage)
                result = {'status': 'PASS', 'integrity': integrity, 'checks': checks(stage)}
        else:
            result = replay(a.output)
        print(json.dumps(result, indent=2))
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        p.exit(1, f'Verification failed: {error}\n')


if __name__ == '__main__':
    main()
