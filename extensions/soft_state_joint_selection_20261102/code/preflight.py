#!/usr/bin/env python3
"""Artificial-only model, publication-gate, stage-order and independent replay checks.

The miniature cohort uses Gaussian episodes and reserved fixture streams. Its
results are software verification, never scientific study evidence. Production
source publication remains mandatory and is not simulated for production data.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/kahm-softstate-joint-mpl'
import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
from artifacts import digest, now, write_json
ROOT = Path(__file__).resolve().parents[1]


def publication_checks(work):
    checks = []
    # Pure receipt checks use obviously artificial commit identities, isolated
    # from every preparation command and from the production study directory.
    from pipeline import verify_publication
    gate_root = work / 'publication_receipt_unit_fixture'
    gate_root.mkdir()
    gate_config = fixture_config()
    gate_config.pop('manufactured_fixture')
    write_json(gate_root / 'config.json', gate_config)
    freeze_time = datetime.now(timezone.utc) - timedelta(seconds=2)
    write_json(gate_root / 'freeze.json', {'created_utc': freeze_time.isoformat(),
                                         'scope': 'artificial receipt-parser unit test only'})
    try:
        verify_publication(gate_root)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing publication receipt was accepted')
    publication = {'schema': 'kahm-soft-state-publication-v1', 'status': 'published',
                   'source_commit': 'a' * 40, 'source_tree_sha': 'b' * 40,
                   'repository': gate_config['source']['repository'],
                   'branch': 'artificial-receipt-unit-fixture',
                   'freeze_sha256': digest(gate_root / 'freeze.json'),
                   'published_utc': (freeze_time + timedelta(seconds=1)).isoformat(),
                   'source_files_verified': True}
    # Each variant gets a distinct directory; no frozen receipt is overwritten.
    for label, change in (('wrong_freeze', {'freeze_sha256': '0' * 64}),
                          ('bad_commit', {'source_commit': 'not-a-commit'}),
                          ('unverified', {'source_files_verified': False}),
                          ('future', {'published_utc': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()})):
        rejected = work / ('publication_reject_' + label)
        rejected.mkdir()
        shutil.copyfile(gate_root / 'config.json', rejected / 'config.json')
        shutil.copyfile(gate_root / 'freeze.json', rejected / 'freeze.json')
        write_json(rejected / 'PUBLICATION.json', {**publication, **change})
        try:
            verify_publication(rejected)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid publication receipt accepted: ' + label)
    write_json(gate_root / 'PUBLICATION.json', publication)
    assert verify_publication(gate_root)['source_commit'] == 'a' * 40
    checks.append({'name': 'publication_receipt_gate_artificial_metadata_only', 'status': 'passed',
                   'production_preparation_called': False, 'remote_publication_claimed': False})
    return checks


def fixture_config():
    c = copy.deepcopy(json.loads((ROOT / 'config.json').read_text()))
    c.update(manufactured_fixture={'purpose': 'staging and independent audit, no scientific outcomes',
                                  'base_generator': 'artificial Gaussian episodes plus fixed lag perturbation'},
             master_seed=8800204, repetitions=2, laws=['linear_ar', 'nonlinear_ar'],
             nonlinear_laws=['nonlinear_ar'], native_cap_order=[1, 2],
             max_episode_length=32, history_tail_length=20, origins_per_episode=3,
             split_episodes={'A': 8, 'B': 4, 'C': 4, 'D': 6},
             split_caps={'A': 24, 'B': 12, 'C': 12, 'D': None}, task_count=8)
    c['model'].update(blocks=[2], max_references=8, bank_row_cap=24)
    c['seed_derivation']['distinct_seed_count'] = 54
    c['fresh_evidence']['new_independent_repetitions'] = 0
    return c


def staged_checks(work, unit_results):
    fixture = work / 'staged_fixture'
    fixture.mkdir()
    shutil.copytree(ROOT / 'code', fixture / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'audit', fixture / 'audit', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'analysis_code', fixture / 'analysis_code', ignore=shutil.ignore_patterns('__pycache__'))
    write_json(fixture / 'config.json', fixture_config())
    (fixture / 'PROTOCOL.md').write_text('Artificial integration fixture only; no production law outcomes.\n')
    write_json(fixture / 'readiness/prior_seed_exclusions.json', {'records': []})
    write_json(fixture / 'readiness/manufactured_preflight.json',
               {'status': 'passed', 'production_data_used': False, 'checks': unit_results,
                'scope': 'Completed unit prerequisites; the miniature stage chain is under test'})
    command = [sys.executable, str(ROOT / 'code/pipeline.py')]
    logs = work / 'chain_logs'
    logs.mkdir()
    checks = []

    def run(phase, *, succeeds=True, label=None):
        path = logs / ((label or phase) + '.log')
        with path.open('x') as stream:
            result = subprocess.run([*command, phase, '--root', str(fixture), '--workers', '1'],
                                    stdout=stream, stderr=subprocess.STDOUT)
        if (result.returncode == 0) != succeeds:
            raise RuntimeError(f'Artificial stage {phase} unexpected status {result.returncode}: {path.read_text()[-6000:]}')
        print(json.dumps({'fixture_phase': label or phase, 'status': 'passed'}), flush=True)
        return {'name': label or phase, 'status': 'passed', 'exit_code': result.returncode,
                'log_sha256': digest(path)}

    checks.append(run('freeze'))
    # Check source integrity without modifying any production source or running data generation.
    from pipeline import verify_freeze
    copied_model = fixture / 'code/model.py'
    pristine = copied_model.read_bytes()
    try:
        copied_model.write_bytes(pristine + b'\n# artificial integrity mutation\n')
        try:
            verify_freeze(fixture)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed frozen artificial source accepted')
    finally:
        copied_model.write_bytes(pristine)
    verify_freeze(fixture)
    checks.append({'name': 'reject_changed_frozen_source', 'status': 'passed'})
    checks.append(run('abc'))
    checks.append(run('d', succeeds=False, label='reject_D_before_fit_seal'))
    assert not (fixture / 'prepared/targets').exists()
    checks.append(run('fit'))
    checks.append(run('d', succeeds=False, label='reject_D_after_fits_without_global_seal'))
    assert not (fixture / 'prepared/targets').exists()
    for phase in ('seal-fits', 'd'):
        checks.append(run(phase))
    checks.append(run('release', succeeds=False, label='reject_scoring_before_prediction_seal'))
    assert not (fixture / 'TARGET_RELEASE.json').exists()
    for phase in ('predict', 'seal-predictions', 'release', 'verify'):
        checks.append(run(phase))
    # Reconstruct every miniature task independently from retained A states and
    # episode/noise arrays. The oracle imports no production model or analyzer.
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(ROOT / 'audit'))
    from verify_soft_state import audit_task
    from independent_oracle import Auditor
    audit = Auditor()
    roster = json.loads((fixture / 'prepared/abc_manifest.json').read_text())['tasks']
    with threadpool_limits(limits=1):
        for task in roster:
            audit_task(fixture, task['task_id'], audit)
    checks.append({'name': 'independent_all_artificial_tasks_replay', 'status': 'passed',
                   'task_count': len(roster), 'checks': audit.checks,
                   'maximum_absolute_numerical_differences': dict(audit.max_absolute_difference),
                   'scope': 'All candidate B paths, joint and standard-only selections, all selected C/D arrays; artificial data only'})
    return fixture, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--unit-only', action='store_true')
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix='kahm-softstate-joint-artificial-'))
    if args.work_dir:
        work.mkdir(parents=True, exist_ok=False)
    from preflight_model import run_model_checks
    unit = run_model_checks(work)
    unit.extend(publication_checks(work))
    statistics_path = ROOT / 'analysis_code/preflight_statistics.py'
    statistics_log = work / 'statistics_preflight.log'
    with statistics_log.open('x') as stream:
        result = subprocess.run([sys.executable, str(statistics_path)], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Manufactured statistics verification failed: ' + statistics_log.read_text()[-6000:])
    unit.append({'name': 'independent_manufactured_six_contrast_statistics', 'status': 'passed',
                 'log_sha256': digest(statistics_log)})
    fixture, chain = (None, []) if args.unit_only else staged_checks(work, unit)
    receipt = {'schema': 'kahm-soft-state-joint-manufactured-preflight-v1', 'status': 'passed',
               'created_utc': now(), 'production_data_used': False,
               'production_generators_executed': False, 'work_directory': str(work),
               'fixture_root': None if fixture is None else str(fixture),
               'unit_checks': unit, 'staged_checks': chain,
               'independent_audit_in_this_receipt': fixture is not None,
               'source_code_sha256': {p.name: digest(p) for p in sorted((ROOT / 'code').glob('*.py'))},
               'independent_audit_source_sha256': {p.name: digest(p) for p in sorted((ROOT / 'audit').glob('*.py'))}}
    if args.output:
        write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
