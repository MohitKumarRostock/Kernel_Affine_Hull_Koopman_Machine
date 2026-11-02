#!/usr/bin/env python3
"""Manufactured-only checks of nesting, native dimensions and staged execution.

No production generator or production seed is used. The optional miniature
chain uses deliberately artificial random episodes, distinct test streams and
small fitting/evaluation roles. Its artifacts are retained for independent
audit; they are not scientific study results.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/kahm-softstate-mpl'

import argparse
import copy
from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

from artifacts import (array_hash, digest, load_portable_state, now, state_fingerprint,
                       write_json, write_portable_state)
from model import ARMS, BlockFeatures, SoftStateSuite, fit_suite, load_runtime, numerical_solver

ROOT = Path(__file__).resolve().parents[1]


def unit_checks(work):
    c = json.loads((ROOT / 'config.json').read_text())
    rng = np.random.default_rng(880001)
    ax, bx = rng.normal(size=(96, 12)), rng.normal(size=(25, 12))
    ay = np.column_stack((ax[:, 0], ax[:, 1] ** 2, ax[:, 2] + .2 * ax[:, 3]))
    by = np.column_stack((bx[:, 0], bx[:, 1] ** 2, bx[:, 2] + .2 * bx[:, 3]))
    mc = {**c['model'], 'blocks': [2], 'max_references': 24, 'bank_row_cap': 96}
    solver = numerical_solver()
    solver.clear_events()
    with threadpool_limits(limits=1):
        suite, detail, validation = fit_suite(ax, ay, bx, by, cap_order=[1, 2, 12, 20],
                                              seed=880002, config=mc)
        predictions, query = suite.predict(bx, tag='manufactured')
    checks = []

    # Exact aliases test ambient rank 12, not a smaller artificial input width.
    for arm in ARMS:
        np.testing.assert_array_equal(validation[f'cap12__{arm}__alpha_paths'],
                                      validation[f'cap20__{arm}__alpha_paths'])
        np.testing.assert_array_equal(validation[f'cap12__{arm}'], validation[f'cap20__{arm}'])
    ranks = [int(local['rank']) for local in suite.candidates['cap12']['bank']['models']]
    if 12 not in ranks:
        raise AssertionError('Cap-alias fixture failed to exercise native full ambient rank')
    checks.append({'name': 'native_cap12_cap20_alias_at_history12', 'status': 'passed',
                   'stored_ranks': ranks, 'all_B_alpha_predictions_bitwise_equal': True})

    # Check actual array prefixes, not merely feature-width accounting.
    for key, candidate in suite.candidates.items():
        geometry = BlockFeatures(candidate['bank'], candidate['scalers'])
        features, diag = geometry.transform(bx, tag='nesting_fixture')
        g, p = features['G'].shape[1], features['P'].shape[1]
        np.testing.assert_array_equal(features['G'], features['P'][:, :g])
        np.testing.assert_array_equal(features['P'], features['K'][:, :p])
        np.testing.assert_array_equal(features['P'], features['R'][:, :p])
        assert diag['shared_prefix_sha256']['P_gate_prefix'] == diag['method_feature_sha256']['G']
        assert diag['shared_prefix_sha256']['K_P_prefix'] == diag['method_feature_sha256']['P']
        assert diag['shared_prefix_sha256']['R_P_prefix'] == diag['method_feature_sha256']['P']
        assert diag['gate_block_sum_max_abs_error'] <= 1e-12
    checks.append({'name': 'actual_block_prefix_identity_and_partition_sums', 'status': 'passed'})

    # Independent linear-system solution checks the affine-ridge objective,
    # n-scaled penalty, free intercept and every alpha choice in one candidate.
    candidate = suite.candidates['cap2']
    geometry = BlockFeatures(candidate['bank'], candidate['scalers'])
    fa, _ = geometry.transform(ax)
    fb, _ = geometry.transform(bx)
    maximum_error = 0.
    for arm in ARMS:
        mean = fa[arm].mean(axis=0)
        xc, yc = fa[arm] - mean, ay - ay.mean(axis=0)
        for i, alpha in enumerate(mc['alpha_grid']):
            coefficient = np.linalg.solve(xc.T @ xc + len(ax) * alpha * np.eye(xc.shape[1]), xc.T @ yc)
            wanted = ay.mean(axis=0) + (fb[arm] - mean) @ coefficient
            actual = validation[f'cap2__{arm}__alpha_paths'][i]
            maximum_error = max(maximum_error, float(np.max(np.abs(wanted - actual))))
            np.testing.assert_allclose(wanted, actual, rtol=2e-9, atol=2e-9)
        paths = validation[f'cap2__{arm}__alpha_paths']
        losses = np.mean((paths - by[None, :, :]) ** 2, axis=1)
        expected_indices = np.argmin(losses, axis=0).tolist()
        assert expected_indices == detail['candidates']['cap2']['arms'][arm]['alpha_indices_by_horizon']
    checks.append({'name': 'independent_affine_ridge_normal_equation_and_B_paths',
                   'status': 'passed', 'maximum_absolute_prediction_difference': maximum_error})

    # A/B selection is audited from saved predictions, independently of the
    # selection function; adding K/R does not select the shared bank.
    means = {}
    for arm in ARMS:
        means[arm] = [float(np.mean(np.mean((validation[f'cap{cap}__{arm}'] - by) ** 2, axis=0)
                                    / suite.loss_scale)) for cap in suite.cap_order]
    p_cap = suite.cap_order[int(np.argmin(means['P']))]
    assert {r['cap'] for r in suite.selections['shared'].values()} == {p_cap}
    for arm in ARMS:
        assert suite.selections['independent'][arm]['cap'] == suite.cap_order[int(np.argmin(means[arm]))]
    checks.append({'name': 'shared_P_objective_and_independent_cap_choices', 'status': 'passed'})

    state_dir = work / 'portable_roundtrip'
    written = write_portable_state(state_dir, suite.to_state())
    restored = SoftStateSuite.from_state(load_portable_state(state_dir))
    replay, _ = restored.predict(bx)
    replay_max_error = 0.
    for key in predictions:
        replay_max_error = max(replay_max_error, float(np.max(np.abs(predictions[key] - replay[key]))))
        # Portable arrays preserve entries exactly; contiguous storage can
        # change a BLAS reduction order at floating-point roundoff scale.
        np.testing.assert_allclose(predictions[key], replay[key], rtol=2e-12, atol=2e-12)
    before = state_fingerprint(restored.to_state())
    restored.predict(bx + .01, tag='artificial_C')
    assert state_fingerprint(restored.to_state()) == before
    assert not any(name.lower().startswith(('c_', 'd_')) for name in inspect.signature(fit_suite).parameters)
    checks.append({'name': 'portable_state_exact_arrays_numerical_forecast_roundtrip_and_query_immutability',
                   'status': 'passed', 'portable_array_count': written['array_count'],
                   'maximum_absolute_forecast_difference': replay_max_error,
                   'forecast_rtol': 2e-12, 'forecast_atol': 2e-12})

    # Constant references exercise an empty local-position block, without
    # silently changing native rank zero into rank one.
    ax0, bx0 = np.ones((40, 12)), np.ones((11, 12))
    ay0, by0 = rng.normal(size=(40, 3)), rng.normal(size=(11, 3))
    with threadpool_limits(limits=1):
        zero, zd, zv = fit_suite(ax0, ay0, bx0, by0, cap_order=[1, 2], seed=880003, config=mc)
        zp, _ = zero.predict(bx0)
    for key, candidate in zero.candidates.items():
        assert all(local['rank'] == 0 for local in candidate['bank']['models'])
        assert zd['candidates'][key]['block_widths']['position'] == 0
        for arm in ARMS:
            np.testing.assert_allclose(zv[f'{key}__{arm}'], np.broadcast_to(ay0.mean(axis=0), (len(bx0), 3)),
                                       rtol=0., atol=1e-14)
            assert np.all(candidate['heads'][arm]['coef'] == 0.)
    assert all(np.isfinite(v).all() for v in zp.values())
    checks.append({'name': 'native_zero_rank_empty_position_block_and_affine_intercept', 'status': 'passed'})

    # Force the documented numerical exception only on an artificial ridge
    # design; the unmodified checked fallback must recover the same objective.
    m = load_runtime()
    solver.clear_events()
    primary = np.linalg.svd
    try:
        def forced_failure(*args, **kwargs):
            raise np.linalg.LinAlgError('manufactured preflight only')
        np.linalg.svd = forced_failure
        heads = m._ridge_path(ax[:, :5], ay, [.01])
    finally:
        np.linalg.svd = primary
    event = solver.events()
    assert len(event) == 1 and event[0]['status'] == 'recovered'
    xc, yc = ax[:, :5] - ax[:, :5].mean(axis=0), ay - ay.mean(axis=0)
    wanted = np.linalg.solve(xc.T @ xc + len(ax) * .01 * np.eye(5), xc.T @ yc)
    np.testing.assert_allclose(heads[0]['coef'], wanted, rtol=1e-10, atol=1e-10)
    checks.append({'name': 'documented_svd_recovery_on_manufactured_exception', 'status': 'passed',
                   'policy_id': event[0]['policy_id']})
    solver.clear_events()
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
             master_seed=8800103, repetitions=2, laws=['linear_ar', 'nonlinear_ar'],
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
    write_json(fixture / 'config.json', fixture_config())
    (fixture / 'PROTOCOL.md').write_text('Artificial integration fixture only; no production law outcomes.\n')
    write_json(fixture / 'readiness/prior_seed_exclusions.json', {'records': []})
    write_json(fixture / 'readiness/manufactured_preflight.json',
               {'status': 'passed', 'production_data_used': False, 'checks': unit_results,
                'scope': 'Unit prerequisites for this miniature integration chain; chain itself is under test'})
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
        return {'name': label or phase, 'status': 'passed', 'exit_code': result.returncode,
                'log_sha256': digest(path)}

    for phase in ('freeze', 'abc'):
        checks.append(run(phase))
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
    return fixture, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--unit-only', action='store_true')
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix='kahm-softstate-artificial-'))
    if args.work_dir:
        work.mkdir(parents=True, exist_ok=False)
    unit = unit_checks(work)
    fixture, chain = (None, []) if args.unit_only else staged_checks(work, unit)
    receipt = {'schema': 'kahm-soft-state-manufactured-preflight-v1', 'status': 'passed',
               'created_utc': now(), 'production_data_used': False,
               'production_generators_executed': False, 'work_directory': str(work),
               'fixture_root': None if fixture is None else str(fixture),
               'unit_checks': unit, 'staged_checks': chain,
               'independent_audit_in_this_receipt': False,
               'source_code_sha256': {p.name: digest(p) for p in sorted((ROOT / 'code').glob('*.py'))}}
    if args.output:
        write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
