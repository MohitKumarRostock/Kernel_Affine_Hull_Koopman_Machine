#!/usr/bin/env python3
"""Manufactured-only verification of joint input-transform/native-cap selection.

The fixtures use artificial random arrays and never call a scientific data
generator. They compare the standard-only path with the preceding published
model and independently recompute fitting and validation objectives.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'

import argparse
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

from artifacts import load_portable_state, now, state_fingerprint, write_json, write_portable_state
from model import (ARMS, SCALINGS, VIEWS, BlockFeatures, SoftStateSuite, candidate_id,
                   fit_candidate, fit_suite, load_runtime, numerical_solver)

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT.parent / 'soft_state_ablation_20261102'


def predecessor_module():
    spec = importlib.util.spec_from_file_location(
        '_joint_preflight_predecessor_model', PREDECESSOR / 'code/model.py')
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the preserved predecessor model')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_model_checks(work_dir):
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'config.json'
    if not config_path.is_file():
        config_path = PREDECESSOR / 'config.json'
    declared = json.loads(config_path.read_text())
    # Subsampling deliberately tests that geometry scalers fit ALL A histories.
    mc = {**declared['model'], 'blocks': [2], 'max_references': 24, 'bank_row_cap': 64}
    caps = [1, 2, 4, 8, 12]
    rng = np.random.default_rng(880041)
    ax, bx = rng.normal(size=(96, 12)), rng.normal(size=(25, 12))
    ax[0, 0] = 9.0
    ay = np.column_stack((ax[:, 0], ax[:, 1] ** 2, ax[:, 2] + .2 * ax[:, 3]))
    by = np.column_stack((bx[:, 0], bx[:, 1] ** 2, bx[:, 2] + .2 * bx[:, 3]))
    checks = []
    solver = numerical_solver()
    solver.clear_events()

    with threadpool_limits(limits=1):
        suite, detail, validation = fit_suite(
            ax, ay, bx, by, cap_order=caps, scaling_order=list(SCALINGS), seed=880042, config=mc)
        before = state_fingerprint(suite.to_state())
        predictions, query = suite.predict(bx, tag='manufactured')
        if state_fingerprint(suite.to_state()) != before:
            raise AssertionError('Prediction mutated a fitted state')
        previous = predecessor_module()
        # Both adapters use exactly the same immutable solver source. Its
        # installation guard permits only one module instance per process.
        previous._SOLVER = solver
        old_suite, old_detail, old_validation = previous.fit_suite(
            ax, ay, bx, by, cap_order=caps, seed=880042, config=mc)
        old_predictions, _ = old_suite.predict(bx, tag='manufactured')
    for cap in caps:
        for arm in ARMS:
            for suffix in ('', '__alpha_paths'):
                np.testing.assert_array_equal(
                    validation[f'standard_cap{cap}__{arm}{suffix}'],
                    old_validation[f'cap{cap}__{arm}{suffix}'])
    for old_view, new_view in (('shared', 'standard_shared'), ('independent', 'standard_independent')):
        for arm in ARMS:
            np.testing.assert_array_equal(predictions[f'{new_view}__{arm}'], old_predictions[f'{old_view}__{arm}'])
            new, old = suite.selections[new_view][arm], old_suite.selections[old_view][arm]
            for field in ('cap', 'alpha_indices_by_horizon', 'alpha_by_horizon', 'B_mean_smse'):
                if new[field] != old[field]:
                    raise AssertionError('Standard-only selection differs from predecessor: ' + field)
    checks.append({'name': 'standard_only_exact_predecessor_predictions_and_selections',
                   'status': 'passed', 'candidate_count': len(caps), 'views': 2,
                   'all_candidate_alpha_paths_bitwise_equal': True,
                   'all_selected_forecasts_bitwise_equal': True})

    m = load_runtime()
    scale = np.where(ax.std(axis=0, ddof=0) > 1e-12, ax.std(axis=0, ddof=0), 1.0)
    for key, candidate in suite.candidates.items():
        bank = candidate['bank']
        mode = candidate['input_scaling']
        np.testing.assert_array_equal(bank['mean'], ax.mean(axis=0))
        np.testing.assert_array_equal(bank['scale'], scale)
        if mode == 'standard':
            if bank.get('geometry_scaler') is not None:
                raise AssertionError('Standard path acquired another input transformation')
        else:
            scaler = bank['geometry_scaler']
            if scaler['fit_rows'] != len(ax) or scaler['mode'] != mode:
                raise AssertionError('Geometry scaler did not fit all A rows')
            if mode == 'minmax':
                np.testing.assert_array_equal(scaler['minimum'], ax.min(axis=0))
                np.testing.assert_array_equal(scaler['maximum'], ax.max(axis=0))
                expected = 2 * (bx - ax.min(axis=0)) / np.ptp(ax, axis=0) - 1
            else:
                np.testing.assert_array_equal(scaler['mean'], ax.mean(axis=0))
                np.testing.assert_array_equal(scaler['scale'], scale)
                if scaler['post_tanh_rescale'] is not False:
                    raise AssertionError('Tanh input map was restandardized')
                expected = np.tanh((bx - ax.mean(axis=0)) / scale)
            np.testing.assert_array_equal(m.transform_geometry(bx, scaler), expected)
        geometry = BlockFeatures(bank, candidate['scalers'])
        features, diagnostic = geometry.transform(bx, tag='nesting_fixture')
        g, p = features['G'].shape[1], features['P'].shape[1]
        np.testing.assert_array_equal(features['G'], features['P'][:, :g])
        np.testing.assert_array_equal(features['P'], features['K'][:, :p])
        np.testing.assert_array_equal(features['P'], features['R'][:, :p])
        if features['K'].shape != features['R'].shape or diagnostic['gate_block_sum_max_abs_error'] > 1e-12:
            raise AssertionError('Matched representation structure changed')
    checks.append({'name': 'A_only_full_row_input_scalers_and_exact_shared_blocks_all_candidates',
                   'status': 'passed', 'candidate_count': len(suite.candidates),
                   'A_rows': len(ax), 'bank_row_cap': mc['bank_row_cap']})

    # B has an explicit role only in penalty/candidate choice, never geometry.
    with threadpool_limits(limits=1):
        for mode in SCALINGS:
            alternate, _, _ = fit_candidate(ax, ay, bx * 30 + 7, by * -9,
                cap=2, input_scaling=mode, seed=880042, config=mc)
            original = suite.candidates[candidate_id(mode, 2)]
            for field in ('bank', 'scalers', 'loss_scale'):
                if state_fingerprint(alternate[field]) != state_fingerprint(original[field]):
                    raise AssertionError('B values affected A-fitted geometry/block statistics: ' + field)
    checks.append({'name': 'changing_B_histories_and_targets_does_not_change_A_geometry_or_block_scalers',
                   'status': 'passed', 'transformations_checked': list(SCALINGS)})

    # Recompute every alpha choice and every pair choice from retained forecasts.
    objectives = {}
    for key in suite.candidate_order:
        objectives[key] = {}
        for arm in ARMS:
            paths = validation[f'{key}__{arm}__alpha_paths']
            losses = np.mean((paths - by[None, :, :]) ** 2, axis=1)
            indices = np.argmin(losses, axis=0)
            record = detail['candidates'][key]['arms'][arm]
            if indices.tolist() != record['alpha_indices_by_horizon']:
                raise AssertionError('B alpha choice disagrees with independently recomputed losses')
            selected = np.column_stack([paths[index, :, h] for h, index in enumerate(indices)])
            np.testing.assert_allclose(validation[f'{key}__{arm}'], selected, rtol=1e-13, atol=1e-13)
            objectives[key][arm] = float(np.mean(np.mean((selected - by) ** 2, axis=0) / suite.loss_scale))
            np.testing.assert_allclose(record['B_mean_smse'], objectives[key][arm], rtol=1e-13, atol=1e-13)
    for view in VIEWS:
        keys = ([candidate_id('standard', cap) for cap in caps]
                if view.startswith('standard_') else suite.candidate_order)
        for arm in ARMS:
            selector = 'P' if view in ('shared', 'standard_shared') else arm
            expected = keys[int(np.argmin([objectives[key][selector] for key in keys]))]
            if suite.selections[view][arm]['candidate_id'] != expected:
                raise AssertionError('Independent B geometry choice differs: ' + view + '/' + arm)
            np.testing.assert_array_equal(validation[f'{view}__{arm}'], validation[f'{expected}__{arm}'])
    for arm in ARMS:
        if suite.selections['independent'][arm]['B_mean_smse'] > suite.selections['standard_independent'][arm]['B_mean_smse']:
            raise AssertionError('A superset search increased the selected B objective')
    checks.append({'name': 'independent_recomputation_of_every_B_alpha_pair_and_standard_subset_choice',
                   'status': 'passed', 'candidate_count': len(suite.candidates),
                   'view_count': len(VIEWS), 'superset_property_applies_to_B_only': True})

    # A separate linear-system solution checks the affine-ridge objective.
    maximum_error = 0.0
    with threadpool_limits(limits=1):
        for mode in SCALINGS:
            key = candidate_id(mode, 2)
            candidate = suite.candidates[key]
            geometry = BlockFeatures(candidate['bank'], candidate['scalers'])
            fa, _ = geometry.transform(ax)
            fb, _ = geometry.transform(bx)
            for arm in ARMS:
                mean = fa[arm].mean(axis=0)
                xc, yc = fa[arm] - mean, ay - ay.mean(axis=0)
                for i, alpha in enumerate(mc['alpha_grid']):
                    coefficient = np.linalg.solve(xc.T @ xc + len(ax) * alpha * np.eye(xc.shape[1]), xc.T @ yc)
                    wanted = ay.mean(axis=0) + (fb[arm] - mean) @ coefficient
                    actual = validation[f'{key}__{arm}__alpha_paths'][i]
                    maximum_error = max(maximum_error, float(np.max(np.abs(wanted - actual))))
                    np.testing.assert_allclose(wanted, actual, rtol=2e-9, atol=2e-9)
    checks.append({'name': 'independent_affine_ridge_normal_equations_for_all_input_transformations',
                   'status': 'passed', 'maximum_absolute_prediction_difference': maximum_error,
                   'unpenalized_intercept': True})

    # Cap 20 is an ambient-rank alias; it must not become a production candidate.
    with threadpool_limits(limits=1):
        for mode in SCALINGS:
            alias, _, alias_arrays = fit_candidate(
                ax, ay, bx, by, cap=20, input_scaling=mode, seed=880042, config=mc)
            base = suite.candidates[candidate_id(mode, 12)]
            ranks = [local['rank'] for local in base['bank']['models']]
            if 12 not in ranks:
                raise AssertionError('Alias fixture did not exercise full ambient native rank')
            for arm in ARMS:
                np.testing.assert_array_equal(alias_arrays[f'{mode}_cap20__{arm}__alpha_paths'],
                                              validation[f'{mode}_cap12__{arm}__alpha_paths'])
    checks.append({'name': 'cap12_cap20_exact_alias_all_transformations_at_history12',
                   'status': 'passed', 'full_ambient_rank_exercised': True})

    written = write_portable_state(work / 'portable_roundtrip', suite.to_state())
    restored = SoftStateSuite.from_state(load_portable_state(work / 'portable_roundtrip'))
    with threadpool_limits(limits=1):
        replay, _ = restored.predict(bx)
        fingerprint = state_fingerprint(restored.to_state())
        restored.predict(bx + .01, tag='artificial_C')
    if state_fingerprint(restored.to_state()) != fingerprint:
        raise AssertionError('Query transformation or prediction changed fitted state')
    replay_error = max(float(np.max(np.abs(predictions[key] - replay[key]))) for key in predictions)
    for key in predictions:
        np.testing.assert_allclose(predictions[key], replay[key], rtol=2e-12, atol=2e-12)
    if any(name.lower().startswith(('c_', 'd_')) for name in inspect.signature(fit_suite).parameters):
        raise AssertionError('Fitting API accepts C or D inputs')
    checks.append({'name': 'portable_state_order_roundtrip_query_immutability_and_AB_only_fitting_API',
                   'status': 'passed', 'portable_array_count': written['array_count'],
                   'forecast_array_count': len(replay), 'maximum_absolute_forecast_difference': replay_error})

    # Deliberate order reversal makes tie resolution observable.
    with threadpool_limits(limits=1):
        ties, _, _ = fit_suite(ax, np.zeros_like(ay), bx, np.zeros_like(by),
            cap_order=[2, 1], scaling_order=['tanh_z_1', 'standard', 'minmax'], seed=880043, config=mc)
    for view in VIEWS:
        expected = 'standard_cap2' if view.startswith('standard_') else 'tanh_z_1_cap2'
        for arm in ARMS:
            choice = ties.selections[view][arm]
            if choice['candidate_id'] != expected or choice['alpha_indices_by_horizon'] != [0, 0, 0]:
                raise AssertionError('Exact ties did not follow the declared candidate/alpha order')
    checks.append({'name': 'exact_ties_follow_scaler_outer_cap_inner_then_first_alpha', 'status': 'passed'})

    ax0, bx0 = np.ones((40, 12)), np.full((11, 12), 2.0)
    ay0, by0 = rng.normal(size=(40, 3)), rng.normal(size=(11, 3))
    with threadpool_limits(limits=1):
        zero, zd, zv = fit_suite(ax0, ay0, bx0, by0, cap_order=[1, 2],
            scaling_order=list(SCALINGS), seed=880044, config={**mc, 'blocks': [1]})
        zero_predictions, _ = zero.predict(bx0)
    for key, candidate in zero.candidates.items():
        if any(local['rank'] != 0 for local in candidate['bank']['models']) or zd['candidates'][key]['block_widths']['position'] != 0:
            raise AssertionError('Constant-reference native rank zero was not retained')
        for arm in ARMS:
            np.testing.assert_allclose(zv[f'{key}__{arm}'], np.broadcast_to(ay0.mean(axis=0), (len(bx0), 3)),
                                       rtol=0., atol=1e-14)
            if np.any(candidate['heads'][arm]['coef'] != 0):
                raise AssertionError('Constant features acquired a nonzero coefficient')
    if any(not np.isfinite(value).all() for value in zero_predictions.values()):
        raise AssertionError('Nonfinite zero-rank forecast')
    constant_minmax = zero.candidates['minmax_cap1']['bank']['geometry_scaler']
    np.testing.assert_array_equal(m.transform_geometry(bx0, constant_minmax), np.zeros_like(bx0))
    minmax = suite.candidates['minmax_cap1']['bank']['geometry_scaler']
    outside = ax.max(axis=0)[None, :] + 2 * np.ptp(ax, axis=0)[None, :]
    np.testing.assert_allclose(m.transform_geometry(outside, minmax), np.full((1, 12), 5.), rtol=1e-14, atol=1e-14)
    tanh = suite.candidates['tanh_z_1_cap1']['bank']['geometry_scaler']
    extreme = np.vstack((tanh['mean'] + 1e6 * tanh['scale'], tanh['mean'] - 1e6 * tanh['scale']))
    np.testing.assert_array_equal(m.transform_geometry(extreme, tanh), np.vstack((np.ones(12), -np.ones(12))))
    checks.append({'name': 'constant_coordinates_native_zero_rank_unclipped_minmax_and_tanh_saturation',
                   'status': 'passed', 'minmax_out_of_range_transformed_value': 5.0,
                   'minmax_constant_A_coordinates_map_to_zero_for_all_queries': True})

    # The serializer stores order explicitly; corrupted order and mismatched
    # candidate identity must fail rather than silently choosing another model.
    mutations = []
    bad_order = copy.deepcopy(suite.to_state())
    bad_order['candidate_order'] = list(reversed(bad_order['candidate_order']))
    mutations.append(bad_order)
    bad_selection = copy.deepcopy(suite.to_state())
    bad_selection['selections']['standard_independent']['K']['input_scaling'] = 'minmax'
    mutations.append(bad_selection)
    for altered in mutations:
        try:
            SoftStateSuite.from_state(altered)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed portable candidate/selection metadata was accepted')
    checks.append({'name': 'changed_serialized_candidate_order_and_selection_identity_rejected', 'status': 'passed'})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix='kahm-joint-model-preflight-'))
    checks = run_model_checks(work)
    result = {'schema': 'kahm-soft-state-joint-model-preflight-v1', 'status': 'passed',
              'created_utc': now(), 'production_data_used': False, 'checks': checks,
              'work_directory': str(work.resolve())}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
