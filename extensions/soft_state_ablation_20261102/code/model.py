"""Four representation-only affine ridge arms on shared native geometry.

Only A fits geometry, block transforms and coefficients. B selects penalties
and caps. No model API accepts C or D targets. Shared blocks are transformed
once and copied into each containing arm, preserving the declared nesting.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

from artifacts import array_hash, state_fingerprint

ROOT = Path(__file__).resolve().parents[1]
INHERITED = ROOT.parent / 'dimension_forecasting'
NUMERICAL = ROOT.parent / 'policy_augmentation_20261101' / 'code'
sys.path.insert(0, str(INHERITED / 'code'))
from focused_dimension_model import load_runtime, runtime_provenance

ARMS = ('G', 'P', 'K', 'R')
VIEWS = ('shared', 'independent')
BLOCKS = ('gates', 'position', 'kahm', 'rbf')
_SOLVER = None


def numerical_solver():
    global _SOLVER
    if _SOLVER is None:
        path = NUMERICAL / 'numerical_solver.py'
        spec = importlib.util.spec_from_file_location('_softstate_inherited_numerical_solver', path)
        if spec is None or spec.loader is None:
            raise ImportError('Cannot locate the inherited checked numerical solver')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SOLVER = module
    return _SOLVER


def expected_keys():
    return {f'{view}__{arm}' for view in VIEWS for arm in ARMS}


def effective_config(cap, config):
    m = load_runtime()
    supplied = dict(config)
    if any(k in supplied for k in ('rank', 'rank_cap', 'dimension_policy')):
        raise ValueError('Pass the native cap separately')
    supplied.update(rank=int(cap), dimension_policy='native')
    result = m.dimension_config(m.DEFAULT_CONFIG, supplied)
    alphas = result['alpha_grid']
    if not alphas or any(not math.isfinite(float(a)) or a <= 0 for a in alphas):
        raise ValueError('Ridge penalties must be positive and finite')
    if len(set(alphas)) != len(alphas):
        raise ValueError('Duplicate ridge penalties are not distinct candidates')
    return result


def _empty_summary(rows):
    return {'rows': int(rows), 'width': 0, 'finite': True, 'minimum': None,
            'maximum': None, 'max_abs': 0., 'max_norm': 0., 'radius': 0., 'violations': 0}


def split_raw_blocks(bank, raw):
    gate_width = len(bank['models'])
    position_width = sum(int(local['rank']) for local in bank['models'])
    common = raw['common']
    if common.shape[1] != gate_width + position_width:
        raise AssertionError('Inherited common feature ordering/width changed')
    # The inherited constructor places ALL gates first, then the gated local
    # principal coordinates in local-model order. No new gate calculation.
    result = {'gates': common[:, :gate_width], 'position': common[:, gate_width:],
              'kahm': raw['kahm'], 'rbf': raw['rbf']}
    if result['kahm'].shape != result['rbf'].shape:
        raise AssertionError('Matched departure widths differ')
    for start, stop in bank['block_slices']:
        if not np.allclose(result['gates'][:, start:stop].sum(axis=1), 1., rtol=0., atol=1e-12):
            raise AssertionError('Partition gates no longer sum to one')
    if np.any(result['gates'] < 0.):
        raise AssertionError('Negative locality membership')
    return result


@dataclass
class BlockFeatures:
    bank: dict
    scalers: dict

    def transform_raw(self, raw, *, nodes=None, tag='query'):
        m = load_runtime()
        nodes = {} if nodes is None else nodes
        raw_blocks = split_raw_blocks(self.bank, raw)
        blocks, summaries = {}, {}
        for block in BLOCKS:
            values = raw_blocks[block]
            if values.shape[1] == 0:
                blocks[block] = np.zeros((len(values), 0), dtype=np.float64)
                summaries[block] = _empty_summary(len(values))
                continue
            with m.timed(nodes, f'{block}_transform_{tag}'):
                scaled = m._standardize(self.scalers[block], values)
                blocks[block] = m.radial_bound(scaled)
            summaries[block] = m._feature_summary(blocks[block], math.sqrt(values.shape[1]))
            if not summaries[block]['finite'] or summaries[block]['violations']:
                raise FloatingPointError('Invalid bounded feature block: ' + block)
        features = {'G': blocks['gates']}
        features['P'] = np.column_stack((blocks['gates'], blocks['position']))
        features['K'] = np.column_stack((features['P'], blocks['kahm']))
        features['R'] = np.column_stack((features['P'], blocks['rbf']))
        g, p = features['G'].shape[1], features['P'].shape[1]
        checks = {'G_in_P_exact': np.array_equal(features['G'], features['P'][:, :g]),
                  'P_in_K_exact': np.array_equal(features['P'], features['K'][:, :p]),
                  'P_in_R_exact': np.array_equal(features['P'], features['R'][:, :p]),
                  'K_R_equal_width': features['K'].shape[1] == features['R'].shape[1]}
        if not all(checks.values()):
            raise AssertionError('Shared transformed blocks differ across arms')
        diagnostic = {'block_features': summaries,
                      'raw_block_sha256': {k: array_hash(v) for k, v in raw_blocks.items()},
                      'transformed_block_sha256': {k: array_hash(v) for k, v in blocks.items()},
                      'method_feature_sha256': {k: array_hash(v) for k, v in features.items()},
                      'shared_prefix_sha256': {
                          'P_gate_prefix': array_hash(features['P'][:, :g]),
                          'K_P_prefix': array_hash(features['K'][:, :p]),
                          'R_P_prefix': array_hash(features['R'][:, :p])},
                      'method_widths': {k: int(v.shape[1]) for k, v in features.items()},
                      'nesting_checks': checks,
                      'transformation': 'A column-standardization, then one radial map per fixed block',
                      'whole_descriptor_radial_map': False}
        return features, diagnostic

    def transform(self, x, *, nodes=None, tag='query'):
        m = load_runtime()
        x = m._matrix(x, 'query histories')
        if x.shape[1] != len(self.bank['mean']):
            raise ValueError('Query history width differs from fitted state')
        raw, diagnostics = m._raw_feature_maps(self.bank, x, nodes, tag)
        features, transformed = self.transform_raw(raw, nodes=nodes, tag=tag)
        diagnostics.update(transformed)
        return features, diagnostics


def local_dimension_records(bank):
    records = []
    for i, local in enumerate(bank['models']):
        record = copy.deepcopy(local['dimension_audit'])
        record.update(model_index=i, block_index=int(local['block']),
                      stored_basis_shape=list(local['p'].shape),
                      stored_reference_shape=list(local['q'].shape),
                      basis_sha256=array_hash(local['p']), reference_sha256=array_hash(local['q']))
        if record['effective_rank'] != local['p'].shape[1] or local['rank'] != local['p'].shape[1]:
            raise AssertionError('Native dimension audit and stored basis differ')
        records.append(record)
    return records


def fit_candidate(ax, ay, bx, by, *, cap, seed, config):
    m = load_runtime()
    effective = effective_config(cap, config)
    nodes = {}
    bank_fit = m._InstrumentedBank(seed=int(seed), config=effective,
                                   geometry_scaling='standard').fit(ax)
    if any(v.get('factor_error') for v in bank_fit.diagnostics['local_models']):
        raise FloatingPointError('Unexpected native geometry factor failure')
    nodes.update(bank_fit.nodes)
    bank = bank_fit.state('kahm')
    for local in bank['models']:
        local.pop('reference_distance', None)
    raw, initial = {}, {}
    for tag, x in (('A', ax), ('B', bx)):
        raw[tag], initial[tag] = m._raw_feature_maps(bank, x, nodes, tag)
    a_blocks = split_raw_blocks(bank, raw['A'])
    scalers = {}
    for block in BLOCKS:
        with m.timed(nodes, f'{block}_scaler_A'):
            if a_blocks[block].shape[1]:
                scalers[block] = m._scaler_fit(a_blocks[block], effective['feature_scale_floor'])
            else:
                scalers[block] = {'mean': np.zeros(0), 'scale': np.zeros(0),
                                  'floor': float(effective['feature_scale_floor']), 'floored_columns': 0}
    geometry = BlockFeatures(bank, scalers)
    features = {}
    for tag in ('A', 'B'):
        features[tag], extra = geometry.transform_raw(raw[tag], nodes=nodes, tag=tag)
        initial[tag].update(extra)
    loss_scale = np.maximum(np.var(ay, axis=0, ddof=0), effective['target_variance_floor'])
    heads, selected_b, validation, records = {}, {}, {}, {}
    key = f'cap{cap}'
    for arm in ARMS:
        with m.timed(nodes, arm + '_ridge_path_A'):
            path = m._ridge_path(features['A'][arm], ay, effective['alpha_grid'])
        with m.timed(nodes, arm + '_selection_B'):
            predictions = np.stack([m._ridge_predict(head, features['B'][arm]) for head in path])
            losses = np.stack([m._mse(by, prediction) for prediction in predictions])
            indices = np.argmin(losses, axis=0)
            heads[arm] = m._combine_heads(path, indices)
            selected_b[arm] = m._ridge_predict(heads[arm], features['B'][arm])
        validation[f'{key}__{arm}__alpha_paths'] = predictions
        validation[f'{key}__{arm}'] = selected_b[arm]
        actual = np.column_stack([predictions[int(index), :, h] for h, index in enumerate(indices)])
        np.testing.assert_allclose(selected_b[arm], actual, rtol=1e-13, atol=1e-13)
        records[arm] = {
            'cap': int(cap), 'arm': arm,
            'alpha_candidates': [{'alpha': float(alpha), 'alpha_index': i,
                                  'B_mse_by_horizon': losses[i].tolist(),
                                  'B_smse_by_horizon': (losses[i] / loss_scale).tolist(),
                                  'status': 'valid'}
                                 for i, alpha in enumerate(effective['alpha_grid'])],
            'alpha_indices_by_horizon': indices.tolist(),
            'alpha_by_horizon': heads[arm]['alpha_by_horizon'].tolist(),
            'B_mse_by_horizon': m._mse(by, selected_b[arm]).tolist(),
            'B_smse_by_horizon': (m._mse(by, selected_b[arm]) / loss_scale).tolist(),
            'B_mean_smse': float(np.mean(m._mse(by, selected_b[arm]) / loss_scale)),
            'feature_width': int(heads[arm]['feature_width']),
            'centered_feature_rank': int(heads[arm]['feature_rank']),
            'coefficient_count': int(heads[arm]['coef'].size),
            'B_prediction_sha256': array_hash(selected_b[arm]),
            'fit_split': 'A', 'selection_split': 'B', 'intercept': 'unpenalized',
            'baseline_retained': False, 'correction_multiplier': None}
    state = {'cap': int(cap), 'bank': bank, 'scalers': scalers, 'heads': heads,
             'loss_scale': loss_scale, 'config': effective}
    diagnostic = {'candidate_id': key, 'cap': int(cap), 'seed': int(seed),
                  'bank': bank_fit.diagnostics, 'local_dimensions': local_dimension_records(bank),
                  'arms': records, 'features_A_B': initial, 'fit_nodes': nodes,
                  'block_widths': {k: int(v.shape[1]) for k, v in a_blocks.items()},
                  'feature_widths': {k: int(v.shape[1]) for k, v in features['A'].items()},
                  'input_scaling': 'A-fitted standardization only',
                  'C_used': False, 'D_used': False}
    return state, diagnostic, validation


@dataclass
class SoftStateSuite:
    cap_order: list
    candidates: dict
    selections: dict
    input_hashes: dict
    loss_scale: np.ndarray

    def to_state(self):
        return {'schema': 'kahm-soft-state-model-v1', 'cap_order': self.cap_order,
                'candidates': self.candidates, 'selections': self.selections,
                'input_hashes': self.input_hashes, 'loss_scale': self.loss_scale,
                'feature_block_order': list(BLOCKS), 'arms': list(ARMS),
                'geometry_scaling': 'standard', 'retained_baseline': False,
                'C_role': 'diagnostic only', 'direct_horizon_heads': True,
                'autonomous_dynamics_fitted': False}

    @classmethod
    def from_state(cls, state):
        if state['schema'] != 'kahm-soft-state-model-v1' or tuple(state['arms']) != ARMS:
            raise ValueError('Unexpected soft-state model schema')
        return cls(state['cap_order'], state['candidates'], state['selections'],
                   state['input_hashes'], state['loss_scale'])

    def predict(self, x, *, tag='query'):
        m = load_runtime()
        x = m._matrix(x, 'fixed query histories')
        predictions, diagnostics, nodes = {}, {}, {}
        selected_ids = {record['candidate_id'] for view in self.selections.values()
                        for record in view.values()}
        for key in (f'cap{cap}' for cap in self.cap_order):
            if key not in selected_ids:
                continue
            state = self.candidates[key]
            geometry = BlockFeatures(state['bank'], state['scalers'])
            features, diagnostics[key] = geometry.transform(x, nodes=nodes, tag=tag)
            for view in VIEWS:
                for arm in ARMS:
                    if self.selections[view][arm]['candidate_id'] != key:
                        continue
                    with m.timed(nodes, f'{view}_{arm}_predict_{tag}'):
                        predictions[f'{view}__{arm}'] = m._ridge_predict(state['heads'][arm], features[arm])
        if set(predictions) != expected_keys():
            raise AssertionError('Incomplete fixed forecast roster')
        shape = (len(x), len(self.loss_scale))
        if any(p.shape != shape or not np.isfinite(p).all() for p in predictions.values()):
            raise FloatingPointError('Incomplete or nonfinite representation forecast')
        return predictions, {'candidates': diagnostics, 'nodes': nodes, 'targets_accepted': False}


def fit_suite(a_x, a_y, b_x, b_y, *, cap_order, seed, config):
    """Fit all four arms per native cap, selecting a shared bank with P alone."""
    m = load_runtime()
    solver = numerical_solver()
    installation = solver.install(m)
    ax, ay, bx, by = [m._matrix(value, name) for name, value in
                      zip(('A_X', 'A_Y', 'B_X', 'B_Y'), (a_x, a_y, b_x, b_y))]
    if (len(ax) != len(ay) or len(bx) != len(by) or ax.shape[1] != bx.shape[1]
            or ay.shape[1] != by.shape[1]):
        raise ValueError('Inconsistent A/B history or target shapes')
    if (not cap_order or len(set(cap_order)) != len(cap_order)
            or any(not isinstance(cap, int) or cap < 1 for cap in cap_order)):
        raise ValueError('Declare a distinct positive integer native cap order')
    candidates, diagnostics, validation = {}, {}, {}
    for cap in cap_order:
        key = f'cap{cap}'
        candidates[key], diagnostics[key], arrays = fit_candidate(
            ax, ay, bx, by, cap=cap, seed=seed, config=config)
        validation.update(arrays)
    choices = {}
    for arm in ARMS:
        scores = [diagnostics[f'cap{cap}']['arms'][arm]['B_mean_smse'] for cap in cap_order]
        choices[arm] = int(np.argmin(scores))
    selections = {view: {} for view in VIEWS}
    for view in VIEWS:
        for arm in ARMS:
            cap_index = choices['P'] if view == 'shared' else choices[arm]
            cap, key = cap_order[cap_index], f'cap{cap_order[cap_index]}'
            record = diagnostics[key]['arms'][arm]
            selections[view][arm] = {
                'cap': cap, 'candidate_id': key, 'cap_index': cap_index,
                'alpha_by_horizon': record['alpha_by_horizon'],
                'alpha_indices_by_horizon': record['alpha_indices_by_horizon'],
                'B_smse_by_horizon': record['B_smse_by_horizon'],
                'B_mean_smse': record['B_mean_smse'],
                'feature_width': record['feature_width'],
                'centered_feature_rank': record['centered_feature_rank'],
                'cap_selection_arm': 'P' if view == 'shared' else arm,
                'cap_selection_split': 'B', 'tie_rule': 'first minimum in declared cap/alpha order'}
            validation[f'{view}__{arm}'] = validation[f'{key}__{arm}'].copy()
    hashes = {name: array_hash(value) for name, value in
              zip(('A_X', 'A_Y', 'B_X', 'B_Y'), (ax, ay, bx, by))}
    loss_scale = candidates[f'cap{cap_order[0]}']['loss_scale'].copy()
    suite = SoftStateSuite(list(cap_order), candidates, selections, hashes, loss_scale)
    summary = {'schema': 'kahm-soft-state-fit-diagnostics-v1', 'selections': selections,
               'candidates': diagnostics, 'input_hashes': hashes, 'loss_scale': loss_scale.tolist(),
               'native_cap_order': list(cap_order), 'arms': list(ARMS),
               'numerical_solver_installation': installation,
               'runtime_provenance': runtime_provenance(),
               'A_B_only': True, 'C_used': False, 'D_used': False,
               'state_fingerprint_before_C': state_fingerprint(suite.to_state())}
    return suite, summary, validation
