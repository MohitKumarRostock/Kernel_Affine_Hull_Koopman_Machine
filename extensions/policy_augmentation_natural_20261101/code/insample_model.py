"""Centered in-sample departures around an unchanged full-A common forecast.

The same corrected numerical implementation used by the simulation study
supplies all maps, scalers, ridge fits, bounds, and calibration operations.
This single-series extension does not manufacture independent episode folds.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
import numpy as np

SYNTHETIC = Path(__file__).resolve().parents[2] / 'policy_augmentation_20261101'
sys.path.insert(0, str(SYNTHETIC / 'code'))
import crossfit_model as inherited
fd, previous, secondary = inherited.fd, inherited.previous, inherited.secondary
CAPS, SCALERS = inherited.CAPS, inherited.SCALERS
SLOPES = ('insample_kahm', 'insample_rbf', 'insample_matched_rbf')
METHODS = ('common', 'shared_bias', *SLOPES)


@dataclass
class InSampleSuite:
    common: object
    bias: np.ndarray
    departures: dict
    departure_candidates: dict
    selections: dict
    training_arrays: dict
    validation_predictions: dict
    config: dict
    diagnostics: dict
    etas: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def frozen_state(self):
        return {'common': self.common.predictive_state(), 'bias': self.bias,
                'departures': self.departures, 'departure_candidates': self.departure_candidates,
                'selections': self.selections, 'training_arrays': self.training_arrays,
                'config': self.config}

    def frozen_hash(self):
        return fd.state_hash(self.frozen_state())

    def _uncalibrated(self, x):
        m = fd.load_runtime()
        common, common_diag = self.common.predict(x)
        shared = common + self.bias
        predictions = {'common': common, 'shared_bias': shared}
        cache, geometry = {}, {}
        for method in SLOPES:
            state = self.departures[method]
            key = inherited._candidate_id(state.scaler, state.cap)
            if key not in cache:
                cache[key], geometry[key] = inherited._checked_raw(state.bank, x, key + '/natural_query')
            features = secondary._bounded(m, cache[key][state.kind], state.feature_scaler)
            predictions[method] = shared + secondary._slope(state.head, features)
        if tuple(predictions) != METHODS or any(not np.isfinite(p).all() for p in predictions.values()):
            raise FloatingPointError('Natural residual forecast roster is invalid')
        return predictions, {'common': common_diag, 'departure_geometry': geometry}

    def calibrate(self, cx, cy):
        if self.etas:
            raise RuntimeError('Natural residual suite already calibrated')
        before = self.frozen_hash()
        m = fd.load_runtime()
        raw, diagnostic = self._uncalibrated(cx)
        output = {k: v.copy() for k, v in raw.items()}
        records = {}
        for method in SLOPES:
            eta, output[method], record = m._select_gamma(cy, raw['shared_bias'], raw[method], self.config['gamma_grid'])
            self.etas[method] = eta
            records[method] = {**record, 'selected_eta': eta.tolist(), 'selection_split': 'C'}
        if self.frozen_hash() != before:
            raise AssertionError('Natural C calibration changed A/B state')
        self.calibration = secondary.json_safe({'choices': records, 'pre_C_hash': before,
            'post_C_hash': before, 'C_X_sha256': fd.array_hash(cx), 'C_Y_sha256': fd.array_hash(cy),
            'shared_baseline_sha256': fd.state_hash((self.common.predictive_state(), self.bias)),
            'query': diagnostic})
        return {'predictions': output, 'eta1_predictions': raw, 'records': self.calibration}

    def predict(self, x):
        if set(self.etas) != set(SLOPES):
            raise RuntimeError('Natural residual calibration is missing')
        m = fd.load_runtime()
        before = self.frozen_hash()
        raw, diagnostic = self._uncalibrated(x)
        output = {k: v.copy() for k, v in raw.items()}
        bounds = {}
        for method in SLOPES:
            output[method] = m._blend(raw['shared_bias'], raw[method], self.etas[method])
            bound = secondary._centered_bound(self.departures[method].head, self.etas[method])
            observed = np.max(np.abs(output[method] - raw['shared_bias']), axis=0)
            limit = np.asarray(bound['correction_bound_by_horizon'])
            violations = int(np.count_nonzero(observed > limit * (1 + 1e-10) + 1e-12))
            if violations:
                raise AssertionError('Natural correction exceeds inherited magnitude bound')
            bounds[method] = {**bound, 'observed_max_abs_correction_by_horizon': observed.tolist(),
                              'query_bound_violations': violations}
        if self.frozen_hash() != before:
            raise AssertionError('Natural query changed A/B state')
        diagnostic.update(bounds=bounds, frozen_state_before=before, frozen_state_after=before)
        return {'predictions': output, 'eta1_predictions': raw, 'diagnostics': secondary.json_safe(diagnostic)}


def fit_insample(ax, ay, bx, by, *, seed, config, reference_primary):
    """Select independent KAHM/RBF caps, scalers and ridge penalties on B."""
    wall, cpu = time.perf_counter(), time.process_time()
    m = fd.load_runtime()
    ax, ay, bx, by = fd._fit_arrays(ax, ay, bx, by)
    expected = {k: fd.array_hash(v) for k, v in (('A_X', ax), ('A_Y', ay), ('B_X', bx), ('B_Y', by))}
    if expected != reference_primary.diagnostics['input_hashes'] or reference_primary.gammas:
        raise ValueError('Natural residual suite requires the identical pre-C primary fit')
    key = reference_primary.selections['selected']['common']
    candidate = reference_primary.candidates[key]
    common_bank = reference_primary.banks[candidate.common_bank]
    base = reference_primary.shared.baseline_fit
    common = inherited.CommonModel(common_bank.diagnostics['arm'], candidate.cap, common_bank.bank,
        common_bank.scalers['common'], candidate.heads['common'], base.input_scaler, base.head)
    common_hash = common.state_hash()
    ca, ca_diag = common.predict(ax)
    cb, cb_diag = common.predict(bx)
    np.testing.assert_array_equal(cb, candidate.b_predictions['common'])
    bias = np.mean(by - cb, axis=0)
    shared_b, residual = cb + bias, ay - ca
    shared_hash = fd.state_hash((common.predictive_state(), bias))
    candidates, validation, records = {}, {}, {'kahm': [], 'rbf': []}
    geometry = {}
    kind_seconds = {'kahm': 0.0, 'rbf': 0.0}
    for cap in CAPS:
        for scaler in SCALERS:
            cid = inherited._candidate_id(scaler, cap)
            bank = reference_primary.banks[previous.bank_id(scaler, cap)]
            ra, da = inherited._checked_raw(bank.bank, ax, cid + '/natural_A')
            rb, db = inherited._checked_raw(bank.bank, bx, cid + '/natural_B')
            geometry[cid] = {'A': da, 'B': db, 'local_dimensions': previous.dimension_records(bank.bank)}
            for kind in ('kahm', 'rbf'):
                ts = time.perf_counter()
                scaler_state = bank.scalers[kind]
                oracle = m._scaler_fit(ra[kind], base.config['feature_scale_floor'])
                np.testing.assert_array_equal(scaler_state['mean'], oracle['mean'])
                np.testing.assert_array_equal(scaler_state['scale'], oracle['scale'])
                fa = secondary._bounded(m, ra[kind], scaler_state)
                fb = secondary._bounded(m, rb[kind], scaler_state)
                path = m._ridge_path(fa, residual, base.config['alpha_grid'])
                paths = np.stack([shared_b + secondary._slope(head, fb) for head in path])
                losses = np.mean((by[None] - paths) ** 2, axis=1)
                indices = np.argmin(losses, axis=0)
                head = m._combine_heads(path, indices)
                prediction = shared_b + secondary._slope(head, fb)
                score, mse = inherited._score(prediction, by, base.loss_scale)
                centered_mean = np.mean(secondary._slope(head, fa), axis=0)
                if np.max(np.abs(centered_mean)) > 1e-10 * max(1.0, float(np.max(np.abs(residual)))):
                    raise AssertionError('Natural centered correction has nonzero A mean')
                secondary._centered_bound(head, np.ones(ay.shape[1]))
                state_key = 'insample__' + cid + '__' + kind
                candidates[state_key] = inherited.DepartureState(scaler, cap, kind, bank.bank,
                    scaler_state, head, 'insample', centered_mean)
                validation[state_key] = prediction
                validation[state_key + '__alpha_paths'] = paths
                records[kind].append({'candidate_id': cid, 'state_key': state_key, 'scaler': scaler,
                    'cap': cap, 'score': score, 'mse_by_horizon': mse.tolist(),
                    'selected_alpha_indices': indices.tolist(),
                    'selected_alpha_by_horizon': head['alpha_by_horizon'].tolist(),
                    'B_mse_by_alpha_horizon': losses.tolist(), 'feature_width': int(head['feature_width']),
                    'B_prediction_sha256': fd.array_hash(prediction),
                    'B_alpha_prediction_sha256': fd.array_hash(paths),
                    'A_mean_centered_correction': centered_mean.tolist(),
                    'shared_baseline_sha256': shared_hash})
                kind_seconds[kind] += time.perf_counter() - ts
    departures, selections = {}, {}
    for kind in ('kahm', 'rbf'):
        index = int(np.argmin([r['score'] for r in records[kind]]))
        chosen = records[kind][index]
        departures['insample_' + kind] = candidates[chosen['state_key']]
        selections[kind] = {'selected_index': index, 'selected_candidate': chosen['candidate_id'],
            'selected_scaler': chosen['scaler'], 'selected_cap': chosen['cap'],
            'selected_alpha_by_horizon': chosen['selected_alpha_by_horizon'], 'candidates': records[kind],
            'selection_split': 'B', 'tie_rule': 'First minimum: cap ascending then scaler order'}
    matched_key = 'insample__' + selections['kahm']['selected_candidate'] + '__rbf'
    departures['insample_matched_rbf'] = candidates[matched_key]
    selections['matched_rbf'] = {'state_key': matched_key,
        'selection_rule': 'KAHM-selected scaler/cap, own RBF B penalty and C multiplier'}
    selections['common'] = {'selected_candidate': key, 'selected_scaler': common.scaler,
        'selected_cap': common.cap, 'common_gamma': 1.0, 'selection_split': 'B',
        'state_sha256': common_hash}
    selections['shared_bias'] = {'value': bias.tolist(), 'fit_split': 'B', 'state_sha256': shared_hash}
    if common.state_hash() != common_hash:
        raise AssertionError('Natural residual fitting changed common state')
    arrays = {'A_full_common': ca, 'A_insample_residual': residual, 'B_common': cb, 'B_shared_bias': shared_b}
    diagnostics = secondary.json_safe({'schema': 'natural-insample-full-common-v1', 'input_hashes': expected,
        'seed': int(seed), 'A_rows': len(ax), 'B_rows': len(bx), 'independent_episode_folds': 0,
        'A_target_variance': base.loss_scale.tolist(), 'full_common_A': ca_diag, 'full_common_B': cb_diag,
        'geometry': geometry, 'feature_configurations_per_kind': 15,
        'B_candidate_scores_per_horizon_per_kind': 15 * len(base.config['alpha_grid']),
        'head_fit_seconds_by_kind': kind_seconds, 'fit_wall_seconds': time.perf_counter() - wall,
        'fit_cpu_seconds': time.process_time() - cpu})
    return InSampleSuite(common, bias, departures, candidates, selections, arrays, validation,
        copy.deepcopy(base.config), diagnostics)
