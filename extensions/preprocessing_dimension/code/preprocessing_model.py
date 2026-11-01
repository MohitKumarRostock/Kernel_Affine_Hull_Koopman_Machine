"""Finite preprocessing/cap search using the published immutable KAHM runtime.

The principal heads refit concatenated features. Fixed-common mechanisms are
implemented separately in fixed_common_secondary.py. No terminal target is an
argument to construction, calibration, or prediction.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import time
import numpy as np

OLD = Path(__file__).resolve().parents[2] / 'dimension_forecasting'
sys.path.insert(0, str(OLD / 'code'))
import focused_dimension_model as fd

SCALERS = ('standard', 'minmax', 'tanh_z_1')
CAPS = (1, 2, 4, 8, 12, 20)
PAIRS = {
    'standard': ('standard', 'standard'),
    'minmax': ('minmax', 'minmax'),
    'tanh': ('tanh_z_1', 'tanh_z_1'),
    'hybrid_tanh_minmax': ('tanh_z_1', 'minmax'),
    'hybrid_minmax_tanh': ('minmax', 'tanh_z_1'),
}
GEOMETRY = fd.GEOMETRY_METHODS
METHODS = fd.FULL_METHODS
CONTEXTS = ('selected', 'cap2', 'cap20', *PAIRS)


def bank_id(scaler, cap):
    return f'{scaler}__cap{cap}'


def candidate_id(pair, cap):
    return f'{pair}__cap{cap}'


def input_scaled(x, arm, audit, base):
    m = fd.load_runtime()
    if arm == 'standard':
        return m._standardize(base.input_scaler, x)
    from geometry_adapter import scaled_histories
    return scaled_histories(x, audit)


def dimension_records(bank):
    result = []
    for j, local in enumerate(bank['models']):
        record = copy.deepcopy(local['dimension_audit'])
        record.update(model_index=j, block=int(local['block']),
                      stored_basis_shape=list(local['p'].shape),
                      stored_reference_shape=list(local['q'].shape),
                      basis_sha256=fd.array_hash(local['p']),
                      reference_sha256=fd.array_hash(local['q']))
        if record['effective_rank'] != local['p'].shape[1]:
            raise AssertionError('Local rank metadata differs from stored basis')
        result.append(record)
    return result


def checked_raw(bank, x, nodes, tag):
    m = fd.load_runtime()
    raw, diagnostics = m._raw_feature_maps(bank, x, nodes, tag)
    if any(not np.isfinite(a).all() for a in raw.values()):
        raise FloatingPointError('Nonfinite raw feature map')
    if any(diagnostics.get(key) for key in ('coordinate_fallback_rows', 'gate_fallback_rows', 'kernel_fallback_rows')):
        raise FloatingPointError('Unexpected coordinate, gate, or kernel fallback')
    if any(row.get('factor_failure') for row in diagnostics.get('local_models', [])):
        raise FloatingPointError('Unexpected reconstruction factor failure')
    return raw, diagnostics


@dataclass
class Bank:
    bank: dict
    scalers: dict
    diagnostics: dict

    def features(self, raw):
        m = fd.load_runtime()
        z = {k: m._standardize(self.scalers[k], raw[k]) for k in self.scalers}
        bounded = {k: m.radial_bound(z[k]) for k in z}
        return z, bounded


def assemble(common, departure):
    zc, bc = common
    zd, bd = departure
    values = {'common': bc['common']}
    for kind in ('kahm', 'rbf'):
        values[kind + '_departure'] = np.column_stack((bc['common'], bd[kind]))
        values[kind + '_unbounded'] = np.column_stack((bc['common'], zd[kind]))
    return values


@dataclass
class Candidate:
    pair: str
    cap: int
    common_bank: str
    departure_bank: str
    heads: dict
    b_predictions: dict
    diagnostics: dict


@dataclass
class PrimarySuite:
    shared: object
    banks: dict
    candidates: dict
    raw_rbf_states: dict
    raw_rbf_audits: dict
    raw_rbf_selected_scaler: str
    selections: dict
    b_scores: dict
    config: dict
    diagnostics: dict
    gammas: dict = field(default_factory=dict)
    shared_gammas: dict = field(default_factory=dict)
    policies: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def _shared_predict(self, x, nodes, tag):
        m = fd.load_runtime()
        base = self.shared.baseline_fit
        with m.timed(nodes, 'baseline_query'):
            baseline = m._ridge_predict(base.head, m._standardize(base.input_scaler, x))
        with m.timed(nodes, 'raw_hgb_query'):
            hgb = np.column_stack([h.predict(x) for h in self.shared.raw_hgb])
        arm = self.raw_rbf_selected_scaler
        states = self.raw_rbf_states[arm]
        with m.timed(nodes, 'raw_rbf_query'):
            z = input_scaled(x, arm, self.raw_rbf_audits[arm], base)
            cache = {}
            columns = []
            for state in states:
                factor = state['features']['bandwidth_factor']
                if factor not in cache:
                    cache[factor] = m._raw_rbf_features(state['features'], z)
                columns.append(m._ridge_predict(state['head'], cache[factor])[:, 0])
            rbf = np.column_stack(columns)
        return {'ridge': baseline, 'raw_rbf': rbf, 'raw_hgb': hgb,
                'persistence': np.repeat(x[:, -1:], baseline.shape[1], axis=1)}

    def predict_uncalibrated(self, x, tag='query'):
        m = fd.load_runtime()
        x = m._matrix(x, 'query histories')
        nodes, bank_diagnostics, features = {}, {}, {}
        shared = self._shared_predict(x, nodes, tag)
        for key, bank in self.banks.items():
            local_nodes = {}
            raw, diag = checked_raw(bank.bank, x, local_nodes, tag)
            features[key] = bank.features(raw)
            nodes.update({key + '/' + k: v for k, v in local_nodes.items()})
            bank_diagnostics[key] = diag
        predictions = {}
        for key, candidate in self.candidates.items():
            f = assemble(features[candidate.common_bank], features[candidate.departure_bank])
            predictions[key] = {}
            for method in GEOMETRY:
                with m.timed(nodes, key + '/' + method + '_query'):
                    predictions[key][method] = shared['ridge'] + m._ridge_predict(candidate.heads[method], f[method])
        return predictions, shared, {'banks': bank_diagnostics, 'nodes': nodes}

    def calibrate(self, cx, cy):
        m = fd.load_runtime()
        before = fd.state_hash((self.shared, self.banks, self.candidates, self.selections,
                                self.raw_rbf_states, self.raw_rbf_selected_scaler))
        raw, shared, query = self.predict_uncalibrated(cx, 'C')
        records, calibrated = {}, {}
        for key, values in raw.items():
            self.gammas[key], records[key], calibrated[key] = {}, {}, {}
            for method in GEOMETRY:
                g, p, rec = m._select_gamma(cy, shared['ridge'], values[method], self.config['gamma_grid'])
                self.gammas[key][method], calibrated[key][method], records[key][method] = g, p, rec
        shared_cal = {'ridge': shared['ridge']}
        shared_records = {}
        for method in fd.SHARED_METHODS[1:]:
            g, p, rec = m._select_gamma(cy, shared['ridge'], shared[method], self.config['gamma_grid'])
            self.shared_gammas[method], shared_cal[method], shared_records[method] = g, p, rec
        for context, mapping in self.selections.items():
            values = {**shared_cal, **{method: calibrated[mapping[method]][method] for method in GEOMETRY}}
            scores = {method: float(np.mean(m._mse(cy, values[method]) / self.shared.baseline_fit.loss_scale))
                      for method in fd.NONKAHM_METHODS}
            choice = min(fd.NONKAHM_METHODS, key=scores.__getitem__)
            self.policies[context] = {'selected_method': choice, 'scores': scores,
                                      'candidate_order': list(fd.NONKAHM_METHODS), 'split': 'C'}
        after = fd.state_hash((self.shared, self.banks, self.candidates, self.selections,
                               self.raw_rbf_states, self.raw_rbf_selected_scaler))
        if before != after:
            raise AssertionError('C calibration mutated frozen development state')
        self.calibration = {'candidate_records': records, 'shared_records': shared_records,
                            'pre_C_hash': before, 'post_C_hash': after,
                            'C_X_sha256': fd.array_hash(cx), 'C_Y_sha256': fd.array_hash(cy),
                            'policies': self.policies, 'query': query}
        return self.views(raw, shared), self.calibration

    def views(self, candidates, shared):
        m = fd.load_runtime()
        arrays = {}
        for mode in ('calibrated', 'gamma1'):
            if mode == 'calibrated':
                cur_shared = {'ridge': shared['ridge'], **{
                    method: m._blend(shared['ridge'], shared[method], self.shared_gammas[method])
                    for method in fd.SHARED_METHODS[1:]}}
                current = {key: {method: m._blend(shared['ridge'], p, self.gammas[key][method])
                                 for method, p in values.items()} for key, values in candidates.items()}
            else:
                cur_shared, current = shared, candidates
            for key, values in current.items():
                for method, p in values.items():
                    arrays[f'candidate__{key}__{mode}__{method}'] = p
            for context, mapping in self.selections.items():
                values = {**cur_shared, **{method: current[mapping[method]][method] for method in GEOMETRY}}
                values['nonkahm_policy'] = values[self.policies[context]['selected_method']]
                for method in METHODS:
                    arrays[f'primary__{context}__{mode}__{method}'] = values[method]
        return arrays

    def predict(self, x):
        before = fd.state_hash(self)
        raw, shared, diagnostics = self.predict_uncalibrated(x)
        arrays = self.views(raw, shared)
        bound_checks = []
        m = fd.load_runtime()
        for key, candidate in self.candidates.items():
            for method, head in candidate.heads.items():
                radius = None if method.endswith('unbounded') else math.sqrt(head['feature_width'])
                bound = m._readout_bound(head, radius, self.gammas[key][method])
                if bound['coefficient_bound_violations']:
                    raise AssertionError('Ridge coefficient bound violation')
                if bound['correction_bound_by_horizon'] is not None:
                    p = arrays[f'candidate__{key}__calibrated__{method}']
                    actual = np.max(np.abs(p - shared['ridge']), axis=0)
                    limit = np.asarray(bound['correction_bound_by_horizon'])
                    if np.any(actual > limit * (1 + 1e-10) + 1e-12):
                        raise AssertionError('Prediction exceeds inherited readout bound')
                    bound['max_abs_correction'] = actual.tolist()
                bound_checks.append({'candidate': key, 'method': method, **bound})
        if before != fd.state_hash(self):
            raise AssertionError('Prediction mutated fitted primary state')
        diagnostics.update(bounds=bound_checks, state_before=before, state_after=before)
        return arrays, diagnostics


def fit_primary(ax, ay, bx, by, *, seed, config):
    m = fd.load_runtime()
    if ax.shape[1] != 12:
        raise ValueError('This frozen study requires ambient history dimension twelve')
    from geometry_adapter import fit_scaler_audit
    shared = fd.fit_shared(ax, ay, bx, by, seed=seed, config=config)
    base = shared.baseline_fit
    raw_states = {'standard': shared.raw_rbf}
    raw_audits = {arm: fit_scaler_audit(ax, arm) for arm in SCALERS}
    raw_b = {'standard': shared.b_predictions['raw_rbf']}
    raw_diagnostics = {'standard': copy.deepcopy(shared.diagnostics['B']['raw_rbf'])}
    for arm in SCALERS[1:]:
        nodes = {}
        za, zb = [input_scaled(x, arm, raw_audits[arm], base) for x in (ax, bx)]
        state, prediction, diagnostic = m._fit_raw_rbf(za, ay, zb, by, zb, seed, base.config, nodes)
        if diagnostic['failures']:
            raise FloatingPointError('Raw RBF declared preprocessing trial failed')
        raw_states[arm], raw_b[arm] = state, prediction
        raw_diagnostics[arm] = {**diagnostic, 'nodes': nodes}
    rbf_scores = {arm: float(np.mean(m._mse(by, raw_b[arm]) / base.loss_scale)) for arm in SCALERS}
    rbf_choice = min(SCALERS, key=rbf_scores.__getitem__)
    banks, fit_features, bank_nodes = {}, {}, {}
    for arm in SCALERS:
        for cap in CAPS:
            key = bank_id(arm, cap)
            effective = fd._config(cap, 'native', config)
            source = m._InstrumentedBank(seed=seed, config=effective, geometry_scaling=arm).fit(ax)
            if any(row.get('factor_error') for row in source.diagnostics['local_models']):
                raise FloatingPointError('Local KAHM factor failure')
            state = source.state('kahm')
            for local in state['models']:
                local.pop('reference_distance', None)
            nodes = dict(source.nodes)
            raw, records = {}, {}
            for split, x in (('A', ax), ('B', bx)):
                raw[split], records[split] = checked_raw(state, x, nodes, split)
            scalers = {kind: m._scaler_fit(raw['A'][kind], effective['feature_scale_floor'])
                       for kind in ('common', 'kahm', 'rbf')}
            bank = Bank(state, scalers, {'arm': arm, 'cap': cap, 'bank': source.diagnostics,
                        'local_dimensions': dimension_records(state), 'features': records, 'nodes': nodes})
            banks[key] = bank
            fit_features[key] = {split: bank.features(raw[split]) for split in ('A', 'B')}
            bank_nodes[key] = nodes
    # Same scaler => same partitions/references across caps. Different scalers
    # need not induce the same partition, which is part of preprocessing.
    reference_checks = 0
    for arm in SCALERS:
        reference = banks[bank_id(arm, 1)].bank
        for cap in CAPS:
            actual = banks[bank_id(arm, cap)].bank
            if actual['block_slices'] != reference['block_slices']:
                raise AssertionError('Partitions changed with cap')
            for left, right in zip(reference['models'], actual['models']):
                for key in ('q', 'center', 'radius', 'block'):
                    np.testing.assert_array_equal(left[key], right[key])
                    reference_checks += 1
    candidates, b_scores = {}, {}
    for pair, (common_arm, departure_arm) in PAIRS.items():
        for cap in CAPS:
            key, cb, db = candidate_id(pair, cap), bank_id(common_arm, cap), bank_id(departure_arm, cap)
            features = {split: assemble(fit_features[cb][split], fit_features[db][split]) for split in ('A', 'B')}
            nodes, heads, predictions, records = {}, {}, {}, {}
            for method in GEOMETRY:
                heads[method], records[method] = m._select_ridge(features['A'][method], base.residual_a,
                    features['B'][method], by, base.predictions_b, config['alpha_grid'], nodes, method)
                predictions[method] = base.predictions_b + m._ridge_predict(heads[method], features['B'][method])
            candidates[key] = Candidate(pair, cap, cb, db, heads, predictions,
                                        {'B': records, 'nodes': nodes, 'common_bank': cb, 'departure_bank': db})
            b_scores[key] = {method: float(np.mean(m._mse(by, p) / base.loss_scale))
                             for method, p in predictions.items()}
    identity_checks = 0
    for pair in PAIRS:
        for method in GEOMETRY:
            np.testing.assert_array_equal(candidates[candidate_id(pair, 12)].b_predictions[method],
                                          candidates[candidate_id(pair, 20)].b_predictions[method])
            identity_checks += 1
    selections = {}
    # Tie order: cap ascending, then standard/minmax/tanh/declared hybrids.
    all_keys = [candidate_id(pair, cap) for cap in CAPS for pair in PAIRS]
    for context in CONTEXTS:
        selections[context] = {}
        for method in GEOMETRY:
            eligible = all_keys
            if context in ('cap2', 'cap20'):
                eligible = [key for key in eligible if candidates[key].cap == int(context[3:])]
            elif context in PAIRS:
                eligible = [key for key in eligible if candidates[key].pair == context]
            elif method == 'common':
                eligible = [key for key in eligible if candidates[key].pair in ('standard', 'minmax', 'tanh')]
            if method == 'common' and context in ('cap2', 'cap20'):
                eligible = [key for key in eligible if candidates[key].pair in ('standard', 'minmax', 'tanh')]
            selections[context][method] = min(eligible, key=lambda key: b_scores[key][method])
    diagnostics = {'input_hashes': {key: fd.array_hash(x) for key, x in
                   zip(('A_X', 'A_Y', 'B_X', 'B_Y'), (ax, ay, bx, by))},
                   'raw_rbf': raw_diagnostics, 'raw_rbf_scores': rbf_scores,
                   'reference_identity_checks': reference_checks,
                   'cap12_cap20_identity_checks': identity_checks,
                   'runtime_provenance': fd.runtime_provenance(),
                   'shared_fit_nodes': shared.diagnostics['fit_nodes'],
                   'architecture': 'Complete concatenated heads; not fixed-common slopes'}
    return PrimarySuite(shared, banks, candidates, raw_states, raw_audits, rbf_choice,
                        selections, b_scores, base.config, diagnostics)
