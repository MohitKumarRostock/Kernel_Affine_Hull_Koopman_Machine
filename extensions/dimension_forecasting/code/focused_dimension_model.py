"""Staged principal ten-procedure comparison with B-only dimension selection.

The verified Step 1 runtime supplies all numerical forecasting primitives.
This adapter separates A/B construction from C calibration and policy choice.
No fitting or prediction API accepts terminal targets.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ridge', 'common', 'kahm_departure', 'rbf_departure',
           'kahm_unbounded', 'rbf_unbounded')
GEOMETRY_METHODS = METHODS[1:]
SHARED_METHODS = ('ridge', 'raw_rbf', 'raw_hgb', 'persistence')
FULL_METHODS = ('ridge', 'raw_rbf', 'raw_hgb', 'persistence', 'common',
                'kahm_departure', 'rbf_departure', 'kahm_unbounded',
                'rbf_unbounded', 'nonkahm_policy')
NONKAHM_METHODS = ('ridge', 'raw_rbf', 'raw_hgb', 'persistence',
                   'common', 'rbf_departure', 'rbf_unbounded')
CONTEXTS = ('historical2', 'native20', 'selected')
_RUNTIME = None
_PROVENANCE = None


def load_runtime():
    global _RUNTIME, _PROVENANCE
    if _RUNTIME is None:
        path = ROOT / 'runtime/step1/forecasting_corrected/code'
        sys.path.insert(0, str(path))
        from geometry_adapter import _implementation
        _RUNTIME, _PROVENANCE = _implementation()
    return _RUNTIME


def runtime_provenance():
    load_runtime()
    return copy.deepcopy(_PROVENANCE)


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)},
                        sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + value.tobytes()).hexdigest()


def state_bytes(state):
    return pickle.dumps(state, protocol=5)


def state_hash(state):
    return hashlib.sha256(state_bytes(state)).hexdigest()


def _fit_arrays(ax, ay, bx, by):
    m = load_runtime()
    arrays = [m._matrix(value, name) for name, value in
              zip(('A_X', 'A_Y', 'B_X', 'B_Y'), (ax, ay, bx, by))]
    ax, ay, bx, by = arrays
    if len(ax) != len(ay) or len(bx) != len(by) or ax.shape[1] != bx.shape[1] or ay.shape[1] != by.shape[1]:
        raise ValueError('Inconsistent A/B feature and target shapes')
    return arrays


def _config(rank_cap, dimension_policy, config):
    m = load_runtime()
    supplied = dict(config or {})
    if 'rank' in supplied or 'dimension_policy' in supplied:
        raise ValueError('Pass rank_cap and dimension_policy explicitly')
    supplied.update(rank=rank_cap, dimension_policy=dimension_policy)
    effective = m.dimension_config(m.DEFAULT_CONFIG, supplied)
    if not effective['alpha_grid'] or any(not np.isfinite(a) or a <= 0 for a in effective['alpha_grid']):
        raise ValueError('Positive finite alpha grid required')
    if (not effective['gamma_grid'] or effective['gamma_grid'][0] != 0
            or any(not np.isfinite(g) or g < 0 or g > 1 for g in effective['gamma_grid'])):
        raise ValueError('Gamma grid must begin at0 and stay in[0,1]')
    return effective


@dataclass
class BaselineFit:
    input_scaler: dict
    head: dict
    predictions_a: np.ndarray
    predictions_b: np.ndarray
    residual_a: np.ndarray
    loss_scale: np.ndarray
    diagnostics: dict
    input_hashes: dict
    nodes: dict
    config: dict


def fit_baseline(a_x, a_y, b_x, b_y, *, config=None):
    """A-only ridge fits, B-only per-horizon alpha selection; no C arguments."""
    m = load_runtime()
    a_x, a_y, b_x, b_y = _fit_arrays(a_x, a_y, b_x, b_y)
    effective = _config(20, 'native', config)
    nodes = {}
    with m.timed(nodes, 'input_scaler_A'):
        scaler = m._scaler_fit(a_x, effective['feature_scale_floor'])
    with m.timed(nodes, 'baseline_transform_A_B'):
        za, zb = [m._standardize(scaler, x) for x in (a_x, b_x)]
    head, diagnostics = m._select_ridge(za, a_y, zb, b_y, np.zeros_like(b_y),
                                      effective['alpha_grid'], nodes, 'baseline')
    with m.timed(nodes, 'baseline_predict_A_B'):
        pa, pb = [m._ridge_predict(head, z) for z in (za, zb)]
    hashes = {name: array_hash(value) for name, value in
              zip(('A_X', 'A_Y', 'B_X', 'B_Y'), (a_x, a_y, b_x, b_y))}
    return BaselineFit(scaler, head, pa, pb, a_y - pa,
        np.maximum(np.var(a_y, axis=0, ddof=0), effective['target_variance_floor']),
        diagnostics, hashes, nodes, effective)


@dataclass
class DimensionCandidate:
    candidate_id: str
    rank_cap: int
    dimension_policy: str
    input_scaler: dict
    baseline: dict
    geometry: object
    heads: dict
    b_predictions: dict
    loss_scale: np.ndarray
    diagnostics: dict
    config: dict

    def predict_uncalibrated(self, x, *, nodes=None, tag='query'):
        """Evaluate fixed A/B state on histories only; never select or fit."""
        m = load_runtime()
        x = m._matrix(x, 'query histories')
        if x.shape[1] != len(self.input_scaler['mean']):
            raise ValueError('Query history width differs from fitted state')
        nodes = {} if nodes is None else nodes
        with m.timed(nodes, 'baseline_transform_' + tag):
            z = m._standardize(self.input_scaler, x)
        with m.timed(nodes, 'baseline_predict_' + tag):
            baseline = m._ridge_predict(self.baseline, z)
        features, diagnostics = self.geometry.transform(x, nodes=nodes, tag=tag)
        predictions = {'ridge': baseline}
        for method in GEOMETRY_METHODS:
            with m.timed(nodes, method + '_predict_' + tag):
                predictions[method] = baseline + m._ridge_predict(self.heads[method], features[method])
            if predictions[method].shape != baseline.shape or not np.isfinite(predictions[method]).all():
                raise FloatingPointError('Invalid fixed candidate forecast: ' + method)
        return predictions, diagnostics

    def local_dimension_records(self):
        records = []
        for index, model in enumerate(self.geometry.bank['models']):
            record = copy.deepcopy(model['dimension_audit'])
            record['model_index'] = index
            record['stored_basis_shape'] = list(model['p'].shape)
            record['stored_reference_shape'] = list(model['q'].shape)
            record['block_index'] = int(model['block'])
            record['basis_sha256'] = array_hash(model['p'])
            record['reference_sha256'] = array_hash(model['q'])
            record['stored_reference_dtype'] = str(model['q'].dtype)
            if record['effective_rank'] != model['p'].shape[1] or record['effective_rank'] != model['rank']:
                raise AssertionError('Actual local basis disagrees with rank audit')
            records.append(record)
        return records


def fit_candidate(a_x, a_y, b_x, b_y, *, candidate_id, rank_cap,
                  dimension_policy='native', seed, config=None, baseline_fit=None):
    """Construct one declared cap using A/B only, preserving source predictors.

    Inherited local numerical guards remain in the source primitives and are
    logged. An unexpected bank/head exception propagates to the study worker;
    it is never replaced by an apparently successful cap candidate.
    """
    m = load_runtime()
    a_x, a_y, b_x, b_y = _fit_arrays(a_x, a_y, b_x, b_y)
    effective = _config(rank_cap, dimension_policy, config)
    base = fit_baseline(a_x, a_y, b_x, b_y, config=config) if baseline_fit is None else baseline_fit
    actual_hashes = {name: array_hash(value) for name, value in
                     zip(('A_X', 'A_Y', 'B_X', 'B_Y'), (a_x, a_y, b_x, b_y))}
    if actual_hashes != base.input_hashes:
        raise ValueError('Shared baseline was fitted from different A/B arrays')
    for key in ('alpha_grid', 'feature_scale_floor', 'target_variance_floor'):
        if base.config[key] != effective[key]:
            raise ValueError('Shared baseline configuration differs: ' + key)
    nodes = {}
    bank = m._InstrumentedBank(seed=seed, config=effective, geometry_scaling='standard').fit(a_x)
    factor_failures = [record for record in bank.diagnostics['local_models'] if record.get('factor_error')]
    if factor_failures:
        raise FloatingPointError('Unexpected local factor failure: ' + repr(factor_failures))
    nodes.update(bank.nodes)
    state = bank.state('kahm')
    for model in state['models']:
        model.pop('reference_distance', None)
    raw, features, feature_diagnostics = {}, {}, {}
    for label, x in (('A', a_x), ('B', b_x)):
        raw[label], feature_diagnostics[label] = m._raw_feature_maps(state, x, nodes, label)
    scalers = {}
    for key in ('common', 'kahm', 'rbf'):
        with m.timed(nodes, f'{key}_scaler_A'):
            scalers[key] = m._scaler_fit(raw['A'][key], effective['feature_scale_floor'])
    geometry = m.ReconstructionFeatures(state, scalers)
    for label in ('A', 'B'):
        features[label], transformed = geometry.transform_raw(raw[label], nodes=nodes, tag=label)
        feature_diagnostics[label].update(transformed)
    heads, b_predictions = {}, {'ridge': base.predictions_b.copy()}
    selection = {'ridge': copy.deepcopy(base.diagnostics)}
    for method in GEOMETRY_METHODS:
        heads[method], selection[method] = m._select_ridge(features['A'][method], base.residual_a,
            features['B'][method], b_y, base.predictions_b, effective['alpha_grid'], nodes, method)
        with m.timed(nodes, method + '_selected_predict_B'):
            b_predictions[method] = base.predictions_b + m._ridge_predict(heads[method], features['B'][method])
    diagnostics = {'candidate_id':candidate_id, 'requested_cap':int(rank_cap),
        'dimension_policy':dimension_policy, 'seed':int(seed),
        'input_hashes':actual_hashes, 'fit_and_alpha_selection_splits':['A','B'],
        'calibration_used':False, 'bank':bank.diagnostics, 'B':selection,
        'features_A_B':feature_diagnostics, 'fit_nodes':nodes,
        'A_target_variance':base.loss_scale.tolist(),
        'B_prediction_sha256':{key:array_hash(value) for key,value in b_predictions.items()},
        'scope':'All five principal geometry heads; source numerical primitives unchanged; A/B construction precedes C'}
    candidate = DimensionCandidate(candidate_id, int(rank_cap), dimension_policy,
        copy.deepcopy(base.input_scaler), copy.deepcopy(base.head), geometry, heads,
        b_predictions, base.loss_scale.copy(), diagnostics, effective)
    diagnostics['local_dimensions'] = candidate.local_dimension_records()
    return candidate


def select_cap(candidates, b_y, *, cap_order, method='kahm_departure'):
    """Freeze one overall cap from pre-C B predictions; ties retain cap order."""
    m = load_runtime()
    b_y = m._matrix(b_y, 'B targets for cap selection')
    if method not in GEOMETRY_METHODS or not cap_order or len(set(cap_order)) != len(cap_order):
        raise ValueError('Invalid declared selection method or candidate order')
    if set(candidates) != set(cap_order):
        raise ValueError('Exactly the declared candidate roster is required')
    records = []
    first = candidates[cap_order[0]]
    for index, key in enumerate(cap_order):
        candidate = candidates[key]
        if candidate.diagnostics['input_hashes']['B_Y'] != array_hash(b_y):
            raise ValueError('Selection targets differ from candidate B targets')
        np.testing.assert_array_equal(candidate.loss_scale, first.loss_scale)
        prediction = candidate.b_predictions[method]
        mse = m._mse(b_y, prediction)
        score = float(np.mean(mse / candidate.loss_scale))
        records.append({'candidate_id':key, 'candidate_index':index,
            'requested_cap':candidate.rank_cap, 'dimension_policy':candidate.dimension_policy,
            'mse_by_horizon':mse.tolist(), 'normalized_score':score,
            'B_prediction_sha256':array_hash(prediction), 'pre_C_state_sha256':state_hash(candidate)})
    chosen_index = int(np.argmin([r['normalized_score'] for r in records]))
    chosen = cap_order[chosen_index]
    return chosen, {'schema':'kahm-cap-selection-v1', 'method':method,
        'selection_split':'B', 'calibration_used':False,
        'score':'mean_h(MSE_B_h / max(Var_A(Y_h,ddof=0),variance_floor))',
        'candidate_order':list(cap_order), 'tie_rule':'First minimum in declared order',
        'selected_candidate_id':chosen, 'selected_index':chosen_index,
        'selected_cap':candidates[chosen].rank_cap, 'candidates':records,
        'A_target_variance':first.loss_scale.tolist(), 'B_target_sha256':array_hash(b_y)}


@dataclass
class CalibratedCandidate:
    candidate: DimensionCandidate
    gammas: dict
    calibration: dict
    bounds: dict

    def predict(self, x):
        m = load_runtime()
        nodes = {}
        raw, diagnostics = self.candidate.predict_uncalibrated(x, nodes=nodes)
        prediction = {'ridge':raw['ridge']}
        for method in GEOMETRY_METHODS:
            with m.timed(nodes, method + '_blend_query'):
                prediction[method] = m._blend(raw['ridge'], raw[method], self.gammas[method])
            bound = copy.deepcopy(self.bounds[method])
            maximum = np.max(np.abs(prediction[method] - raw['ridge']), axis=0)
            limit = bound['correction_bound_by_horizon']
            bound.update(max_abs_correction_by_horizon=maximum.tolist(),
                         violations=(None if limit is None else int(np.count_nonzero(
                             maximum > np.asarray(limit) * (1+1e-10) + 1e-12))))
            diagnostics.setdefault('bounds', {})[method] = bound
        diagnostics['gamma_by_horizon'] = {k:v.tolist() for k,v in self.gammas.items()}
        return {'predictions':prediction,
                'uncalibrated_predictions':{k:raw[k] for k in GEOMETRY_METHODS},
                'diagnostics':diagnostics, 'nodes':nodes}


def calibrate(candidate, c_x, c_y):
    """Apply the unchanged C-only gamma selection to a frozen A/B candidate."""
    m = load_runtime()
    c_x, c_y = m._matrix(c_x, 'C histories'), m._matrix(c_y, 'C targets')
    if len(c_x) != len(c_y):
        raise ValueError('C history/target row counts differ')
    before = state_hash(candidate)
    nodes = {}
    raw, query_diagnostics = candidate.predict_uncalibrated(c_x, nodes=nodes, tag='C')
    gammas, records, bounds = {}, {}, {}
    calibrated = {'ridge':raw['ridge']}
    for method in GEOMETRY_METHODS:
        with m.timed(nodes, method + '_calibrate_C'):
            gammas[method], calibrated[method], records[method] = m._select_gamma(
                c_y, raw['ridge'], raw[method], candidate.config['gamma_grid'])
        bounds[method] = m._readout_bound(candidate.heads[method],
            (None if method.endswith('unbounded') else
             math.sqrt(candidate.heads[method]['feature_width'])), gammas[method])
    if state_hash(candidate) != before:
        raise AssertionError('C calibration mutated the frozen A/B candidate')
    calibration = {'C':records, 'query_diagnostics':query_diagnostics,
        'C_X_sha256':array_hash(c_x), 'C_Y_sha256':array_hash(c_y),
        'candidate_pre_C_state_sha256':before,
        'candidate_post_C_state_sha256':before,
        'uncalibrated_prediction_sha256':{k:array_hash(v) for k,v in raw.items()},
        'calibrated_prediction_sha256':{k:array_hash(v) for k,v in calibrated.items()},
        'nodes':nodes}
    return CalibratedCandidate(candidate, gammas, calibration, bounds), raw, calibrated


@dataclass
class SharedCompetitors:
    """Cap-independent A/B states, including the unchanged original raw grids."""
    baseline_fit: BaselineFit
    raw_rbf: list
    raw_hgb: list
    b_predictions: dict
    diagnostics: dict
    config: dict

    def predict_uncalibrated(self, x, *, nodes=None, tag='query'):
        m = load_runtime()
        x = m._matrix(x, 'shared query histories')
        if x.shape[1] != len(self.baseline_fit.input_scaler['mean']):
            raise ValueError('Shared query width differs from A')
        nodes = {} if nodes is None else nodes
        with m.timed(nodes, 'baseline_transform_' + tag):
            z = m._standardize(self.baseline_fit.input_scaler, x)
        with m.timed(nodes, 'baseline_predict_' + tag):
            baseline = m._ridge_predict(self.baseline_fit.head, z)
        result = {'ridge':baseline}
        with m.timed(nodes, 'persistence_predict_' + tag):
            result['persistence'] = np.repeat(x[:, -1:], baseline.shape[1], axis=1)
        cache = {}
        with m.timed(nodes, 'raw_rbf_predict_' + tag):
            values = []
            for state in self.raw_rbf:
                key = state['features']['bandwidth_factor']
                if key not in cache:
                    cache[key] = m._raw_rbf_features(state['features'], z)
                values.append(m._ridge_predict(state['head'], cache[key])[:,0])
            result['raw_rbf'] = np.column_stack(values)
        with m.timed(nodes, 'raw_hgb_predict_' + tag):
            result['raw_hgb'] = np.column_stack([model.predict(x) for model in self.raw_hgb])
        for method,value in result.items():
            if value.shape != baseline.shape or not np.isfinite(value).all():
                raise FloatingPointError('Invalid shared forecast: ' + method)
        return {method:result[method] for method in SHARED_METHODS}


def fit_shared(a_x,a_y,b_x,b_y,*,seed,config=None):
    """Fit the original raw grids once, selecting exclusively on B.

    The original helpers' final query argument is B. It is used only to
    evaluate their selected states; the original candidate grids are unchanged.
    The helpers' C-labelled timing nodes are renamed to reflect this B query.
    """
    m=load_runtime()
    a_x,a_y,b_x,b_y=_fit_arrays(a_x,a_y,b_x,b_y)
    base=fit_baseline(a_x,a_y,b_x,b_y,config=config)
    effective=base.config
    nodes=copy.deepcopy(base.nodes)
    za,zb=[m._standardize(base.input_scaler,x) for x in (a_x,b_x)]
    rbf,pb_rbf,diag_rbf=m._fit_raw_rbf(za,a_y,zb,b_y,zb,seed,effective,nodes)
    hgb,pb_hgb,diag_hgb=m._fit_raw_hgb(a_x,a_y,b_x,b_y,b_x,seed,effective,nodes)
    for name,record in (('raw_rbf',diag_rbf),('raw_hgb',diag_hgb)):
        if record['failures']:
            raise FloatingPointError('Unexpected declared raw grid failure: ' + name + repr(record['failures']))
    for old in ('raw_rbf_transform_A_B_C','raw_rbf_predict_C','raw_hgb_predict_C'):
        if old in nodes:
            nodes[old.replace('_C','_B_initial_query')]=nodes.pop(old)
    pb={'ridge':base.predictions_b.copy(),'raw_rbf':pb_rbf,'raw_hgb':pb_hgb,
        'persistence':np.repeat(b_x[:,-1:],a_y.shape[1],axis=1)}
    if any(not np.isfinite(value).all() for value in pb.values()):
        raise FloatingPointError('Nonfinite shared B predictions')
    diagnostics={'B':{'ridge':copy.deepcopy(base.diagnostics),'raw_rbf':diag_rbf,
        'raw_hgb':diag_hgb,'persistence':{'status':'fixed','selection_split':None}},
        'fit_nodes':nodes,'input_hashes':copy.deepcopy(base.input_hashes),
        'B_prediction_sha256':{key:array_hash(value) for key,value in pb.items()},
        'calibration_used':False,'query_argument_in_original_raw_helpers':'B, not C',
        'raw_grid_failures':[]}
    return SharedCompetitors(base,rbf,hgb,pb,diagnostics,effective)


@dataclass
class CalibratedShared:
    shared: SharedCompetitors
    gammas: dict
    calibration: dict

    def predict(self,x):
        m=load_runtime();nodes={}
        raw=self.shared.predict_uncalibrated(x,nodes=nodes)
        predictions={'ridge':raw['ridge']}
        for method in SHARED_METHODS[1:]:
            with m.timed(nodes,method+'_blend_query'):
                predictions[method]=m._blend(raw['ridge'],raw[method],self.gammas[method])
        return {'predictions':predictions,'uncalibrated_predictions':raw,'nodes':nodes,
            'diagnostics':{'failures':[],'recovered_prediction_cells':{method:0 for method in SHARED_METHODS},
                           'gamma_by_horizon':{key:value.tolist() for key,value in self.gammas.items()}}}


def calibrate_shared(shared,c_x,c_y):
    m=load_runtime();c_x=m._matrix(c_x,'C histories');c_y=m._matrix(c_y,'C targets')
    if len(c_x)!=len(c_y):raise ValueError('C shared rows differ')
    before=state_hash(shared);nodes={}
    raw=shared.predict_uncalibrated(c_x,nodes=nodes,tag='C')
    calibrated={'ridge':raw['ridge']};gammas={}
    records={'ridge':{'gamma_by_horizon':[0.]*c_y.shape[1],
        'selected_mse_by_horizon':m._mse(c_y,raw['ridge']).tolist(),'status':'fixed_baseline'}}
    for method in SHARED_METHODS[1:]:
        with m.timed(nodes,method+'_calibrate_C'):
            gammas[method],calibrated[method],records[method]=m._select_gamma(
                c_y,raw['ridge'],raw[method],shared.config['gamma_grid'])
    if state_hash(shared)!=before:raise AssertionError('C mutated shared A/B state')
    record={'C':records,'C_X_sha256':array_hash(c_x),'C_Y_sha256':array_hash(c_y),
        'shared_pre_C_state_sha256':before,'shared_post_C_state_sha256':before,
        'uncalibrated_prediction_sha256':{k:array_hash(v) for k,v in raw.items()},
        'calibrated_prediction_sha256':{k:array_hash(v) for k,v in calibrated.items()},'nodes':nodes}
    return CalibratedShared(shared,gammas,record),raw,calibrated


def select_context_policies(selections,candidate_c,shared_c,c_y,variance):
    """Select the original complete non-KAHM procedure after B caps are fixed."""
    m=load_runtime();contexts={}
    for context in CONTEXTS:
        started_wall,started_cpu=time.perf_counter(),time.process_time()
        mapping={method:(selections[method]['selected_candidate_id'] if context=='selected' else context)
                 for method in GEOMETRY_METHODS}
        values={**shared_c,**{method:candidate_c[mapping[method]][method] for method in GEOMETRY_METHODS}}
        scores=[float(np.mean(m._mse(c_y,values[method])/variance)) for method in NONKAHM_METHODS]
        chosen=NONKAHM_METHODS[int(np.argmin(scores))]
        contexts[context]={'method_to_candidate':mapping,'selected_policy':chosen,
            'policy':{'selected_method':chosen,'selection_split':'C',
                'score':'mean horizon C MSE / A target variance (ddof=0, floored)',
                'A_target_variance':np.asarray(variance).tolist(),
                'candidates':[{'method':method,'normalized_mse':score} for method,score in zip(NONKAHM_METHODS,scores)],
                'candidate_order':list(NONKAHM_METHODS),'cap_choices_already_frozen_on_B':True},
            'role':'primary' if context=='selected' else 'secondary'}
        contexts[context]['selection_cost']={'wall_seconds':time.perf_counter()-started_wall,
                                             'cpu_seconds':time.process_time()-started_cpu}
    return contexts


def prediction_views(candidate_results,shared_result,contexts,matched_candidate_id,*,nodes=None):
    """Materialize transparent array views without new fitting or selection."""
    arrays={};m=load_runtime();nodes={} if nodes is None else nodes
    for mode in ('calibrated','gamma1'):
        def values(result):
            return (result['predictions'] if mode=='calibrated' else
                    {'ridge':result['predictions']['ridge'],**result['uncalibrated_predictions']})
        shared=values(shared_result)
        candidates={name:values(result) for name,result in candidate_results.items()}
        for name,result in candidates.items():
            arrays.update({f'{name}__{mode}__{method}':value for method,value in result.items()})
        arrays.update({f'shared__{mode}__{method}':shared[method] for method in SHARED_METHODS})
        for name,context in contexts.items():
            current={**shared,**{method:candidates[context['method_to_candidate'][method]][method]
                                 for method in GEOMETRY_METHODS}}
            with m.timed(nodes,f'{name}/{mode}/policy_forward'):
                current['nonkahm_policy']=current[context['selected_policy']].copy()
            arrays.update({f'context_{name}__{mode}__{method}':current[method] for method in FULL_METHODS})
        arrays.update({f'matched__{mode}__{method}':candidates[matched_candidate_id][method]
                       for method in GEOMETRY_METHODS})
    return arrays
