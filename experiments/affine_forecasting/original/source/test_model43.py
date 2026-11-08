"""Artificial-only integration checks for the Step 43 method adapters.

No source measurement, prior fitted model or assessment outcome is opened.
These checks exercise the pinned native factory, analytical ridge examples,
seed plumbing, all conventional configurations, numerical acceptance boundaries,
the limited binding control and the prescribed timing order.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import hadamard

import model43 as m


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_reference(deployment, coordinates):
    head = deployment["head"]
    linear = F(head["intercept"]) + sum(
        (F(coef) * F.from_float(float(value))
         for coef, value in zip(head["coefficients"], coordinates)), F(0))
    normalized = min(F(1), max(F(0), linear))
    return F(deployment["target_min"]) + F(deployment["span"]) * normalized


def run(output):
    start = time.perf_counter()
    groups = Counter()
    def check(ok, group):
        if not ok:
            raise AssertionError(group)
        groups[group] += 1
    def rejects(fn, group, exception=m.MethodError):
        try:
            fn()
        except exception:
            groups[group] += 1
        else:
            raise AssertionError("Expected rejection: " + group)

    provenance = json.loads((ROOT / 'analysis/inherited_method_provenance43.json').read_text())
    for row in provenance['files']:
        path = ROOT / row['path']
        check(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'inherited_source_identity')
    configs = m.affine_configs()
    check(len(configs) == 90 and len({x['candidate_id'] for x in configs}) == 90, 'fixed_affine_roster')
    check(Counter(x['family'] for x in configs) == {'raw': 10, 'nystrom': 20, 'kahm': 60}, 'fixed_family_counts')
    check({x['ridge_lambda'] for x in configs} == {'1/1000000', '1/10000', '1/100', '1', '100'}, 'fixed_penalties')
    for seed in m.SEEDS:
        rows = m.baseline_configs(seed)
        check(len(rows) == 44 and len({x['candidate_id'] for x in rows}) == 44, 'baseline_roster')
        for expected, row in zip(m.baseline_configs(3901), rows):
            adjusted = deepcopy(expected)
            if 'random_state' in adjusted['parameters']:
                adjusted['parameters']['random_state'] = seed
            check(adjusted == row, 'seed_substitution_only')
    rejects(lambda: m.baseline_configs(7), 'undeclared_seed_rejected')

    # An orthogonal analytical example distinguishes n*lambda from lambda.
    H = hadamard(64).astype(np.float64)[:, 1:25]
    y = [F(1, 2) + F(int(row[0]), 4) + F(int(row[1]), 8) for row in H]
    cfgs = [x for x in configs if x['map'] == 'raw_standard']
    exact_case = None
    projection_case = None
    for scale in (F(1), F(1, 16)):
        Z = H * float(scale)
        for cfg in cfgs:
            result = m.fit_affine(cfg, Z, y, {'target_min': '0', 'span': '1'},
                                  feature_id='artificial:orthogonal', map_sha256='a' * 64,
                                  map_bytes=123, training_binding={'origin': 'artificial_hadamard'})
            fitted, head = result['fit'], result['head']
            source = list(map(F, fitted['projection']['source_coefficients']))
            lam = F(cfg['ridge_lambda'])
            expected = [F(1, 4) * scale / (scale * scale + lam),
                        F(1, 8) * scale / (scale * scale + lam)] + [F(0)] * 22
            for value, target in zip(source, expected):
                check(abs(float(value - target)) <= 2e-15 * max(1, abs(float(target))), 'analytical_mean_loss_ridge')
            beta = list(map(F, head['coefficients']))
            check(sum((x*x for x in beta), F(0)) <= 4, 'projected_exact_norm')
            center = [sum((F.from_float(float(v)) for v in Z[:, j]), F(0)) / len(Z) for j in range(24)]
            check(F(head['intercept']) + sum((a*b for a,b in zip(beta,center)),F(0)) == sum(y,F(0))/len(y), 'exact_training_recentering')
            check(head['output_interval'] == ['0', '1'], 'fixed_range')
            if scale == 1 and cfg['lambda_token'] == '1e-6':
                exact_case = result
                check(not fitted['projection']['projected'], 'known_inactive_projection')
            if scale == F(1, 16) and cfg['lambda_token'] == '1e-6':
                projection_case = result
                check(fitted['projection']['projected'], 'known_active_projection')
    check(exact_case is not None and projection_case is not None, 'analytical_fixtures_retained')
    malformed = dict(cfgs[0], ridge_lambda='1/7')
    rejects(lambda: m.fit_affine(malformed,H,y,{'target_min':'0','span':'1'},feature_id='x',map_sha256='a'*64,map_bytes=123), 'off_grid_head_rejected')

    # Input-only native fits on reproducible artificial inputs at all three seeds.
    rng = np.random.default_rng(430343)
    X = rng.normal(size=(64,24))
    X += np.linspace(-.4,.4,len(X))[:,None] * np.linspace(.1,.8,24)[None,:]
    query = rng.normal(size=(3,24))
    yf = .5 + .18*np.tanh(X[:,0] + .25*X[:,1])
    synthetic_y = [F.from_float(float(v)) for v in yf]
    normal = {'target_min':'-3/2','span':'17/3'}
    first_features = {}
    changed_kahm = Counter()
    fitted_banks = {}
    head_count = 0
    checked_query_predictions = 0
    for seed in m.SEEDS:
        maps, meta = m.fit_representations(X, seed)
        fitted_banks[seed] = maps
        check(sorted(maps) == m.map_names(), 'native_map_roster')
        check(meta['seed'] == seed and meta['forecast_labels_used'] is False and meta['no_heldout_input_or_label_used'] is True, 'native_training_only_metadata')
        check(meta['training_shape'] == [64,24], 'native_artificial_shape')
        for name, mapping in maps.items():
            Z = m.transform_one(mapping,X)
            zq = np.vstack([m.transform_one(mapping,row[None,:])[0] for row in query])
            check(Z.shape[0] == 64 and np.isfinite(Z).all() and np.isfinite(zq).all(), 'native_feature_finiteness')
            wire = m.serialize_map(mapping)
            loaded = m.deserialize_map(wire)
            replay = m.transform_one(loaded, query[:1])[0]
            check(np.array_equal(replay.view(np.uint64),zq[0].view(np.uint64)), 'serialized_transform_replay')
            if seed == 3901:
                first_features[name] = (Z.copy(),zq.copy())
            elif mapping.family in ('raw','nystrom'):
                check(np.array_equal(Z.view(np.uint64),first_features[name][0].view(np.uint64)) and np.array_equal(zq.view(np.uint64),first_features[name][1].view(np.uint64)), 'deterministic_raw_nystrom_seed_replay')
            else:
                changed_kahm[seed] += not np.array_equal(zq.view(np.uint64),first_features[name][1].view(np.uint64))
            for cfg in [x for x in configs if x['map'] == name]:
                fitted = m.fit_affine(cfg,Z,synthetic_y,normal,
                                      feature_id=f'artificial:{seed}:{name}',map_sha256=hashlib.sha256(wire).hexdigest(),
                                      map_bytes=len(wire),training_binding={'origin':'deterministic_artificial_normal_fixture'})
                head_count += 1
                b = list(map(F,fitted['head']['coefficients']))
                check(sum((v*v for v in b),F(0)) <= 4, 'native_grid_head_norm')
                for i,z in enumerate(zq):
                    reported = m.predict_one(fitted['prepared'],z)
                    record = m.certify(fitted['prepared'],z,reported,f'artificial:q{i}')
                    reference = independent_reference(fitted['deployment'],z)
                    error = abs(F.from_float(reported)-reference)
                    check(F(record['exact_physical']) == reference and F(record['absolute_physical_error']) == error, 'independent_rational_reference')
                    check(record['accepted'] == (error <= F(normal['span'])/2**40), 'independent_budget_predicate')
                    check(record['binding_only']['accepted'], 'honest_binding_control')
                    checked_query_predictions += 1
        if seed != 3901:
            check(changed_kahm[seed] > 0, 'seed_changes_native_fixture')

    baseline_fits = 0
    payloads = {}
    for cfg in m.baseline_configs(3901):
        payload = m.fit_baseline(cfg,X,synthetic_y,normal)
        values = m.predict_baseline(payload,query)
        baseline_fits += 1
        payloads[cfg['candidate_id']] = payload
        check(values.shape == (3,) and np.isfinite(values).all(), 'baseline_fit_predict_contract')
        if cfg['kind'] == 'ridge':
            check(payload['model'].alpha == len(X)*float(F(cfg['ridge_lambda'])), 'baseline_mean_loss_penalty')
        if cfg['kind'] == 'persistence':
            check(np.array_equal(values,query[:,-1]), 'persistence_uses_last_observed_value')
        if cfg['kind'] == 'mean':
            target = float(F(normal['target_min'])+F(normal['span'])*sum(synthetic_y,F(0))/len(synthetic_y))
            check(np.array_equal(values,np.full(3,target)), 'training_mean_physical_inverse')
    for seed in (4301,4302):
        for kind in ('hgb','extra_trees','random_forest'):
            cfg = next(x for x in m.baseline_configs(seed) if x['kind'] == kind)
            payload = m.fit_baseline(cfg,X,synthetic_y,normal)
            baseline_fits += 1
            check(payload['model'].random_state == seed and np.isfinite(m.predict_baseline(payload,query)).all(), 'stochastic_baseline_seed_reaches_estimator')

    # Exact budget boundary, accepted/rejected float32 outcomes and controls.
    pp = exact_case['prepared']
    z = np.zeros(24,dtype=np.float64)
    check(independent_reference(exact_case['deployment'],z) == F(1,2), 'exact_half_reference_fixture')
    budget = F(1,2**40)
    for reported,expected in [(0.5,True),(float(F(1,2)+budget),True),
                              (math.nextafter(float(F(1,2)+budget),math.inf),False),
                              (float(F(1,2)-budget),True),
                              (math.nextafter(float(F(1,2)-budget),-math.inf),False)]:
        record = m.certify(pp,z,reported,'boundary')
        check(record['accepted'] == expected and record['binding_only']['accepted'], 'inclusive_budget_and_consistent_bad_packet')
    assay_same = m.prediction_assay(pp,z,m.predict_one(pp,z),'half')
    check(assay_same['cases']['physical_output_float32']['accepted'] and not assay_same['quantization_changed_reported_value'], 'valid_float32_value_remains_accepted')
    shifted = z.copy();shifted[0]=.1
    assay_changed = m.prediction_assay(pp,shifted,m.predict_one(pp,shifted),'shifted')
    changed = assay_changed['cases']['physical_output_float32']
    check(assay_changed['quantization_changed_reported_value'] and not changed['accepted'] and changed['binding_only']['accepted'], 'float32_numeric_rejection_beyond_binding')
    challenges = m.misbinding_assay(pp,z,0.5,'first',stale_features=shifted)
    check(len(challenges['challenges']) == 3 and all(x['status']=='REJECTED_AS_EXPECTED' for x in challenges['challenges']), 'three_misbinding_reasons')
    equal_challenge = m.misbinding_assay(pp,z,0.5,'first',stale_features=z.copy())
    check(equal_challenge['challenges'][-1]['status']=='NOT_APPLICABLE_EQUAL_FEATURE_BYTES', 'equal_stale_features_not_counted')
    bad_numeric = deepcopy(assay_same['cases']['binary64']['certificate'])
    bad_numeric['reference_interval_hex'] = [0.0.hex(),0.0.hex()]
    check(m.binding_only_check(pp,z,bad_numeric,query_id='half',reported=0.5)['accepted'], 'binding_control_explicitly_omits_arithmetic')
    rejects(lambda: m.cert.verify_certificate(pp,z,bad_numeric,expected_query_id='half',reported_prediction=0.5), 'full_checker_rejects_false_reference',m.cert.CertificateError)
    huge = m.fit_affine(cfgs[0],H,y,{'target_min':'0','span':str(10**40)},feature_id='artificial:overflow',map_sha256='b'*64,map_bytes=123)
    rejects(lambda: m.prediction_assay(huge['prepared'],z,m.predict_one(huge['prepared'],z),'overflow'), 'nonfinite_float32_stops_without_dropping')

    # Very small timing smoke test, separate from any scientific latency result.
    mapping = fitted_banks[3901]['raw_standard']
    Z = m.transform_one(mapping,X)
    wire = m.serialize_map(mapping)
    timing_head = m.fit_affine(cfgs[0],Z,synthetic_y,normal,feature_id='artificial:timing',map_sha256=hashlib.sha256(wire).hexdigest(),map_bytes=len(wire))
    timing_z = m.transform_one(mapping,query[:1])[0]
    value = m.predict_one(timing_head['prepared'],timing_z)
    first = m.benchmark_query(timing_head['prepared'],mapping,query[:1],timing_z,'timing',value,query_index=2,sample_ordinal=0,warmups=1,repeats=2)
    second = m.benchmark_query(timing_head['prepared'],mapping,query[:1],timing_z,'timing',value,query_index=4,sample_ordinal=1,warmups=1,repeats=2)
    check(first['primitive_order'][:2] == list(reversed(second['primitive_order'][:2])), 'timing_alternates_sample_ordinal_despite_original_parity')
    for record in (first,second):
        check(all(len(x)==2 and all(type(v)is int and v>=0 for v in x) for x in record['nanoseconds'].values()), 'timing_raw_repetitions_retained')
    static = m.benchmark_static_context(timing_head['deployment'],warmups=1,repeats=2)
    check(len(static['nanoseconds']) == 2, 'static_prepare_timing_smoke')
    baseline_timing = m.benchmark_baseline(payloads['baseline::ridge_1e6'],query[:1],warmups=1,repeats=2)
    check(len(baseline_timing['nanoseconds']) == 2, 'baseline_timing_smoke')

    record = {'schema':'step43-artificial-method-checks-v1','status':'PASS',
              'utc':datetime.now(timezone.utc).isoformat(),'checks':sum(groups.values()),
              'groups':dict(groups),'artificial_representation_banks':3,
              'artificial_maps':54,'artificial_grid_heads':head_count,
              'artificial_grid_query_predictions':checked_query_predictions,
              'artificial_baseline_fits':baseline_fits,
              'analytical_head_fits':11,'new_public_measurements_read':0,
              'real_model_fits':0,'scientific_latency_benchmark_performed':False,
              'test_source_sha256':sha(Path(__file__)),
              'model_source_sha256':sha(ROOT/'source/model43.py'),
              'inherited_provenance_sha256':sha(ROOT/'analysis/inherited_method_provenance43.json'),
              'design_sha256':sha(ROOT/'design43.json'),
              'wall_seconds':time.perf_counter()-start,
              'scope':'Artificial method/seed/serialization/analytical-ridge/numerical-assay/timing-order checks; no real cohort or controller execution'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(record,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/method_artificial_checks43.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Retain the existing artificial-check receipt')
    run(args.output)
