#!/usr/bin/env python3
"""Replay the frozen S43 numerical claims without fitting models or network I/O.

Binary64 features are trusted input coordinates, interpreted as exact dyadic
rationals.  This script independently recomputes each clipped affine reference
with Fraction, then invokes the original integer-arithmetic packet verifier.
It also recomputes selection and assessment losses from retained forecasts and
targets, and reaggregates the original timing repetitions (not a new benchmark).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'original/source'))
import certificate41 as verifier


def require(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


def load(name):
    with gzip.open(HERE / 'witnesses' / name, 'rt') as handle:
        return json.load(handle)


def binary(hexadecimal):
    value = float.fromhex(hexadecimal)
    require(math.isfinite(value), 'Nonfinite forecast or feature')
    return F.from_float(value)


def risk(predictions, targets):
    require(len(predictions) == len(targets) and predictions, 'Invalid loss roster')
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    sse = sum((error * error for error in errors), F(0))
    n = len(errors)
    return {'n': n, 'sse': str(sse), 'mse': str(sse / n),
            'mae': str(sum(map(abs, errors), F(0)) / n),
            'rmse': math.sqrt(float(sse / n))}


def compare_risk(actual, expected, label):
    for field in ('n', 'sse', 'mse', 'mae'):
        require(actual[field] == expected[field], f'{label}: {field} changed')
    require(math.isclose(actual['rmse'], expected['rmse'], rel_tol=1e-14, abs_tol=1e-14),
            f'{label}: RMSE changed')


def exact_reference(deployment, features):
    """Simple rational oracle, separate from the cached integer verifier."""
    head = deployment['head']
    require(len(features) == len(head['coefficients']), 'Feature dimension changed')
    value = F(head['intercept']) + sum(
        (F(weight) * coordinate for weight, coordinate in zip(head['coefficients'], features)), F(0))
    low, high = map(F, head['output_interval'])
    clipped = min(max(value, low), high)
    return F(deployment['target_min']) + F(deployment['span']) * clipped


def replay_development(data):
    result = {}
    checked_losses = 0
    checked_selections = 0
    for task, seeds in data['tasks'].items():
        result[task] = {}
        for seed, record in seeds.items():
            candidates = record['configurations']
            forecasts = defaultdict(list)
            targets = []
            ids = []
            for fold in record['folds'].values():
                ids += fold['ids']
                targets += list(map(F, fold['raw_targets']))
                require(set(fold['predictions_hex']) == set(candidates), 'Candidate roster changed')
                for candidate, values in fold['predictions_hex'].items():
                    forecasts[candidate] += list(map(binary, values))
            require(len(set(ids)) == len(ids), 'Development folds overlap')
            losses = {candidate: risk(values, targets) for candidate, values in forecasts.items()}
            for candidate, loss in losses.items():
                compare_risk(loss, record['expected_pooled_risks'][candidate], f'{task}/{seed}/{candidate}')
                checked_losses += 1
            selections = {}
            for policy in ('raw', 'nystrom', 'kahm', 'all_map', 'ordinary', 'ridge'):
                def eligible(candidate):
                    config = candidates[candidate]
                    if policy in ('raw', 'nystrom', 'kahm'):
                        return config.get('family') == policy
                    if policy == 'all_map':
                        return config['recipe'] == 'affine'
                    if policy == 'ordinary':
                        return config['recipe'] == 'baseline'
                    return config.get('baseline_kind') == 'ridge'
                pool = [candidate for candidate in candidates if eligible(candidate)]
                selected = min(pool, key=lambda candidate: (F(losses[candidate]['mse']), candidate))
                expected = record['expected_selections'][policy]
                require(selected == expected['candidate_id'], f'{task}/{seed}/{policy}: selected head changed')
                require(len(pool) == expected['pool_size'], 'Selection pool size changed')
                selections[policy] = {'candidate_id': selected, 'pool_size': len(pool), 'physical_risk': losses[selected]}
                checked_selections += 1
            result[task][seed] = selections
    return {'pooled_candidate_losses_recomputed': checked_losses,
            'policy_selections_recomputed': checked_selections, 'selections': result}


def replay_assessment(data, development):
    deployments = data['deployments']
    prepared = {key: verifier.prepare_deployment(value, expected_deployment_id=value['deployment_id'],
                                                expected_map_sha256=value['map_sha256'])
                for key, value in deployments.items()}
    counts = Counter()
    results = {}
    for task, record in data['tasks'].items():
        forecasts = defaultdict(list)
        exact_forecasts = defaultdict(list)
        monthly = defaultdict(lambda: defaultdict(list))
        monthly_targets = defaultdict(list)
        targets = []
        for query in record['queries']:
            target = F(query['raw_target'])
            targets.append(target)
            monthly_targets[query['month']].append(target)
            for evaluation_id, evaluation in query['affine_evaluations'].items():
                features = list(map(binary, evaluation['features_hex']))
                vector = np.array([float(value) for value in features], dtype=np.float64)
                deployment = deployments[evaluation_id]
                reference = exact_reference(deployment, features)
                budget = F(deployment['span']) * F(deployment['normalized_error_budget'])
                for case_name, case in evaluation['cases'].items():
                    reported = float.fromhex(case['reported_physical_hex'])
                    error = abs(F.from_float(reported) - reference)
                    mathematically_valid = error <= budget
                    packet = case['certificate']
                    lower, upper = map(binary, packet['reference_interval_hex'])
                    require(lower <= reference <= upper, 'Certificate misses exact affine reference')
                    require(binary(packet['absolute_error_upper_hex']) >= error, 'Invalid outward error bound')
                    try:
                        checked = verifier.verify_certificate(prepared[evaluation_id], vector, packet,
                            expected_query_id=query['query_id'], reported_prediction=reported)
                        accepted = True
                        require(F(checked['exact_reference_physical']) == reference,
                                'Independent rational and integer references disagree')
                        require(F(checked['absolute_error_exact']) == error, 'Verifier error mismatch')
                    except verifier.CertificateError:
                        accepted = False
                    require(accepted == mathematically_valid, 'Numerical verifier decision changed')
                    counts[f'{case_name}_accepted' if accepted else f'{case_name}_rejected'] += 1
                    if case_name == 'binary64':
                        require(accepted, 'Honest binary64 certificate rejected')
                    else:
                        honest = float.fromhex(evaluation['cases']['binary64']['reported_physical_hex'])
                        require(float(np.float32(honest)).hex() == reported.hex(), 'Float32 conversion changed')
                        counts['quantization_changed_value'] += (honest.hex() != reported.hex())
                for alias in evaluation['aliases']:
                    require(query['predictions'][alias]['evaluation_id'] == evaluation_id, 'Prediction alias changed')
                    exact_forecasts[alias].append(reference)
            for alias, prediction in query['predictions'].items():
                seed, policy = alias.split('::')
                selected = development['selections'][task][seed][policy]['candidate_id']
                require(prediction['candidate_id'] == selected, 'Assessment used a different selected candidate')
                forecast = binary(prediction['reported_physical_hex'])
                forecasts[alias].append(forecast)
                monthly[query['month']][alias].append(forecast)
                if prediction['evaluation_id'].startswith('affine::'):
                    packet_prediction = query['affine_evaluations'][prediction['evaluation_id']]['cases']['binary64']['reported_physical_hex']
                    require(prediction['reported_physical_hex'] == packet_prediction, 'Alias forecast differs from certificate')
            counts['assessment_queries'] += 1
        losses = {alias: risk(values, targets) for alias, values in forecasts.items()}
        for alias, loss in losses.items():
            compare_risk(loss, record['expected_policy_results'][alias]['physical_risk'], f'{task}/{alias}')
        exact_losses = {alias: risk(values, targets) for alias, values in exact_forecasts.items()}
        for alias, loss in exact_losses.items():
            compare_risk(loss, record['expected_policy_results'][alias]['exact_reference_risk'], f'{task}/{alias}/exact')
        month_losses = {}
        for month, policies in monthly.items():
            month_losses[month] = {alias: risk(values, monthly_targets[month]) for alias, values in policies.items()}
            for alias, loss in month_losses[month].items():
                compare_risk(loss, record['expected_monthly_results'][month][alias], f'{task}/{month}/{alias}')
        comparisons = {}
        for policy in ('ordinary', 'ridge'):
            raw = F(losses['3901::raw']['sse'])
            reference = F(losses['3901::'+policy]['sse'])
            passed = 20 * raw <= 21 * reference
            require(passed == record['expected_primary_comparisons'][policy]['within_five_percent_mse'],
                    'Primary five-percent MSE criterion changed')
            comparisons[policy] = {'within_five_percent_mse': passed,
                'mse_ratio': str(raw / reference), 'relative_mse_change_percent': 100 * float(raw / reference - 1)}
            counts['primary_comparisons_passed'] += passed
        # Replay all prescribed wrong-query, wrong-deployment, and stale-feature
        # packets against the external expected context of the first query.
        first = record['queries'][0]
        for evaluation_id, challenges in record['misbindings'].items():
            evaluation = first['affine_evaluations'][evaluation_id]
            vector = np.array([float.fromhex(x) for x in evaluation['features_hex']], dtype=np.float64)
            reported = float.fromhex(evaluation['cases']['binary64']['reported_physical_hex'])
            for challenge in challenges:
                try:
                    verifier.verify_certificate(prepared[evaluation_id], vector, challenge['certificate'],
                        expected_query_id=first['query_id'], reported_prediction=reported)
                except verifier.CertificateError:
                    counts['misbinding_rejected'] += 1
                    counts['misbinding_'+challenge['challenge']] += 1
                else:
                    raise AssertionError('Prescribed misbinding packet accepted')
        results[task] = {'n': len(targets), 'physical_risks': losses, 'exact_reference_risks': exact_losses,
                         'monthly_physical_risks': month_losses, 'primary_comparisons': comparisons}
    expected = {'assessment_queries': 270, 'binary64_accepted': 1350,
                'physical_output_float32_rejected': 1349, 'physical_output_float32_accepted': 1,
                'quantization_changed_value': 1350, 'misbinding_rejected': 45,
                'primary_comparisons_passed': 6}
    for key, value in expected.items():
        require(counts[key] == value, f'Coverage/outcome changed: {key}')
    require(len(prepared) == 15, 'Distinct deployment roster changed')
    return {'distinct_deployments': len(prepared), 'counts': dict(counts), 'tasks': results}


def replay_timings(data):
    """Summaries of actual retained observations; no runtime guarantee."""
    result = {}
    for task, record in data['tasks'].items():
        models = {}
        for evaluation_id, model in record['models'].items():
            components = defaultdict(list)
            for query in model['queries']:
                samples = query['nanoseconds']
                if isinstance(samples, list):
                    samples = {'selected_conventional_prediction': samples}
                for component, values in samples.items():
                    require(len(values) == record['configuration']['measured_repeats'], 'Timing repeat count changed')
                    require(all(type(value) is int and value >= 0 for value in values), 'Invalid timing sample')
                    components[component].append(statistics.median(values) / 1e6)
            summaries = {}
            for component, observations in components.items():
                summary = {'queries': len(observations), 'min_ms': min(observations),
                    'max_ms': max(observations), 'p50_ms': float(np.percentile(observations, 50)),
                    'p95_ms': float(np.percentile(observations, 95))}
                expected = model['expected_component_summary'][component]
                for key, value in summary.items():
                    require(math.isclose(value, expected[key], rel_tol=1e-13, abs_tol=1e-15), 'Timing reaggregation changed')
                summaries[component] = summary
            models[evaluation_id] = {'aliases': model['aliases'], 'components': summaries}
        primary = next(value for value in models.values() if '3901::raw' in value['aliases'])
        components = primary['components']
        result[task] = {'models': models,
            'raw_policy_certified_to_plain_p50_ratio': components['certified_prediction']['p50_ms'] / components['prediction_without_certificate']['p50_ms'],
            'raw_policy_cached_checker_to_direct_exact_p50_ratio': components['certificate_verification']['p50_ms'] / components['direct_exact_reference']['p50_ms']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON verification report outside this experiment directory')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.is_relative_to(HERE), 'Write replay results outside the retained experiment directory')
    started = time.perf_counter()
    development = replay_development(load('development.json.gz'))
    assessment = replay_assessment(load('assessment.json.gz'), development)
    timings = replay_timings(load('timing.json.gz'))
    report = {'schema': 'affine-forecasting-reviewer-replay-v1', 'status': 'PASS',
        'scope': {'trusted_stored_binary64_features': True, 'exact_affine_numerical_verification': True,
                  'development_selection_recomputed_from_retained_forecasts': True,
                  'assessment_losses_recomputed': True, 'original_timing_observations_reaggregated': True,
                  'fresh_training_run': False, 'new_timing_benchmark': False,
                  'ideal_native_features_certified': False, 'population_risk_claim': False,
                  'future_label_coverage_claim': False, 'cryptographic_authenticity_claim': False},
        'development': development, 'assessment': assessment, 'retained_timing_summaries': timings,
        'replay_seconds': time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'replay_seconds': report['replay_seconds'], 'counts': assessment['counts']}, sort_keys=True))


if __name__ == '__main__':
    main()
