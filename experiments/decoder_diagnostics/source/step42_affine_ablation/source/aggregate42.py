"""Predefined descriptive contrasts; no assessment-driven replacement of selectors."""
from fractions import Fraction as F
import math
import common42 as c
import prepare42 as d


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def effect_summary(rows, reference_mse):
    reference = F(reference_mse)
    if reference <= 0 or not rows:
        raise ValueError('Positive reference risk and nonempty contrast set required')
    differences = [F(row['mse_difference']) for row in rows]
    scaled = [x / reference for x in differences]
    threshold = F(1, 10**10)
    return {'pairs': len(rows), 'exact_improved': sum(x < 0 for x in differences),
            'exact_equal': sum(x == 0 for x in differences),
            'exact_worsened': sum(x > 0 for x in differences),
            'beyond_display_threshold_improved': sum(x < -threshold for x in scaled),
            'within_display_threshold': sum(abs(x) <= threshold for x in scaled),
            'beyond_display_threshold_worsened': sum(x > threshold for x in scaled),
            'mean_mse_difference': str(sum(differences, F(0)) / len(differences)),
            'median_mse_difference': str(median(differences)),
            'mean_reference_percent': 100 * float(sum(scaled, F(0)) / len(scaled)),
            'median_reference_percent': 100 * float(median(scaled)),
            'largest_absolute_reference_percent': 100 * float(max(map(abs, scaled))),
            'display_threshold_relative_to_reference_mse': str(threshold),
            'reference_mse': str(reference), 'independent_replications': False}


def candidate_key(row, **changes):
    values = {key: row[key] for key in ('map', 'lambda_token', 'project', 'recenter', 'clip')}
    values.update(changes)
    return (values['map'], values['lambda_token'], values['project'], values['recenter'], values['clip'])


def contrast_context(rows, reference_mse):
    lookup = {candidate_key(row): row for row in rows}
    factors = {}
    for factor in ('project', 'recenter', 'clip'):
        pairs = []
        for off in rows:
            if off[factor]:
                continue
            on = lookup[candidate_key(off, **{factor: True})]
            pairs.append({'factor': factor, 'map': off['map'], 'lambda_token': off['lambda_token'],
                          'off_candidate_id': off['candidate_id'], 'on_candidate_id': on['candidate_id'],
                          'other_switches': {k: off[k] for k in ('project', 'recenter', 'clip') if k != factor},
                          'mse_difference': str(F(on['physical_risk']['mse']) - F(off['physical_risk']['mse'])),
                          'exact_reference_mse_difference': str(F(on['exact_reference_risk']['mse']) - F(off['exact_reference_risk']['mse'])),
                          'projection_applied_in_pair': bool(on.get('projection_active', False) or off.get('projection_active', False)),
                          'projection_applied_fold_count': max(on.get('projection_active_fold_count', 0), off.get('projection_active_fold_count', 0))})
        active = [x for x in pairs if x['projection_applied_in_pair'] or x['projection_applied_fold_count']]
        inactive = [x for x in pairs if x not in active]
        factors[factor] = {'pairs': pairs, 'summary': effect_summary(pairs, reference_mse),
                           'projection_active_summary': effect_summary(active, reference_mse) if active else None,
                           'projection_inactive_summary': effect_summary(inactive, reference_mse) if inactive else None}
    interactions = []
    for row in rows:
        if row['project'] or row['recenter']:
            continue
        q00 = row
        q10 = lookup[candidate_key(row, project=True)]
        q01 = lookup[candidate_key(row, recenter=True)]
        q11 = lookup[candidate_key(row, project=True, recenter=True)]
        delta = (F(q11['physical_risk']['mse']) - F(q01['physical_risk']['mse'])
                 - F(q10['physical_risk']['mse']) + F(q00['physical_risk']['mse']))
        interactions.append({'map': row['map'], 'lambda_token': row['lambda_token'], 'clip': row['clip'],
                             'candidate_ids_P0R0_P1R0_P0R1_P1R1': [x['candidate_id'] for x in (q00, q10, q01, q11)],
                             'mse_difference': str(delta),
                             'definition': 'MSE11-MSE01-MSE10+MSE00'})
    regularization = []
    for row in rows:
        if row['lambda_token'] == '1e-6':
            continue
        original = lookup[candidate_key(row, lambda_token='1e-6')]
        regularization.append({'map': row['map'], 'lambda_token': row['lambda_token'],
                               'switches': {key: row[key] for key in ('project', 'recenter', 'clip')},
                               'reference_candidate_id': original['candidate_id'], 'candidate_id': row['candidate_id'],
                               'mse_difference': str(F(row['physical_risk']['mse']) - F(original['physical_risk']['mse']))})
    assert all(len(x['pairs']) == 360 for x in factors.values())
    assert len(interactions) == 180 and len(regularization) == 576
    return {'reference_mse': str(F(reference_mse)), 'factor_contrasts': factors,
            'projection_recenter_interactions': {'rows': interactions,
                                                 'summary': effect_summary(interactions, reference_mse)},
            'regularization_contrasts': {'rows': regularization,
                                         'summary': effect_summary(regularization, reference_mse)}}


def clipping_range_diagnostic(contexts, predictions, configs):
    """Use exact reference losses so arithmetic roundoff cannot break the theorem."""
    rows = []
    for cfg in configs:
        if cfg['clip']:
            continue
        off_id = cfg['candidate_id']
        on_id = off_id[:-1] + '1'
        sums = {'inside': F(0), 'outside': F(0)}
        counts = {'inside': 0, 'outside': 0}
        for stage, context in contexts.items():
            lower = F(context['normalization']['target_min'])
            upper = lower + F(context['normalization']['span'])
            off = predictions[stage][off_id]['exact_physical']
            on = predictions[stage][on_id]['exact_physical']
            for a, b, target in zip(off, on, context['evaluation']['raw_targets']):
                where = 'inside' if lower <= target <= upper else 'outside'
                delta = (F(b) - target)**2 - (F(a) - target)**2
                if where == 'inside' and delta > 0:
                    raise RuntimeError('Exact clipping worsened an in-range squared loss')
                sums[where] += delta; counts[where] += 1
        rows.append({'off_candidate_id': off_id, 'on_candidate_id': on_id,
                     'map': cfg['map'], 'lambda_token': cfg['lambda_token'],
                     'project': cfg['project'], 'recenter': cfg['recenter'],
                     'inside_n': counts['inside'], 'outside_n': counts['outside'],
                     'inside_exact_sse_difference': str(sums['inside']),
                     'outside_exact_sse_difference': str(sums['outside']),
                     'inside_clipping_never_increased_exact_squared_error': True})
    assert len(rows) == 360
    return rows


def target_range_counts(context):
    targets = context['evaluation']['raw_targets']
    lower = F(context['normalization']['target_min'])
    upper = lower + F(context['normalization']['span'])
    return {'n': len(targets), 'target_min': str(lower), 'target_max': str(upper),
            'below': sum(y < lower for y in targets),
            'inside': sum(lower <= y <= upper for y in targets),
            'above': sum(y > upper for y in targets)}


def aggregate():
    c.assert_frozen()
    protocol = c.read(c.ROOT / 'protocol42.json')
    prior = c.read(c.ROOT / 'inputs/prior41/output/result41.json')
    result = {'schema': 'step42-controlled-ablation-result-v1', 'step': 42,
              'study_status': 'DIAGNOSTIC_ALREADY_EXPOSED_PERIODS',
              'new_primary_generalization_claim': False, 'prior_step41_status': prior['primary_status'],
              'protocol_seal_sha256': c.sha(c.ROOT / 'protocol_seal42.json'),
              'input_manifest_sha256': c.sha(c.ROOT / 'inputs/manifest42.json'),
              'counts': protocol['counts'], 'tasks': {}, 'independent_global_audit_pending': True}
    for task in protocol['tasks']:
        if c.read(c.ROOT / 'audit' / task / 'bridge_gate42.json')['status'] != 'PASS':
            raise RuntimeError('Independent certificate bridge gate required')
        selection = c.read(c.ROOT / 'output' / task / 'selection42.json')
        final = c.read(c.ROOT / 'output' / task / 'final/scores42.json')
        final_lookup = {r['candidate_id']: r for r in final['rows']}
        development = []
        for row in selection['candidate_rows']:
            folds = row['folds']
            mae = sum((F(f['physical_risk']['mae']) * f['n'] for f in folds.values()), F(0)) / row['n']
            emae = sum((F(f['exact_reference_risk']['mae']) * f['n'] for f in folds.values()), F(0)) / row['n']
            development.append({**row,
                                'physical_risk': {'n': row['n'], 'sse': row['pooled_physical_sse'],
                                                  'mse': row['pooled_physical_mse'], 'rmse': math.sqrt(float(F(row['pooled_physical_mse']))), 'mae': str(mae)},
                                'exact_reference_risk': {'n': row['n'], 'sse': row['pooled_exact_reference_sse'],
                                                         'mse': row['pooled_exact_reference_mse'], 'rmse': math.sqrt(float(F(row['pooled_exact_reference_mse']))), 'mae': str(emae)},
                                'projection_active_fold_count': sum(f['projection_active'] for f in folds.values()),
                                'clipping_count': sum(f['clipping_count'] for f in folds.values())})
        contexts = {stage: d.load_stage(task, stage, root=c.ROOT) for stage in c.STAGES}
        prediction_rows = {stage: {r['candidate_id']: r for r in c.read(c.ROOT / 'output' / task / stage / 'predictions42.json')['rows']}
                           for stage in c.STAGES}
        old_selection = contexts['f3']['inherited_selection']
        old_test = prior['tasks'][task]
        ordinary_dev = old_selection['selections']['ordinary_rolling']['pooled_physical_mse']
        ordinary_test = old_test['selections']['ordinary_rolling']['physical_test_risk']['mse']
        contexts_to_use = {'development': {s: contexts[s] for s in ('f1', 'f2', 'f3')},
                           'assessment': {'final': contexts['final']}}
        contrasts = {'schema': 'step42-exact-descriptive-contrasts-v1', 'task': task,
                     'study_status': result['study_status'],
                     'development': contrast_context(development, ordinary_dev),
                     'assessment': contrast_context(final['rows'], ordinary_test)}
        for label, pieces in contexts_to_use.items():
            contrasts[label]['clipping_by_target_range'] = clipping_range_diagnostic(pieces, prediction_rows, protocol['configurations'])
        cref = c.save(c.ROOT / 'output' / task / 'contrasts42.json', contrasts)
        selected_results = {}
        for spec in protocol['selection']['specifications']:
            label = spec['label']; selected = selection['selections'][label]
            row = final_lookup[selected['candidate_id']]
            pool = [r for r in final['rows'] if all(r[key] == value for key, value in spec['fixed'].items())]
            hindsight = min(pool, key=lambda r: (F(r['physical_risk']['mse']), r['candidate_id']))
            risk = F(row['physical_risk']['mse']); reference = F(ordinary_test)
            dev_choice_mse = F(selected['pooled_physical_mse'])
            selected_results[label] = {
                'selection': selected, 'configuration': {key: row[key] for key in protocol['configurations'][0]},
                'assessment_risk': row['physical_risk'], 'exact_reference_assessment_risk': row['exact_reference_risk'],
                'assessment_mse_ratio_to_original_selected_ordinary': float(risk / reference),
                'assessment_mse_reduction_percent_vs_original_selected_ordinary': 100 * float(1 - risk / reference),
                'assessment_within_original_five_percent_tolerance_descriptively': 20 * risk <= 21 * reference,
                'assessment_strictly_lower_than_original_selected_ordinary': risk < reference,
                'projection_active': row['projection_active'], 'clipping_count': row['clipping_count'],
                'coefficient_squared_norm': row['coefficient_squared_norm'],
                'recenter_delta': row['recenter_delta'], 'structural_claims': row['structural_claims'],
                'maximum_absolute_numerical_error': row['max_abs_numerical_error'],
                'assessment_rank_of_selected_in_pool': 1 + sum(F(x['physical_risk']['mse']) < risk for x in pool),
                'hindsight_only': {'candidate_id': hindsight['candidate_id'], 'assessment_mse': hindsight['physical_risk']['mse'],
                                   'selection_regret_mse': str(risk - F(hindsight['physical_risk']['mse'])),
                                   'not_an_alternative_selected_predictor': True},
                'development_selected_mse': str(dev_choice_mse),
            }
        inherited = selected_results['original41_replay']
        if inherited['assessment_risk'] != old_test['selections']['affine_rolling']['physical_test_risk']:
            raise RuntimeError('Original full assessment scores not recovered')
        bridge = c.read(c.ROOT / 'output' / task / 'certificate_bridge/certificates42.json')
        bridge_summary = {'selectors': bridge['aliases'], 'unique_candidates': len(bridge['records']),
                          'packets': sum(len(x['rows']) for x in bridge['records'].values()),
                          'accepted_packets': sum(row['accepted'] for x in bridge['records'].values() for row in x['rows']),
                          'rejected_over_budget_packets': sum(not row['accepted'] for x in bridge['records'].values() for row in x['rows']),
                          'all_verified': bridge['all_query_certificates_verified_by_inherited_checker'],
                          'records_sha256': c.sha(c.ROOT / 'output' / task / 'certificate_bridge/certificates42.json')}
        result['tasks'][task] = {'assessment_n': final['n'], 'development_n': selection['development_n'],
                                'target_range_counts': {stage: target_range_counts(context) for stage, context in contexts.items()},
                                'selection_results': selected_results,
                                'prior_comparators': {label: old_test['selections'][label]
                                                      for label in ('ordinary_rolling', 'ridge_rolling', 'affine_rolling')},
                                'effect_summaries': {context: {factor: contrasts[context]['factor_contrasts'][factor]['summary']
                                                               for factor in ('project', 'recenter', 'clip')}
                                                     for context in ('development', 'assessment')},
                                'contrast_artifact': cref, 'certificate_bridge': bridge_summary,
                                'unique_final_head_ids': len({v['head_id'] for v in c.read(c.ROOT / 'output' / task / 'final/variants42.json')['variants'].values()}),
                                'all_720_assessment_configurations_retained': True}
    result['completed_utc'] = c.utc()
    ref = c.save(c.ROOT / 'output/result42.json', result)
    c.save(c.ROOT / 'output/aggregate_seal42.json',
           {'schema': 'step42-aggregate-seal-v1', 'protocol_seal_sha256': result['protocol_seal_sha256'],
            'bound_files': {ref['path']: ref['sha256'],
                            **{data['contrast_artifact']['path']: data['contrast_artifact']['sha256']
                               for data in result['tasks'].values()}}})
    print(c.canonical_bytes({'status': 'COMPLETE', 'study_status': result['study_status'], **ref}).decode(), end='')


if __name__ == '__main__':
    aggregate()
