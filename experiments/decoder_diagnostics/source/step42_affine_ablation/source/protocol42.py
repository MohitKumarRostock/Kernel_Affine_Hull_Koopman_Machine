"""Seal a retrospective controlled diagnostic before its new grid is executed."""
import importlib.metadata
import platform
from pathlib import Path
import common42 as c


def protocol():
    return {
        'schema': 'step42-controlled-head-ablation-protocol-v1', 'step': 42,
        'created_utc': c.utc(), 'status': 'DIAGNOSTIC_ALREADY_EXPOSED_PERIODS',
        'question': 'Which head restrictions and development selection choices explain the accuracy deficits in the inherited affine pipeline?',
        'tasks': list(c.TASKS), 'stages': list(c.STAGES), 'maps': c.maps(),
        'configurations': c.configurations(),
        'counts': {'maps': 18, 'lambdas': 5, 'factorial_recipes': 8,
                   'configurations_per_stage': 720, 'base_solves_per_stage': 90,
                   'stages_per_task': 4, 'tasks': 2, 'base_solves_total': 720,
                   'variant_payloads_total': 5760, 'selectors_per_task': 177},
        'data': {
            'source': 'Immutable Step41 training/validation/assessment features and exact labels',
            'new_outcome_information': False, 'new_domains': False,
            'new_assessment_periods': False, 'feature_maps_refitted': False,
            'training_features': 'Byte-identical stored training matrices',
            'evaluation_features': 'Byte-identical validation matrices; ordered concatenation of retained one-query assessment matrices',
            'normalization': 'Each original stage training-only exact target minimum and positive span',
            'no_training_on_evaluation_labels': True,
            'retrospective_design': 'Step41 outcomes and selected configurations were known when this diagnostic was designed',
        },
        'fitting': {
            'lambda_grid': [{'token': t, 'value': v} for t, v in c.LAMBDA_GRID],
            'base_solve': 'Same centered binary64 ridge arithmetic as Step41: solve(Zc.T@Zc+n*lambda*I, Zc.T@yc)',
            'base_intercept': 'binary64 mean(y)-mean(Z)@base_beta, interpreted as exact dyadic',
            'norm_projection': 'If enabled and exact base coefficient norm squared exceeds 4, contract by 2/upper_sqrt(norm_squared), using the inherited 80-bit outward square-root rule',
            'recenter_on': 'Exact training target mean minus transformed beta dot exact training feature means',
            'recenter_off': 'Retain original fitted binary64 intercept, including when projection changes coefficients',
            'clipping_on': 'Clip normalized output to [0,1]',
            'clipping_off': 'No output clipping; no bounded-output claim',
            'kappa': '2', 'projection_bits': 80,
            'recenter_interpretation': 'With inactive projection, the contrast isolates exact-vs-binary64 intercept recentering; with active projection it includes mean preservation after coefficient shrinkage',
        },
        'prediction': {
            'operational': 'One numpy.dot per query, same arithmetic order as Step41, optional clipping, then binary64 inverse target scaling',
            'reference': 'Retained exact rational head applied to exact stored binary64 features, optional exact clipping, exact inverse scaling',
            'metric': 'Exact physical-unit squared loss of actual reported binary64 predictions against exact inherited raw labels',
            'secondary_metrics': ['MAE', 'RMSE', 'exact_reference_MSE', 'maximum_numerical_discrepancy'],
            'all_candidate_predictions_retained': True,
        },
        'selection': {
            'folds': ['f1', 'f2', 'f3'],
            'criterion': 'Exact pooled physical SSE divided by total development observations',
            'specifications': c.selector_specs(),
            'main_comparisons': ['original41_replay', 'recipe_P1R1C1',
                                 'raw_recipe_P1R1C1', 'nystrom_recipe_P1R1C1',
                                 'kahm_recipe_P1R1C1'],
            'all_eight_global_recipe_selectors_reported': True,
            'new_ties': 'Lexical candidate ID; fixed before this diagnostic run',
            'original_replay_ties': 'Original Step41 f3 deployment byte count then candidate identity',
            'assessment_reselection': False,
            'assessment_oracles': 'Explicit hindsight diagnostics only; never substitutes for a selected predictor',
        },
        'diagnostics': {
            'paired_factor_contrasts': 'MSE(on)-MSE(off) at fixed map, lambda and other switches; exact rational differences; development and assessment separately',
            'factor_pairs_per_task_context': {'projection': 360, 'recenter': 360, 'clipping': 360},
            'projection_recenter_interaction': 'MSE(P1,R1)-MSE(P0,R1)-MSE(P1,R0)+MSE(P0,R0), fixed map/lambda/clipping',
            'projection_recenter_interactions_per_task_context': 180,
            'regularization_contrasts': 'Each of four nonlegacy lambdas minus lambda1e-6 at fixed map and switches',
            'regularization_pairs_per_task_context': 576,
            'activation': ['projection_active', 'clipping_count', 'recenter_delta', 'coefficient_squared_norm'],
            'clipping_count_meaning': 'Operational raw normalized value outside [0,1] and clipping enabled; exact-reference activation is retained separately',
            'target_range_diagnostic': 'Retain below/inside/above training-target-range counts per stage and exact clipping loss differences inside versus outside that range; exact clipping cannot increase squared loss for an inside-range target',
            'effect_summary': 'Exact equal/improve/worsen counts, median and mean normalized MSE differences using the same original selected ordinary reference MSE within each task/context; no independent-replication interpretation',
            'negligible_effect_display_threshold': 'Absolute paired MSE difference <= 1/10000000000 times the original selected conventional reference MSE; descriptive numerical threshold only',
            'inferential_p_values': False, 'population_success_claim': False,
        },
        'assurance': {
            'scope': 'Numerical agreement for specified fitted heads on supplied stored features; structural claims tracked per variant',
            'old_norm_class_retained_only_if': 'Exact coefficient norm squared <=4',
            'old_range_claim_retained_only_if': 'Clipping on',
            'training_optimizer_certified': False, 'native_feature_transform_certified': False,
            'future_label_coverage': False, 'generalization_guarantee': False,
            'no_automatic_reuse_of_step41_certificate_for_unbounded_variants': True,
            'certificate_bridge_selectors': ['original41_replay', 'recipe_P1R1C1',
                                             'recipe_P1R0C1', 'raw_recipe_P1R1C1',
                                             'nystrom_recipe_P1R1C1', 'kahm_recipe_P1R1C1'],
            'certificate_bridge': 'Convert eligible selected clipped heads with exact norm squared <=4 to the original Step41 schema and check every retained assessment query with the unchanged certificate41 producer/verifier; deduplicate candidate IDs',
            'certificate_budget': 'training_span * 2^-40',
            'certificate_rejection_policy': 'Retain a correctly rejected over-budget prediction as an empirical certificate-acceptance outcome; stop on an incorrect acceptance, incorrect rejection, malformed packet, or identity mismatch',
            'new_latency_claim': False,
        },
        'validation': {
            'inherited_branch': 'All 18 maps in all four stages recover original rational coefficients/intercepts and operational evaluation prediction bytes at lambda1e-6 P1R1C1',
            'original_selector': 'Original Step41 winner and exact pooled/assessment errors must be recovered',
            'independent_audit': 'Separate mathematical reconstruction of head transforms, exact query references, scores, selectors and contrasts; verification solve replays allowed and recorded',
        },
        'execution': {
            'threads': 1, 'producer_passes': 1, 'outcome_based_reruns': False,
            'failure_policy': 'Stop, preserve evidence, and report any blocked or failed gate; no silent modification of the frozen run',
            'stage_budget_seconds': 1800, 'selection_budget_seconds': 180,
            'aggregate_budget_seconds': 600,
            'verification_replay_is_distinct_from_a_new_scientific_grid_run': True,
        },
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {name: importlib.metadata.version(name)
                                     for name in ['numpy', 'scipy', 'scikit-learn', 'threadpoolctl']}},
        'prior_primary_status': 'Step41 remains NOT_ESTABLISHED on both periods',
        'next_claim_requirement': 'Any revised primary generalization claim requires a newly frozen method/selector and previously unused assessment',
    }


def seal():
    if (c.ROOT / 'protocol_seal42.json').exists():
        raise RuntimeError('Step42 protocol already frozen')
    c.save(c.ROOT / 'protocol42.json', protocol())
    files = [c.ROOT / 'protocol42.json']
    for folder in ('source', 'inputs', 'analysis'):
        files += [p for p in (c.ROOT / folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    receipt = {'schema': 'step42-protocol-source-seal-v1', 'sealed_utc': c.utc(),
               'data_already_exposed': True, 'new_grid_not_executed_before_seal': True,
               'bound_files': {c.relative(p): c.sha(p) for p in sorted(set(files))}}
    ref = c.save(c.ROOT / 'protocol_seal42.json', receipt)
    print(c.canonical_bytes({'status': 'SEALED', 'bound_file_count': len(receipt['bound_files']), **ref}).decode(), end='')


if __name__ == '__main__':
    seal()
