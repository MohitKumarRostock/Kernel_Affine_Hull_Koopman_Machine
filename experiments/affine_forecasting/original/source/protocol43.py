"""Freeze complete code, source metadata, method and criteria before numeric access."""
import argparse
from importlib.metadata import version
import platform

import common43 as c
import model43 as m
import execute43


def build():
    root = c.ROOT
    if (root / 'protocol_seal43.json').exists():
        raise FileExistsError('Frozen study cannot be modified')
    if any((root / 'output').iterdir()):
        raise RuntimeError('Scientific output must be empty before protocol freeze')
    design = c.read_json(root / 'design43.json')
    design_seal = c.read_json(root / 'design_seal43.json')
    if design_seal['design_sha256'] != c.sha_file(root / 'design43.json'):
        raise RuntimeError('Preaccess design changed')
    manifest = c.read_json(root / 'inputs/data43/manifest43.json')
    if manifest['measurement_values_decoded_or_parsed'] != 0:
        raise RuntimeError('New numeric measurements were already examined')
    expected_counts = {
        'household_power2008': {'train': [151, 181, 212, 243], 'validation': [30, 31, 31], 'assessment': 90},
        'jena_temperature2018': {'train': [149, 179, 210, 241], 'validation': [30, 31, 31], 'assessment': 90},
        'metro_traffic2018': {'train': [137, 167, 197, 226], 'validation': [30, 30, 29], 'assessment': 90},
    }
    for task in c.TASKS:
        value = manifest['tasks'][task]
        actual = {'train': [len(value['stages'][s]['train']) for s in c.STAGES],
                  'validation': [len(value['stages'][s]['validation']) for s in c.STAGES[:-1]],
                  'assessment': len(value['assessment'])}
        if actual != expected_counts[task]:
            raise RuntimeError('Final timestamp-only cohort roster differs: ' + task)
    checks = ['analysis/method_artificial_checks43.json', 'analysis/data_artificial_checks43.json',
              'analysis/audit_artificial_checks43.json', 'analysis/producer_artificial_checks43.json',
              'analysis/cost_artificial_checks43.json']
    for name in checks:
        receipt = c.read_json(root / name)
        if receipt.get('status') != 'PASS':
            raise RuntimeError('Required artificial component check did not pass: ' + name)
    protocol = {
        'schema': 'step43-selected-policy-transfer-protocol-v1', 'step': 43,
        'status': 'PREDECLARED_FRESH_TASK_PERIOD_ASSESSMENT', 'created_utc': c.utc_now(),
        'design': design, 'design_seal': c.file_ref(root / 'design_seal43.json'),
        'tasks': list(c.TASKS), 'stages': list(c.STAGES), 'seeds': list(c.SEEDS),
        'policies': list(c.POLICIES), 'primary_seed': 3901, 'primary_policy': 'raw',
        'data_manifest': c.file_ref(root / 'inputs/data43/manifest43.json'),
        'cohort_counts': expected_counts,
        'maps': m.map_names(), 'affine_configurations': m.affine_configs(),
        'baseline_configurations': {str(seed): m.baseline_configs(seed) for seed in c.SEEDS},
        'representations': m.rep.DEFAULT_CONFIG,
        'selection': {'loss': 'exact physical-unit MSE of reported binary64 predictions',
                      'pooling': 'sum fold SSE divided by sum fold n',
                      'tie_rule': 'lexical candidate_id',
                      'pool_sizes': {'raw': 10, 'nystrom': 20, 'kahm': 60, 'all_map': 90, 'ordinary': 44, 'ridge': 5}},
        'prediction': {'operation': 'one query at a time for maps, affine np.dot and conventional prediction',
                       'assessment_labels_after_all_seed_policy_predictions': True,
                       'exact_targets': 'Original decimal tokens interpreted as rational values',
                       'no_full_assessment_grid': True},
        'certificate': {'producer_and_verifier': 'unchanged certificate41.py',
                        'normalized_budget': '1/1099511627776',
                        'reference': 'exact fitted clipped affine model on supplied stored binary64 features',
                        'primary_requires_all_raw_seed3901_packets_accepted': True,
                        'deduplication': 'exact deployment identity across selected seed/policy aliases',
                        'float32_condition': 'Round physical output through float32 then back to binary64; self-consistent packet',
                        'misbinding_query_index': 0, 'stale_feature_query_index': 1,
                        'misbinding_after_complete_assessment': True},
        'timing': {'primary_seed': 3901, 'queries_per_task': 32,
                   'index_rule': 'floor((2*j+1)*n/(2*m)), j=0,...,m-1; m=min(32,n)',
                   'warmups': 3, 'repeats': 10, 'static_repeats': 100,
                   'counterbalance': 'full verified/unverified order by selected sample ordinal parity',
                   'declared_primitives': design['cost_assay']['primitives'],
                   'measure_after_all_assessments_and_before_final_audit': True,
                   'no_speedup_success_threshold': True},
        'counts': {'development_seed_banks': 27, 'final_seed_banks': 9,
                   'development_affine_fits': 2430, 'final_selected_affine_fits': 27,
                   'development_conventional_candidates': 1188,
                   'final_selected_conventional_fits_range': [9, 18],
                   'assessment_queries': 270, 'assessment_policy_alias_predictions': 4860,
                   'development_candidate_query_predictions': 109746},
        'execution': {'producer_grid_runs': 1, 'outcome_based_reruns': 0, 'numerical_threads': 1,
                      'python_hash_seed': 4303, 'command_budget_seconds': 3600,
                      'phase_budget_seconds': {'stage': 1800, 'select': 180, 'seal-selection': 180,
                                               'assess': 1800, 'misbinding': 300, 'aggregate': 600},
                      'commands': [{'label': label, 'arguments': args} for label, args in execute43.command_plan()],
                      'failure_policy': 'Stop, retain evidence, and report; no silent postfreeze edits or reruns'},
        'artificial_checks': [c.file_ref(root / name) for name in checks],
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}},
        'claim_scope': 'Finite-cohort tolerance on three known-domain periods, one historical and two forward; no statistical noninferiority or broad AI superiority claim',
        'prior_step41_primary_status_unchanged': 'NOT_ESTABLISHED',
        'prior_step42_status_unchanged': 'DIAGNOSTIC_ALREADY_EXPOSED_PERIODS',
    }
    c.write_json(root / 'protocol43.json', protocol)
    return protocol


def freeze():
    root = c.ROOT
    build()
    files = [root / name for name in ('design43.json', 'design_seal43.json', 'protocol43.json')]
    for directory in ('source', 'inputs', 'analysis'):
        files.extend(path for path in (root / directory).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
    c.seal_files(root / 'protocol_seal43.json', files, schema='step43-protocol-source-seal-v1',
                 frozen_before_new_numeric_measurement_access=True,
                 public_measurement_values_decoded_or_parsed=0, scientific_fits=0,
                 assessment_outcomes_examined=0, design_seal_sha256=c.sha_file(root / 'design_seal43.json'))
    print(c.canonical_bytes({'status': 'FROZEN', 'bound_files': len(set(files)),
                             'protocol_seal_sha256': c.sha_file(root / 'protocol_seal43.json')}).decode(), end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze'])
    parser.parse_args()
    freeze()
