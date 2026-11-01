#!/usr/bin/env python3
"""Verify new risk-manuscript displays against audited per-repetition records.

This is a presentation-data check, with no model fits or new experiments.
It does not import the table generator or either scientific implementation.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


LAWS = {'harmonic': 'Harmonic', 'quasiperiodic': 'Quasiperiodic',
        'nonlinear_ar': 'Nonlinear AR', 'logistic': 'Logistic',
        'vanderpol': 'Van der Pol', 'duffing': 'Duffing'}
ARMS = {'cap2': 'KAHM cap 2', 'native20': 'KAHM native 20',
        'selected_cap': 'KAHM selected cap', 'rbf': 'RBF',
        'affine': 'Affine hull', 'ridge_ar': 'Ridge AR',
        'polynomial': 'Quadratic direct', 'persistence': 'Persistence',
        'constant': 'Constant'}
KAHM = ['cap2', 'native20', 'selected_cap']


def review(manuscript, study):
    manuscript, study = Path(manuscript), Path(study)
    analysis = study / 'runs/20261101/analysis'
    audit_path = study / 'audit/post_release/final_results_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed'
    file_hashes = json.loads((analysis / 'analysis_files.json').read_text())
    provenance = json.loads((manuscript / 'dimension_study/provenance.json').read_text())
    for name, record in provenance['files'].items():
        original = study / record['source_relative_to_study_extension']
        copied = manuscript / 'dimension_study' / name
        assert sha(original) == sha(copied) == record['sha256'], name
    cache, inputs, verified = {}, {}, {}

    def data(name):
        if name not in cache:
            p = analysis / (name + '.csv')
            assert sha(p) == file_hashes[p.name]
            cache[name] = pd.read_csv(p)
            inputs[p.name] = sha(p)
        return cache[name]

    def table_rows(name):
        text = (manuscript / 'tables' / name).read_text()
        if r'\begin{longtable}' in text:
            body = text.split(r'\endfoot', 1)[1].split(r'\end{longtable}', 1)[0]
        else:
            body = text.split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
        return rows_from_text(body)

    def rows_from_text(text):
        # A starred TeX row break prevents a page break; it is not data.
        return [[x.strip() for x in re.sub(r'\\\\\*?\s*$', '', line.strip()).split('&')]
                for line in text.splitlines()
                if '&' in line and re.search(r'\\\\\*?\s*$', line)]

    expanded = (manuscript / 'supplement_sections/12_dimension_policy.tex').read_text()
    expanded_rows = Counter(map(tuple, rows_from_text(expanded)))

    def check(name, expected, in_supplement=True):
        actual = table_rows(name)
        assert actual == expected, {'file': name, 'actual': actual, 'expected': expected}
        if in_supplement:
            assert not (Counter(map(tuple, actual)) - expanded_rows), name + ' missing expanded rows'
        verified[name] = {'sha256': sha(manuscript / 'tables' / name),
                          'rows': len(actual), 'cells': sum(map(len, actual)),
                          'expanded_supplement_rows_checked': in_supplement}

    def f(value, p=6):
        return f'{float(value):.{p}f}'

    selected = data('selected_forecasts')
    endpoints = data('endpoints')
    comparisons = data('paired_contrasts')
    ranks = data('selected_bank_local_ranks')
    assert len(selected) == 234 and len(endpoints) == 1872 and len(comparisons) == 936
    means = selected.groupby(['system', 'arm']).test_mean_mse.mean()
    expected = []
    for law, label in LAWS.items():
        values = [means[law, arm] for arm in KAHM]
        expected.append([label] + [f(v) for v in values] +
                        [f'{100 * (values[2] / values[0] - 1):+.2f}'])
    check('dimension_policy_summary.tex', expected, in_supplement=False)

    expected = []
    for law, label in LAWS.items():
        for arm, arm_label in ARMS.items():
            sub = selected[(selected.system == law) & (selected.arm == arm)]
            assert len(sub) == (3 if law in ('vanderpol', 'duffing') else 5)
            expected.append([label if arm == 'cap2' else '', arm_label] +
                [f(sub['test_mse_h' + str(h)].mean(), 5) for h in (1, 4, 12)] +
                [f'{sub.test_mean_mse.mean():.5f} ({sub.test_mean_mse.std(ddof=1):.5f})'])
    check('dimension_policy_full_results.tex', expected)

    expected = []
    for law, label in LAWS.items():
        sub = selected[(selected.system == law) & (selected.arm == 'selected_cap')]
        expected.append([label] + [str(int((sub.cap == c).sum())) for c in (1, 2, 4)] +
                        [str(int((sub.history == h).sum())) for h in (4, 12)])
    check('dimension_policy_choices.tex', expected)
    check('dimension_policy_ranks.tex', [[ARMS[arm]] + [
        str(int(((ranks.arm == arm) & (ranks.effective_rank == rank)).sum()))
        for rank in (1, 2, 3, 4, 12)] for arm in KAHM])

    expected = []
    all_decisions = {}
    for law, label in LAWS.items():
        for arm in KAHM:
            ep = endpoints[(endpoints.system == law) & (endpoints.arm == arm)]
            values = []
            for tol in (.2, .35, .5):
                counts = Counter()
                for (job, alias), group in ep.groupby(['job', 'alias']):
                    accept = (group.mse_upper <= tol ** 2).all()
                    exclude = (group.mse_lower > tol ** 2).any()
                    assert not (accept and exclude)
                    result = 'accepted' if accept else 'excluded' if exclude else 'unresolved'
                    counts[result] += 1
                    all_decisions[job, alias, tol] = result
                values.append('/'.join(str(counts[k]) for k in ['accepted', 'excluded', 'unresolved']))
            expected.append([label if arm == 'cap2' else '', ARMS[arm]] + values +
                            [str(int((ep.mse_lower > 0).sum()))])
    check('dimension_policy_resolution.tex', expected)
    expected = []
    for law, label in LAWS.items():
        sub = comparisons[(comparisons.system == law) & (comparisons.type == 'retained_prefix')]
        expected.append([label] + [str(int(((sub.contrast == arm) & (sub.improvement_lower > 0)).sum()))
            for arm in KAHM] + [str(len(sub) // 3)])
    check('dimension_policy_improvements.tex', expected)

    old = selected[selected.arm == 'cap2'].set_index('job')
    new = selected[selected.arm == 'selected_cap'].set_index('job')
    delta = new.test_mean_mse - old.test_mean_mse
    k_selected = endpoints[endpoints.arm.isin(KAHM) & endpoints.selected_forecast]
    chosen_caps = new.cap.value_counts()
    prefix = comparisons[comparisons.type == 'retained_prefix']
    policy = comparisons[comparisons.type != 'retained_prefix']
    accepts = {}
    for tol in (.2, .35, .5):
        per_arm = []
        for arm in KAHM:
            ep = k_selected[k_selected.arm == arm]
            per_arm.append(sum((group.mse_upper <= tol ** 2).all() for _, group in ep.groupby('job')))
        assert len(set(per_arm)) == 1
        accepts[tol] = per_arm[0]
    expected_macros = {
        'DimRepetitions': len(old), 'DimDevelopmentFits': 70 * len(old),
        'DimEvaluatedModels': len(endpoints) // 3, 'DimRiskTails': 2 * len(endpoints),
        'DimImprovementTails': len(comparisons), 'DimTotalTails': 2 * len(endpoints) + len(comparisons),
        'DimBetterRuns': int((delta < 0).sum()), 'DimEqualRuns': int((delta == 0).sum()),
        'DimWorseRuns': int((delta > 0).sum()), 'DimCapOneCount': chosen_caps[1],
        'DimCapTwoCount': chosen_caps[2], 'DimCapFourCount': chosen_caps[4],
        'DimSelectedForecasts': len(selected[selected.arm.isin(KAHM)]),
        'DimSelectedLowerZero': int((k_selected.mse_lower == 0).sum()),
        'DimPositivePrefix': int((prefix.improvement_lower > 0).sum()),
        'DimPrefixTails': len(prefix), 'DimPositivePolicy': int((policy.improvement_lower > 0).sum()),
        'DimPolicyTails': len(policy), 'DimAllLowerZero': int((endpoints.mse_lower == 0).sum()),
        'DimAllLowerCount': len(endpoints), 'DimAcceptLow': accepts[.2],
        'DimAcceptMedium': accepts[.35], 'DimAcceptHigh': accepts[.5],
        'DimNativeDuffingIncrease': f(100 * (means['duffing', 'native20'] / means['duffing', 'cap2'] - 1), 2),
        'DimAuditTableRows': audit['counts']['export_rows_verified'],
        'DimAuditPredictionJoins': audit['counts']['full_prediction_loss_joins'],
    }
    expected_macros = {k: str(v) for k, v in expected_macros.items()}
    macro_file = manuscript / 'tables/dimension_policy_numbers.tex'
    actual_macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}', macro_file.read_text()))
    assert actual_macros == expected_macros, (actual_macros, expected_macros)
    assert (new.history == old.history).all() and (new.prefix_stage == old.prefix_stage).all()
    assert (selected[selected.arm.isin(KAHM)].n_features == 30).all()
    verified[macro_file.name] = {'sha256': sha(macro_file), 'macros': len(actual_macros)}
    return {'status': 'passed', 'scope': 'Manuscript cells checked against audited per-repetition outcomes and endpoints; no new experiments.',
            'audit_sha256': sha(audit_path), 'input_csv_sha256': inputs,
            'copied_input_files_exact': len(provenance['files']),
            'verified_displays': verified, 'table_count': 6,
            'table_rows': sum(v.get('rows', 0) for v in verified.values()),
            'table_cells_including_labels': sum(v.get('cells', 0) for v in verified.values()),
            'macro_count': len(actual_macros)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manuscript', type=Path)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.manuscript, args.study)
    result['verifier_sha256'] = sha(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'table_count', 'table_rows',
          'table_cells_including_labels', 'macro_count', 'copied_input_files_exact']}, indent=2))
