#!/usr/bin/env python3
"""Compare manuscript display cells with the already independently audited CSVs.

This does not import the manuscript builder, fitting code, or analysis code;
it performs no new experiment or statistical analysis.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
import subprocess
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


LABELS = {
    'Raw ridge': 'ridge', 'Raw RBF': 'raw_rbf',
    'Boosted trees': 'raw_hgb', 'Persistence': 'persistence',
    'Common features': 'common', 'Bounded KAHM departure': 'kahm_departure',
    'Bounded RBF departure': 'rbf_departure',
    'Unbounded KAHM departure': 'kahm_unbounded',
    'Unbounded RBF departure': 'rbf_unbounded',
    'Selected non-KAHM': 'nonkahm_policy',
}
LAWS = ['nonlinear_ar', 'vanderpol', 'duffing', 'linear_ar']
LAW_LABELS = {'Nonlinear AR': 'nonlinear_ar', 'Van der Pol': 'vanderpol',
              'Duffing': 'duffing', 'Linear AR': 'linear_ar'}
CAPS = [1, 2, 4, 8, 12, 20]


def review(manuscript, study):
    manuscript, study = Path(manuscript), Path(study)
    displays = manuscript / 'generated/dimension_forecasting'
    summary = json.loads((study / 'analysis/analysis_summary.json').read_text())
    audit = json.loads((study / 'audit/post_release/final_results_audit.json').read_text())
    assert audit['status'] == 'passed' and summary['status'] == 'completed'
    inputs, verified = {}, {}
    cache = {}

    def data(name):
        if name not in cache:
            path = study / 'analysis' / (name + '.csv')
            digest = sha(path)
            assert summary['tables'][path.name]['sha256'] == digest, path
            inputs[path.name] = digest
            cache[name] = pd.read_csv(path)
        return cache[name]

    def one(name, **where):
        frame = data(name)
        for key, value in where.items():
            frame = frame.loc[frame[key] == value]
        assert len(frame) == 1, (name, where, len(frame))
        return frame.iloc[0]

    def rows(name):
        text = (displays / name).read_text()
        body = text.split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
        return [[v.strip() for v in line.strip().removesuffix(r'\\').split('&')]
                for line in body.splitlines() if line.strip()]

    def check(name, expected):
        actual = rows(name)
        assert actual == expected, {'file': name, 'actual': actual, 'expected': expected}
        verified[name] = {'sha256': sha(displays / name), 'rows': len(actual),
                          'cells': sum(map(len, actual))}

    def f(value, places=6):
        return f'{float(value):.{places}f}'

    def loss(method, view='context_selected', panel='nonlinear', mode='calibrated'):
        return one('panel_summary', method=method, view=view, panel=panel, mode=mode).smse

    check('aggregate_table.tex', [[label, f(loss(m)), f(loss(m, panel='linear'))]
                                 for label, m in LABELS.items()])
    primary_rows = []
    for label, m in [(k, v) for k, v in LABELS.items()
                     if v in ['common', 'rbf_departure', 'raw_rbf', 'raw_hgb', 'nonkahm_policy']]:
        r = one('primary_contrasts', comparator=m, panel='nonlinear')
        primary_rows.append((r.contrast_id, [label, f(r.candidate_mean_smse, 5),
            f(r.comparator_mean_smse, 5), f(r.point_relative_gain_percent, 2),
            f(r.margin_gap_mean), f(r.simultaneous_one_sided_lower),
            'Yes' if r.criterion_established else 'No']))
    order = ['common', 'rbf_departure', 'raw_rbf', 'raw_hgb', 'nonkahm_policy']
    primary_rows = [next(v for key, v in primary_rows if key == 'nonlinear_vs_' + m)
                    for m in order]
    r = one('primary_contrasts', comparator='ridge', panel='linear')
    primary_rows.append(['Raw ridge (linear NI)', f(r.candidate_mean_smse, 5),
        f(r.comparator_mean_smse, 5), f(r.point_relative_gain_percent, 2),
        f(r.margin_gap_mean), f(r.simultaneous_one_sided_lower),
        'Yes' if r.criterion_established else 'No'])
    check('primary_table.tex', primary_rows)
    check('law_noise_table.tex', [[label] + [f(one('condition_summary', law=law,
        noise_level=eta, view='context_selected', method=m, mode='calibrated').smse, 5)
        for law in LAWS for eta in (0, .1)] for label, m in LABELS.items()])

    variants = [('Fixed cap 2', 'context_historical2')] + [
        ('Native cap ' + str(c), 'native' + str(c)) for c in CAPS] + [
        ('B-selected cap', 'context_selected')]
    check('dimension_table.tex', [[label, f(loss('kahm_departure', view)),
        f(loss('common', view)), f(loss('rbf_departure', view)),
        f(100 * (1 - loss('kahm_departure', view) / loss('kahm_departure', 'context_historical2')), 2)]
        for label, view in variants])
    check('decomposition_table.tex', [[label] + [f(one('decomposition_summary',
        panel='nonlinear', view=view, mode='calibrated')[key]) for key in
        ['complete_change', 'common_change', 'departure_increment_change', 'new_departure_increment']]
        for label, view in [('KAHM-selected bank', 'matched'), ('Native cap 20', 'native20')]])

    geometries = [(label, 'historical2' if view == 'context_historical2' else view)
                  for label, view in variants[:-1]]
    expected = []
    for label, candidate in geometries:
        local = data('local_dimensions').query('candidate_id == @candidate')
        banks = data('geometry_summary_by_task').query('candidate_id == @candidate')
        ranks, width = local.effective_rank, banks.common_width
        assert len(local) == 2688 and len(banks) == 192
        assert (banks.stacked_basis_rank == 12).all()
        expected.append([label, f'{ranks.min()}--{ranks.max()}', f(ranks.mean(), 3),
            str(int((ranks == 12).sum())), f'{width.min()}--{width.max()}', '168'])
    check('geometry_table.tex', expected)

    cap_data = data('cap_selections')
    geom_methods = [m for m in LABELS.values() if m in set(cap_data.method)]
    expected = []
    for label, m in LABELS.items():
        if m not in geom_methods:
            continue
        for panel, flag in [('Nonlinear', False), ('Linear', True)]:
            sub = cap_data[(cap_data.method == m) & ((cap_data.law == 'linear_ar') == flag)]
            assert len(sub) == (48 if flag else 144)
            count = sub.selected_cap.value_counts()
            expected.append([label, panel] + [str(int(count.get(c, 0))) for c in CAPS])
    check('cap_table.tex', expected)

    expected = []
    resource_specs = [(label, 'selected', m) for label, m in LABELS.items()] + [
        ('KAHM, fixed cap 2', 'historical2', 'kahm_departure'),
        ('KAHM, native cap 20', 'native20', 'kahm_departure')]
    for label, context, method in resource_specs:
        row = one('resource_summary', context=context, method=method)
        expected.append([label, f(row.fit_and_selection_wall_seconds, 3),
            f(row.fit_and_selection_cpu_seconds, 3), f(row.deployment_pickle_bytes / 1024, 1),
            f(row.deployment_array_bytes / 1024, 1)])
    check('resource_table.tex', expected)

    expected = []
    for label, m in LABELS.items():
        if m not in ('common', 'kahm_departure', 'rbf_departure'):
            continue
        feature = data('feature_checks').query('method == @m')
        bound = data('bound_checks').query('method == @m')
        expected.append([label, f((feature.maximum_norm / feature.radius).max(), 4),
            f(bound.maximum_coefficient_bound_ratio.max(), 4),
            f(bound.maximum_D_correction_bound_ratio.max(), 4),
            str(int(feature.bound_violations.sum() + bound.coefficient_bound_violations.sum()
                    + bound.D_correction_bound_violations.sum()))])
    check('bounds_table.tex', expected)

    gamma = data('C_gamma_choices')
    selected_gamma = gamma.merge(cap_data[['task_id', 'method', 'selected_candidate']],
                                on=['task_id', 'method'], how='left')
    selected_gamma = selected_gamma[(selected_gamma.candidate_id == selected_gamma.selected_candidate)
                                   | (selected_gamma.candidate_id == 'shared')]
    assert not selected_gamma.duplicated(['task_id', 'method', 'horizon']).any()
    policy = data('C_policy_choices').query("context == 'selected'")
    activation = {}
    expected = []
    for label, m in list(LABELS.items())[1:]:
        if m == 'nonkahm_policy':
            sub = selected_gamma.merge(policy[['task_id', 'selected_method']], on='task_id')
            sub = sub[sub.method == sub.selected_method]
            assert not (policy.selected_method == 'ridge').any()
        else:
            sub = selected_gamma[selected_gamma.method == m]
        assert len(sub) == 576, (m, len(sub))
        nl = int(((sub.law != 'linear_ar') & (sub.gamma != 0)).sum())
        linear = int(((sub.law == 'linear_ar') & (sub.gamma != 0)).sum())
        counts = sub.gamma.value_counts()
        activation[m] = (nl, linear)
        expected.append([label, str(nl), str(linear)] + [str(int(counts.get(g, 0))) for g in (0, .25, .5, 1)])
    check('activation_table.tex', expected)
    policy_methods = ['ridge', 'raw_rbf', 'raw_hgb', 'persistence', 'common', 'rbf_departure', 'rbf_unbounded']
    expected = []
    for label, law in LAW_LABELS.items():
        for eta in (0., .1):
            sub = policy[(policy.law == law) & (policy.noise_level == eta)]
            assert len(sub) == 24
            counts = sub.selected_method.value_counts()
            expected.append([label, str(eta)] + [str(int(counts.get(m, 0))) for m in policy_methods])
    check('policy_table.tex', expected)

    specs = [('KAHM, fixed cap 2', 'context_historical2', 'kahm_departure'),
             ('KAHM, native cap 20', 'context_native20', 'kahm_departure')] + [
             (label, 'context_selected', m) for label, m in LABELS.items()]
    check('horizon_table.tex', [[label] + [f(one('panel_horizon_summary', panel='nonlinear',
        view=view, method=m, mode='calibrated', horizon=h).smse) for h in (1, 4, 12)]
        for label, view, m in specs])

    macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}', (displays / 'numbers.tex').read_text()))
    expected_macros = {}
    for macro, comparator in [('DFGainCommon', 'common'), ('DFGainRBF', 'rbf_departure'),
                             ('DFGainRawRBF', 'raw_rbf'), ('DFGainHGB', 'raw_hgb')]:
        expected_macros[macro] = f(100 * (1 - loss('kahm_departure') / loss(comparator)), 2)
    expected_macros.update({
        'DFWorsePolicy': f(100 * (loss('kahm_departure') / loss('nonkahm_policy') - 1), 2),
        'DFDimensionGain': f(100 * (1 - loss('kahm_departure') / loss('kahm_departure', 'context_historical2')), 2),
        'DFMaxErrorRatio': f(summary['maximum_selected_error_ratio'], 4),
        'DFKActiveNonlinear': str(activation['kahm_departure'][0]),
        'DFKActiveLinear': str(activation['kahm_departure'][1]),
        'DFKFitSeconds': f(one('resource_summary', context='selected', method='kahm_departure').fit_and_selection_wall_seconds, 3),
        'DFRFitSeconds': f(one('resource_summary', context='selected', method='rbf_departure').fit_and_selection_wall_seconds, 3),
        'DFPeakProcessMiB': f(data('task_resources').peak_rss_kib.max() / 1024, 1),
        'DFTargetCells': f'{192 * 1664 * 3:,}',
    })
    assert macros == expected_macros, (macros, expected_macros)
    verified['numbers.tex'] = {'sha256': sha(displays / 'numbers.tex'), 'macros': len(macros)}

    figure = displays / 'law_noise_contrasts.pdf'
    text = subprocess.check_output(['pdftotext', '-raw', str(figure), '-'], text=True)
    shown = re.findall(r'(-?\d+\.\d{2})%', text)
    wanted = []
    for comparator in ['rbf_departure', 'nonkahm_policy']:
        for law in LAWS:
            for eta in (0, .1):
                args = dict(law=law, noise_level=eta, view='context_selected', mode='calibrated')
                k = one('condition_summary', method='kahm_departure', **args).smse
                c = one('condition_summary', method=comparator, **args).smse
                wanted.append(f(100 * (1 - k/c), 2))
    assert Counter(shown) == Counter(wanted), (shown, wanted)
    verified[figure.name] = {'sha256': sha(figure), 'percentage_labels': len(wanted),
                            'scope': 'Numeric label multiset; visual positions reviewed separately.'}
    return {'status': 'passed', 'scope': 'Independent manuscript display-to-audited-table comparison; no new experiments.',
            'manuscript': str(manuscript), 'study': str(study), 'input_csv_sha256': inputs,
            'audit_sha256': sha(study / 'audit/post_release/final_results_audit.json'),
            'verified_displays': verified, 'table_count': 12,
            'table_rows': sum(x.get('rows', 0) for x in verified.values()),
            'table_cells_including_labels': sum(x.get('cells', 0) for x in verified.values()),
            'macro_count': len(macros), 'figure_labels': len(wanted)}


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
          'table_cells_including_labels', 'macro_count', 'figure_labels']}, indent=2))
