#!/usr/bin/env python3
"""Descriptive summaries and manuscript assets from every frozen-study row."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    folder = args.results
    config = json.loads((folder/'protocol.json').read_text())
    run = json.loads((folder/'run_summary.json').read_text())
    endpoint_rows = read_csv(folder/'endpoints.csv')
    decision_rows = read_csv(folder/'decisions.csv')
    methods = ['block_2', 'block_4', 'block_8', 'block_16', 'stratified_learned_coarse',
               'stratified_learned_fine', 'full_erm', 'fitted_upper']
    summary = []
    for regime in config['regimes']:
        for prefix in config['prefixes']:
            for n in config['calibration_sizes']:
                for method in methods:
                    rows = [r for r in endpoint_rows if (r['regime'], int(r['prefix']), int(r['n']), r['method']) == (regime, prefix, n, method)]
                    if not rows:
                        continue
                    values = np.array([float(r['upper'] if method == 'fitted_upper' else r['lower']) for r in rows])
                    score_means = [float(r['score_mean']) for r in rows if r['score_mean']]
                    summary.append({'regime': regime, 'prefix': prefix, 'cost': config['retained_costs'][prefix-1],
                        'n': n, 'method': method, 'banks': len(rows), 'mean_endpoint': mean(values),
                        'sd_endpoint': float(np.std(values, ddof=1)), 'minimum_endpoint': float(values.min()),
                        'maximum_endpoint': float(values.max()), 'mean_uncalibrated_score': mean(score_means) if score_means else '',
                        'maximum_qp_gap': max(float(r['maximum_gap']) for r in rows),
                        'unconverged_programs': sum(int(r['unconverged_programs']) for r in rows)})
    write_csv(folder/'summary.csv', summary)
    decisions = []
    for regime in config['regimes']:
        for n in config['calibration_sizes']:
            selected = [r for r in decision_rows if r['regime'] == regime and int(r['n']) == n]
            for method in dict.fromkeys(r['method'] for r in selected):
                rows = [r for r in selected if r['method'] == method]
                decisions.append({'regime': regime, 'n': n, 'method': method, 'banks': len(rows),
                    'upper_acceptances': sum(r['upper_accepted'] == 'True' for r in rows),
                    'minimum_cost_certificates': sum(r['minimum_cost_certified'] == 'True' for r in rows),
                    'joint_infeasibility_certificates': sum(r['all_classes_excluded'] == 'True' for r in rows),
                    'mean_lower_cost2': mean([float(r['lower_cost2']) for r in rows]),
                    'mean_lower_cost6': mean([float(r['lower_cost6']) for r in rows])})
    write_csv(folder/'decision_summary.csv', decisions)
    lookup = {(r['regime'], r['n'], r['method']): r for r in decisions}
    table = [r'\begin{table}[t]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{llrrrrrrr}', r'\toprule',
             r'Regime & Episodes & Upper & $b=2$ & $b=4$ & $b=8$ & $b=16$ & Strat. & ERM \\', r'\midrule']
    for ri, regime in enumerate(config['regimes']):
        for n in config['calibration_sizes']:
            key = 'minimum_cost_certificates' if regime == 'repairable' else 'joint_infeasibility_certificates'
            counts = []
            for method in ['block_2', 'block_4', 'block_8', 'block_16', 'stratified_pair', 'full_erm']:
                row = lookup.get((regime, n, method))
                counts.append('--' if row is None else f"{row[key]}/20")
            upper = lookup[regime, n, 'block_2']['upper_acceptances']
            table.append(f'{regime.capitalize()} & {n:,} & {upper}/20 & '+' & '.join(counts)+r' \\')
        if ri == 0:
            table.append(r'\midrule')
    table.extend([r'\bottomrule', r'\end{tabular}',
        r'\caption{Fresh calibration conditional on all 40 archived learned KAHM banks. Upper counts accepted fitted forecasts; lower-method columns count minimum cost six in the repairable regime and exclusion of both classes in the irreducible regime. All 1,760 endpoints share total failure probability $0.05$. Strat. takes the maximum of the two separately allocated learned-partition bounds. Full-sample ERM is evaluated only at $n=256$, on the same observations as the blocks; dashes mean uncomputed. Counts describe the fixed bank collection, not independent new training repetitions.}',
        r'\label{tab:learned-kahm-hierarchy}', r'\end{table}'])
    (folder/'learned_kahm_hierarchy.tex').write_text('\n'.join(table)+'\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.65), sharey=True)
    colors = {'block_2': '#566473', 'block_4': '#1f77b4', 'block_8': '#d28a16',
              'block_16': '#ab4a6e', 'stratified_pair': '#18856b', 'full_erm': '#111111'}
    display = {'block_2': 'Pairs', 'block_4': 'Block 4', 'block_8': 'Block 8',
               'block_16': 'Block 16', 'stratified_pair': 'Stratified pairs', 'full_erm': 'Full ERM'}
    for axis, regime, cost in zip(axes, config['regimes'], [2, 6]):
        for method in colors:
            data = [lookup[regime, n, method] for n in config['calibration_sizes'] if (regime, n, method) in lookup]
            x = [r['n'] for r in data]
            y = [r[f'mean_lower_cost{cost}'] for r in data]
            axis.plot(x, y, marker='D' if method == 'full_erm' else 'o',
                      linestyle='None' if method == 'full_erm' else '-', color=colors[method],
                      linewidth=1.6, markersize=5, label=display[method])
        axis.axhline(config['mse_tolerance'], color='#767676', linestyle='--', linewidth=1.)
        axis.set_xscale('log', base=4)
        axis.set_xticks(config['calibration_sizes'], [f'{n:,}' for n in config['calibration_sizes']])
        axis.set_xlabel('Independent calibration episodes')
        axis.set_title(f'{regime.capitalize()}: cost-{cost} class')
        axis.grid(axis='y', color='#e0e0e0', linewidth=.6)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_ylim(0., .46)
    axes[0].set_ylabel('Mean calibrated risk lower endpoint')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=[0., .10, 1., 1.])
    fig.savefig(folder/'learned_kahm_hierarchy.pdf', bbox_inches='tight')
    fig.savefig(folder/'learned_kahm_hierarchy.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    report = {'scope': config['scope'], 'source_run_sha256': hashlib.sha256((folder/'run_summary.json').read_bytes()).hexdigest(),
              'run_summary': run, 'decision_summary': decisions,
              'assets': {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
                         ['summary.csv', 'decision_summary.csv', 'learned_kahm_hierarchy.tex',
                          'learned_kahm_hierarchy.pdf', 'learned_kahm_hierarchy.png']}}
    (folder/'analysis_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'optimization': {k: run[k] for k in ['qp_count', 'unconverged_qp_count', 'maximum_normalized_gap', 'maximum_feasibility_residual']},
                      'decisions': decisions}, indent=2))


if __name__ == '__main__':
    main()
