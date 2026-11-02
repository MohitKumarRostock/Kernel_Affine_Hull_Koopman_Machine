#!/usr/bin/env python3
"""Recompute the four primary comparisons from retained repetition losses.

This checks the statistical summaries conditional on the supplied losses. It
neither regenerates simulations nor authenticates the upstream fitted models.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import math
import statistics
from scipy.stats import t

CONTRASTS = (
    ('P_vs_G_nonlinear', 'nonlinear', 'P', 'G', 0.97),
    ('K_vs_P_nonlinear', 'nonlinear', 'K', 'P', 0.97),
    ('K_vs_R_nonlinear', 'nonlinear', 'K', 'R', 0.97),
    ('K_vs_P_linear', 'linear', 'K', 'P', 1.05),
)

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-12):
        raise AssertionError(f'{name}: {actual!r} != {expected!r}')

def verify(evidence: Path) -> dict:
    path = evidence / 'repetition_losses.csv'
    rows = read_csv(path)
    losses = {}
    for row in rows:
        key = (row['panel'], int(row['repetition']), row['forecast_key'])
        if key in losses:
            raise AssertionError(f'Duplicate loss key: {key}')
        value = float(row['smse'])
        if not math.isfinite(value) or value < 0:
            raise AssertionError(f'Invalid loss: {key}')
        losses[key] = value
    reference = {r['contrast_id']: r for r in read_csv(evidence/'primary_contrasts.csv')}
    paired = {(r['contrast_id'], int(r['repetition'])): r
              for r in read_csv(evidence/'primary_paired_repetitions.csv')}
    if set(reference) != {c[0] for c in CONTRASTS} or len(paired) != 128:
        raise AssertionError('Incomplete or altered primary family')
    critical = float(t.ppf(1 - 0.05 / 4, 31))
    results = []
    for name, panel, candidate, comparator, factor in CONTRASTS:
        a = [losses[panel, r, 'shared__'+candidate] for r in range(32)]
        b = [losses[panel, r, 'shared__'+comparator] for r in range(32)]
        gaps = [x-factor*y for x,y in zip(a,b)]
        for r,gap in enumerate(gaps):
            close(gap, float(paired[name,r]['margin_difference']), name+f': pair {r}')
        result = {
            'contrast_id': name, 'panel': panel, 'candidate': candidate,
            'comparator': comparator, 'comparator_factor': factor,
            'n_repetitions': 32, 'candidate_mean_smse': statistics.mean(a),
            'comparator_mean_smse': statistics.mean(b),
            'margin_difference_mean': statistics.mean(gaps),
            'margin_difference_sd': statistics.stdev(gaps), 'critical_t': critical,
            'point_relative_gain_percent': 100*(1-statistics.mean(a)/statistics.mean(b)),
        }
        result['simultaneous_one_sided_upper'] = result['margin_difference_mean'] + critical*result['margin_difference_sd']/math.sqrt(32)
        result['criterion_established'] = result['simultaneous_one_sided_upper'] < 0
        for key in ('candidate_mean_smse', 'comparator_mean_smse', 'margin_difference_mean',
                    'margin_difference_sd', 'critical_t', 'point_relative_gain_percent',
                    'simultaneous_one_sided_upper'):
            close(result[key], float(reference[name][key]), name+': '+key)
        expected = reference[name]['criterion_established'].lower() == 'true'
        if result['criterion_established'] != expected:
            raise AssertionError(f'{name}: inconsistent decision')
        results.append(result)
    panels = read_csv(evidence/'panel_summary.csv')
    for row in panels:
        values = [losses[row['panel'],r,row['forecast_key']] for r in range(32)]
        close(statistics.mean(values),float(row['mean_smse']), 'panel mean')
        close(statistics.stdev(values),float(row['repetition_sd']), 'panel SD')
    return {
        'verification': 'passed',
        'scope': 'Recomputation from retained repetition-level losses; no new model fit or simulation.',
        'input_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(evidence.glob('*.csv'))},
        'primary_family_size':4, 'paired_contrasts_checked':128,
        'panel_summaries_checked':len(panels),
        'criteria_established':sum(r['criterion_established'] for r in results),
        'primary_results':results,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'evidence')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=verify(args.evidence)
    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    print(text)

if __name__ == '__main__':
    main()
