#!/usr/bin/env python3
"""Check arithmetic conditional on the released primary-summary statistics.

This does not read repetition-level losses, reconstruct forecasts, or refit models.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from scipy.stats import t

EXPECTED_SHA256 = '2e4c632c030fa69dc0b0f61c09f670633d5570b703c675d42d369e6ea4127818'
LAWS = ('linear_ar', 'nonlinear_ar', 'van_der_pol', 'duffing')
DECODERS = ('linear', 'nonlinear')


def close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-12):
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')

def main() -> None:
    path = Path(__file__).resolve().parent / 'primary_contrasts.csv'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise AssertionError('The released primary-summary bytes do not match their recorded identity.')
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
    if len(rows) != 40 or len({row['id'] for row in rows}) != 40:
        raise AssertionError('Expected exactly 40 distinct primary rows.')
    seen = set()
    critical = float(t.ppf(1 - 0.05 / 80, 63))
    counts = Counter()
    for row in rows:
        law, decoder = row['law'], row['decoder']
        if law not in LAWS or decoder not in DECODERS or row['view'] != 'independent' or int(row['n']) != 64:
            raise AssertionError('Unexpected law, decoder, selection rule, or sample size.')
        candidate, reference = row['candidate'], row['reference']
        allowed = {
            (f'K__local__{decoder}', f'K__mem__{decoder}'),
            (f'K__norm__{decoder}', f'K__mem__{decoder}'),
            (f'K__local__{decoder}', f'A__local__{decoder}'),
            (f'K__local__{decoder}', f'R__local__{decoder}'),
            (f'K__local__{decoder}', f'raw__raw__{decoder}'),
        }
        key = (law, decoder, candidate, reference)
        if (candidate, reference) not in allowed or key in seen:
            raise AssertionError(f'Unexpected or duplicated comparison: {key}')
        seen.add(key)
        mean = float(row['mean_difference']); sd = float(row['paired_sd'])
        if sd <= 0 or not math.isfinite(sd):
            raise AssertionError('This release has no zero-variance Student exception.')
        se = sd / 8; half = critical * se
        close(float(row['critical_t']), critical, 'critical value')
        close(float(row['standard_error']), se, 'standard error')
        close(float(row['halfwidth']), half, 'half-width')
        close(float(row['lower']), mean - half, 'lower endpoint')
        close(float(row['upper']), mean + half, 'upper endpoint')
        a = float(row['candidate_mean_smse']); b = float(row['reference_mean_smse'])
        if a < 0 or b <= 0:
            raise AssertionError('Invalid mean loss.')
        close(mean, a-b, 'difference of means')
        close(float(row['relative_gain_percent']), 100*(1-a/b), 'ratio of means')
        decision = ('candidate_lower_mean_loss' if mean+half < 0 else
                    'candidate_higher_mean_loss' if mean-half > 0 else 'unresolved')
        if row['classification'] != decision:
            raise AssertionError('Incorrect interval decision.')
        counts[decision] += 1
    expected = {'candidate_lower_mean_loss':16, 'candidate_higher_mean_loss':7, 'unresolved':17}
    if dict(counts) != expected:
        raise AssertionError('Unexpected complete-family decision counts.')
    print(json.dumps({'status':'passed', 'rows_checked':len(rows), 'decision_counts':dict(counts),
        'input_sha256':EXPECTED_SHA256,
        'scope':'Summary-statistic arithmetic only; no repetition-level reconstruction or model replay.'}, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
