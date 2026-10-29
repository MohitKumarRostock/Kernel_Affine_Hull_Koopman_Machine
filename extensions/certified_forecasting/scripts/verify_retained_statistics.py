#!/usr/bin/env python3
"""Independently check the original empirical tables and retained scalar records.

The verifier uses only the Python standard library and does not import the
experimental runners, plotting scripts, or certificate implementation. It
reconstructs table selection/aggregation, the two-atom minimum risk, PenDigits
writer aggregation, and confidence arithmetic from retained sufficient
statistics. It does not refit models or recompute synthetic episode moments
from the original NPZ arrays.

Example, from the extension directory::

    python scripts/verify_retained_statistics.py         --output verification/retained_statistics.json

The output must be a new file outside the archived results directory.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import hashlib
import csv
import gzip
import json
import math
import statistics as st
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Extension root containing results/ and paper/')
parser.add_argument('--paper', type=Path, help='Optional manuscript directory; defaults to <root>/paper')
parser.add_argument('--output', type=Path, required=True, help='New JSON output outside <root>/results')
args = parser.parse_args()
EXT = args.root.resolve()
RES = EXT / 'results'
TABLE = (args.paper.resolve() if args.paper else EXT / 'paper') / 'tables'
output = args.output.resolve()
if output.exists():
    parser.error('The output already exists; choose a new verification file.')
if output == RES or RES in output.parents:
    parser.error('Verification output must be outside the archived results directory.')
checks = []
errors = []

def chk(ok, msg):
    (checks if ok else errors).append(msg)

def close(a, b, tol=2e-13):
    return abs(a - b) <= tol

def readruns(name):
    return [json.loads(p.read_text()) for p in sorted((RES / name).glob('*/results.json'))]

def choose(rs, fam, tie=True):
    cs = [c for c in rs if c['family'] == fam]
    return min(cs, key=(lambda c: (st.fmean(c['validation_mse']), c['n_features'], c['name'])) if tie else lambda c: st.fmean(c['validation_mse']))

def table_rows(name):
    rows = []
    for line in (TABLE / name).read_text().splitlines():
        if '&' in line and (not line.lstrip().startswith('\\')):
            rows.append([p.strip().rstrip('\\').strip() for p in line.split('&')])
    return rows
MAIN = readruns('main_v1')
OSC = readruns('oscillators_v1')
SENS = {k: readruns(f'sensitivity_v1_k{k}') for k in [1, 4, 16]}
chk(len(MAIN) == 20 and sum((len(r['results']) for r in MAIN)) == 800, 'Main campaign: 20 runs, 800 candidate records')
chk(len(OSC) == 6 and sum((len(r['results']) for r in OSC)) == 240, 'Oscillators: 6 runs, 240 candidate records')
chk(sum(map(len, SENS.values())) == 36 and sum((len(r['results']) for rs in SENS.values() for r in rs)) == 756, 'Sensitivity: 36 runs, 756 candidate records')
label = {'harmonic': 'Harmonic', 'quasiperiodic': 'Quasiperiodic', 'nonlinear_ar': 'Nonlinear AR', 'logistic': 'Logistic', 'vanderpol': 'Van der Pol', 'duffing': 'Duffing'}
fams = {'KAHM blocks': 'kahm_nested', 'RBF blocks': 'rbf_nested', 'Affine-hull blocks': 'affine_nested', 'Hard KAHM states': 'kahm_hard', 'Ridge AR': 'ridge_ar', 'Quadratic regression': 'polynomial', 'KAHM rollout': 'kahm_koopman', 'Polynomial rollout': 'polynomial_koopman', 'Persistence': 'persistence'}
for row in table_rows('forecast_results.tex'):
    if row[0] not in fams:
        continue
    vals = []
    for sys in ['harmonic', 'quasiperiodic', 'nonlinear_ar', 'logistic']:
        selected = [choose(r['results'], fams[row[0]], False) for r in MAIN if r['system'] == sys]
        vals += [f"{st.fmean((math.sqrt(c['test_mse'][h]) for c in selected)):.3f}" for h in [0, 2]]
    chk(row[1:] == vals, f'Forecast table: {row[0]}')
for row in table_rows('certified_selection.tex'):
    sys = next((s for s, l in label.items() if l == row[0]), None)
    if sys not in ['harmonic', 'quasiperiodic', 'nonlinear_ar', 'logistic']:
        continue
    vals = []
    for family in ['kahm_nested', 'all']:
        for eps in [0.2, 0.35, 0.5]:
            passed = []
            for run in MAIN:
                if run['system'] != sys:
                    continue
                eligible = [c for c in run['results'] if family == 'all' or c['family'] == family]
                eligible = [c for c in eligible if all((z['mse_upper'] <= eps ** 2 for z in c['certificates']))]
                if eligible:
                    passed.append(min(eligible, key=lambda c: (c['n_features'], c['history'], st.fmean(c['validation_mse'])))['n_features'])
            med = str(int(st.median(passed))) if passed else '--'
            vals.append(f'{len(passed)}/5 ({med})')
    chk(row[1:] == vals, f'Certified-selection table: {sys}')
for row in table_rows('improvements.tex'):
    sys = next((s for s, l in label.items() if l == row[0]), None)
    if sys not in ['harmonic', 'quasiperiodic', 'nonlinear_ar', 'logistic']:
        continue
    vals = []
    for h in [1, 4, 12]:
        gs = [g for r in MAIN if r['system'] == sys for g in r['paired_improvements'] if g['horizon'] == h]
        vals.append(f"{sum((g['improvement_lower'] > 0 for g in gs))}/{len(gs)}")
    chk(row[1:] == vals, f'Improvement table: {sys}')
osc_fam = {**fams, 'Affine blocks': 'affine_nested', 'Quadratic direct': 'polynomial', 'Quadratic rollout': 'polynomial_koopman'}
for row in table_rows('oscillators_table.tex'):
    sys = next((s for s, l in label.items() if l == row[0]), None)
    if sys not in ['vanderpol', 'duffing']:
        continue
    cs = [choose(r['results'], osc_fam[row[1]]) for r in OSC if r['system'] == sys]
    vals = []
    for h in range(3):
        rr = [math.sqrt(c['test_mse'][h]) for c in cs]
        vals.append(f'${st.fmean(rr):.3f}\\pm{st.stdev(rr):.3f}$')
    vals.append(f"{st.fmean((c['certificates'][2]['rmse_upper'] for c in cs)):.3f}")
    chk(row[2:] == vals, f'Oscillator table: {sys}/{row[1]}')
for row in table_rows('sensitivity_results.tex'):
    sys = next((s for s, l in label.items() if l == row[0]), None)
    if sys not in ['harmonic', 'quasiperiodic', 'nonlinear_ar', 'logistic']:
        continue
    k = int(row[1])
    cs = [choose(r['results'], 'kahm_nested') for r in SENS[k] if r['system'] == sys]
    vals = [f"{st.fmean((c['n_features'] for c in cs)):.1f}"]
    vals += [f"{st.fmean((math.sqrt(c['test_mse'][h]) for c in cs)):.4f}" for h in range(3)]
    vals.append(f"{sum((all((z['rmse_upper'] <= 0.35 for z in c['certificates'])) for c in cs))}/3")
    chk(row[2:] == vals, f'Sensitivity table: {sys}/kappa={k}')
# Confidence arithmetic uses the archived sufficient statistics.
allruns = MAIN + OSC + [r for rs in SENS.values() for r in rs]
maxbounddiff = 0.0
ncert = 0
npair = 0
maxgap = 0.0
for run in allruns:
    for c in run['results']:
        maxgap = max(maxgap, c.get('optimization_gap', 0))
        for z in c['certificates']:
            ncert += 1
            for sign, key, outkey in [(1, 'upper_details', 'mse_upper'), (-1, 'lower_details', 'mse_lower')]:
                d = z[key]
                n = d['n_independent']
                b = d['bound']
                alpha = d['delta_per_tail']
                radius = math.sqrt(2 * d['variance'] * math.log(2 / alpha) / n) + 7 * b * math.log(2 / alpha) / (3 * (n - 1))
                expected = min(b, d['mean'] + radius) if sign == 1 else max(0, d['mean'] - radius)
                maxbounddiff = max(maxbounddiff, abs(radius - d['radius']), abs(expected - z[outkey]))
                chk(close(radius, d['radius']) and close(expected, z[outkey]), f"Bound moments {run['system']}/{run['seed']}/{c['name']}/{ncert}/{key}")
    for g in run['paired_improvements']:
        npair += 1
        n = g['n_independent']
        b = g['bound']
        alpha = g['delta_per_tail']
        radius = math.sqrt(2 * g['variance'] * math.log(2 / alpha) / n) + 7 * b * math.log(2 / alpha) / (3 * (n - 1))
        expected = max(0, g['mean'] - radius) - 4
        maxbounddiff = max(maxbounddiff, abs(expected - g['improvement_lower']))
        chk(close(expected, g['improvement_lower']), f"Paired-bound moments {run['system']}/{run['seed']}/{npair}")
chk(maxgap < 1e-07, 'All recorded optimization gaps below 1e-7')
for system in ['harmonic', 'quasiperiodic', 'nonlinear_ar', 'logistic']:
    for seed in [2701, 2702, 2703]:
        ss = [next((r for r in SENS[k] if r['system'] == system and r['seed'] == seed))['streams'] for k in SENS]
        chk(ss[0] == ss[1] == ss[2], f'Sensitivity stream reuse {system}/{seed}')
# PenDigits is reaggregated from individual writer-loss records.
P = json.loads((RES / 'pendigits_v1b/results.json').read_text())
PS = json.loads((RES / 'pendigits_v1b/summary.json').read_text())
FC = json.loads((RES / 'pendigits_v1b/frozen_candidates.json').read_text())
with (RES / 'pendigits_v1b/writer_losses.csv').open() as f:
    writer_rows = list(csv.DictReader(f))
groups = defaultdict(list)
for w in writer_rows:
    groups[int(w['seed']), int(w['coordinate']), w['candidate'], int(w['horizon'])].append(float(w['writer_mean_squared_loss']))
maxwriterdiff = 0
for c in P['results']:
    for hidx, h in enumerate([1, 2, 4]):
        values = groups[c['seed'], c['coordinate'], c['name'], h]
        chk(len(values) == 13, 'PenDigits writer aggregation count')
        diff = abs(st.fmean(values) - c['cohort_writer_mean_mse'][hidx])
        maxwriterdiff = max(maxwriterdiff, diff)
        chk(diff < 2e-16, 'PenDigits independent writer-loss reaggregation')
        expbound = min(4, c['cohort_writer_mean_mse'][hidx] + 4 * math.sqrt(math.log(720 / 0.05) / (2 * 13)))
        chk(close(expbound, c['conditional_iid_writer_mse_upper'][hidx]), 'PenDigits Hoeffding arithmetic')
for frozen in FC:
    chk(frozen['n_training_strokes'] == 480 and frozen['n_validation_strokes'] == 717, 'PenDigits development sample counts')
    for family, name in frozen['selected_by_family'].items():
        c = choose(frozen['candidates'], family)
        chk(c['name'] == name, 'PenDigits validation-only model choice')
for row in table_rows('pendigits_results.tex'):
    summary = next((s for s in PS['families'] if s['label'] == row[0]), None)
    if summary is None:
        continue
    family = summary['family']
    cs = [c for c in P['results'] if c['family'] == family and c['selected_before_test']]
    vals = [f"{st.fmean((c['n_features'] for c in cs)):.1f}"]
    for h in range(3):
        pooled = [math.sqrt(st.fmean((c['cohort_writer_mean_mse'][h] for c in cs if c['seed'] == seed))) for seed in [1701, 1702, 1703]]
        vals.append(f'${st.fmean(pooled):.4f}\\pm{st.stdev(pooled):.4f}$')
        chk(close(st.fmean(pooled), summary['mean_coordinate_pooled_rmse_over_seeds'][h]), f'PenDigits summary mean/{family}/{h}')
        chk(close(st.stdev(pooled), summary['sd_coordinate_pooled_rmse_over_seeds'][h]), f'PenDigits summary SD/{family}/{h}')
    chk(row[1:] == vals, f'PenDigits table: {family}')
chk(len(P['results']) == 240 and len(writer_rows) == 9360, 'PenDigits retained candidate/writer-row counts')
writer_counts = {int(w['writer']): int(w['n_eligible_strokes']) for w in writer_rows}
chk(len(writer_counts) == 13 and sum(writer_counts.values()) == 1751 and (5 not in writer_counts), 'PenDigits eligible writers/digits and writer 5 exclusion')
chk(P['origins'] == [7, 11, 15, 19, 23, 27] and len(P['held_out_record_ids']) == 1751, 'PenDigits origins and held-out record identifiers')
# The controlled two-atom experiment has an analytically known optimum.
with gzip.open(RES / 'known_risk_v1/trials.csv.gz', 'rt') as f:
    known = list(csv.DictReader(f))
viol = Counter()
maxoraclediff = 0.0
knownmeans = {}
for r in known:
    p = float(r['p'])
    a = float(r['softness'])
    k = float(r['kappa'])
    oracle = p * (1 - p) * max(2 - k * math.sqrt(2) * a, 0) ** 2
    maxoraclediff = max(maxoraclediff, abs(oracle - float(r['oracle_mse'])))
    for key, ok in [('pair_valid', float(r['pair_lower']) <= oracle + 1e-12), ('centered_valid', float(r['centered_lower']) <= oracle + 1e-12), ('upper_valid', float(r['fitted_population_mse']) <= float(r['fitted_upper']) + 1e-12)]:
        viol[key] += not ok
        chk(int(ok) == int(r[key]), f'Known-risk flag/{key}')
for a in [0.0, 0.6]:
    rs = [r for r in known if float(r['p']) == 0.5 and float(r['kappa']) == 1 and (int(r['n']) == 4096) and (float(r['softness']) == a)]
    knownmeans[str(a)] = {k: st.fmean((float(r[k]) for r in rs)) for k in ['oracle_mse', 'pair_lower', 'centered_lower', 'fitted_upper']}
chk(len(known) == 28800 and dict(viol) == {'pair_valid': 1, 'centered_valid': 0, 'upper_valid': 0}, 'Known-risk trial and violation counts')
chk(maxoraclediff < 1e-14, 'Known-risk analytic oracle recomputation')
for prefix, expected in [('Forecast table:', 9), ('Certified-selection table:', 4), ('Improvement table:', 4), ('Oscillator table:', 16), ('Sensitivity table:', 12), ('PenDigits table:', 12)]:
    observed = sum((item.startswith(prefix) for item in checks + errors))
    chk(observed == expected, f'{prefix} expected {expected} table rows, observed {observed}')
input_paths = [p for campaign in ['main_v1', 'oscillators_v1', 'sensitivity_v1_k1', 'sensitivity_v1_k4', 'sensitivity_v1_k16'] for p in sorted((RES / campaign).glob('*/results.json'))]
input_paths += [RES / 'pendigits_v1b' / name for name in ['results.json', 'summary.json', 'frozen_candidates.json', 'writer_losses.csv']]
input_paths += [RES / 'known_risk_v1/trials.csv.gz']
input_paths += [TABLE / name for name in ['forecast_results.tex', 'certified_selection.tex', 'improvements.tex', 'oscillators_table.tex', 'sensitivity_results.tex', 'pendigits_results.tex']]
input_hashes = {str(p.relative_to(EXT) if EXT in p.parents else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
summary = {'passed_checks': len(checks), 'failed_checks': errors, 'candidate_records': sum((len(r['results']) for r in allruns)), 'certificate_pairs_recomputed_from_stored_moments': ncert, 'paired_bounds_recomputed_from_stored_moments': npair, 'maximum_bound_arithmetic_difference': maxbounddiff, 'maximum_recorded_optimization_gap': maxgap, 'pendigits_maximum_writer_reaggregation_difference': maxwriterdiff, 'pendigits_writer_counts': writer_counts, 'known_risk_trials': len(known), 'known_risk_failures': dict(viol), 'known_risk_maximum_oracle_difference': maxoraclediff, 'known_risk_selected_means': knownmeans, 'table_checks': [c for c in checks if 'table:' in c.lower()], 'limitations': ['No model fitting or costly experiment replay performed.', 'Bound arithmetic uses saved sufficient statistics; raw synthetic NPZ archives are not independently reprocessed here.', 'Writer inference and iid sampling are not validated by numeric reaggregation.']}
summary['schema'] = 'retained-statistics-verification-v1'
summary['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
summary['input_sha256'] = input_hashes
summary['status'] = 'PASS' if not errors else 'FAIL'
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x') as stream:
    json.dump(summary, stream, indent=2)
    stream.write('\n')
print(json.dumps({key: summary[key] for key in ['status', 'candidate_records', 'certificate_pairs_recomputed_from_stored_moments', 'paired_bounds_recomputed_from_stored_moments', 'maximum_bound_arithmetic_difference', 'known_risk_trials', 'known_risk_failures', 'failed_checks']}, indent=2))
if errors:
    raise SystemExit(1)
