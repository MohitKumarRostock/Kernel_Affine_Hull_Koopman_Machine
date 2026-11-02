"""Independent reconstruction of the frozen block-hierarchy benchmark.

Imports neither the experiment runner nor either optimization implementation.
QP evidence is checked from weak duality with independently rebuilt constraints;
all retained categorical draws, score summaries and confidence endpoints are
recomputed. Exact count enumerations independently check the two analytic laws.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import tarfile

import numpy as np


TOL = 2e-9


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_lines(path):
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            yield json.loads(line)


def bernoulli_kl(q, p):
    if q == 0:
        return -math.log1p(-p)
    if q == 1:
        return -math.log(p)
    if p <= 0 or p >= 1:
        return math.inf
    return q * math.log(q / p) + (1 - q) * (math.log1p(-q) - math.log1p(-p))


def lower_kl(mean, count, bound, alpha):
    """Independent bisection; returns the mathematical endpoint, no guard."""
    q = min(1.0, max(0.0, mean / bound))
    if q == 0:
        return 0.0
    if q == 1:
        return bound * math.exp(-math.log(1 / alpha) / count)
    left, right = 0.0, q
    target = math.log(1 / alpha) / count
    for _ in range(100):
        middle = (left + right) / 2
        if bernoulli_kl(q, middle) > target:
            left = middle
        else:
            right = middle
    return bound * (left + right) / 2


def qp_check(law, weights, record):
    """Rebuild the 1D constrained problem; evaluate its dual independently."""
    u = np.asarray(law['u'], dtype=np.longdouble)
    y = np.asarray(law['y'], dtype=np.longdouble)
    weights = np.asarray(weights, dtype=np.longdouble)
    active = weights > 0
    locations = sorted(set(u[active]))
    mass, means, within = [], [], np.longdouble(0)
    for point in locations:
        selected = active & (u == point)
        probability = weights[selected].sum()
        mean = (weights[selected] * y[selected]).sum() / probability
        mass.append(probability)
        means.append(mean)
        within += (weights[selected] * (y[selected] - mean) ** 2).sum()
    mass, means = np.asarray(mass), np.asarray(means)
    locations = np.asarray(locations)
    size = len(locations)
    require(np.allclose(locations.astype(float), record['support'], rtol=0, atol=1e-14), 'QP support differs')
    # The archived solver's inequalities are adjacent increasing differences,
    # adjacent decreasing differences, then the upper and lower target box.
    rows, bounds = [], []
    for direction in (1, -1):
        for j in range(size - 1):
            row = np.zeros(size, dtype=np.longdouble)
            row[j], row[j + 1] = -direction, direction
            rows.append(row)
            bounds.append(np.longdouble(law['kappa']) * (locations[j + 1] - locations[j]))
    for direction in (1, -1):
        for j in range(size):
            row = np.zeros(size, dtype=np.longdouble)
            row[j] = direction
            rows.append(row)
            bounds.append(np.longdouble(1))
    matrix, bounds = np.asarray(rows), np.asarray(bounds)
    lam = np.asarray(record['dual_multipliers'], dtype=np.longdouble)
    fitted = np.asarray(record['primal'], dtype=np.longdouble)
    require(lam.shape == (len(rows),), 'Unexpected dual dimension')
    require(np.isfinite(lam).all() and np.min(lam) >= 0, 'Nonfeasible dual multiplier')
    require(fitted.shape == (size,) and np.isfinite(fitted).all(), 'Invalid fitted values')
    violation = max(np.longdouble(0), np.max(matrix @ fitted - bounds))
    # Check all pair constraints separately from the adjacent dual system.
    pair_violation = max(np.longdouble(0), np.max(
        np.abs(fitted[:, None] - fitted[None, :])
        - np.longdouble(law['kappa']) * np.abs(locations[:, None] - locations[None, :])))
    require(violation <= 1e-10 and pair_violation <= 1e-10, 'Primal not Lipschitz feasible')
    at = matrix.T @ lam
    dual = within + lam @ (matrix @ means - bounds) - np.sum(at ** 2 / (4 * mass))
    primal = within + np.sum(mass * (fitted - means) ** 2)
    recorded_raw = float(record['dual_raw'])
    require(abs(float(dual) - recorded_raw) <= 1e-10, 'Independent dual value differs')
    require(float(record['dual_guard']) >= 0, 'Negative numerical guard')
    recorded_lower = float(record['lower'])
    recorded_upper = float(record['upper'])
    require(abs(recorded_lower - max(0, recorded_raw - float(record['dual_guard']))) <= 1e-13,
            'Lower value does not equal guarded dual')
    require(recorded_lower <= max(0, float(dual)) + 1e-12, 'Lower exceeds independently evaluated dual')
    require(recorded_upper >= float(primal) - 1e-12, 'Upper is below independently evaluated primal')
    require(recorded_lower <= recorded_upper + 1e-12, 'Weak-duality violation')
    require(abs(recorded_upper - recorded_lower - float(record['gap'])) <= 1e-13, 'Incorrect recorded gap')
    require(float(record['gap']) <= 1e-8, 'Unresolved optimization gap')
    # Independent closed forms for the constant and rare-conflict supports.
    if len(set(law['u'])) == 1:
        target_mean = np.sum(weights * y) / weights.sum()
        exact = np.sum(weights * (y - target_mean) ** 2)
        require(recorded_lower <= exact + 1e-11 <= recorded_upper + 2e-11, 'Constant oracle mismatch')
    if law['name'] == 'rare_conflict':
        a, c = weights[1], weights[2]
        exact = a * c / (a + c) if a + c else np.longdouble(0)
        require(recorded_lower <= exact + 1e-11 <= recorded_upper + 2e-11, 'Rare-conflict oracle mismatch')
    return {
        'dual_error': abs(float(dual) - recorded_raw),
        'max_violation': float(max(violation, pair_violation)),
        'gap': float(record['gap']),
    }


def enumerate_rare(b, epsilon):
    total = 0.0
    for a in range(b + 1):
        for c in range(b - a + 1):
            common = b - a - c
            probability = math.comb(b, a) * math.comb(b - a, c) * epsilon ** (a + c) * (1 - 2 * epsilon) ** common
            score = a * c / (a + c) / (b - 1) if a + c else 0.0
            total += probability * score
    return total


def enumerate_noise(b, points):
    total = 0.0
    for observations in product(range(2 * points), repeat=b):
        counts = [0] * points
        sums = [0] * points
        for atom in observations:
            location, target = atom // 2, 2 * (atom % 2) - 1
            counts[location] += 1
            sums[location] += target
        sse = sum(n - s * s / n for n, s in zip(counts, sums) if n)
        total += sse / (b - 1)
    return total / (2 * points) ** b


def verify(results, output):
    require(not output.exists(), 'Refusing to overwrite an existing audit')
    freeze = json.loads((results / 'protocol_freeze.json').read_text())
    summary = json.loads((results / 'run_summary.json').read_text())
    laws = {law['name']: law for law in freeze['laws']}
    require(len(laws) == 8, 'Expected all eight frozen laws')
    require(freeze['sample_sizes'] == [256, 1024, 4096], 'Sample sizes changed')
    require(freeze['block_sizes'] == [2, 4, 8, 16, 32], 'Block sizes changed')
    require(freeze['repetitions'] == 100 and freeze['seed'] == 202610291901, 'Frozen random design changed')
    alpha = float(freeze['alpha_per_endpoint'])
    require(abs(alpha * 18 - .05) < 1e-15, 'Failure probability allocation differs')
    for name, expected in summary['files_sha256'].items():
        require(sha256(results / name) == expected, 'Result checksum mismatch: ' + name)
    require(sha256(results / 'source_snapshot.tar.gz') == freeze['snapshot_sha256'], 'Snapshot checksum mismatch')
    with tarfile.open(results / 'source_snapshot.tar.gz', 'r:gz') as archive:
        for name, expected in freeze['source_sha256'].items():
            handle = archive.extractfile(name)
            require(handle is not None and hashlib.sha256(handle.read()).hexdigest() == expected,
                    'Archived source checksum mismatch: ' + name)

    exact_risks = {
        'collapsed_binary': 1.0,
        'two_atom_partial': (1 - .6 / math.sqrt(2)) ** 2,
        'three_point_v': 2 / 9,
        'five_point_zigzag': .54,
        'partial_collapse': 19 / 75,
        'noisy_affine': .25,
        'rare_conflict': .025,
        'affine_mismatch': 0.0,
    }
    require(set(laws) == set(exact_risks), 'Unexpected law identity')
    max_dual_error = max_gap = max_violation = 0.0
    population_rows = json.loads((results / 'population.json').read_text())
    require({r['law'] for r in population_rows} == set(laws), 'Population table omits a law')
    for row in population_rows:
        law = laws[row['law']]
        check = qp_check(law, law['probabilities'], row['qp'])
        max_dual_error = max(max_dual_error, check['dual_error'])
        max_violation = max(max_violation, check['max_violation'])
        risk = exact_risks[row['law']]
        require(row['optimum_lower'] <= risk + 1e-10 <= row['optimum_upper'] + 2e-10,
                'Population optimum misses analytic value')
        u, y, p = (np.asarray(law[k]) for k in ('u', 'y', 'probabilities'))
        pair = .5 * np.maximum(abs(y[:, None] - y[None, :]) - law['kappa'] * abs(u[:, None] - u[None, :]), 0) ** 2
        require(abs(float(p @ pair @ p) - row['pair_witness']) < 1e-13, 'Population pair witness differs')

    cache = {}
    for row in json_lines(results / 'qp_certificates.jsonl.gz'):
        law = laws[row['law']]
        counts = np.asarray(row['counts'])
        require(len(counts) == len(law['u']) and np.all(counts >= 0) and counts.sum() >= 2,
                'Invalid cached count vector')
        require(np.all(counts.astype(int) == counts), 'Fractional sample count')
        key = (row['law'], tuple(int(v) for v in counts))
        require(key not in cache, 'Duplicate cached QP record')
        check = qp_check(law, counts / counts.sum(), row)
        max_dual_error = max(max_dual_error, check['dual_error'])
        max_gap = max(max_gap, check['gap'])
        max_violation = max(max_violation, check['max_violation'])
        cache[key] = float(row['lower'])
    require(len(cache) == summary['distinct_optimization_programs'], 'Cached program count differs')
    require(abs(max_gap - summary['maximum_empirical_qp_gap']) < 1e-13, 'Maximum gap differs')

    with (results / 'endpoints.csv').open(newline='') as handle:
        endpoint_rows = list(csv.DictReader(handle))
    endpoints = {}
    for row in endpoint_rows:
        key = (row['law'], int(row['repetition']), int(row['n']), row['method'])
        require(key not in endpoints, 'Duplicate endpoint record')
        endpoints[key] = row
    require(len(endpoints) == 8 * 100 * 3 * 6 == summary['endpoint_count'], 'Endpoint count differs')
    children = np.random.SeedSequence(freeze['seed']).spawn(800)
    law_indices = {name: i for i, name in enumerate(laws)}
    used_endpoints, seen_streams, used_cache = set(), set(), set()
    max_summary_difference = max_endpoint_difference = 0.0
    violations = 0
    pointwise_violations = 0
    score_count = 0
    for stream in json_lines(results / 'samples.jsonl.gz'):
        name, repetition = stream['law'], int(stream['repetition'])
        key = (name, repetition)
        require(key not in seen_streams and 0 <= repetition < 100, 'Duplicate or invalid sample stream')
        seen_streams.add(key)
        law = laws[name]
        sample = np.asarray(stream['sample'], dtype=int)
        require(sample.shape == (4096,) and sample.min() >= 0 and sample.max() < len(law['u']), 'Invalid categorical draws')
        expected = np.random.default_rng(children[100 * law_indices[name] + repetition]).choice(
            len(law['probabilities']), size=4096, p=law['probabilities'])
        require(np.array_equal(sample, expected), 'Raw draw does not replay from frozen random stream')
        u, y = (np.asarray(law[k]) for k in ('u', 'y'))
        risk = exact_risks[name]
        stream_failure = False
        for n in freeze['sample_sizes']:
            prefix = sample[:n]
            for b in freeze['block_sizes'] + [n]:
                method = 'full_erm' if b == n else 'block_' + str(b)
                row_key = (name, repetition, n, method)
                row = endpoints[row_key]
                used_endpoints.add(row_key)
                if method == 'full_erm':
                    counts = tuple(np.bincount(prefix, minlength=len(u)).tolist())
                    cache_key = (name, counts)
                    used_cache.add(cache_key)
                    mean, count, bound = cache[cache_key], n, 4.0
                else:
                    chunks = prefix.reshape(-1, b)
                    if b == 2:
                        left, right = chunks[:, 0], chunks[:, 1]
                        scores = .5 * np.maximum(abs(y[left] - y[right]) - law['kappa'] * abs(u[left] - u[right]), 0) ** 2
                    else:
                        counts = np.stack([(chunks == atom).sum(axis=1) for atom in range(len(u))], axis=1)
                        scores = []
                        for values in counts:
                            cache_key = (name, tuple(values.tolist()))
                            used_cache.add(cache_key)
                            scores.append(cache[cache_key] * b / (b - 1))
                        scores = np.asarray(scores)
                    count = len(scores)
                    mean = float(np.mean(scores))
                    bound = 4 * (b // 2) * (b - b // 2) / (b * (b - 1))
                    require(np.min(scores) >= 0 and np.max(scores) <= bound + 1e-9, 'Block score outside sharp range')
                    score_count += count
                    for column, value in (('variance', float(np.var(scores, ddof=1))), ('score_min', float(scores.min())), ('score_max', float(scores.max()))):
                        difference = abs(float(row[column]) - value)
                        max_summary_difference = max(max_summary_difference, difference)
                        require(difference <= 1e-12, 'Incorrect block summary: ' + column)
                difference = abs(float(row['mean']) - mean)
                max_summary_difference = max(max_summary_difference, difference)
                require(difference <= 1e-12, 'Incorrect empirical mean')
                require(int(row['independent_scores']) == count, 'Independent score count differs')
                require(int(row['block_size']) == b, 'Block size differs')
                require(abs(float(row['bound']) - bound) < 1e-14 and abs(float(row['alpha']) - alpha) < 1e-15,
                        'Range or allocation differs')
                endpoint = lower_kl(mean, count, bound, alpha)
                difference = abs(float(row['lower']) - endpoint)
                max_endpoint_difference = max(max_endpoint_difference, difference)
                require(difference <= 2e-10, 'Independent KL endpoint differs')
                failed = float(row['lower']) > risk + 1e-9
                require(int(row['coverage_failure']) == int(failed), 'Incorrect coverage failure flag')
                pointwise_violations += int(failed)
                stream_failure |= failed
            # Six methods exactly: five blocks and one full-sample comparator.
        violations += int(stream_failure)
    require(len(seen_streams) == 800 == summary['law_repetition_streams'], 'Sample stream count differs')
    require(used_endpoints == set(endpoints), 'Unused or missing endpoint')
    require(used_cache == set(cache), 'Unused or missing cached optimization')
    require(violations == summary['simultaneous_coverage_failures'], 'Simultaneous coverage count differs')

    with (results / 'analytic_hierarchy.csv').open(newline='') as handle:
        analytic_rows = list(csv.DictReader(handle))
    analytic_keys, max_formula_difference = set(), 0.0
    curves = {}
    for row in analytic_rows:
        b, parameter = int(row['block_size']), float(row['parameter'])
        name = row['law']
        key = (name, parameter, b)
        require(key not in analytic_keys, 'Duplicate analytic record')
        analytic_keys.add(key)
        if name == 'rare_conflict':
            value = (2 * b * parameter - 1 + (1 - 2 * parameter) ** b) / (4 * (b - 1))
            risk = parameter / 2
        else:
            require(name == 'separated_noise' and parameter in (4, 16, 64), 'Unknown analytic law')
            value = (b - parameter * (1 - (1 - 1 / parameter) ** b)) / (b - 1)
            risk = 1.0
        for column, expected in (('witness', value), ('risk', risk), ('ratio', value / risk)):
            difference = abs(float(row[column]) - expected)
            max_formula_difference = max(max_formula_difference, difference)
            require(difference <= 1e-13, 'Analytic table value differs')
        curves.setdefault((name, parameter), []).append((b, value))
    expected_keys = {(name, parameter, b) for b in range(2, 513) for name, parameters in
                     (('rare_conflict', (.01, .05, .10)), ('separated_noise', (4, 16, 64))) for parameter in parameters}
    require(analytic_keys == expected_keys, 'Analytic grid incomplete')
    for values in curves.values():
        require(np.all(np.diff([value for _, value in sorted(values)]) >= -1e-13), 'Analytic hierarchy is not monotone')
    enumeration_checks, max_enumeration_difference = 0, 0.0
    for b in range(2, 9):
        for epsilon in (.01, .05, .10):
            enumerated = enumerate_rare(b, epsilon)
            formula = (2 * b * epsilon - 1 + (1 - 2 * epsilon) ** b) / (4 * (b - 1))
            difference = abs(enumerated - formula)
            max_enumeration_difference = max(max_enumeration_difference, difference)
            require(difference <= 2e-14, 'Rare count enumeration differs from formula')
            enumeration_checks += 1
    for b in range(2, 6):
        for points in (2, 3, 4):
            enumerated = enumerate_noise(b, points)
            formula = (b - points * (1 - (1 - 1 / points) ** b)) / (b - 1)
            difference = abs(enumerated - formula)
            max_enumeration_difference = max(max_enumeration_difference, difference)
            require(difference <= 2e-12, 'Noise sequence enumeration differs from formula')
            enumeration_checks += 1

    audit = {
        'verified': True,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': sha256(Path(__file__)),
        'independent_of_runner_and_optimization_imports': True,
        'source_archive_hashes_verified': len(freeze['source_sha256']),
        'population_oracles_checked': len(population_rows),
        'cached_dual_certificates_checked': len(cache),
        'categorical_streams_replayed': len(seen_streams),
        'categorical_draws_replayed': len(seen_streams) * 4096,
        'individual_block_scores_reconstructed': score_count,
        'endpoints_reconstructed': len(endpoints),
        'simultaneous_coverage_failures': violations,
        'pointwise_coverage_failures': pointwise_violations,
        'maximum_independent_dual_error': max_dual_error,
        'maximum_primal_constraint_violation': max_violation,
        'maximum_empirical_primal_dual_gap': max_gap,
        'maximum_score_summary_difference': max_summary_difference,
        'maximum_endpoint_difference': max_endpoint_difference,
        'analytic_table_rows_checked': len(analytic_rows),
        'maximum_analytic_table_difference': max_formula_difference,
        'independent_exact_enumeration_checks': enumeration_checks,
        'maximum_enumeration_difference': max_enumeration_difference,
        'numerical_scope': 'Floating-point reconstruction and weak-duality checks, not interval arithmetic or a proof of coverage from finite repetitions.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.results, args.output)
