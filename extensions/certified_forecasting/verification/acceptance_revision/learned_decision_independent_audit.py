#!/usr/bin/env python3
"""Independent structural and numerical audit of the learned-KAHM study.

Imports no project module. Rebuilds KMeans references and every local
regularized KAHM matrix from training samples, evaluates the published folding
formula directly from numeric JSON, and recomputes selected validation losses,
all calibration endpoints, and all minimum-cost decisions. Existing outputs
are never overwritten. This is numerical verification, not verified arithmetic.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans


def read_json(path):
    return json.loads(Path(path).read_text())


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if not isinstance(value, dict):
        return value
    if '__array__' in value:
        a = np.asarray(value['data'], dtype=value['dtype']).reshape(value['shape'])
        assert a.dtype.kind in 'fiub'
        return a
    if '__kahm_block__' in value:
        return decode(value['__kahm_block__'])
    if '__slice__' in value:
        return slice(*value['__slice__'])
    return {key: decode(item) for key, item in value.items()}


def generate(seed, count, regime):
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1., 1.]), (count, 2))
    x = signs * (.9 + rng.uniform(-.1, .1, (count, 2)))
    if regime == 'repairable':
        source = x
    else:
        source = rng.choice(np.array([-1., 1.]), (count, 2))
        source *= .9 + rng.uniform(-.1, .1, (count, 2))
    y = .8 * np.prod(source, axis=1) + rng.uniform(-.05, .05, count)
    return x, y


class Checks:
    def __init__(self):
        self.count = 0
        self.maximum = {}

    def close(self, actual, recorded, category, atol=5e-9, rtol=1e-10):
        x, y = np.asarray(actual), np.asarray(recorded)
        assert x.shape == y.shape, (category, x.shape, y.shape)
        difference = float(np.max(np.abs(x-y), initial=0.))
        self.maximum[category] = max(self.maximum.get(category, 0.), difference)
        self.count += x.size
        assert np.allclose(x, y, atol=atol, rtol=rtol), (category, difference)


def scaling(bank, x):
    span = np.where(bank['constant_features_'], 1., bank['input_range_'])
    z = 2*(x-bank['input_min_'])/span-1
    z[:, bank['constant_features_']] = 0
    return z


def local_audit(ae, references, checks):
    checks.close(ae['y_data'], references, 'training_reference_values', atol=0, rtol=0)
    pc = ae['PC']
    checks.close(pc.T @ pc, np.eye(pc.shape[1]), 'pca_orthonormality', atol=1e-12)
    projected = pc.T @ references
    checks.close(projected, ae['y_data_n_subspace'], 'projected_references', atol=1e-12)
    covariance = np.cov(projected)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() > 0
    precision = np.linalg.inv(covariance)
    checks.close(ae['sqrt_weights_matrix'] @ ae['sqrt_weights_matrix'], precision,
                 'covariance_inverse', atol=5e-8)
    differences = projected.T[:, None, :] - projected.T[None, :, :]
    squared = np.einsum('ijk,kl,ijl->ij', differences, precision, differences)
    kernel = np.exp(-squared/(2*projected.shape[0]))
    reference_energy = float(np.mean(references**2))
    regularizer = 2.5*reference_energy
    for _ in range(100):
        inverse = np.linalg.inv(kernel + regularizer*np.eye(len(kernel)))
        residual = references - references @ (kernel @ inverse).T
        next_regularizer = 2*reference_energy + float(np.mean(residual**2))
        if abs(next_regularizer-regularizer) <= 1e-3:
            break
        regularizer = next_regularizer
    checks.close(inverse, ae['B'], 'local_kahm_inverse', atol=5e-10)


def bank_audit(bank, train_x, checks):
    assert bank['kind'] == 'kahm'
    assert bank['input_scaling'] == bank['input_scaling_'] == 'minmax'
    assert bank['cluster_counts'] == [2, 4]
    assert bank['omega'] == 16 and bank['tau'] == 1e-6
    checks.close(bank['input_min_'], train_x.min(0), 'training_extrema', atol=0, rtol=0)
    checks.close(bank['input_max_'], train_x.max(0), 'training_extrema', atol=0, rtol=0)
    checks.close(bank['input_range_'], train_x.max(0)-train_x.min(0), 'training_extrema', atol=0, rtol=0)
    checks.close(bank['fit_indices_'], np.arange(len(train_x)), 'geometry_training_indices', atol=0, rtol=0)
    z = scaling(bank, train_x)
    checks.close(z.min(0), -np.ones(2), 'training_scaled_range', atol=1e-14)
    checks.close(z.max(0), np.ones(2), 'training_scaled_range', atol=1e-14)
    local_count = 0
    for block_index, block in enumerate(bank['blocks_']):
        expected_seed = bank['seed'] + 104729*(block_index+1)
        assert block['seed'] == expected_seed
        count = bank['cluster_counts'][block_index]
        fitted = KMeans(n_clusters=count, n_init='auto', random_state=expected_seed).fit(z)
        centers = np.column_stack([z[fitted.labels_ == j].mean(0) for j in range(count)])
        model = block['model_']
        assert model['n_clusters'] == count
        assert model['cluster_strategy'] == 'kmeans_state'
        assert model['scaling_enabled'] is False and model['l2_normalization_enabled'] is False
        checks.close(centers, model['cluster_centers'], 'learned_cluster_centers', atol=1e-12)
        rng = np.random.default_rng(expected_seed)
        for cluster, autoencoders in enumerate(model['classifier']):
            indices = np.flatnonzero(fitted.labels_ == cluster)
            assert len(indices) > 48
            chosen = np.sort(rng.choice(indices, 48, replace=False))
            references = z[chosen].T.astype(np.float32).astype(float)
            assert len(autoencoders) == 1
            local_audit(autoencoders[0], references, checks)
            local_count += 1
    return local_count


def features(bank, x):
    z = scaling(bank, x).T
    blocks = []
    for block in bank['blocks_']:
        distances = []
        for aes in block['model_']['classifier']:
            ae_distances = []
            for ae in aes:
                output = np.empty_like(z)
                weighted_reference = (ae['sqrt_weights_matrix'] @ ae['y_data_n_subspace']).T
                for start in range(0, z.shape[1], 1000):
                    stop = min(z.shape[1], start+1000)
                    projected = ae['PC'].T @ z[:, start:stop]
                    weighted_query = (ae['sqrt_weights_matrix'] @ projected).T
                    # Direct coordinate differences, independently of the inherited
                    # sum-of-squares expansion used by distance_matrix.py.
                    differences = weighted_query[:, None, :] - weighted_reference[None, :, :]
                    squared_distance = np.sum(differences**2, axis=2)
                    kernel = np.exp(-squared_distance/(2*projected.shape[0]))
                    weights = kernel @ ae['B']
                    normalizer = weights.sum(axis=1, keepdims=True)
                    normalizer[normalizer == 0] = 1.
                    output[:, start:stop] = ae['y_data'] @ (weights/normalizer).T
                radial = 1-np.exp(-np.linalg.norm(output-z, axis=0))
                denominator = np.linalg.norm(output, axis=0)*np.linalg.norm(z, axis=0)
                cosine = np.divide(np.sum(output*z, axis=0), denominator,
                                   out=np.zeros(z.shape[1]), where=denominator != 0)
                angular = np.arccos(np.clip(cosine, -1, 1))/np.pi
                ae_distances.append(np.hypot(radial, angular)/np.sqrt(2))
            distances.append(np.min(ae_distances, axis=0))
        score = np.array(distances).T
        log_weight = block['omega']*np.log(1-np.clip(score, 0, 1)+block['tau'])
        exp_weight = np.exp(log_weight-log_weight.max(axis=1, keepdims=True))
        blocks.append(exp_weight/exp_weight.sum(axis=1, keepdims=True))
    return np.column_stack(blocks)


def mean_bounds(values, bound, alpha):
    n = len(values)
    mean, variance = float(np.mean(values)), float(np.var(values, ddof=1))
    radius = math.sqrt(2*variance*math.log(2/alpha)/n)
    radius += 7*bound*math.log(2/alpha)/(3*(n-1))
    return max(0., mean-radius), min(bound, mean+radius), mean


def pair_scores(f, y, slices, kappa):
    last = len(y)//2*2
    distance = np.maximum.reduce([
        np.linalg.norm(f[:last:2, s]-f[1:last:2, s], axis=1) for s in slices])
    return np.maximum(np.abs(y[:last:2]-y[1:last:2])-kappa*distance, 0)**2/2


def min_cost(rows, predicate):
    costs = [r['cost'] for r in rows if predicate(r)]
    return min(costs) if costs else None


def audit(folder, output):
    if output.exists():
        raise FileExistsError(output)
    config = read_json(folder/'protocol.json')
    results = read_json(folder/'results.json')
    protocol_freeze = read_json(folder/'protocol_freeze.json')
    development_freeze = read_json(folder/'development_freeze.json')
    repair = read_json(folder/'precalibration_freeze_issue/repair.json')
    original = read_json(folder/'precalibration_freeze_issue/development_freeze.json')
    original_protocol = read_json(folder/'precalibration_freeze_issue/protocol_freeze.json')
    started = read_json(folder/'calibration_started.json')
    assert repair['design_models_seeds_calibration_sizes_and_tolerance_changed'] is False
    assert repair['calibration_generated'] is False
    assert original['frozen_models_sha256'] == development_freeze['frozen_models_sha256'] == digest(folder/'frozen_models.json.gz')
    assert original['development_scores_sha256'] == development_freeze['development_scores_sha256'] == digest(folder/'development_scores.csv')
    assert original_protocol['protocol_sha256'] == protocol_freeze['protocol_sha256'] == digest(folder/'protocol.json')
    assert original['frozen_at_utc'] < repair['recorded_at_utc'] < started['started_at_utc'] < results['completed_at_utc']
    assert started['development_freeze_sha256'] == digest(folder/'development_freeze.json')
    current_sources = read_json(folder/'source_snapshot.json')
    old_sources = {x['path']:x for x in read_json(folder/'precalibration_freeze_issue/source_snapshot.json')['files']}
    assert digest(folder/'source_snapshot.tar.gz') == current_sources['archive_sha256'] == protocol_freeze['source_snapshot_sha256']
    with tarfile.open(folder/'source_snapshot.tar.gz', 'r:gz') as archive:
        imported_modules = set()
        for source in current_sources['files']:
            assert source == old_sources[source['path']]
            contents = archive.extractfile(source['path']).read()
            assert hashlib.sha256(contents).hexdigest() == source['sha256']
            if source['path'].endswith('.py'):
                tree = ast.parse(contents)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules.update(x.name for x in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_modules.add(node.module)
    assert not any('block_witness' in module for module in imported_modules)
    excluded = set(old_sources) - {x['path'] for x in current_sources['files']}
    assert excluded == {'extensions/certified_forecasting/src/certified_forecasting/block_witness.py'}

    rows = read_gzip(folder/'interval_details.json.gz')
    details = {(r['record']['regime'], r['record']['repetition'], r['record']['n_calibration'], r['record']['prefix']): r for r in rows}
    with open(folder/'decisions.csv', newline='') as stream:
        decisions = {(r['regime'], int(r['repetition']), int(r['n_calibration'])):r for r in csv.DictReader(stream)}
    checks = Checks()
    count = len(config['regimes'])*config['n_repetitions_per_regime']*len(config['calibration_sizes'])*2
    assert count == config['n_class_size_records'] == 320
    upper_alpha = config['upper_budget']/count
    pair_alpha = config['unconditional_lower_budget']/count
    eta = config['stratified_lower_budget']/(count*2)
    checks.close(count*(upper_alpha+pair_alpha+2*eta), .05, 'confidence_allocation', atol=0, rtol=0)
    target_diameter = config['target_bounds'][1]-config['target_bounds'][0]
    loss_bound = target_diameter**2
    assert config['target_bounds'] == [-.85, .85] and config['mse_tolerance'] == .1
    assert config['kappa'] == 2
    checks.close(results['known_bayes_risk']['repairable'], .05**2/3, 'known_bayes_risks')
    checks.close(results['known_bayes_risk']['irreducible'], .8**2*(.81+.01/3)**2+.05**2/3, 'known_bayes_risks')
    local_models = 0
    endpoint_count = 0
    decision_count = 0
    score_count = 0
    data_seeds = set()
    selected_not_converged = 0
    fitted_not_converged = 0
    maximum_norm = 0.
    maximum_selected_gap = 0.
    outside_fractions = []
    summary = {}
    for record in read_gzip(folder/'frozen_models.json.gz'):
        regime, repetition = record['regime'], record['repetition']
        regime_index = config['regimes'].index(regime)
        base = 98100000+1000*regime_index+10*repetition
        expected_seeds = dict(training=base+1, validation=base+2, calibration=base+3)
        assert record['data_seeds'] == expected_seeds
        for seed in expected_seeds.values():
            assert seed not in data_seeds
            data_seeds.add(seed)
        train_x, train_y = generate(base+1, config['n_train'], regime)
        valid_x, valid_y = generate(base+2, config['n_validation'], regime)
        for key, value in [('train_x',train_x),('train_y',train_y),('valid_x',valid_x),('valid_y',valid_y)]:
            assert array_digest(value) == record['development_hashes'][key]
        bank = decode(record['bank'])
        local_models += bank_audit(bank, train_x, checks)
        assert bank['block_slices_'] == [slice(0,2), slice(2,6)]
        vf = features(bank, valid_x)
        for prefix, model in enumerate(record['models'], 1):
            scores = []
            for candidate in model['candidates']:
                width = candidate['cost']
                assert width == [2,6][prefix-1]
                coefficient = np.asarray(candidate['coefficients'])
                norm = sum(np.linalg.norm(coefficient[s]) for s in bank['block_slices_'][:prefix])
                assert norm <= 2+1e-12
                maximum_norm = max(maximum_norm, float(norm))
                checks.close(norm, candidate['norm_sum'], 'coefficient_norms')
                fitted_not_converged += int(not candidate['converged'])
                prediction = np.clip(vf[:,:width] @ coefficient + candidate['intercept'], -.85, .85)
                mse = float(np.mean((valid_y-prediction)**2))
                checks.close(mse, candidate['validation_mse'], 'validation_mse')
                scores.append(mse)
            assert model['selected'] == model['candidates'][int(np.argmin(scores))]
            selected_not_converged += int(not model['selected']['converged'])
            maximum_selected_gap = max(maximum_selected_gap, model['selected']['optimization_gap'])
        x, y = generate(base+3, max(config['calibration_sizes']), regime)
        assert np.all(np.abs(y) <= .85)
        f = features(bank, x)
        checks.close(f[:,:2].sum(1), np.ones(len(f)), 'simplex_membership')
        checks.close(f[:,2:].sum(1), np.ones(len(f)), 'simplex_membership')
        assert np.all(f >= 0)
        outside_fractions.append(float(np.mean((x<bank['input_min_']) | (x>bank['input_max_']))))
        labels = [np.argmax(f[:,s], axis=1) for s in bank['block_slices_']]
        for n in config['calibration_sizes']:
            verified = []
            for prefix, model in enumerate(record['models'], 1):
                selected = model['selected']
                width = selected['cost']
                slices = bank['block_slices_'][:prefix]
                prediction = np.clip(f[:n,:width] @ np.asarray(selected['coefficients']) + selected['intercept'], -.85, .85)
                losses = (y[:n]-prediction)**2
                _, upper, mse = mean_bounds(losses, loss_bound, upper_alpha)
                witness = pair_scores(f[:n,:width], y[:n], slices, 2)
                unconditional, _, witness_mean = mean_bounds(witness, loss_bound/2, pair_alpha)
                stored = details[regime,repetition,n,prefix]
                row = stored['record']
                checks.close([upper,mse,unconditional,witness_mean],
                    [row['upper'],row['mse'],row['unconditional_lower'],row['unconditional_witness']], 'unconditional_endpoints')
                stratified = []
                for partition, m in enumerate([2,4]):
                    labels_n = labels[partition][:n]
                    counts = np.bincount(labels_n, minlength=m)
                    radius = np.sqrt(np.log(4*m/eta)/(2*n))
                    lo, hi = np.maximum(0, counts/n-radius), np.minimum(1,counts/n+radius)
                    conditional_lowers = []
                    for cell in range(m):
                        indices = np.flatnonzero(labels_n == cell)
                        if len(indices) < 4:
                            lower = 0.
                        else:
                            q = pair_scores(f[indices,:width], y[indices], slices, 2)
                            lower, _, _ = mean_bounds(q, loss_bound/2, eta/(2*m))
                        conditional_lowers.append(lower)
                    lp = linprog(conditional_lowers, A_eq=np.ones((1,m)), b_eq=[1.], bounds=list(zip(lo,hi)), method='highs')
                    assert lp.success
                    stratified.append(float(lp.fun))
                checks.close(stratified, [row['coarse_stratified_lower'],row['fine_stratified_lower']], 'stratified_endpoints')
                lower = max(unconditional, *stratified)
                checks.close(lower, row['primary_lower'], 'selected_lower')
                assert row['accepted'] == (upper <= .1)
                assert row['excluded'] == (lower > .1)
                verified.append(dict(cost=width, accepted=upper<=.1, excluded=lower>.1,
                    excluded_unconditional=unconditional>.1, excluded_stratified=max(stratified)>.1))
                endpoint_count += 1
                score_count += len(losses)
            upper_cost = min_cost(verified, lambda v:v['accepted'])
            lower_cost = min_cost(verified, lambda v:not v['excluded'])
            lower_unconditional = min_cost(verified, lambda v:not v['excluded_unconditional'])
            lower_stratified = min_cost(verified, lambda v:not v['excluded_stratified'])
            computed = dict(upper_choice=upper_cost, cost_lower=lower_cost, cost_upper=upper_cost,
                unconditional_cost_lower=lower_unconditional, stratified_cost_lower=lower_stratified,
                minimum_cost_certified=upper_cost is not None and upper_cost==lower_cost,
                all_classes_excluded=lower_cost is None,
                unconditional_minimum_cost_certified=upper_cost is not None and upper_cost==lower_unconditional,
                unconditional_all_excluded=lower_unconditional is None,
                stratified_minimum_cost_certified=upper_cost is not None and upper_cost==lower_stratified,
                stratified_all_excluded=lower_stratified is None)
            recorded = decisions[regime,repetition,n]
            for key, value in computed.items():
                assert recorded[key] == ('' if value is None else str(value)), (key,recorded[key],value)
            key = (regime,n)
            summary.setdefault(key, dict(upper_acceptances=0,minimum_cost_certificates=0,joint_infeasibility_certificates=0))
            summary[key]['upper_acceptances'] += int(upper_cost is not None)
            summary[key]['minimum_cost_certificates'] += int(computed['minimum_cost_certified'])
            summary[key]['joint_infeasibility_certificates'] += int(computed['all_classes_excluded'])
            decision_count += 1
        print(f'Independent KAHM algebra verified {regime} repetition {repetition+1}', flush=True)
    assert len(data_seeds) == 120
    for row in results['summary']:
        for key,value in summary[row['regime'],row['n_calibration']].items():
            assert value == row[key]
    report = dict(passed=True,
        scope='No project imports; independent KMeans/reference/regularizer/folding algebra and endpoint/decision reconstruction from portable numerical JSON.',
        caveat='Numerical verification is not interval arithmetic. Recorded freeze order is checked; logs are not independent timestamp attestation. Risk endpoints bound population quantities; the observed calibration MSEs are not exact population risks.',
        actual_kahm_banks=40, learned_blocks=80, local_reference_models_reconstructed=local_models,
        independent_data_seeds=len(data_seeds), all_sources_match_precalibration_repair_manifest=True,
        repaired_archive_excluded_only_unimported_block_module=True,
        selected_models=80, candidate_models=240, endpoints=endpoint_count, decisions=decision_count,
        calibration_loss_terms_across_prefixes=score_count,
        floating_scalar_checks=checks.count, maximum_absolute_discrepancies=checks.maximum,
        maximum_coefficient_sum_norm=maximum_norm,
        candidate_fits_not_reaching_requested_tolerance=fitted_not_converged,
        selected_fits_not_reaching_requested_tolerance=selected_not_converged,
        maximum_selected_optimization_gap=maximum_selected_gap,
        future_coordinates_outside_training_range_fraction_min=min(outside_fractions),
        future_coordinates_outside_training_range_fraction_max=max(outside_fractions),
        familywise_failure_probability=.05,
        independent_reference_bayes_risks=dict(repairable=.05**2/3,irreducible=.8**2*(.81+.01/3)**2+.05**2/3),
        irreducible_class_optimum_known='Both full Lipschitz classes contain the constant zero forecast and histories are independent of targets, so each class optimum equals the known Bayes risk 0.4242004444444446.',
        repairable_class_optima='Not computed exactly; lower endpoints and fitted-risk upper endpoints imply the stated minimum-cost decisions on their joint confidence event.',
        restrictions=['Development pilots informed the chosen constructed distribution and tolerance.',
                      'Counts are repeated fitting/calibration runs under this designed mechanism, not natural-task prevalence.',
                      'Feature-coordinate cost concerns the two declared retained KAHM representations, not runtime or all possible predictors.'],
        results_sha256=digest(folder/'results.json'))
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments = parser.parse_args()
    audit(arguments.results, arguments.output)
