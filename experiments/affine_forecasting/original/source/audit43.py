"""Independent evidence and numerical audit for the predeclared Step 43 study.

This module never imports the producer model/scoring/selection implementation.
It reconstructs exact losses, choices, clipped affine references, certificate
claims and acceptance decisions independently. Stored native map transforms and
classical estimator prediction calls are re-executed under the declared frozen
libraries; those are explicitly operational replays, not new candidate fits or
certificates for ideal native feature arithmetic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from functools import lru_cache
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import struct
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('household_power2008', 'jena_temperature2018', 'metro_traffic2018')
SEEDS = (3901, 4301, 4302)
FOLDS = ('f1', 'f2', 'f3')
POLICIES = ('raw', 'nystrom', 'kahm', 'all_map', 'ordinary', 'ridge')
CERTIFIED_POLICIES = ('raw', 'nystrom', 'kahm', 'all_map')
LAMBDA_PAIRS = (('1e-6', '1/1000000'), ('1e-4', '1/10000'),
                ('1e-2', '1/100'), ('1', '1'), ('100', '100'))
ASSURANCE = 'CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES'
BUDGET = F(1, 1 << 40)
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
SCOPE = {
    'assurance': ASSURANCE,
    'reference': 'FIXED_STORED_RATIONAL_AFFINE_HEAD',
    'affine_training_optimum_certified': False,
    'native_transform_error_enclosed': False,
    'raw_input_robustness_certified': False,
    'future_response_coverage_claim': False,
    'population_risk_claim': False,
    'cryptographic_authenticity_claim': False,
}


class AuditError(AssertionError):
    """Retain scientific evidence and stop on an unsupported or changed record."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, 'tolist'):
        return canonical(value.tolist())
    if hasattr(value, 'item'):
        return canonical(value.item())
    return value


def canonical_bytes(value):
    return (json.dumps(canonical(value), sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def legacy_bytes(value):
    return json.dumps(canonical(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def legacy_digest(value):
    return hashlib.sha256(legacy_bytes(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_bytes(value))


def safe_path(relative, *, root=ROOT):
    require(type(relative) is str and bool(relative), 'Missing relative path')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts and '.' not in path.parts,
            'Unsafe evidence path: ' + relative)
    resolved = root / path
    require(not resolved.is_symlink(), 'Evidence path is a symbolic link: ' + relative)
    return resolved


def bound_ref(record, *, root=ROOT):
    require(type(record) is dict and {'path', 'sha256'} <= set(record), 'Malformed file reference')
    path = safe_path(record['path'], root=root)
    require(type(record['sha256']) is str and SHA256.fullmatch(record['sha256'])
            and sha(path) == record['sha256'], 'Changed evidence: ' + record['path'])
    if 'bytes' in record:
        require(type(record['bytes']) is int and path.stat().st_size == record['bytes'],
                'Evidence byte count differs: ' + record['path'])
    return path


def verify_bindings(bindings, *, root=ROOT):
    require(type(bindings) is dict and bool(bindings), 'Missing evidence bindings')
    for name, expected in bindings.items():
        bound_ref({'path': name, 'sha256': expected}, root=root)


def q(value, name='rational'):
    require(type(value) is str, name + ' must be an explicit canonical rational string')
    try:
        result = F(value)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise AuditError(name + ' is not rational') from error
    require(str(result) == value, name + ' is not canonical')
    return result


def binary64(value):
    require(type(value) is str, 'Missing canonical binary64 hex value')
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError) as error:
        raise AuditError('Invalid binary64 hex value') from error
    require(math.isfinite(number) and number.hex() == value, 'Noncanonical/nonfinite binary64 value')
    return number


def binary64_fraction(value):
    return F.from_float(binary64(value))


def array(value, *, dimension=None):
    value = np.asarray(value)
    require(value.dtype.kind == 'f' and value.dtype.itemsize == 8 and np.isfinite(value).all(),
            'Expected finite stored binary64 features')
    require(value.ndim in (1, 2) and value.size > 0, 'Invalid stored feature shape')
    if dimension is not None:
        require(value.shape[-1] == dimension, 'Stored feature dimension differs')
    return np.asarray(value, dtype=np.float64)


def array_digest(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(value.shape).encode('ascii'))
    result.update(value.dtype.str.encode('ascii'))
    result.update(value.tobytes())
    return result.hexdigest()


def bit_equal(left, right):
    left, right = array(left), array(right)
    return left.shape == right.shape and np.array_equal(left.view(np.uint64), right.view(np.uint64))


def maps_roster():
    return sorted(['raw_' + s for s in ('minmax01', 'standard')]
                  + [f'nystrom_{s}_w{w}' for s in ('minmax01', 'standard') for w in (8, 24)]
                  + [f'kahm_{s}_cap{c}_p{p}' for s in ('minmax01', 'standard')
                     for c in (4, 12, 20) for p in (1, 2)])


def affine_configs():
    rows = []
    for name in maps_roster():
        family, scaling = name.split('_')[:2]
        for token, penalty in LAMBDA_PAIRS:
            rows.append({'candidate_id': name + '::lambda_' + token, 'map': name,
                         'family': family, 'scaling': scaling, 'lambda_token': token,
                         'ridge_lambda': penalty, 'recipe': 'affine', 'kappa': '2'})
    return rows


def exact_risk(predictions_hex, targets):
    require(len(predictions_hex) == len(targets) and len(targets) > 0,
            'Risk requires equal nonempty prediction/target rosters')
    losses = [binary64_fraction(p) - F(y) for p, y in zip(predictions_hex, targets)]
    sse = sum((x*x for x in losses), F(0))
    mse = sse / len(losses)
    return {'n': len(losses), 'sse': str(sse), 'mse': str(mse),
            'rmse': math.sqrt(float(mse)),
            'mae': str(sum(map(abs, losses), F(0)) / len(losses))}


def exact_reference_risk(values, targets):
    require(len(values) == len(targets) and len(targets) > 0,
            'Reference risk requires equal nonempty rosters')
    losses = [F(p) - F(y) for p, y in zip(values, targets)]
    sse = sum((x*x for x in losses), F(0))
    mse = sse / len(losses)
    return {'n': len(losses), 'sse': str(sse), 'mse': str(mse),
            'rmse': math.sqrt(float(mse)),
            'mae': str(sum(map(abs, losses), F(0)) / len(losses))}


def compare_risk(record, expected, name):
    require(record == expected, 'Exact risk differs: ' + name)


def choose_pooled(fold_rows, candidates):
    """Pool sums and observation counts, never equally weight unequal folds."""
    candidates = list(candidates)
    require(len(candidates) == len(set(candidates)) and bool(candidates), 'Invalid selection pool')
    scores = []
    for cid in candidates:
        records = [fold[cid] for fold in fold_rows]
        n = sum(row['n'] for row in records)
        require(n > 0 and all(type(row['n']) is int and row['n'] > 0 for row in records),
                'Invalid development fold count')
        sse = sum((q(row['sse']) for row in records), F(0))
        require(sse >= 0, 'Negative development loss')
        scores.append({'candidate_id': cid, 'n': n, 'sse': str(sse), 'mse': str(sse / n)})
    ordered = sorted(scores, key=lambda row: (q(row['mse']), row['candidate_id']))
    return ordered[0], scores


def tolerance(raw_sse, reference_sse):
    raw, reference = F(raw_sse), F(reference_sse)
    require(raw >= 0 and reference >= 0, 'Negative primary SSE')
    return {'raw_sse': str(raw), 'reference_sse': str(reference),
            'sse_ratio': float(raw / reference) if reference else None,
            'within_five_percent': 20 * raw <= 21 * reference}


def class_id(head):
    descriptor = {'schema': 'step35-stored-feature-decoder-class-v1',
                  'feature_id': head['feature_id'],
                  'metric': 'Euclidean_distance_on_supplied_feature_coordinates',
                  'decoder_class': 'all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space',
                  'kappa': head['kappa'], 'output_interval': head['output_interval'],
                  'assurance': ASSURANCE}
    return 'class35:' + legacy_digest(descriptor)


def audit_head(head):
    expected_keys = {'schema', 'kind', 'feature_id', 'class_id', 'dimension', 'kappa',
                     'output_interval', 'assurance', 'coefficients', 'intercept'}
    require(type(head) is dict and set(head) == expected_keys, 'Affine head schema fields differ')
    require(head['schema'] == 'step35-stored-feature-head-v1' and head['kind'] == 'clipped_affine'
            and head['assurance'] == ASSURANCE and head['output_interval'] == ['0', '1']
            and q(head['kappa']) == 2, 'Affine class or scope differs')
    require(type(head['feature_id']) is str and bool(head['feature_id'])
            and head['class_id'] == class_id(head), 'Affine feature/class identity differs')
    require(type(head['dimension']) is int and 1 <= head['dimension'] <= 4096
            and type(head['coefficients']) is list and len(head['coefficients']) == head['dimension'],
            'Affine dimension differs')
    coefficients = [q(x, 'coefficient') for x in head['coefficients']]
    q(head['intercept'], 'intercept')
    require(sum((x*x for x in coefficients), F(0)) <= 4, 'Exact coefficient norm exceeds 2')
    return 'head35:' + legacy_digest(head)


def upper_sqrt(value, bits=80):
    value = F(value)
    require(value >= 0, 'Negative square-root argument')
    scale = 1 << bits
    lower_integer = math.isqrt((value.numerator * scale * scale) // value.denominator)
    lower = F(lower_integer, scale)
    upper = lower if lower*lower == value else F(lower_integer + 1, scale)
    require(lower*lower <= value <= upper*upper and upper - lower <= F(1, scale),
            'Independent rational square-root bound failed')
    return upper


def audit_affine_fit(receipt, features=None, targets=None, *, ridge_lambda,
                     feature_id=None, training_binding=None):
    head = receipt['head']
    hid = audit_head(head)
    require(receipt['schema'] == 'step35-admissible-affine-head-v1' and receipt['head_id'] == hid
            and receipt['class_id'] == head['class_id'] and receipt['feature_id'] == head['feature_id']
            and receipt['native_transform_error_enclosed'] is False and receipt['population_risk_claim'] is False,
            'Affine fit identity or scope differs')
    if feature_id is not None:
        require(head['feature_id'] == feature_id, 'Affine fit uses another representation')
    if training_binding is not None:
        require(receipt['training_binding'] == training_binding, 'Affine training binding differs')
    projection = receipt['projection']
    source = [q(x) for x in projection['source_coefficients']]
    source_intercept = q(projection['source_intercept'])
    require(len(source) == head['dimension'], 'Ridge source dimension differs')
    require(all(F.from_float(float(x)) == x for x in source + [source_intercept]),
            'Ridge proposals are not exact stored binary64 values')
    squared = sum((x*x for x in source), F(0))
    factor = F(1) if squared <= 4 else F(2) / upper_sqrt(squared)
    transformed = [factor*x for x in source]
    require(q(projection['source_squared_norm']) == squared and q(projection['factor']) == factor
            and projection['projected'] is (factor != 1)
            and [q(x) for x in head['coefficients']] == transformed,
            'Exact projection differs from the frozen P1 rule')
    centers = [q(x) for x in projection['center_features']]
    center_target = q(projection['center_target'])
    intercept = center_target - sum((b*z for b, z in zip(transformed, centers)), F(0))
    require(len(centers) == len(source) and projection['intercept_recentered'] is True
            and q(head['intercept']) == intercept, 'Exact R1 intercept recentering differs')
    norm2 = sum((x*x for x in transformed), F(0))
    require(receipt['membership'] == {
        'status': 'VERIFIED_EXACT_AFFINE_CLASS_MEMBERSHIP',
        'coefficient_squared_norm': str(norm2), 'kappa_squared': '4',
        'proof': 'Cauchy-Schwarz followed by nonexpansive range clipping.'},
        'Affine membership receipt differs')
    if features is not None:
        z = array(features)
        require(z.ndim == 2 and z.shape[1] == len(source) and len(z) == len(targets) > 0,
                'Affine training feature/target roster differs')
        ys = [F(y) for y in targets]
        expected_centers = [sum((F.from_float(float(v)) for v in z[:, j]), F(0)) / len(z)
                            for j in range(z.shape[1])]
        require(centers == expected_centers and center_target == sum(ys, F(0)) / len(ys),
                'Affine exact centers use another training cohort')
        yf = np.asarray([float(y) for y in ys], dtype=np.float64)
        zmean, ymean = z.mean(axis=0), yf.mean()
        xc, yc = z - zmean, yf - ymean
        penalty = len(z) * float(F(ridge_lambda))
        require(penalty > 0 and math.isfinite(penalty), 'Invalid ridge penalty')
        matrix = xc.T @ xc + penalty * np.eye(z.shape[1])
        rhs = xc.T @ yc
        beta = np.linalg.solve(matrix, rhs)
        b = float(ymean - zmean @ beta)
        require(source == [F.from_float(float(x)) for x in beta] and source_intercept == F.from_float(b),
                'Floating ridge proposal differs from prescribed training-only solve')
        residual = matrix @ beta - rhs
        denominator = np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(beta, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf)
        backward = float(np.linalg.norm(residual, ord=np.inf) / denominator) if denominator else 0.0
        require(math.isfinite(backward) and backward <= 256 * z.shape[1] * np.finfo(float).eps,
                'Ridge solve has excessive scaled residual')
    return {'head_id': hid, 'projected': factor != 1, 'coefficient_squared_norm': str(norm2),
            'training_optimizer_certified': False}


def audit_deployment(deployment, *, head=None, candidate_id=None, map_sha256=None,
                     map_bytes=None, normalization=None):
    keys = {'schema', 'head', 'head_id', 'candidate_id', 'target_min', 'span', 'map_sha256',
            'map_bytes', 'normalized_error_budget', 'scope', 'deployment_id'}
    require(type(deployment) is dict and set(deployment) == keys
            and deployment['schema'] == 'step41-affine-deployment-v1', 'Deployment schema differs')
    hid = audit_head(deployment['head'])
    require(deployment['head_id'] == hid and deployment['scope'] == SCOPE
            and deployment['normalized_error_budget'] == str(BUDGET), 'Deployment scope/accuracy budget differs')
    body = {k: v for k, v in deployment.items() if k != 'deployment_id'}
    require(deployment['deployment_id'] == 'deploy41:' + legacy_digest(body), 'Deployment identity differs')
    require(type(deployment['candidate_id']) is str and bool(deployment['candidate_id'])
            and type(deployment['map_bytes']) is int and deployment['map_bytes'] > 0
            and type(deployment['map_sha256']) is str and SHA256.fullmatch(deployment['map_sha256']),
            'Deployment map binding differs')
    minimum, span = q(deployment['target_min']), q(deployment['span'])
    require(span > 0 and float(span) > 0 and all(math.isfinite(float(x)) for x in (minimum, span, minimum + span)),
            'Deployment target normalization is invalid')
    for name, expected in [('head', head), ('candidate_id', candidate_id), ('map_sha256', map_sha256), ('map_bytes', map_bytes)]:
        if expected is not None:
            require(deployment[name] == expected, 'Deployment trusted context differs: ' + name)
    if normalization is not None:
        require(deployment['target_min'] == normalization['target_min'] and deployment['span'] == normalization['span'],
                'Deployment target normalization uses another cohort')
    return deployment['deployment_id']


def feature_bytes(vector):
    vector = array(vector)
    require(vector.ndim == 1, 'Certificate feature input must be a vector')
    return b'step41-feature-vector-v1\0' + struct.pack('>I', len(vector)) + b''.join(struct.pack('>d', float(x)) for x in vector)


def feature_digest(vector):
    return hashlib.sha256(feature_bytes(vector)).hexdigest()


def exact_prediction(head, vector, minimum, span):
    """Independent common-denominator integer dot product, exact clipping/scale."""
    vector = array(vector, dimension=head['dimension'])
    require(vector.ndim == 1, 'Exact prediction requires one query')
    coefficients = [q(x) for x in head['coefficients']]
    intercept = q(head['intercept'])
    denominator = math.lcm(*(x.denominator for x in coefficients + [intercept]))
    ratios = [float(value).as_integer_ratio() for value in vector]
    binary_denominator = max(den for _, den in ratios)
    require(all(binary_denominator % den == 0 for _, den in ratios), 'Features are not dyadic')
    numerator = intercept.numerator * (denominator // intercept.denominator) * binary_denominator
    numerator += sum(b.numerator * (denominator // b.denominator) * n * (binary_denominator // d)
                     for b, (n, d) in zip(coefficients, ratios))
    total_denominator = denominator * binary_denominator
    normalized = F(0) if numerator <= 0 else F(1) if numerator >= total_denominator else F(numerator, total_denominator)
    return F(minimum) + F(span) * normalized


def operational_prediction(deployment, vector):
    head = deployment['head']
    vector = array(vector, dimension=head['dimension'])
    beta = np.asarray([float(q(x)) for x in head['coefficients']], dtype=np.float64)
    raw = float(np.dot(beta, vector)) + float(q(head['intercept']))
    require(math.isfinite(raw), 'Nonfinite affine intermediate')
    value = float(q(deployment['target_min'])) + float(q(deployment['span'])) * min(1.0, max(0.0, raw))
    require(math.isfinite(value), 'Nonfinite physical prediction')
    return float(value)


def tight_hull(value):
    value = F(value)
    nearest = float(value)
    require(math.isfinite(nearest), 'Exact reference is outside finite binary64 range')
    rounded = F.from_float(nearest)
    low = math.nextafter(nearest, -math.inf) if rounded > value else nearest
    high = math.nextafter(nearest, math.inf) if rounded < value else nearest
    if value == 0:
        low = high = 0.0
    return [low.hex(), high.hex()]


def tight_upper(value):
    value = F(value)
    require(value >= 0, 'Negative error magnitude')
    nearest = float(value)
    upper = math.nextafter(nearest, math.inf) if F.from_float(nearest) < value else nearest
    require(math.isfinite(upper), 'Error bound is outside finite binary64 range')
    return (0.0 if value == 0 else upper).hex()


def expected_packet(deployment, vector, *, query_id, reported_hex):
    reference = exact_prediction(deployment['head'], vector, deployment['target_min'], deployment['span'])
    error = abs(binary64_fraction(reported_hex) - reference)
    return {'schema': 'step41-affine-query-certificate-v1',
            'deployment_id': deployment['deployment_id'], 'query_id': query_id,
            'feature_sha256': feature_digest(vector), 'reported_prediction_hex': reported_hex,
            'reference_interval_hex': tight_hull(reference), 'absolute_error_upper_hex': tight_upper(error)}


def binding_only(deployment, vector, packet, *, query_id, reported_hex):
    """Explicitly limited identity check; not numerical validation/authentication."""
    fields = {'schema', 'deployment_id', 'query_id', 'feature_sha256', 'reported_prediction_hex',
              'reference_interval_hex', 'absolute_error_upper_hex'}
    return (type(packet) is dict and set(packet) == fields
            and packet.get('schema') == 'step41-affine-query-certificate-v1'
            and packet.get('deployment_id') == deployment['deployment_id']
            and packet.get('query_id') == query_id
            and packet.get('feature_sha256') == feature_digest(vector)
            and packet.get('reported_prediction_hex') == reported_hex)


def audit_packet(deployment, vector, packet, *, query_id, reported_hex, check_static=True):
    if check_static:
        audit_deployment(deployment)
    wanted = expected_packet(deployment, vector, query_id=query_id, reported_hex=reported_hex)
    require(packet == wanted, 'Packet identity or tight arithmetic claim differs')
    reference = exact_prediction(deployment['head'], vector, deployment['target_min'], deployment['span'])
    error = abs(binary64_fraction(reported_hex) - reference)
    budget = q(deployment['span']) * BUDGET
    return {'accepted': error <= budget, 'exact_physical': str(reference),
            'absolute_error': str(error), 'normalized_error': str(error / q(deployment['span'])),
            'physical_budget': str(budget), 'binding_only_accepted': True,
            'packet_json_bytes': len(legacy_bytes(packet)), 'feature_payload_bytes': len(feature_bytes(vector))}


def expected_verification(deployment, vector, *, query_id, reported_hex):
    reference = exact_prediction(deployment['head'], vector, deployment['target_min'], deployment['span'])
    reported = binary64_fraction(reported_hex)
    error = abs(reported - reference)
    budget = q(deployment['span']) * BUDGET
    if error > budget:
        return {'status': 'REJECTED_OVER_BUDGET',
                'reason': 'Reported prediction exceeds the frozen discrepancy budget'}
    return {'schema': 'step41-affine-query-verification-v1', 'status': 'PASS',
            'deployment_id': deployment['deployment_id'], 'query_id': query_id,
            'feature_sha256': feature_digest(vector), 'exact_reference_physical': str(reference),
            'exact_reported_physical': str(reported), 'absolute_error_exact': str(error),
            'normalized_error_exact': str(error / q(deployment['span'])),
            'physical_error_budget': str(budget), 'normalized_error_budget': str(BUDGET),
            'verified_exact_error': True, 'accepted_prediction_accuracy': True,
            'target_label_accessed': False, 'scope': dict(SCOPE)}


def float32_hex(reported_hex):
    rounded = float(np.float32(binary64(reported_hex)))
    require(math.isfinite(rounded), 'Predeclared float32 output path became nonfinite')
    return rounded.hex()


def binding_record(accepted, reason=None):
    require(accepted == (reason is None), 'Invalid binding-control expected status')
    return {'status': 'PASS' if accepted else 'REJECTED', 'accepted': accepted, 'reason': reason,
            'numerical_discrepancy_checked': False, 'authentication_claim': False,
            'scope': 'Schema/deployment/query/feature/reported-value consistency only'}


def audit_certificate_record(record, deployment, vector, *, query_id, reported_hex):
    checked = audit_packet(deployment, vector, record['certificate'], query_id=query_id,
                           reported_hex=reported_hex, check_static=False)
    expected = {'reported_physical_hex': reported_hex, 'exact_physical': checked['exact_physical'],
                'absolute_physical_error': checked['absolute_error'],
                'physical_error_budget': checked['physical_budget'],
                'accepted': checked['accepted'], 'expected_acceptance': checked['accepted'],
                'certificate': expected_packet(deployment, vector, query_id=query_id, reported_hex=reported_hex),
                'verification': expected_verification(deployment, vector, query_id=query_id, reported_hex=reported_hex),
                'binding_only': binding_record(True),
                'packet_json_bytes': checked['packet_json_bytes'], 'feature_payload_bytes': checked['feature_payload_bytes']}
    require(record == expected, 'Certificate wrapper, acceptance, or checker result differs')
    return checked


def audit_output_assay(record, deployment, vector, *, query_id, reported_hex):
    require(set(record) == {'schema', 'query_id', 'target_label_accessed', 'quantization_changed_reported_value',
                           'cases', 'scope'} and record['schema'] == 'step43-numerical-output-assay-v1'
            and record['query_id'] == query_id and record['target_label_accessed'] is False
            and record['scope'] == 'Numerical-contract enforcement beyond identity-binding consistency; no statistical-accuracy or authentication claim',
            'Numerical assay scope or schema differs')
    rounded = float32_hex(reported_hex)
    require(record['quantization_changed_reported_value'] is (rounded != reported_hex)
            and set(record['cases']) == {'binary64', 'physical_output_float32'}, 'Float32 assay case roster differs')
    honest = audit_certificate_record(record['cases']['binary64'], deployment, vector,
                                     query_id=query_id, reported_hex=reported_hex)
    quantized = audit_certificate_record(record['cases']['physical_output_float32'], deployment, vector,
                                        query_id=query_id, reported_hex=rounded)
    return {'binary64': honest, 'physical_output_float32': quantized,
            'quantization_changed_reported_value': rounded != reported_hex}


def audit_misbinding(record, deployment, vector, stale_vector, *, query_id, reported_hex):
    original = expected_packet(deployment, vector, query_id=query_id, reported_hex=reported_hex)
    wrong_query = dict(original, query_id=query_id + '::step43_wrong_query')
    prefix, suffix = deployment['deployment_id'].split(':', 1)
    wrong_id = prefix + ':' + ('0' if suffix[0] != '0' else '1') + suffix[1:]
    wrong_deployment = dict(original, deployment_id=wrong_id)
    stale_hash = feature_digest(stale_vector)
    differs = stale_hash != original['feature_sha256']
    variants = [('wrong_query', wrong_query, 'Certificate query ID differs', 'query_id'),
                ('wrong_deployment', wrong_deployment, 'Certificate deployment differs', 'deployment_id')]
    if differs:
        variants.append(('stale_feature', expected_packet(deployment, stale_vector, query_id=query_id,
                                                         reported_hex=reported_hex),
                         'Certificate feature bytes differ', 'feature_bytes'))
    rows = []
    for name, packet, reason, binding_reason in variants:
        require(not binding_only(deployment, vector, packet, query_id=query_id, reported_hex=reported_hex),
                'Misbinding fixture unexpectedly matches the trusted context')
        rows.append({'challenge': name, 'status': 'REJECTED_AS_EXPECTED', 'certificate': packet,
                     'full_rejection': reason, 'binding_only': binding_record(False, binding_reason)})
    if not differs:
        rows.append({'challenge': 'stale_feature', 'status': 'NOT_APPLICABLE_EQUAL_FEATURE_BYTES',
                     'certificate': None, 'full_rejection': None, 'binding_only': None})
    expected = {'schema': 'step43-misbinding-assay-v1', 'query_id': query_id,
                'feature_sha256': original['feature_sha256'], 'stale_feature_sha256': stale_hash,
                'stale_features_differ': differs, 'challenges': rows,
                'target_label_accessed': False, 'authentication_claim': False,
                'scope': 'Fixed model/query/feature content-binding contract checks'}
    require(record == expected, 'Misbinding challenge definition, applicability or rejection differs')
    return {'challenges': 3, 'applicable': 3 if differs else 2,
            'rejected_as_expected': 3 if differs else 2, 'stale_features_differ': differs}


def linear_quantile(values, probability):
    require(bool(values) and 0 <= probability <= 1, 'Invalid timing quantile')
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def validate_events(path):
    rows, previous, prior_utc = [], None, None
    for index, line in enumerate(Path(path).read_bytes().splitlines(keepends=True)):
        record = json.loads(line)
        require(line == canonical_bytes(record) and record['index'] == index
                and record['previous_event_sha256'] == previous, 'Event canonical hash chain differs')
        stamp = datetime.fromisoformat(record['utc'])
        require(stamp.tzinfo is not None and (prior_utc is None or prior_utc <= stamp),
                'Event timestamps are not ordered/timezone-aware')
        require(type(record['monotonic_ns']) is int and record['monotonic_ns'] >= 0,
                'Event monotonic clock is invalid')
        previous, prior_utc = hashlib.sha256(line).hexdigest(), stamp
        rows.append(record)
    return rows


def publish_gate(path, kind, bindings, **fields):
    protocol_seal = ROOT / 'protocol_seal43.json'
    record = {'schema': 'step43-independent-audit-v1', 'status': 'PASS', 'phase': kind,
              'utc': datetime.now(timezone.utc).isoformat(),
              'protocol_seal_sha256': sha(protocol_seal),
              'auditor_sha256': sha(Path(__file__)),
              'source_bindings': {str(Path(p).relative_to(ROOT)): sha(p) for p in sorted(set(map(Path, bindings)))},
              'scientific_candidate_searches_by_auditor': 0,
              'fresh_candidate_fits_by_auditor': 0,
              'ideal_feature_transform_certified': False,
              'population_risk_claim': False,
              **fields}
    save(path, record)
    return record


def csv_fields(raw, separator):
    """Independent byte CSV parser: no measurement string/numeric decoding."""
    delimiter = separator.encode('ascii')[0] if isinstance(separator, str) else separator[0]
    raw = raw.rstrip(b'\r\n')
    if delimiter == ord(';'):
        require(b'"' not in raw, 'Unexpected quoted Household field')
        return raw.split(bytes([delimiter]))
    result, current = [], bytearray()
    quoted = ended = False
    pos = 0
    while pos < len(raw):
        char = raw[pos]
        if quoted:
            if char == ord('"'):
                if pos + 1 < len(raw) and raw[pos + 1] == ord('"'):
                    current.append(char)
                    pos += 2
                    continue
                quoted, ended = False, True
            else:
                current.append(char)
        elif char == delimiter:
            result.append(bytes(current))
            current, ended = bytearray(), False
        elif char == ord('"') and not current and not ended:
            quoted = True
        else:
            require(char != ord('"') and not ended, 'Invalid source CSV quoting')
            current.append(char)
        pos += 1
    require(not quoted, 'Unclosed source CSV field')
    result.append(bytes(current))
    return result


@lru_cache(maxsize=4096)
def household_date(raw):
    parts = raw.strip().split(b'/')
    require(len(parts) == 3 and all(x.isdigit() for x in parts), 'Invalid Household date')
    day, month, year = map(int, parts)
    return year, month, day


def timestamp(spec, fields):
    kind = spec['timestamp_kind']
    if kind == 'household_date_time':
        parts = fields[1].strip().split(b':')
        require(len(parts) == 3 and all(x.isdigit() for x in parts), 'Invalid Household time')
        stamp = datetime(*household_date(fields[0]), *map(int, parts))
    else:
        text = ' '.join(fields[index].decode('ascii').strip() for index in spec['timestamp_indices'])
        if kind == 'iso_local':
            stamp = datetime.fromisoformat(text)
            require(stamp.isoformat(sep=' ', timespec='seconds') == text, 'Noncanonical Metro timestamp')
        else:
            require(kind == 'jena_date_time', 'Unknown source timestamp convention')
            stamp = None
            for fmt in spec['timestamp_formats']:
                try:
                    stamp = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            require(stamp is not None, 'Invalid Jena timestamp')
    require(stamp.tzinfo is None and stamp.second == 0 and stamp.minute % spec['cadence_minutes'] == 0,
            'Source cadence or timezone differs')
    return stamp


def iso(value):
    return value.isoformat(timespec='seconds')


def query_lags(query):
    target = datetime.fromisoformat(query['target_timestamp'])
    origin = target - timedelta(hours=24)
    require(query['origin_timestamp'] == iso(origin), 'Forecast origin is not 24 hours before target')
    return [iso(origin - timedelta(hours=23-j)) for j in range(24)]


def calendar_queries(task, index, start, stop):
    result, excluded = [], []
    date = datetime.fromisoformat(start).replace(hour=23)
    end = datetime.fromisoformat(stop).replace(hour=23)
    while date < end:
        item = {'id': task + ':' + iso(date), 'task': task, 'target_timestamp': iso(date),
                'origin_timestamp': iso(date - timedelta(hours=24))}
        stamps = [item['target_timestamp'], *query_lags(item)]
        absent = [x for x in stamps if x not in index]
        missing = [x for x in stamps if x in index and index[x]['declared_target_missing']]
        if absent or missing:
            excluded.append({'target_timestamp': item['target_timestamp'],
                             'absent_timestamps': absent, 'declared_missing_timestamps': missing})
        else:
            result.append(item)
        date += timedelta(days=1)
    return result, excluded


def source_payloads(task, manifest, *, cache=None):
    if cache is not None and task in cache:
        return cache[task]
    result = {}
    for item in manifest['tasks'][task]['archives']:
        path = bound_ref(item)
        with zipfile.ZipFile(path) as archive:
            require(archive.namelist() == [item['member']], 'Source archive membership differs')
            member = archive.read(item['member'])
        require(len(member) == item['member_bytes'] and hashlib.sha256(member).hexdigest() == item['member_sha256'],
                'Source member bytes differ')
        payload = gzip.decompress(member) if item['nested_gzip'] else member
        require(len(payload) == item['payload_bytes'] and hashlib.sha256(payload).hexdigest() == item['payload_sha256'],
                'Source decompressed payload differs')
        require(item['source_id'] not in result, 'Duplicate source identifier')
        result[item['source_id']] = payload
    if cache is not None:
        cache[task] = result
    return result


def audit_calendar(task, manifest, design):
    """Reconstruct timestamp/missing-marker projection without numeric access."""
    spec = manifest['tasks'][task]['spec']
    source = read(bound_ref(manifest['tasks'][task]['source']))
    saved = read(bound_ref(manifest['tasks'][task]['calendar']))
    require(source['source_spec'] == spec and source['archives'] == manifest['tasks'][task]['archives'],
            'Source specification or archive roster differs')
    payloads = source_payloads(task, manifest)
    lower = datetime.fromisoformat(design['tasks'][task]['development_start']) - timedelta(days=2)
    upper = datetime.fromisoformat(design['tasks'][task]['assessment_stop_exclusive'])
    headers = None
    groups = {}
    first = previous = None
    native_rows = unique_native = duplicate_excess = 0
    years, missing_counts, member_stats = Counter(), Counter(), []
    markers = {x.encode('ascii') for x in spec['missing_markers']}
    for item in source['archives']:
        raw = payloads[item['source_id']]
        stream = io.BytesIO(raw)
        heading = stream.readline().removeprefix(b'\xef\xbb\xbf')
        current = [x.decode(spec['header_encoding']) for x in csv_fields(heading, spec['separator'])]
        if headers is None:
            headers = current
        require(headers == current and headers[spec['target_index']] == spec['target_header']
                and [headers[i] for i in spec['timestamp_indices']] == spec['timestamp_headers'],
                'Source header columns differ')
        count = 0
        for line_number, raw_line in enumerate(stream, 2):
            stop = stream.tell()
            fields = csv_fields(raw_line, spec['separator'])
            require(len(fields) == len(headers), 'Source field count differs')
            stamp = timestamp(spec, fields)
            require(previous is None or previous <= stamp, 'Source dates are not ordered')
            native_rows += 1
            count += 1
            if stamp == previous:
                duplicate_excess += 1
            else:
                unique_native += 1
                years[str(stamp.year)] += 1
            first = stamp if first is None else first
            previous = stamp
            if stamp.minute or not lower <= stamp < upper:
                continue
            missing = fields[spec['target_index']].strip(b' \t') in markers if markers else False
            if markers:
                missing_counts['declared_missing' if missing else 'not_declared_missing'] += 1
            key = iso(stamp)
            row = groups.setdefault(key, {'timestamp': key, 'records': [], 'declared_target_missing': False})
            row['records'].append({'source_id': item['source_id'], 'source_line': line_number,
                                   'byte_start': stop - len(raw_line), 'byte_stop': stop})
            row['declared_target_missing'] = row['declared_target_missing'] or missing
        member_stats.append({'source_id': item['source_id'], 'payload_sha256': hashlib.sha256(raw).hexdigest(),
                             'payload_bytes': len(raw), 'timestamp_rows': count})
    expected = {'schema': 'step43-calendar-byte-projection-v1', 'task': task, 'headers': headers,
                'source_spec': spec, 'rows': list(groups.values()), 'member_stats': member_stats,
                'timestamp_rows': native_rows, 'unique_native_timestamps': unique_native,
                'duplicate_record_excess': duplicate_excess, 'source_timestamp_counts_by_year': dict(sorted(years.items())),
                'first_source_timestamp': iso(first), 'last_source_timestamp': iso(previous),
                'projected_hourly_timestamps': len(groups), 'projected_start': iso(lower),
                'projected_stop_exclusive': iso(upper), 'parsed_data_columns': spec['timestamp_indices'],
                'target_values_decoded_or_parsed': 0, 'other_measurement_values_decoded_or_parsed': 0,
                'declared_missing_markers': spec['missing_markers'], 'missing_status_inspected': bool(markers),
                'missing_status_counts': dict(missing_counts), 'duplicate_target_agreement_examined': False}
    require(saved == expected, 'Independent timestamp-only calendar differs: ' + task)
    task_record, task_design = manifest['tasks'][task], design['tasks'][task]
    for stage, month in zip(FOLDS, task_design['validation_months']):
        record = task_record['stages'][stage]
        beginning = month + '-01'
        stamp = datetime.fromisoformat(beginning)
        following = datetime(stamp.year + (stamp.month == 12), stamp.month % 12 + 1, 1).date().isoformat()
        train, train_excluded = calendar_queries(task, groups, task_design['development_start'], beginning)
        validation, validation_excluded = calendar_queries(task, groups, beginning, following)
        require(record == {'train_start': task_design['development_start'], 'train_stop_exclusive': beginning,
                           'validation_start': beginning, 'validation_stop_exclusive': following,
                           'train': train, 'validation': validation,
                           'counts': {'train': len(train), 'validation': len(validation)},
                           'excluded': {'train': train_excluded, 'validation': validation_excluded}},
                'Independent development date roster differs')
        require(len(train) >= 32 and len(validation) >= 7, 'Frozen minimum development cohort unmet')
    final, excluded = calendar_queries(task, groups, task_design['development_start'], task_design['assessment_start'])
    require(task_record['stages']['final'] == {
        'train_start': task_design['development_start'], 'train_stop_exclusive': task_design['assessment_start'],
        'train': final, 'validation': [], 'counts': {'train': len(final), 'validation': 0},
        'excluded': {'train': excluded, 'validation': []}}, 'Independent final training roster differs')
    assessment, excluded = calendar_queries(task, groups, task_design['assessment_start'], task_design['assessment_stop_exclusive'])
    require(task_record['assessment'] == assessment and task_record['assessment_excluded'] == excluded
            and len(final) >= 32 and len(assessment) >= 30, 'Independent assessment date roster differs')
    require(task_record['counts'] == {'stages': {stage: task_record['stages'][stage]['counts'] for stage in (*FOLDS, 'final')},
                                      'assessment': len(assessment)}, 'Calendar counts differ')
    return {'source_timestamp_rows_checked': native_rows, 'projected_hourly_timestamps': len(groups),
            'assessment_queries': len(assessment), 'numeric_measurement_values_decoded': 0}


def audit_exposure(manifest, design):
    evidence = read(bound_ref(manifest['exposure']))
    require(evidence['status'] == 'PASS' and evidence['absolute_historical_never_seen_claim'] is False
            and evidence['new_domains_claim'] is False and evidence['new_measurement_values_decoded'] == 0,
            'Freshness scope is overstated')
    bound_ref(evidence['search'])
    for proof in evidence['proof_files']:
        bound_ref(proof)
    by_domain = {'household': set(), 'jena': set(), 'metro': set()}
    receipt_count = 0
    for trace in evidence['traces']:
        old = read(bound_ref(trace))
        union = set()
        for receipt in old['receipts']:
            require(SHA256.fullmatch(receipt['prior_sha256']), 'Invalid retained prior receipt hash')
            union.update(receipt['access']['requested_timestamps'])
        require(old['unique_requested_timestamps'] == sorted(union)
                and old['unique_timestamp_count'] == len(union) and old['receipt_count'] == len(old['receipts'])
                and old['numeric_values_copied'] is False and old['minimum'] == min(union)
                and old['maximum'] == max(union), 'Prior numeric-authorization union differs')
        require(old['year_counts'] == dict(sorted(Counter(x[:4] for x in union).items())), 'Prior authorization year counts differ')
        domain = old['task'].split('_', 1)[0]
        by_domain[domain].update(union)
        receipt_count += len(old['receipts'])
    for task in TASKS:
        proposed = set()
        spec = design['tasks'][task]
        target = datetime.fromisoformat(spec['assessment_start']).replace(hour=23)
        end = datetime.fromisoformat(spec['assessment_stop_exclusive']).replace(hour=23)
        while target < end:
            item = {'target_timestamp': iso(target), 'origin_timestamp': iso(target - timedelta(hours=24))}
            proposed.add(item['target_timestamp'])
            proposed.update(query_lags(item))
            target += timedelta(days=1)
        previous = by_domain[task.split('_', 1)[0]]
        require(not proposed & previous, 'Fresh assessment overlaps prior numeric authorization')
        expected = {'assessment_and_history_timestamp_count': len(proposed),
                    'prior_numeric_timestamp_count': len(previous), 'overlap_count': 0,
                    'prospective_timestamps_sha256': hashlib.sha256(canonical_bytes(sorted(proposed))).hexdigest(),
                    'prior_timestamps_sha256': hashlib.sha256(canonical_bytes(sorted(previous))).hexdigest(),
                    'freshness_scope': spec['freshness_scope']}
        require(evidence['disjoint_assessment'][task] == manifest['tasks'][task]['freshness'] == expected,
                'Prior-exposure disjointness record differs')
    return {'prior_authorization_receipts_checked': receipt_count, 'new_assessment_periods_disjoint_from_retained_access': True}


def protocol_check(*, deep=True):
    design, design_seal = read(ROOT / 'design43.json'), read(ROOT / 'design_seal43.json')
    require(design_seal['schema'] == 'step43-preaccess-design-seal-v1'
            and design_seal['design_sha256'] == sha(ROOT / 'design43.json')
            and design_seal['measurement_values_decoded_or_parsed'] == 0
            and design_seal['scientific_fits'] == 0, 'Preaccess design seal differs')
    seal = read(ROOT / 'protocol_seal43.json')
    require(seal['schema'] == 'step43-protocol-source-seal-v1', 'Final protocol seal schema differs')
    if deep:
        verify_bindings(seal['bound_files'])
    protocol = read(ROOT / 'protocol43.json')
    require(seal['bound_files']['protocol43.json'] == sha(ROOT / 'protocol43.json')
            and seal['bound_files']['design43.json'] == sha(ROOT / 'design43.json')
            and seal['bound_files']['design_seal43.json'] == sha(ROOT / 'design_seal43.json'),
            'Final freeze does not bind the fixed preaccess design')
    require(design['schema'] == 'step43-preaccess-design-v1' and set(design['tasks']) == set(TASKS)
            and design['method']['seeds'] == list(SEEDS) and design['method']['primary_seed'] == 3901
            and design['method']['ridge_lambdas'] == [x[1] for x in LAMBDA_PAIRS]
            and design['method']['affine_candidates_per_stage'] == 90
            and design['method']['baseline_candidates_per_development_stage'] == 44
            and design['access_and_execution']['producer_grid_runs'] == 1
            and design['access_and_execution']['outcome_based_reruns'] is False,
            'Frozen scientific panel or single-execution rule differs')
    manifest = read(ROOT / 'inputs/data43/manifest43.json')
    require(manifest['schema'] == 'step43-data-manifest-v1' and manifest['task_order'] == list(TASKS)
            and manifest['stages'] == [*FOLDS, 'final'] and manifest['measurement_values_decoded_or_parsed'] == 0
            and manifest['design_sha256'] == sha(ROOT / 'design43.json')
            and manifest['design_seal_sha256'] == sha(ROOT / 'design_seal43.json'), 'Input manifest/freeze ancestry differs')
    bound_ref(manifest['prepare_source'])
    require(protocol['schema'] == 'step43-selected-policy-transfer-protocol-v1' and protocol['design'] == design
            and protocol['tasks'] == list(TASKS) and protocol['stages'] == [*FOLDS, 'final']
            and protocol['seeds'] == list(SEEDS) and protocol['policies'] == list(POLICIES)
            and protocol['primary_seed'] == 3901 and protocol['primary_policy'] == 'raw'
            and protocol['maps'] == maps_roster() and protocol['affine_configurations'] == affine_configs()
            and protocol['baseline_configurations'] == {str(seed): baseline_configs(seed) for seed in SEEDS}
            and protocol['representations'] == REPRESENTATION_CONFIG,
            'Final protocol differs from independently expanded design/configuration rosters')
    require(protocol['data_manifest'] == reference(ROOT / 'inputs/data43/manifest43.json')
            and protocol['certificate']['normalized_budget'] == str(BUDGET)
            and protocol['certificate']['primary_requires_all_raw_seed3901_packets_accepted'] is True
            and protocol['execution']['producer_grid_runs'] == 1
            and protocol['execution']['outcome_based_reruns'] == 0
            and protocol['execution']['numerical_threads'] == 1
            and seal['frozen_before_new_numeric_measurement_access'] is True
            and seal['public_measurement_values_decoded_or_parsed'] == seal['scientific_fits'] == seal['assessment_outcomes_examined'] == 0,
            'Final protocol budget/execution/access claims differ')
    return protocol, manifest, design, seal


def preflight():
    protocol, manifest, design, seal = protocol_check()
    require(not (ROOT / 'output/data_access43.jsonl').exists(), 'Numeric data access preceded independent preflight')
    require(not (ROOT / 'output/selected_roster43.json').exists()
            and not (ROOT / 'output/assessment').exists(), 'Scientific selection/assessment preceded preflight')
    checks = {}
    for task in TASKS:
        checks[task] = audit_calendar(task, manifest, design)
    exposure = audit_exposure(manifest, design)
    inherited = read(ROOT / 'analysis/inherited_method_provenance43.json')
    require(inherited['status'] == 'PASS' and inherited['byte_identical'] is True
            and inherited['files_count'] == len(inherited['files']) == 19
            and inherited['new_measurement_values_read'] is False and inherited['real_model_fits'] == 0,
            'Inherited method provenance differs')
    for row in inherited['files']:
        bound_ref(row)
    artificial_paths = []
    for path in sorted((ROOT / 'analysis').glob('*checks43.json')):
        value = read(path)
        require(value['status'] == 'PASS', 'Artificial check is not PASS: ' + path.name)
        artificial_paths.append(path)
    own = read(ROOT / 'analysis/audit_artificial_checks43.json')
    require(own['auditor_sha256'] == sha(Path(__file__)) and own['test_sha256'] == sha(ROOT / 'source/test_audit43.py')
            and own['new_measurement_values_read'] == own['real_model_fits'] == 0,
            'Independent auditor artificial check binds stale source')
    bindings = [ROOT / 'protocol43.json', ROOT / 'protocol_seal43.json', ROOT / 'design43.json',
                ROOT / 'design_seal43.json', ROOT / 'inputs/data43/manifest43.json',
                ROOT / 'analysis/inherited_method_provenance43.json', *artificial_paths]
    bindings += [safe_path(x) for x in seal['bound_files']]
    return publish_gate(ROOT / 'audit/preflight43.json', 'preflight', bindings,
                        task_calendar_checks=checks, **exposure,
                        protocol_bound_files=len(seal['bound_files']), inherited_files_checked=19,
                        new_measurement_values_decoded_or_parsed=0)


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def reference(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


_GATE_CACHE = set()


def gate(name, **fields):
    value = read(ROOT / name)
    require(value.get('status') == 'PASS' and value.get('protocol_seal_sha256') == sha(ROOT / 'protocol_seal43.json')
            and value.get('auditor_sha256') == sha(Path(__file__))
            and all(value.get(k) == v for k, v in fields.items()), 'Required independent gate differs: ' + name)
    key = (name, sha(ROOT / name))
    if key not in _GATE_CACHE:
        verify_bindings(value['source_bindings'])
        _GATE_CACHE.add(key)
    return value


@lru_cache(maxsize=3)
def baseline_configs(seed):
    """Independent expansion of the 44 declared conventional settings."""
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
    rows = [{'token': 'persistence', 'kind': 'persistence', 'parameters': {}},
            {'token': 'training_mean', 'kind': 'mean', 'parameters': {}}]
    for token, penalty in zip(('1e6', '1e4', '1e2', '1', '100'), (x[1] for x in LAMBDA_PAIRS)):
        rows.append({'token': 'ridge_' + token, 'kind': 'ridge', 'ridge_lambda': penalty,
                     'parameters': {'fit_intercept': True, 'solver': 'cholesky'}})
    for nodes, leaf, l2 in [(a, b, 1) for a in (7, 15, 31) for b in (5, 20)] + [(15, b, 10) for b in (5, 20)]:
        params = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=nodes,
                 min_samples_leaf=leaf, l2_regularization=l2, early_stopping=False, random_state=seed).get_params()
        rows.append({'token': f'hgb_nodes{nodes}_leaf{leaf}_l2{l2}', 'kind': 'hgb', 'parameters': params})
    settings = [(c, e, 'scale') for c in (1, 10, 100) for e in (.01, .1)] + [(c, .05, g) for c in (1, 10) for g in (.01, .1)]
    for index, (c, e, gamma) in enumerate(settings):
        rows.append({'token': f'svr_{index:02d}', 'kind': 'svr', 'parameters': SVR(C=c, epsilon=e, gamma=gamma,
            kernel='rbf', tol=1e-5, cache_size=256, max_iter=100000).get_params()})
    for index, (alpha, gamma) in enumerate((a, g) for a in (.01, 1, 100) for g in (.01, .1, 1)):
        rows.append({'token': f'krr_{index:02d}', 'kind': 'krr', 'parameters': KernelRidge(alpha=alpha,
                     gamma=gamma, kernel='rbf').get_params(), 'target_centering': True})
    for kind, cls, leaves in [('extra_trees', ExtraTreesRegressor, (1, 5, 20)),
                              ('random_forest', RandomForestRegressor, (1, 5))]:
        for leaf in leaves:
            for fraction in (1., .5):
                rows.append({'token': f'{kind}_leaf{leaf}_features{fraction}', 'kind': kind,
                             'parameters': cls(n_estimators=256, min_samples_leaf=leaf, max_features=fraction,
                                               n_jobs=1, random_state=seed).get_params()})
    rows = [{**row, 'candidate_id': 'baseline::' + row['token']} for row in rows]
    require(len(rows) == len({x['candidate_id'] for x in rows}) == 44, 'Independent conventional roster is invalid')
    return rows


def predict_baseline(payload, X):
    X = array(X, dimension=24)
    kind = payload['kind']
    if kind == 'persistence':
        result = X[:, -1]
    elif kind == 'mean':
        result = np.full(len(X), float(q(payload['value'])))
    else:
        normal = payload['normalization']
        result = float(F(normal['target_min'])) + float(F(normal['span'])) * (
            payload['model'].predict(payload['scaler'].transform(X)) + payload['offset'])
    return array(np.asarray(result, dtype=np.float64))


def audit_baseline(payload, configuration, X, y, normalization):
    require(payload['kind'] == configuration['kind']
            and canonical(payload['normalization']) == {k: normalization[k] for k in ('target_min', 'span')},
            'Conventional model normalization or family differs')
    kind = payload['kind']
    if kind == 'persistence':
        require(payload['rule'] == 'last_observed_physical_own_target', 'Persistence rule changed')
        return
    if kind == 'mean':
        expected = q(normalization['target_min']) + q(normalization['span']) * sum(y, F(0)) / len(y)
        require(q(payload['value']) == expected, 'Training-mean baseline uses another cohort')
        return
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    stored = payload['scaler']
    require(type(stored) is StandardScaler and payload['fit_n'] == len(y)
            and stored.n_samples_seen_ == len(y), 'Conventional training/scaler count differs')
    for name in ('mean_', 'var_', 'scale_'):
        require(bit_equal(getattr(stored, name), getattr(scaler, name)), 'Conventional scaler is not training-only')
    model = payload['model']
    parameters = dict(configuration['parameters'])
    if kind == 'ridge':
        parameters['alpha'] = len(y) * float(q(configuration['ridge_lambda']))
    params = model.get_params()
    require(all(params.get(k) == v for k, v in parameters.items()), 'Conventional model hyperparameters differ')
    expected_class = {'ridge': 'Ridge', 'hgb': 'HistGradientBoostingRegressor', 'svr': 'SVR',
                      'krr': 'KernelRidge', 'extra_trees': 'ExtraTreesRegressor',
                      'random_forest': 'RandomForestRegressor'}[kind]
    require(type(model).__name__ == expected_class and model.n_features_in_ == 24,
            'Conventional estimator class or feature count differs')
    expected_offset = float(np.asarray([float(v) for v in y]).mean()) if kind == 'krr' else 0.
    require(payload['offset'] == expected_offset, 'Conventional target centering differs')
    if kind == 'svr':
        require(model.fit_status_ == 0, 'Frozen SVR budget was exhausted')


NUMERIC_TOKEN = re.compile(rb'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')


def source_number(spec, token):
    token = token.strip(b' \t')
    require(NUMERIC_TOKEN.fullmatch(token), 'Source target violates its frozen decimal grammar')
    value = F(token.decode('ascii'))
    domain = spec['numeric_domain']
    require((domain == 'finite_nonnegative_decimal' and value >= 0)
            or (domain == 'finite_nonnegative_integer' and value >= 0 and value.denominator == 1)
            or (domain == 'finite_decimal_at_least_minus273_15' and value >= F('-273.15')),
            'Source target violates its predeclared physical domain')
    require(math.isfinite(float(value)), 'Source target cannot be retained as finite binary64')
    return value


class AuthorizedSource:
    """Replay only times already authorized by retained producer access receipts."""
    def __init__(self, task, manifest):
        gate('audit/preflight43.json')
        self.task, self.manifest = task, manifest
        self.spec = manifest['tasks'][task]['spec']
        self.calendar = read(bound_ref(manifest['tasks'][task]['calendar']))
        self.index = {x['timestamp']: x for x in self.calendar['rows']}
        self.payloads = source_payloads(task, manifest)
        self.protocol = sha(ROOT / 'protocol_seal43.json')
        self.manifest_hash = sha(ROOT / 'inputs/data43/manifest43.json')
        self.tokens = 0

    def values(self, access_ref, *, stage, phase, times, maximum):
        receipt = read(bound_ref(access_ref))
        spec = self.manifest['tasks'][self.task]
        require(receipt['schema'] == 'step43-authorized-numeric-access-v1' and receipt['status'] == 'PASS'
                and receipt['task'] == self.task and receipt['stage'] == stage and receipt['phase'] == phase
                and receipt['protocol_seal_sha256'] == self.protocol and receipt['data_manifest_sha256'] == self.manifest_hash
                and receipt['requested_timestamps'] == times and receipt['maximum_allowed_timestamp'] == maximum
                and receipt['maximum_decoded_timestamp'] == max(times)
                and max(times) <= maximum and len(times) == len(set(times))
                and receipt['unique_measurement_timestamps'] == len(times)
                and receipt['source'] == spec['source'] and receipt['calendar'] == spec['calendar']
                and receipt['other_measurement_values_decoded_or_parsed'] == 0
                and receipt['outcome_based_row_filtering'] is False, 'Authorized source access scope differs')
        expected_archives = [{k: row[k] for k in ('path', 'sha256', 'payload_sha256')} for row in spec['archives']]
        require(receipt['source_archives'] == expected_archives, 'Numeric access source archive bindings differ')
        if phase == 'validation':
            require(receipt['fit_seal'] == reference(ROOT / f'output/{self.task}/{stage}/fit_seal43.json'),
                    'Validation numeric access lacks its complete fit seal')
        else:
            require(receipt['fit_seal'] is None, 'Unexpected fit seal on source access')
        if phase.startswith('assessment_'):
            selection_hash = sha(ROOT / 'output/selection_seal43.json')
            gate('audit/preassessment43.json', selection_seal_sha256=selection_hash)
            require(receipt['selection_seal_sha256'] == selection_hash, 'Assessment numeric access selection differs')
        else:
            require(receipt['selection_seal_sha256'] is None, 'Development access contains assessment identity')
        references, duplicates, result = [], [], {}
        for stamp in times:
            grouped = self.index[stamp]
            require(not grouped['declared_target_missing'], 'Numeric access includes a frozen missing timestamp')
            numbers = []
            for offset in grouped['records']:
                payload = self.payloads[offset['source_id']]
                fields = csv_fields(payload[offset['byte_start']:offset['byte_stop']], self.spec['separator'])
                require(len(fields) == len(self.calendar['headers']) and iso(timestamp(self.spec, fields)) == stamp,
                        'Numeric source row offset or timestamp differs')
                numbers.append(source_number(self.spec, fields[self.spec['target_index']]))
                references.append({'timestamp': stamp, **offset})
            require(numbers and all(value == numbers[0] for value in numbers), 'Source duplicate values disagree')
            if len(numbers) > 1:
                duplicates.append({'timestamp': stamp, 'source_records': len(numbers), 'exact_agreement': True})
            result[stamp] = numbers[0]
        require(receipt['source_row_references'] == references and receipt['duplicate_agreement'] == duplicates
                and receipt['measurement_token_decodes'] == len(references), 'Numeric source provenance differs')
        self.tokens += len(references)
        return result


def audit_cohort(task, stage, partition, manifest, source):
    base = ROOT / f'output/data43/{task}/{stage}'
    record = read(base / (partition + '.json'))
    queries = manifest['tasks'][task]['stages'][stage][partition]
    section = manifest['tasks'][task]['stages'][stage]
    completion = read(base / (partition + '_completed43.json'))
    verify_bindings(completion['bound_files'])
    expected_files = {base / (partition + suffix) for suffix in ('_started43.json', '_access43.json', '_X.npy', '.json')}
    expected_files.add(base / 'normalization43.json')
    require(completion['schema'] == 'step43-cohort-access-completion-v1'
            and set(completion['bound_files']) == {p.relative_to(ROOT).as_posix() for p in expected_files}
            and completion['status'] == 'PASS' and completion['task'] == task and completion['stage'] == stage
            and completion['partition'] == partition and completion['protocol_seal_sha256'] == source.protocol
            and completion['data_manifest_sha256'] == source.manifest_hash,
            'Numeric cohort completion identity differs')
    require(record['schema'] == 'step43-causal-cohort-v1' and record['task'] == task and record['stage'] == stage
            and record['partition'] == partition and record['protocol_seal_sha256'] == source.protocol
            and record['data_manifest_sha256'] == source.manifest_hash and record['queries'] == queries
            and record['ids'] == [x['id'] for x in queries] and record['n'] == len(queries)
            and record['other_measurement_predictors_used'] is False and record['target_values_clipped'] is False,
            'Frozen numeric cohort roster or scope differs')
    times = sorted({time for query in queries for time in [query['target_timestamp'], *query_lags(query)]})
    maximum = iso(datetime.fromisoformat(section[partition + '_stop_exclusive']) - timedelta(seconds=1))
    values = source.values(record['access'], stage=stage, phase=partition, times=times, maximum=maximum)
    targets = [values[x['target_timestamp']] for x in queries]
    histories = [[values[t] for t in query_lags(query)] for query in queries]
    X = array(np.load(bound_ref(record['X']), allow_pickle=False), dimension=24)
    require(bit_equal(X, np.asarray(histories, dtype=np.float64)) and X.shape == (len(queries), 24),
            'Stored raw feature coordinates differ from authorized exact histories')
    normal = read(bound_ref(record['normalization']))
    require(record['normalization'] == reference(base / 'normalization43.json'), 'Cohort normalization path differs')
    require(normal['schema'] == 'step43-training-target-normalization-v1' and normal['task'] == task
            and normal['stage'] == stage and normal['fitted_partition'] == 'train'
            and normal['future_data_used'] is False and normal['target_values_clipped'] is False,
            'Target normalization scope differs')
    lo, hi, span = q(normal['target_min']), q(normal['target_max']), q(normal['span'])
    require(span > 0 and hi - lo == span, 'Target normalization range differs')
    if partition == 'train':
        require(lo == min(targets) and hi == max(targets)
                and normal['train_target_ids_sha256'] == digest(record['ids'])
                and normal['raw_train_targets_sha256'] == digest(targets)
                and normal['training_access'] == record['access'], 'Target scaling uses another training cohort')
    normalized = [(value - lo) / span for value in targets]
    persistence = [row[-1] for row in histories]
    require(record['raw_targets'] == list(map(str, targets)) and record['y'] == list(map(str, normalized))
            and record['raw_persistence'] == list(map(str, persistence))
            and record['persistence'] == [str((value - lo) / span) for value in persistence],
            'Stored exact target/persistence values differ from authorized source')
    return {'record': record, 'X': X, 'targets': targets, 'y': normalized, 'normalization': normal,
            'reference': reference(base / (partition + '.json'))}


REPRESENTATION_CONFIG = {
    'dimension': 24, 'scalings': ['minmax01', 'standard'], 'nystrom_widths': [8, 24],
    'kahm_caps': [4, 12, 20], 'kahm_prefixes': [1, 2], 'cluster_counts': [8, 16],
    'max_fit': 512, 'max_references': 24, 'tau': 1e-6, 'omega': 2., 'probe_count': 32,
    'nystrom_eigen_floor': 1e-12, 'temporal_selection': 'equal_row_stratum_midpoints',
    'metric_normalization': 'reciprocal_median_positive_pair_distance'}


def temporal_indices(n, m):
    require(type(n) is int and type(m) is int and n >= m > 0, 'Invalid fixed temporal strata')
    return [(2*j + 1)*n // (2*m) for j in range(m)]


def operational_map(mapping, X):
    # No producer model wrapper is imported. Existing native map arithmetic is
    # replayed, with no map refit or ideal-real arithmetic assertion.
    return array(mapping.core.transform(X) * float(mapping.metric_scale))


def audit_scaler(scaler, scaling, X):
    constant = X.max(axis=0) == X.min(axis=0)
    offset = X.min(axis=0) if scaling == 'minmax01' else X.mean(axis=0)
    divisor = X.max(axis=0) - X.min(axis=0) if scaling == 'minmax01' else X.std(axis=0, ddof=0)
    divisor = np.where(constant, 1., divisor)
    require(scaler.mode == scaling and scaler.training_rows == len(X)
            and bit_equal(scaler.offset, offset) and bit_equal(scaler.divisor, divisor)
            and np.array_equal(scaler.constant_mask, constant), 'Map scaler is not the prescribed training-only scaler')


def audit_representation_metadata(metadata, X, seed):
    probes = temporal_indices(len(X), 32)
    require(metadata['schema'] == 'step35-frozen-representations-v1' and metadata['seed'] == seed
            and metadata['training_shape'] == list(X.shape) and metadata['training_sha256'] == array_digest(X)
            and metadata['probe_indices'] == probes and metadata['probe_sha256'] == array_digest(X[probes])
            and metadata['config'] == REPRESENTATION_CONFIG and metadata['config_sha256'] == legacy_digest(REPRESENTATION_CONFIG)
            and metadata['factory_source_sha256'] == sha(ROOT / 'source/representations35.py')
            and metadata['forecast_labels_used'] is False and metadata['no_heldout_input_or_label_used'] is True,
            'Representation factory cohort/configuration/scope differs')
    rows = {row['name']: row for row in metadata['maps']}
    require(len(rows) == len(metadata['maps']) == 18 and set(rows) == set(maps_roster())
            and set(metadata['map_order']) == set(rows), 'Full factory metadata roster differs')
    for scaling in ('minmax01', 'standard'):
        row = metadata['scalers'][scaling]
        lower, upper = X.min(axis=0), X.max(axis=0)
        mask = upper == lower
        offset = lower if scaling == 'minmax01' else X.mean(axis=0)
        divisor = upper - lower if scaling == 'minmax01' else X.std(axis=0, ddof=0)
        divisor = np.where(mask, 1., divisor)
        require(row['mode'] == scaling and bit_equal(np.array(row['offset']), offset)
                and bit_equal(np.array(row['divisor']), divisor) and row['constant_mask'] == mask.tolist()
                and row['training_rows'] == len(X) and row['heldout_clipping'] is False
                and row['nonconstant_floor'] is None, 'Factory scaler metadata differs')
    cost = metadata['native_cost_accounting']
    require(cost['regrouped_banks_used_for_deployment'] is False
            and cost['unused_regrouped_fitting_included_in_physical_factory_cost'] is True,
            'Native unused-fit physical cost accounting differs')
    require(math.isfinite(metadata['factory_total_seconds']) and metadata['factory_total_seconds'] > 0,
            'Representation factory duration is invalid')
    return rows


def audit_map(mapping, detail, metadata, X, seed):
    from scipy.spatial.distance import cdist, pdist
    name, family, scaling = mapping.name, mapping.family, mapping.scaling
    require(name == detail['name'] == metadata['name'] and family == detail['family'] == metadata['family']
            and scaling == detail['scaling'] == metadata['scaling']
            and mapping.n_features_in_ == 24 and mapping.n_features_out_ == detail['dimension'] == metadata['actual_width']
            and metadata['payload_sha256'] == detail['map']['sha256']
            and metadata['payload_bytes'] == detail['map']['bytes'], 'Stored map identity or payload metadata differs')
    scaler = mapping.core if family == 'raw' else (mapping.core.scaler if family == 'nystrom' else mapping.core.bank.scaler)
    audit_scaler(scaler, scaling, X)
    probes = X[temporal_indices(len(X), 32)]
    unscaled = array(mapping.core.transform(probes))
    distances = pdist(unscaled, 'euclidean')
    positive = distances[distances > 0]
    median = float(np.median(positive)) if len(positive) else 1.
    scalar = 1. / median if len(positive) else 1.
    require(float(mapping.metric_scale) == metadata['metric_scale'] == scalar
            and metadata['unscaled_probe_distance_median'] == median
            and metadata['positive_probe_pair_count'] == len(positive)
            and metadata['metric_scale_fallback_one'] is (len(positive) == 0)
            and metadata['metric_scaling_is_training_only'] is True, 'Training-only feature metric differs')
    if family == 'nystrom':
        width = int(name.rsplit('w', 1)[1])
        indices = temporal_indices(len(X), width)
        landmarks = scaler.transform(X[indices])
        distances = pdist(scaler.transform(probes), 'sqeuclidean')
        positive = distances[distances > 0]
        gamma = 1. / float(np.median(positive)) if len(positive) else 1.
        gram = np.exp(-gamma * cdist(landmarks, landmarks, 'sqeuclidean'))
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2.)
        projection = eigenvectors / np.sqrt(np.maximum(eigenvalues, 1e-12))[None, :]
        require(metadata['landmark_indices'] == indices and metadata['landmark_sha256'] == array_digest(landmarks)
                and bit_equal(mapping.core.landmarks, landmarks) and bit_equal(mapping.core.projection, projection)
                and float(mapping.core.gamma) == metadata['gamma'] == gamma and metadata['eigen_floor'] == 1e-12,
                'Nyström training landmarks/bandwidth/eigensystem differs')
    elif family == 'kahm':
        cap, prefix = int(name.split('_cap')[1].split('_')[0]), int(name.rsplit('_p', 1)[1])
        bank = mapping.core.bank
        require(metadata['cap'] == bank.cap == cap and metadata['prefix'] == prefix and bank.seed == seed
                and metadata['omega'] == mapping.core.omega == 2. and metadata['tau'] == bank.tau == 1e-6
                and metadata['requested_block_sizes'] == list(bank.requested_counts) == [8, 16][:prefix]
                and metadata['actual_block_sizes'] == list(bank.block_sizes)
                and len(bank.blocks) == prefix and bank.kind == 'learned', 'Frozen native KAHM deployment configuration differs')
    else:
        require(family == 'raw' and mapping.requested_width == 24, 'Unknown selected representation')


def audit_bank(task, stage, seed, cohort, selected=None, *, replay_fit=True):
    import representations35 as operational_classes
    base = ROOT / f'output/{task}/{stage}/seed_{seed}'
    path = base / 'fits43.json'
    record = read(path)
    normal = cohort['normalization']
    require(record['schema'] == 'step43-stage-seed-fit-v1' and record['task'] == task and record['stage'] == stage
            and record['seed'] == seed and record['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json')
            and record['training_cohort'] == cohort['reference'] and record['normalization'] == normal,
            'Fit-bank identity/cohort/normalization differs')
    affine = affine_configs()
    baselines = baseline_configs(seed)
    if selected is not None:
        wanted = {selected[f'{seed}::{policy}']['candidate_id'] for policy in POLICIES}
        affine = [row for row in affine if row['candidate_id'] in wanted]
        baselines = [row for row in baselines if row['candidate_id'] in wanted]
        require(len(affine) == 3 and 1 <= len(baselines) <= 2, 'Final selected identity deduplication differs')
    require(record['affine_rows'] == affine, 'Fit bank affine candidate roster differs')
    expected_baselines = {row['candidate_id']: row for row in baselines}
    require(len(record['baseline_rows']) == len(baselines)
            and {row['candidate_id'] for row in record['baseline_rows']} == set(expected_baselines),
            'Fit bank conventional candidate roster differs')
    metadata = read(bound_ref(record['representation_metadata']))
    metadata_rows = audit_representation_metadata(metadata, cohort['X'], seed)
    require(set(record['maps']) == {row['map'] for row in affine}, 'Retained map roster differs from needed candidate maps')
    maps, Zs = {}, {}
    for name, detail in record['maps'].items():
        map_path = bound_ref(detail['map'])
        mapping = operational_classes.deserialize_map(map_path.read_bytes())
        audit_map(mapping, detail, metadata_rows[name], cohort['X'], seed)
        feature_path = bound_ref(detail['training_features'])
        Z = array(np.load(feature_path, allow_pickle=False), dimension=detail['dimension'])
        require(bit_equal(Z, operational_map(mapping, cohort['X'])), 'Stored training feature transform replay differs')
        fid = 'feature43:' + digest({'task': task, 'stage': stage, 'map_sha256': detail['map']['sha256'],
                                    'training_feature_sha256': detail['training_features']['sha256']})
        require(detail['feature_id'] == fid, 'Feature identity does not bind task/stage/map/training coordinates')
        maps[name], Zs[name] = mapping, Z
    heads = read(bound_ref(record['heads']))
    require(set(heads) == {row['candidate_id'] for row in affine}, 'Affine head-bank roster differs')
    for configuration in affine:
        cid, name = configuration['candidate_id'], configuration['map']
        detail, row = record['maps'][name], heads[cid]
        require(row['head'] == row['fit']['head'], 'Retained affine head and fit receipt differ')
        training_binding = {'task': task, 'stage': stage, 'cohort_path': cohort['reference']['path'],
                            'cohort_sha256': cohort['reference']['sha256'], 'training_features': detail['training_features'],
                            'map_sha256': detail['map']['sha256']}
        audit_affine_fit(row['fit'], Zs[name] if replay_fit else None, cohort['y'] if replay_fit else None,
                         ridge_lambda=configuration['ridge_lambda'], feature_id=detail['feature_id'],
                         training_binding=training_binding)
        audit_deployment(row['deployment'], head=row['head'], candidate_id=cid,
                         map_sha256=detail['map']['sha256'], map_bytes=detail['map']['bytes'], normalization=normal)
    payloads, baseline_rows, configurations = {}, {}, {row['candidate_id']: row for row in affine}
    for row in record['baseline_rows']:
        cid, configuration = row['candidate_id'], expected_baselines[row['candidate_id']]
        require({k: v for k, v in row.items() if k not in {'payload', 'pickle_sha256', 'pickle_bytes', 'recipe', 'baseline_kind'}}
                == configuration and row['recipe'] == 'baseline' and row['baseline_kind'] == configuration['kind'],
                'Conventional baseline configuration differs')
        raw = gzip.decompress(bound_ref(row['payload']).read_bytes())
        require(hashlib.sha256(raw).hexdigest() == row['pickle_sha256'] and len(raw) == row['pickle_bytes'],
                'Stored conventional estimator pickle identity differs')
        payload = pickle.loads(raw)
        audit_baseline(payload, configuration, cohort['X'], cohort['y'], normal)
        payloads[cid], baseline_rows[cid] = payload, row
        configurations[cid] = {**configuration, 'recipe': 'baseline', 'baseline_kind': configuration['kind']}
    require(math.isfinite(record['wall_seconds']) and record['wall_seconds'] > 0, 'Fit-bank wall duration is invalid')
    return {'record': record, 'reference': reference(path), 'maps': maps, 'heads': heads,
            'baselines': payloads, 'baseline_rows': baseline_rows, 'configurations': configurations}


def audit_validation(task, stage, seed, cohort, bank):
    base = ROOT / f'output/{task}/{stage}/seed_{seed}'
    prediction_path = base / 'validation_predictions43.json'
    predictions = read(prediction_path)
    scores = read(base / 'validation_scores43.json')
    require(predictions['schema'] == 'step43-validation-predictions-v1' and predictions['task'] == task
            and predictions['stage'] == stage and predictions['seed'] == seed
            and predictions['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json')
            and predictions['fit_sha256'] == bank['reference']['sha256']
            and predictions['validation_cohort'] == cohort['reference']
            and predictions['query_ids'] == cohort['record']['ids'], 'Validation prediction ancestry differs')
    configurations = bank['configurations']
    require(set(predictions['maps']) == set(bank['maps'])
            and set(predictions['candidate_predictions']) == set(configurations), 'Validation roster is incomplete')
    require(scores['schema'] == 'step43-validation-scores-v1' and scores['task'] == task and scores['stage'] == stage
            and scores['seed'] == seed and scores['n'] == len(cohort['targets'])
            and scores['prediction_sha256'] == sha(prediction_path)
            and set(scores['candidate_rows']) == set(configurations), 'Validation score roster/ancestry differs')
    features = {}
    for name, mapping in bank['maps'].items():
        stored = array(np.load(bound_ref(predictions['maps'][name]), allow_pickle=False))
        expected = np.vstack([operational_map(mapping, cohort['X'][i:i+1])[0] for i in range(len(cohort['X']))])
        require(bit_equal(stored, expected), 'Validation operational feature replay differs')
        features[name] = stored
    recomputed = {}
    for cid, cfg in configurations.items():
        row = predictions['candidate_predictions'][cid]
        if cfg['recipe'] == 'affine':
            deployment = bank['heads'][cid]['deployment']
            vectors = features[cfg['map']]
            reported = [operational_prediction(deployment, z).hex() for z in vectors]
            exact = [str(exact_prediction(deployment['head'], z, deployment['target_min'], deployment['span'])) for z in vectors]
            require(row['exact_physical'] == exact, 'Validation exact affine reference differs')
            reference_risk = exact_reference_risk(exact, cohort['targets'])
        else:
            reported = [float(predict_baseline(bank['baselines'][cid], cohort['X'][i:i+1])[0]).hex()
                        for i in range(len(cohort['X']))]
            require(row['exact_physical'] is None, 'Conventional estimator was assigned an affine reference')
            reference_risk = None
        require(row['reported_physical_hex'] == reported, 'Validation physical prediction operational replay differs')
        expected = {'configuration': cfg, 'physical_risk': exact_risk(reported, cohort['targets']),
                    'exact_reference_risk': reference_risk}
        require(scores['candidate_rows'][cid] == expected, 'Validation score/configuration is not independently reproduced')
        recomputed[cid] = expected
    return recomputed


def pooled_risk(parts):
    n = sum(row['n'] for row in parts)
    require(n > 0, 'Empty pooled development cohort')
    sse = sum((q(row['sse']) for row in parts), F(0))
    mae = sum((q(row['mae'])*row['n'] for row in parts), F(0)) / n
    return {'n': n, 'sse': str(sse), 'mse': str(sse / n), 'rmse': math.sqrt(float(sse / n)), 'mae': str(mae)}


def pool_for(policy, configurations):
    return [cid for cid, row in configurations.items()
            if ((policy in ('raw', 'nystrom', 'kahm') and row.get('family') == policy)
                or (policy == 'all_map' and row['recipe'] == 'affine')
                or (policy == 'ordinary' and row['recipe'] == 'baseline')
                or (policy == 'ridge' and row.get('baseline_kind') == 'ridge'))]


def audit_selection(task, seed, folds):
    path = ROOT / f'output/{task}/selection_seed_{seed}43.json'
    value = read(path)
    ids = set(folds['f1'])
    require(all(set(fold) == ids for fold in folds.values()) and len(ids) == 134,
            'Independent development score bank is incomplete')
    configurations = {cid: folds['f1'][cid]['configuration'] for cid in ids}
    expected_rows = {}
    for cid in sorted(ids):
        require(all(fold[cid]['configuration'] == configurations[cid] for fold in folds.values()),
                'Candidate configuration changed across development folds')
        parts = {stage: folds[stage][cid]['physical_risk'] for stage in FOLDS}
        expected_rows[cid] = {'configuration': configurations[cid], 'physical_risk': pooled_risk(list(parts.values())),
                              'folds': parts}
    selections = {}
    for policy in POLICIES:
        pool = pool_for(policy, configurations)
        require(len(pool) == {'raw': 10, 'nystrom': 20, 'kahm': 60, 'all_map': 90, 'ordinary': 44, 'ridge': 5}[policy],
                'Development policy pool count differs')
        winner = min(pool, key=lambda cid: (q(expected_rows[cid]['physical_risk']['mse']), cid))
        selections[policy] = {'candidate_id': winner, 'pool_size': len(pool),
                              'physical_risk': expected_rows[winner]['physical_risk']}
    require(value['schema'] == 'step43-development-selection-v1' and value['task'] == task and value['seed'] == seed
            and value['development_n'] == next(iter(expected_rows.values()))['physical_risk']['n']
            and value['candidate_rows'] == expected_rows and value['selections'] == selections,
            'Exact pooled development selection differs')
    expected_refs = {stage: reference(ROOT / f'output/{task}/{stage}/seed_{seed}/validation_scores43.json') for stage in FOLDS}
    require(value['fold_scores'] == expected_refs, 'Development selection does not bind all three audited folds')
    return {f'{seed}::{policy}': {**row, 'seed': seed, 'policy': policy} for policy, row in selections.items()}


def audit_fit_seal(task, stage):
    base = ROOT / f'output/{task}/{stage}'
    path = base / 'fit_seal43.json'
    seal = read(path)
    require(seal['schema'] == 'step43-stage-fit-seal-v1' and seal['task'] == task and seal['stage'] == stage
            and seal['seeds'] == list(SEEDS) and seal['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json'),
            'Complete all-seed fit seal differs')
    verify_bindings(seal['bound_files'])
    expected = {p.relative_to(ROOT).as_posix() for p in base.glob('seed_*/**/*')
                if p.is_file() and p.name not in {'validation_scores43.json', 'validation_predictions43.json'}
                and '/features/validation/' not in p.as_posix()}
    require(set(seal['bound_files']) == expected, 'Fit seal omits model bytes or incorrectly includes later validation')
    completion = read(base / 'stage_completed43.json')
    require(completion['status'] == 'COMPLETE' and completion['task'] == task and completion['stage'] == stage
            and completion['fit_seal_sha256'] == sha(path), 'Stage completion does not bind the fit bank')
    return seal


def one_event(events, kind, **fields):
    matches = [row for row in events if row['kind'] == kind and all(row.get(k) == v for k, v in fields.items())]
    require(len(matches) == 1, 'Expected exactly one retained event: ' + kind + ' ' + str(fields))
    return matches[0]


def before(left, right, message):
    require(left['monotonic_ns'] < right['monotonic_ns']
            and datetime.fromisoformat(left['utc']) <= datetime.fromisoformat(right['utc']), message)


def audit_development_events(data_events, events):
    expected = []
    for task in TASKS:
        for stage in (*FOLDS, 'final'):
            sections = ('train',) if stage == 'final' else ('train', 'validation')
            seal_event = one_event(events, 'stage_models_sealed', task=task, stage=stage)
            seal = read(bound_ref(seal_event['fit_seal']))
            for partition in sections:
                fields = {'task': task, 'stage': stage, 'partition': partition}
                start = one_event(data_events, 'cohort_access_started', **fields)
                finish = one_event(data_events, 'cohort_access_completed', **fields)
                before(start, finish, 'A cohort completion precedes its authorized start')
                for key in ('cohort', 'access', 'completion'):
                    bound_ref(finish[key])
                marker = read(ROOT / f'output/data43/{task}/{stage}/{partition}_started43.json')
                require(marker['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json')
                        and marker['task'] == task and marker['stage'] == stage and marker['partition'] == partition,
                        'Cohort start marker differs')
                if partition == 'train':
                    before(finish, seal_event, 'Stage model sealing precedes completed training access')
                else:
                    before(seal_event, start, 'Validation target access preceded all-seed fit sealing')
                    require(start['fit_seal'] == seal_event['fit_seal'] == marker['fit_seal']
                            and datetime.fromisoformat(seal['utc']) <= datetime.fromisoformat(start['utc']),
                            'Validation accesses a different all-seed fit seal')
                expected += [start['index'], finish['index']]
    developmental = [row['index'] for row in data_events if row['kind'].startswith('cohort_')]
    require(sorted(expected) == developmental and len(expected) == 42,
            'Unexpected, repeated, failed or missing developmental numeric access')
    return {'training_cohorts': 12, 'validation_cohorts': 9, 'development_access_events': 42}


def log_prefix(path):
    raw = Path(path).read_bytes()
    return {'path': Path(path).relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'lines': len(raw.splitlines()), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify_log_prefix(record):
    raw = safe_path(record['path']).read_bytes()[:record['bytes']]
    require(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
            and len(raw.splitlines()) == record['lines'], 'A previously audited event prefix changed')


def preassessment():
    _, manifest, _, _ = protocol_check()
    gate('audit/preflight43.json')
    require(not (ROOT / 'output/assessment').exists(), 'Assessment predictions existed before selected-model audit')
    data_events = validate_events(ROOT / 'output/data_access43.jsonl')
    events = validate_events(ROOT / 'output/events43.jsonl')
    require(not any(row['kind'].startswith('assessment_') for row in data_events + events)
            and not any((ROOT / f'output/data43/{task}/assessment').exists() for task in TASKS),
            'Assessment numeric access preceded independent selection audit')
    chronology = audit_development_events(data_events, events)
    roster = read(ROOT / 'output/selected_roster43.json')
    require(roster['schema'] == 'step43-selected-roster-v1'
            and roster['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json')
            and roster['selection_uses_development_only'] is True and roster['assessment_targets_accessed'] == 0
            and set(roster['tasks']) == set(TASKS), 'Global selected roster identity/scope differs')
    summary, total_numeric_tokens = {}, 0
    for task in TASKS:
        source = AuthorizedSource(task, manifest)
        folds = {seed: {} for seed in SEEDS}
        for stage in FOLDS:
            audit_fit_seal(task, stage)
            training = audit_cohort(task, stage, 'train', manifest, source)
            validation = audit_cohort(task, stage, 'validation', manifest, source)
            for seed in SEEDS:
                bank = audit_bank(task, stage, seed, training)
                folds[seed][stage] = audit_validation(task, stage, seed, validation, bank)
                del bank
        expected_selections = {}
        for seed in SEEDS:
            expected_selections.update(audit_selection(task, seed, folds[seed]))
        require(roster['tasks'][task]['selections'] == expected_selections
                and roster['tasks'][task]['selection_files'] == [reference(ROOT / f'output/{task}/selection_seed_{seed}43.json') for seed in SEEDS],
                'Global selected roster does not equal independently audited choices')
        audit_fit_seal(task, 'final')
        training = audit_cohort(task, 'final', 'train', manifest, source)
        final_models = {}
        for seed in SEEDS:
            bank = audit_bank(task, 'final', seed, training, expected_selections)
            final_models[str(seed)] = {
                'affine_deployments': {cid: row['deployment']['deployment_id'] for cid, row in bank['heads'].items()},
                'conventional_payloads': {cid: row['pickle_sha256'] for cid, row in bank['baseline_rows'].items()},
                'bank': bank['reference']}
            del bank
        summary[task] = {'selections': expected_selections, 'final_models': final_models,
                         'development_n': next(iter(expected_selections.values()))['physical_risk']['n'],
                         'assessment_query_count': len(manifest['tasks'][task]['assessment'])}
        total_numeric_tokens += source.tokens
        print(json.dumps({'audit_progress': 'development_and_selection_reproduced', 'task': task}), flush=True)
        del source, training, folds
    selection_path = ROOT / 'output/selection_seal43.json'
    selection = read(selection_path)
    require(selection['schema'] == 'step43-selection-seal-v1'
            and selection['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json'), 'Selection/final-model seal differs')
    verify_bindings(selection['bound_files'])
    expected_paths = {ROOT / 'output/selected_roster43.json'}
    for task in TASKS:
        expected_paths.update(ROOT / f'output/{task}/selection_seed_{seed}43.json' for seed in SEEDS)
        expected_paths.update(path for path in (ROOT / f'output/{task}/final').rglob('*') if path.is_file())
    require(set(selection['bound_files']) == {p.relative_to(ROOT).as_posix() for p in expected_paths},
            'Selection seal does not exactly close selected identities and final models')
    one_event(events, 'selection_and_final_models_sealed')
    bindings = [ROOT / 'protocol_seal43.json', ROOT / 'audit/preflight43.json', selection_path,
                ROOT / 'output/selected_roster43.json']
    for task in TASKS:
        bindings.extend(path for path in (ROOT / f'output/{task}').rglob('*') if path.is_file())
        bindings.extend(path for path in (ROOT / f'output/data43/{task}').rglob('*') if path.is_file())
    return publish_gate(ROOT / 'audit/preassessment43.json', 'preassessment', bindings,
                        selection_seal_sha256=sha(selection_path), tasks=summary,
                        development_seed_banks_checked=27, final_selected_seed_banks_checked=9,
                        development_candidate_scores_recomputed=27*134,
                        assessment_targets_accessed_before_gate=0,
                        previously_authorized_source_tokens_replayed=total_numeric_tokens,
                        retained_event_prefixes=[log_prefix(ROOT / name) for name in
                                                ('output/data_access43.jsonl', 'output/events43.jsonl')],
                        **chronology)


def evaluation_id(bank, cid):
    if cid in bank['heads']:
        return 'affine::' + bank['heads'][cid]['deployment']['deployment_id']
    return 'baseline::' + digest({'candidate_id': cid, 'pickle_sha256': bank['baseline_rows'][cid]['pickle_sha256']})


def comparison(raw, other):
    require(raw['n'] == other['n'], 'Primary comparison has unequal query cohorts')
    a, b = q(raw['sse']), q(other['sse'])
    return {'raw_risk': raw, 'reference_risk': other, 'mse_ratio': str(a / b) if b else None,
            'within_five_percent_mse': 20*a <= 21*b}


def audit_query(task, index, query, manifest, source, banks, selected, data_events, events, previous_targets):
    qdir = ROOT / f'output/data43/{task}/assessment/queries/{index:04d}'
    path = ROOT / f'output/assessment/{task}/queries/{index:04d}/predictions43.json'
    prediction = read(path)
    history_path = qdir / 'history43.json'
    history = read(history_path)
    history_ref = reference(history_path)
    selection_hash = sha(ROOT / 'output/selection_seal43.json')
    normal = banks[3901]['record']['normalization']
    normal_ref = reference(ROOT / f'output/data43/{task}/final/normalization43.json')
    fields = {'task': task, 'query_index': index, 'query_id': query['id']}
    order = [one_event(data_events, kind, **fields) for kind in (
        'assessment_history_started', 'assessment_history_completed', 'assessment_predictions_sealed',
        'assessment_target_started', 'assessment_target_revealed')]
    for left, right in zip(order, order[1:]):
        before(left, right, 'Query history/prediction/target chronology differs')
    committed = one_event(events, 'assessment_query_predictions_committed', **fields)
    before(order[1], committed, 'Query prediction preceded completed authorized history')
    before(committed, order[2], 'Query byte-sealing preceded complete prediction commit')
    require(committed['prediction'] == reference(path) and order[1]['history'] == history_ref
            and order[1]['access'] == history['access'], 'Query chronology events bind different evidence')
    for key in ('task', 'query_index'):
        require(history[key] == fields[key], 'Assessment history identity differs')
    require(history['schema'] == 'step43-causal-assessment-history-v1' and history['query_id'] == query['id']
            and history['target_timestamp'] == query['target_timestamp'] and history['origin_timestamp'] == query['origin_timestamp']
            and history['protocol_seal_sha256'] == source.protocol and history['data_manifest_sha256'] == source.manifest_hash
            and history['selection_seal_sha256'] == selection_hash and history['normalization'] == normal_ref
            and history['current_target_decoded'] is False and history['other_measurement_predictors_used'] is False,
            'Assessment causal-history source identity or scope differs')
    lags = query_lags(query)
    require(history['history_timestamps'] == lags
            and history['previously_revealed_targets_in_history'] == sorted(set(previous_targets) & set(lags)),
            'Assessment history is not the fixed causal 24-hour snapshot')
    values = source.values(history['access'], stage='final', phase='assessment_history', times=lags,
                           maximum=query['origin_timestamp'])
    raw = [values[stamp] for stamp in lags]
    lo, span = q(normal['target_min']), q(normal['span'])
    require(history['raw_values'] == list(map(str, raw)) and q(history['raw_persistence']) == raw[-1]
            and q(history['persistence']) == (raw[-1] - lo) / span, 'Assessment history exact values differ')
    X = array(np.load(bound_ref(history['X']), allow_pickle=False), dimension=24)
    require(X.shape == (1, 24) and bit_equal(X, np.asarray([raw], dtype=np.float64)), 'Assessment history float coordinates differ')
    require(prediction['schema'] == 'step43-query-predictions-v1' and prediction['task'] == task
            and prediction['query_id'] == query['id'] and prediction['query_index'] == index
            and prediction['protocol_seal_sha256'] == source.protocol and prediction['selection_seal_sha256'] == selection_hash
            and prediction['history_receipt_sha256'] == history_ref['sha256'] and prediction['history'] == history_ref
            and set(prediction['predictions']) == set(selected), 'Complete selected-model query commitment differs')
    require(bit_equal(np.load(bound_ref(prediction['raw_features']), allow_pickle=False), X),
            'Producer raw query coordinates differ from authorized history')
    byte_seal = read(qdir / 'predictions_seal43.json')
    seal_ref = reference(qdir / 'predictions_seal43.json')
    require(byte_seal['schema'] == 'step43-query-prediction-byte-seal-v1' and byte_seal['task'] == task
            and byte_seal['query_index'] == index and byte_seal['query_id'] == query['id']
            and byte_seal['protocol_seal_sha256'] == source.protocol and byte_seal['selection_seal_sha256'] == selection_hash
            and byte_seal['predictions'] == reference(path) and byte_seal['history'] == history_ref
            and byte_seal['prediction_keys'] == sorted(selected) and byte_seal['prediction_count'] == 18
            and byte_seal['current_target_decoded'] is False
            and order[2]['prediction_seal'] == order[3]['prediction_seal'] == seal_ref
            and order[2]['predictions'] == reference(path) and order[2]['prediction_count'] == 18
            and order[2]['current_target_decoded'] is False,
            'Target authorization does not bind all 18 actual preceding forecasts')
    groups = {}
    for alias, chosen in selected.items():
        bank, cid = banks[chosen['seed']], chosen['candidate_id']
        eid = evaluation_id(bank, cid)
        group = groups.setdefault(eid, {'bank': bank, 'candidate_id': cid, 'aliases': []})
        require(group['candidate_id'] == cid, 'Conflicting deduplicated model identity')
        group['aliases'].append(alias)
    require(set(prediction['evaluations']) == set(groups), 'Query ran omitted/unselected/non-deduplicated model identities')
    audited, vectors = {}, {}
    for eid, group in groups.items():
        bank, cid = group['bank'], group['candidate_id']
        evaluation = prediction['evaluations'][eid]
        cfg = bank['configurations'][cid]
        require(evaluation['candidate_id'] == cid and evaluation['aliases'] == group['aliases']
                and evaluation['recipe'] == cfg['recipe'] and evaluation['fit_bank'] == bank['reference'],
                'Query evaluation changes selected candidate, aliases or final fit bytes')
        if cfg['recipe'] == 'affine':
            deployment = bank['heads'][cid]['deployment']
            detail = bank['record']['maps'][cfg['map']]
            require(evaluation['deployment'] == deployment and evaluation['map'] == detail['map']
                    and evaluation['map_name'] == cfg['map'], 'Query uses another stored affine deployment/map')
            bound_ref(evaluation['map'])
            z = array(np.load(bound_ref(evaluation['features']), allow_pickle=False))
            require(z.shape == (1, detail['dimension']) and bit_equal(z, operational_map(bank['maps'][cfg['map']], X)),
                    'Assessment native feature transform replay differs')
            z = z[0]
            reported = operational_prediction(deployment, z).hex()
            require(evaluation['reported_physical_hex'] == reported, 'Assessment affine operational forecast differs')
            audited[eid] = audit_output_assay(evaluation['assay'], deployment, z,
                                               query_id=query['id'], reported_hex=reported)
            vectors[eid] = z
        else:
            row = bank['baseline_rows'][cid]
            require(evaluation['payload'] == row['payload'] and evaluation['pickle_sha256'] == row['pickle_sha256'],
                    'Assessment conventional payload differs from selected fit')
            reported = float(predict_baseline(bank['baselines'][cid], X)[0]).hex()
            require(evaluation['reported_physical_hex'] == reported, 'Assessment conventional operational forecast differs')
        for alias in group['aliases']:
            expected = {'candidate_id': cid, 'evaluation_id': eid, 'reported_physical_hex': reported}
            if cfg['recipe'] == 'affine':
                expected['exact_physical'] = str(audited[eid]['binary64']['exact_physical'])
            require(prediction['predictions'][alias] == expected, 'Policy alias does not point to its actual scored evaluation')
    # Only now inspect the stored target, after checking its preceding commit and
    # all actual selected-model values. This reconstructs producer order; it is
    # evidence consistency, not cryptographic proof of historical execution.
    target_path = qdir / 'target43.json'
    target = read(target_path)
    require(target['schema'] == 'step43-revealed-assessment-target-v1' and target['task'] == task
            and target['query_id'] == query['id'] and target['query_index'] == index
            and target['target_timestamp'] == query['target_timestamp'] and target['origin_timestamp'] == query['origin_timestamp']
            and target['protocol_seal_sha256'] == source.protocol and target['data_manifest_sha256'] == source.manifest_hash
            and target['selection_seal_sha256'] == selection_hash and target['normalization'] == normal_ref
            and target['history'] == history_ref and target['prediction_seal'] == seal_ref
            and target['target_value_clipped'] is False and order[4]['target'] == reference(target_path)
            and order[4]['access'] == target['access'] and order[4]['next_index'] == index + 1,
            'Revealed target identity/source/prediction ancestry differs')
    truth = source.values(target['access'], stage='final', phase='assessment_target',
                         times=[query['target_timestamp']], maximum=query['target_timestamp'])[query['target_timestamp']]
    require(q(target['raw_target']) == truth and q(target['y']) == (truth - lo) / span,
            'Revealed target exact source value or unclipped normalization differs')
    return {'prediction': prediction, 'target': truth, 'audited': audited, 'vectors': vectors,
            'row': {'query_index': index, 'query_id': query['id'], 'month': query['target_timestamp'][:7],
                    'prediction': reference(path), 'target': reference(target_path)}}


def audit_task_assessment(task, manifest, data_events, events):
    source = AuthorizedSource(task, manifest)
    selected = read(ROOT / 'output/selected_roster43.json')['tasks'][task]['selections']
    training = audit_cohort(task, 'final', 'train', manifest, source)
    banks = {seed: audit_bank(task, 'final', seed, training, selected, replay_fit=False) for seed in SEEDS}
    queries = manifest['tasks'][task]['assessment']
    report = read(ROOT / f'output/assessment/{task}/assessment43.json')
    require(report['schema'] == 'step43-task-assessment-v1' and report['task'] == task and report['n'] == len(queries)
            and report['protocol_seal_sha256'] == source.protocol
            and report['selection_seal_sha256'] == sha(ROOT / 'output/selection_seal43.json'),
            'Task assessment identity or query count differs')
    predictions, references, accepted = ({key: [] for key in selected} for _ in range(3))
    summary = {name: {'packets': 0, 'accepted': 0, 'rejected_over_budget': 0, 'binding_only_accepted': 0}
               for name in ('binary64', 'physical_output_float32')}
    summary['quantization_changed_reported_value'] = 0
    rows, targets, previous_targets, first_two, query_records = [], [], [], [], []
    for index, query in enumerate(queries):
        checked = audit_query(task, index, query, manifest, source, banks, selected, data_events, events, previous_targets)
        rows.append(checked['row'])
        targets.append(checked['target'])
        query_records.append(checked['prediction'])
        previous_targets.append(query['target_timestamp'])
        if index < 2:
            first_two.append(checked)
        for alias, row in checked['prediction']['predictions'].items():
            predictions[alias].append(row['reported_physical_hex'])
            if 'exact_physical' in row:
                references[alias].append(row['exact_physical'])
                accepted[alias].append(checked['audited'][row['evaluation_id']]['binary64']['accepted'])
        for value in checked['audited'].values():
            for name in ('binary64', 'physical_output_float32'):
                summary[name]['packets'] += 1
                summary[name]['accepted'] += int(value[name]['accepted'])
                summary[name]['rejected_over_budget'] += int(not value[name]['accepted'])
                summary[name]['binding_only_accepted'] += 1
            summary['quantization_changed_reported_value'] += int(value['quantization_changed_reported_value'])
    results = {alias: {'candidate_id': selected[alias]['candidate_id'],
                      'physical_risk': exact_risk(predictions[alias], targets),
                      'exact_reference_risk': exact_reference_risk(references[alias], targets) if references[alias] else None,
                      'accepted_packets': sum(accepted[alias]), 'packet_count': len(accepted[alias])} for alias in selected}
    monthly = {}
    for month in sorted({row['month'] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row['month'] == month]
        monthly[month] = {alias: exact_risk([predictions[alias][i] for i in indices], [targets[i] for i in indices])
                          for alias in selected}
    primary = {policy: comparison(results['3901::raw']['physical_risk'], results['3901::' + policy]['physical_risk'])
               for policy in ('ordinary', 'ridge')}
    coverage = {'packets': len(accepted['3901::raw']), 'accepted': sum(accepted['3901::raw']),
                'all_accepted': len(accepted['3901::raw']) == len(queries) and all(accepted['3901::raw'])}
    require(report['queries'] == rows and report['policy_results'] == results and report['monthly_results'] == monthly
            and report['primary_comparisons'] == primary and report['primary_certificate_coverage'] == coverage
            and report['certificate_summary'] == summary, 'Assessment exact scores/months/primary criterion/packet summaries differ')
    state = read(ROOT / f'output/data43/{task}/assessment/state43.json')
    require(state['status'] == 'COMPLETE' and state['next_index'] == len(queries) and state['pending_query'] is None
            and state['previously_revealed_targets'] == previous_targets and state['protocol_seal_sha256'] == source.protocol
            and state['selection_seal_sha256'] == sha(ROOT / 'output/selection_seal43.json'),
            'Assessment access state did not close the exact chronological query roster')
    misbinding = read(ROOT / f'output/assessment/{task}/misbinding43.json')
    affine_ids = set(first_two[0]['audited'])
    require(misbinding['schema'] == 'step43-selected-head-misbinding-v1' and misbinding['task'] == task
            and misbinding['protocol_seal_sha256'] == source.protocol and misbinding['query_index'] == 0
            and misbinding['stale_query_index'] == 1 and set(misbinding['records']) == affine_ids
            and misbinding['performed_after_assessment'] == reference(ROOT / f'output/assessment/{task}/assessment43.json')
            and misbinding['query_prediction'] == rows[0]['prediction'] and misbinding['stale_query_prediction'] == rows[1]['prediction'],
            'Misbinding challenge sample or model roster changed')
    challenges = []
    for eid in sorted(affine_ids):
        evaluation = first_two[0]['prediction']['evaluations'][eid]
        challenges.append(audit_misbinding(misbinding['records'][eid], evaluation['deployment'], first_two[0]['vectors'][eid],
                                           first_two[1]['vectors'][eid], query_id=queries[0]['id'],
                                           reported_hex=evaluation['reported_physical_hex']))
    print(json.dumps({'audit_progress': 'assessment_reproduced', 'task': task, 'queries': len(queries)}), flush=True)
    return {'report': report, 'banks': banks, 'query_records': query_records, 'selected': selected,
            'summary': {'queries': len(queries), 'policy_alias_predictions': len(queries)*18,
                        'deduplicated_affine_packets': summary['binary64']['packets'], 'certificate_summary': summary,
                        'primary_comparisons': primary, 'primary_certificate_coverage': coverage,
                        'misbinding_challenges': sum(x['challenges'] for x in challenges),
                        'misbinding_applicable': sum(x['applicable'] for x in challenges),
                        'misbinding_rejected_as_expected': sum(x['rejected_as_expected'] for x in challenges),
                        'previously_authorized_source_tokens_replayed': source.tokens}}


def timing_samples(values, count):
    require(isinstance(values, list) and len(values) == count
            and all(type(value) is int and value >= 0 for value in values), 'Retained raw timing sample roster is invalid')
    return float(linear_quantile(values, .5))


def timing_summary(values):
    return {'queries': len(values), 'p50_ms': float(linear_quantile(values, .5))/1e6,
            'p95_ms': float(linear_quantile(values, .95))/1e6,
            'min_ms': float(min(values))/1e6, 'max_ms': float(max(values))/1e6}


def audit_cost(task, checked, design, events):
    path = ROOT / f'output/benchmarks/{task}/cost43.json'
    value = read(path)
    report, queries, selected = checked['report'], checked['query_records'], checked['selected']
    bank = checked['banks'][3901]
    count = min(32, len(queries))
    indices = temporal_indices(len(queries), count)
    aliases = ['3901::' + policy for policy in (*CERTIFIED_POLICIES, 'ordinary', 'ridge')]
    require(value['schema'] == 'step43-selected-model-cost-assay-v1' and value['status'] == 'COMPLETE'
            and value['task'] == task and value['seed'] == 3901
            and value['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json')
            and value['selection_seal_sha256'] == sha(ROOT / 'output/selection_seal43.json')
            and value['preassessment_gate_sha256'] == sha(ROOT / 'audit/preassessment43.json')
            and value['configuration'] == design['cost_assay'] and value['assessment_n'] == len(queries)
            and value['query_indices'] == indices and value['sample_query_count'] == count
            and value['primary_policy_aliases'] == aliases and value['new_fits'] == 0
            and value['new_source_measurement_decodes'] == 0 and value['population_runtime_claim'] is False
            and value['speedup_threshold'] is None, 'Selected-model cost design or scope differs')
    verify_bindings(value['source_bindings'])
    required = {'protocol_seal43.json', 'protocol43.json', 'design43.json', 'audit/preassessment43.json',
                'output/selection_seal43.json', 'output/selected_roster43.json',
                *(f'output/assessment/{t}/assessment43.json' for t in TASKS)}
    require(required <= set(value['source_bindings']), 'Cost evidence omits prior complete assessments or selected identities')
    groups = {}
    for alias in aliases:
        cid = selected[alias]['candidate_id']
        eid = evaluation_id(bank, cid)
        groups.setdefault(eid, {'candidate_id': cid, 'recipe': bank['configurations'][cid]['recipe'], 'aliases': []})['aliases'].append(alias)
    require(set(value['deduplicated_models']) == set(groups), 'Cost model roster substitutes/repeats/omits selected identities')
    samples = 0
    for eid, group in groups.items():
        model = value['deduplicated_models'][eid]
        cid = group['candidate_id']
        require(model['candidate_id'] == cid and model['recipe'] == group['recipe'] and model['aliases'] == group['aliases']
                and model['evaluation_id'] == eid and len(model['queries']) == count,
                'Cost model identity or query roster differs')
        affine = group['recipe'] == 'affine'
        if affine:
            deployment = bank['heads'][cid]['deployment']
            map_record = bank['record']['maps'][bank['configurations'][cid]['map']]
            static = model['static_context']
            require(model['deployment_id'] == deployment['deployment_id'] and model['map'] == map_record['map']
                    and static['warmups'] == 3 and static['repeats'] == 100
                    and static['static_deployment_json_bytes'] == len(legacy_bytes(deployment))
                    and model['static_context_median_ms'] == timing_samples(static['nanoseconds'], 100)/1e6,
                    'Static deployment preparation timing or byte record differs')
            samples += 100
            medians = {name: [] for name in ('prediction_without_certificate', 'certified_prediction',
                                            'certificate_verification', 'direct_exact_reference')}
            sizes = {name: [] for name in ('packet_json_bytes', 'feature_payload_bytes', 'static_deployment_json_bytes', 'map_bytes')}
        else:
            baseline = bank['baseline_rows'][cid]
            require(model['payload'] == baseline['payload'] and model['pickle_sha256'] == baseline['pickle_sha256']
                    and model['pickle_bytes'] == baseline['pickle_bytes'], 'Timed conventional estimator differs')
            medians = {'selected_conventional_prediction': []}
        for ordinal, (index, row) in enumerate(zip(indices, model['queries'])):
            query, query_ref = queries[index], report['queries'][index]['prediction']
            evaluation = query['evaluations'][eid]
            require(row['query_index'] == index and row['sample_ordinal'] == ordinal
                    and row['query_id'] == query['query_id'] and row['prediction'] == query_ref
                    and row['raw_features'] == query['raw_features']
                    and row['reported_physical_hex'] == evaluation['reported_physical_hex']
                    and row['warmups'] == 3 and row['repeats'] == 10, 'Cost query sample/prediction identity differs')
            for ref in (query_ref, query['raw_features']):
                require(value['source_bindings'].get(ref['path']) == ref['sha256'], 'Cost sample is not bound by exact retained query bytes')
            if affine:
                honest = evaluation['assay']['cases']['binary64']
                order = ['prediction_without_certificate', 'certified_prediction']
                if ordinal % 2:
                    order.reverse()
                order += ['certificate_verification', 'direct_exact_reference']
                require(row['primitive_order'] == order and set(row['nanoseconds']) == set(medians)
                        and set(row['median_nanoseconds']) == set(medians)
                        and row['features'] == evaluation['features']
                        and row['verification_status'] == honest['verification']['status']
                        and row['expected_acceptance'] == honest['expected_acceptance']
                        and row['absolute_physical_error'] == honest['absolute_physical_error']
                        and row['physical_error_budget'] == honest['physical_error_budget'],
                        'Cost counterbalancing or actual certificate acceptance differs')
                require(value['source_bindings'].get(row['features']['path']) == row['features']['sha256'],
                        'Cost feature input hash is not bound')
                expected_sizes = {'packet_json_bytes': honest['packet_json_bytes'],
                                  'feature_payload_bytes': honest['feature_payload_bytes'],
                                  'static_deployment_json_bytes': len(legacy_bytes(deployment)),
                                  'map_bytes': deployment['map_bytes']}
                for name, expected in expected_sizes.items():
                    require(type(row[name]) is int and row[name] == expected, 'Cost payload accounting differs')
                    sizes[name].append(expected)
                for name in medians:
                    median = timing_samples(row['nanoseconds'][name], 10)
                    require(row['median_nanoseconds'][name] == median, 'Cost per-query median differs from raw samples')
                    medians[name].append(median)
                    samples += 10
                require(row['direct_exact_reference_scope'] == 'Same cached integer reference as checker; no packet parsing, identity checks or discrepancy reporting'
                        and row['timing_scope'] == 'Local one-query execution; excludes disk/network I/O and process startup',
                        'Direct-reference/checker timing scope changed')
            else:
                median = timing_samples(row['nanoseconds'], 10)
                require(row['median_nanoseconds'] == median, 'Conventional cost median differs')
                medians['selected_conventional_prediction'].append(median)
                samples += 10
        require(model['component_summary'] == {name: timing_summary(values) for name, values in medians.items()},
                'Cost across-query p50/p95 summaries differ from raw timings')
        if affine:
            expected_sizes = {name: {'min': min(values), 'median': float(linear_quantile(values, .5)), 'max': max(values)}
                              for name, values in sizes.items()}
            require(model['byte_summary'] == expected_sizes, 'Cost payload size summaries differ')
    start, finish = one_event(events, 'cost_started', task=task), one_event(events, 'cost_completed', task=task)
    before(start, finish, 'Cost completion preceded measurement start')
    require(finish['cost'] == reference(path) and start['query_indices'] == indices
            and start['selected_evaluation_ids'] == list(groups) and finish['selected_models'] == len(groups)
            and finish['sample_queries'] == count, 'Cost event identity/roster differs')
    for other in TASKS:
        completed = one_event(events, 'producer_phase_completed', phase='assess', task=other)
        before(completed, start, 'Cost ran before every assessment finished')
        challenged = one_event(events, 'producer_phase_completed', phase='misbinding', task=other)
        before(challenged, start, 'Cost ran before fixed misbinding challenges finished')
    require(datetime.fromisoformat(value['started_utc']) <= datetime.fromisoformat(value['completed_utc'])
            and math.isfinite(value['wall_seconds']) and value['wall_seconds'] > 0, 'Cost timing envelope is invalid')
    return {'selected_deduplicated_models': len(groups), 'query_sample_size': count,
            'retained_raw_timing_samples_checked': samples, 'timing_repetitions_rerun_by_auditor': 0,
            'benchmark_claim': 'Local observed samples; no population runtime or speedup threshold claim'}


def expected_command_plan():
    plan = [('preflight', ['source/audit43.py', 'preflight'])]
    plan += [(f'development_{task}_{stage}', ['source/run43.py', 'stage', '--task', task, '--stage', stage])
             for task in TASKS for stage in FOLDS]
    plan += [('selection', ['source/run43.py', 'select'])]
    plan += [(f'final_fit_{task}', ['source/run43.py', 'stage', '--task', task, '--stage', 'final']) for task in TASKS]
    plan += [('selection_seal', ['source/run43.py', 'seal-selection']),
             ('preassessment_audit', ['source/audit43.py', 'preassessment'])]
    plan += [(f'assessment_{task}', ['source/run43.py', 'assess', '--task', task]) for task in TASKS]
    plan += [(f'misbinding_{task}', ['source/run43.py', 'misbinding', '--task', task]) for task in TASKS]
    plan += [(f'cost_{task}', ['source/cost43.py', '--task', task]) for task in TASKS]
    plan += [('aggregate', ['source/run43.py', 'aggregate']), ('final_audit', ['source/audit43.py', 'final'])]
    return [{'label': label, 'arguments': arguments} for label, arguments in plan]


def audit_controller(protocol, current_index):
    plan = expected_command_plan()
    require(len(plan) == 27 and protocol['execution']['commands'] == plan, 'Frozen controller command roster changed')
    started = read(ROOT / 'output/execution_started43.json')
    require(started['schema'] == 'step43-controller-start-v1' and started['commands'] == plan
            and started['command_count'] == 27 and started['producer_grid_runs'] == 1 and started['outcome_based_reruns'] == 0
            and started['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json'), 'Single-pass controller start differs')
    rows = validate_events(ROOT / 'output/execution_events43.jsonl')
    begin = one_event(rows, 'execution_started')
    require(begin['command_count'] == 27 and begin['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json'),
            'Controller event ancestry differs')
    expected_count = 1 + 2*current_index + 1
    require(len(rows) == expected_count and not (ROOT / 'output/execution_failed43.json').exists(),
            'Repeated, failed, extra or reordered controller commands exist')
    for index in range(current_index + 1):
        start = rows[1 + 2*index]
        require(start['kind'] == 'command_started' and start['command_index'] == index
                and start['label'] == plan[index]['label'] and start['arguments'] == plan[index]['arguments'],
                'Controller command order differs')
        if index < current_index:
            finish = rows[2 + 2*index]
            require(finish['kind'] == 'command_completed' and finish['command_index'] == index
                    and finish['label'] == start['label'] and finish['arguments'] == start['arguments']
                    and finish['returncode'] == 0 and math.isfinite(finish['wall_seconds'])
                    and 0 <= finish['wall_seconds'] <= protocol['execution']['command_budget_seconds'] + 1,
                    'Prior command failed or exceeded its declared wall budget')
            before(start, finish, 'Controller completed before starting a command')
            bound_ref(finish['stdout'])
            bound_ref(finish['stderr'])
    return {'completed_commands_before_current_audit': current_index,
            'command_roster_count': len(plan), 'current_command': plan[current_index]['label'],
            'event_prefix': log_prefix(ROOT / 'output/execution_events43.jsonl')}


def final():
    protocol, manifest, design, _ = protocol_check()
    controller = audit_controller(protocol, 26)
    pre = gate('audit/preassessment43.json', selection_seal_sha256=sha(ROOT / 'output/selection_seal43.json'))
    for prefix in pre['retained_event_prefixes']:
        verify_log_prefix(prefix)
    data_events, events = (validate_events(ROOT / name) for name in ('output/data_access43.jsonl', 'output/events43.jsonl'))
    audit_development_events(data_events, events)
    require(len(data_events) == 42 + 5*sum(len(manifest['tasks'][task]['assessment']) for task in TASKS)
            and not any('failed' in row['kind'] for row in data_events + events),
            'Repeated, failed, omitted or unexpected numeric-access events exist')
    tasks, costs, reports = {}, {}, {}
    for task in TASKS:
        checked = audit_task_assessment(task, manifest, data_events, events)
        tasks[task] = checked['summary']
        costs[task] = audit_cost(task, checked, design, events)
        reports[task] = checked['report']
        del checked
    result_path = ROOT / 'output/result43.json'
    result = read(result_path)
    predictive = all(row['within_five_percent_mse'] for task in tasks.values() for row in task['primary_comparisons'].values())
    coverage = all(task['primary_certificate_coverage']['all_accepted'] for task in tasks.values())
    established = predictive and coverage
    require(result['schema'] == 'step43-selected-policy-transfer-result-v1'
            and result['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json') and result['tasks'] == reports
            and result['primary_policy'] == 'raw-family P1R1C1' and result['primary_seed'] == 3901
            and result['primary_predictive_criterion_met'] is predictive
            and result['primary_certificate_coverage_met'] is coverage
            and result['primary_criterion_met_pending_independent_audit'] is established
            and result['primary_status_if_independent_audit_passes'] == ('ESTABLISHED_ON_THREE_FIXED_PERIODS' if established else 'NOT_ESTABLISHED')
            and result['independent_final_audit_pending'] is True and result['population_noninferiority_or_significance_claim'] is False
            and result['unseen_domain_claim'] is False and result['assessment_query_count'] == sum(x['queries'] for x in tasks.values())
            and result['prior_step41_primary_status_unchanged'] == 'NOT_ESTABLISHED'
            and result['prior_step42_status_unchanged'] == 'DIAGNOSTIC_ALREADY_EXPOSED_PERIODS',
            'Aggregate primary conjunction, prior-result scope or independent-audit status differs')
    for field, stem, name in [('assessment_files', 'assessment', 'assessment43.json'),
                              ('misbinding_files', 'assessment', 'misbinding43.json'),
                              ('cost_files', 'benchmarks', 'cost43.json')]:
        require(result[field] == {task: reference(ROOT / f'output/{stem}/{task}/{name}') for task in TASKS},
                'Aggregate task evidence reference differs')
    for field, path in [('selected_roster', 'output/selected_roster43.json'),
                         ('selection_seal', 'output/selection_seal43.json'),
                         ('preassessment_audit', 'audit/preassessment43.json')]:
        require(result[field] == reference(ROOT / path), 'Aggregate selected-model audit binding differs')
    aggregate = read(ROOT / 'output/aggregate_seal43.json')
    require(aggregate['schema'] == 'step43-aggregate-seal-v1'
            and aggregate['protocol_seal_sha256'] == sha(ROOT / 'protocol_seal43.json'), 'Aggregate evidence seal differs')
    verify_bindings(aggregate['bound_files'])
    expected = {'output/result43.json', *(ref['path'] for field in ('assessment_files', 'cost_files', 'misbinding_files') for ref in result[field].values())}
    require(set(aggregate['bound_files']) == expected, 'Aggregate seal does not exactly close result and assay evidence')
    # The current auditor stdout and controller log are still being appended.
    # Bind only completed immutable artifacts, and bind the controller prefix.
    active_logs = {'output/logs/26_final_audit.stdout.txt', 'output/logs/26_final_audit.stderr.txt',
                   'output/execution_events43.jsonl'}
    bindings = [path for path in (ROOT / 'output').rglob('*') if path.is_file()
                and path.relative_to(ROOT).as_posix() not in active_logs]
    bindings += [ROOT / name for name in ('protocol_seal43.json', 'protocol43.json',
                                         'audit/preflight43.json', 'audit/preassessment43.json')]
    return publish_gate(ROOT / 'audit/postrun_audit43.json', 'final', bindings,
                        selection_seal_sha256=sha(ROOT / 'output/selection_seal43.json'),
                        result_sha256=sha(result_path), tasks=tasks, cost_checks=costs,
                        primary_predictive_criterion_met=predictive,
                        primary_certificate_coverage_met=coverage,
                        primary_criterion_met=established,
                        primary_status='ESTABLISHED_ON_THREE_FIXED_PERIODS' if established else 'NOT_ESTABLISHED',
                        primary_tolerances_checked=6, primary_tolerance='20*SSE_raw <= 21*SSE_reference',
                        zero_reference_rule='Cross multiplication; zero passes only with zero raw loss; ratio remains null',
                        primary_seed=3901, secondary_promoted_to_primary=False,
                        experimental_units='270 fixed dated targets nested within three dependent series; seeds are computational views',
                        confidence_intervals_or_population_inference=False,
                        chronology_claim='Consistency of byte-bound retained records and ordered event chains; no external attestation or authenticity claim',
                        controller=controller,
                        scope=SCOPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'preassessment', 'final'))
    args = parser.parse_args()
    try:
        if args.phase == 'preflight':
            protocol, _, _, _ = protocol_check()
            audit_controller(protocol, 0)
        elif args.phase == 'preassessment':
            protocol, _, _, _ = protocol_check()
            audit_controller(protocol, 15)
        result = globals()[args.phase]()
        print(json.dumps({'status': result['status'], 'phase': args.phase,
                          'primary_status': result.get('primary_status')}), flush=True)
    except Exception as failure:
        import traceback
        path = ROOT / f'audit/{args.phase}_failed43.json'
        record = {'schema': 'step43-independent-audit-failure-v1', 'status': 'FAIL_RETAINED',
                  'phase': args.phase, 'utc': datetime.now(timezone.utc).isoformat(),
                  'error_type': type(failure).__name__, 'reason': str(failure), 'traceback': traceback.format_exc(),
                  'auditor_sha256': sha(Path(__file__))}
        if not path.exists():
            save(path, record)
        print(json.dumps(record), flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
