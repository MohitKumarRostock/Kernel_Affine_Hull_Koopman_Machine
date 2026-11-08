"""Artificial-only checks of the independent Step 43 arithmetic/audit kernels.

No study manifest, public measurement archive, real cohort, or fitted scientific
model is accessed. The retained legacy producer is used only to construct small
artificial examples; the auditor independently reconstructs their mathematics.
"""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

import audit43 as audit
import affine41
import certificate41 as cert
import finite_feature35 as ff

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    checks = 0
    rejected = 0

    def check(value, message):
        nonlocal checks
        checks += 1
        if not value:
            raise AssertionError(message)

    def fails(function, message):
        nonlocal rejected
        try:
            function()
        except (audit.AuditError, ValueError, KeyError):
            rejected += 1
            check(True, message)
        else:
            check(False, message)

    # Exact finite-cohort margin, including the explicitly frozen zero case.
    for raw, reference, expected in [(21, 20, True), (F(21, 20), 1, True),
                                      (F(21, 20) + F(1, 10**30), 1, False),
                                      (0, 0, True), (F(1, 10**30), 0, False), (0, 1, True)]:
        record = audit.tolerance(raw, reference)
        check(record['within_five_percent'] is expected, 'Exact margin or zero-reference behavior')
        check((record['sse_ratio'] is None) == (reference == 0), 'Zero-reference display rule')
    fails(lambda: audit.tolerance(-1, 1), 'Negative SSE rejected')

    # Pooled SSE selects b; averaging fold MSE would incorrectly select a.
    folds = [{'a': {'n': 9, 'sse': '9'}, 'b': {'n': 9, 'sse': '0'}},
             {'a': {'n': 1, 'sse': '0'}, 'b': {'n': 1, 'sse': '4'}}]
    winner, rows = audit.choose_pooled(folds, ['a', 'b'])
    check(winner == {'candidate_id': 'b', 'n': 10, 'sse': '4', 'mse': '2/5'}, 'Unequal-fold pooled selector')
    tie, _ = audit.choose_pooled([{'z': {'n': 1, 'sse': '1'}, 'a': {'n': 1, 'sse': '1'}}], ['z', 'a'])
    check(tie['candidate_id'] == 'a', 'Lexical tie independent of row order')
    fails(lambda: audit.choose_pooled(folds, ['a', 'a']), 'Duplicate selection candidate rejected')

    risk = audit.exact_risk([0.5.hex(), (-0.25).hex()], [F(1, 3), F(1, 4)])
    check(risk['sse'] == '5/18' and risk['mse'] == '5/36' and risk['mae'] == '1/3', 'Independent exact loss')
    check(len(audit.affine_configs()) == 90 and len(audit.maps_roster()) == 18, 'Declared affine search roster')
    check(len({x['candidate_id'] for x in audit.affine_configs()}) == 90, 'Unique candidate identities')

    # Known rational reference, exact clipping and signed-zero content binding.
    fitted = ff.make_affine_head(['2/3', '-1/4'], '1/7', feature_id='artificial:known', project=False)
    deployment = cert.make_deployment(fitted['head'], target_min='-5', span='7', map_sha256='a'*64,
                                     map_bytes=123, candidate_id='artificial:known')
    audit.audit_deployment(deployment)
    known = [([0.25, 0.5], F(-89, 24)), ([4.0, -1.0], F(2)), ([-4.0, 1.0], F(-5))]
    for vector, expected in known:
        check(audit.exact_prediction(fitted['head'], np.array(vector), '-5', '7') == expected,
              'Independent integer clipped reference')
    check(audit.feature_digest(np.array([0.0])) != audit.feature_digest(np.array([-0.0])),
          'Feature binding retains signed zero')

    # Exact projection and recentering, with a separate floating solve replay.
    artificial_fits = 0
    for scale, constant in [(0.001, False), (1.0, False), (1.0, True)]:
        Z = np.array([[scale * (i-4), scale * ((i*i) % 7-3), scale*((i % 3)-1)] for i in range(9)], dtype=np.float64)
        targets = [F(1, 2) if constant else F(i, 8) for i in range(9)]
        for penalty in ('1/1000000', '1/100', '100'):
            receipt = affine41.fit_affine(Z, targets, feature_id='artificial:ridge', ridge_lambda=penalty,
                                         training_binding={'artificial': True})
            artificial_fits += 1
            checked = audit.audit_affine_fit(receipt, Z, targets, ridge_lambda=penalty,
                                             feature_id='artificial:ridge', training_binding={'artificial': True})
            check(checked['head_id'] == receipt['head_id'], 'Independent projection/fit reconstruction')
            check(F(checked['coefficient_squared_norm']) <= 4, 'Exact projected norm')
            broken = deepcopy(receipt)
            broken['projection']['center_target'] = str(F(broken['projection']['center_target']) + F(1, 100))
            fails(lambda: audit.audit_affine_fit(broken, Z, targets, ridge_lambda=penalty), 'Changed training center rejected')

    # Legacy checker and third, independent integer audit agree on valid and
    # invalid reported values; a self-consistent over-budget packet is retained
    # as a legitimate rejection rather than an audit implementation error.
    packet_cases = 0
    accepted = rejected_over_budget = 0
    for head_case in [('1/3', '0'), ('0', '1/2')]:
        fit = ff.make_affine_head([head_case[0]], head_case[1], feature_id='artificial:certificate', project=False)
        deploy = cert.make_deployment(fit['head'], target_min='0', span='1', map_sha256='b'*64,
                                      map_bytes=100, candidate_id='artificial:certificate')
        prepared = cert.prepare_deployment(deploy, expected_deployment_id=deploy['deployment_id'], expected_map_sha256='b'*64)
        vector = np.array([1.0])
        honest = cert.predict_float(prepared, vector)
        values = [honest, float(np.float32(honest)), honest + 1e-4]
        for value in values:
            packet = cert.produce_certificate(prepared, vector, value, query_id='artificial:q0')
            record = audit.audit_packet(deploy, vector, packet, query_id='artificial:q0', reported_hex=value.hex())
            try:
                verified = cert.verify_certificate(prepared, vector, packet, expected_query_id='artificial:q0', reported_prediction=value)
                legacy_accepted = True
            except cert.CertificateError as error:
                check(str(error) == 'Reported prediction exceeds the frozen discrepancy budget', 'Only intended budget rejection')
                verified = {'status': 'REJECTED_OVER_BUDGET', 'reason': str(error)}
                legacy_accepted = False
            check(record['accepted'] is legacy_accepted, 'Independent exact budget predicate')
            check(verified == audit.expected_verification(deploy, vector, query_id='artificial:q0', reported_hex=value.hex()),
                  'Complete independent verification record')
            check(audit.binding_only(deploy, vector, packet, query_id='artificial:q0', reported_hex=value.hex()),
                  'Self-consistent numerical violation can pass binding-only control')
            packet_cases += 1
            accepted += legacy_accepted
            rejected_over_budget += not legacy_accepted
            mutated = deepcopy(packet)
            mutated['absolute_error_upper_hex'] = (1.0).hex()
            fails(lambda: audit.audit_packet(deploy, vector, mutated, query_id='artificial:q0', reported_hex=value.hex()),
                  'Loose/error-bound packet rejected')
            wrong = dict(packet, query_id='artificial:other')
            fails(lambda: audit.audit_packet(deploy, vector, wrong, query_id='artificial:q0', reported_hex=value.hex()),
                  'Wrong expected query rejected')
        check(audit.float32_hex(honest.hex()) == float(np.float32(honest)).hex(), 'Prescribed output-only quantization')
    check(accepted > 0 and rejected_over_budget > 0, 'Both valid and invalid numerical cases covered')

    # Serialization and event evidence reject changed order, values, or bytes.
    for text in ['nan', 'inf', '0x1p+0', '0x1.0000000000000p+0 ']:
        fails(lambda t=text: audit.binary64(t), 'Noncanonical binary64 rejected')
    for text in ['2/2', '0.5', ' 1']:
        fails(lambda t=text: audit.q(t), 'Noncanonical rational rejected')
    with tempfile.TemporaryDirectory(prefix='step43_audit_artificial_') as directory:
        path = Path(directory) / 'events.jsonl'
        first = {'index': 0, 'kind': 'artificial_begin', 'utc': '2026-01-01T00:00:00+00:00',
                 'monotonic_ns': 1, 'previous_event_sha256': None}
        line = audit.canonical_bytes(first)
        second = {'index': 1, 'kind': 'artificial_end', 'utc': '2026-01-01T00:00:01+00:00',
                  'monotonic_ns': 2, 'previous_event_sha256': __import__('hashlib').sha256(line).hexdigest()}
        path.write_bytes(line + audit.canonical_bytes(second))
        check(len(audit.validate_events(path)) == 2, 'Canonical event chain')
        second['previous_event_sha256'] = 'f'*64
        path.write_bytes(line + audit.canonical_bytes(second))
        fails(lambda: audit.validate_events(path), 'Broken hash ancestry rejected')
    check(audit.linear_quantile([0, 10, 20], 0.95) == 19.0, 'Independent linear percentile')
    check(audit.linear_quantile([1, 9], 0.5) == 5.0, 'Independent even-count median')

    # Reuse the explicitly manufactured producer component fixture to exercise
    # the actual bank/score/selection/assay schemas. No source gate is forged,
    # no natural archive is opened, and no additional candidate is fitted.
    fixture = ROOT / 'analysis/producer_fixture43'
    notice = audit.read(fixture / 'fixture_manifest43.json')
    check(notice['all_values_manufactured'] is True and notice['scientific_audit_gate'] is False
          and notice['scope'] == 'ARTIFICIAL_COMPONENT_ONLY', 'Manufactured producer fixture provenance')
    actual_sha, actual_bound = audit.sha, audit.bound_ref
    original_root = audit.ROOT
    def fixture_sha(path):
        path = Path(path)
        if path == fixture / 'protocol_seal43.json':
            return notice['artificial_protocol_identifier']
        if path == fixture / 'source/representations35.py':
            return actual_sha(ROOT / 'source/representations35.py')
        return actual_sha(path)
    def cohort(stage, partition):
        path = fixture / f'output/data43/household_power2008/{stage}/{partition}.json'
        record = audit.read(path)
        return {'record': record, 'X': np.load(fixture / record['X']['path'], allow_pickle=False),
                'targets': list(map(F, record['raw_targets'])), 'y': list(map(F, record['y'])),
                'normalization': audit.read(fixture / record['normalization']['path']),
                'reference': {'path': path.relative_to(fixture).as_posix(), 'sha256': actual_sha(path),
                              'bytes': path.stat().st_size}}
    artifact_cases = 0
    with patch.object(audit, 'ROOT', fixture), patch.object(audit, 'sha', fixture_sha), \
            patch.object(audit, 'bound_ref', lambda ref, **kw: actual_bound(ref, root=fixture)):
        bank = audit.audit_bank('household_power2008', 'f1', 3901, cohort('f1', 'train'))
        check(len(bank['heads']) == 90 and len(bank['baselines']) == 44, 'Actual manufactured complete bank independent audit')
        scores = audit.audit_validation('household_power2008', 'f1', 3901, cohort('f1', 'validation'), bank)
        check(len(scores) == 134, 'Actual manufactured validation independent score reconstruction')
        artifact_cases += 134
        del bank
        for seed in audit.SEEDS:
            folds = {stage: audit.read(fixture / f'output/household_power2008/{stage}/seed_{seed}/validation_scores43.json')['candidate_rows']
                     for stage in audit.FOLDS}
            selections = audit.audit_selection('household_power2008', seed, folds)
            check(len(selections) == 6, 'Actual manufactured pooled selection interface')
            artifact_cases += 6
        selected = audit.read(fixture / 'output/selected_roster43.json')['tasks']['household_power2008']['selections']
        bank = audit.audit_bank('household_power2008', 'final', 3901, cohort('final', 'train'), selected)
        check(len(bank['heads']) == 3, 'Actual selected manufactured final bank independent audit')
        assessment = audit.read(fixture / 'output/assessment/household_power2008/assessment43.json')
        queries = [audit.read(fixture / row['prediction']['path']) for row in assessment['queries']]
        for query in queries:
            for eid, evaluation in query['evaluations'].items():
                if evaluation['recipe'] != 'affine':
                    continue
                z = np.load(fixture / evaluation['features']['path'], allow_pickle=False)[0]
                checked = audit.audit_output_assay(evaluation['assay'], evaluation['deployment'], z,
                                                   query_id=query['query_id'], reported_hex=evaluation['reported_physical_hex'])
                check(checked['binary64']['binding_only_accepted'] is True, 'Actual manufactured honest/float32 assay schema')
                artifact_cases += 2
        challenge = audit.read(fixture / 'output/assessment/household_power2008/misbinding43.json')
        for eid, record in challenge['records'].items():
            evaluation = queries[0]['evaluations'][eid]
            z = np.load(fixture / evaluation['features']['path'], allow_pickle=False)[0]
            stale = np.load(fixture / queries[1]['evaluations'][eid]['features']['path'], allow_pickle=False)[0]
            checked = audit.audit_misbinding(record, evaluation['deployment'], z, stale,
                                             query_id=queries[0]['query_id'], reported_hex=evaluation['reported_physical_hex'])
            check(checked['applicable'] == checked['rejected_as_expected'], 'Actual manufactured binding challenge schema')
            artifact_cases += 3
    check(audit.ROOT == original_root and audit.sha is actual_sha and audit.bound_ref is actual_bound,
          'Manufactured fixture redirects restored')
    # Cost component closure uses already measured manufactured fixture samples.
    # Its missing scientific workflow is represented only by in-memory labelled
    # event fixtures; no fake gate is written or accepted by a gate function.
    cost_fixture = ROOT / 'analysis/cost_fixture43'
    cost_notice = audit.read(cost_fixture / 'fixture_notice43.json')
    check(cost_notice['status'] == 'ARTIFICIAL_STUB_NOT_A_SCIENTIFIC_GATE', 'Manufactured cost fixture identity')
    task = audit.TASKS[0]
    bank_record = audit.read(cost_fixture / f'output/{task}/final/seed_3901/fits43.json')
    cost_bank = {'record': bank_record,
                 'heads': audit.read(cost_fixture / bank_record['heads']['path']),
                 'baseline_rows': {row['candidate_id']: row for row in bank_record['baseline_rows']},
                 'configurations': {row['candidate_id']: row for row in bank_record['affine_rows'] + bank_record['baseline_rows']}}
    cost_report = audit.read(cost_fixture / f'output/assessment/{task}/assessment43.json')
    cost_queries = [audit.read(cost_fixture / row['prediction']['path']) for row in cost_report['queries']]
    cost_selected = audit.read(cost_fixture / 'output/selected_roster43.json')['tasks'][task]['selections']
    cost_events = audit.validate_events(cost_fixture / 'output/events43.jsonl')
    start_event = next(row for row in cost_events if row['kind'] == 'cost_started')
    cost_events += [{'kind': 'producer_phase_completed', 'phase': phase, 'task': name,
                     'utc': start_event['utc'], 'monotonic_ns': start_event['monotonic_ns'] - 2,
                     'fixture_scope': 'ARTIFICIAL_IN_MEMORY_TIMING_COMPONENT_ONLY'}
                    for phase in ('assess', 'misbinding') for name in audit.TASKS]
    actual_bindings = audit.verify_bindings
    with patch.object(audit, 'ROOT', cost_fixture), \
            patch.object(audit, 'bound_ref', lambda ref, **kw: actual_bound(ref, root=cost_fixture)), \
            patch.object(audit, 'verify_bindings', lambda refs, **kw: actual_bindings(refs, root=cost_fixture)):
        checked = audit.audit_cost(task, {'report': cost_report, 'query_records': cost_queries,
                                          'selected': cost_selected, 'banks': {3901: cost_bank}},
                                   audit.read(cost_fixture / 'design43.json'), cost_events)
        check(checked['selected_deduplicated_models'] == 2 and checked['retained_raw_timing_samples_checked'] == 200,
              'Actual manufactured cost schema, primitive medians, p95, byte accounting and deduplication')
    check(audit.ROOT == original_root and audit.verify_bindings is actual_bindings,
          'Manufactured cost component redirects restored')
    for seed in audit.SEEDS:
        configs = audit.baseline_configs(seed)
        check(len(configs) == 44 and sum(row['kind'] == 'ridge' for row in configs) == 5,
              'Independently expanded conventional comparison pool')
    check(audit.temporal_indices(90, 32)[:4] == [1, 4, 7, 9], 'Original query index differs from timing sample ordinal')
    fails(lambda: audit.timing_samples([1.0]*10, 10), 'Noninteger raw timing samples rejected')

    result = {'schema': 'step43-independent-auditor-artificial-checks-v1', 'status': 'PASS',
              'checks': checks, 'deliberate_invalid_cases_rejected': rejected,
              'artificial_ridge_fits': artificial_fits,
              'artificial_independent_ridge_replays': artificial_fits,
              'artificial_certificate_cases': packet_cases,
              'artificial_accepted_packets': accepted,
              'artificial_correct_budget_rejections': rejected_over_budget,
              'new_measurement_values_read': 0, 'real_model_fits': 0,
              'manufactured_artifact_cases': artifact_cases,
              'manufactured_component_fixture': 'analysis/producer_fixture43',
              'manufactured_cost_raw_samples_checked': 200,
              'manufactured_fixture_manifest_sha256': audit.sha(fixture / 'fixture_manifest43.json'),
              'scope': 'Artificial arithmetic and actual manufactured bank/validation/selection/assay component interfaces; no full scientific controller rehearsal or scientific gate is fabricated.',
              'auditor_sha256': audit.sha(ROOT / 'source/audit43.py'),
              'test_sha256': audit.sha(Path(__file__)),
              'wall_seconds': time.perf_counter() - started}
    path = ROOT / 'analysis/audit_artificial_checks43.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    # This is a prefreeze artificial receipt, regenerated if audited source is
    # refined before the final protocol/source seal. Scientific files are never
    # overwritten by this test.
    path.write_bytes(audit.canonical_bytes(result))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
