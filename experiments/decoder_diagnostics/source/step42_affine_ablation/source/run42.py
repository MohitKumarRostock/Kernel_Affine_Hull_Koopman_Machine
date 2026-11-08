"""Single generation pass over the frozen retrospective head factorial."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from fractions import Fraction as F
import signal
import sys
import time
from pathlib import Path
import numpy as np
import common42 as c
import heads42 as h
import prepare42 as d


def progress(event, **fields):
    print(c.canonical_bytes({'event': event, **fields}).decode(), end='', flush=True)


def _timeout(*args):
    raise TimeoutError('Frozen Step42 phase budget exhausted')


def _start(directory, phase, **fields):
    c.assert_frozen()
    p = c.read(c.ROOT / 'protocol42.json')
    c.save(directory / (phase + '_started42.json'),
           {'phase': phase, 'utc': c.utc(), 'protocol_seal_sha256': c.sha(c.ROOT / 'protocol_seal42.json'),
            'study_status': p['status'], **fields})
    signal.signal(signal.SIGALRM, _timeout)
    seconds = p['execution'].get(phase + '_budget_seconds', 1800)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return time.perf_counter()


def _finish(directory, phase, start, **fields):
    signal.setitimer(signal.ITIMER_REAL, 0)
    c.save(directory / (phase + '_completed42.json'),
           {'phase': phase, 'status': 'COMPLETE', 'utc': c.utc(),
            'wall_seconds': time.perf_counter() - start, **fields})
    seal_path = directory / (phase + '_seal42.json')
    files = [p for p in directory.rglob('*') if p.is_file() and p != seal_path]
    record = {'schema': 'step42-' + phase + '-seal-v1', 'utc': c.utc(),
              'protocol_seal_sha256': c.sha(c.ROOT / 'protocol_seal42.json'),
              'bound_files': {c.relative(p): c.sha(p) for p in sorted(files)}}
    ref = c.save(seal_path, record)
    progress('COMPLETE', phase=phase, **fields, seal=ref)


def stage(task, stage_name):
    dest = c.ROOT / 'output' / task / stage_name
    started = _start(dest, 'stage', task=task, stage=stage_name)
    if stage_name == 'final':
        gate = c.read(c.ROOT / 'audit' / task / 'selection_gate42.json')
        if gate['status'] != 'PASS':
            raise RuntimeError('Independent selection gate must pass before final evaluation')
    context = d.load_stage(task, stage_name, root=c.ROOT)
    normal = context['normalization']
    configs = c.read(c.ROOT / 'protocol42.json')['configurations']
    bases, variants, predictions, scores = {}, {}, [], []
    legacy_heads = legacy_predictions = 0
    for index, cfg in enumerate(configs):
        name = cfg['map']; cid = cfg['candidate_id']; key = cfg['base_key']
        meta = context['maps'][name]
        if key not in bases:
            bases[key] = h.fit_base(context['train']['features'][name], context['train']['y'],
                                   feature_id=meta['feature_id'], ridge_lambda=cfg['ridge_lambda'],
                                   training_binding={'task': task, 'stage': stage_name,
                                                     'cohort_path': context['train']['path'],
                                                     'cohort_sha256': context['train']['sha256'],
                                                     'feature_path': meta['training_features']['path'],
                                                     'feature_sha256': meta['training_features']['sha256'],
                                                     'map_sha256': meta['map']['sha256']})
        variant = h.make_variant(bases[key], project=cfg['project'], recenter=cfg['recenter'],
                                 clip=cfg['clip'], kappa='2', bits=80)
        variants[cid] = variant
        pp = h.prepare_prediction(variant, target_min=normal['target_min'], span=normal['span'])
        operational, exact = [], []
        for zz in context['evaluation']['features'][name]:
            operational.append(h.predict_details(pp, zz))
            exact.append(h.exact_details(pp, zz))
        row = {'candidate_id': cid,
               'reported_physical_hex': [x['reported_physical_hex'] for x in operational],
               'raw_normalized_hex': [x['raw_normalized_hex'] for x in operational],
               'exact_physical': [x['exact_physical'] for x in exact],
               'exact_raw_normalized': [x['raw_normalized'] for x in exact],
               'clipping_active': [x['clipping_active'] for x in operational],
               'clipping_side': [x['clipping_side'] for x in operational],
               'exact_clipping_active': [x['clipping_active'] for x in exact],
               'exact_clipping_side': [x['clipping_side'] for x in exact]}
        predictions.append(row)
        difference = [abs(c.exact_number(a) - F(b))
                      for a, b in zip(row['reported_physical_hex'], row['exact_physical'])]
        tr = variant['transformation']
        score = {**cfg,
                 'physical_risk': c.score(row['reported_physical_hex'], context['evaluation']['raw_targets'], hex_predictions=True),
                 'exact_reference_risk': c.score(row['exact_physical'], context['evaluation']['raw_targets']),
                 'projection_active': tr['projection_active'],
                 'projection_would_activate': F(tr['source_squared_norm']) > 4,
                 'clipping_count': sum(row['clipping_active']),
                 'exact_clipping_count': sum(row['exact_clipping_active']),
                 'max_abs_numerical_error': str(max(difference)),
                 'coefficient_squared_norm': tr['coefficient_squared_norm'],
                 'source_squared_norm': tr['source_squared_norm'],
                 'recenter_delta': tr['recenter_delta'],
                 'structural_claims': variant['structural_claims']}
        scores.append(score)
        if cfg['lambda_token'] == '1e-6' and cfg['project'] and cfg['recenter'] and cfg['clip']:
            old = c.read(c.ROOT / meta['inherited_head']['path'])
            for field in ('coefficients', 'intercept', 'dimension', 'feature_id', 'output_interval'):
                if variant['head'][field] != old[field]:
                    raise RuntimeError('Inherited head replay mismatch: ' + cid + '/' + field)
            inherited = context['inherited_predictions']['predictions'][name + '::affine']
            if row['reported_physical_hex'] != [x['reported_physical_hex'] for x in inherited]:
                raise RuntimeError('Inherited reported predictions changed: ' + cid)
            if row['exact_physical'] != [x['exact_physical'] for x in inherited]:
                raise RuntimeError('Inherited exact predictions changed: ' + cid)
            legacy_heads += 1; legacy_predictions += len(inherited)
        if (index + 1) % 40 == 0:
            progress('GRID', task=task, stage=stage_name, completed=index + 1, total=len(configs))
    header = {'task': task, 'stage': stage_name, 'role': context['evaluation_role'],
              'input_manifest_sha256': c.sha(c.ROOT / 'inputs/manifest42.json')}
    c.save(dest / 'bases42.json', {**header, 'bases': bases})
    c.save(dest / 'variants42.json', {**header, 'variants': variants})
    c.save(dest / 'predictions42.json', {**header, 'query_ids': context['evaluation']['ids'], 'rows': predictions})
    c.save(dest / 'scores42.json', {**header, 'n': len(context['evaluation']['ids']), 'rows': scores})
    c.save(dest / 'legacy_recovery42.json', {**header, 'status': 'PASS', 'head_count': legacy_heads,
                                           'prediction_count': legacy_predictions,
                                           'exact_head_and_prediction_recovery': True})
    _finish(dest, 'stage', started, task=task, stage=stage_name, base_solves=len(bases),
            variants=len(variants), evaluation_predictions=sum(len(x['reported_physical_hex']) for x in predictions),
            legacy_heads=legacy_heads, legacy_predictions=legacy_predictions)


def selection(task):
    dest = c.ROOT / 'output' / task / 'selection'
    started = _start(dest, 'selection', task=task)
    folds = {}
    for name in ('f1', 'f2', 'f3'):
        if c.read(c.ROOT / 'audit' / task / name / 'stage_gate42.json')['status'] != 'PASS':
            raise RuntimeError('Independent development gate required')
        folds[name] = c.read(c.ROOT / 'output' / task / name / 'scores42.json')
    previous = d.load_stage(task, 'f3', root=c.ROOT)
    lookup = {name: {r['candidate_id']: r for r in data['rows']} for name, data in folds.items()}
    n = sum(x['n'] for x in folds.values())
    rows = []
    for cfg in c.read(c.ROOT / 'protocol42.json')['configurations']:
        cid = cfg['candidate_id']
        sse = sum((F(lookup[name][cid]['physical_risk']['sse']) for name in folds), F(0))
        exact_sse = sum((F(lookup[name][cid]['exact_reference_risk']['sse']) for name in folds), F(0))
        rows.append({**cfg, 'n': n, 'pooled_physical_sse': str(sse),
                     'pooled_physical_mse': str(sse / n), 'pooled_exact_reference_sse': str(exact_sse),
                     'pooled_exact_reference_mse': str(exact_sse / n),
                     'legacy_f3_deployment_bytes': previous['maps'][cfg['map']]['original_deployment_payload_bytes'],
                     'folds': {name: {'n': data['n'], 'physical_risk': lookup[name][cid]['physical_risk'],
                                      'exact_reference_risk': lookup[name][cid]['exact_reference_risk'],
                                      'projection_active': lookup[name][cid]['projection_active'],
                                      'clipping_count': lookup[name][cid]['clipping_count']}
                               for name, data in folds.items()}})
    specs = c.read(c.ROOT / 'protocol42.json')['selection']['specifications']
    selections = {spec['label']: c.select(rows, spec) for spec in specs}
    old = previous['inherited_selection']['selections']['affine_rolling']
    chosen = selections['original41_replay']
    original_name = chosen['candidate_id'].split('::')[0] + '::affine'
    if original_name != old['candidate_id'] or chosen['pooled_physical_mse'] != old['pooled_physical_mse']:
        raise RuntimeError('Original Step41 selector is not exactly recovered')
    record = {'schema': 'step42-development-selection-v1', 'task': task,
              'study_status': 'DIAGNOSTIC_ALREADY_EXPOSED_PERIODS', 'development_n': n,
              'candidate_rows': rows, 'selections': selections,
              'fold_score_sha256': {name: c.sha(c.ROOT / 'output' / task / name / 'scores42.json') for name in folds},
              'selection_uses_development_only': True, 'assessment_reselection': False,
              'assessment_outcomes_previously_known_from_step41': True,
              'legacy_selector_recovered': True}
    # Stable location used by downstream stages; phase receipts stay in a subdir.
    ref = c.save(c.ROOT / 'output' / task / 'selection42.json', record)
    c.save(dest / 'selection_binding42.json', ref)
    _finish(dest, 'selection', started, task=task, selectors=len(selections),
            selected_main=selections['recipe_P1R1C1']['candidate_id'])


def bridge(task):
    dest = c.ROOT / 'output' / task / 'certificate_bridge'
    started = _start(dest, 'bridge', task=task)
    if c.read(c.ROOT / 'audit' / task / 'final' / 'stage_gate42.json')['status'] != 'PASS':
        raise RuntimeError('Independent final stage gate required')
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'legacy42'))
    import certificate41 as cert
    p = c.read(c.ROOT / 'protocol42.json')
    selections = c.read(c.ROOT / 'output' / task / 'selection42.json')['selections']
    aliases = {label: selections[label]['candidate_id']
               for label in p['assurance']['certificate_bridge_selectors']}
    variants = c.read(c.ROOT / 'output' / task / 'final' / 'variants42.json')['variants']
    pred = {row['candidate_id']: row for row in c.read(c.ROOT / 'output' / task / 'final' / 'predictions42.json')['rows']}
    context = d.load_stage(task, 'final', root=c.ROOT)
    records = {}
    for cid in sorted(set(aliases.values())):
        variant = variants[cid]; name = cid.split('::')[0]; meta = context['maps'][name]
        head = h.as_legacy_head(variant)
        deployment = cert.make_deployment(head, target_min=context['normalization']['target_min'],
                                          span=context['normalization']['span'],
                                          map_sha256=meta['map']['sha256'], map_bytes=meta['map_payload_bytes'],
                                          candidate_id=cid)
        pp = cert.prepare_deployment(deployment, expected_deployment_id=deployment['deployment_id'],
                                     expected_map_sha256=meta['map']['sha256'])
        rows = []
        for i, (qid, zz) in enumerate(zip(context['evaluation']['ids'], context['evaluation']['features'][name])):
            reported = cert.predict_float(pp, zz)
            if reported.hex() != pred[cid]['reported_physical_hex'][i]:
                raise RuntimeError('Certificate bridge changed actual prediction')
            packet = cert.produce_certificate(pp, zz, reported, query_id=qid)
            budget = F(context['normalization']['span']) / 2**40
            error = abs(F.from_float(reported) - F(pred[cid]['exact_physical'][i]))
            expected_acceptance = error <= budget
            try:
                checked = cert.verify_certificate(pp, zz, packet, expected_query_id=qid,
                                                   reported_prediction=reported)
                if not expected_acceptance:
                    raise RuntimeError('Inherited checker accepted an over-budget prediction')
                accepted = True
            except cert.CertificateError as exc:
                if expected_acceptance or str(exc) != 'Reported prediction exceeds the frozen discrepancy budget':
                    raise RuntimeError('Inherited checker rejected for an unexpected reason') from exc
                checked = {'status': 'REJECTED_OVER_BUDGET', 'reason': str(exc)}
                accepted = False
            rows.append({'query_id': qid, 'reported_physical_hex': reported.hex(),
                         'exact_physical': pred[cid]['exact_physical'][i],
                         'certificate': packet, 'verification': checked,
                         'accepted': accepted, 'expected_acceptance': expected_acceptance,
                         'absolute_physical_error': str(error),
                         'packet_json_bytes': len(cert.encode_certificate(packet))})
        records[cid] = {'candidate_id': cid, 'map': name, 'head': head, 'deployment': deployment,
                        'aliases': [label for label, value in aliases.items() if value == cid],
                        'query_ids': context['evaluation']['ids'], 'rows': rows,
                        'physical_error_budget': str(F(context['normalization']['span']) / 2**40)}
    c.save(dest / 'certificates42.json', {'schema': 'step42-inherited-certificate-bridge-v1',
                                        'task': task, 'aliases': aliases, 'records': records,
                                        'certificate_source_sha256': c.sha(Path(cert.__file__)),
                                        'all_query_certificates_verified_by_inherited_checker': all(row['accepted'] for record in records.values() for row in record['rows']),
                                        'future_label_coverage': False, 'native_transform_error_enclosed': False})
    _finish(dest, 'bridge', started, task=task, unique_candidates=len(records),
            packets=sum(len(x['rows']) for x in records.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['stage', 'selection', 'bridge'])
    parser.add_argument('--task', choices=c.TASKS, required=True)
    parser.add_argument('--stage', choices=c.STAGES)
    args = parser.parse_args()
    if args.phase == 'stage':
        if args.stage is None:
            parser.error('--stage is required')
        stage(args.task, args.stage)
    elif args.phase == 'selection':
        selection(args.task)
    else:
        bridge(args.task)


if __name__ == '__main__':
    main()
