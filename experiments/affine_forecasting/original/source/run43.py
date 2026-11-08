"""Frozen producer for selected-policy transfer; assessment labels are gated."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[_name] = '1'

import argparse
from fractions import Fraction as F
import gzip
import hashlib
import math
import pickle
from pathlib import Path
import signal
import time
import traceback

import numpy as np

import common43 as c
import data43 as d
import model43 as m

ROOT = c.ROOT
OUT = ROOT / 'output'


def progress(kind, **fields):
    print(c.canonical_bytes({'kind': kind, **fields}).decode(), end='', flush=True)


def save_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value)
    return c.file_ref(path)


def save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.save(stream, np.asarray(value, dtype=np.float64), allow_pickle=False)
    return c.file_ref(path)


def ref_read(record):
    path = ROOT / record['path']
    if c.sha_file(path) != record['sha256']:
        raise RuntimeError('Referenced bytes changed: ' + record['path'])
    return path


def risk(values, targets):
    values, targets = list(map(F, values)), list(map(F, targets))
    if not values or len(values) != len(targets):
        raise ValueError('Risk requires a nonempty matched query cohort')
    errors = [a-b for a, b in zip(values, targets)]
    n = len(errors)
    sse = sum((x*x for x in errors), F(0))
    return {'n': n, 'sse': str(sse), 'mse': str(sse/n),
            'rmse': math.sqrt(float(sse/n)),
            'mae': str(sum(map(abs, errors), F(0))/n)}


def reported_risk(hex_values, targets):
    return risk([F.from_float(float.fromhex(x)) for x in hex_values], targets)


def pooled_risk(rows):
    n = sum(row['n'] for row in rows)
    sse = sum((F(row['sse']) for row in rows), F(0))
    sae = sum((F(row['mae'])*row['n'] for row in rows), F(0))
    if n <= 0:
        raise ValueError('Empty development pool')
    return {'n': n, 'sse': str(sse), 'mse': str(sse/n),
            'rmse': math.sqrt(float(sse/n)), 'mae': str(sae/n)}


def compare(raw, reference):
    if raw['n'] != reference['n']:
        raise ValueError('Primary comparison cohorts differ')
    a, b = F(raw['sse']), F(reference['sse'])
    return {'raw_risk': raw, 'reference_risk': reference,
            'mse_ratio': str(a/b) if b else None,
            'within_five_percent_mse': 20*a <= 21*b}


def bank_path(task, stage, seed):
    return OUT / task / stage / ('seed_' + str(seed))


def load_context(task, stage, seed):
    path = bank_path(task, stage, seed) / 'fits43.json'
    bank = c.read_json(path)
    if bank['protocol_seal_sha256'] != c.protocol_identity():
        raise RuntimeError('Fit bank belongs to another protocol')
    maps = {name: m.deserialize_map(ref_read(row['map']).read_bytes())
            for name, row in bank['maps'].items()}
    heads = c.read_json(ref_read(bank['heads']))
    prepared = {cid: m.prepare_deployment(row['deployment'],
                                         expected_deployment_id=row['deployment']['deployment_id'],
                                         expected_map_sha256=row['deployment']['map_sha256'])
                for cid, row in heads.items()}
    baselines, baseline_rows = {}, {}
    for row in bank['baseline_rows']:
        raw = gzip.decompress(ref_read(row['payload']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != row['pickle_sha256']:
            raise RuntimeError('Baseline pickle content identity changed')
        baselines[row['candidate_id']] = pickle.loads(raw)
        baseline_rows[row['candidate_id']] = row
    return {'fits': bank, 'fits_ref': c.file_ref(path), 'maps': maps,
            'heads': heads, 'prepared': prepared, 'baselines': baselines,
            'baseline_rows': baseline_rows}


def _baseline_configuration(row):
    return {key: value for key, value in row.items()
            if key not in ('payload', 'pickle_sha256', 'pickle_bytes', 'recipe', 'baseline_kind')}


def fit_bank(task, stage, seed, cohort):
    started = time.perf_counter()
    destination = bank_path(task, stage, seed)
    destination.mkdir(parents=True, exist_ok=False)
    configurations = m.affine_configs()
    baseline_configs = m.baseline_configs(seed)
    if stage == 'final':
        selected = c.read_json(OUT / 'selected_roster43.json')['tasks'][task]['selections']
        wanted = {selected[f'{seed}::{policy}']['candidate_id'] for policy in c.POLICIES}
        configurations = [row for row in configurations if row['candidate_id'] in wanted]
        baseline_configs = [row for row in baseline_configs if row['candidate_id'] in wanted]
        if len(configurations) != 3 or not 1 <= len(baseline_configs) <= 2:
            raise RuntimeError('Final bank differs from selected family/ordinary/ridge identities')
    X, y, normal = cohort['X'], cohort['y'], cohort['normalization']
    maps, metadata = m.fit_representations(X, seed)
    c.write_json(destination / 'representation_metadata43.json', metadata)
    details, train_features = {}, {}
    for name in sorted({row['map'] for row in configurations}):
        mapping = maps[name]
        payload = m.serialize_map(mapping)
        map_ref = save_bytes(destination / 'maps' / (name + '.pkl'), payload)
        Z = m.transform_one(mapping, X)
        features_ref = save_array(destination / 'features/train' / (name + '.npy'), Z)
        feature_id = 'feature43:' + c.digest({'task': task, 'stage': stage,
                                             'map_sha256': map_ref['sha256'],
                                             'training_feature_sha256': features_ref['sha256']})
        details[name] = {'name': name, 'family': mapping.family, 'scaling': mapping.scaling,
                         'dimension': int(Z.shape[1]), 'feature_id': feature_id,
                         'map': map_ref, 'training_features': features_ref}
        train_features[name] = Z
    heads = {}
    for cfg in configurations:
        detail = details[cfg['map']]
        fitted = m.fit_affine(
            cfg, train_features[cfg['map']], y, normal,
            feature_id=detail['feature_id'], map_sha256=detail['map']['sha256'],
            map_bytes=detail['map']['bytes'],
            training_binding={'task': task, 'stage': stage,
                              'cohort_path': cohort['path'], 'cohort_sha256': cohort['sha256'],
                              'training_features': detail['training_features'],
                              'map_sha256': detail['map']['sha256']})
        heads[cfg['candidate_id']] = {key: fitted[key] for key in ('fit', 'head', 'deployment')}
    c.write_json(destination / 'heads43.json', heads)
    baseline_rows = []
    for cfg in baseline_configs:
        fitted = m.fit_baseline(cfg, X, y, normal)
        raw = pickle.dumps(fitted, protocol=5)
        payload = gzip.compress(raw, compresslevel=6, mtime=0)
        ref = save_bytes(destination / 'baselines' / (cfg['token'] + '.pkl.gz'), payload)
        baseline_rows.append({**cfg, 'recipe': 'baseline', 'baseline_kind': cfg['kind'],
                              'payload': ref, 'pickle_sha256': hashlib.sha256(raw).hexdigest(),
                              'pickle_bytes': len(raw)})
    record = {'schema': 'step43-stage-seed-fit-v1', 'task': task, 'stage': stage, 'seed': seed,
              'protocol_seal_sha256': c.protocol_identity(),
              'training_cohort': c.file_ref(ROOT / cohort['path']),
              'normalization': normal,
              'representation_metadata': c.file_ref(destination / 'representation_metadata43.json'),
              'maps': details, 'heads': c.file_ref(destination / 'heads43.json'),
              'affine_rows': configurations, 'baseline_rows': baseline_rows,
              'wall_seconds': time.perf_counter()-started}
    c.write_json(destination / 'fits43.json', record)
    progress('FIT_BANK_COMPLETE', task=task, stage=stage, seed=seed,
             affine_heads=len(configurations), baseline_fits=len(baseline_rows),
             wall_seconds=record['wall_seconds'])


def validate_bank(task, stage, seed, cohort):
    context = load_context(task, stage, seed)
    destination = bank_path(task, stage, seed)
    X, targets = cohort['X'], cohort['raw_targets']
    Zs, feature_refs = {}, {}
    for name, mapping in context['maps'].items():
        Z = np.vstack([m.transform_one(mapping, X[i:i+1])[0] for i in range(len(X))])
        Zs[name] = Z
        feature_refs[name] = save_array(destination / 'features/validation' / (name + '.npy'), Z)
    predictions, scores = {}, {}
    for cfg in context['fits']['affine_rows']:
        cid = cfg['candidate_id']
        pp, Z = context['prepared'][cid], Zs[cfg['map']]
        physical = [m.predict_one(pp, z).hex() for z in Z]
        references = [str(m.exact_reference(pp, z)) for z in Z]
        predictions[cid] = {'reported_physical_hex': physical, 'exact_physical': references}
        scores[cid] = {'configuration': cfg, 'physical_risk': reported_risk(physical, targets),
                       'exact_reference_risk': risk(references, targets)}
    for row in context['fits']['baseline_rows']:
        cid = row['candidate_id']
        values = [float(m.predict_baseline(context['baselines'][cid], X[i:i+1])[0]).hex()
                  for i in range(len(X))]
        predictions[cid] = {'reported_physical_hex': values, 'exact_physical': None}
        scores[cid] = {'configuration': {**_baseline_configuration(row), 'recipe': 'baseline',
                                        'baseline_kind': row['kind']},
                       'physical_risk': reported_risk(values, targets), 'exact_reference_risk': None}
    pred_path = destination / 'validation_predictions43.json'
    c.write_json(pred_path, {'schema': 'step43-validation-predictions-v1', 'task': task,
                            'stage': stage, 'seed': seed,
                            'protocol_seal_sha256': c.protocol_identity(),
                            'fit_sha256': context['fits_ref']['sha256'],
                            'validation_cohort': c.file_ref(ROOT / cohort['path']),
                            'query_ids': cohort['ids'], 'maps': feature_refs,
                            'candidate_predictions': predictions})
    c.write_json(destination / 'validation_scores43.json',
                 {'schema': 'step43-validation-scores-v1', 'task': task, 'stage': stage,
                  'seed': seed, 'n': len(targets), 'prediction_sha256': c.sha_file(pred_path),
                  'candidate_rows': scores})
    progress('VALIDATION_BANK_COMPLETE', task=task, stage=stage, seed=seed,
             candidates=len(predictions), n=len(targets))


def stage(task, stage):
    if task not in c.TASKS or stage not in c.STAGES:
        raise ValueError('Unknown task/stage')
    destination = OUT / task / stage
    c.write_json(destination / 'stage_started43.json',
                 {'task': task, 'stage': stage, 'utc': c.utc_now(),
                  'protocol_seal_sha256': c.protocol_identity()})
    training = d.load_training(task, stage)
    for seed in c.SEEDS:
        fit_bank(task, stage, seed, training)
    files = [path for path in destination.glob('seed_*/**/*') if path.is_file()]
    c.seal_files(destination / 'fit_seal43.json', files, schema='step43-stage-fit-seal-v1',
                 task=task, stage=stage, seeds=list(c.SEEDS), protocol_seal_sha256=c.protocol_identity())
    c.event('stage_models_sealed', task=task, stage=stage,
            fit_seal=c.file_ref(destination / 'fit_seal43.json'))
    if stage != 'final':
        validation = d.load_validation(task, stage)
        for seed in c.SEEDS:
            validate_bank(task, stage, seed, validation)
    c.write_json(destination / 'stage_completed43.json',
                 {'status': 'COMPLETE', 'task': task, 'stage': stage, 'utc': c.utc_now(),
                  'fit_seal_sha256': c.sha_file(destination / 'fit_seal43.json')})


def select():
    roster = {'schema': 'step43-selected-roster-v1', 'protocol_seal_sha256': c.protocol_identity(),
              'selection_uses_development_only': True, 'assessment_targets_accessed': 0, 'tasks': {}}
    for task in c.TASKS:
        all_selections, selection_refs = {}, []
        for seed in c.SEEDS:
            fold_paths = {s: bank_path(task, s, seed) / 'validation_scores43.json'
                          for s in c.STAGES[:-1]}
            folds = {s: c.read_json(path) for s, path in fold_paths.items()}
            ids = sorted(folds['f1']['candidate_rows'])
            if len(ids) != 134 or any(sorted(x['candidate_rows']) != ids for x in folds.values()):
                raise RuntimeError('Incomplete development score grid')
            rows = {}
            for cid in ids:
                parts = {s: f['candidate_rows'][cid] for s, f in folds.items()}
                configuration = parts['f1']['configuration']
                if any(part['configuration'] != configuration for part in parts.values()):
                    raise RuntimeError('Candidate configuration changed across folds')
                rows[cid] = {'configuration': configuration,
                             'physical_risk': pooled_risk([part['physical_risk'] for part in parts.values()]),
                             'folds': {s: part['physical_risk'] for s, part in parts.items()}}
            selections = {}
            for policy in c.POLICIES:
                pool = []
                for cid, row in rows.items():
                    cfg = row['configuration']
                    eligible = ((policy in ('raw', 'nystrom', 'kahm') and cfg.get('family') == policy)
                                or (policy == 'all_map' and cfg['recipe'] == 'affine')
                                or (policy == 'ordinary' and cfg['recipe'] == 'baseline')
                                or (policy == 'ridge' and cfg.get('baseline_kind') == 'ridge'))
                    if eligible:
                        pool.append(cid)
                winner = min(pool, key=lambda cid: (F(rows[cid]['physical_risk']['mse']), cid))
                selections[policy] = {'candidate_id': winner, 'pool_size': len(pool),
                                      'physical_risk': rows[winner]['physical_risk']}
                all_selections[f'{seed}::{policy}'] = {**selections[policy], 'seed': seed, 'policy': policy}
            path = OUT / task / f'selection_seed_{seed}43.json'
            c.write_json(path, {'schema': 'step43-development-selection-v1', 'task': task, 'seed': seed,
                                'development_n': sum(f['n'] for f in folds.values()),
                                'fold_scores': {s: c.file_ref(path) for s, path in fold_paths.items()},
                                'candidate_rows': rows, 'selections': selections})
            selection_refs.append(c.file_ref(path))
            progress('SELECTION_COMPLETE', task=task, seed=seed,
                     selected={k: v['candidate_id'] for k, v in selections.items()})
        roster['tasks'][task] = {'selections': all_selections, 'selection_files': selection_refs}
    c.write_json(OUT / 'selected_roster43.json', roster)


def seal_selection():
    roster = c.read_json(OUT / 'selected_roster43.json')
    files = [OUT / 'selected_roster43.json']
    for task in c.TASKS:
        files.extend(ROOT / row['path'] for row in roster['tasks'][task]['selection_files'])
        files.extend(path for path in (OUT / task / 'final').rglob('*') if path.is_file())
    c.seal_files(OUT / 'selection_seal43.json', files,
                 schema='step43-selection-seal-v1', protocol_seal_sha256=c.protocol_identity())
    c.event('selection_and_final_models_sealed', selection_seal=c.file_ref(OUT / 'selection_seal43.json'))


def _evaluation_id(context, cid):
    if cid in context['prepared']:
        return 'affine::' + context['prepared'][cid].deployment_id
    row = context['baseline_rows'][cid]
    return 'baseline::' + c.digest({'candidate_id': cid, 'pickle_sha256': row['pickle_sha256']})


def assess(task):
    c.require_gate('audit/preassessment43.json', protocol_seal_sha256=c.protocol_identity(),
                   selection_seal_sha256=c.sha_file(OUT / 'selection_seal43.json'))
    selected = c.read_json(OUT / 'selected_roster43.json')['tasks'][task]['selections']
    contexts = {seed: load_context(task, 'final', seed) for seed in c.SEEDS}
    queries = d.assessment_queries(task)
    rows, targets = [], []
    all_predictions = {alias: [] for alias in selected}
    all_references = {alias: [] for alias in selected}
    acceptance = {alias: {'accepted': 0, 'packets': 0} for alias in selected}
    summary = {case: {'packets': 0, 'accepted': 0, 'rejected_over_budget': 0,
                      'binding_only_accepted': 0} for case in ('binary64', 'physical_output_float32')}
    summary['quantization_changed_reported_value'] = 0
    for index, query in enumerate(queries):
        history = d.load_assessment_history(task, index)
        qid = history['query_id']
        X = history['X']
        destination = OUT / 'assessment' / task / 'queries' / f'{index:04d}'
        raw_ref = save_array(destination / 'raw_X.npy', X)
        predictions, evaluations, feature_cache = {}, {}, {}
        for alias, selection in selected.items():
            seed = selection['seed']
            cid = selection['candidate_id']
            context = contexts[seed]
            evaluation_id = _evaluation_id(context, cid)
            if evaluation_id not in evaluations:
                if cid in context['prepared']:
                    cfg = next(row for row in context['fits']['affine_rows'] if row['candidate_id'] == cid)
                    name = cfg['map']
                    detail = context['fits']['maps'][name]
                    map_hash = detail['map']['sha256']
                    if map_hash not in feature_cache:
                        Z = m.transform_one(context['maps'][name], X)
                        zref = save_array(destination / 'features' / (map_hash + '.npy'), Z)
                        feature_cache[map_hash] = (Z[0], zref)
                    z, zref = feature_cache[map_hash]
                    pp = context['prepared'][cid]
                    value = m.predict_one(pp, z)
                    assay = m.prediction_assay(pp, z, value, qid)
                    evaluation = {'recipe': 'affine', 'candidate_id': cid, 'aliases': [],
                                  'reported_physical_hex': value.hex(),
                                  'fit_bank': context['fits_ref'],
                                  'deployment': context['heads'][cid]['deployment'],
                                  'features': zref, 'map': detail['map'], 'map_name': name,
                                  'assay': assay}
                    for case in ('binary64', 'physical_output_float32'):
                        record = assay['cases'][case]
                        summary[case]['packets'] += 1
                        summary[case]['accepted'] += int(record['accepted'])
                        summary[case]['rejected_over_budget'] += int(not record['accepted'])
                        summary[case]['binding_only_accepted'] += int(record['binding_only']['accepted'])
                    summary['quantization_changed_reported_value'] += int(assay['quantization_changed_reported_value'])
                else:
                    value = float(m.predict_baseline(context['baselines'][cid], X)[0])
                    evaluation = {'recipe': 'baseline', 'candidate_id': cid, 'aliases': [],
                                  'reported_physical_hex': value.hex(), 'fit_bank': context['fits_ref'],
                                  'payload': context['baseline_rows'][cid]['payload'],
                                  'pickle_sha256': context['baseline_rows'][cid]['pickle_sha256']}
                evaluations[evaluation_id] = evaluation
            evaluation = evaluations[evaluation_id]
            evaluation['aliases'].append(alias)
            prediction = {'candidate_id': cid, 'evaluation_id': evaluation_id,
                          'reported_physical_hex': evaluation['reported_physical_hex']}
            if evaluation['recipe'] == 'affine':
                honest = evaluation['assay']['cases']['binary64']
                prediction['exact_physical'] = honest['exact_physical']
                all_references[alias].append(honest['exact_physical'])
                acceptance[alias]['packets'] += 1
                acceptance[alias]['accepted'] += int(honest['accepted'])
            predictions[alias] = prediction
            all_predictions[alias].append(prediction['reported_physical_hex'])
        history_path = history.get('path') or history.get('history_receipt_path')
        if not history_path:
            raise RuntimeError('Assessment history omitted its retained receipt path')
        pred_path = destination / 'predictions43.json'
        c.write_json(pred_path, {'schema': 'step43-query-predictions-v1',
                                'task': task, 'query_id': qid, 'query_index': index,
                                'protocol_seal_sha256': c.protocol_identity(),
                                'selection_seal_sha256': c.sha_file(OUT / 'selection_seal43.json'),
                                'history_receipt_sha256': history['history_receipt_sha256'],
                                'history': c.file_ref(ROOT / history_path), 'raw_features': raw_ref,
                                'predictions': predictions, 'evaluations': evaluations})
        c.event('assessment_query_predictions_committed', task=task, query_index=index,
                query_id=qid, prediction=c.file_ref(pred_path))
        target = d.reveal_assessment_target(task, index, pred_path.relative_to(ROOT).as_posix())
        targets.append(F(target['raw_target']))
        rows.append({'query_index': index, 'query_id': qid,
                     'month': query['target_timestamp'][:7],
                     'prediction': c.file_ref(pred_path), 'target': c.file_ref(ROOT / target['path'])})
        if (index+1) % 10 == 0 or index+1 == len(queries):
            progress('ASSESSMENT_QUERY', task=task, completed=index+1, total=len(queries))
    policy_results, monthly = {}, {}
    for alias, selection in selected.items():
        refs = all_references[alias]
        policy_results[alias] = {'candidate_id': selection['candidate_id'],
                                 'physical_risk': reported_risk(all_predictions[alias], targets),
                                 'exact_reference_risk': risk(refs, targets) if refs else None,
                                 'accepted_packets': acceptance[alias]['accepted'],
                                 'packet_count': acceptance[alias]['packets']}
    for month in sorted({row['month'] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row['month'] == month]
        monthly[month] = {alias: reported_risk([all_predictions[alias][i] for i in indices],
                                              [targets[i] for i in indices]) for alias in selected}
    raw = policy_results['3901::raw']
    primary = {policy: compare(raw['physical_risk'], policy_results['3901::' + policy]['physical_risk'])
               for policy in ('ordinary', 'ridge')}
    record = {'schema': 'step43-task-assessment-v1', 'task': task, 'n': len(rows),
              'protocol_seal_sha256': c.protocol_identity(),
              'selection_seal_sha256': c.sha_file(OUT / 'selection_seal43.json'),
              'queries': rows, 'policy_results': policy_results, 'monthly_results': monthly,
              'primary_comparisons': primary,
              'primary_certificate_coverage': {'packets': raw['packet_count'],
                                               'accepted': raw['accepted_packets'],
                                               'all_accepted': raw['packet_count'] == raw['accepted_packets'] == len(rows)},
              'certificate_summary': summary}
    c.write_json(OUT / 'assessment' / task / 'assessment43.json', record)


def misbinding(task):
    assessment = c.read_json(OUT / 'assessment' / task / 'assessment43.json')
    first = c.read_json(ref_read(assessment['queries'][0]['prediction']))
    second = c.read_json(ref_read(assessment['queries'][1]['prediction']))
    records = {}
    for eid, evaluation in first['evaluations'].items():
        if evaluation['recipe'] != 'affine':
            continue
        if eid not in second['evaluations']:
            raise RuntimeError('Selected evaluation identities changed between queries')
        z = np.load(ref_read(evaluation['features']), allow_pickle=False)[0]
        stale = np.load(ref_read(second['evaluations'][eid]['features']), allow_pickle=False)[0]
        pp = m.prepare_deployment(evaluation['deployment'],
                                  expected_deployment_id=evaluation['deployment']['deployment_id'],
                                  expected_map_sha256=evaluation['map']['sha256'])
        records[eid] = m.misbinding_assay(
            pp, z, float.fromhex(evaluation['reported_physical_hex']), first['query_id'],
            stale_features=stale, honest_record=evaluation['assay']['cases']['binary64'])
    c.write_json(OUT / 'assessment' / task / 'misbinding43.json',
                 {'schema': 'step43-selected-head-misbinding-v1', 'task': task,
                  'protocol_seal_sha256': c.protocol_identity(),
                  'performed_after_assessment': c.file_ref(OUT / 'assessment' / task / 'assessment43.json'),
                  'query_index': 0, 'stale_query_index': 1,
                  'query_prediction': assessment['queries'][0]['prediction'],
                  'stale_query_prediction': assessment['queries'][1]['prediction'], 'records': records})


def aggregate():
    tasks, cost_refs, assessment_refs, misbinding_refs = {}, {}, {}, {}
    for task in c.TASKS:
        path = OUT / 'assessment' / task / 'assessment43.json'
        tasks[task] = c.read_json(path)
        assessment_refs[task] = c.file_ref(path)
        cost_refs[task] = c.file_ref(OUT / 'benchmarks' / task / 'cost43.json')
        misbinding_refs[task] = c.file_ref(OUT / 'assessment' / task / 'misbinding43.json')
    predictive = all(comparison['within_five_percent_mse']
                     for task in tasks.values() for comparison in task['primary_comparisons'].values())
    coverage = all(task['primary_certificate_coverage']['all_accepted'] for task in tasks.values())
    records = {'schema': 'step43-selected-policy-transfer-result-v1',
               'protocol_seal_sha256': c.protocol_identity(),
               'study_status': 'FRESH_TASK_PERIODS_IN_KNOWN_DOMAINS',
               'freshness_scope': 'Not numerically examined in the retained workflow before final protocol freeze; no absolute historical never-seen claim',
               'primary_policy': 'raw-family P1R1C1', 'primary_seed': 3901,
               'primary_predictive_criterion_met': predictive,
               'primary_certificate_coverage_met': coverage,
               'primary_criterion_met_pending_independent_audit': predictive and coverage,
               'primary_status_if_independent_audit_passes': ('ESTABLISHED_ON_THREE_FIXED_PERIODS'
                                                             if predictive and coverage else 'NOT_ESTABLISHED'),
               'population_noninferiority_or_significance_claim': False,
               'unseen_domain_claim': False,
               'assessment_query_count': sum(task['n'] for task in tasks.values()),
               'tasks': tasks, 'assessment_files': assessment_refs,
               'cost_files': cost_refs, 'misbinding_files': misbinding_refs,
               'selected_roster': c.file_ref(OUT / 'selected_roster43.json'),
               'selection_seal': c.file_ref(OUT / 'selection_seal43.json'),
               'preassessment_audit': c.file_ref(ROOT / 'audit/preassessment43.json'),
               'independent_final_audit_pending': True,
               'prior_step41_primary_status_unchanged': 'NOT_ESTABLISHED',
               'prior_step42_status_unchanged': 'DIAGNOSTIC_ALREADY_EXPOSED_PERIODS'}
    c.write_json(OUT / 'result43.json', records)
    files = [OUT / 'result43.json', *(ROOT / ref['path'] for ref in assessment_refs.values()),
             *(ROOT / ref['path'] for ref in cost_refs.values()),
             *(ROOT / ref['path'] for ref in misbinding_refs.values())]
    c.seal_files(OUT / 'aggregate_seal43.json', files,
                 schema='step43-aggregate-seal-v1', protocol_seal_sha256=c.protocol_identity())
    progress('AGGREGATE_COMPLETE', primary_predictive_criterion_met=predictive,
             primary_certificate_coverage_met=coverage,
             primary_status_if_audit_passes=records['primary_status_if_independent_audit_passes'])


def _timeout(*args):
    raise TimeoutError('Frozen phase budget exhausted; retain failure and stop')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['stage', 'select', 'seal-selection', 'assess', 'misbinding', 'aggregate'])
    parser.add_argument('--task', choices=c.TASKS)
    parser.add_argument('--stage', choices=c.STAGES)
    args = parser.parse_args()
    protocol_hash = c.protocol_identity(deep=True)
    c.require_gate('audit/preflight43.json', protocol_seal_sha256=protocol_hash)
    protocol = c.read_json(ROOT / 'protocol43.json')
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, protocol['execution']['phase_budget_seconds'][args.phase])
    started = time.perf_counter()
    c.event('producer_phase_started', phase=args.phase, task=args.task, stage=args.stage,
            protocol_seal_sha256=protocol_hash)
    try:
        if args.phase == 'stage':
            stage(args.task, args.stage)
        elif args.phase in ('assess', 'misbinding'):
            globals()[args.phase](args.task)
        elif args.phase == 'seal-selection':
            seal_selection()
        else:
            globals()[args.phase]()
        signal.setitimer(signal.ITIMER_REAL, 0)
        c.event('producer_phase_completed', phase=args.phase, task=args.task, stage=args.stage,
                wall_seconds=time.perf_counter()-started)
    except Exception as failure:
        signal.setitimer(signal.ITIMER_REAL, 0)
        record = {'phase': args.phase, 'task': args.task, 'stage': args.stage,
                  'type': type(failure).__name__, 'reason': str(failure),
                  'traceback': traceback.format_exc(), 'utc': c.utc_now()}
        path = OUT / 'failures' / f'{args.phase}_{args.task}_{args.stage}43.json'
        if not path.exists():
            c.write_json(path, record)
        progress('STOP_RETAINED_FAILURE', **record)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
