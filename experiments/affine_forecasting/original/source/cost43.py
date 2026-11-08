"""Post-assessment cost assay for the actual primary-seed selected identities.

No fitting, reselection or new measurement decoding occurs.  Complete retained
assessment records and an independently passed preassessment gate are required.
Scientific latency is measured once with the frozen warmup/repetition counts.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path
import time
import traceback

import model43 as m
import common43 as c
import numpy as np


CERTIFIED_POLICIES = ('raw', 'nystrom', 'kahm', 'all_map')
PRIMARY_POLICIES = CERTIFIED_POLICIES + ('ordinary', 'ridge')


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_query_indices(n, limit=32):
    require(type(n) is int and n > 0 and type(limit) is int and limit > 0,
            'Invalid query-sampling counts')
    count = min(n, limit)
    return [((2*j+1)*n)//(2*count) for j in range(count)]


def linear_quantile(values, probability):
    require(bool(values) and 0 <= probability <= 1, 'Invalid cost quantile')
    ordered = sorted(values)
    index = (len(ordered)-1)*probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high]-ordered[low])*(index-low)


def timing_summary(medians):
    require(bool(medians) and all(math.isfinite(x) and x >= 0 for x in medians),
            'Invalid retained timing medians')
    return {'queries': len(medians),
            'p50_ms': float(linear_quantile(medians, .5))/1e6,
            'p95_ms': float(linear_quantile(medians, .95))/1e6,
            'min_ms': float(min(medians))/1e6, 'max_ms': float(max(medians))/1e6}


def size_summary(values):
    require(bool(values) and all(type(x) is int and x >= 0 for x in values),
            'Invalid retained byte counts')
    return {'min': min(values), 'median': float(linear_quantile(values,.5)), 'max': max(values)}


def benchmark_task(task, *, root=c.ROOT, context_loader=None):
    require(task in c.TASKS, 'Task is outside frozen study roster')
    root = Path(root)
    target = root/'output/benchmarks'/task/'cost43.json'
    require(not target.exists(), 'Retain the existing cost assay; no repeated benchmark run')
    protocol_sha = c.protocol_identity(root=root)
    preassessment = c.require_gate('audit/preassessment43.json', root=root,
                                  protocol_seal_sha256=protocol_sha)
    selection_seal_sha = c.sha_file(root/'output/selection_seal43.json')
    gate_binding = preassessment.get('selection_seal_sha256')
    if gate_binding is None:
        gate_binding = preassessment.get('source_bindings',{}).get('output/selection_seal43.json')
    require(gate_binding == selection_seal_sha,
            'Preassessment PASS does not bind the current selected identities')
    bindings = {}
    def bind(relative):
        path = root/relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'Unsafe relative cost evidence path')
        bindings[relative] = c.sha_file(path)
        return path
    def load_json(relative):
        return c.read_json(bind(relative))
    def ref_path(reference):
        require(isinstance(reference,dict) and {'path','sha256'} <= set(reference),
                'Malformed cost evidence reference')
        path = bind(reference['path'])
        require(bindings[reference['path']] == reference['sha256'],
                'Cost evidence hash differs: '+reference['path'])
        if 'bytes' in reference:
            require(path.stat().st_size == reference['bytes'], 'Cost evidence size differs')
        return path

    for relative in ('protocol_seal43.json','protocol43.json','design43.json',
                     'audit/preassessment43.json','output/selection_seal43.json',
                     'source/cost43.py','source/model43.py','source/certificate41.py',
                     'source/common43.py','source/run43.py','source/interface43.json'):
        bind(relative)
    design = c.read_json(root/'design43.json')
    settings = design['cost_assay']
    require(settings['seed'] == 3901 and settings['warmups'] == 3
            and settings['measured_repeats'] == 10 and settings['static_context_repeats'] == 100,
            'Cost repetition/seed design differs')
    assessments = {}
    for name in c.TASKS:
        assessment = load_json(f'output/assessment/{name}/assessment43.json')
        require(assessment['task'] == name and assessment['n'] == len(assessment['queries'])
                and assessment['n'] > 0 and assessment['protocol_seal_sha256'] == protocol_sha
                and assessment['selection_seal_sha256'] == selection_seal_sha,
                'A complete matching task assessment is required before timing')
        assessments[name] = assessment
    assessment = assessments[task]
    chosen_indices = selected_query_indices(assessment['n'])
    roster = load_json('output/selected_roster43.json')
    require(roster['protocol_seal_sha256'] == protocol_sha,
            'Selected policy roster belongs to another protocol')
    q0 = c.read_json(ref_path(assessment['queries'][0]['prediction']))
    primary_keys = ['3901::'+policy for policy in PRIMARY_POLICIES]
    groups = {}
    for key in primary_keys:
        selection = roster['tasks'][task]['selections'][key]
        prediction = q0['predictions'][key]
        require(prediction['candidate_id'] == selection['candidate_id'],
                'Cost selection differs from frozen development choice')
        identity = prediction['evaluation_id']
        evaluation = q0['evaluations'][identity]
        require(evaluation['candidate_id'] == prediction['candidate_id'],
                'Selected evaluation candidate differs')
        group = groups.setdefault(identity, {'candidate_id':prediction['candidate_id'],
                                            'recipe':evaluation['recipe'], 'aliases':[]})
        require(group['candidate_id'] == prediction['candidate_id']
                and group['recipe'] == evaluation['recipe'], 'Conflicting deduplicated cost identity')
        group['aliases'].append(key)

    if context_loader is None:
        from run43 import load_context
        context_loader = load_context
    context = context_loader(task,'final',3901)
    ref_path(context['fits_ref'])
    fit_bank = context['fits']
    require(fit_bank['task'] == task and fit_bank['stage'] == 'final' and fit_bank['seed'] == 3901,
            'Benchmark context is not the selected primary-seed final bank')
    ref_path(fit_bank['heads'])
    ref_path(fit_bank['representation_metadata'])
    affine_configs = {row['candidate_id']:row for row in fit_bank['affine_rows']}
    started_utc = c.utc_now()
    start = time.perf_counter()
    c.event('cost_started', root=root, task=task, seed=3901,
            selected_evaluation_ids=list(groups), query_indices=chosen_indices,
            protocol_seal_sha256=protocol_sha, selection_seal_sha256=selection_seal_sha)
    records = {}
    for identity, group in groups.items():
        candidate_id, recipe = group['candidate_id'], group['recipe']
        rows = []
        if recipe == 'affine':
            cfg = affine_configs[candidate_id]
            mapping = context['maps'][cfg['map']]
            prepared = context['prepared'][candidate_id]
            deployment = context['heads'][candidate_id]['deployment']
            require('affine::'+prepared.deployment_id == identity
                    and deployment['deployment_id'] == prepared.deployment_id,
                    'Selected affine identity differs from the primary fit bank')
            map_record = fit_bank['maps'][cfg['map']]
            ref_path(map_record['map']);ref_path(map_record['training_features'])
            static = m.benchmark_static_context(deployment,warmups=3,repeats=100)
            for ordinal,index in enumerate(chosen_indices):
                query_record = assessment['queries'][index]
                query = c.read_json(ref_path(query_record['prediction']))
                require(query['query_index'] == index and query['query_id'] == query_record['query_id'],
                        'Benchmark query ordering differs')
                evaluation = query['evaluations'][identity]
                require(evaluation['deployment'] == deployment and evaluation['map'] == map_record['map'],
                        'Query affine deployment/map differs from selected bank')
                for key in group['aliases']:
                    require(query['predictions'][key]['evaluation_id'] == identity
                            and query['predictions'][key]['reported_physical_hex'] == evaluation['reported_physical_hex'],
                            'Benchmark alias prediction differs')
                X = np.load(ref_path(query['raw_features']),allow_pickle=False)
                Z = np.load(ref_path(evaluation['features']),allow_pickle=False)
                require(X.shape == (1,24) and Z.ndim == 2 and Z.shape[0] == 1,
                        'Retained query raw/feature array shape differs')
                honest = evaluation['assay']['cases']['binary64']
                reported = float.fromhex(evaluation['reported_physical_hex'])
                require(honest['reported_physical_hex'] == reported.hex(), 'Honest assay value differs')
                row = m.benchmark_query(prepared,mapping,X,Z[0],query['query_id'],reported,
                                        query_index=index,sample_ordinal=ordinal,warmups=3,repeats=10)
                row.update({'prediction':query_record['prediction'],
                            'raw_features':query['raw_features'],'features':evaluation['features'],
                            'reported_physical_hex':reported.hex(),
                            'verification_status':honest['verification']['status'],
                            'expected_acceptance':honest['expected_acceptance']})
                require(F(row['absolute_physical_error']) == F(honest['absolute_physical_error'])
                        and (F(row['absolute_physical_error']) <= F(row['physical_error_budget'])) == honest['expected_acceptance'],
                        'Cost record and assessed exact numerical predicate differ')
                rows.append(row)
            summaries = {name:timing_summary([row['median_nanoseconds'][name] for row in rows])
                         for name in rows[0]['nanoseconds']}
            sizes = {name:size_summary([row[name] for row in rows])
                     for name in ('packet_json_bytes','feature_payload_bytes',
                                  'static_deployment_json_bytes','map_bytes')}
            record = {**group,'evaluation_id':identity,'map':map_record['map'],
                      'deployment_id':deployment['deployment_id'],'static_context':static,
                      'static_context_median_ms':float(linear_quantile(static['nanoseconds'],.5))/1e6,
                      'queries':rows,'component_summary':summaries,'byte_summary':sizes}
        else:
            require(recipe == 'baseline', 'Unknown selected model recipe')
            payload = context['baselines'][candidate_id]
            baseline_row = context['baseline_rows'][candidate_id]
            expected_identity = 'baseline::'+c.digest({'candidate_id':candidate_id,
                                                      'pickle_sha256':baseline_row['pickle_sha256']})
            require(expected_identity == identity,
                    'Selected baseline identity differs from primary fit bank')
            ref_path(baseline_row['payload'])
            for ordinal,index in enumerate(chosen_indices):
                query_record = assessment['queries'][index]
                query = c.read_json(ref_path(query_record['prediction']))
                require(query['query_index'] == index and query['query_id'] == query_record['query_id'],
                        'Benchmark baseline query ordering differs')
                evaluation = query['evaluations'][identity]
                for key in group['aliases']:
                    require(query['predictions'][key]['evaluation_id'] == identity
                            and query['predictions'][key]['reported_physical_hex'] == evaluation['reported_physical_hex'],
                            'Baseline benchmark alias prediction differs')
                X = np.load(ref_path(query['raw_features']),allow_pickle=False)
                result = m.benchmark_baseline(payload,X,warmups=3,repeats=10)
                require(result['reported_physical_hex'] == evaluation['reported_physical_hex'],
                        'Selected conventional prediction replay changed')
                rows.append({'query_id':query['query_id'],'query_index':index,'sample_ordinal':ordinal,
                             'prediction':query_record['prediction'],'raw_features':query['raw_features'],
                             **result})
            record = {**group,'evaluation_id':identity,'payload':baseline_row['payload'],
                      'pickle_sha256':baseline_row['pickle_sha256'],'pickle_bytes':baseline_row['pickle_bytes'],
                      'queries':rows,'component_summary':{
                          'selected_conventional_prediction':timing_summary([row['median_nanoseconds'] for row in rows])}}
        records[identity] = record
    out = {'schema':'step43-selected-model-cost-assay-v1','status':'COMPLETE',
           'task':task,'seed':3901,'protocol_seal_sha256':protocol_sha,
           'selection_seal_sha256':selection_seal_sha,
           'preassessment_gate_sha256':bindings['audit/preassessment43.json'],
           'configuration':settings,'assessment_n':assessment['n'],
           'query_indices':chosen_indices,'sample_query_count':len(chosen_indices),
           'primary_policy_aliases':primary_keys,'deduplicated_models':records,
           'source_bindings':bindings,'started_utc':started_utc,'completed_utc':c.utc_now(),
           'wall_seconds':time.perf_counter()-start,
           'new_fits':0,'new_source_measurement_decodes':0,
           'population_runtime_claim':False,'speedup_threshold':None,
           'summary_convention':'Per-query median of10 repeats; linear p50/p95 across selected queries; direct full-inference totals are measured, not sums of component timings',
           'scope':'Actual primary-seed selected identities on retained query arrays after all assessments; disk/network/startup excluded from primitive samples'}
    c.write_json(target,out)
    c.event('cost_completed',root=root,task=task,cost=c.file_ref(target,root=root),
            selected_models=len(records),sample_queries=len(chosen_indices))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True,choices=c.TASKS)
    args=parser.parse_args()
    result=benchmark_task(args.task)
    print(json.dumps({'status':'COMPLETE','task':args.task,
                      'models':len(result['deduplicated_models']),
                      'sample_queries':result['sample_query_count'],
                      'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':
    try:
        main()
    except Exception as failure:
        print(json.dumps({'status':'FAILED_RETAINED','type':type(failure).__name__,
                          'reason':str(failure),'traceback':traceback.format_exc()}))
        raise
