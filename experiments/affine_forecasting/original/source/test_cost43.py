"""Manufactured-only integration check of the retained selected-model cost API.

The fixture is deliberately not a scientific study: its protocol/gate documents
are labelled artificial, and the genuine gate functions reject them.  Only
within this test are those two gate calls redirected.  Real run43.load_context,
map/head/model replay, packet generation/checking and timing summaries execute.
No natural source observations, assessment outcomes, or previous fits are read.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[_name] = '1'

from collections import Counter
from fractions import Fraction as F
import gzip
import hashlib
from pathlib import Path
import pickle
import time
from unittest.mock import patch

import numpy as np

import common43 as c
import cost43 as cost
import model43 as m
import run43 as runner


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'analysis/cost_fixture43'


def run():
    start = time.perf_counter()
    checks = Counter()
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks[name] += 1
    def rejects(function, name):
        try:
            function()
        except ValueError:
            checks[name] += 1
        else:
            raise AssertionError('Expected refusal: '+name)
    def write(relative, record):
        c.write_json(FIXTURE/relative, record)
        return c.file_ref(FIXTURE/relative, root=FIXTURE)
    def bytefile(relative, contents):
        path = FIXTURE/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(contents)
        return c.file_ref(path,root=FIXTURE)
    def arrayfile(relative, value):
        path = FIXTURE/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.save(stream,np.asarray(value,dtype=np.float64),allow_pickle=False)
        return c.file_ref(path,root=FIXTURE)

    if FIXTURE.exists():
        raise FileExistsError('Retain the existing artificial fixture: '+str(FIXTURE))
    FIXTURE.mkdir(parents=True)
    artificial = {'status':'ARTIFICIAL_STUB_NOT_A_SCIENTIFIC_GATE',
                  'scope':'Manufactured cost adapter fixture only; no model selection or study validity claim'}
    write('fixture_notice43.json', artificial)
    source_names = ('cost43.py','model43.py','certificate41.py','common43.py',
                    'run43.py','interface43.json')
    for name in source_names:
        bytefile('source/'+name,(ROOT/'source'/name).read_bytes())
    bytefile('design43.json',(ROOT/'design43.json').read_bytes())
    write('protocol43.json',artificial)
    write('protocol_seal43.json',artificial)
    protocol_sha = c.sha_file(FIXTURE/'protocol_seal43.json')
    write('output/selection_seal43.json',artificial)
    selection_sha = c.sha_file(FIXTURE/'output/selection_seal43.json')
    write('audit/preassessment43.json',{**artificial,'selection_seal_sha256':selection_sha})
    rejects(lambda:c.protocol_identity(root=FIXTURE),'genuine_protocol_rejects_artificial_stub')
    rejects(lambda:c.require_gate('audit/preassessment43.json',root=FIXTURE),
            'genuine_gate_rejects_artificial_stub')

    for n in (1,2,30,31,32,33,91,365):
        indices=cost.selected_query_indices(n)
        count=min(n,32)
        check(indices==[((2*j+1)*n)//(2*count) for j in range(count)]
              and len(set(indices))==count and indices==sorted(indices)
              and 0<=indices[0]<=indices[-1]<n,'fixed_equal_stratum_query_indices')
    for invalid in (0,-1,1.0,True):
        rejects(lambda:cost.selected_query_indices(invalid),'invalid_query_count_refused')
    check(cost.timing_summary([0.,10.,20.])=={
        'queries':3,'p50_ms':1e-5,'p95_ms':1.9e-5,'min_ms':0.,'max_ms':2e-5},
        'linear_quantile_summary')
    check(cost.size_summary([3,1,8,4])=={'min':1,'median':3.5,'max':8},'byte_summary')

    # One actual raw-map/affine fit and one actual conventional ridge fit are
    # enough to exercise both serialization branches.  Four certified labels
    # deliberately alias this one component; this is not a selection test.
    task=c.TASKS[0]
    rows=np.arange(32,dtype=np.float64)[:,None]
    axes=np.arange(24,dtype=np.float64)[None,:]
    X=np.sin((rows+1)*(axes+1)/31.)+rows/64.+axes/128.
    y=[F(i,31) for i in range(32)]
    normal={'target_min':'-3','span':'7'}
    scaler=m.rep._geometry().InputScaler.fit(X,'standard')
    mapping,metadata=m.rep._freeze_metric('raw_standard','raw','standard',24,scaler,X)
    cfg=next(row for row in m.affine_configs()
             if row['map']=='raw_standard' and row['ridge_lambda']=='1')
    baseline_cfg=next(row for row in m.baseline_configs(3901)
                      if row['kind']=='ridge' and row['ridge_lambda']=='1')
    cid,bid=cfg['candidate_id'],baseline_cfg['candidate_id']
    prefix=f'output/{task}/final/seed_3901'
    wire=m.serialize_map(mapping)
    map_ref=bytefile(prefix+'/maps/raw_standard.pkl',wire)
    Z=m.transform_one(mapping,X)
    zref=arrayfile(prefix+'/features/train/raw_standard.npy',Z)
    fitted=m.fit_affine(cfg,Z,y,normal,feature_id='artificial:cost:raw_standard',
                        map_sha256=map_ref['sha256'],map_bytes=len(wire),
                        training_binding={'origin':'manufactured_cost_fixture'})
    head={key:fitted[key] for key in ('fit','head','deployment')}
    heads_ref=write(prefix+'/heads43.json',{cid:head})
    metadata_ref=write(prefix+'/representation_metadata43.json',{
        **artificial,'manufactured_raw_component':metadata})
    payload=m.fit_baseline(baseline_cfg,X,y,normal)
    raw=pickle.dumps(payload,protocol=5)
    payload_ref=bytefile(prefix+'/baselines/ridge_1.pkl.gz',gzip.compress(raw,mtime=0))
    baseline_row={**baseline_cfg,'recipe':'baseline','baseline_kind':'ridge',
                  'payload':payload_ref,'pickle_sha256':hashlib.sha256(raw).hexdigest(),
                  'pickle_bytes':len(raw)}
    bank={'schema':'step43-stage-seed-fit-v1','task':task,'stage':'final','seed':3901,
          'protocol_seal_sha256':protocol_sha,'normalization':normal,
          'heads':heads_ref,'representation_metadata':metadata_ref,
          'maps':{'raw_standard':{'name':'raw_standard','family':'raw','scaling':'standard',
              'dimension':24,'feature_id':'artificial:cost:raw_standard',
              'map':map_ref,'training_features':zref}},
          'affine_rows':[cfg],'baseline_rows':[baseline_row]}
    bank_ref=write(prefix+'/fits43.json',bank)

    primary=['3901::'+name for name in cost.PRIMARY_POLICIES]
    certified_keys=primary[:4]
    baseline_keys=primary[4:]
    affine_id='affine::'+fitted['prepared'].deployment_id
    baseline_id='baseline::'+c.digest({'candidate_id':bid,
                                      'pickle_sha256':baseline_row['pickle_sha256']})
    selections={key:{'candidate_id':cid if key in certified_keys else bid} for key in primary}
    write('output/selected_roster43.json',{'schema':'step43-selected-roster-v1',
        'protocol_seal_sha256':protocol_sha,'tasks':{task:{'selections':selections}}})
    queries=[]
    for index in range(2):
        qid='artificial:cost:query:'+str(index)
        qbase=f'output/assessment/{task}/queries/{index:04d}'
        raw_X=X[index:index+1]+(index+1)/1000.
        raw_ref=arrayfile(qbase+'/raw_X.npy',raw_X)
        z=m.transform_one(mapping,raw_X)[0]
        z_ref=arrayfile(qbase+'/features/raw_standard.npy',z[None,:])
        prediction=m.predict_one(fitted['prepared'],z)
        ordinary=float(m.predict_baseline(payload,raw_X)[0])
        evaluations={affine_id:{'recipe':'affine','candidate_id':cid,
            'aliases':certified_keys+['4301::raw'],
            'reported_physical_hex':prediction.hex(),'fit_bank':bank_ref,
            'deployment':fitted['deployment'],'features':z_ref,'map':map_ref,
            'map_name':'raw_standard','assay':m.prediction_assay(fitted['prepared'],z,prediction,qid)},
            baseline_id:{'recipe':'baseline','candidate_id':bid,
                'aliases':baseline_keys+['4302::ridge'],
                'reported_physical_hex':ordinary.hex(),'fit_bank':bank_ref,
                'payload':payload_ref,'pickle_sha256':baseline_row['pickle_sha256']}}
        predictions={key:{'candidate_id':cid if key in certified_keys else bid,
            'evaluation_id':affine_id if key in certified_keys else baseline_id,
            'reported_physical_hex':prediction.hex() if key in certified_keys else ordinary.hex()}
            for key in primary}
        qref=write(qbase+'/predictions43.json',{'schema':'step43-query-predictions-v1',
            'task':task,'query_index':index,'query_id':qid,
            'protocol_seal_sha256':protocol_sha,'selection_seal_sha256':selection_sha,
            'raw_features':raw_ref,'predictions':predictions,'evaluations':evaluations})
        queries.append({'query_index':index,'query_id':qid,'prediction':qref})
    for name in c.TASKS:
        write(f'output/assessment/{name}/assessment43.json',{
            'schema':'step43-task-assessment-v1','task':name,'n':len(queries),
            'protocol_seal_sha256':protocol_sha,'selection_seal_sha256':selection_sha,
            'queries':queries,'fixture_scope':artificial['scope']})

    actual_file_ref=c.file_ref
    def fixture_file_ref(path,*,root=FIXTURE):
        return actual_file_ref(path,root=root)
    def fixture_protocol(**kwargs):
        check(not kwargs.get('root') or kwargs['root']==FIXTURE,'artificial_protocol_redirect_scope')
        return protocol_sha
    def fixture_gate(relative,**kwargs):
        check(relative=='audit/preassessment43.json' and kwargs['root']==FIXTURE
              and kwargs['protocol_seal_sha256']==protocol_sha,'artificial_gate_redirect_scope')
        return c.read_json(FIXTURE/relative)
    with patch.object(runner,'ROOT',FIXTURE),patch.object(runner,'OUT',FIXTURE/'output'),\
            patch.object(c,'file_ref',fixture_file_ref),\
            patch.object(c,'protocol_identity',fixture_protocol),\
            patch.object(c,'require_gate',fixture_gate):
        loaded=runner.load_context(task,'final',3901)
        check(loaded['fits']==bank and loaded['fits_ref']==bank_ref,
              'real_runner_context_reads_component_bank')
        check(runner._evaluation_id(loaded,cid)==affine_id
              and runner._evaluation_id(loaded,bid)==baseline_id,'actual_runner_identity_convention')
        result=cost.benchmark_task(task,root=FIXTURE,context_loader=runner.load_context)
        rejects(lambda:cost.benchmark_task(task,root=FIXTURE,context_loader=runner.load_context),
                'existing_cost_record_not_overwritten')
    check(c.protocol_identity is not fixture_protocol and c.require_gate is not fixture_gate,
          'artificial_gate_redirects_restored')
    check(result['status']=='COMPLETE' and result['new_fits']==0
          and result['new_source_measurement_decodes']==0,'adapter_completed_without_new_fits_or_decodes')
    check(result['query_indices']==[0,1] and result['sample_query_count']==2,'two_retained_queries_sampled')
    check(result['primary_policy_aliases']==primary
          and set(result['deduplicated_models'])=={affine_id,baseline_id},'selected_identities_deduplicated')
    for identity,record in result['deduplicated_models'].items():
        expected_keys=certified_keys if identity==affine_id else baseline_keys
        check(record['aliases']==expected_keys,'sensitivity_aliases_excluded_from_primary_cost')
        for ordinal,row in enumerate(record['queries']):
            check(row['query_index']==ordinal and row['sample_ordinal']==ordinal,
                  'query_index_and_counterbalance_ordinal_retained')
            if identity==affine_id:
                check(row['primitive_order'][:2]==(['prediction_without_certificate','certified_prediction']
                      if ordinal%2==0 else ['certified_prediction','prediction_without_certificate']),
                      'full_inference_pair_order_counterbalanced')
                for key,values in row['nanoseconds'].items():
                    check(len(values)==10 and all(type(value) is int and value>=0 for value in values)
                          and row['median_nanoseconds'][key]==float(np.median(values)),
                          'raw_primitive_samples_and_medians')
                check(row['verification_status']=='PASS' and row['expected_acceptance'] is True,
                      'honest_assessed_acceptance_replayed')
            else:
                check(len(row['nanoseconds'])==10 and row['median_nanoseconds']==float(np.median(row['nanoseconds'])),
                      'baseline_raw_samples_and_median')
        if identity==affine_id:
            check(len(record['static_context']['nanoseconds'])==100,'static_context_100_samples')
            for key,summary in record['component_summary'].items():
                check(summary==cost.timing_summary([row['median_nanoseconds'][key] for row in record['queries']]),
                      'affine_component_summary_reproduced')
        else:
            check(record['component_summary']['selected_conventional_prediction']==cost.timing_summary(
                [row['median_nanoseconds'] for row in record['queries']]),'baseline_summary_reproduced')
    for relative,digest in result['source_bindings'].items():
        check(c.sha_file(FIXTURE/relative)==digest,'retained_cost_evidence_hash_binding')

    receipt={'schema':'step43-artificial-cost-adapter-checks-v1','status':'PASS',
             'checks':sum(checks.values()),'groups':dict(checks),
             'wall_seconds':time.perf_counter()-start,
             'source_sha256':{name:c.sha_file(ROOT/'source'/name)
                              for name in (*source_names,'test_cost43.py')},
             'fixture':c.file_ref(FIXTURE/'fixture_notice43.json'),
             'artificial_cost':c.file_ref(FIXTURE/f'output/benchmarks/{task}/cost43.json'),
             'natural_source_measurements_read':0,'real_assessment_outcomes_read':0,
             'manufactured_affine_fits':1,'manufactured_conventional_fits':1,
             'genuine_protocol_and_gate_rejected_artificial_stubs':True,
             'gate_redirects_scoped_to_test_and_restored':True,
             'real_runner_context_loader_executed':True,
             'scope':'Interface, serialization, replay, deduplication, timing counts/order/summaries only; artificial timings are not study latency evidence; no scientific selection or gate PASS is claimed'}
    c.write_json(ROOT/'analysis/cost_artificial_checks43.json',receipt)
    print(c.canonical_bytes({'status':'PASS','checks':receipt['checks'],
                             'wall_seconds':receipt['wall_seconds']}).decode(),end='')


if __name__=='__main__':
    run()
