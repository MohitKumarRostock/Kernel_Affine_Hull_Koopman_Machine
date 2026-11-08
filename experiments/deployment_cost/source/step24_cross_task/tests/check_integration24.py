"""Pure manufactured integration checks; no data loader, generator or fit runs."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'step23_cross_task'
SOURCE_PATHS=[Path(__file__).resolve(),*[ROOT/'src'/name for name in
    ('common24.py','pipeline24.py','fit24.py','models24.py','geometry24.py')],
    PARENT/'PROTOCOL.json',PARENT/'protocol/policy23.py',
    PARENT/'protocol/candidate_templates23.json']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


IMPORT_HASHES={str(p.relative_to(ROOT.parent)):sha(p) for p in SOURCE_PATHS}
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(PARENT/'protocol'))
import common24 as C
import pipeline24 as P
import fit24 as F
import models24 as M
import geometry24 as G
from policy23 import select_candidates, observed_utility_gate


def rejects(fn):
    try:fn()
    except (AssertionError,ValueError,TypeError,RuntimeError,ArithmeticError,FileNotFoundError):
        return True
    raise AssertionError('A manufactured invalid/corrupted case was accepted')


def exact_reference(y,p):
    return sum((Fraction.from_float(float(a))-Fraction.from_float(float(b)))**2
               for a,b in zip(y,p))/len(y)


def main():
    receipt_path=ROOT/'tests/integration_preflight24.json'
    if receipt_path.exists():raise FileExistsError('Manufactured integration receipt is write-once')
    checks=[];details={};rejected=0
    tiny=np.nextafter(0.,1.)
    pairs=[
        (np.array([0.,-0.,1.,-1.]),np.array([tiny,-tiny,np.nextafter(1.,np.inf),np.nextafter(-1.,-np.inf)])),
        (np.array([2.**52,2.**52+1.]),np.array([2.**52+1.,2.**52-1.])),
        (np.array([1e100,-1e100,1e-100]),np.array([np.nextafter(1e100,np.inf),np.nextafter(-1e100,0.),0.])),
        (np.array([tiny,0.]),np.array([0.,tiny])),
        (np.array([1.,2.,3.]),np.array([1.,2.,3.])),
        (np.array([.1,.2,.3]),np.array([.3,.2,.1])),
    ]
    for y,p in pairs:
        actual=C.exact_mse(y,p);expected=exact_reference(y,p)
        assert actual==expected
        enclosure=C.bracket(actual)
        assert Fraction.from_float(enclosure['lower'])<=actual<=Fraction.from_float(enclosure['upper'])
    assert C.exact_sum_dyadics([1e100,1.,-1e100])==Fraction(1)
    assert C.exact_mse(*pairs[3])>0 and float(C.exact_mse(*pairs[3]))==0.
    for fn in [lambda:C.exact_mse([np.nan],[0.]),lambda:C.exact_mse([1.,2.],[1.]),
               lambda:C.exact_mse([[1.]],[[0.]])]:
        assert rejects(fn);rejected+=1
    checks.append('Exact integer/dyadic MSE equals independent Fraction arithmetic for6 cancellation, subnormal, nextafter, large-scale and zero cases; outward brackets contain exact values')

    # Same three manufactured observations yield exactly10 and11 MSE.
    y=np.zeros(3);reference_risk=C.exact_mse(y,np.array([1.,2.,5.]));boundary_risk=C.exact_mse(y,np.array([1.,4.,4.]))
    assert reference_risk==10 and boundary_risk==11
    records=[{'id':'Cacc','index':0,'group':'RAW','validation_mse':10.,'complete_numeric_bytes':100},
             {'id':'Ccost','index':1,'group':'UNFITTED','validation_mse':11.,'complete_numeric_bytes':10},
             {'id':'P','index':2,'group':'P','validation_mse':11.,'complete_numeric_bytes':5},
             {'id':'CS','index':3,'group':'CS','validation_mse':np.nextafter(11.,np.inf),'complete_numeric_bytes':4}]
    selected=select_candidates(records)
    assert selected['C_accuracy']['id']=='Cacc' and selected['C_cost']['id']=='Ccost'
    assert selected['P_cost']['id']=='P' and selected['CS_cost'] is None
    observed=lambda cid,group,risk,cost:{'id':cid,'group':group,'test_mse_exact':risk,
                'complete_numeric_bytes':cost,'median_warm_batch_nanoseconds_per_row':Fraction(100)}
    ca=observed('Cacc','RAW',reference_risk,100);cc=observed('Ccost','UNFITTED',reference_risk,10)
    k=observed('P','P',boundary_risk,5)
    assert observed_utility_gate(k,ca,cc)['all_three_pass']
    assert not observed_utility_gate(dict(k,test_mse_exact=boundary_risk+Fraction(1,10**40)),ca,cc)['accuracy_pass']
    assert not observed_utility_gate(dict(k,complete_numeric_bytes=10),ca,cc)['memory_pass']
    assert not observed_utility_gate(None,ca,cc)['all_three_pass']
    checks.append('Exact MSE feeds the11/10 utility gate correctly: equality passes accuracy/latency, any positive exact risk excess fails, equal storage fails, and missing candidates remain missing')

    # Full fixed template roster with purely manufactured validation costs/losses.
    templates=C.templates();assert len(templates)==654
    wanted=next(t for t in templates if t['group']=='P' and t['scaling']=='standard'
                and t['cap']==8 and t['prefix']==14 and t['omega']==64.
                and t['family']=='hgb' and t['max_leaf_nodes']==15)
    binding={k:wanted[k] for k in ('scaling','cap','prefix','omega')}
    full=[]
    for template in templates:
        spec={**template,**binding} if template['group']=='MATCHED_DIAGNOSTIC' else dict(template)
        full.append({**spec,'validation_mse':2.+spec['index']*1e-5,
                     'complete_numeric_bytes':100000+10*spec['index']})
    by_id={r['id']:r for r in full}
    by_id[wanted['id']]['validation_mse']=.9
    raw=next(r for r in full if r['group']=='RAW' and r['family']=='ridge')
    raw['validation_mse']=1.
    cs=next(r for r in full if r['group']=='CS' and r['family']=='ridge')
    cs['validation_mse']=1.08
    for row in full:
        if row['group']=='MATCHED_DIAGNOSTIC':row['validation_mse']=.01+row['index']*1e-6
    for index,name,risk,cost in [(654,'persistence',1.05,16),(655,'daily_seasonal',1.08,24),(656,'training_mean',1.3,8)]:
        full.append({'id':'b_'+name,'index':index,'group':'UNFITTED','view':'BASELINE',
                     'family':'unfitted','baseline_name':name,'validation_mse':risk,'complete_numeric_bytes':cost})
    selectors=select_candidates(full);roster=P.choose_roster(full,selectors,binding)
    assert selectors['P_accuracy']['id']==wanted['id']
    assert selectors['C_accuracy']['id']==raw['id'] and selectors['C_cost']['id']=='b_persistence'
    assert sum(r.get('family')=='hgb' for r in full)==300
    assert len(roster['fitted_ids'])<=21 and 2*len(roster['fitted_ids'])<=42
    all_by_id={r['id']:r for r in full}
    for view in ('P','CS','LR0','LRM','LW'):
        for family in ('ridge','hgb'):
            chosen=all_by_id[roster['names'][f'matched_{view}_{family}']]
            assert chosen['view']==view and chosen['family']==family
            assert all(chosen[k]==binding[k] for k in ('scaling','cap','prefix'))
            if view!='CS':assert chosen['omega']==binding['omega']
    reverse=P.choose_roster(list(reversed(full)),select_candidates(list(reversed(full))),binding)
    assert roster==reverse
    details['manufactured_roster']={'template_count':654,'hgb_templates':300,
        'selected_fitted_ids':len(roster['fitted_ids']),'secondary_fits_if_executed':2*len(roster['fitted_ids']),
        'P_binding':binding,'low_loss_diagnostics_excluded_from_primary_selectors':True}
    checks.append('All654 template slots resolve into the correct matched-view/family roster; primary selection excludes diagnostic wins, order is deterministic, and secondary configuration count stays below42')

    ridge={'kind':'ridge','coef':np.array([.3,-.2,.1]),'intercept':np.float64(.2),'output_bounds':M.output_bounds('hmm')}
    rawpipe={'pipeline_kind':'raw','head':ridge,'task_kind':'hmm'}
    testX=np.linspace(-1.,1.,30).reshape(10,3)
    M.audit_export(P.predict(rawpipe,testX),P.predict(rawpipe,testX,reference=True),{'offset':0.,'scale':1.})
    nys={'kind':'nystrom','scaler':{'offset':np.array([0.,1.]),'divisor':np.array([2.,3.]),
                                'constant_mask':np.array([False,False])},
         'landmarks':np.array([[-.5,.25],[.1,-.3],[1.,.7]]),'gamma':np.float64(.4),
         'coef':np.array([.7,-.4,.25]),'intercept':np.float64(.1),'output_bounds':M.output_bounds('hmm')}
    nyspipe={'pipeline_kind':'nystrom','model':nys,'task_kind':'hmm'}
    query=np.linspace(-2.,2.,518).reshape(259,2)
    nominal=P.predict(nyspipe,query)
    def forbidden_nominal(*args,**kwargs):raise AssertionError('Independent Nystrom route invoked nominal kernel/predictor')
    with patch.object(M,'predict_nystrom',forbidden_nominal),patch.object(M,'rbf_kernel',forbidden_nominal):
        independent=P.predict(nyspipe,query,reference=True)
    audit=M.audit_export(nominal,independent,{'offset':0.,'scale':1.},'manufactured independent compact kernel')
    details['independent_nystrom']=audit
    checks.append('Reference Nyström replay bypasses both nominal predictor and kernel, covers multiple128-row chunks, and agrees within the fixed export tolerance')

    # Owning-array encoder fixtures exercise complete geometry payload cost
    # without fitting, scoring, or invoking the native runtime.
    fixture_blocks=[]
    for width in (2,4,8,16):
        classifier=[]
        for _ in range(width):
            ae={'y_data_n_subspace':np.array([[0.,1.]]),
                'PC':np.array([[1.],[0.],[0.]]),'B':np.eye(2),
                'y_data':np.array([[0.,1.],[0.,0.],[0.,0.]]),
                'sqrt_weights_matrix':np.eye(3),'kerneltype':'gaussian'}
            classifier.append((ae,))
        fixture_blocks.append({'classifier':tuple(classifier),'cluster_centers':np.zeros((3,width)),
                               'n_clusters':np.int64(width),'classifier_dir':None})
    fixture_bank={'kind':'kahm_encoder','scaler':{'offset':np.zeros(3),'divisor':np.ones(3),
                                              'constant_mask':np.zeros(3,dtype=bool)},
                  'blocks':tuple(fixture_blocks),'block_sizes':np.array([2,4,8,16],dtype=np.int64),
                  'score_batch_size':np.int64(1000)}
    geometry_head={'kind':'ridge','coef':np.linspace(-.1,.1,30),'intercept':np.float64(.2),
                   'output_bounds':M.output_bounds('hmm')}
    fixture_banks={G.bank_key('minmax01',2):fixture_bank}
    geometry_p=P.make_pipeline({'group':'P','view':'P','scaling':'minmax01','cap':2,
                              'prefix':30,'omega':64.},geometry_head,fixture_banks,'hmm')
    geometry_cs=P.make_pipeline({'group':'CS','view':'CS','scaling':'minmax01','cap':2,
                               'prefix':30,'omega':None},geometry_head,fixture_banks,'hmm')
    costs={}
    payloads={'raw':rawpipe,'nystrom':nyspipe,'geometry_P':geometry_p,'geometry_CS':geometry_cs,
              'LR0_transform':{'transform':G.view_payload('LR0',(2,4,8,16),64.),'head':ridge},
              'LRM_transform':{'transform':G.view_payload('LRM',(2,4,8,16),64.),'head':ridge}}
    for name,payload in payloads.items():
        before=P.payload_cost(payload)['complete_numeric_bytes']
        restored=pickle.loads(pickle.dumps(payload,protocol=5))
        after=P.payload_cost(restored)['complete_numeric_bytes']
        assert before==after,(name,before,after)
        costs[name]=before
    details['pickle_stable_costs']=costs
    assert costs['geometry_P']-costs['geometry_CS']==16
    checks.append('Complete numeric payload cost is stable through serialization for raw, compact-kernel and all-block Walsh transformation payloads')

    unexpected={'calls':0}
    def no_development_or_fit(*args,**kwargs):
        unexpected['calls']+=1
        raise AssertionError('A pure integration test attempted data access or fitting')
    with tempfile.TemporaryDirectory(prefix='step24_pure_integration_') as temporary:
        tmp=Path(temporary)
        blob=tmp/'cache.bin';blob.write_bytes(b'manufactured cache bytes')
        F._cache_receipt(blob,'context-A',{'parent':'upstream-A'})
        F._verify_cache(blob,'context-A',{'parent':'upstream-A'})
        for fn in [lambda:F._verify_cache(blob,'context-B',{'parent':'upstream-A'}),
                   lambda:F._verify_cache(blob,'context-A',{'parent':'upstream-B'})]:
            assert rejects(fn);rejected+=1
        original=blob.read_bytes();blob.write_bytes(original+b'corrupted')
        assert rejects(lambda:F._verify_cache(blob,'context-A',{'parent':'upstream-A'}));rejected+=1
        blob.write_bytes(original)
        orphan=tmp/'uncommitted.bin';orphan.write_bytes(b'no receipt')
        assert rejects(lambda:F._verify_cache(orphan,'context-A'));rejected+=1

        # A committed candidate record is verified without invoking its fit path.
        candidate_folder=tmp/'candidate';spec=next(t for t in templates if t['group']=='RAW' and t['family']=='ridge')
        cid=spec['id'];hp=candidate_folder/'heads'/f'{cid}.pkl';pp=candidate_folder/'validation_predictions'/f'{cid}.npy'
        rp=candidate_folder/'candidate_records'/f'{cid}.json'
        C.write_pickle(hp,ridge);pp.parent.mkdir(parents=True);np.save(pp,np.array([.2,.3]),allow_pickle=False)
        context='candidate-context';upstream={}
        record={'id':cid,'source_spec':spec,'context_sha256':context,'upstream':upstream,
                'validation_mse':.065,'complete_numeric_bytes':56,
                'head_sha256':C.sha(hp),'prediction_sha256':C.sha(pp)}
        C.write_json(rp,record);F._cache_receipt(rp,context,upstream)
        dummy=SimpleNamespace(context=context,upstream=lambda s:upstream)
        with patch.object(F,'_fit_one',no_development_or_fit):
            result=F._candidate(spec,candidate_folder,None,None,None,dummy,'hmm')
            assert result==record
            assert rejects(lambda:F._candidate(spec,candidate_folder,None,None,None,
                       SimpleNamespace(context='wrong-context',upstream=lambda s:upstream),'hmm'));rejected+=1
            old_record=rp.read_bytes();tampered=dict(record,validation_mse=0.)
            rp.write_text(json.dumps(tampered))
            assert rejects(lambda:F._candidate(spec,candidate_folder,None,None,None,dummy,'hmm'));rejected+=1
            rp.write_bytes(old_record)
            old_head=hp.read_bytes();hp.write_bytes(old_head+b'changed')
            assert rejects(lambda:F._candidate(spec,candidate_folder,None,None,None,dummy,'hmm'));rejected+=1
            hp.write_bytes(old_head)

        # Completion replay verifies all sealed outputs before an early return.
        fake_run=tmp/'run';fake_run.mkdir()
        C.write_json(fake_run/'implementation_seal.json',{'manufactured_fixture_only':True})
        task=fake_run/'hmm_0';task.mkdir();out=task/'chosen.fixture';out.write_text('synthetic output')
        done={'implementation_seal_sha256':C.sha(fake_run/'implementation_seal.json'),
              'sealed_outputs':{'chosen.fixture':C.sha(out)},'primary_fits':0,'secondary_fits':0}
        C.write_json(task/'fit_complete.json',done)
        with patch.object(F,'RUN',fake_run),patch.object(F,'check_implementation_seal',lambda:{'manufactured_fixture_only':True}),\
             patch.object(F,'log',lambda *a,**k:None),patch.object(F.D,'load_development',no_development_or_fit):
            F.fit_task('hmm_0')
            out.write_text('changed output')
            assert rejects(lambda:F.fit_task('hmm_0'));rejected+=1
    assert unexpected['calls']==0
    checks.append('Cache/candidate/completion checkpoints reject wrong context, changed upstream, uncommitted files, altered candidate loss, corrupt model bytes and modified completed outputs; no data/fit path invoked')

    final_hashes={str(p.relative_to(ROOT.parent)):sha(p) for p in SOURCE_PATHS}
    if final_hashes!=IMPORT_HASHES:raise RuntimeError('Implementation sources changed during the manufactured check')
    receipt={'schema':'step24-pure-integration-preflight-v1','passed':True,
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'checks':checks,'check_groups':len(checks),'corrupt_or_invalid_cases_rejected':rejected,
        'details':details,'model_fits':0,'benchmark_trajectories_generated':0,
        'real_dataset_rows_parsed':0,'ECG_or_reserve_access':False,
        'unexpected_development_or_fit_calls':unexpected['calls'],'source_sha256':final_hashes}
    with receipt_path.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256','details')},indent=2))


if __name__=='__main__':main()
