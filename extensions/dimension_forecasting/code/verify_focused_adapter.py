#!/usr/bin/env python3
"""Numerical and staging gates using only manufactured, non-study inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['NUMBA_DISABLE_JIT']='1'
import sys
sys.dont_write_bytecode=True
import copy
import argparse
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

from focused_dimension_model import (load_runtime,fit_shared,fit_candidate,calibrate,
    calibrate_shared,select_context_policies,prediction_views,
    select_cap,state_hash,METHODS,GEOMETRY_METHODS,FULL_METHODS,SHARED_METHODS)
from dimension_worker import fit_task,predict_task,write_json,write_npz,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir',type=Path,default=ROOT/'verification/schema_fixture')
    parser.add_argument('--receipt',type=Path,default=ROOT/'verification/focused_adapter_tests.json')
    args=parser.parse_args()
    rng=np.random.default_rng(91847266)
    ax,ay,bx,by,cx,cy=[rng.normal(size=(n,w)) for n,w in
                      [(96,12),(96,3),(48,12),(48,3),(48,12),(48,3)]]
    query=rng.normal(size=(19,12))
    checks=0;records=[]
    source=load_runtime()
    with threadpool_limits(limits=1):
        shared=fit_shared(ax,ay,bx,by,seed=4721)
        base=shared.baseline_fit
        shared_cal,shared_raw_c,shared_cal_c=calibrate_shared(shared,cx,cy)
        shared_query=shared_cal.predict(query)
        for cap,policy in [(2,'legacy_normal_departure'),(20,'native')]:
            full=source.fit_suite(ax,ay,bx,by,cx,cy,seed=4721,
                                 config={'rank':cap,'dimension_policy':policy})
            focused=fit_candidate(ax,ay,bx,by,candidate_id=f'cap{cap}',rank_cap=cap,
                                  dimension_policy=policy,seed=4721,baseline_fit=base)
            calibrated,raw_c,cal_c=calibrate(focused,cx,cy)
            a,b=full.predict(query),calibrated.predict(query)
            choices={method:{'selected_candidate_id':'historical2'} for method in GEOMETRY_METHODS}
            contexts=select_context_policies(choices,{'historical2':cal_c,'native20':cal_c},
                                             shared_cal_c,cy,base.loss_scale)
            views=prediction_views({'historical2':b,'native20':b},shared_query,contexts,'historical2')
            for method in FULL_METHODS:
                np.testing.assert_array_equal(a['predictions'][method],
                    views['context_selected__calibrated__'+method]);checks+=1
            assert contexts['selected']['selected_policy']==full.selected_policy;checks+=1
            assert contexts['selected']['policy']['candidates']==full.diagnostics['policy']['candidates'];checks+=1
            for method in SHARED_METHODS:
                assert shared.diagnostics['B'][method]==full.diagnostics['B'][method];checks+=1
                assert shared_cal.calibration['C'][method]==full.diagnostics['C'][method];checks+=1
                if method!='ridge':
                    np.testing.assert_array_equal(shared_cal.gammas[method],full.gammas[method]);checks+=1
            for method in METHODS:
                np.testing.assert_array_equal(a['predictions'][method],b['predictions'][method]);checks+=1
                if method in GEOMETRY_METHODS:
                    np.testing.assert_array_equal(a['uncalibrated_predictions'][method],
                                                  b['uncalibrated_predictions'][method]);checks+=1
                    np.testing.assert_array_equal(full.gammas[method],calibrated.gammas[method]);checks+=1
                    np.testing.assert_array_equal(full.heads[method]['coef'],focused.heads[method]['coef']);checks+=1
                    np.testing.assert_array_equal(full.heads[method]['alpha_by_horizon'],
                                                  focused.heads[method]['alpha_by_horizon']);checks+=1
            before=state_hash(focused)
            calibrate(focused,cx,cy*100+3)
            assert state_hash(focused)==before;checks+=1
            zero_target=focused.predict_uncalibrated(cx)[0]['ridge']
            zero,_,_=calibrate(focused,cx,zero_target)
            out=zero.predict(query)
            for method in GEOMETRY_METHODS:
                np.testing.assert_array_equal(zero.gammas[method],np.zeros(3));checks+=1
                np.testing.assert_array_equal(out['predictions'][method],out['predictions']['ridge']);checks+=1
            records.append({'cap':cap,'policy':policy,'fixed_cap_principal_parity':'exact',
                            'complete_ten_method_parity':'exact','raw_B_grids_and_C_policy_parity':'exact',
                            'C_does_not_mutate_AB_state':True,'gamma_zero_exact_baseline':True})
        one=fit_candidate(ax,ay,bx,by,candidate_id='native1',rank_cap=1,seed=4721,baseline_fit=base)
        two=fit_candidate(ax,ay,bx,by,candidate_id='native2',rank_cap=2,seed=4721,baseline_fit=base)
        for method in GEOMETRY_METHODS:
            one.b_predictions[method]=base.predictions_b.copy()
            two.b_predictions[method]=base.predictions_b.copy()
        selected,record=select_cap({'native1':one,'native2':two},by,
                                   cap_order=('native1','native2'))
        assert selected=='native1';checks+=1
        assert 'c_y' not in inspect.signature(select_cap).parameters;checks+=1
        assert 'c_x' not in inspect.signature(fit_candidate).parameters;checks+=1
        assert 'c_x' not in inspect.signature(fit_shared).parameters;checks+=1
        assert tuple(inspect.signature(type(calibrated).predict).parameters)==('self','x');checks+=1
        with patch.object(source._InstrumentedBank,'_fit_kahm_factor',side_effect=RuntimeError('injected factor fault')):
            try:fit_candidate(ax,ay,bx,by,candidate_id='failure',rank_cap=2,seed=4721,baseline_fit=base)
            except FloatingPointError as error:
                assert 'injected factor fault' in str(error);checks+=1
            else:raise AssertionError('Injected factor fault did not stop fitting')
        broken=copy.deepcopy(shared.diagnostics['B']['raw_hgb'])
        broken['failures']=[{'exception':'injected raw grid fault'}]
        with patch.object(source,'_fit_raw_hgb',return_value=(shared.raw_hgb,shared.b_predictions['raw_hgb'],broken)):
            try:fit_shared(ax,ay,bx,by,seed=4721)
            except FloatingPointError as error:
                assert 'injected raw grid fault' in str(error);checks+=1
            else:raise AssertionError('Raw candidate failure did not stop fitting')
        before=state_hash(shared)
        calibrate_shared(shared,cx,cy*100+3)
        assert state_hash(shared)==before;checks+=1
        zero_shared,_,_=calibrate_shared(shared,cx,shared_raw_c['ridge'])
        for method in SHARED_METHODS[1:]:
            np.testing.assert_array_equal(zero_shared.gammas[method],np.zeros(3));checks+=1
            np.testing.assert_array_equal(zero_shared.predict(query)['predictions'][method],
                                          zero_shared.predict(query)['predictions']['ridge']);checks+=1
        with patch.object(source._InstrumentedBank,'fit',side_effect=AssertionError('query must not fit')):
            before=state_hash(calibrated)
            calibrated.predict(query)
            assert state_hash(calibrated)==before;checks+=1
        fixture=args.fixture_dir
        if fixture.exists():
            raise FileExistsError('Preserve existing schema fixture; choose fresh verification directory')
        fixture.mkdir(parents=True)
        input_path=fixture/'abc.npz'
        write_npz(input_path,{'A_X':ax,'A_Y':ay,'B_X':bx,'B_Y':by,'C_X':cx,'C_Y':cy})
        task={'task_id':'manufactured_fixture','law':'manufactured','history':12,
              'seed':91847266,'repetition':0,'noise_index':0,'noise_level':0.,'fit_seed':4721,
              'input':'abc.npz','sha256':sha256_file(input_path),'manufactured_only':True}
        fit=fit_task(input_path,task,fixture/'results',freeze_sha256='manufactured_fixture_no_study_data')
        lock=fixture/'FIT_LOCK.json'
        write_json(lock,{'scope':'Manufactured schema fixture only','fit_summary_sha256':sha256_file(fixture/'results/fit_summary.json')})
        # Query generation occurs after every manufactured fixture state is fitted.
        query_path=fixture/'queries.npz'
        write_npz(query_path,{'D_X':rng.normal(size=(19,12))})
        predict_task(fixture/'results',query_path,fixture/'predictions',fit_lock_sha256=sha256_file(lock))
        checks+=4
    result={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Manufactured iid matrices only; no production outcomes or study fits',
            'assertions':checks,'parity_fixtures':records,
            'cap_tie_ascending':True,'injected_factor_failure_stops':True,'injected_raw_grid_failure_stops':True,
            'fixed_query_no_state_mutation':True,'schema_fixture':str(fixture),
            'historical_native2_identity':fit['historical_native2_identity']}
    target=args.receipt
    write_json(target,result)
    print(json.dumps({'status':'passed','assertions':checks,'schema_fixture':str(fixture)},indent=2))


if __name__=='__main__':main()
