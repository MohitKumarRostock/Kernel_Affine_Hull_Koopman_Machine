"""Recompute primary forecast risks, complete numeric costs and utility decisions."""
from pathlib import Path
import argparse, hashlib, json, pickle, sys, time
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/step24_cross_task/src'))
import numpy as np
import common24 as C
import models24 as M
import pipeline24 as P

def rational(d):return F(int(d['numerator']),int(d['denominator']))
def exact_mse(y,p):
    # Differences are formed as exact rationals, before squaring.
    check(len(y)>0 and np.asarray(y).shape==np.asarray(p).shape and np.asarray(y).ndim==1,'Finite scalar target/prediction roster')
    check(np.isfinite(y).all() and np.isfinite(p).all(),'Finite forecast values')
    return sum(((F.from_float(float(a))-F.from_float(float(b)))**2 for a,b in zip(y,p)),F())/len(y)
def check(ok,msg):
    if not ok:raise AssertionError(msg)

def run(recompute_predictions=False):
    results=[];models_checked=0;max_prediction_error=0.0
    for task in ['hmm_0','hmm_1','hmm_2','hmm_3','hmm_4','appliances']:
        folder=ROOT/'data'/task; record=json.loads((folder/'record.json').read_text())
        with np.load(folder/'observations.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        with (folder/'models.pkl').open('rb') as f:models=pickle.load(f)
        risks={}; costs={}; times={}
        for cid,model in models.items():
            prediction=arrays['prediction_'+cid]
            risks[cid]=exact_mse(arrays['y'],prediction)
            check(len(prediction)==record['metrics'][cid]['rows'],f'{task}/{cid}: observation count')
            check(risks[cid]==rational(record['metrics'][cid]['mse_exact']),f'{task}/{cid}: exact MSE')
            costs[cid]=M.numeric_inventory(model)['numeric_bytes']
            check(costs[cid]==record['costs'][cid]['complete_numeric_bytes'],f'{task}/{cid}: complete model bytes')
            samples=sorted(record['timings'][cid]['warm_batch_nanoseconds'])
            check(len(samples)==9,f'{task}/{cid}: timing sample count')
            times[cid]=F(samples[4],record['batch_rows'])
            check(times[cid]==rational(record['timings'][cid]['median_warm_batch_nanoseconds_per_row']),f'{task}/{cid}: latency median')
            if recompute_predictions:
                rebuilt=P.predict(model,arrays['X'])
                error=float(np.max(np.abs(rebuilt-prediction)))
                max_prediction_error=max(max_prediction_error,error)
                check(np.allclose(rebuilt,prediction,rtol=1e-10,atol=1e-9),f'{task}/{cid}: raw-input prediction replay')
            models_checked+=1
        ca=record['names']['C_accuracy']; cc=record['names']['C_cost'];cap=F(11,10)*risks[ca]
        for name in ['P_cost','CS_cost']:
            cid=record['names'][name];expected=record['expected_gates'][name]
            accuracy=cid is not None and risks[cid]<=cap
            memory=cid is not None and costs[cid]<costs[cc]
            latency=cid is not None and times[cid]<=times[cc]
            actual={'accuracy_pass':accuracy,'memory_pass':memory,'latency_pass':latency,'all_three_pass':accuracy and memory and latency}
            for k,v in actual.items():check(v==expected[k],f'{task}/{name}: {k}')
            results.append({'task':task,'procedure':name,**actual})
    summary={name:{k:sum(r['procedure']==name and r[k] for r in results) for k in ['accuracy_pass','memory_pass','latency_pass','all_three_pass']} for name in ['P_cost','CS_cost']}
    check(all(v=={'accuracy_pass':6,'memory_pass':0,'latency_pass':0,'all_three_pass':0} for v in summary.values()),'Primary panel counts')
    return {'status':'PASS','tasks':6,'models_checked':models_checked,'procedures':summary,'rows':results,
            'raw_input_predictions_recomputed':recompute_predictions,'maximum_prediction_absolute_difference':max_prediction_error,
            'scope':'Exact MSE from stored binary64 forecasts; numeric inventories reconstructed from retained model payloads; medians recomputed from original timing samples. Timings are not remeasured; no new training.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path);ap.add_argument('--predict',action='store_true');args=ap.parse_args()
    start=time.perf_counter();result=run(args.predict);result['wall_seconds']=time.perf_counter()-start
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
