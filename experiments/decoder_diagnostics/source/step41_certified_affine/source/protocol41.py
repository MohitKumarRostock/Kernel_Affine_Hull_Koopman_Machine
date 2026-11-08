"""Freeze protocol and complete scientific source before new values are decoded."""
import platform,importlib.metadata
from pathlib import Path
import common41 as c
import models41 as m
import representations35 as rep
def maps():
    return sorted(['raw_'+s for s in ('minmax01','standard')]+['nystrom_'+s+'_w'+str(w) for s in ('minmax01','standard') for w in (8,24)]+['kahm_'+s+'_cap'+str(k)+'_p'+str(p) for s in ('minmax01','standard') for k in (4,12,20) for p in (1,2)])
def protocol():
    import data41 as d
    names=maps();baselines=m.baseline_configs();affine=[{'candidate_id':name+'::affine','map':name,'recipe':'affine','token':'affine'} for name in names]
    return {'schema':'step41-affine-deployment-protocol-v1','step':41,'created_utc':c.utc(),'seed':3901,
      'question':'Does the unchanged certified affine selector stay within a fixed 5% test-MSE tolerance of a stronger conventional selector on two forward periods, and what does deployed prediction verification cost?',
      'data':d.data_contract(),'tasks':list(d.task_ids()),'stages':['f1','f2','f3','final'],'maps':names,'representations':rep.DEFAULT_CONFIG,
      'heads':{'configurations':affine,'affine_ridge_lambda':'1/1000000','kappa':'2','normalized_range':['0','1'],'fit_recipe':'Byte-identical inherited fit_affine function; ridge then exact norm projection/recentering and clipping','training_optimality_claim':False},
      'baselines':{'configurations':baselines,'count':44,'scaling':'training-only StandardScaler per fitted baseline','same24lags':True,'target_scaling':'fold training target_min/span; KRR additionally centers normalizedtrainingtarget','clipped':False,'class_membership_certified':False},
      'selection':{'folds':['f1','f2','f3'],'criterion':'sum physical operational-prediction SSE / total development queries','tie_breakers':['exact pooled physical MSE','f3 map+deployment or baselinepayload bytes','lexical candidate_id'],
        'selectors':['affine_rolling','ordinary_rolling','ridge_rolling','hgb_rolling','svr_rolling','krr_rolling','extra_trees_rolling','random_forest_rolling','raw_affine','nystrom_affine','kahm_affine'],
        'primary':'affine_rolling','reference':'ordinary_rolling','test_reselection':False,'final_refit':'All62 configurations on permitted trainingthroughAugust; selectedIDsunchanged'},
      'certificate':{'reference':'exact clipped affine predictor with storedrational coefficients on storedbinary64features and exacttargetnormalization','normalized_error_budget':'1/1099511627776','physical_error_budget':'training_span * normalized_error_budget','native_transform_error_enclosed':False,'raw_input_robustness':False,'future_label_coverage':False,'population_risk':False,'acceptance':'all query identities bind; exact arithmetic and tight outward reference/error interval checks; recomputed exact physicalerror<=physicalbudget'},
      'evaluation':{'prediction_ids':[r['candidate_id'] for r in affine+baselines],'primary_metric':'Exact squaredloss of reportedbinary64physical predictions against exactrawtarget values','exact_affine_reference':'secondary; all retained',
        'primary_per_task':'baselineMSE>0 and20*affineMSE<=21*baselineMSE andallvaliditygatesPASS','overall':'Both continuationcohorts pass','tolerance_rationale':'A predeclared practical5% MSE tolerance to assess the accuracycost of a verifiablepredictor; not a statisticalnoninferiority margin, SLA, or communitystandard',
        'secondary':['strictlylowerMSE','atleast5%lowerMSE','MAE','allfamilies','monthlydescriptive','exactreferenceordering'],
        'all_candidate_outcomes_retained':True,'outcome_based_reselection':False,'population_significance_test':False},
      'timing':{'scope':'selected final affine predictor only, same stored test queries after assessment; no fitting','threads':1,'warmups_per_query_primitive':3,'repeats_per_query_primitive':10,'static_prepare_repeats':100,
        'primitives':['transform','head_inference','certificate_construction','certificate_serialization','certificate_verification','prediction_without_certificate','certified_prediction','ordinary_prediction'],
        'summary':'perquerymedian across10repeats; p50,p95 acrossqueries using numpy quantilemethodlinear; raw nanosecondsretained','runtime_threshold':None,'speedup_claim':False,
        'payloads':['mapbytes','staticdeploymentbytes','certificateJSONbytes','requiredfeaturebytes','certificateplusfeaturebytes'],
        'verification_includes':'packetJSONparse plus check withcachedtrustedstaticcontext; staticprepareincludesdeploymentJSONparse; fullcertifiedprediction timestransform,inference,construction,serialization,parse,verification directly',
        'ordinary_prediction':'selectedordinarybaseline, itsinputscaling/prediction/inversetargetscaling on sameonequeryphysicalX',
        'exclude':'disk/networkI/O andprocessstartup; nopeakRAMclaim; componenttimesmustnotbesummedwithdirecttotals'},
      'execution':{'single_primary_pass':True,'scientific_reruns_allowed':False,'threads':1,'phase_budget_seconds':{'fit':1800,'validate':1800,'select':120,'test':3600,'benchmark':600,'aggregate':120},'failure_policy':'STOP_RETAIN_FAILURE_NO_RETUNING'},
      'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib','threadpoolctl']}},
      'prior_evidence':{'step40_archive_sha256':'3da58e36ed6a318ab748604f8b58a3ac293ee96f831987874e64fc38ca3f70ee','step40_primary':'NOT_ESTABLISHED','motivation':'Predeclared affine secondary findings; no relabeling Step40 primary','inherited_sources_sha256':c.sha(c.ROOT/'inputs/inherited41.json'),'new_domains':False,'new_forward_periods':True,'reserved_patients_accessed':0}}
def seal():
    import data41 as d
    if (c.ROOT/'protocol_seal41.json').exists():raise RuntimeError('Protocol already frozen')
    p=protocol();c.save(c.ROOT/'protocol41.json',p)
    files=[c.ROOT/'protocol41.json']
    for folder in ('source','inputs','analysis'):
        files.extend(x for x in (c.ROOT/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc')
    record={'schema':'step41-protocol-source-seal-v1','sealed_utc':c.utc(),'protocol_sha256':c.sha(c.ROOT/'protocol41.json'),'bound_files':{c.relative(x):c.sha(x) for x in sorted(set(files))},'new_measurement_tokens_decoded':0,'single_primary_pass':True}
    ref=c.save(c.ROOT/'protocol_seal41.json',record);d.event('protocol_frozen',protocol_seal_sha256=ref['sha256'],bound_file_count=len(record['bound_files']))
    print(c.canonical_bytes({'status':'SEALED',**ref,'files':len(record['bound_files'])}).decode(),end='')
if __name__=='__main__':seal()
