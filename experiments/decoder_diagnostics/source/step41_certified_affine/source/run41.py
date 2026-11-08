"""Single-pass chronological affine deployment experiment."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
import argparse,pickle,time,signal,traceback,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import common41 as c
import data41 as d
import models41 as m
import affine41 as a
import representations35 as rep
import certificate41 as cert
ROOT=c.ROOT;OUT=ROOT/'output'
def progress(event,**fields): print(c.canonical_bytes({'event':event,**fields}).decode(),end='',flush=True)
def base(task,stage):return OUT/task/stage
def timeout(*args):raise TimeoutError('Frozen phase budget exhausted; no rerun')
def start(phase,p,task=None,stage=None):
    dest=OUT/(task or 'global')/(stage or phase)
    c.save(dest/(phase+'_started41.json'),{'phase':phase,'task':task,'stage':stage,'utc':c.utc(),'protocol_seal_sha256':c.sha(ROOT/'protocol_seal41.json')})
    d.event('phase_started',phase=phase,task=task,stage=stage)
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,p['execution']['phase_budget_seconds'][phase]);return time.perf_counter()
def end(phase,t,task=None,stage=None,**fields):
    signal.setitimer(signal.ITIMER_REAL,0)
    rec={'phase':phase,'task':task,'stage':stage,'status':'COMPLETE','wall_seconds':time.perf_counter()-t,'utc':c.utc(),**fields}
    ref=c.save(OUT/(task or 'global')/(stage or phase)/(phase+'_completed41.json'),rec)
    d.event('phase_completed',phase=phase,task=task,stage=stage,receipt_path=ref['path'],receipt_sha256=ref['sha256']);progress('COMPLETE',**rec)
def seal(path,schema,files,**fields):
    rec={'schema':schema,'sealed_utc':c.utc(),'protocol_seal_sha256':c.sha(ROOT/'protocol_seal41.json'),'bound_files':{c.relative(x):c.sha(x) for x in sorted(set(files))},**fields}
    ref=c.save(path,rec);d.event('evidence_sealed',path=ref['path'],sha256=ref['sha256'],schema=schema,task=fields.get('task'),stage=fields.get('stage'));return rec
def feature_save(path,z):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.save(f,np.asarray(z,dtype=np.float64),allow_pickle=False)
    return {'path':c.relative(path),'sha256':c.sha(path)}
def fit(p,task,stage):
    t=start('fit',p,task,stage);cohort=d.load_training(task,stage);dest=base(task,stage)
    X=cohort['X'];y=list(map(F,cohort['y']));normal=cohort['normalization']
    maps,metadata=rep.fit_representations(X,p['representations'],p['seed']);c.save(dest/'representation_metadata41.json',metadata)
    details={};rows=[];timings=[]
    for i,name in enumerate(p['maps']):
        t0=time.perf_counter();r=maps[name];raw=rep.serialize_map(r);mapref=c.save_bytes(dest/'maps'/(name+'.pkl'),raw)
        z=rep.transform_one(r,X);fref=feature_save(dest/'features/train'/(name+'.npy'),z)
        feature_id='feature41:'+c.digest({'task':task,'stage':stage,'map_sha256':mapref['sha256'],'train_feature_sha256':fref['sha256']})
        binding={'task':task,'stage':stage,'cohort_path':cohort['path'],'cohort_sha256':cohort['sha256'],'feature_path':fref['path'],'feature_sha256':fref['sha256'],'map_sha256':mapref['sha256']}
        fitted=a.fit_affine(z,y,feature_id=feature_id,kappa='2',ridge_lambda=p['heads']['affine_ridge_lambda'],training_binding=binding)
        fitref=c.save(dest/'fits'/name/'affine.json',fitted);head=fitted['head'];href=c.save(dest/'heads'/name/'affine.json',head)
        cid=name+'::affine';deployment=cert.make_deployment(head,target_min=normal['target_min'],span=normal['span'],map_sha256=mapref['sha256'],map_bytes=len(raw),candidate_id=cid)
        dpref=c.save(dest/'deployments'/(name+'.json'),deployment)
        prepared=cert.prepare_deployment(deployment,expected_map_sha256=mapref['sha256'])
        di={'name':name,'family':r.family,'scaling':r.scaling,'dimension':int(z.shape[1]),'feature_id':feature_id,'map_path':mapref['path'],'map_sha256':mapref['sha256'],'map_payload_bytes':len(raw),'training_features':fref}
        details[name]=di
        rows.append({'candidate_id':cid,'map':name,'token':'affine','recipe':'affine','family':r.family,'scaling':r.scaling,'dimension':int(z.shape[1]),'feature_id':feature_id,
          'head_path':href['path'],'head_sha256':href['sha256'],'fit_path':fitref['path'],'fit_sha256':fitref['sha256'],'deployment_path':dpref['path'],'deployment_sha256':dpref['sha256'],'deployment_id':deployment['deployment_id'],
          'head_payload_bytes':(ROOT/href['path']).stat().st_size,'map_payload_bytes':len(raw),'static_deployment_bytes':(ROOT/dpref['path']).stat().st_size,'deployment_payload_bytes':len(raw)+(ROOT/dpref['path']).stat().st_size})
        timings.append({'candidate_id':cid,'seconds':time.perf_counter()-t0});progress('FIT_AFFINE',task=task,stage=stage,index=i+1,total=18)
    baseline_rows=[]
    for i,cfg in enumerate(p['baselines']['configurations']):
        t0=time.perf_counter();model=m.fit_baseline(cfg,X,y,normal)
        pref=c.save_bytes(dest/'baselines'/(cfg['token']+'.pkl'),pickle.dumps(model,protocol=5))
        baseline_rows.append({**cfg,'recipe':'baseline','baseline_kind':cfg['kind'],'family':'ordinary','payload_path':pref['path'],'payload_sha256':pref['sha256'],'deployment_payload_bytes':pref['bytes']})
        timings.append({'candidate_id':cfg['candidate_id'],'seconds':time.perf_counter()-t0})
        if (i+1)%8==0:progress('FIT_BASELINE',task=task,stage=stage,index=i+1,total=44)
    assert [r['candidate_id'] for r in rows+baseline_rows]==p['evaluation']['prediction_ids']
    manifest={'schema':'step41-fit-manifest-v1','task':task,'stage':stage,'training_cohort':{'path':cohort['path'],'sha256':cohort['sha256']},'normalization':normal,'training_n':len(y),'maps':details,'candidate_rows':rows,'baseline_rows':baseline_rows,'group_timings':timings,'representation_seconds':metadata['factory_total_seconds'],'selection_seal_sha256':c.sha(OUT/task/'selection_seal41.json') if stage=='final' else None}
    c.save(dest/'fit_manifest41.json',manifest);end('fit',t,task,stage,training_n=len(y),affine_fits=18,baselines=44)
    seal(dest/'fit_seal41.json','step41-fit-seal-v1',[x for x in dest.rglob('*') if x.is_file()],task=task,stage=stage,manifest_sha256=c.sha(dest/'fit_manifest41.json'))
def load_models(task,stage):
    manifest=c.read(base(task,stage)/'fit_manifest41.json');maps={};prepared={};baselines={}
    for name,row in manifest['maps'].items():
        if c.sha(ROOT/row['map_path'])!=row['map_sha256']:raise ValueError('Map mutated')
        maps[name]=rep.deserialize_map((ROOT/row['map_path']).read_bytes())
    for row in manifest['candidate_rows']:
        if c.sha(ROOT/row['deployment_path'])!=row['deployment_sha256']:raise ValueError('Deployment mutated')
        prepared[row['candidate_id']]=cert.prepare_deployment(c.read(ROOT/row['deployment_path']),expected_deployment_id=row['deployment_id'],expected_map_sha256=manifest['maps'][row['map']]['map_sha256'])
    for row in manifest['baseline_rows']:
        if c.sha(ROOT/row['payload_path'])!=row['payload_sha256']:raise ValueError('Baseline mutated')
        baselines[row['candidate_id']]=pickle.loads((ROOT/row['payload_path']).read_bytes())
    return manifest,maps,prepared,baselines
def evaluate(p,task,stage,context,X,ids,role,dest):
    manifest,maps,prepared,baselines=context;predictions={};map_records={}
    for name in p['maps']:
        z=rep.transform_one(maps[name],X);fref=feature_save(dest/'features'/role/(name+'.npy'),z);map_records[name]={'features':fref};cid=name+'::affine';pp=prepared[cid];pr=[]
        for qid,zz in zip(ids,z):
            value=cert.predict_float(pp,zz);witness=cert.produce_certificate(pp,zz,value,query_id=qid)
            verified=cert.verify_certificate(pp,zz,witness,expected_query_id=qid,reported_prediction=value)
            exact=cert.exact_prediction(pp,zz)
            pr.append({'query_id':qid,'reported_physical_hex':float(value).hex(),'exact_physical':str(exact),'certificate':witness})
        predictions[cid]=pr
    for row in manifest['baseline_rows']:
        values=m.predict_baseline(baselines[row['candidate_id']],X)
        predictions[row['candidate_id']]=[{'query_id':qid,'reported_physical_hex':float(v).hex(),'exact_physical':None,'certificate':None} for qid,v in zip(ids,values)]
    assert list(predictions)==p['evaluation']['prediction_ids']
    return {'schema':'step41-prediction-bundle-v1','task':task,'stage':stage,'role':role,'query_ids':ids,'predictions':predictions,'maps':map_records,'fit_seal_sha256':c.sha(base(task,stage)/'fit_seal41.json')}
def validate(p,task,stage):
    t=start('validate',p,task,stage);cohort=d.load_validation(task,stage);dest=base(task,stage);context=load_models(task,stage)
    bundle=evaluate(p,task,stage,context,cohort['X'],cohort['ids'],'validation',dest);pref=c.save(dest/'validation_predictions41.json',bundle);rows=[]
    for r in context[0]['candidate_rows']+context[0]['baseline_rows']:
        preds=bundle['predictions'][r['candidate_id']]
        rows.append({**r,'physical_risk':c.score(preds,cohort['raw_targets']),'exact_reference_risk':c.score(preds,cohort['raw_targets'],True) if r['recipe']=='affine' else None})
    c.save(dest/'validation_scores41.json',{'schema':'step41-validation-scores-v1','task':task,'stage':stage,'n':len(cohort['ids']),'cohort_sha256':cohort['sha256'],'validation_predictions_sha256':pref['sha256'],'candidate_rows':rows})
    end('validate',t,task,stage,validation_n=len(cohort['ids']))
    frozen=c.read(dest/'fit_seal41.json')['bound_files'];files=[x for x in dest.rglob('*') if x.is_file() and c.relative(x) not in frozen]
    seal(dest/'completion_seal41.json','step41-fold-completion-seal-v1',files,task=task,stage=stage,fit_seal_sha256=c.sha(dest/'fit_seal41.json'),validation_scores_sha256=c.sha(dest/'validation_scores41.json'))
def select(p,task):
    t=start('select',p,task);folds={}
    for stage in ('f1','f2','f3'):
        gate=c.read(ROOT/'audit'/task/stage/'validation_gate41.json')
        if gate['status']!='PASS' or gate['completion_seal_sha256']!=c.sha(base(task,stage)/'completion_seal41.json'):raise RuntimeError('Independent development gate mismatch')
        folds[stage]=c.read(base(task,stage)/'validation_scores41.json')
    rows=[];n=sum(f['n'] for f in folds.values())
    for cid in p['evaluation']['prediction_ids']:
        records={stage:next(r for r in f['candidate_rows'] if r['candidate_id']==cid) for stage,f in folds.items()};last=records['f3'];sse=sum((F(r['physical_risk']['sse']) for r in records.values()),F(0))
        rsse=sum((F(r['exact_reference_risk']['sse']) for r in records.values()),F(0)) if last['recipe']=='affine' else None
        row={k:last.get(k) for k in ('candidate_id','map','token','recipe','family','scaling','baseline_kind')}
        row.update({'folds':{stage:{'n':folds[stage]['n'],'physical_risk':r['physical_risk'],'exact_reference_risk':r['exact_reference_risk'],'payload_bytes':r['deployment_payload_bytes']} for stage,r in records.items()},'pooled_physical_mse':str(sse/n),'pooled_exact_reference_mse':str(rsse/n) if rsse is not None else None,'selection_payload_bytes':last['deployment_payload_bytes']});rows.append(row)
    selections={label:c.select(rows,label) for label in p['selection']['selectors']}
    rec={'schema':'step41-development-selection-v1','task':task,'candidate_rows':rows,'selections':selections,'development_n':n,'fold_score_sha256':{s:c.sha(base(task,s)/'validation_scores41.json') for s in folds},'selection_uses_development_only':True,'test_targets_accessed':0}
    ref=c.save(OUT/task/'selection41.json',rec);end('select',t,task,selected_ids={k:v['candidate_id'] for k,v in selections.items()})
    seal(OUT/task/'selection_seal41.json','step41-selection-seal-v1',[ROOT/ref['path']],task=task,selection_sha256=ref['sha256'],selected_ids={k:v['candidate_id'] for k,v in selections.items()})
def test(p,task):
    t=start('test',p,task);dest=base(task,'test');context=load_models(task,'final');selection=c.read(OUT/task/'selection41.json');queries=d.test_queries(task)
    allpred={cid:[] for cid in p['evaluation']['prediction_ids']};ys=[];queryrows=[]
    for i,q in enumerate(queries):
        history=d.load_test_history(task,q);bundle=evaluate(p,task,'final',context,history['X'],[q['id']],f'q{i:03d}',dest)
        bundle.update({'schema':'step41-sealed-test-predictions-v1','query_id':q['id'],'query_index':i,'model_seal_sha256':c.sha(base(task,'final')/'fit_seal41.json'),'history_receipt_sha256':history['history_receipt_sha256'],'selection_seal_sha256':c.sha(OUT/task/'selection_seal41.json')})
        ref=c.save(dest/'predictions'/f'q{i:03d}.json',bundle);target=d.reveal_test_target(task,q,ref['path'],ref['sha256']);ys.append(target['raw_target'])
        for cid in allpred:allpred[cid].extend(bundle['predictions'][cid])
        queryrows.append({'id':q['id'],'month':q['target_timestamp'][:7],'prediction_path':ref['path'],'prediction_sha256':ref['sha256'],'target_path':target['path'],'target_sha256':target['sha256']})
        if (i+1)%10==0:progress('TEST_QUERY',task=task,index=i+1,total=len(queries))
    rows=[]
    for r in context[0]['candidate_rows']+context[0]['baseline_rows']:
        preds=allpred[r['candidate_id']];rows.append({**r,'physical_test_risk':c.score(preds,ys),'exact_reference_test_risk':c.score(preds,ys,True) if r['recipe']=='affine' else None,'test_predictions_digest':c.digest(preds)})
    selected={label:next(r for r in rows if r['candidate_id']==s['candidate_id']) for label,s in selection['selections'].items()}
    primary=c.primary(selected['affine_rolling']['physical_test_risk'],selected['ordinary_rolling']['physical_test_risk']);monthly={}
    for month in sorted({q['month'] for q in queryrows}):
        ix=[i for i,q in enumerate(queryrows) if q['month']==month]
        monthly[month]={label:c.score([allpred[row['candidate_id']][i] for i in ix],[ys[i] for i in ix]) for label,row in selected.items()}
    c.save(OUT/task/'primary_result41.json',{'schema':'step41-primary-result-v1','task':task,'n':len(ys),'query_ids':[q['id'] for q in queryrows],'raw_targets':ys,'queries':queryrows,'candidate_rows':rows,'selections':selected,'primary_gate':primary,'monthly_results':monthly,'validity_audit_pending':True,'selection_sha256':c.sha(OUT/task/'selection41.json')})
    end('test',t,task,test_n=len(ys),primary=primary)
    files=[x for x in dest.rglob('*') if x.is_file()]+[OUT/task/'primary_result41.json']
    seal(dest/'completion_seal41.json','step41-test-completion-seal-v1',files,task=task,primary_result_sha256=c.sha(OUT/task/'primary_result41.json'))
def benchmark(p,task):
    t=start('benchmark',p,task);gate=c.read(ROOT/'audit'/task/'postrun_gate41.json')
    if gate['status']!='PASS':raise RuntimeError('Postrun gate required')
    result=c.read(OUT/task/'primary_result41.json');row=result['selections']['affine_rolling'];context=load_models(task,'final');pp=context[2][row['candidate_id']];mapping=context[1][row['map']]
    deployment=c.read(ROOT/row['deployment_path']);configuration=p['timing'];repeats=configuration['repeats_per_query_primitive'];warmups=configuration['warmups_per_query_primitive']
    deployment_bytes=c.canonical_bytes(deployment)
    static_kwargs={'expected_deployment_id':row['deployment_id'],'expected_map_sha256':context[0]['maps'][row['map']]['map_sha256']}
    for _ in range(3):cert.prepare_deployment(json.loads(deployment_bytes),**static_kwargs)
    static=[]
    for _ in range(configuration['static_prepare_repeats']):
        t0=time.perf_counter_ns();cert.prepare_deployment(json.loads(deployment_bytes),**static_kwargs);static.append(time.perf_counter_ns()-t0)
    records=[]
    for i,q in enumerate(result['queries']):
        bundle=c.read(ROOT/q['prediction_path']);pr=bundle['predictions'][row['candidate_id']][0]
        X=np.load(OUT/task/'test/history'/f'{i:03d}_X.npy',allow_pickle=False);z=np.load(ROOT/bundle['maps'][row['map']]['features']['path'],allow_pickle=False)[0]
        witness=pr['certificate'];value=float.fromhex(pr['reported_physical_hex']);wire=cert.encode_certificate(witness)
        ordinary=context[3][result['selections']['ordinary_rolling']['candidate_id']]
        def unverified_prediction():return cert.predict_float(pp,rep.transform_one(mapping,X)[0])
        def certified_prediction():
            zz=rep.transform_one(mapping,X)[0];vv=cert.predict_float(pp,zz)
            packet=cert.encode_certificate(cert.produce_certificate(pp,zz,vv,query_id=q['id']))
            return cert.verify_certificate(pp,zz,json.loads(packet),expected_query_id=q['id'],reported_prediction=vv)
        funcs={'transform':lambda:rep.transform_one(mapping,X),'head_inference':lambda:cert.predict_float(pp,z),'certificate_construction':lambda:cert.produce_certificate(pp,z,value,query_id=q['id']),'certificate_serialization':lambda:cert.encode_certificate(witness),'certificate_verification':lambda:cert.verify_certificate(pp,z,json.loads(wire),expected_query_id=q['id'],reported_prediction=value),
               'prediction_without_certificate':unverified_prediction,'certified_prediction':certified_prediction,'ordinary_prediction':lambda:m.predict_baseline(ordinary,X)}
        if cert.predict_float(pp,z).hex()!=pr['reported_physical_hex'] or not np.array_equal(rep.transform_one(mapping,X)[0].view(np.uint64),z.view(np.uint64)):raise RuntimeError('Deployment replay changed')
        samples={}
        for name,fn in funcs.items():
            for _ in range(warmups):fn()
            samples[name]=[]
            for _ in range(repeats):
                t0=time.perf_counter_ns();fn();samples[name].append(time.perf_counter_ns()-t0)
        packet=len(cert.encode_certificate(witness));featurebytes=len(cert.feature_payload_bytes(z))
        records.append({'query_id':q['id'],'prediction_sha256':q['prediction_sha256'],'nanoseconds':samples,'median_nanoseconds':{k:float(np.median(v)) for k,v in samples.items()},'certificate_json_bytes':packet,'feature_bytes':featurebytes,'certificate_plus_feature_bytes':packet+featurebytes,'absolute_physical_prediction_error':str(abs(c.number(pr['reported_physical_hex'])-F(pr['exact_physical'])))})
    summaries={}
    for name in configuration['primitives']:
        med=[r['median_nanoseconds'][name] for r in records];summaries[name]={'p50_ms':float(np.quantile(med,.5,method='linear'))/1e6,'p95_ms':float(np.quantile(med,.95,method='linear'))/1e6,'min_ms':min(med)/1e6,'max_ms':max(med)/1e6}
    sizes={key:{'min':min(r[key] for r in records),'median':float(np.median([r[key] for r in records])),'max':max(r[key] for r in records)} for key in ('certificate_json_bytes','feature_bytes','certificate_plus_feature_bytes')}
    out={'schema':'step41-deployment-benchmark-v1','task':task,'candidate_id':row['candidate_id'],'deployment_sha256':row['deployment_sha256'],'map_sha256':context[0]['maps'][row['map']]['map_sha256'],'n':len(records),'configuration':configuration,'records':records,'component_summary':summaries,'packet_summary':sizes,'static_prepare_nanoseconds':static,'static_prepare_median_ms':float(np.median(static))/1e6,'map_bytes':row['map_payload_bytes'],'static_deployment_bytes':row['static_deployment_bytes'],'deployment_payload_bytes':row['deployment_payload_bytes'],'maximum_physical_error':str(max(F(r['absolute_physical_prediction_error']) for r in records)),'physical_error_budget':str(F(context[0]['normalization']['span'])/2**40),'postrun_gate_sha256':c.sha(ROOT/'audit'/task/'postrun_gate41.json')}
    c.save(OUT/task/'benchmark41.json',out);end('benchmark',t,task,test_n=len(records))
def aggregate(p):
    t=start('aggregate',p);results={};benchmarks={};gates={}
    for task in p['tasks']:
        for label in ('postrun','benchmark'):
            path=ROOT/'audit'/task/(label+'_gate41.json');gate=c.read(path)
            if gate['status']!='PASS':raise RuntimeError('Final independent gate failed')
            gates[c.relative(path)]=c.sha(path)
        results[task]=c.read(OUT/task/'primary_result41.json');benchmarks[task]=c.read(OUT/task/'benchmark41.json')
    success=all(r['primary_gate']['within_five_percent_mse'] for r in results.values())
    c.save(OUT/'result41.json',{'schema':'step41-complete-result-v1','primary_status':'ESTABLISHED_ON_FIXED_PERIODS' if success else 'NOT_ESTABLISHED','scope':'two forward periods of known domains, no population or broadAI superiority claim','tasks':results,'benchmarks':benchmarks,'independent_gates':gates,'protocol_seal_sha256':c.sha(ROOT/'protocol_seal41.json'),'step40_primary_unchanged':'NOT_ESTABLISHED'})
    end('aggregate',t,primary_status='ESTABLISHED_ON_FIXED_PERIODS' if success else 'NOT_ESTABLISHED')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fit','validate','select','test','benchmark','aggregate']);parser.add_argument('--task');parser.add_argument('--stage');args=parser.parse_args()
    p=d.verify_protocol_seal()
    try:
        if args.phase in ('fit','validate'):globals()[args.phase](p,args.task,args.stage)
        elif args.phase=='aggregate':aggregate(p)
        else:globals()[args.phase](p,args.task)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL,0);failure={'phase':args.phase,'task':args.task,'stage':args.stage,'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'utc':c.utc()}
        path=OUT/'failures'/f'{args.phase}_{args.task}_{args.stage}.json'
        if not path.exists():c.save(path,failure)
        progress('STOP_RETAIN_FAILURE',**failure);raise SystemExit(2)
if __name__=='__main__':main()
