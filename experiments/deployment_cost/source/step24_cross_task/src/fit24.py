"""Full fixed-budget development fitting; no test observation access."""
from __future__ import annotations
import sys
import time
import resource
from pathlib import Path
import numpy as np
from common24 import *
import data24 as D
import geometry24 as G
import models24 as M
import pipeline24 as P
sys.path.insert(0,str(PARENT/'protocol'))
from policy23 import select_candidates


def _basis_key(spec):return f"{spec['scaling']}_m{spec['landmark_count']}_g{spec['gamma_multiplier']:g}"


def _cache_receipt(path,context,upstream=None):
    path=Path(path)
    write_json(path.with_name(path.name+'.receipt.json'),{
        'context_sha256':context,'sha256':sha(path),'upstream':upstream or {}})


def _verify_cache(path,context,upstream=None):
    path=Path(path);receipt=read_json(path.with_name(path.name+'.receipt.json'))
    assert receipt['context_sha256']==context and receipt['sha256']==sha(path),'Stale or changed cache'
    assert receipt['upstream']==(upstream or {}),'Changed upstream cache context'


class DevelopmentFeatures:
    def __init__(self,folder,data,banks,context):
        self.folder=folder;self.data=data;self.banks=banks
        self.context=context;self.bank_digest=sha(folder/'geometry_banks.pkl')
        self.score_train={};self.score_val={};self.nystrom_bases={}
        self.current_key=None;self.current_features=None
        self.nystrom_current_key=None;self.nystrom_current_features=None
        cache=folder/'geometry_scores.npz'
        if cache.exists():
            _verify_cache(cache,context,{'banks':self.bank_digest})
            with np.load(cache,allow_pickle=False) as f:
                for key in banks:
                    self.score_train[key]=f[key+'_train'].copy()
                    self.score_val[key]=f[key+'_validation'].copy()
        else:
            rows={}
            for key,bank in banks.items():
                self.score_train[key]=G.score_bank(bank,data['train_X'])
                self.score_val[key]=G.score_bank(bank,data['validation_X'])
                rows[key+'_train']=self.score_train[key]
                rows[key+'_validation']=self.score_val[key]
            np.savez_compressed(cache,**rows)
            _cache_receipt(cache,context,{'banks':self.bank_digest})
        self.score_digest=sha(cache)
        self.diagnostics={}

    def pair(self,spec):
        group=spec['group']
        if group=='RAW':return self.data['train_X'],self.data['validation_X']
        if group=='NYSTROM':
            key=_basis_key(spec)
            if key!=self.nystrom_current_key:
                base=self.basis(spec)
                self.nystrom_current_features=(M.nystrom_features(base,self.data['train_X']),
                                               M.nystrom_features(base,self.data['validation_X']))
                self.nystrom_current_key=key
            return self.nystrom_current_features
        bank_key=G.bank_key(spec['scaling'],spec['cap'])
        key=(bank_key,spec['prefix'],spec['view'],spec.get('omega'))
        if key!=self.current_key:
            transform=G.view_payload(spec['view'],P.block_sizes(spec['prefix']),spec.get('omega'),1e-6)
            st=self.score_train[bank_key][:,:spec['prefix']]
            sv=self.score_val[bank_key][:,:spec['prefix']]
            self.current_features=(G.transform_scores(st,transform),G.transform_scores(sv,transform))
            self.current_key=key
            self.diagnostics[str(key)]={'train':G.representation_diagnostics(st,transform,self.current_features[0]),
                                       'validation':G.representation_diagnostics(sv,transform,self.current_features[1])}
        return self.current_features

    def basis(self,spec):
        key=_basis_key(spec)
        if key not in self.nystrom_bases:
            p=self.folder/'nystrom_training'/f'{key}.pkl'
            if p.exists():
                _verify_cache(p,self.context);base=read_pickle(p)
                _verify_cache(p.with_suffix('.json'),self.context)
                _verify_cache(p.with_suffix('.npz'),self.context)
            else:
                start=time.perf_counter()
                base,evidence=M.fit_nystrom_base(self.data['support'],spec['scaling'],spec['landmark_count'],spec['gamma_multiplier'])
                write_pickle(p,base)
                write_json(p.with_suffix('.json'),evidence['summary'])
                np.savez_compressed(p.with_suffix('.npz'),**evidence['arrays'])
                for q in (p,p.with_suffix('.json'),p.with_suffix('.npz')):_cache_receipt(q,self.context)
            self.nystrom_bases[key]=base
        return self.nystrom_bases[key]

    def upstream(self,spec):
        if spec['group']=='NYSTROM':
            self.basis(spec)
            return {'basis':sha(self.folder/'nystrom_training'/(_basis_key(spec)+'.pkl'))}
        if spec['group'] in ('P','CS','MATCHED_DIAGNOSTIC'):
            return {'banks':self.bank_digest,'scores':self.score_digest}
        return {}


def _fit_one(spec,y,target_map,cache,kind):
    train,val=cache.pair(spec)
    head,pred,diagnostics=M.fit_head(spec,train,y,val,target_map,kind)
    if spec['group']=='NYSTROM':
        before=M.predict_head(head,val,kind,clip=False)
        compact=M.collapse_nystrom(cache.basis(spec),head)
        raw=M.predict_nystrom(compact,cache.data['validation_X'],kind,clip=False)
        compact_pred=M.predict_nystrom(compact,cache.data['validation_X'],kind)
        diagnostics['nystrom_unclipped_export']=M.audit_export(before,raw,target_map,'collapsed Nystrom unbounded')
        diagnostics['nystrom_clipped_export']=M.audit_export(pred,compact_pred,target_map,'collapsed Nystrom output')
        head=compact;pred=compact_pred
    return head,pred,diagnostics


def _candidate(spec,folder,y,yval,target_map,cache,kind):
    cid=spec['id'];record_path=folder/'candidate_records'/f'{cid}.json'
    head_path=folder/'heads'/f'{cid}.pkl';pred_path=folder/'validation_predictions'/f'{cid}.npy'
    upstream=cache.upstream(spec)
    if record_path.exists():
        _verify_cache(record_path,cache.context,upstream)
        record=read_json(record_path)
        if record['source_spec']!=spec:raise RuntimeError('Checkpoint candidate differs from frozen template')
        assert record['context_sha256']==cache.context and record['upstream']==upstream
        assert head_path.is_file() and pred_path.is_file()
        assert sha(head_path)==record['head_sha256'] and sha(pred_path)==record['prediction_sha256']
        return record
    start=time.perf_counter()
    head,pred,diagnostics=_fit_one(spec,y,target_map,cache,kind)
    pipe=P.make_pipeline(spec,head,cache.banks,kind)
    cost=P.payload_cost(pipe)
    record={**spec,'label':P.spec_label(spec),'source_spec':spec,
            'context_sha256':cache.context,'upstream':upstream,
            'validation_mse':float(np.mean((yval-pred)**2)),
            'complete_numeric_bytes':cost['complete_numeric_bytes'],
            'serialized_model_bytes':cost['serialized_model_bytes'],
            'fit_export_inventory_seconds':time.perf_counter()-start,'fit_diagnostics':diagnostics}
    if not np.isfinite(record['validation_mse']):raise ArithmeticError('Nonfinite validation loss')
    write_pickle(head_path,head);pred_path.parent.mkdir(parents=True,exist_ok=True);np.save(pred_path,pred,allow_pickle=False)
    record.update(head_sha256=sha(head_path),prediction_sha256=sha(pred_path))
    write_json(record_path,record)
    _cache_receipt(record_path,cache.context,upstream)
    return record


def fit_task(task):
    implementation=check_implementation_seal();kind=task_kind(task)
    folder=RUN/task;folder.mkdir(parents=True,exist_ok=True)
    completion=folder/'fit_complete.json'
    if completion.exists():
        done=read_json(completion)
        if done['implementation_seal_sha256']!=sha(RUN/'implementation_seal.json'):raise RuntimeError('Completion source mismatch')
        for rel,digest in done['sealed_outputs'].items():assert sha(folder/rel)==digest,(task,rel)
        log(task,'fit_already_complete',primary_fits=done['primary_fits'],secondary_fits=done['secondary_fits']);return
    started=time.perf_counter();log(task,'development_start')
    stored=folder/'development.npz'
    if stored.exists():
        context_record=read_json(folder/'context.json')
        assert context_record['implementation_seal_sha256']==sha(RUN/'implementation_seal.json')
        assert context_record['task']==task
        assert context_record['development_npz_sha256']==sha(stored)
        assert context_record['data_access_sha256']==sha(folder/'data_access_development.json')
        data={k:v for k,v in np.load(stored,allow_pickle=False).items()}
        meta=read_json(folder/'data_access_development.json')
        assert meta['implementation_seal_sha256']==sha(RUN/'implementation_seal.json')
        data.update(horizons=meta['returned_horizons'],primary_column=meta['returned_primary_column'],metadata=meta['loader_metadata'])
    else:
        data=D.load_development(task)
        meta={'loader_metadata':data['metadata'],'returned_horizons':data['horizons'],
              'returned_primary_column':data['primary_column'],'implementation_seal_sha256':sha(RUN/'implementation_seal.json')}
        write_json(folder/'data_access_development.json',meta)
        np.savez_compressed(stored,**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
    trainX=data['train_X'];valX=data['validation_X'];horizons=list(map(int,data['horizons']));primary_col=int(data['primary_column'])
    y=data['train_Y'][:,primary_col];yv=data['validation_Y'][:,primary_col]
    maps={str(h):M.fit_target_map(data['train_Y'][:,j]) for j,h in enumerate(horizons)}
    target_map=maps[str(horizons[primary_col])]
    context_record={'task':task,'implementation_seal_sha256':sha(RUN/'implementation_seal.json'),
                    'development_npz_sha256':sha(stored),'data_access_sha256':sha(folder/'data_access_development.json'),
                    'horizons':horizons,'primary_column':primary_col,'target_maps':jsonable(maps),
                    'array_sha256':{k:array_sha(v) for k,v in data.items() if isinstance(v,np.ndarray)}}
    if (folder/'context.json').exists():assert read_json(folder/'context.json')==context_record
    else:write_json(folder/'context.json',context_record)
    context=sha(folder/'context.json')
    log(task,'development_loaded',support_rows=len(data['support']),train_rows=len(trainX),validation_rows=len(valX),features=trainX.shape[1])
    bank_path=folder/'geometry_banks.pkl'
    if bank_path.exists():
        _verify_cache(bank_path,context);_verify_cache(folder/'geometry_fit_evidence.json',context)
        banks=read_pickle(bank_path)
    else:
        seed=next(x['geometry_seed'] for x in protocol()['benchmark_contracts']['task_ids'] if x['id']==task)
        banks,evidence=G.fit_banks(data['support'],seed)
        write_pickle(bank_path,banks);write_json(folder/'geometry_fit_evidence.json',evidence)
        _cache_receipt(bank_path,context);_cache_receipt(folder/'geometry_fit_evidence.json',context)
    cache=DevelopmentFeatures(folder,data,banks,context)
    records=[];roster=templates();main=[s for s in roster if s['group']!='MATCHED_DIAGNOSTIC']
    for j,spec in enumerate(main,1):
        records.append(_candidate(spec,folder,y,yv,target_map,cache,kind))
        if j%24==0 or j==len(main):log(task,'primary_fit_progress',completed=j,total=654,elapsed_seconds=round(time.perf_counter()-started,2))
    p_winner=min((r for r in records if r['group']=='P'),key=lambda r:(r['validation_mse'],r['index']))
    binding={k:p_winner[k] for k in ('scaling','cap','prefix','omega')}
    for spec0 in roster:
        if spec0['group']!='MATCHED_DIAGNOSTIC':continue
        spec={**spec0,**binding}
        records.append(_candidate(spec,folder,y,yv,target_map,cache,kind))
    assert len(records)==654 and sum(r['family']=='hgb' for r in records)==300
    names=protocol()['comparators']['unfitted_'+kind]
    for index,name in enumerate(names,654):
        spec={'id':'b_'+name,'index':index,'group':'UNFITTED','family':'unfitted','view':'BASELINE','baseline_name':name}
        pipe=P.baseline_pipeline(name,kind,horizons[primary_col],target_map['offset'])
        pred=P.predict(pipe,valX);cost=P.payload_cost(pipe)
        record={**spec,'label':name,'source_spec':spec,'validation_mse':float(np.mean((yv-pred)**2)),
                'complete_numeric_bytes':cost['complete_numeric_bytes'],'serialized_model_bytes':cost['serialized_model_bytes']}
        records.append(record)
        pp=folder/'validation_predictions'/f"{spec['id']}.npy"
        np.save(pp,pred,allow_pickle=False)
    selectors=select_candidates(records)
    selection=P.choose_roster(records,selectors,binding)
    by_id={r['id']:r for r in records};primary=horizons[primary_col]
    models={str(h):{} for h in horizons};selected_meta={str(h):{} for h in horizons}
    for cid in selection['selected_ids']:
        spec=by_id[cid]
        if spec['group']=='UNFITTED':pipe=P.baseline_pipeline(spec['baseline_name'],kind,primary,target_map['offset'])
        else:pipe=P.make_pipeline(spec,read_pickle(folder/'heads'/f'{cid}.pkl'),banks,kind)
        models[str(primary)][cid]=pipe
        predicted=P.predict(pipe,valX)
        cached=np.load(folder/'validation_predictions'/f'{cid}.npy',allow_pickle=False)
        audit=M.audit_export(cached,predicted,target_map,'selected full-pipeline validation replay')
        # A discontinuous head changing its selected validation loss is not silently accepted.
        if not np.array_equal(cached,predicted):
            raise ArithmeticError('Selected full-pipeline validation differs from selection forecasts')
        selected_meta[str(primary)][cid]={'source_spec':spec['source_spec'],'cost':P.payload_cost(pipe),'validation_replay':audit}
    secondary_count=0;secondary_predictions={}
    for col,h in enumerate(horizons):
        if h==primary:continue
        tm=maps[str(h)];yy=data['train_Y'][:,col]
        for cid in selection['selected_ids']:
            record=by_id[cid];spec=record['source_spec']
            p=folder/'secondary_heads'/f'h{h}_{cid}.pkl'
            if record['group']=='UNFITTED':
                pipe=P.baseline_pipeline(record['baseline_name'],kind,h,tm['offset'])
                vp=P.predict(pipe,valX);diag={'unfitted':True}
            else:
                if p.exists():
                    _verify_cache(p,context,cache.upstream(spec))
                    saved=read_pickle(p);head=saved['head'];vp=saved['validation_prediction'];diag=saved['diagnostics']
                    assert saved['horizon']==h and saved['source_spec']==spec and saved['target_map']==jsonable(tm)
                else:
                    head,vp,diag=_fit_one(spec,yy,tm,cache,kind)
                    write_pickle(p,{'head':head,'validation_prediction':vp,'diagnostics':diag,
                                    'horizon':h,'source_spec':spec,'target_map':jsonable(tm)})
                    _cache_receipt(p,context,cache.upstream(spec))
                pipe=P.make_pipeline(spec,head,banks,kind);secondary_count+=1
            models[str(h)][cid]=pipe;secondary_predictions[f'h{h}_{cid}']=vp
            selected_meta[str(h)][cid]={'source_spec':spec,'cost':P.payload_cost(pipe),'fit_diagnostics':diag,
                                        'validation_outcomes_used_for_selection':False}
    assert secondary_count<=42
    assert sha(stored)==context_record['development_npz_sha256']
    assert sha(folder/'data_access_development.json')==context_record['data_access_sha256']
    for receipt_path in folder.rglob('*.receipt.json'):
        target=receipt_path.with_name(receipt_path.name[:-len('.receipt.json')])
        receipt=read_json(receipt_path)
        _verify_cache(target,context,receipt['upstream'])
    selection.update(primary_horizon=primary,horizons=horizons,selectors={k:(None if v is None else (v['id'] if isinstance(v,dict) and 'id' in v else v)) for k,v in selectors.items()},
                     source_protocol_sha256=implementation['protocol_sha256'],secondary_fits=secondary_count,
                     primary_fits=654,all_test_data_unopened=True)
    # Consolidated evidence and selected portable models, with no cross-model cost amortization.
    write_json(folder/'records.json',records,replace=True)
    write_json(folder/'selected.json',selection,replace=True)
    write_json(folder/'selected_model_metadata.json',selected_meta,replace=True)
    write_json(folder/'target_maps.json',maps,replace=True)
    write_json(folder/'representation_diagnostics_development.json',cache.diagnostics,replace=True)
    write_pickle(folder/'selected_models.pkl',models)
    valpred={r['id']:np.load(folder/'validation_predictions'/f"{r['id']}.npy",allow_pickle=False) for r in records}
    np.savez_compressed(folder/'validation_predictions_all.npz',**valpred)
    np.savez_compressed(folder/'secondary_validation_predictions.npz',**secondary_predictions)
    done={'passed':True,'task':task,'completed_utc':now(),'implementation_seal_sha256':sha(RUN/'implementation_seal.json'),
          'primary_fits':654,'secondary_fits':secondary_count,'selected_model_count_per_horizon':len(selection['selected_ids']),
          'development_seconds':time.perf_counter()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'benchmark_test_access':False,'ECG_or_reserve_access':False}
    outputs=['context.json','selected.json','records.json','selected_model_metadata.json','selected_models.pkl',
             'validation_predictions_all.npz','secondary_validation_predictions.npz','target_maps.json',
             'data_access_development.json','development.npz','geometry_banks.pkl','geometry_fit_evidence.json',
             'geometry_scores.npz','representation_diagnostics_development.json']
    done['sealed_outputs']={rel:sha(folder/rel) for rel in outputs}
    write_json(completion,done);log(task,'development_complete',**{k:v for k,v in done.items() if k not in ('task','passed')})


def freeze_all():
    implementation=check_implementation_seal()
    if (RUN/'fit_seal.json').exists():check_fit_seal();return
    tasks={};primary_count=0;secondary_count=0
    for task in TASKS:
        folder=RUN/task;done=read_json(folder/'fit_complete.json')
        if not done['passed'] or done['benchmark_test_access']:raise RuntimeError('Invalid fit completion')
        if done['implementation_seal_sha256']!=sha(RUN/'implementation_seal.json'):raise RuntimeError('Mixed implementation fits')
        for rel,digest in done['sealed_outputs'].items():assert sha(folder/rel)==digest,(task,rel)
        selected=read_json(folder/'selected.json');records=read_json(folder/'records.json')
        assert selected['primary_fits']==654 and len([r for r in records if r['group']!='UNFITTED'])==654
        files=['fit_complete.json','context.json','selected.json','records.json','selected_model_metadata.json','selected_models.pkl',
               'validation_predictions_all.npz','secondary_validation_predictions.npz','target_maps.json','data_access_development.json',
               'development.npz','geometry_banks.pkl','geometry_fit_evidence.json','geometry_scores.npz','representation_diagnostics_development.json']
        tasks[task]={'files':{rel:sha(folder/rel) for rel in files},'primary_fits':654,'secondary_fits':done['secondary_fits']}
        primary_count+=654;secondary_count+=done['secondary_fits']
    assert primary_count==3924 and secondary_count<=252
    seal={'schema':'step24-fit-seal-v1','passed':True,'sealed_utc':now(),
          'protocol_sha256':implementation['protocol_sha256'],'source_sha256':implementation['source_sha256'],
          'archive_sha256':implementation['archive_sha256'],'all_task_ids':TASKS,
          'implementation_seal_sha256':sha(RUN/'implementation_seal.json'),'tasks':tasks,
          'primary_fits':primary_count,'secondary_fits':secondary_count,
          'all_test_data_unopened':True,'ECG_or_reserve_access':False}
    write_json(RUN/'fit_seal.json',seal);log('all','fit_sealed',primary_fits=primary_count,secondary_fits=secondary_count)
