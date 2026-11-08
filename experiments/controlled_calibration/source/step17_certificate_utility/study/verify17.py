"""Independent saved-evidence audit of the Step 17 prospective comparison.

No Step17 producer, data, model-selection or certificate module is imported.
The validated common native feature engine is reused for explicit sentinels;
native geometry is checked against support provenance, not independently fit.
"""
import os
for _option in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'BLIS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT',
                'VECLIB_MAXIMUM_THREADS'):
    os.environ[_option]='1'
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json,pickle,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
RUN=HERE/'run'
for _directory in ('step6_geometry_regimes/src','step11_output_accuracy',
                   'step11_output_accuracy/engine'):
    sys.path.insert(0,str(ROOT/_directory))
import audit_math17 as A
near,require=A.near,A.require
REGIMES=('repairable','irreducible')
SCALES=('minmax01','standard')
TEMPERATURES=(.5,4.,16.,64.)
RIDGES=(0.,1e-4,1e-2)
SIZES=dict(support=256,train=512,validation=256,cal=4096,test=1024)
SAMPLE_SIZES=(256,1024,4096)
BLOCK_SIZES=(2,4,8,16,32)
ALPHA=.05/1008
BOUNDS=(-.85,.85)
KAPPA=2.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with Path(path).open('rb') as h:return pickle.load(h)


def write(path,record):
    with Path(path).open('x') as h:
        json.dump(record,h,indent=2,allow_nan=False,
                  default=lambda x:x.tolist() if isinstance(x,np.ndarray)
                  else x.item() if isinstance(x,np.generic) else str(x))
        h.write('\n')


def arrsha(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str(x.shape).encode()+x.dtype.str.encode()+x.tobytes()).hexdigest()


def update(maxima,key,value):
    maxima[key]=max(maxima.get(key,0.),float(value))


def timing(value):
    require(not isinstance(value,bool) and np.isscalar(value) and np.isfinite(value)
            and value>=0,'invalid elapsed time')


def identity(record,case):
    for key in ('case','regime','regime_index','repetition'):
        require(record[key]==case[key],'case identity differs: '+key)


def case_roster(rows,cases):
    require(len(rows)==len(cases)==24,'aggregate case count differs')
    for row,case in zip(rows,cases,strict=True):identity(row,case)


def independent_streams():
    result={}
    for regime_index in range(2):
        for repetition in range(12):
            i=regime_index*12+repetition
            for pool_index,pool in enumerate(('support','train','validation','cal','test','geometry')):
                result[f'r{i}_{pool}']=int(np.random.SeedSequence(
                    (2026110617,1195724617,regime_index,repetition,pool_index)
                    ).generate_state(1,dtype=np.uint64)[0])
    return result


def independent_draw(n,seed,regime=None):
    rng=np.random.default_rng(seed)
    def history():
        signs=rng.choice(np.array([-1.,1.]),size=(n,2))
        radii=.9+rng.uniform(-.1,.1,size=(n,2))
        return signs*radii
    x=history()
    if regime is None:return dict(X=x)
    driver=x.copy() if regime=='repairable' else history()
    noise=rng.uniform(-.05,.05,size=n)
    y=.8*driver[:,0]*driver[:,1]+noise
    return dict(X=x,Y=y,driver=driver,noise=noise)


def check_data(saved,n,seed,regime=None):
    expected=independent_draw(n,seed,regime)
    require(set(saved)==set(expected),'generator evidence field mismatch')
    maximum=0.
    for key,value in expected.items():
        maximum=max(maximum,near(saved[key],value,3e-16,0.))
    if regime is not None:
        require(np.min(saved['Y'])>=BOUNDS[0] and np.max(saved['Y'])<=BOUNDS[1],
                'generator target range violated')
    return maximum


def roster():
    return [dict(index=i,id=f'g{i}',scaling=scale,cap=cap,concentration=temp,
                 blocks=[2,4],maximum_prefix=6)
            for i,(scale,cap,temp) in enumerate(
                (s,c,t) for s in SCALES for c in (1,2) for t in TEMPERATURES)]


def slices(prefix):
    return [(0,2)] if prefix==2 else [(0,2),(2,6)]


def check_geometry(banks,support,metadata,seed):
    n=len(support)
    require(set(banks)=={(s,q) for s in SCALES for q in (1,2)},'geometry menu')
    require(metadata['seed']==seed and metadata['native_base_seed']==seed%2**31,'geometry seed')
    require(metadata['unlabeled_shape']==[n,2] and
            metadata['unlabeled_sha256']==arrsha(support),'support identity')
    require(metadata['forecast_labels_used'] is False,'geometry used forecast labels')
    require(metadata['fit_indices']==list(range(n)),'geometry row selection')
    require(metadata['maximal_partition_fit_count']==2 and
            metadata['native_kmeans_block_fit_count']==4 and
            metadata['native_local_ae_fit_count']==48,'geometry fitting census')
    require(len(metadata['banks'])==8,'unused regrouped fitting work omitted')
    for scale in SCALES:
        offset=support.min(0) if scale=='minmax01' else support.mean(0)
        divisor=np.ptp(support,axis=0) if scale=='minmax01' else support.std(0)
        require(np.all(divisor>0),'unexpected degenerate fixture support')
        scaled=(support-offset)/divisor
        rounded=scaled.astype(np.float32).astype(float)
        for cap in (1,2):
            bank=banks[(scale,cap)]
            require(bank.kind=='learned' and bank.cap==cap and bank.seed==seed,'bank identity')
            require(tuple(bank.block_sizes)==(2,4) and tuple(bank.requested_counts)==(2,4),'nested blocks')
            require(bank.tau==1e-6 and bank.n_features_in_==2,'bank input parameters')
            near(bank.fit_indices,np.arange(n),0.,0.)
            near(bank.scaler.offset,offset,0.,0.)
            near(bank.scaler.divisor,divisor,0.,0.)
            require(bank.scaler.mode==scale and bank.scaler.training_rows==n,'scaler provenance')
            count=0
            for bi,block in enumerate(bank.blocks):
                require(block['n_clusters']==(2,4)[bi] and len(block['classifier'])==(2,4)[bi],
                        'native block roster')
                for j,group in enumerate(block['classifier']):
                    require(len(group)==1,'native local encoder multiplicity')
                    encoder=group[0]
                    refs=np.asarray(encoder['y_data']).T
                    require(refs.shape[1]==2 and 2<=len(refs)<=24,'reference dimension/count')
                    near(refs,banks[(scale,3-cap)].blocks[bi]['classifier'][j][0]['y_data'].T,0.,0.)
                    direct=np.any(np.all(refs[:,None,:]==rounded[None,:,:],axis=2),axis=1)
                    if not direct.all():
                        distances=((scaled[:,None]-scaled[None,:])**2).sum(2)
                        np.fill_diagonal(distances,np.inf)
                        auxiliary=(.95*scaled+.05*scaled[np.argmin(distances,axis=1)]).astype(np.float32).astype(float)
                        possible=np.any(np.all(refs[:,None,:]==auxiliary[None,:,:],axis=2),axis=1)
                        require(np.all(direct|possible),'reference has no support origin')
                    rank=np.shape(encoder['PC'])[1]
                    require(1<=rank<=min(cap,2,len(refs)-1),'native local rank')
                    audit=encoder['dimension_audit']
                    require(audit['effective_rank']==rank and audit['requested_cap']==cap and
                            audit['ambient_dimension']==2 and audit['local_reference_count']==len(refs),
                            'native dimension audit')
                    require(bank.audit['local_models'][count]['reference_sha256']==arrsha(encoder['y_data']),
                            'reference hash changed')
                    count+=1
            require(count==6,'local model count')
    for path,digest in metadata['source_sha256'].items():
        require(sha(ROOT/path)==digest,'native source identity')


def simplex(F):
    F=np.asarray(F)
    require(F.ndim==2 and F.shape[1]==6 and np.isfinite(F).all(),'soft-state dimensions')
    require(F.min()>=0. and F.max()<=1.+3e-13,'soft-state range')
    near(F[:,:2].sum(1),np.ones(len(F)),3e-13,0.)
    near(F[:,2:6].sum(1),np.ones(len(F)),3e-13,0.)


def feature_replay(bank,X,temperature,F,rows):
    from certified import certified_bank
    from soft_state import power_state
    ix=np.asarray(rows,dtype=int)
    result=certified_bank(bank,np.ascontiguousarray(X[ix]),prefix=6,
                         precision=(80,128,192),score_atol=1e-10,F_atol=1e-9)
    require(np.asarray(result['accepted']).all(),'native feature replay rejected')
    p,lo,hi=power_state(result['score_lo'],result['score_hi'],(2,4),temperature,tau=1e-6)
    require(np.max(np.maximum(p-lo,hi-p))<=1e-9,'soft-state enclosure width')
    return near(F[ix],p,3e-10,2e-11)


def check_head(model,F,Y,prefix,ridge):
    require(model['alpha']==ridge,'ridge penalty changed')
    require(model['prefix']==prefix and model['kappa']==KAPPA and tuple(model['bounds'])==BOUNDS
            and [tuple(b) for b in model['blocks']]==slices(prefix),'head task/metric')
    require(model['tolerance']==1e-7 and model['max_iter']==20000 and model['inward_margin']==1e-12,
            'head optimization budget')
    d=A.head_values(F[:,:prefix],Y,model,slices(prefix),ridge)
    raw=np.asarray(model['coef_before_guard'])
    raw_norm=float(sum(np.sqrt(np.sum(np.asarray(raw[lo:hi],np.longdouble)**2)) for lo,hi in slices(prefix)))
    factor=1. if raw_norm==0 else min(1.,(KAPPA-1e-12)/raw_norm)
    near(model['raw_norm_sum'],raw_norm,2e-13,2e-13)
    near(model['inward_factor'],factor,2e-14,2e-14)
    near(model['coef'],np.asarray(model['coef_before_guard'])*model['inward_factor'],3e-14,3e-14)
    near(model['intercept'],d['intercept_expected'],3e-13,3e-13)
    near(model['x_mean'],F[:,:prefix].mean(0),3e-14,3e-14)
    near(model['y_mean'],float(np.mean(Y)),3e-14,3e-14)
    near(model['norm_sum'],d['norm_sum'],3e-12,3e-12)
    near(model['objective'],d['objective'],3e-11,3e-11)
    near(model['gap'],d['gap'],3e-10,3e-10)
    require(type(model['converged']) is bool,'convergence flag must be literal Boolean')
    require(type(model['iterations']) is int and 0<=model['iterations']<=model['max_iter'],
            'invalid iteration count')
    require(model['converged']==(d['gap']<=model['tolerance']*max(1.,abs(d['objective']))),
            'convergence status differs from objective-gap rule')
    return d


def prediction(F,model,prefix):
    return np.clip(F[:,:prefix]@np.asarray(model['coef'])+float(model['intercept']),*BOUNDS)


def audit_certificate(F,Y,pred,record):
    require(record['sample_sizes']==list(SAMPLE_SIZES) and
            record['block_sizes']==list(BLOCK_SIZES),'certificate menu')
    require(record['alpha']==ALPHA and record['kappa']==KAPPA and
            record['target_bounds']==list(BOUNDS),'certificate estimand')
    require(record['solver_config']==dict(max_iter=4000,tolerance=1e-7,check_every=20),
            'certificate optimization budget')
    feature_slices=[tuple(s) for s in record['block_slices']]
    require(feature_slices==slices(F.shape[1]),'certificate feature metric')
    duals={}
    counts=Counter();maxima={}
    for b in BLOCK_SIZES:
        K=len(Y)//b
        evidence=record['block_evidence'][str(b)]
        z=A.audit_duals(F[:K*b].reshape(K,b,-1),Y[:K*b].reshape(K,b),
                       feature_slices,KAPPA,evidence)
        duals[b]=z['lower_sse']
        counts['block_programs']+=K
        counts['programs_at_iteration_limit']+=int((~np.asarray(evidence['converged'])).sum())
        for key in ('max_replay_error','max_gap','max_feasibility_residual'):
            update(maxima,key,z[key])
        near(evidence['gap'],(z['upper_sse']-z['lower_sse'])/(b-1),3e-10,3e-10)
        require(np.asarray(evidence['converged']).dtype==np.bool_ and
                np.asarray(evidence['converged']).shape==(K,),'block convergence shape/dtype')
        iterations=np.asarray(evidence['iterations'])
        require(iterations.shape==(K,) and np.issubdtype(iterations.dtype,np.integer)
                and np.all(iterations>=0) and np.all(iterations<=record['solver_config']['max_iter']),
                'block iteration limit')
        timing(record['block_solve_seconds'][str(b)])
    reconstructed=A.score_endpoints(Y,pred,duals,SAMPLE_SIZES,ALPHA,BOUNDS)
    require([v['n'] for v in record['records']]==list(SAMPLE_SIZES),'calibration prefix roster')
    for item in record['records']:
        n=item['n'];want=reconstructed[n]
        require(item['confidence_events']==dict(fitted_upper=1,block_lower=5,recycled_lower=1),
                'wrong confidence allocation census')
        require(item['same_solved_dual_values'] is True,'different solver paths used for baseline')
        near(item['upper']['mean'],want['mean_loss'],2e-12,2e-12)
        near(item['upper']['upper'],want['upper'],2e-11,2e-11)
        require(item['upper']['n']==n and item['upper']['alpha']==ALPHA and
                item['upper']['bound']==1.7**2,'upper endpoint range/count')
        near(item['upper']['kl_radius'],np.log(1/ALPHA)/n,2e-15,2e-15)
        for block,pooled,b in zip(item['block'],item['recycled_per_block'],BLOCK_SIZES,strict=True):
            K=n//b
            sse_sum=float(np.sum(duals[b][:K]))
            near(block['lower'],want['blocks'][b],2e-11,2e-11)
            near(block['mean'],sse_sum/(K*(b-1)),2e-11,2e-11)
            near(block['sse_sum'],sse_sum,2e-9,2e-11)
            near(block['kl_radius'],np.log(1/ALPHA)/K,2e-15,2e-15)
            near(pooled['mean'],want['pooled'][b]['empirical_lower'],2e-11,2e-11)
            near(pooled['lower'],want['pooled'][b]['lower'],2e-11,2e-11)
            near(pooled['sse_sum'],sse_sum,2e-9,2e-11)
            near(pooled['kl_radius'],np.log(1/ALPHA)/n,2e-15,2e-15)
            require(block['n']==n//b and block['block_size']==b and block['alpha']==ALPHA,
                    'block independent sample count/allocation')
            require(block['n_blocks']==K and block['normalization']=='SSE/(b-1)',
                    'block mean metadata')
            near(block['bound'],1.7**2*(b*b//4)/(b*(b-1)),0.,0.)
            require(pooled['n']==n and pooled['alpha']==ALPHA and pooled['bound']==1.7**2,
                    'pooled independent sample count/allocation')
            require(pooled['covered']==b*(n//b) and pooled['uncovered']==n%b,
                    'pooled sample coverage')
            require(pooled['n_blocks']==K and pooled['block_size']==b and
                    pooled['normalization']=='sum block dual SSE/n','pooled mean metadata')
        near(item['hierarchy_lower'],want['block_best'],2e-11,2e-11)
        # Conservative numerical bisection may return zero where an independent
        # root is tiny positive; replay ties on individually checked stored roots.
        stored_block_winner=max(item['block'],key=lambda r:r['lower'])
        require(item['hierarchy_selected_block_size']==stored_block_winner['block_size'],'wrong block winner')
        near(item['hierarchy_lower'],stored_block_winner['lower'],0.,0.)
        near(item['recycled']['lower'],want['pool_best'],2e-11,2e-11)
        stored_pooled_winner=max(item['recycled_per_block'],key=lambda r:r['mean'])
        require(item['recycled']['selected_block_size']==stored_pooled_winner['block_size'],'wrong pooled winner')
        require(item['recycled']['selected_partition_index']==BLOCK_SIZES.index(stored_pooled_winner['block_size']),
                'wrong pooled winner index')
        for key in ('mean','n','bound','alpha','kl_radius'):
            near(item['recycled'][key],stored_pooled_winner[key],0.,0.)
        near(item['recycled']['per_partition_lower'],[r['lower'] for r in item['recycled_per_block']],0.,0.)
        require(item['recycled']['one_event_for_all_partitions'] is True,'pooled event scope')
        near(item['combined_lower'],max(want['block_best'],want['pool_best']),2e-11,2e-11)
        require(item['combined_has_no_additional_event'] is True,'combined endpoint scope')
        require(item['combined_selected_method']==('hierarchy' if item['hierarchy_lower']>=item['recycled']['lower']
                else 'recycled'),'combined method label')
        require(set(item['postprocessing_seconds'])=={'upper','block_KL','recycled_KL'},'postprocessing timing roster')
        for value in item['postprocessing_seconds'].values():timing(value)
        require(item['recycled']['mean']<=item['upper']['mean']+2e-10,
                'empirical lower value exceeds feasible fitted loss')
        counts['one_sided_events']+=7
    return reconstructed,counts,maxima


def same(a,b):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),'mapping fields differ')
        for key in a:same(a[key],b[key])
    elif isinstance(a,(np.ndarray,list,tuple)):
        if isinstance(a,np.ndarray):near(a,b,0.,0.)
        else:
            require(len(a)==len(b),'sequence length mismatch')
            for x,y in zip(a,b,strict=True):same(x,y)
    elif isinstance(a,(bool,np.bool_)) or isinstance(b,(bool,np.bool_)):
        require(type(a) is bool and type(b) is bool and a is b,'stored Boolean differs')
    else:require(a==b,'stored scalar differs')


def frozen():
    f=read(RUN/'freeze.json')
    require(f['NREP']==12 and f['NCASES']==24 and f['sizes']==SIZES,'study sample budget')
    require(f['prefixes']==[2,6] and f['sample_sizes']==list(SAMPLE_SIZES) and
            f['certificate_block_sizes']==list(BLOCK_SIZES),'study class/sample/block roster')
    require(f['alpha']==ALPHA and f['confidence_events']==1008 and f['delta']==.05,
            'joint probability allocation')
    require(f['kappa']==2. and f['target_bounds']==list(BOUNDS) and
            f['mse_tolerance']==.1 and f['primary_n']==1024,'task or decision changed')
    require(f['geometry_roster']==roster() and f['ridges']==list(RIDGES),'development menu')
    require(f['geometry_config']==dict(scalings=list(SCALES),caps=[1,2],cluster_counts=[2,4],
        max_fit=256,max_references=24,tau=1e-6),'geometry settings')
    require(f['workers']==4 and f['numerical_threads_per_worker']==1,'execution resources')
    require(f['feature_slices']==dict(support=[0,256],train=[256,768],validation=[768,1024]),
            'development row usage')
    require(f['streams']==independent_streams() and len(set(f['streams'].values()))==144,'seed schedule')
    require(len({f['streams'][f'r{i}_geometry']%2**31 for i in range(24)})==24,'native seed collision')
    old=read(ROOT/'step10_stable_normalization/results/run_20261105/freeze.json')
    reserved=set(old['excluded_prior_scalar_streams'])|set(old['predictive_streams'].values())|set(old['numerical_streams'].values())
    reserved.update(old.get('historical_streams_retained_only_for_paired_diagnostics',{}).values())
    for folder in ('step11_output_accuracy/prediction/run','step12_predictive_geometry/study/run',
                   'step13_temporal_geometry/study/run','step14_linear_rollout/study/run',
                   'step15_external_dynamics/study/run','step16_observable_adequacy/study/run'):
        previous=read(ROOT/folder/'freeze.json')
        reserved.update(previous['streams'].values());reserved.update(previous.get('manufactured_seeds',()))
    require(f['excluded_prior_seeds']==sorted(reserved) and not reserved.intersection(f['streams'].values()),
            'prior stream reservation check')
    require(f['manufactured_seeds']==list(range(2026110617000,2026110617100)) and
            not set(f['manufactured_seeds']).intersection(f['streams'].values()),'manufactured stream separation')
    expected_cases=[dict(case=i,regime=REGIMES[i//12],regime_index=i//12,repetition=i%12) for i in range(24)]
    require(f['cases']==expected_cases,'complete learning repetition roster')
    require(f['protocol']==read(HERE/'protocol17.json') and
            f['protocol_sha256']==sha(HERE/'protocol17.json'),'frozen protocol differs')
    for filename,digest in f['sources'].items():require(sha(ROOT/filename)==digest,'frozen source changed: '+filename)
    for name in ('runner17.py','controller17.py','data17.py','certificate17.py','block_solver17.py',
                 'verify17.py','audit_math17.py','check_audit17.py','check_runner17.py','check_certificate17.py'):
        path=HERE/name
        require(f['sources'].get(path.relative_to(ROOT).as_posix())==sha(path),'required source is not frozen: '+name)
    for stem in ('runner','certificate','auditor'):
        gate=read(HERE.parent/'tests'/f'{stem}_preflight.json')
        require(gate['passed'] is True and gate['no_new_scientific_data'] is True,'manufactured preflight failed')
    for name,digest in f['predecessor_sha256'].items():
        require(sha(ROOT/'step16_observable_adequacy/study/run'/name)==digest,'predecessor changed')
    require(read(ROOT/'step16_observable_adequacy/study/run/verification.json')['passed'] is True,'prior study gate')
    return f


def stage_digest(stage):
    record=read(RUN/(stage+'_seal.json'))
    require(record['freeze_sha256']==sha(RUN/'freeze.json'),'stage detached from freeze')
    actual={p.relative_to(RUN).as_posix():sha(p) for p in sorted((RUN/stage).rglob('*')) if p.is_file()}
    require(record['files']==actual,'sealed evidence changed: '+stage)
    return sha(RUN/(stage+'_seal.json'))


def checked_events(events,n,folding,powers):
    require(len(events)==folding+powers,'feature-event count')
    counts=Counter();maxima={}
    for event in events:
        require(event['n']==n and event['seconds']>=0,'feature event rows/time')
        if event['kind']=='folding_scores':
            counts['folding_calls']+=1;counts['native_query_rows']+=n
            require(sum(event['precision_counts'].values())==n,'precision accounting')
            require(set(event['precision_counts']).issubset({'80','128','192'}),'precision ladder')
            require(event['max_score_error']<=1e-10 and event['max_F_error']<=1e-9,'native output gate')
            for key in ('max_score_error','max_F_error'):update(maxima,key,event[key])
        else:
            require(event['kind']=='power_state','unknown feature event')
            counts['power_calls']+=1;counts['power_query_rows']+=n
            require(event['max_P_error']<=1e-9 and event['max_block_sum_residual']<=2e-13,'soft-state output gate')
            for key in ('max_P_error','max_block_sum_residual'):update(maxima,key,event[key])
    require(counts['folding_calls']==folding and counts['power_calls']==powers,'feature stage census')
    return counts,maxima


def check_payload(cost,bank,head,prefix):
    blocks=bank.blocks[:1 if prefix==2 else 2]
    expected=dict(scaler_offset=bank.scaler.offset.nbytes,scaler_divisor=bank.scaler.divisor.nbytes,
        scaler_constant_mask=bank.scaler.constant_mask.nbytes,
        retained_centers=sum(b['cluster_centers'].nbytes for b in blocks),
        native_encoder_arrays=sum(e[k].nbytes for b in blocks for group in b['classifier']
            for e in group for k in ('y_data_n_subspace','PC','B','y_data','sqrt_weights_matrix')),
        head_coefficient=np.asarray(head['coef']).nbytes,scalar_parameters=40)
    require(cost['components']==expected and cost['numeric_bytes']==sum(expected.values()),'deployment payload census')
    require(cost['prefix']==prefix and cost['native_states']==prefix and
            cost['feature_blocks']==len(blocks),'deployment prefix identity')


def check_selection(paths,grid,models,selected):
    require(grid.shape==(16,2,3) and np.isfinite(grid).all(),'candidate score grid')
    triples=[(float(grid[gi,1,ri]),gi,ri) for gi in range(16) for ri in range(3)]
    _,gi,large_ri=min(triples)
    require(selected['geometry_index']==gi and selected['geometry_id']==f'g{gi}',
            'largest-prefix geometry winner differs')
    require(selected['geometry_spec']==roster()[gi] and selected['largest_prefix_ridge_index']==large_ri,
            'selected geometry specification')
    require(selected['nested_geometry_shared'] is True and selected['validation_selection_only'] is True,
            'nested development selection semantics')
    for pi,prefix in enumerate((2,6)):
        ri=min(range(3),key=lambda i:(grid[gi,pi,i],i))
        chosen=selected['selected_heads'][str(prefix)]
        require(chosen['ridge_index']==ri and chosen['ridge']==RIDGES[ri] and chosen['prefix']==prefix,
                'conditional ridge winner differs')
        near(chosen['validation_mse'],grid[gi,pi,ri],0.,0.)
        same(models[str(prefix)],paths[f'g{gi}']['heads'][str(prefix)][ri])


def fit(out):
    require(not (RUN/'cal').exists() and not (RUN/'test').exists(),'fit audit must precede future data')
    f=frozen();seal=stage_digest('fit');counts=Counter();maxima={}
    for case in f['cases']:
        i=case['case'];folder=RUN/'fit'/f'r{i}'
        pools=dict(np.load(folder/'pools.npz'))
        receipt=read(folder/'data_receipt.json')
        identity(receipt,case)
        for pool in ('support','train','validation'):
            data={k[len(pool)+1:]:v for k,v in pools.items() if k.startswith(pool+'_')}
            update(maxima,'generator_replay_error',check_data(data,SIZES[pool],f['streams'][f'r{i}_{pool}'],
                None if pool=='support' else case['regime']))
            require(receipt['pool_seeds'][pool]==f['streams'][f'r{i}_{pool}'],'fit receipt stream')
        require(receipt['calibration_directory_absent'] is True and receipt['terminal_directory_absent'] is True
                and receipt['support_targets_generated'] is False,'fit chronology or support labels')
        require(receipt['freeze_sha256']==sha(RUN/'freeze.json'),'fit sampling freeze link')
        banks=load(folder/'banks.pkl')
        check_geometry(banks,pools['support_X'],read(folder/'geometry_metadata.json'),f['streams'][f'r{i}_geometry'])
        features=load(folder/'allfeaturecandidates.pkl');paths=load(folder/'allheadpaths.pkl')
        require(set(features)==set(paths)=={s['id'] for s in roster()},'full candidate evidence')
        X=np.vstack((pools['support_X'],pools['train_X'],pools['validation_X']))
        grid=np.load(folder/'grid.npz')['validation_losses']
        for spec in roster():
            F=features[spec['id']];simplex(F)
            require(F.shape==(1024,6),'development feature sample count')
            path=paths[spec['id']]
            same(path['spec'],spec)
            near(path['validation_losses'],grid[spec['index']],0.,0.)
            update(maxima,'feature_replay_error',feature_replay(banks[(spec['scaling'],spec['cap'])],X,
                spec['concentration'],F,[0,31,255,256,511,767,768,1023]))
            for pi,prefix in enumerate((2,6)):
                require(len(path['heads'][str(prefix)])==3,'head path count')
                for ri,ridge in enumerate(RIDGES):
                    model=path['heads'][str(prefix)][ri]
                    details=check_head(model,F[256:768],pools['train_Y'],prefix,ridge)
                    pred=prediction(F[768:1024],model,prefix)
                    loss=float(np.mean((pred-pools['validation_Y'])**2))
                    near(grid[spec['index'],pi,ri],loss,2e-13,2e-13)
                    update(maxima,'maximum_head_gap',details['gap'])
                    update(maxima,'maximum_coefficient_norm_sum',details['norm_sum'])
                    counts['candidate_heads']+=1;counts['nonconverged_heads']+=not model['converged']
        models,selected=load(folder/'models.pkl'),read(folder/'selected.json')
        identity(selected,case)
        check_selection(paths,grid,models,selected)
        chosen=load(folder/'selected_bank.pkl');spec=selected['geometry_spec']
        original=banks[(spec['scaling'],spec['cap'])]
        require(chosen.cap==original.cap and chosen.seed==original.seed and tuple(chosen.block_sizes)==(2,4),
                'selected bank identity')
        require(chosen.scaler.mode==original.scaler.mode and chosen.scaler.training_rows==original.scaler.training_rows
                and chosen.tau==original.tau,'selected scaler/bank parameters changed')
        for name in ('offset','divisor','constant_mask'):
            require(np.array_equal(getattr(chosen.scaler,name),getattr(original.scaler,name)),
                    'selected scaler differs from fitted geometry')
        for cb,ob in zip(chosen.blocks,original.blocks,strict=True):
            for cg,og in zip(cb['classifier'],ob['classifier'],strict=True):
                for name in ('PC','B','y_data','y_data_n_subspace','sqrt_weights_matrix'):
                    near(cg[0][name],og[0][name],0.,0.)
        costs=read(folder/'costs.json')
        for prefix in (2,6):check_payload(costs[str(prefix)],chosen,models[str(prefix)],prefix)
        require(costs['2']['numeric_bytes']<costs['6']['numeric_bytes'],'retained cost does not increase')
        c,m=checked_events(read(folder/'events.json'),1024,4,16);counts.update(c)
        for key,value in m.items():update(maxima,key,value)
        summary=read(folder/'summary.json')
        identity(summary,case)
        all_heads=[h for path in paths.values() for local in path['heads'].values() for h in local]
        require(summary['nonconverged_heads']==sum(not h['converged'] for h in all_heads),
                'head convergence census')
        near(summary['maximum_head_gap'],max(h['gap'] for h in all_heads),0.,0.)
        near(summary['maximum_norm_sum'],max(h['norm_sum'] for h in all_heads),0.,0.)
        same(summary['selected'],selected);same(summary['costs'],costs)
        for key in ('seconds','geometry_fit_seconds','shared_encoding_seconds','heads_and_validation_seconds',
                    'summed_head_fit_seconds'):timing(summary[key])
        require(summary['fit_candidate_heads']==96 and summary['feature_configurations']==16
                and summary['native_regrouped_banks_fitted_and_charged']==4,'fit cost scope')
        require(summary['no_calibration_reselection'] is True and summary['no_postselection_refit'] is True,
                'fit freeze semantics')
        counts['cases']+=1;counts['feature_configurations']+=16
    summary=read(RUN/'fit/summary.json')
    case_roster(summary['by_case'],f['cases'])
    require(summary['candidate_heads']==2304 and summary['feature_configurations']==384,'fit aggregate census')
    require(summary['all_selections_completed_before_calibration'] is True and
            summary['all_selections_completed_before_terminal'] is True,'all selections chronology')
    for i,row in enumerate(summary['by_case']):same(row,read(RUN/'fit'/f'r{i}'/'summary.json'))
    result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(RUN/'freeze.json'),
        fit_seal_sha256=seal,checked_before_calibration_generation=True,checked_before_terminal_generation=True,
        counts=dict(counts),maxima=maxima,verifier_source_sha256=sha(HERE/'verify17.py'),
        helper_source_sha256=sha(HERE/'audit_math17.py'),
        numerical_scope='independent data/head/selection replay; common-engine feature sentinels; no independent geometry refit')
    write(out,result);return result


def decisions(reconstructed,costs,regime,n):
    result={}
    uppers={p:reconstructed[p][n]['upper'] for p in (2,6)}
    for method in ('hierarchy','recycled','combined'):
        lows={p:(reconstructed[p][n]['block_best'] if method=='hierarchy'
            else reconstructed[p][n]['pool_best'] if method=='recycled'
            else max(reconstructed[p][n]['block_best'],reconstructed[p][n]['pool_best'])) for p in (2,6)}
        compatible=[p for p in (2,6) if lows[p]<=.1]
        accepted=[p for p in (2,6) if uppers[p]<=.1]
        low=min(compatible) if compatible else None
        high=min(accepted) if accepted else None
        refinement=lows[2]>.1 and uppers[6]<=.1
        excluded=all(lows[p]>.1 for p in (2,6))
        result[method]=dict(lower_by_prefix={str(p):v for p,v in lows.items()},
            upper_by_prefix={str(p):v for p,v in uppers.items()},
            excluded_by_prefix={str(p):bool(lows[p]>.1) for p in (2,6)},
            accepted_by_prefix={str(p):bool(uppers[p]<=.1) for p in (2,6)},
            compatible_minimum_prefix=low,adequate_minimum_prefix=high,
            cost_lower_numeric_bytes=None if low is None else costs[str(low)]['numeric_bytes'],
            cost_upper_numeric_bytes=None if high is None else costs[str(high)]['numeric_bytes'],
            minimum_prefix_certified=bool(high is not None and low==high),
            refinement_certified=bool(refinement),all_classes_excluded=bool(excluded),
            resolved_primary_regime_question=bool(refinement if regime=='repairable' else excluded))
    h,p=(result[m]['resolved_primary_regime_question'] for m in ('hierarchy','recycled'))
    result['comparison']=dict(paired_resolution='both' if h and p else 'hierarchy_only' if h else 'recycled_only' if p else 'neither',
        same_fitted_upper_endpoints=True,no_additional_optimization_for_recycled=True,primary_n=n==1024,
        regime_question='refinement' if regime=='repairable' else 'joint_class_exclusion')
    result['upper_only']=dict(adequate_minimum_prefix=result['hierarchy']['adequate_minimum_prefix'],
        class_exclusions_available=False,resolution='fitted adequacy only; does not exclude an entire class')
    return result


def check_decisions(saved,want):
    require(set(saved)==set(want),'decision methods differ')
    for method in saved:
        require(set(saved[method])==set(want[method]),'decision fields differ')
        for key,value in want[method].items():
            actual=saved[method][key]
            if key in ('lower_by_prefix','upper_by_prefix'):
                require(set(actual)==set(value),'decision prefix keys')
                for prefix in value:near(actual[prefix],value[prefix],2e-11,2e-11)
            elif isinstance(value,bool):require(actual is value,'decision Boolean differs')
            elif isinstance(value,dict):same(actual,value)
            else:require(actual==value,'decision differs: '+method+'/'+key)


def check_eval_input(f,stage,case):
    i=case['case'];folder=RUN/stage/f'r{i}';fitfolder=RUN/'fit'/f'r{i}'
    data=dict(np.load(folder/'observations.npz'))
    error=check_data(data,SIZES[stage],f['streams'][f'r{i}_{stage}'],case['regime'])
    receipt=read(folder/'data_receipt.json')
    identity(receipt,case)
    require(receipt['pool_seed']==f['streams'][f'r{i}_{stage}'],'evaluation receipt stream differs')
    require(receipt['freeze_sha256']==sha(RUN/'freeze.json') and
            receipt['fit_seal_sha256']==sha(RUN/'fit_seal.json') and
            receipt['precalibration_fit_gate_sha256']==sha(RUN/'fit_validation.json'),
            'evaluation missing precalibration gate link')
    require(receipt['selected_models_sha256']==sha(fitfolder/'models.pkl') and
            receipt['selected_bank_sha256']==sha(fitfolder/'selected_bank.pkl') and
            receipt['frozen_models_unchanged'] is True,'selection changed during evaluation')
    if stage=='cal':require(receipt['terminal_directory_absent'] is True,'calibration after terminal generation')
    else:
        require(receipt['cal_seal_sha256']==sha(RUN/'cal_seal.json') and
                receipt['preterminal_cal_gate_sha256']==sha(RUN/'cal_validation.json'),
                'terminal generation missing calibration gate')
    features=dict(np.load(folder/'features.npz'));simplex(features['F'])
    require(features['F'].shape==(SIZES[stage],6),'evaluation feature shape')
    for key in ('F_lo','F_hi','score_lo','score_hi'):
        require(features[key].shape==features['F'].shape and np.isfinite(features[key]).all(),
                'evaluation enclosure shape or finite values')
    require(np.all(features['F_lo']<=features['F']) and np.all(features['F']<=features['F_hi']),
            'evaluation nominal feature outside enclosure')
    require(np.max(np.maximum(features['F']-features['F_lo'],features['F_hi']-features['F']))<=1e-9,
            'evaluation feature enclosure width')
    selected=read(fitfolder/'selected.json');models=load(fitfolder/'models.pkl');bank=load(fitfolder/'selected_bank.pkl')
    feature_error=feature_replay(bank,data['X'],selected['geometry_spec']['concentration'],features['F'],
        [0,SIZES[stage]//4,SIZES[stage]//2,SIZES[stage]-1])
    saved=dict(np.load(folder/'predictions.npz'))
    predicted=np.column_stack([prediction(features['F'],models[str(p)],p) for p in (2,6)])
    prediction_error=near(saved['pred'],predicted,3e-13,3e-13)
    near(saved['loss'],(data['Y'][:,None]-predicted)**2,3e-13,3e-13)
    near(saved['prefixes'],[2,6],0.,0.)
    return data,features['F'],predicted,saved,dict(generator_replay_error=error,
        feature_replay_error=feature_error,prediction_replay_error=prediction_error)


def verify_calibration(f):
    counts=Counter();maxima={};all_reconstructed={}
    for case in f['cases']:
        i=case['case'];folder=RUN/'cal'/f'r{i}'
        data,F,pred,saved,m=check_eval_input(f,'cal',case)
        for key,v in m.items():update(maxima,key,v)
        evidence=load(folder/'dual_records.pkl');record=read(folder/'calibration.json')
        identity(record,case)
        require(record['alpha']==ALPHA and record['confidence_events_per_case']==42 and
                record['same_solved_duals'] is True,'calibration event family')
        reconstructed={}
        for pi,prefix in enumerate((2,6)):
            result,c,m=audit_certificate(F[:,:prefix],data['Y'],pred[:,pi],evidence[str(prefix)])
            same(record['by_prefix'][str(prefix)],evidence[str(prefix)]['records'])
            reconstructed[prefix]=result;counts.update(c)
            for key,v in m.items():update(maxima,key,v)
        costs=read(RUN/'fit'/f'r{i}'/'costs.json')
        for n in SAMPLE_SIZES:check_decisions(record['decisions'][str(n)],decisions(reconstructed,costs,case['regime'],n))
        all_reconstructed[i]=reconstructed
        c,m=checked_events(read(folder/'events.json'),4096,1,1);counts.update(c)
        for key,v in m.items():update(maxima,key,v)
        summary=read(folder/'summary.json')
        identity(summary,case)
        same(summary['decisions'],record['decisions']);same(summary['by_prefix'],record['by_prefix'])
        for prefix in ('2','6'):
            same(summary['block_solve_seconds'][prefix],evidence[prefix]['block_solve_seconds'])
            timing(summary['certificate_seconds'][prefix])
            for b in BLOCK_SIZES:
                e=evidence[prefix]['block_evidence'][str(b)]
                expected=dict(programs=len(e['sse_lower']),unconverged=int(np.sum(~e['converged'])),
                    max_gap=float(np.max(e['gap'])),
                    max_primal_feasibility_residual=float(np.max(e['feasibility_residual'])),
                    max_iterations=int(np.max(e['iterations'])))
                same(summary['solver_diagnostics'][prefix][str(b)],expected)
        timing(summary['seconds']);timing(summary['shared_encoding_seconds'])
        require(summary['models_unchanged_sha256']==sha(RUN/'fit'/f'r{i}'/'models.pkl') and
                summary['geometry_unchanged_sha256']==sha(RUN/'fit'/f'r{i}'/'selected_bank.pkl'),
                'calibration changed selected models')
        counts['cases']+=1
    aggregate=read(RUN/'cal/summary.json')
    case_roster(aggregate['by_case'],f['cases'])
    require([(r['regime'],r['n']) for r in aggregate['paired_resolution']]==
            [(regime,n) for regime in REGIMES for n in SAMPLE_SIZES],
            'paired-resolution roster is incomplete or duplicated')
    require(aggregate['confidence_events']==1008 and aggregate['alpha']==ALPHA and aggregate['delta']==.05,
            'aggregate confidence family')
    require(aggregate['cases']==24 and aggregate['selected_classes']==48 and
            aggregate['class_sample_records']==144 and aggregate['primary_n']==1024 and
            aggregate['block_family_events_per_class_sample']==5 and aggregate['recycled_events_per_class_sample']==1,
            'aggregate class/event census')
    require(aggregate['no_new_optimization_for_recycled'] is True and
            aggregate['same_per_endpoint_alpha'] is True and
            aggregate['calibration_completed_before_terminal'] is True,'calibration scope/gate')
    require(aggregate['numerical_block_programs']==counts['block_programs'],'block program census')
    require(counts['one_sided_events']==1008,'audited confidence event count')
    for i,row in enumerate(aggregate['by_case']):same(row,read(RUN/'cal'/f'r{i}'/'summary.json'))
    for entry in aggregate['paired_resolution']:
        regime,n=entry['regime'],entry['n']
        rows=[r for r in aggregate['by_case'] if r['regime']==regime]
        count={k:sum(r['decisions'][str(n)]['comparison']['paired_resolution']==k for r in rows)
               for k in ('both','hierarchy_only','recycled_only','neither')}
        require(entry['paired_counts']==count and entry['repetitions']==12,'paired resolution counts')
        for method in ('hierarchy','recycled','combined'):
            require(entry['resolved_by_method'][method]==sum(r['decisions'][str(n)][method]['resolved_primary_regime_question'] for r in rows),
                    'method resolution aggregate')
    return counts,maxima,all_reconstructed


def check_prior_gates(require_cal=False):
    fitgate=read(RUN/'fit_validation.json')
    require(fitgate['passed'] is True and fitgate['checked_before_calibration_generation'] is True and
            fitgate['checked_before_terminal_generation'] is True,'invalid precalibration fit gate')
    require(fitgate['freeze_sha256']==sha(RUN/'freeze.json') and
            fitgate['fit_seal_sha256']==sha(RUN/'fit_seal.json'),'fit gate identity')
    if require_cal:
        calgate=read(RUN/'cal_validation.json')
        require(calgate['passed'] is True and calgate['checked_before_terminal_generation'] is True,'invalid preterminal cal gate')
        require(calgate['freeze_sha256']==sha(RUN/'freeze.json') and
                calgate['fit_seal_sha256']==sha(RUN/'fit_seal.json') and
                calgate['cal_seal_sha256']==sha(RUN/'cal_seal.json') and
                calgate['precalibration_fit_gate_sha256']==sha(RUN/'fit_validation.json'),'calibration gate identity')


def cal(out):
    require(not (RUN/'test').exists(),'calibration audit must precede terminal generation')
    f=frozen();stage_digest('fit');seal=stage_digest('cal');check_prior_gates()
    counts,maxima,_=verify_calibration(f)
    result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(RUN/'freeze.json'),
        fit_seal_sha256=sha(RUN/'fit_seal.json'),cal_seal_sha256=seal,
        precalibration_fit_gate_sha256=sha(RUN/'fit_validation.json'),
        checked_before_terminal_generation=True,counts=dict(counts),maxima=maxima,
        verifier_source_sha256=sha(HERE/'verify17.py'),helper_source_sha256=sha(HERE/'audit_math17.py'),
        numerical_scope='independent data, feasible signed-dual, KL and decision replay; guards are not interval proofs')
    write(out,result);return result


def chronology():
    ftime=datetime.fromisoformat(read(RUN/'freeze.json')['utc'])
    last=ftime
    for stage,next_gate in (('fit','fit_validation.json'),('cal','cal_validation.json'),('test',None)):
        stage_record=read(RUN/stage/'stage_receipt.json')
        start=datetime.fromisoformat(stage_record['started']);finish=datetime.fromisoformat(stage_record['finished'])
        require(last<=start<=finish,'original stage chronology')
        require(stage_record['completed_first_run'] is True and stage_record['cases']==24,'stage completion roster')
        seal_time=datetime.fromisoformat(read(RUN/(stage+'_seal.json'))['utc'])
        require(finish<=seal_time,'stage seal chronology')
        for i in range(24):
            record=read(RUN/stage/f'r{i}'/'data_receipt.json')
            require(start<=datetime.fromisoformat(record['started'])<=datetime.fromisoformat(record['finished'])<=finish,
                    'episode generation outside stage')
        if next_gate:
            last=datetime.fromisoformat(read(RUN/next_gate)['utc'])
            require(seal_time<=last,'audit occurred before seal')
        else:last=seal_time


def terminal(out):
    f=frozen()
    for stage in ('fit','cal','test'):stage_digest(stage)
    check_prior_gates(True);chronology()
    calcounts,calmax,_=verify_calibration(f)
    counts=Counter();maxima={};computed=[]
    for case in f['cases']:
        i=case['case'];folder=RUN/'test'/f'r{i}'
        data,F,pred,saved,m=check_eval_input(f,'test',case)
        for key,v in m.items():update(maxima,key,v)
        variance=.05**2/3 if case['regime']=='repairable' else .8**2*(.9**2+.1**2/3)**2+.05**2/3
        means=.8*data['X'][:,0]*data['X'][:,1] if case['regime']=='repairable' else np.zeros(len(F))
        near(saved['conditional_mean'],means,3e-16,0.)
        near(saved['conditional_variance'],np.full(len(F),variance),3e-16,0.)
        conditional=(means[:,None]-pred)**2+variance
        near(saved['conditional_expected_loss'],conditional,3e-13,3e-13)
        results=read(folder/'results.json')
        fitted_selection=read(RUN/'fit'/f'r{i}'/'selected.json')
        fitted_costs=read(RUN/'fit'/f'r{i}'/'costs.json')
        require([r['prefix'] for r in results]==[2,6],'terminal class roster')
        for pi,row in enumerate(results):
            identity(row,case)
            same(row['selected'],fitted_selection['selected_heads'][str(row['prefix'])])
            same(row['geometry'],fitted_selection['geometry_spec'])
            same(row['costs'],fitted_costs[str(row['prefix'])])
            near(row['observed_mse'],float(np.mean((pred[:,pi]-data['Y'])**2)),2e-13,2e-13)
            near(row['mean_conditional_expected_loss'],float(np.mean(conditional[:,pi])),2e-13,2e-13)
            near(row['population_bayes_risk'],variance,2e-16,0.)
            require(row['population_class_minimum_known'] is (case['regime']=='irreducible'),
                    'oracle scope differs')
            if case['regime']=='irreducible':near(row['known_population_class_minimum'],variance,2e-16,0.)
            else:require(row['known_population_class_minimum'] is None,'unsupported repairable class optimum')
            require(row['conditional_expected_loss_is_population_risk'] is False,'Monte Carlo conditional loss called exact population risk')
            computed.append(row)
        summary=read(folder/'summary.json');same(summary['results'],results)
        identity(summary,case)
        same(summary['frozen_calibration_decisions'],read(RUN/'cal'/f'r{i}'/'calibration.json')['decisions'])
        timing=read(folder/'deployment_timing.json');same(summary['standalone_deployment'],timing)
        for prefix in ('2','6'):
            t=timing[prefix]
            require(t['rows']==1024 and t['prefix']==int(prefix) and t['warm_process_uncached_features'] is True,
                    'deployment timing scope')
            require(t['encoder_seconds']>=0 and t['head_seconds']>=0,'negative deployment time')
            near(t['total_seconds'],t['encoder_seconds']+t['head_seconds'],0.,0.)
            c,m=checked_events(t['events'],1024,1,1);counts.update(c)
            for key,v in m.items():update(maxima,key,v)
        c,m=checked_events(read(folder/'events.json'),1024,1,1);counts.update(c)
        for key,v in m.items():update(maxima,key,v)
        require(summary['max_standalone_prediction_error']<=1e-11,'prefix deployment prediction changed')
        counts['terminal_episodes']+=1024;counts['terminal_scalar_forecasts']+=2048;counts['cases']+=1
    aggregate=read(RUN/'test/summary.json')
    case_roster(aggregate['by_case'],f['cases'])
    require([(r['regime'],r['prefix']) for r in aggregate['aggregates']]==
            [(regime,prefix) for regime in REGIMES for prefix in (2,6)],
            'terminal aggregate roster is incomplete or duplicated')
    require(aggregate['terminal_episodes']==24576 and aggregate['terminal_scalar_forecasts']==49152 and
            aggregate['terminal_targets']==24576,'terminal aggregate census')
    for i,row in enumerate(aggregate['by_case']):same(row,read(RUN/'test'/f'r{i}'/'summary.json'))
    for row in aggregate['aggregates']:
        local=[r for r in computed if r['regime']==row['regime'] and r['prefix']==row['prefix']]
        require(len(local)==row['repetitions']==12,'terminal mean repetitions')
        near(row['mean_observed_mse'],float(np.mean([r['observed_mse'] for r in local])),2e-14,2e-14)
        near(row['mean_conditional_expected_loss'],float(np.mean([r['mean_conditional_expected_loss'] for r in local])),2e-14,2e-14)
        near(row['mean_numeric_payload_bytes'],float(np.mean([r['costs']['numeric_bytes'] for r in local])),0.,0.)
    result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(RUN/'freeze.json'),
        fit_seal_sha256=sha(RUN/'fit_seal.json'),cal_seal_sha256=sha(RUN/'cal_seal.json'),
        test_seal_sha256=sha(RUN/'test_seal.json'),precalibration_fit_gate_sha256=sha(RUN/'fit_validation.json'),
        preterminal_cal_gate_sha256=sha(RUN/'cal_validation.json'),chronology_checked=True,
        counts=dict(counts),maxima=maxima,calibration_counts=dict(calcounts),calibration_maxima=calmax,
        aggregates=aggregate['aggregates'],paired_resolution=read(RUN/'cal/summary.json')['paired_resolution'],
        verifier_source_sha256=sha(HERE/'verify17.py'),helper_source_sha256=sha(HERE/'audit_math17.py'),
        independent_data_and_head_replay=True,independent_signed_dual_and_KL_replay=True,
        geometry_independently_refitted=False,common_interval_feature_sentinels=True,
        end_to_end_interval_certificate=False,scientific_units='24 complete independent learning repetitions; 12 per fixed regime')
    write(out,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('fit','cal','terminal'))
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(__debug__,'assertions must remain enabled')
    result=globals()[args.stage](args.out)
    print(json.dumps(dict(stage=args.stage,passed=result['passed'],counts=result['counts'])),flush=True)


if __name__=='__main__':main()
