"""Independent saved-evidence audit for the Step18 finite-cohort experiment.

No scientific producer, model selection routine, or certificate solver is
imported. Public artifact classes and the common forward engine may be loaded
for deserialization and explicitly identified forward replay. Audit code is
separately hashed; it never fits or selects a scientific model.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse,csv,hashlib,json,pickle,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PROJ=HERE.parent
RUN=HERE/'run'
REF=PROJ/'references'/'ptbxl_v103'
for folder in ('step6_geometry_regimes/src','step11_output_accuracy',
               'step11_output_accuracy/engine'):
    sys.path.insert(0,str(ROOT/folder))
import audit_math18 as A
require,near=A.require,A.near


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(path):
    with Path(path).open('rb') as h:return pickle.load(h)
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(path,value):
    with Path(path).open('x') as h:
        json.dump(value,h,indent=2,allow_nan=False,
                  default=lambda x:x.tolist() if isinstance(x,np.ndarray)
                  else x.item() if isinstance(x,np.generic) else str(x))
        h.write('\n')
def audit_identity():
    return {'utc':datetime.now(timezone.utc).isoformat(),
            'verification_sources':{str(p.relative_to(ROOT)):sha(p) for p in
                                    (Path(__file__),HERE/'audit_math18.py')},
            'numerical_scope':'Independent guarded floating replay, not an interval proof'}


def audit_roster():
    patients={}
    with (REF/'ptbxl_database.csv').open(newline='') as h:
        for r in csv.DictReader(h):
            p=int(float(r['patient_id']));e=int(r['ecg_id'])
            require(p==float(r['patient_id']),'patient identifier not integer')
            if p not in patients or e<patients[p][0]:patients[p]=(e,r['filename_hr'])
    ids=sorted(patients)
    require(len(ids)==18869,'population patient count')
    order=np.random.Generator(np.random.PCG64(2026110618)).permutation(ids)
    pool=order[5120:13312]
    order2=np.random.Generator(np.random.PCG64(2026110619)).permutation(8192)
    expected={'support':order[:512],'train':order[512:3584],
              'validation':order[3584:5120], 'calibration':pool[order2[:4096]],
              'terminal':pool[order2[4096:]], 'reserve':order[13312:]}
    with (REF/'patient_roster18.csv').open(newline='') as h:rows=list(csv.DictReader(h))
    require(len(rows)==len(ids) and len({r['patient_id'] for r in rows})==len(ids),
            'roster duplicate or omission')
    result={}
    for stage,wanted in expected.items():
        part=[r for r in rows if r['stage']==stage]
        near([int(r['patient_id']) for r in part],wanted,0,0)
        near([int(r['stage_index']) for r in part],np.arange(len(part)),0,0)
        for r in part:
            p=int(r['patient_id']);origin=155+int(hashlib.sha256(
                ('step18-origin-v1:'+str(p)).encode()).hexdigest(),16)%4825
            require(int(r['origin'])==origin and (int(r['ecg_id']),r['filename_hr'])==patients[p],
                    'record/origin selection differs')
            if stage in ('calibration','terminal'):
                k=int(r['evaluation_pool_index']);i=int(r['evaluation_order'])
                require(pool[k]==p and pool[order2[i]]==p,'evaluation sampling membership')
            else:require(int(r['evaluation_pool_index'])==-1,'nonpool patient indexed in pool')
        result[stage]=part
    return result


def audit_data(stage,roster):
    folder=RUN/'data'/stage
    z=arrays(folder/'arrays.npz');r=read(folder/'receipt.json')
    expected={'X','patient_id','ecg_id','origin'}|({'Y'} if stage!='support' else set())
    require(set(z)==expected,'data fields or support target exposure')
    require(z['X'].shape==(len(roster[stage]),32) and np.isfinite(z['X']).all(),
            'history shape/finite gate')
    if 'Y'in z:
        require(z['Y'].shape==(len(z['X']),) and np.isfinite(z['Y']).all()
                and np.all(np.abs(z['Y'])<=1),'target range/shape')
    for key in ('patient_id','ecg_id','origin'):
        near(z[key],[int(row[key]) for row in roster[stage]],0,0)
    require(r['artifact_sha256']==sha(folder/'arrays.npz'),'data artifact hash')
    require(r['freeze']['sha256']==sha(RUN/'freeze.json'),'data detached from freeze')
    require(not (folder/'FAILED.json').exists(),'failed extraction')
    return z,r


def check_freeze():
    f=read(RUN/'freeze.json')
    for p,s in f['sources'].items():require(sha(ROOT/p)==s,'scientific source changed: '+p)
    require(not (RUN/'data'/'reserve').exists(),'reserve opened')
    return f


def feature_slices(prefix):
    return [(0,2),(2,6),(6,14)][: {2:1,6:2,14:3}[prefix]]


def audit_kahm_payload(cost,bank,head,prefix):
    n={2:1,6:2,14:3}[prefix];blocks=bank.blocks[:n]
    components={'scaler_offset':bank.scaler.offset.nbytes,
                'scaler_divisor':bank.scaler.divisor.nbytes,
                'scaler_constant_mask':bank.scaler.constant_mask.nbytes,
                'retained_centers':sum(b['cluster_centers'].nbytes for b in blocks),
                'native_encoder_arrays':sum(e[k].nbytes for b in blocks
                    for group in b['classifier'] for e in group for k in
                    ('y_data_n_subspace','PC','B','y_data','sqrt_weights_matrix')),
                'head_coefficient':np.asarray(head['coef']).nbytes,
                'scalar_parameters':40}
    require(cost['components']==components,'KAHM deployment inventory')
    require(cost['numeric_bytes']==sum(components.values()),'KAHM bytes')


def audit_cal_certificate(cert,F,Y,pred,radii,prefix,kappa):
    require(cert['sample_sizes']==[512,2048,4096] and cert['block_size']==32,
            'certificate sample/block roster')
    require(cert['alpha']==.05/9 and cert['population_size']==8192 and
            cert['target_bounds']==[-1,1] and cert['kappa']==kappa,'certificate estimand')
    slices=feature_slices(prefix)
    require(cert['block_slices']==slices or cert['block_slices']==[list(x) for x in slices],
            'certificate metric')
    maxima={'dual_replay':0.,'gap':0.,'feasibility':0.};programs=limits=0
    out=[]
    for row,n in zip(cert['records'],(512,2048,4096),strict=True):
        require(row['n']==n and row['statistical_denominator']==n,'sample denominator')
        require(row['confidence_events']=={'class_lower':1},'event allocation')
        local=[]
        for summary,name in zip(row['partitions'],('consecutive','greedy_nearest'),strict=True):
            e=cert['block_evidence'][str(n)][name]
            idx=A.independent_partition(F[:n],32,name)
            near(e['indices'],idx,0,0)
            v=A.dual_values(F[:n],Y[:n],idx,radii[:n],slices,kappa,e)
            for key in ('sse_lower','sse_upper','dual_raw','dual_guard','gap','feasibility_residual'):
                err=near(e[key],v[key],2e-10,2e-11)
                maxima['dual_replay']=max(maxima['dual_replay'],err)
            maxima['gap']=max(maxima['gap'],float(np.max(v['gap'])))
            maxima['feasibility']=max(maxima['feasibility'],float(np.max(v['feasibility_residual'])))
            programs+=n//32;limits+=int(np.sum(~np.asarray(e['nominal_converged'])))
            mean=float(np.sum(v['sse_lower'],dtype=np.longdouble)/n)
            near(summary['empirical_minimum_lower'],mean,2e-11,2e-11)
            require(summary['covered']==n and summary['uncovered']==0,'partition coverage')
            require(summary['partition']==name,'partition name')
            local.append(mean)
        # Use the declared saved floating tie rule only after both means have
        # been independently checked. Reduction-order ulps must not invent an
        # audit failure for an otherwise identical computational maximum.
        winner=int(np.argmax([s['empirical_minimum_lower'] for s in row['partitions']]))
        require(local[winner]>=max(local)-2e-11,'selected partition is not maximal')
        lower=A.kl_lower(max(local),n,4,.05/9)
        near(row['lower'],lower,2e-11,2e-11)
        require(row['selected_partition']==('consecutive','greedy_nearest')[winner],
                'wrong computational partition maximum')
        require(max(local)<=float(np.mean((Y[:n]-pred[:n])**2))+2e-9,
                'dual exceeds feasible nominal head loss')
        out.append({'n':n,'lower':lower,'empirical_lower':max(local),
                    'partition':row['selected_partition']})
    require(programs==cert['total_block_programs']==416,'QP census for one class')
    return {'endpoints':out,'programs':programs,'iteration_limits':limits,'maxima':maxima}


def stage_seal(stage):
    r=read(RUN/(stage+'_seal.json'))
    actual={str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/stage).rglob('*')) if p.is_file()}
    require(r['files']==actual and r['freeze_sha256']==sha(RUN/'freeze.json'),
            'sealed stage changed: '+stage)
    return sha(RUN/(stage+'_seal.json'))


def feature_replay(bank,X,concentration,F,prefix,indices):
    from certified import certified_bank
    from soft_state import power_state
    ix=np.asarray(indices,dtype=int)
    e=certified_bank(bank,np.ascontiguousarray(X[ix]),prefix=prefix,
                    precision=(80,128,192),score_atol=1e-12,F_atol=1e-9)
    require(np.all(e['accepted']),'forward replay failed')
    f,lo,hi=power_state(e['score_lo'],e['score_hi'],(2,4,8)[:len(feature_slices(prefix))],
                      concentration,tau=1e-6,atol=1e-9)
    return near(F[ix],f,3e-10,2e-11)


def geometry_audit(banks,U):
    require(set(banks)=={(s,q) for s in ('minmax01','standard') for q in (2,8,20)},
            'six geometry banks')
    rank_rows=[]
    for scale in ('minmax01','standard'):
        offset=U.min(0) if scale=='minmax01' else U.mean(0)
        divisor=U.max(0)-U.min(0) if scale=='minmax01' else U.std(0)
        constant=U.max(0)==U.min(0);divisor=np.where(constant,1,divisor)
        scaled=(U-offset)/divisor;scaled[:,constant]=0
        possible={r.tobytes() for r in scaled.astype(np.float32).astype(float)}
        augmented=None
        for cap in (2,8,20):
            bank=banks[(scale,cap)]
            require(bank.kind=='learned' and bank.cap==cap and bank.seed==2026110618,
                    'geometry identity')
            require(bank.block_sizes==(2,4,8) and bank.n_features_in_==32 and bank.tau==1e-6,
                    'geometry dimensions')
            near(bank.scaler.offset,offset,0,0);near(bank.scaler.divisor,divisor,0,0)
            near(bank.scaler.constant_mask,constant,0,0)
            require(bank.scaler.training_rows==512 and bank.scaler.mode==scale,'scaler provenance')
            near(bank.fit_indices,np.arange(512),0,0)
            for bi,block in enumerate(bank.blocks):
                require(len(block['classifier'])==(2,4,8)[bi],'native state count')
                for j,group in enumerate(block['classifier']):
                    require(len(group)==1,'native encoder multiplicity')
                    e=group[0];refs=np.asarray(e['y_data']).T;rank=e['PC'].shape[1]
                    require(refs.shape[1]==32 and 2<=len(refs)<=24 and
                            1<=rank<=min(cap,32,len(refs)-1),'native reference/rank dimensions')
                    for r in refs:
                        if r.tobytes() not in possible:
                            if augmented is None:
                                sq=np.zeros((512,512))
                                for k in range(32):sq+=(scaled[:,k,None]-scaled[None,:,k])**2
                                np.fill_diagonal(sq,np.inf)
                                aux=(.95*scaled+.05*scaled[np.argmin(sq,axis=1)]).astype(np.float32).astype(float)
                                augmented={a.tobytes() for a in aux}
                            require(r.tobytes() in augmented,'reference has no support origin')
                    near(e['y_data'],banks[(scale,20)].blocks[bi]['classifier'][j][0]['y_data'],0,0)
                    audit=e['dimension_audit']
                    require(audit['effective_rank']==rank and audit['requested_cap']==cap and
                            audit['ambient_dimension']==32 and audit['local_reference_count']==len(refs),
                            'native dimension audit')
                    rank_rows.append(dict(scaling=scale,cap=cap,block=bi,state=j,rank=rank,references=len(refs)))
    return rank_rows


def fit_audit(write_gate=True):
    f=check_freeze();roster=audit_roster();fitsha=stage_seal('fit')
    if write_gate:
        require(not (RUN/'cal').exists() and not (RUN/'data'/'calibration').exists(),
                'fit audit must precede calibration access')
    data={s:audit_data(s,roster)[0] for s in ('support','train','validation')}
    obs=arrays(RUN/'fit/observations.npz')
    for s,z in data.items():
        near(obs[s+'_X'],z['X'],0,0);near(obs[s+'_patient_id'],z['patient_id'],0,0)
        if s!='support':near(obs[s+'_Y'],z['Y'],0,0)
    banks=load(RUN/'fit/banks.pkl');ranks=geometry_audit(banks,data['support']['X'])
    features=arrays(RUN/'fit/feature_candidates.npz')
    models=load(RUN/'fit/candidate_heads.pkl');rows=read(RUN/'fit/kahm_candidates.json')
    saved=arrays(RUN/'fit/kahm_validation_predictions.npz')
    selected=read(RUN/'fit/selected.json');sm=load(RUN/'fit/selected_models.pkl')
    require(len(rows)==len(models)==648 and saved['predictions'].shape==(1536,648),'KAHM candidate census')
    X=np.concatenate((data['train']['X'],data['validation']['X']))
    ymax=data['validation']['Y'];max_feature=max_head=max_val=0.
    gi=0
    for scale in ('minmax01','standard'):
        for cap in (2,8,20):
            for omega in (.5,4.,16.,64.):
                key=f'g{gi:02d}';gi+=1;F=features[key]
                require(F.shape==(4608,14) and np.isfinite(F).all(),'feature candidate shape')
                for a,b in feature_slices(14):near(F[:,a:b].sum(1),np.ones(len(F)),3e-13,0)
                require(np.min(F)>=0 and np.max(F)<=1+2e-13,'simplex range')
                # Six banks, both extreme concentrations, common validated engine.
                if omega in (.5,64.):
                    max_feature=max(max_feature,feature_replay(banks[(scale,cap)],X,omega,F,14,(0,3071,4607)))
    for i,r in enumerate(rows):
        cid=f'k{i:03d}';require(r['id']==cid and r['index']==i,'head menu order')
        # Independent Cartesian-menu identity, not producer-generated identifiers.
        g=i//27;inside=i%27;prefix=(2,6,14)[inside//9]
        kap=(1.,2.,4.)[(inside%9)//3];alpha=(0.,1e-4,1e-2)[inside%3]
        require(r['geometry']['id']==f'g{g:02d}' and r['prefix']==prefix and
                r['kappa']==kap and r['alpha']==alpha,'head candidate specification')
        head=models[cid];F=features[r['geometry']['id']][:,:prefix]
        require(head['prefix']==prefix and head['alpha']==alpha and head['kappa']==kap,
                'head stored specification')
        d=A.head_values(F[:3072],data['train']['Y'],head,feature_slices(prefix))
        for name,key in (('objective','objective'),('gap','gap'),('norm_sum','norm_sum')):
            max_head=max(max_head,near(head[name],d[key],3e-10,3e-10))
        near(head['intercept'],d['expected_intercept'],3e-12,3e-12)
        require(head['converged']==(head['gap']<=1e-7*max(1,abs(head['objective']))),'head convergence flag')
        pv=np.clip(F[3072:]@head['coef']+head['intercept'],-1,1)
        max_val=max(max_val,near(saved['predictions'][:,i],pv,2e-12,2e-12))
        near(r['validation_mse'],np.mean((ymax-pv)**2),2e-12,2e-12)
    for p in ('2','6','14'):
        winner=min((r for r in rows if r['prefix']==int(p)),key=lambda r:(r['validation_mse'],r['index']))
        require(selected['kahm'][p]==winner and sm[p]['candidate_id']==winner['id'],'KAHM winner selection')
        near(sm[p]['head']['coef'],models[winner['id']]['coef'],0,0)
        audit_kahm_payload(winner['cost'],sm[p]['bank'],sm[p]['head'],int(p))
    br=read(RUN/'fit/baseline_candidates.json');bm=load(RUN/'fit/baseline_candidates.pkl')
    bs=load(RUN/'fit/baseline_selected.pkl');bp=arrays(RUN/'fit/baseline_validation_predictions.npz')
    require(len(br)==len(bm)==140 and len(bs)==13,'baseline candidate/winner census')
    max_baseline=0.
    for i,r in enumerate(br):
        require(r['id']==f'b{i:03d}' and r['index']==i,'baseline menu order')
        m=bm[r['id']];pv=A.baseline_predict(m,data['validation']['X'])
        max_baseline=max(max_baseline,near(bp['predictions'][:,i],pv,2e-8,2e-9))
        near(r['validation_mse'],np.mean((ymax-pv)**2),2e-9,2e-9)
        require(A.numeric_bytes(m)==r['numeric_bytes'],'baseline payload census')
        if 'scaler'in m:
            sc=m['scaler'];U=data['support']['X']
            offset=U.min(0) if sc['mode']=='minmax01' else U.mean(0)
            divisor=U.max(0)-U.min(0) if sc['mode']=='minmax01' else U.std(0)
            constant=U.max(0)==U.min(0)
            near(sc['offset'],offset,0,0);near(sc['divisor'],np.where(constant,1,divisor),0,0)
            near(sc['constant_mask'],constant,0,0)
    for slot in bs:
        win=min((r for r in br if r['selection_slot']==slot),key=lambda r:(r['validation_mse'],r['index']))
        require(selected['baseline'][slot]==win and bs[slot]['candidate_id']==win['id'],'baseline winner')
        near(A.baseline_predict(bs[slot]['model'],data['validation']['X']),
             bp['predictions'][:,win['index']],2e-8,2e-9)
    ref=min(br,key=lambda r:(r['validation_mse'],r['index']))
    require(selected['reference']==ref,'reference selection')
    result={**audit_identity(),'passed':True,'freeze_sha256':sha(RUN/'freeze.json'),
            'fit_seal_sha256':fitsha,'checked_before_calibration_access':write_gate,
            'patient_counts':{s:len(z['X']) for s,z in data.items()},'reserved_patients':5557,
            'heads_checked':648,'baselines_checked':140,'baseline_winners_checked':13,
            'native_rank_rows':ranks,'max_forward_replay_error':max_feature,
            'forward_replay_scope':'common MPFI engine on36sentinel rows; no independent geometry refit',
            'max_head_arithmetic_error':max_head,'max_validation_replay_error':max_val,
            'max_baseline_replay_error':max_baseline}
    if write_gate:write(RUN/'fit_validation.json',result)
    return result


def evaluation_audit(stage):
    check_freeze();stage_seal(stage);roster=audit_roster()
    data_stage='calibration' if stage=='cal' else 'terminal'
    data,_=audit_data(data_stage,roster)
    obs=arrays(RUN/stage/'observations.npz')
    for k,v in data.items():near(obs[k],v,0,0)
    F=arrays(RUN/stage/'features.npz');P=arrays(RUN/stage/'predictions.npz')
    models=load(RUN/'fit/selected_models.pkl');baselines=load(RUN/'fit/baseline_selected.pkl')
    maximum_prediction=maximum_feature=0.;radii_out={}
    for p,m in models.items():
        f,lo,hi=(F[key+'_'+p] for key in ('F','F_lo','F_hi'))
        require(f.shape==(4096,int(p)),'selected feature dimensions')
        rad=A.block_radii(f,lo,hi,feature_slices(int(p)))
        near(F['block_radius_'+p],rad,2e-15,2e-12)
        require(np.all(F['block_radius_'+p]>=rad-2e-15),'feature radius underestimation')
        radii_out[p]=F['block_radius_'+p]
        pred=np.clip(f@m['head']['coef']+m['head']['intercept'],-1,1)
        maximum_prediction=max(maximum_prediction,near(P['pred_'+p],pred,2e-12,2e-12))
        # Independent high-precision endpoint calculation; producer uses
        # outward multiply-add at every coordinate instead.
        w=np.asarray(m['head']['coef'],dtype=np.longdouble)
        pl=np.longdouble(m['head']['intercept'])+np.sum(np.minimum(
            lo.astype(np.longdouble)*w,hi.astype(np.longdouble)*w),axis=1,dtype=np.longdouble)
        ph=np.longdouble(m['head']['intercept'])+np.sum(np.maximum(
            lo.astype(np.longdouble)*w,hi.astype(np.longdouble)*w),axis=1,dtype=np.longdouble)
        pl=np.clip(pl,-1,1);ph=np.clip(ph,-1,1)
        require(np.all(P['prediction_lo_'+p].astype(np.longdouble)<=pl) and
                np.all(P['prediction_hi_'+p].astype(np.longdouble)>=ph),'head interval enclosure')
        near(P['loss_nominal_'+p],(data['Y']-pred)**2,2e-15,2e-15)
        y=data['Y'].astype(np.longdouble)
        upper=np.maximum((y-P['prediction_lo_'+p])**2,(y-P['prediction_hi_'+p])**2)
        require(np.all(P['loss_upper_'+p].astype(np.longdouble)>=upper),'loss upper enclosure')
        maximum_feature=max(maximum_feature,feature_replay(m['bank'],data['X'],m['concentration'],
                                f,int(p),(0,2047,4095)))
    slots=P['baseline_slots'].tolist()
    require(slots==sorted(baselines),'baseline evaluation order')
    for i,slot in enumerate(slots):
        pp=A.baseline_predict(baselines[slot]['model'],data['X'])
        maximum_prediction=max(maximum_prediction,near(P['baseline_predictions'][:,i],pp,2e-8,2e-9))
    near(P['baseline_losses'],(data['Y'][:,None]-P['baseline_predictions'])**2,2e-15,2e-15)
    exactloss=(data['Y'].astype(np.longdouble)[:,None]-P['baseline_predictions'].astype(np.longdouble))**2
    require(np.all(P['baseline_loss_lower'].astype(np.longdouble)<=exactloss) and
            np.all(P['baseline_loss_upper'].astype(np.longdouble)>=exactloss),'baseline loss enclosure')
    return data,F,P,{'max_prediction_replay_error':maximum_prediction,
                    'max_forward_replay_error':maximum_feature},radii_out


def cal_audit(write_gate=True):
    check_freeze();fitsha=stage_seal('fit');calsha=stage_seal('cal')
    fitgate=read(RUN/'fit_validation.json');require(fitgate['passed'] is True,'fit gate')
    if write_gate:
        require(not (RUN/'terminal').exists() and not (RUN/'data'/'terminal').exists(),
                'cal audit precedes terminal access')
    data,F,P,metrics,radii=evaluation_audit('cal')
    evidence=load(RUN/'cal/certificate_evidence.pkl');models=load(RUN/'fit/selected_models.pkl')
    out={}
    for p in ('2','6','14'):
        out[p]=audit_cal_certificate(evidence[p],F['F_'+p],data['Y'],P['pred_'+p],
                                    radii[p],int(p),models[p]['head']['kappa'])
    require(sum(r['programs'] for r in out.values())==1248,'complete solver census')
    result={**audit_identity(),'passed':True,'freeze_sha256':sha(RUN/'freeze.json'),
            'fit_seal_sha256':fitsha,'cal_seal_sha256':calsha,
            'precalibration_fit_gate_sha256':sha(RUN/'fit_validation.json'),
            'checked_before_terminal_access':write_gate,'certificates':out,
            'class_sample_events':9,'joint_failure_probability':.05,
            'all_saved_duals_recomputed':True,**metrics}
    if write_gate:write(RUN/'cal_validation.json',result)
    return result


def final_audit():
    f=check_freeze();roster=audit_roster()
    for stage in ('fit','cal','terminal','census'):stage_seal(stage)
    require(read(RUN/'fit_validation.json')['passed'] and
            read(RUN/'cal_validation.json')['passed'],'prior audit gates')
    data,F,P,metrics,radii=evaluation_audit('terminal')
    C=arrays(RUN/'cal/predictions.npz');out=read(RUN/'census/results.json')
    costs=read(RUN/'fit/costs.json');selection=read(RUN/'fit/selected.json')
    require(out['population_size']==8192 and len(out['rows'])==16,'census population/model count')
    ref=out['reference'];require(ref['candidate_id']==selection['reference']['id'],'census reference identity')
    near(out['threshold'],1.1*ref['mse'],0,0)
    threshold_exact=np.longdouble(11)*np.longdouble(ref['mse_lower'])/np.longdouble(10)
    require(np.longdouble(out['threshold_lower'])<=threshold_exact,'adequacy threshold direction')
    threshold_exact=np.longdouble(11)*np.longdouble(ref['mse_upper'])/np.longdouble(10)
    require(np.longdouble(out['threshold_upper'])>=threshold_exact,'exclusion threshold direction')
    slots=P['baseline_slots'].tolist();largest_guard=0.
    for r in out['rows']:
        if r['family']=='kahm':
            p=str(r['prefix']);col=None
            nom=np.concatenate((C['loss_nominal_'+p],P['loss_nominal_'+p]))
            lows=np.concatenate((C['loss_lower_'+p],P['loss_lower_'+p]))
            ups=np.concatenate((C['loss_upper_'+p],P['loss_upper_'+p]))
            cost=costs['kahm'][p]
        else:
            col=slots.index(r['id'])
            nom=np.concatenate((C['baseline_losses'][:,col],P['baseline_losses'][:,col]))
            lows=np.concatenate((C['baseline_loss_lower'][:,col],P['baseline_loss_lower'][:,col]))
            ups=np.concatenate((C['baseline_loss_upper'][:,col],P['baseline_loss_upper'][:,col]))
            cost=costs['baseline'][r['id']]
        near(r['mse'],float(np.mean(nom)),0,0)
        require(r['evaluation_rows']==8192 and r['numeric_bytes']==cost['numeric_bytes'],'census count/payload')
        require(np.longdouble(r['mse_lower'])<=np.mean(lows.astype(np.longdouble)) and
                np.longdouble(r['mse_upper'])>=np.mean(ups.astype(np.longdouble)),
                'census arithmetic enclosure')
        largest_guard=max(largest_guard,r['mse_upper']-r['mse'])
        require(r['within_ten_percent_reference_mse']==(r['mse_upper']<=out['threshold_lower']),
                'adequacy decision')
        require(r['numeric_payload_smaller_than_reference']==(r['numeric_bytes']<ref['numeric_bytes']),
                'storage decision')
        require(r['meets_joint_criterion']==(r['mse_upper']<=out['threshold_lower'] and
                                            r['numeric_bytes']<ref['numeric_bytes']),'primary decision')
        near(r['mse_ratio_to_reference'],r['mse']/ref['mse'],2e-15,2e-15)
    rows=out['rows']
    for name,risk in (('nominal_frontier','mse'),('conservative_frontier','mse_upper')):
        ids=[r['id'] for r in rows if not any(q['numeric_bytes']<=r['numeric_bytes'] and
              q[risk]<=r[risk] and(q['numeric_bytes']<r['numeric_bytes'] or q[risk]<r[risk])
              for q in rows)]
        require([r['id'] for r in out[name]]==ids,'Pareto frontier')
    require(out['selected_kahm_joint_success']==any(r['meets_joint_criterion'] for r in rows
                                                 if r['family']=='kahm'),'primary aggregate')
    certificates=read(RUN/'cal/calibration.json')['by_prefix'];consistency=[]
    for p,records in certificates.items():
        fitrow=next(r for r in rows if r['id']=='kahm_'+p)
        for v in records:
            d=out['class_exclusions'][p][str(v['n'])]
            require(d['insufficient']==(v['lower']>out['threshold_upper']),'class exclusion decision')
            near(d['class_lower'],v['lower'],0,0)
            # A rare statistical coverage failure is a scientific result,
            # not an excuse to delete or rerun an experiment.
            if v['lower']>fitrow['mse_upper']:
                consistency.append({'prefix':int(p),'n':v['n'],'lower':v['lower'],
                                    'feasible_census_upper':fitrow['mse_upper']})
    timing=read(RUN/'terminal/deployment_timing.json')
    require(len(timing)==16,'deployment timing roster')
    for name,r in timing.items():
        require(r['rows']==512 and len(r['warm_query_seconds'])==3,'timing sample/repeats')
        near(r['warm_median_seconds'],np.median(r['warm_query_seconds']),0,0)
        require(all(np.isfinite(v) and v>=0 for v in r['warm_query_seconds']),'query elapsed time')
    result={**audit_identity(),'passed':True,'freeze_sha256':sha(RUN/'freeze.json'),
            'stage_seal_sha256':{s:sha(RUN/(s+'_seal.json')) for s in ('fit','cal','terminal','census')},
            'fit_validation_sha256':sha(RUN/'fit_validation.json'),
            'cal_validation_sha256':sha(RUN/'cal_validation.json'),
            'patient_counts':{s:len(v) for s,v in roster.items()},'evaluation_rows':8192,
            'scientific_fits':1,'untouched_reserve_patients':5557,
            'selected_kahm_joint_success':out['selected_kahm_joint_success'],
            'lower_exceeds_feasible_census_risk_events':consistency,
            'maximum_canonical_risk_upper_minus_nominal':largest_guard,
            'no_model_selection_after_validation':True,**metrics}
    write(RUN/'verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('roster','fit','cal','final','replay'))
    args=parser.parse_args()
    if args.stage=='roster':
        out={k:len(v) for k,v in audit_roster().items()}
    elif args.stage=='fit':out=fit_audit()
    elif args.stage=='cal':out=cal_audit()
    elif args.stage=='final':out=final_audit()
    else:
        # Replays all saved arithmetic without fitting or opening raw records.
        out={'fit':fit_audit(False),'cal':cal_audit(False),
             'terminal':evaluation_audit('terminal')[3],
             'sealed_census_sha256':stage_seal('census'),'passed':True}
    print(json.dumps(out,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x))


if __name__=='__main__':main()
