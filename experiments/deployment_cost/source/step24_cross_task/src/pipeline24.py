"""Complete deployment paths, selected rosters, and independent head replay."""
from __future__ import annotations
import pickle
import numpy as np
import geometry24 as G
import models24 as M
from common24 import task_kind


def block_sizes(prefix):
    values=[];total=0
    for k in (2,4,8,16):
        values.append(k);total+=k
        if total==prefix:return values
    raise ValueError('Invalid frozen state prefix')


def make_pipeline(spec,head,banks,kind):
    group=spec['group']
    if group in ('P','CS','MATCHED_DIAGNOSTIC'):
        bank=G.compact_bank(banks[G.bank_key(spec['scaling'],spec['cap'])],spec['prefix'])
        transform=G.view_payload(spec['view'],block_sizes(spec['prefix']),spec.get('omega'),1e-6)
        return {'pipeline_kind':'geometry','geometry':bank,'transform':transform,'head':head,'task_kind':kind}
    if group=='RAW':return {'pipeline_kind':'raw','head':head,'task_kind':kind}
    if group=='NYSTROM':return {'pipeline_kind':'nystrom','model':head,'task_kind':kind}
    raise ValueError('No fitted deployment path for group')


def baseline_pipeline(name,kind,horizon,mean):
    bounds=np.array([-1.,1.]) if kind=='hmm' else np.array([0.])
    p={'pipeline_kind':'baseline','name':name,'task_kind':kind,'output_bounds':bounds}
    if name=='training_mean':p['mean']=np.float64(mean)
    elif name=='persistence':p['column']=np.int64(31 if kind=='hmm' else 0)
    elif name=='daily_seasonal' and kind=='appliances':p['column']=np.int64(144-horizon)
    else:raise ValueError('Invalid fixed baseline')
    return p


def features(pipeline,X):
    if pipeline['pipeline_kind']=='geometry':
        scores=G.score_compact(pipeline['geometry'],X)
        return G.transform_scores(scores,pipeline['transform'])
    if pipeline['pipeline_kind']=='raw':return np.asarray(X,dtype=np.float64)
    raise ValueError('Feature replay applies to explicit-head pipelines')


def predict_nystrom_reference(model,X,kind):
    """Independent explicit distance accumulation and ordered kernel readout."""
    # Reuse the fixed affine input map; do not call the nominal kernel/predictor.
    scaled=M.apply_input_scaler(X,model['scaler'])
    landmarks=np.asarray(model['landmarks'],dtype=np.float64)
    coef=np.asarray(model['coef'],dtype=np.float64)
    result=np.empty(len(scaled),dtype=np.float64)
    for first in range(0,len(scaled),128):
        block=scaled[first:first+128]
        d2=np.zeros((len(block),len(landmarks)),dtype=np.float64)
        for col in range(block.shape[1]):
            delta=block[:,col,None]-landmarks[None,:,col]
            d2+=delta*delta
        kernel=np.exp(-float(model['gamma'])*d2)
        value=np.zeros(len(block),dtype=np.float64)
        for j in range(len(coef)):value+=kernel[:,j]*coef[j]
        result[first:first+len(block)]=value+model['intercept']
    return M.output_map(result,kind)


def predict(pipeline,X,*,reference=False,return_features=False):
    X=np.asarray(X,dtype=np.float64)
    mode=pipeline['pipeline_kind'];kind=pipeline['task_kind'];F=None
    if mode in ('geometry','raw'):
        F=features(pipeline,X)
        method=M.predict_head_reference if reference else M.predict_head
        p=method(pipeline['head'],F,kind)
    elif mode=='nystrom':
        p=(predict_nystrom_reference(pipeline['model'],X,kind) if reference
           else M.predict_nystrom(pipeline['model'],X,kind))
    elif mode=='baseline':
        p=(np.full(len(X),pipeline['mean']) if pipeline['name']=='training_mean'
           else X[:,int(pipeline['column'])].copy())
        p=M.output_map(p,kind)
    else:raise ValueError('Unknown deployment path')
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=(len(X),) or not np.isfinite(p).all():raise ArithmeticError('Invalid pipeline forecast')
    return (p,F) if return_features else p


def payload_cost(pipeline):
    inventory=M.numeric_inventory(pipeline)
    return {'complete_numeric_bytes':inventory['numeric_bytes'],
            'serialized_model_bytes':len(pickle.dumps(pipeline,protocol=5)),
            'inventory':inventory['components']}


def choose_roster(records,selectors,p_binding):
    by_id={r['id']:r for r in records}
    names={}
    def winner(name,rows):
        if not rows:raise ValueError('Missing frozen group '+name)
        names[name]=min(rows,key=lambda r:(r['validation_mse'],r['index']))['id']
    for group in ('P','CS'):
        for family in ('ridge','hgb'):
            winner(f'{group}_{family}',[r for r in records if r['group']==group and r['family']==family])
    for family in ('ridge','hgb'):
        winner(f'RAW_{family}',[r for r in records if r['group']=='RAW' and r['family']==family])
    winner('NYSTROM_ridge',[r for r in records if r['group']=='NYSTROM'])
    for name in ('C_accuracy','C_cost','P_accuracy','P_cost','CS_accuracy','CS_cost'):
        names[name]=None if selectors[name] is None else selectors[name]['id']
    for view in ('P','CS','LR0','LRM','LW'):
        for family in ('ridge','hgb'):
            matched=[]
            for r in records:
                if r.get('view')!=view or r.get('family')!=family:continue
                if any(r.get(k)!=p_binding[k] for k in ('scaling','cap','prefix')):continue
                if view!='CS' and r.get('omega')!=p_binding['omega']:continue
                matched.append(r)
            winner(f'matched_{view}_{family}',matched)
    for r in records:
        if r['group']=='UNFITTED':names[r['baseline_name']]=r['id']
    ids=sorted({v for v in names.values() if v is not None},key=lambda cid:by_id[cid]['index'])
    fitted=[cid for cid in ids if by_id[cid]['group']!='UNFITTED']
    if len(fitted)>21:raise AssertionError('Secondary-horizon fit budget exceeded')
    return {'names':names,'selected_ids':ids,'fitted_ids':fitted,'binding':p_binding}


def spec_label(spec):
    if spec['group']=='UNFITTED':return spec['baseline_name']
    text=f"{spec['view']} / {spec['family']}"
    if 'scaling' in spec:text+=f" / {spec['scaling']}"
    if 'cap' in spec:text+=f" / cap {spec['cap']} / state {spec['prefix']}"
    if spec.get('omega') is not None:text+=f" / omega {spec['omega']:g}"
    if 'landmark_count' in spec:text+=f" / {spec['landmark_count']} landmarks / gamma x{spec['gamma_multiplier']:g}"
    if 'alpha' in spec:text+=f" / alpha {spec['alpha']:g}"
    if 'max_leaf_nodes' in spec:text+=f" / {spec['max_leaf_nodes']} leaves"
    return text
