"""Write-once records and exact score arithmetic for the frozen Step41 study."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import hashlib, json, math
ROOT=Path(__file__).resolve().parents[1]
def canon(v):
    if isinstance(v,F): return str(v)
    if isinstance(v,dict): return {str(k):canon(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [canon(x) for x in v]
    if hasattr(v,'item'): return v.item()
    return v
def canonical_bytes(v): return (json.dumps(canon(v),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(v): return hashlib.sha256(canonical_bytes(v)).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p): return str(Path(p).resolve().relative_to(ROOT))
def read(p): return json.loads(Path(p).read_text())
def utc(): return datetime.now(timezone.utc).isoformat()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(canonical_bytes(v))
    return {'path':relative(p),'sha256':sha(p)}
def save_bytes(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
    return {'path':relative(p),'sha256':sha(p),'bytes':len(b)}
def number(h):
    x=float.fromhex(h)
    if not math.isfinite(x):raise ValueError('Nonfinite prediction')
    return F.from_float(x)
def score(predictions,targets,reference=False):
    if len(predictions)!=len(targets) or not targets:raise ValueError('Score roster')
    ps=[F(p['exact_physical']) if reference and p.get('exact_physical') is not None else number(p['reported_physical_hex']) for p in predictions]
    y=list(map(F,targets));sse=sum(((a-b)**2 for a,b in zip(ps,y)),F(0));mae=sum((abs(a-b) for a,b in zip(ps,y)),F(0))/len(y)
    return {'n':len(y),'sse':str(sse),'mse':str(sse/len(y)),'rmse':math.sqrt(float(sse/len(y))),'mae':str(mae)}
def selector_pool(rows,label):
    if label=='affine_rolling':return [r for r in rows if r['recipe']=='affine']
    if label=='ordinary_rolling':return [r for r in rows if r['recipe']=='baseline']
    if label in ('raw_affine','nystrom_affine','kahm_affine'):
        return [r for r in rows if r['recipe']=='affine' and r['family']==label.split('_')[0]]
    if label.endswith('_rolling'):return [r for r in rows if r.get('baseline_kind')==label[:-8]]
    raise ValueError('Unknown selector')
def select(rows,label):
    pool=selector_pool(rows,label)
    if not pool:raise ValueError('Empty selection pool')
    r=min(pool,key=lambda r:(F(r['pooled_physical_mse']),r['selection_payload_bytes'],r['candidate_id']))
    return {'candidate_id':r['candidate_id'],'pool_size':len(pool),'pooled_physical_mse':r['pooled_physical_mse'],'selection_payload_bytes':r['selection_payload_bytes']}
def primary(affine,ordinary):
    a=F(affine['mse']);b=F(ordinary['mse']);positive=b>0
    return {'positive_reference_mse':positive,'within_five_percent_mse':positive and 20*a<=21*b,'strictly_lower_mse':a<b,'at_least_five_percent_lower_mse':positive and 20*a<=19*b,'mse_ratio':float(a/b) if positive else None,'mse_reduction_percent':100*float(1-a/b) if positive else None,'scope':'FIXED_PERIOD_PRACTICAL_COMPARISON_NOT_POPULATION_NONINFERIORITY'}
