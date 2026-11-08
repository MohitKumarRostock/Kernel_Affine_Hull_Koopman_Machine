"""Shared deterministic paths, records, and source seals for Step24."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import hashlib
import json
import math
import pickle
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
PARENT=BASE/'step23_cross_task'
RUN=ROOT/'run'
TASKS=[f'hmm_{i}' for i in range(5)]+['appliances']


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def array_sha(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def jsonable(value):
    if isinstance(value,Fraction):return {'numerator':str(value.numerator),'denominator':str(value.denominator)}
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def fraction(value):
    if isinstance(value,Fraction):return value
    if isinstance(value,dict):return Fraction(int(value['numerator']),int(value['denominator']))
    if isinstance(value,int):return Fraction(value)
    raise TypeError('Expected exact rational record')


def write_json(path,value,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    text=json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n'
    temp=path.with_name(path.name+'.pending')
    temp.write_text(text);os.replace(temp,path)


def write_pickle(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending')
    with temp.open('wb') as f:pickle.dump(value,f,protocol=5)
    os.replace(temp,path)


def read_pickle(path):
    with Path(path).open('rb') as f:return pickle.load(f)


def protocol():return read_json(PARENT/'PROTOCOL.json')


def templates():return read_json(PARENT/'protocol/candidate_templates23.json')


def task_kind(task):
    if task not in TASKS:raise ValueError('Unknown fixed task')
    return 'appliances' if task=='appliances' else 'hmm'


def source_map():
    paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    paths.append(ROOT/'protocol/IMPLEMENTATION.json')
    paths.append(PARENT/'protocol/protocol_seal23.json')
    paths.extend(PARENT/row['path'] for row in read_json(PARENT/'protocol/protocol_seal23.json')['files']
                 if Path(row['path']).suffix in {'.py','.json'} and not row['path'].startswith(('audit/','tests/','theory/','reproducibility/')))
    return {str(p.relative_to(BASE)):sha(p) for p in sorted(set(paths))}


def check_implementation_seal():
    p=RUN/'implementation_seal.json';seal=read_json(p)
    assert seal['schema']=='step24-implementation-seal-v1' and seal['passed']
    assert seal['protocol_sha256']==sha(PARENT/'PROTOCOL.json')
    assert seal['source_sha256']==source_map(),'Implementation source changed after seal'
    return seal


def check_fit_seal():
    implementation=check_implementation_seal();seal=read_json(RUN/'fit_seal.json')
    assert seal['schema']=='step24-fit-seal-v1' and seal['passed']
    assert seal['all_task_ids']==TASKS
    assert set(seal['tasks'])==set(TASKS),'Fit seal is missing a task record'
    assert seal['protocol_sha256']==implementation['protocol_sha256']
    assert seal['source_sha256']==implementation['source_sha256']
    assert seal['implementation_seal_sha256']==sha(RUN/'implementation_seal.json')
    for task,record in seal['tasks'].items():
        for rel,digest in record['files'].items():assert sha(RUN/task/rel)==digest,(task,rel)
    return seal


def exact_sum_dyadics(values):
    total=0; exponent=0
    for value in values:
        n,d=float(value).as_integer_ratio();k=d.bit_length()-1
        if k>exponent:total <<= (k-exponent);exponent=k
        total += n << (exponent-k)
    return Fraction(total,1<<exponent)


def exact_mse(y,prediction):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64)
    if y.shape!=p.shape or y.ndim!=1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Finite equal-shape scalar forecasts and labels required')
    total=0;exponent=0
    for a,b in zip(y,p):
        an,ad=float(a).as_integer_ratio();bn,bd=float(b).as_integer_ratio()
        ak=ad.bit_length()-1;bk=bd.bit_length()-1;k=max(ak,bk)
        n=(an<<(k-ak))-(bn<<(k-bk));term=n*n;k*=2
        if k>exponent:total <<= (k-exponent);exponent=k
        total += term << (exponent-k)
    return Fraction(total,len(y)*(1<<exponent))


def bracket(value):
    value=Fraction(value);nearest=float(value);represented=Fraction.from_float(nearest)
    return {'value':nearest,'lower':nearest if represented<=value else math.nextafter(nearest,-math.inf),
            'upper':nearest if represented>=value else math.nextafter(nearest,math.inf)}


def risk_record(y,p):
    value=exact_mse(y,p)
    return {'mse':float(value),'mse_exact':value,'mse_enclosure':bracket(value),
            'rmse':math.sqrt(float(value)), 'nominal_numpy_mse':float(np.mean((np.asarray(y)-np.asarray(p))**2)),
            'rows':len(y)}


def log(task,event,**fields):
    record={'utc':now(),'task':task,'event':event,**fields}
    print(json.dumps(jsonable(record),allow_nan=False),flush=True)


def fail_record(task,stage,exc):
    import traceback
    path=RUN/(task or '_global')/('FAILED_'+stage+'.json')
    write_json(path,{'passed':False,'utc':now(),'task':task,'stage':stage,
                    'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()},replace=True)
