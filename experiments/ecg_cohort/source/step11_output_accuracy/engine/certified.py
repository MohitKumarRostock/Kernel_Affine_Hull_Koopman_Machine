"""Step 11 validated forward KAHM scores/coordinates (MPFI+MPFR on Linux LP64).

The target treats all stored binary32/binary64 values as exact, uses the
native ideal-real scaling and Gaussian ratio on s!=0, the native zero-norm
cosine convention, encoder minima, and canonical mathematical Helmert blocks
(common last) when Q=None. No denominator floor or dropped state. Forward only.
See README.md for assumptions, supported inputs, returned intervals and tests.
"""
from pathlib import Path
import ctypes as ct
import hashlib
import os
import subprocess
import tempfile
import numpy as np

SRC=Path(__file__).with_name('interval.cpp')
DS=ct.POINTER(ct.c_double); IS=ct.POINTER(ct.c_int)
def _dp(x):return x.ctypes.data_as(DS)
def _ip(x):return x.ctypes.data_as(IS)
def _load():
    # A per-process private build excludes stale/colliding shared binaries.
    cache=tempfile.mkdtemp(prefix='kahm_step11_')
    so=Path(cache)/'interval.so'
    subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared','-o',str(so),str(SRC),'-l:libmpfi.so.0','-l:libmpfr.so.6','-lgmp'],check=True)
    h=ct.CDLL(str(so))
    so.unlink();Path(cache).rmdir()
    h.k11_new.argtypes=[ct.c_int];h.k11_new.restype=ct.c_void_p
    h.k11_free.argtypes=[ct.c_void_p];h.k11_free.restype=None
    h.k11_add.argtypes=[ct.c_void_p]+[ct.c_int]*4+[DS]*7;h.k11_add.restype=ct.c_int
    h.k11_eval.argtypes=[ct.c_void_p]+[ct.c_int]*4+[IS,IS,DS,DS,ct.c_double,DS,DS,IS,DS,DS,DS,DS];h.k11_eval.restype=ct.c_int
    h.k11_mpfi.restype=h.k11_mpfr.restype=ct.c_char_p
    return h
_L=None
def lib():
    global _L
    if _L is None:_L=_load()
    return _L
def metadata():
    h=lib();return {'MPFI':h.k11_mpfi().decode(),'MPFR':h.k11_mpfr().decode(),'cpp_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'public_format':'binary64 lower RNDD / upper RNDU','scope':'forward scores and canonical coordinates, no derivative or predictive-head certificate'}
def f64(a,nd=None):
    # Native fitted arrays and physical queries are binary32/binary64; small
    # integer literals useful for manufactured tests must also be exact.
    x=np.asarray(a)
    if x.dtype.kind=='f' and x.dtype.itemsize<=8:pass
    elif x.dtype.kind in 'iu' and np.all(np.abs(x.astype(np.longdouble))<=2**53):pass
    else:raise ValueError('Exact finite binary64-compatible input required')
    r=np.ascontiguousarray(x,dtype=np.float64)
    if not np.isfinite(r).all() or (nd is not None and r.ndim!=nd):raise ValueError('Invalid array rank or nonfinite input.')
    return r
def integer(x):
    if not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_)):raise ValueError('Integer size or precision required')
    return int(x)
def _budget(x):
    if not isinstance(x,(float,int,np.float64,np.float32)) or not np.isfinite(x) or x<0:raise ValueError('Invalid tolerance')
    return float(x)
class Prepared:
    def __init__(self,AEs,states,blocks,scale,offset,Q=None,*,bits=80,tau=1e-6,scaling='affine',constant_mask=None):
        bits=integer(bits);self._h=None
        if not 53<=bits<=4096:raise ValueError('Unsupported precision')
        self.bits=bits; self.tau=float(f64([tau],1)[0]);self.sc=f64(scale,1).copy();self.of=f64(offset,1).copy();self.d=len(self.sc)
        self.mask=np.zeros(self.d,bool) if constant_mask is None else np.asarray(constant_mask,bool)
        if self.mask.shape!=(self.d,) or self.of.shape!=(self.d,) or not 1<=self.d<=256 or self.tau<=0 or scaling not in ('affine','standard'):raise ValueError('Invalid scalers, tau, dimension or mode')
        if scaling=='standard' and np.any(self.sc[~self.mask]<=0):raise ValueError('Invalid scaler divisor')
        self.sc[self.mask]=1.;self.of[self.mask]=0.
        sz=[integer(v) for v in blocks];self.bk=np.asarray(sz,np.int32)
        self.ns=len(states);self.ne=len(AEs)
        if not sz or min(sz)<1 or sum(sz)!=self.ns or not 1<=self.ns<=self.ne<=4096:raise ValueError('Invalid state layout')
        idx=[];self.grp=np.full(self.ne,-1,np.int32)
        for j,g in enumerate(states):
            if not len(g):raise ValueError('Empty state')
            for k0 in g:
                k=integer(k0)
                if not 0<=k<self.ne or self.grp[k]!=-1:raise ValueError('Encoder assignments must be a partition')
                self.grp[k]=j
        if np.any(self.grp<0):raise ValueError('Encoder unassigned')
        self.Q=None
        if Q is not None:
            qs=[f64(q,2) for q in Q]
            if len(qs)!=len(sz) or any(q.shape!=(k,k) for k,q in zip(sz,qs)):raise ValueError('Invalid Q')
            self.Q=np.concatenate([q.ravel() for q in qs])
        self.lib=lib();self._h=self.lib.k11_new(bits)
        if not self._h:raise MemoryError('Bank allocation failed')
        try:
            for ae in AEs:
                if str(ae.get('kerneltype','Gaussian')).lower()!='gaussian':raise ValueError('Only Gaussian supported')
                pc,refs,S,B,Y=[f64(ae[k],2) for k in ('PC','y_data_n_subspace','sqrt_weights_matrix','B','y_data')]
                q,n=refs.shape;d=self.d
                if not 1<=q<=d or not 1<=n<=1024 or pc.shape!=(d,q) or S.shape!=(q,q) or B.shape!=(n,n) or Y.shape!=(d,n):raise ValueError('Invalid parameter shapes')
                if self.lib.k11_add(self._h,d,q,n,scaling=='standard',*map(_dp,(pc,refs,S,B,Y,self.sc,self.of))):raise RuntimeError('Encoder preparation failed')
        except BaseException:
            self.close();raise
    def close(self):
        if self._h:self.lib.k11_free(self._h);self._h=None
    def __del__(self):
        if getattr(self,'_h',None):self.close()
    def run(self,X):
        X=f64(X,2)
        if not self._h or X.shape[1]!=self.d or max(1,len(X))*self.ne>2**26:raise ValueError('Invalid query or exceeded batch resource limit')
        if self.mask.any():X=X.copy();X[:,self.mask]=0.
        n=len(X);e=(n,self.ne);s=(n,self.ns)
        d={k:np.full(e if k.startswith('encoder') else s,np.nan,np.float64) for k in ('encoder_lo','encoder_hi','score_lo','score_hi','F_lo','F_hi')}
        d['encoder_status']=np.full(e,-1,np.int32)
        ret=self.lib.k11_eval(self._h,n,self.d,self.ns,len(self.bk),_ip(self.grp),_ip(self.bk),_dp(X),None if self.Q is None else _dp(self.Q),self.tau,_dp(d['encoder_lo']),_dp(d['encoder_hi']),_ip(d['encoder_status']),_dp(d['score_lo']),_dp(d['score_hi']),_dp(d['F_lo']),_dp(d['F_hi']))
        if ret:raise RuntimeError('Native interval evaluation failed')
        return d
def evaluate(AEs,states,blocks,X,scale,offset,Q=None,*,bits=80,tau=1e-6,scaling='affine',constant_mask=None):
    p=Prepared(AEs,states,blocks,scale,offset,Q,bits=bits,tau=tau,scaling=scaling,constant_mask=constant_mask)
    try:return p.run(X)
    finally:p.close()
def _miderr(lo,hi):
    with np.errstate(invalid='ignore',under='ignore',over='ignore'):
        m=lo+(hi-lo)*0.5
        err=np.maximum(m-lo,hi-m)
        err=np.where(err==0,0.,np.nextafter(err,np.inf))
    return m,err
class Adaptive:
    """Private state for one fitted bank/configuration, to amortize preparation across calls."""
    def __init__(self,AEs,states,blocks,scale,offset,Q=None,*,precision=(80,128,192),score_atol=1e-10,F_atol=1e-9,tau=1e-6,scaling='affine',constant_mask=None):
        self.p=tuple(integer(v) for v in precision)
        if not self.p or len(self.p)>16 or list(self.p)!=sorted(set(self.p)) or self.p[0]<53 or self.p[-1]>4096:raise ValueError('Invalid declared precision schedule')
        self.score_atol=_budget(score_atol);self.F_atol=_budget(F_atol)
        self.args=(AEs,states,blocks,scale,offset,Q);self.kw={'scaling':scaling,'constant_mask':constant_mask,'tau':tau}
        self.engines={}
    def close(self):
        for b in self.engines.values():b.close()
        self.engines={}
    def run(self,X):
        X=f64(X,2);n=len(X);idx=np.arange(n);finished=np.zeros(n,bool);pr=np.zeros(n,np.int32);attempts=np.zeros(n,np.int32)
        traces=[]
        for bits in self.p:
            if bits not in self.engines:self.engines[bits]=Prepared(*self.args,bits=bits,**self.kw)
            r=self.engines[bits].run(X[idx])
            m,e=_miderr(r['score_lo'],r['score_hi']);f,fe=_miderr(r['F_lo'],r['F_hi'])
            r.update(scores=m,F=f,score_error_bound=e,F_error_bound=fe)
            accept=(np.all(r['encoder_status']==0,axis=1)&np.all(np.isfinite(e)&(e<=self.score_atol),axis=1)&np.all(np.isfinite(fe)&(fe<=self.F_atol),axis=1))
            if bits==self.p[0]:out={k:v.copy() for k,v in r.items()}
            else:
                for k,v in r.items():out[k][idx]=v
            attempts[idx]+=1;pr[idx]=bits;finished[idx[accept]]=True
            traces.append({'bits':bits,'evaluated_rows':len(idx),'accepted_rows':int(np.sum(accept))})
            # A point proof of exactly s=0 is terminal, not rescued by precision.
            exact=np.any(r['encoder_status']==2,axis=1)
            idx=idx[~(accept|exact)]
            if not len(idx):break
        out['scores'][~finished]=np.nan;out['F'][~finished]=np.nan
        out.update(accepted=finished,precision_bits=pr,attempts=attempts,trace=traces)
        return out
def adaptive(AEs,states,blocks,X,scale,offset,Q=None,**kwargs):
    a=Adaptive(AEs,states,blocks,scale,offset,Q,**kwargs)
    try:return a.run(X)
    finally:a.close()
def bank_adapter(bank,prefix=None,**kwargs):
    blocks=[];states=[];AEs=[];total=0
    for k,b in zip(bank.block_sizes,bank.blocks,strict=True):
        k=integer(k)
        if prefix is not None and total>=prefix:break
        if len(b['classifier'])!=k:raise ValueError('Bank state group mismatch')
        blocks.append(k)
        for gg in b['classifier']:
            gg=gg if isinstance(gg,(tuple,list)) else (gg,)
            states.append(tuple(range(len(AEs),len(AEs)+len(gg))));AEs.extend(gg)
        total+=k
    if prefix is not None and prefix!=total:raise ValueError('Prefix must end at a complete hierarchy block')
    return Adaptive(AEs,states,blocks,bank.scaler.divisor,bank.scaler.offset,None,tau=bank.tau,scaling='standard',constant_mask=bank.scaler.constant_mask,**kwargs)
def certified_bank(bank,X,prefix=None,**kwargs):
    a=bank_adapter(bank,prefix,**kwargs)
    try:return a.run(X)
    finally:a.close()
