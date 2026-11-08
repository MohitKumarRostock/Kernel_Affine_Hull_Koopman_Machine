"""Normalized power soft state: enclosure plus binary64 simplex approximation.
For every block and coordinate the bounds enclose the exact real power state
for ANY scores in the supplied box. Public binary64 nominal values normalize
weights of midpoint scores and round once from 160-bit arithmetic. Their
sum is one to roundoff, not an assertion of exact floating-point summation.
"""
from pathlib import Path
import ctypes as C,subprocess,hashlib,tempfile,os
import numpy as np
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
SCORE_TO_P_ATOL=1e-9
def radius(p,l,h):
    v=np.maximum(np.abs(p-l),np.abs(h-p))
    return np.where(v==0,0.,np.nextafter(v,np.inf))
@lru_cache(None)
def _lib():
    src=ROOT/'softsrc/power_box.cpp'
    so=Path(tempfile.gettempdir())/('geometry_step11_power_box_'+hashlib.sha256(src.read_bytes()).hexdigest()[:24]+'.so')
    if not so.exists():
        tmp=so.with_suffix('.tmp.'+str(os.getpid()))
        subprocess.run(['g++','-O2','-std=c++17','-fPIC','-shared',str(src),'-Wl,-l:libmpfr.so.6','-o',str(tmp)],check=True)
        os.replace(tmp,so)
    lib=C.CDLL(str(so))
    pi=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
    pa=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
    lib.power_box.argtypes=[C.c_int,C.c_int,C.c_int,pi,pa,pa,C.c_double,C.c_double,pa,pa,pa]
    lib.power_box.restype=C.c_int;lib.power_version.restype=C.c_char_p
    return lib
def power_state(lo,hi,blocks,omega,tau=1e-6,*,atol=SCORE_TO_P_ATOL):
    """Returns (nominal,lower,upper); enforces all coordinates and simplex residual.
    Inputs are binary64 outward score endpoints, physical model tau & omega.
    None disables only the width gate, permitting interval-validation fixtures.
    """
    l=np.ascontiguousarray(lo,dtype=np.float64);h=np.ascontiguousarray(hi,dtype=np.float64)
    if any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 1<=k<=256 for k in blocks):raise ValueError('invalid block sizes')
    blocks=np.asarray(blocks,dtype=np.int32)
    if l.ndim!=2 or l.shape!=h.shape or not len(l) or l.shape[1]!=int(blocks.sum()) or l.size>=2**31:raise ValueError('invalid score enclosure layout')
    if atol is not None and (not np.isfinite(atol) or atol<0):raise ValueError('invalid absolute budget')
    out=[np.empty_like(l) for i in range(3)]
    code=_lib().power_box(*l.shape,len(blocks),blocks,l,h,float(omega),float(tau),*out)
    if code:raise ValueError(f'power_box domain status {code}')
    p,L,H=out
    if not (np.isfinite(out).all() and np.all((L<=H)&(p>=0)&(p<=1))): raise ArithmeticError('nonfinite or invalid power state')
    i=0
    for k in blocks:
        if np.max(abs(p[:,i:i+k].sum(axis=1)-1))>4*np.finfo(float).eps*k:raise ArithmeticError('nominal probability normalization')
        i+=k
    if atol is not None and np.max(radius(p,L,H))>atol:raise ArithmeticError('soft-state absolute output budget not met')
    return tuple(out)
