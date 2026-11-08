"""Nested disjoint block witnesses, importing the immutable Step18 routines.

No data access or computation occurs on import. Every new optimizer starts
from zero. Embedded child witnesses are evaluated independently afterward.
Guarded floating values here are diagnostics; exact19 supplies official values.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PARENT_STUDY = BASE / 'step18_natural_decisions/study'
if str(PARENT_STUDY) not in sys.path:
    sys.path.insert(0, str(PARENT_STUDY))
import certificate18 as Q
import block_solver18 as S
if Path(Q.__file__).resolve() != PARENT_STUDY/'certificate18.py':
    raise ImportError('A different certificate18 source was already imported')
if Path(S.__file__).resolve() != PARENT_STUDY/'block_solver18.py':
    raise ImportError('A different block_solver18 source was already imported')

FAMILIES = ('consecutive', 'greedy_nearest')
BASE_BLOCK = 32
LEVELS = (32, 64, 128)
CHUNK_BLOCKS = 16
SOLVER = dict(max_iter=4000, tolerance=1e-7, check_every=20)


def source_files():
    return [Path(__file__).resolve(), Path(Q.__file__).resolve(), Path(S.__file__).resolve()]


def feature_slices(prefix):
    if prefix not in (2,6,14):
        raise ValueError('Unknown frozen feature prefix')
    return [slice(a,b) for a,b in ((0,2),(2,6),(6,14)) if b<=prefix]


def nested_partitions(features, family, *, base_size=BASE_BLOCK, levels=LEVELS):
    """Larger groups concatenate base groups; their greedy rule is not rerun."""
    x=np.asarray(features,dtype=float)
    if tuple(levels)!=tuple(sorted(set(levels))) or levels[0]!=base_size:
        raise ValueError('Nested levels must start at the declared base')
    if any(b%base_size or len(x)%b for b in levels):
        raise ValueError('Complete nested groups required')
    started=time.perf_counter()
    base=Q.make_partition(x,base_size,family)
    elapsed=time.perf_counter()-started
    partitions={b:np.ascontiguousarray(base.reshape(-1,b)) for b in levels}
    maps={}
    for j,b in enumerate(levels):
        previous=None if j==0 else levels[j-1]
        maps[b]=dict(base_group_ids=np.arange(len(base),dtype=np.int64).reshape(-1,b//base_size),
            child_block_ids=np.empty((len(x)//b,0),dtype=np.int64) if previous is None else
                            np.arange(len(x)//previous,dtype=np.int64).reshape(-1,b//previous),
            child_size=previous)
    return partitions,maps,elapsed


def edge_embedding_map(child_size,parent_size):
    if child_size<2 or parent_size%child_size:
        raise ValueError('Parent must contain complete children')
    ii,jj=np.triu_indices(parent_size,1)
    lookup=np.full((parent_size,parent_size),-1,dtype=np.int64)
    lookup[ii,jj]=np.arange(len(ii))
    ci,cj=np.triu_indices(child_size,1)
    return np.stack([lookup[ci+k*child_size,cj+k*child_size]
                     for k in range(parent_size//child_size)])


def embed_children(children,parent_size,child_size):
    """Children are (parent blocks, children per parent, child edges)."""
    children=np.asarray(children,dtype=float)
    mapping=edge_embedding_map(child_size,parent_size)
    if children.ndim!=3 or children.shape[1:]!=mapping.shape or not np.isfinite(children).all():
        raise ValueError('Child dual shape/finite mismatch')
    alpha=np.zeros((len(children),parent_size*(parent_size-1)//2),dtype=float)
    for j,index in enumerate(mapping):
        alpha[:,index]=children[:,j,:]
    return alpha,mapping


def nominal_signed_bracket(alpha,x,y,kappa,slices,primal,sse_upper):
    """Build the unchanged correction routine's input for an arbitrary witness.

    The reused parent optimizer primal is feasible for the full parent program.
    Concatenating child primals would generally fail new cross-child constraints.
    """
    _,distance=S._geometry(x,slices)
    count,b=y.shape
    ii,jj=np.triu_indices(b,1)
    divergence=np.zeros((count,b),dtype=float)
    for v in range(b):
        divergence[:,v]=np.sum(alpha[:,ii==v],axis=1)-np.sum(alpha[:,jj==v],axis=1)
    linear=np.sum(alpha*(y[:,ii]-y[:,jj]),axis=1)
    penalty=kappa*np.sum(abs(alpha)*distance[:,ii,jj],axis=1)
    quadratic=.25*np.sum(divergence*divergence,axis=1)
    raw=linear-penalty-quadratic
    guard=1e-11*(1+abs(linear)+abs(penalty)+abs(quadratic))
    lower=np.maximum(0.,raw-guard)
    violation=np.maximum(0.,np.max(abs(primal[:,ii]-primal[:,jj])-kappa*distance[:,ii,jj],axis=1))
    gap=(sse_upper-lower)/(b-1)
    return dict(block_size=b,signed_dual=alpha,primal=primal,
        sse_lower=lower,sse_upper=sse_upper,lower=lower/(b-1),upper=sse_upper/(b-1),
        dual_raw=raw,dual_guard=guard,gap=gap,iterations=np.zeros(count,dtype=np.int64),
        converged=gap<=SOLVER['tolerance'],feasibility_residual=violation)


def compute_chunk(features,targets,radii,indices,kappa,slices,*,
                  children=None,child_size=None,predictions=None):
    """Run every new QP and retain new/embedded/chosen duals and full diagnostics."""
    index=np.asarray(indices,dtype=np.int64)
    x,y,eps=features[index],targets[index],radii[index]
    count,b=index.shape
    started=time.perf_counter()
    raw=S.lipschitz_block_brackets(x,y,kappa,block_slices=slices,**SOLVER)
    solve_seconds=time.perf_counter()-started
    started=time.perf_counter()
    new=Q.correct_dual_bracket(raw,x,y,eps,slices,kappa,SOLVER['tolerance'])
    new_correction_seconds=time.perf_counter()-started
    started=time.perf_counter()
    if children is None:
        inherited=np.zeros_like(raw['signed_dual'])
        mapping=np.empty((0,0),dtype=np.int64)
        embedded_kind='zero'
    else:
        inherited,mapping=embed_children(children,b,child_size)
        embedded_kind='zero_extended_guarded_child_witnesses'
    embedded_raw=nominal_signed_bracket(inherited,x,y,kappa,slices,
                                       raw['primal'],raw['sse_upper'])
    embedding_seconds=time.perf_counter()-started
    started=time.perf_counter()
    embedded=Q.correct_dual_bracket(embedded_raw,x,y,eps,slices,kappa,SOLVER['tolerance'])
    embedded_correction_seconds=time.perf_counter()-started
    if not np.array_equal(new['distance_upper'],embedded['distance_upper']):
        raise ArithmeticError('Candidate correction changed parent distances')
    choose_embedded=embedded['sse_lower']>new['sse_lower']
    chosen_alpha=np.where(choose_embedded[:,None],embedded['signed_dual'],new['signed_dual'])
    chosen_lower=np.where(choose_embedded,embedded['sse_lower'],new['sse_lower'])
    chosen_gap=(new['sse_upper']-chosen_lower)/(b-1)
    constant=np.sum((y-y.mean(axis=1)[:,None])**2,axis=1)
    if np.any(chosen_lower>constant+1e-8) or np.any(chosen_lower<0):
        raise ArithmeticError('Selected witness violates the feasible constant SSE bound')
    arrays=dict(indices=index,nominal_distance=new['nominal_distance'],
        distance_upper=new['distance_upper'],feature_block_radius=eps,
        child_edge_to_parent_edge=mapping,
        chosen_source=choose_embedded.astype(np.int8),chosen_signed_dual=chosen_alpha,
        chosen_sse_lower=chosen_lower,chosen_sse_upper=new['sse_upper'],
        chosen_gap=chosen_gap,chosen_converged=chosen_gap<=SOLVER['tolerance'],
        constant_sse=constant,
        new_nominal_primal_feasibility_residual=raw['feasibility_residual'],
        embedded_nominal_primal_feasibility_residual=embedded_raw['feasibility_residual'])
    shared={'nominal_distance','distance_upper','feature_block_radius'}
    for name,candidate in [('new',new),('embedded',embedded)]:
        for key,value in candidate.items():
            if isinstance(value,np.ndarray) and key not in shared:
                # Both candidates use the same new-parent primal; keep its
                # actual array under each candidate for simple independent replay.
                arrays[name+'_'+key]=value
    for name in ['dual_linear','dual_penalty','dual_quadratic','dual_raw','dual_guard',
                 'nominal_sse_lower','sse_lower_uncertainty_adjustment']:
        arrays['chosen_'+name]=np.where(choose_embedded,embedded[name],new[name])
    if predictions is not None:
        p=np.asarray(predictions)[index];ii,jj=np.triu_indices(b,1)
        violation=np.maximum(0.,np.max(abs(p[:,ii]-p[:,jj])-kappa*new['distance_upper'][:,ii,jj],axis=1))
        sse=np.sum((y-p)**2,axis=1)
        if np.any(chosen_lower>sse+1e-8) or np.max(violation)>1e-8:
            raise ArithmeticError('Witness conflicts with the frozen fitted feasible predictor')
        arrays.update(fitted_sse=sse,fitted_relaxed_feasibility_residual=violation)
    if not all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in 'fc'):
        raise ArithmeticError('Nonfinite saved witness evidence')
    edges=b*(b-1)//2
    stats=dict(blocks=count,block_size=b,edges_per_block=edges,
        complete_undirected_edges=count*edges,signed_constraint_pairs=count*edges,
        one_sided_inequalities=2*count*edges,
        new_block_programs=count,new_solver_iteration_limits=int(np.sum(~raw['converged'])),
        chosen_iteration_gap_limits=int(np.sum(chosen_gap>SOLVER['tolerance'])),
        chosen_embedded_count=int(np.sum(choose_embedded)),chosen_new_count=int(np.sum(~choose_embedded)),
        maximum_chosen_gap=float(np.max(chosen_gap)),
        maximum_new_iterations=int(np.max(raw['iterations'])),
        maximum_nominal_primal_violation=float(np.max(raw['feasibility_residual'])),
        maximum_relaxed_primal_violation=float(np.max(new['feasibility_residual'])),
        new_optimization_seconds=solve_seconds,
        new_radius_correction_seconds=new_correction_seconds,
        child_embedding_seconds=embedding_seconds,
        embedded_radius_correction_seconds=embedded_correction_seconds,
        saved_array_bytes=sum(int(a.nbytes) for a in arrays.values()),
        new_signed_witness_bytes=int(raw['signed_dual'].nbytes),
        embedded_signed_witness_bytes=int(inherited.nbytes),
        chosen_signed_witness_bytes=int(chosen_alpha.nbytes),
        embedded_kind=embedded_kind,
        chosen_source_codes={'0':'new_optimizer','1':embedded_kind},
        chosen_tie_rule='new optimizer on equal guarded SSE',
        chosen_primal_source='new optimizer full-parent feasible primal',
        numerical_scope='Step18 guarded floating diagnostic, not official exact decision')
    return arrays,stats


def guarded_empirical_lower(sse_lower,population_size):
    total=Q._guarded_sum_lower(np.asarray(sse_lower))
    return 0. if total==0 else float(np.nextafter(total/population_size,0.))

