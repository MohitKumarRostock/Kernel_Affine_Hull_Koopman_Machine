"""Full finite-cohort Lipschitz class-risk brackets, conditional on stored z.

OSQP proposes numerical vectors only. Exact integer/rational witnesses establish
every reported bound. No files, source datasets, training arrays or test outcomes
are opened by this module. Diagnostic assignments use assessment labels and are
not trained forecasting heads.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_name]="1"
from fractions import Fraction as F
import hashlib
import json
from math import isqrt,lcm
import time
import warnings
import numpy as np
import scipy.sparse as sparse
import osqp
import finite_feature35 as ff

ASSURANCE=ff.ASSURANCE
BITS=80
GRID_BITS=40
GAP_MEAN=F(1,100000000)
SOLVER_SETTINGS={"max_iter":20000,"eps_abs":1e-8,"eps_rel":1e-8,
    "polishing":True,"polish_refine_iter":3,"check_termination":25,
    "scaling":10,"rho":0.1,"sigma":1e-6,"alpha":1.6,
    "adaptive_rho":True,"adaptive_rho_interval":50,"adaptive_rho_tolerance":5.0,
    "warm_starting":False,"scaled_termination":False,"time_limit":30.0,"verbose":False}


def canon(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):canon(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [canon(v) for v in value]
    return value


def canonical_bytes(value):
    return (json.dumps(canon(value),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()


def clipped(value):return min(F(1),max(F(0),value))


def quantize(value,bits=GRID_BITS):
    """Exact nearest integer on a dyadic grid, with ties to even."""
    value=F(value);whole,remainder=divmod(value.numerator*(1<<bits),value.denominator)
    if 2*remainder>value.denominator or (2*remainder==value.denominator and whole%2):whole+=1
    return whole


def distance_graph(z,*,bits=BITS,kappa=F(2)):
    z=np.asarray(z,dtype=np.float64)
    if z.ndim!=2 or min(z.shape)<1 or not np.isfinite(z).all():raise ValueError("Nonempty finite binary64 matrix required")
    qz=[[F.from_float(float(v)) for v in row] for row in z]
    exponent=max(v.denominator.bit_length()-1 for row in qz for v in row)
    integers=[[v.numerator << (exponent-(v.denominator.bit_length()-1)) for v in row] for row in qz]
    n=len(z);pairs=[];floors=[];exact=[];zero=0
    for i in range(n):
        for j in range(i+1,n):
            squared=sum((a-b)**2 for a,b in zip(integers[i],integers[j]))
            if exponent<=bits:
                scaled=squared << (2*(bits-exponent));floor=isqrt(scaled);is_exact=floor*floor==scaled
            else:
                floor=isqrt(squared) >> (exponent-bits)
                is_exact=(floor*floor << (2*(exponent-bits)))==squared
            index=len(pairs);pairs.append((i,j));floors.append(floor)
            if is_exact:exact.append(index)
            if squared==0:zero+=1
    kap=F(kappa)
    if kap<0:raise ValueError("Sensitivity must be nonnegative")
    scale=(1<<bits)*kap.denominator
    weights=[[0]*n for _ in range(n)]
    active=[]
    for index,((i,j),floor) in enumerate(zip(pairs,floors)):
        weight=kap.numerator*floor;weights[i][j]=weights[j][i]=weight
        if weight<scale:active.append(index)
    receipt={"schema":"step37-complete-exact-distance-graph-v1","bits":bits,
        "common_binary_exponent":exponent,"pair_order":"lexicographic_i_then_j_i_lt_j",
        "pair_count":len(pairs),"floor_scaled":[str(v) for v in floors],"exact_pair_indices":exact,
        "nonredundant_pair_indices":active,"zero_distance_pair_count":zero,
        "constraint_integer_scale":str(scale),"coordinates_interpreted_as_exact_binary64":True,
        "omitted_qp_pairs_are_redundant_given_output_interval":True}
    return receipt,pairs,floors,set(exact),weights,scale


def dense_dijkstra(initial,weights):
    """Multi-source shortest paths with nonnegative exact integer edge weights."""
    n=len(initial);best=list(initial);done=[False]*n;order=[]
    for _ in range(n):
        i=min((i for i in range(n) if not done[i]),key=lambda i:(best[i],i))
        done[i]=True;order.append(i)
        for j in range(n):
            if not done[j]:
                candidate=best[i]+weights[i][j]
                if candidate<best[j]:best[j]=candidate
    return best,order


def verify_values(values,targets,pairs,weights,scale):
    if not all(F(0)<=v<=F(1) for v in values):raise ArithmeticError("Primal violates output range")
    den=lcm(*(v.denominator for v in values))
    nums=[v.numerator*(den//v.denominator) for v in values]
    for i,j in pairs:
        if abs(nums[i]-nums[j])*scale>weights[i][j]*den:
            raise ArithmeticError("Primal violates a conservative complete-graph constraint")
    sse=sum(((v-y)**2 for v,y in zip(values,targets)),F(0))
    return sse


def repair_proposal(raw,targets,weights,scale,pairs,source):
    grid=[quantize(clipped(F(v))) for v in raw]
    grid_scale=1<<GRID_BITS
    if scale%grid_scale:raise ArithmeticError("Constraint grid must contain proposal grid")
    q=[v*(scale//grid_scale) for v in grid]
    low,low_order=dense_dijkstra(q,weights)
    complement,high_order=dense_dijkstra([scale-v for v in q],weights)
    high=[scale-v for v in complement]
    if not all(0<=a<=seed<=b<=scale for a,seed,b in zip(low,q,high)):
        raise ArithmeticError("Envelope order or range failed")
    lo=[F(v,scale) for v in low];delta=[F(b-a,scale) for a,b in zip(low,high)]
    denominator=sum((d*d for d in delta),F(0))
    theta=F(0) if not denominator else clipped(sum((d*(y-a) for d,y,a in zip(delta,targets,lo)),F(0))/denominator)
    values=[a+theta*d for a,d in zip(lo,delta)]
    sse=verify_values(values,targets,pairs,weights,scale)
    return {"source":source,"proposal_grid_bits":GRID_BITS,"proposal_grid_integers":grid,
            "envelope_scale":str(scale),"minorant_grid":[str(v) for v in low],
            "majorant_grid":[str(v) for v in high],"minorant_visit_order":low_order,
            "majorant_visit_order":high_order,"theta":theta,"values":values,"sse":sse,"mean":sse/len(values)}


def numerical_proposal(targets,graph,pairs,floors,*,settings):
    n=len(targets);active=graph["nonredundant_pair_indices"];scale=int(graph["constraint_integer_scale"])
    # The primary κ=2 case is encoded in the exact integer edge weights.
    kap_numerator=int(graph["kappa_numerator"])
    rows=[];columns=[];entries=[]
    for k,index in enumerate(active):
        i,j=pairs[index];rows.extend((k,k));columns.extend((i,j));entries.extend((1.,-1.))
    incidence=sparse.csc_matrix((entries,(rows,columns)),shape=(len(active),n))
    A=sparse.vstack((sparse.eye(n,format="csc"),incidence),format="csc")
    radii=np.asarray([float(F(kap_numerator*floors[k],scale)) for k in active])
    l=np.concatenate((np.zeros(n),-radii));u=np.concatenate((np.ones(n),radii))
    q=-2*np.asarray([float(y) for y in targets],dtype=np.float64)
    record={"solver":"OSQP","version":osqp.__version__,"settings":dict(settings),
        "objective":"unhalved_SSE_without_constant_P2I_q_minus2y","box_duals_used_in_certificate":False,
        "raw_primal_binary64":None,"raw_edge_dual_binary64":None,"status":None,
        "status_val":None,"iterations":0,"fallback_reason":None,"solver_warnings":[],
        "constraint_lower_bounds_rounded_to_binary64_for_proposal_only":True}
    started=time.perf_counter()
    try:
        model=osqp.OSQP(algebra="builtin")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.setup(P=2*sparse.eye(n,format="csc"),q=q,A=A,l=l,u=u,**settings)
            solution=model.solve(raise_error=False)
            record["solver_warnings"]=[str(w.message) for w in caught]
        record.update(status=str(solution.info.status),status_val=int(solution.info.status_val),iterations=int(solution.info.iter))
        if solution.x is not None and len(solution.x)==n and np.isfinite(solution.x).all():
            record["raw_primal_binary64"]=[str(F.from_float(float(v))) for v in solution.x]
        if solution.y is not None and len(solution.y)==n+len(active) and np.isfinite(solution.y[n:]).all():
            record["raw_edge_dual_binary64"]=[str(F.from_float(float(v))) for v in solution.y[n:]]
        if record["raw_primal_binary64"] is None or record["raw_edge_dual_binary64"] is None:
            record["fallback_reason"]="NONFINITE_OR_MISSING_NUMERICAL_PROPOSAL"
    except Exception as exc:
        record["fallback_reason"]="NUMERICAL_SOLVER_EXCEPTION: "+type(exc).__name__+": "+str(exc)
    record["proposal_wall_seconds"]=time.perf_counter()-started
    return record


def weak_dual(targets,graph,pairs,floors,exact,numerical,kappa):
    n=len(targets);s=[F(0)]*n;terms=[];penalty=F(0);multipliers=[]
    raw=numerical["raw_edge_dual_binary64"]
    if raw is not None:
        for index,value in zip(graph["nonredundant_pair_indices"],raw):
            lam=F(quantize(F(value)),1<<GRID_BITS)
            if not lam:continue
            i,j=pairs[index];s[i]+=lam;s[j]-=lam
            radius=F(kappa)*F(floors[index]+(index not in exact),1<<graph["bits"])
            penalty+=abs(lam)*radius
            multipliers.append({"pair_index":index,"value":str(lam)})
    minimizers=[clipped(y-si/2) for y,si in zip(targets,s)]
    terms=[(v-y)**2+si*v for v,y,si in zip(minimizers,targets,s)]
    value=sum(terms,F(0))-penalty
    return {"grid_bits":GRID_BITS,"signed_multipliers":multipliers,"incidence":s,
            "box_minimizers":minimizers,"coordinate_terms":terms,"penalty":penalty,"sse":value,"mean":value/n,
            "distance_endpoint":"upper","valid_for_any_finite_signed_multipliers":True}


def diagnose(z,y,*,ids,feature_id,class_id,epsilon,heads,task="artificial",map_name="artificial",
             kappa="2",settings=None,input_binding=None):
    started=time.perf_counter();settings=dict(SOLVER_SETTINGS if settings is None else settings)
    targets=list(map(F,y));epsilon=F(epsilon);kap=F(kappa)
    z=np.asarray(z,dtype=np.float64)
    if len(targets)!=len(z) or len(ids)!=len(z) or len(set(ids))!=len(ids):raise ValueError("Cohort roster differs")
    if class_id!=ff.class_identity(feature_id,kap,("0","1")):raise ValueError("Wrong class identity")
    if epsilon<0 or not heads:raise ValueError("Nonnegative fixed tolerance and inherited heads required")
    graph,pairs,floors,exact,weights,scale=distance_graph(z,kappa=kap)
    graph["kappa_numerator"]=kap.numerator
    graph_seconds=time.perf_counter()-started
    floor_values=[(clipped(v)-v)**2 for v in targets]
    floor_sse=sum(floor_values,F(0))
    numerical=numerical_proposal(targets,graph,pairs,floors,settings=settings)
    repair_start=time.perf_counter()
    candidates=[repair_proposal(targets,targets,weights,scale,pairs,"clipped_target_envelope")]
    if numerical["raw_primal_binary64"] is not None:
        candidates.append(repair_proposal(numerical["raw_primal_binary64"],targets,weights,scale,pairs,"solver_envelope"))
    constant=clipped(sum(targets,F(0))/len(targets));constant_values=[constant]*len(targets)
    constant_sse=verify_values(constant_values,targets,pairs,weights,scale)
    candidates.append({"source":"clipped_mean_constant","value":constant,"values":constant_values,
                       "sse":constant_sse,"mean":constant_sse/len(targets)})
    chosen=min(candidates,key=lambda c:(c["sse"],c["source"]))
    dual=weak_dual(targets,graph,pairs,floors,exact,numerical,kap)
    lower=max(floor_sse/len(targets),dual["mean"])
    best_head=min(heads,key=lambda h:(F(h["decision_risk"]["upper"]),h["candidate_id"]))
    head_upper=F(best_head["decision_risk"]["upper"])
    new_upper=chosen["mean"]
    upper=min(new_upper,head_upper)
    upper_source=chosen["source"] if new_upper<=head_upper else "inherited_head:"+best_head["candidate_id"]
    if not F(0)<=lower<=upper:raise ArithmeticError("Certified class-risk bounds conflict")
    status="ADEQUATE" if upper<=epsilon else "INADEQUATE" if lower>epsilon else "UNRESOLVED"
    inherited=[]
    for head in heads:
        lo,hi=F(head["decision_risk"]["lower"]),F(head["decision_risk"]["upper"])
        if not F(0)<=lo<=hi:raise ValueError("Invalid inherited risk enclosure")
        if hi<lower:raise ArithmeticError("Inherited admissible head contradicts class lower bound")
        inherited.append({"candidate_id":head["candidate_id"],"head_id":head["head_id"],
            "recipe":head["recipe"],"decision_risk":dict(head["decision_risk"]),
            "decision_prediction_sha256":head["decision_prediction_sha256"],
            "excess_risk":{"lower":max(F(0),lo-upper),"upper":max(F(0),hi-lower)},
            "verified_eligible":hi<=epsilon,"verified_ineligible":lo>epsilon,
            "positive_certified_excess":lo>upper})
    return canon({"schema":"step37-full-graph-class-diagnostic-v1","task":task,"map":map_name,
        "assurance":ASSURANCE,"feature_id":feature_id,"class_id":class_id,"n":len(targets),"dimension":int(z.shape[1]),
        "ids":list(ids),"epsilon":epsilon,"kappa":kap,"output_interval":["0","1"],
        "input_binding":input_binding or {},"targets_digest":digest(targets),"distance":graph,
        "numerical":numerical,"primal_candidates":candidates,"selected_primal_source":chosen["source"],
        "new_primal_mean":new_upper,"dual":dual,
        "range_floor":{"singleton_sse":floor_values,"sse":floor_sse,"mean":floor_sse/len(targets)},
        "class_risk":{"lower":lower,"upper":upper,"width":upper-lower,"upper_source":upper_source,
                      "lower_source":"rounded_edge_dual" if dual["mean"]>=floor_sse/len(targets) else "common_range_floor"},
        "diagnosis":status,"gap_target_mean":GAP_MEAN,"gap_resolved":upper-lower<=GAP_MEAN,
        "inherited_heads":inherited,"existing_eligible_head_count":sum(h["verified_eligible"] for h in inherited),
        "new_finite_cohort_adequacy":status=="ADEQUATE" and not any(h["verified_eligible"] for h in inherited),
        "timing":{"exact_graph_seconds":graph_seconds,"numerical_proposal_seconds":numerical["proposal_wall_seconds"],
                  "exact_witness_and_summary_seconds":time.perf_counter()-repair_start,"total_seconds":time.perf_counter()-started},
        "scope":{"diagnostic_uses_decision_labels":True,"assignment_is_a_trained_prediction_head":False,
                 "new_forecast_predictions_computed":0,"native_transform_error_enclosed":False,
                 "population_risk_claim":False,"future_response_coverage_claim":False,
                 "causal_optimizer_failure_claim":False,"reserved_patients_accessed":0},
        "extension_justification":"Every verified complete-graph assignment extends to a globally kappa-Lipschitz real-valued decoder by McShane/Whitney cones; clipping to I preserves the range, interpolation and Lipschitz bound."})
