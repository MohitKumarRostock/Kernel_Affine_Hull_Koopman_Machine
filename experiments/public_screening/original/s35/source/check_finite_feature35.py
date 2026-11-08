"""Artificial-only preflight and direct final-witness checks for Step35.

The witness check functions below independently reconstruct exact algebra. They
do not call the producer's distance, dual, primal, or evaluation routines. The
preflight driver imports the producer only to construct artificial test cases.
No public dataset, native model, feature transform, or scientific runtime is
loaded. A successful preflight establishes component behavior, not study merit.
"""
from __future__ import annotations

import argparse
import builtins
from copy import deepcopy
from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
from math import isqrt
from pathlib import Path
import random
import time


def _f(value):
    if type(value) is float:
        return F.from_float(value)
    return F(value)


def _canon(value):
    if isinstance(value,F):
        return str(value)
    if type(value) is float:
        return str(F.from_float(value))
    if isinstance(value,dict):
        return {str(k):_canon(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_canon(v) for v in value]
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_canon(value),sort_keys=True,separators=(",",":"),
        ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def _require(condition,message):
    if not condition:
        raise AssertionError(message)


def _box(value):
    a,b=_f(value["lower"]),_f(value["upper"])
    _require(a<=b,"reversed interval")
    _require(_f(value["width"])==b-a,"interval width differs")
    return a,b


def _clip(value,low,high):
    return max(low,min(high,value))


def _squared(x,y):
    _require(len(x)==len(y),"dimension differs")
    return sum(((a-b)**2 for a,b in zip(x,y)),F(0))


def _distance(record,x,y,bits,*,require_tight=True):
    squared=_squared(x,y)
    low,high=_box(record["distance"])
    _require(low>=0 and low*low<=squared<=high*high,"distance does not enclose Euclidean value")
    _require(_f(record["squared_exact"])==squared,"squared-distance witness differs")
    if require_tight:
        scale=1<<bits
        root=isqrt((squared.numerator<<(2*bits))//squared.denominator)
        a=F(root,scale)
        b=a if a*a==squared else F(root+1,scale)
        _require((low,high)==(a,b),"distance is not the declared integer-root enclosure")
        _require(record["scaled_sqrt_floor"]==root,"integer-root witness differs")
    return low,high


def verify_head(record):
    head=record["head"] if "head" in record else record
    _require(head["schema"]=="step35-stored-feature-head-v1","head schema differs")
    _require(head["assurance"]=="CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES","head assurance differs")
    head_id="head35:"+_hash(head)
    if "head" in record:
        _require(record["head_id"]==head_id,"head content identity differs")
    k=_f(head["kappa"]);low,high=map(_f,head["output_interval"])
    _require(k>=0 and low<high,"invalid class")
    descriptor={"schema":"step35-stored-feature-decoder-class-v1","feature_id":head["feature_id"],
        "metric":"Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class":"all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa":str(k),"output_interval":[str(low),str(high)],"assurance":head["assurance"]}
    _require(head["class_id"]=="class35:"+_hash(descriptor),"head class content identity differs")
    if head["kind"]=="clipped_affine":
        beta=list(map(_f,head["coefficients"]))
        _require(len(beta)==head["dimension"],"affine dimension differs")
        _require(sum((v*v for v in beta),F(0))<=k*k,"affine norm exceeds sensitivity")
        _f(head["intercept"])
    elif head["kind"]=="symmetric_lipschitz_extension":
        z=[list(map(_f,row)) for row in head["anchor_features"]]
        values=list(map(_f,head["anchor_values"]))
        _require(1<=len(z)<=32 and len(z)==len(values),"extension roster differs")
        _require(all(len(row)==head["dimension"] for row in z),"extension dimension differs")
        _require(all(low<=v<=high for v in values),"extension anchor outside range")
        _require(len(set(head["anchor_ids"]))==len(z),"extension anchor identifiers overlap")
    else:
        raise AssertionError("unknown head formula")
    return {"head_id":head_id,"class_id":head["class_id"],"kind":head["kind"],
        "status":"PASS_EXACT_GLOBAL_FEATURE_CLASS_MEMBERSHIP"}


def verify_qp_receipt(record,*,require_tight_distances=True):
    _require(record["schema"]=="step35-stored-feature-finite-qp-v1","QP schema differs")
    _require(record["kind"] in ("class","training"),"QP purpose differs")
    _require(record["assurance"]=="CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES","QP assurance differs")
    z=[list(map(_f,row)) for row in record["features_exact"]]
    y=list(map(_f,record["targets_exact"]));n=len(z)
    _require(1<=n<=32 and n==record["n"]==len(y),"QP row count differs")
    _require(len(set(record["query_ids"]))==n,"QP identifiers overlap")
    _require(all(len(row)==record["dimension"] for row in z),"QP feature dimension differs")
    _require(record["feature_digest"]==_hash(z) and record["target_digest"]==_hash(y),"QP data digest differs")
    k=_f(record["kappa"]);low,high=map(_f,record["output_interval"])
    _require(k>=0 and low<high,"QP class invalid")
    table=record["distance_table"];pairs=list(combinations(range(n),2));distances={}
    _require(len(table["pairs"])==len(pairs),"QP distance roster is incomplete")
    _require(table["query_ids"]==record["query_ids"] and table["features_exact"]==record["features_exact"],"QP distance source differs")
    _require(table["feature_digest"]==_hash(z),"QP distance feature digest differs")
    for (i,j),pair in zip(pairs,table["pairs"]):
        _require(pair["indices"]==[i,j],"QP distance order differs")
        distances[i,j]=_distance(pair,z[i],z[j],table["bits"],require_tight=require_tight_distances)
    primal=record["primal"];v=list(map(_f,primal["values"]))
    _require(len(v)==n and all(low<=a<=high for a in v),"primal range differs")
    for (i,j),(a,b) in distances.items():
        _require(abs(v[i]-v[j])<=k*a,"primal lower-distance constraint failed")
    primal_sse=sum(((a-b)**2 for a,b in zip(v,y)),F(0))
    _require(_f(primal["sse"])==primal_sse and _f(primal["mean"])==primal_sse/n,"primal loss differs")
    dual=record["dual"];s=[F(0)]*n;penalty=F(0);seen=set()
    endpoint=0 if record["kind"]=="training" else 1
    _require(dual["distance_endpoint"]==("lower" if endpoint==0 else "upper"),"dual distance direction differs")
    for edge in dual["edges"]:
        i,j=edge["indices"];pair=(i,j)
        _require(pair in distances and pair not in seen,"dual edge unknown or repeated")
        seen.add(pair);lam=_f(edge["lambda"])
        d=distances[pair][endpoint]
        _require(_f(edge["distance_endpoint"])==d and _f(edge["constraint_radius"])==k*d,"dual edge endpoint differs")
        s[i]+=lam;s[j]-=lam;penalty+=k*abs(lam)*d
    minimizers=[_clip(a-b/2,low,high) for a,b in zip(y,s)]
    terms=[(a-b)**2+c*a for a,b,c in zip(minimizers,y,s)]
    raw=sum(terms,F(0))-penalty;dual_sse=max(F(0),raw)
    _require(list(map(_f,dual["incidence"]))==s,"dual incidence differs")
    _require(list(map(_f,dual["box_minimizers"]))==minimizers,"dual boxed minimizers differ")
    _require(list(map(_f,dual["coordinate_minima"]))==terms,"dual box minimum values differ")
    _require(_f(dual["penalty"])==penalty and _f(dual["unclamped_sse"])==raw,"dual objective terms differ")
    _require(_f(dual["sse"])==dual_sse and _f(dual["mean"])==dual_sse/n,"dual loss differs")
    _require(0<=dual_sse<=primal_sse,"final primal-dual bracket contradicts")
    optimization=record["optimization"]
    _require(_box(optimization["objective_bracket_sse"])==(dual_sse,primal_sse),"optimization SSE bracket differs")
    _require(_box(optimization["objective_bracket_mean"])==(dual_sse/n,primal_sse/n),"optimization mean bracket differs")
    _require(_f(optimization["gap_sse"])==primal_sse-dual_sse,"optimization gap differs")
    if record["status"]=="VERIFIED_BRACKET_TO_DECLARED_GAP":
        _require(primal_sse-dual_sse<=_f(optimization["gap_target_sse"]),"claimed optimization gap is unresolved")
    work=record["work"]
    _require(work["coordinate_updates"]<=work["max_updates"],"coordinate budget exceeded")
    _require(work["coordinate_updates"]==work["initialization_coordinate_attempts"]+work["cyclic_coordinate_attempts"],"coordinate budget decomposition differs")
    _require(work["initialization_included_in_coordinate_budget"] is True,"initialization omitted from budget")
    _require(record["targets_clipped"] is False and record["population_risk_claim"] is False,"unsupported target or population claim")
    if record["kind"]=="class":
        _require(_box(record["class_risk"])==(dual_sse/n,primal_sse/n),"class-risk bracket differs")
        _require(record["witness_uses_assessment_targets"] is True and record["deployable_head_returned"] is False,"class witness role differs")
    else:
        verify_head(record)
        _require(record["head"]["anchor_features"]==record["features_exact"],"head anchors differ")
        _require(record["head"]["anchor_values"]==primal["values"],"head values differ")
    return {"status":"PASS_DIRECT_FINAL_PRIMAL_DUAL_WITNESS","kind":record["kind"],
        "n":n,"dimension":record["dimension"],"pair_checks":len(pairs),"dual_edges":len(seen),
        "coordinate_attempts":work["coordinate_updates"],"gap_sse":str(primal_sse-dual_sse)}


def verify_prediction_receipt(record):
    _require(record["schema"]=="step35-stored-feature-prediction-receipt-v1","prediction schema differs")
    verify_head(record);head=record["head"]
    z=[list(map(_f,row)) for row in record["features_exact"]]
    _require(record["n"]==len(z)==len(record["rows"])==len(record["predictions"]),"prediction row roster differs")
    _require(record["feature_digest"]==_hash(z),"prediction feature digest differs")
    k=_f(head["kappa"]);low,high=map(_f,head["output_interval"])
    fallback=record["status"]=="VALID_PREDICTOR_RANGE_FALLBACK"
    distance_count=0
    for i,(vector,row) in enumerate(zip(z,record["rows"])):
        _require(row["query_id"]==record["query_ids"][i] and row["query_index"]==i,"prediction query identity differs")
        if fallback:
            predicted=(low,high)
        elif head["kind"]=="clipped_affine":
            beta=list(map(_f,head["coefficients"]))
            raw=_f(head["intercept"])+sum((a*b for a,b in zip(beta,vector)),F(0))
            _require(_box(row["raw_prediction"])==(raw,raw),"affine raw output differs")
            value=_clip(raw,low,high);predicted=(value,value)
        else:
            anchors=[list(map(_f,a)) for a in head["anchor_features"]]
            values=list(map(_f,head["anchor_values"]));minus=[];plus=[]
            _require(len(row["anchor_distances"])==len(anchors),"prediction anchor distance roster differs")
            for j,(anchor,value,drecord) in enumerate(zip(anchors,values,row["anchor_distances"])):
                _require(drecord["anchor_index"]==j and drecord["anchor_id"]==head["anchor_ids"][j],"prediction anchor identity differs")
                a,b=_distance(drecord,vector,anchor,record["distance_bits"])
                distance_count+=1;minus.append((value-k*b,value-k*a));plus.append((value+k*a,value+k*b))
            lower=(max(a for a,b in minus),max(b for a,b in minus))
            upper=(min(a for a,b in plus),min(b for a,b in plus))
            raw=((lower[0]+upper[0])/2,(lower[1]+upper[1])/2)
            _require(_box(row["lower_envelope"])==lower and _box(row["upper_envelope"])==upper,"extension envelope differs")
            _require(_box(row["raw_prediction"])==raw,"extension raw enclosure differs")
            predicted=tuple(_clip(v,low,high) for v in raw)
        _require(_box(row["prediction"])==predicted==_box(record["predictions"][i]),"prediction enclosure differs")
        _require(low<=predicted[0]<=predicted[1]<=high,"prediction enclosure outside range")
    _require(record["distance_evaluations"]==distance_count,"prediction distance cost differs")
    if record["targets_exact"] is not None:
        y=list(map(_f,record["targets_exact"]));losses=[]
        _require(record["target_digest"]==_hash(y),"prediction target digest differs")
        for prediction,target in zip(record["predictions"],y):
            a,b=_box(prediction);a-=target;b-=target
            losses.append((F(0) if a<=0<=b else min(a*a,b*b),max(a*a,b*b)))
        for pair,retained,row in zip(losses,record["loss"]["per_row"],record["rows"]):
            _require(_box(retained)==pair==_box(row["squared_loss"]),"per-row arbitrary-target loss differs")
        means=(sum((a for a,b in losses),F(0))/len(y),sum((b for a,b in losses),F(0))/len(y))
        _require(_box(record["loss"])==means,"mean loss enclosure differs")
        _require(record["targets_outside_predictor_range"]==sum(not low<=a<=high for a in y),"target-range count differs")
    _require(record["targets_clipped"] is False and record["population_risk_claim"] is False,"prediction claim exceeds scope")
    return {"status":"PASS_DIRECT_PREDICTION_AND_LOSS_WITNESS","n":len(z),
        "kind":head["kind"],"distance_checks":distance_count,"fallback":fallback}


def run_preflight():
    import finite_feature35 as m
    tests=[];witnesses=[];rejections=0
    def ok(name):
        tests.append(name)
    def rejects(name,call):
        nonlocal rejections
        try:
            call()
        except (m.FeatureCertificateError,AssertionError):
            rejections+=1;ok(name)
        else:
            raise AssertionError("Expected rejection: "+name)
    _require(m.rational(0.1)==F.from_float(0.1)!=F(1,10),"binary64 must be exactified rather than decimalized")
    ok("binary64_0_1_is_its_exact_dyadic")
    for value in [F(0),F(1),F(2),F(1,3),F(1,2**200),F(2**170+1,2**40)]:
        for bits in [1,8,80,128]:
            b=m.sqrt_enclosure(value,bits=bits);a,c=_box(b)
            _require(a*a<=value<=c*c and c-a<=F(1,1<<bits),"integer-root enclosure failed")
    ok("integer_square_root_exact_irrational_tiny_large_and_four_precisions")
    for name,z,y,k,expected in [
        ("single_target_above_range",[[0]],[2],1,F(1)),
        ("single_target_below_range",[[0]],[-1],1,F(1)),
        ("duplicate_features_conflicting_targets",[[0],[0]],[0,1],2,F(1,4)),
        ("unconstrained_pair",[[0],[1]],[0,1],1,F(0)),
        ("active_pair_constraint",[[0],[F(1,4)]],[0,1],1,F(9,64)),
        ("outside_range_conflicting_targets",[[0],[0]],[-1,2],2,F(9,4)),
        ("zero_sensitivity",[[0],[1]],[0,1],0,F(1,4)),
    ]:
        for kind,method in [("class",m.class_risk_certificate),("training",m.fit_anchor_decoder)]:
            result=method(z,y,feature_id="artificial:"+name,kappa=k)
            witnesses.append(verify_qp_receipt(result))
            _require(_box(result["optimization"]["objective_bracket_mean"])==(expected,expected),"analytic QP value differs")
            if kind=="training":
                receipt=m.evaluate_extension(result,z,y)
                witnesses.append(verify_prediction_receipt(receipt))
                for p,v in zip(receipt["predictions"],result["primal"]["values"]):
                    _require(_box(p)[0]<=_f(v)<=_box(p)[1],"extension fails anchor interpolation")
        ok(name)
    # Box-coordinate optimum, including flat regions, negative multipliers,
    # arbitrary targets and old incidences. The verification uses exact
    # supergradient conditions and nearby competitors, not the root routine.
    rng=random.Random(35001);coordinate_cases=0
    for low,high in [(F(0),F(1)),(F(-2),F(3))]:
        for _ in range(600):
            yi,yj,si,sj,old=[F(rng.randint(-40,40),4) for _ in range(5)]
            radius=F(rng.randint(0,32),8)
            proposal=m._coordinate_optimum(yi,yj,si,sj,old,radius,low,high)
            ai=yi-(si-old)/2;bj=yj-(sj+old)/2
            derivative=_clip(ai-proposal/2,low,high)-_clip(bj+proposal/2,low,high)
            _require((proposal>0 and derivative==radius) or (proposal<0 and derivative==-radius) or
                (proposal==0 and abs(derivative)<=radius),"coordinate KKT condition failed")
            def objective(t):
                a=si+t-old;b=sj-t+old
                u=_clip(yi-a/2,low,high);v=_clip(yj-b/2,low,high)
                return (u-yi)**2+a*u+(v-yj)**2+b*v-radius*abs(t)
            optimum=objective(proposal)
            for delta in [F(-2),F(-1,1024),F(0),F(1,1024),F(2)]:
                _require(objective(proposal+delta)<=optimum,"coordinate exact optimum failed")
            _require(m._coordinate_change(yi,yj,si,sj,old,proposal,radius,low,high)==optimum-objective(old),"coordinate local change differs")
            coordinate_cases+=1
    ok("range_aware_coordinate_optimum_1200_exact_KKT_and_local_checks")
    # Deliberately broad but valid lower/upper bounds demonstrate directional
    # use. They need not themselves form a metric for conservative feasibility.
    z=[[0,0,0],[F(1,2),0,0]];ids=["wide0","wide1"]
    table=m.distance_table(z,ids=ids)
    table["pairs"][0]["distance"]={"lower":"0","upper":"1","width":"1"}
    training=m.fit_anchor_decoder(z,[0,1],feature_id="artificial:wide",ids=ids,kappa=1,distances=table)
    assessed=m.class_risk_certificate(z,[0,1],feature_id="artificial:wide",ids=ids,kappa=1,distances=table)
    witnesses+=[verify_qp_receipt(training,require_tight_distances=False),verify_qp_receipt(assessed,require_tight_distances=False)]
    _require(_box(training["optimization"]["objective_bracket_mean"])==(F(1,4),F(1,4)),"training did not use lower distance")
    _require(_box(assessed["class_risk"])==(F(0),F(1,4)),"class dual did not use upper distance")
    ok("fit_lower_and_class_upper_distance_directions_differ")
    z3=[[0,0],[1,0],[2,0]]
    t3=m.distance_table(z3)
    t3["pairs"][0]["distance"]={"lower":"0","upper":"1","width":"1"}
    t3["pairs"][2]["distance"]={"lower":"0","upper":"1","width":"1"}
    r3=m.fit_anchor_decoder(z3,[-2,F(1,3),3],feature_id="artificial:nonmetric",distances=t3,max_updates=60)
    witnesses.append(verify_qp_receipt(r3,require_tight_distances=False));ok("nonmetric_lower_table_contraction_remains_feasible")
    affine=m.make_affine_head([3,4],0,feature_id="artificial:affine",kappa=2,
        center_features=[F(1,3),F(-1,7)],center_target=F(2,5))
    verify_head(affine)
    _require(_f(affine["membership"]["coefficient_squared_norm"])<=4,"projected affine exceeds sensitivity")
    aq=[[F(1,3),F(-1,7)],[10,-10],[-10,10]]
    ap=m.evaluate_affine(affine,aq,[F(2,5),-2,3])
    witnesses.append(verify_prediction_receipt(ap));_require(_box(ap["predictions"][0])==(F(2,5),F(2,5)),"center recentering differs")
    ok("affine_exact_projection_recenter_clipping_and_arbitrary_targets")
    az=m.make_affine_head([1,-3],F(2,3),feature_id="artificial:zero-affine",kappa=0)
    azp=m.evaluate_affine(az,aq,[-1,0,2]);witnesses.append(verify_prediction_receipt(azp))
    _require(all(_box(p)==(F(2,3),F(2,3)) for p in azp["predictions"]),"zero-sensitivity affine is not constant")
    ok("zero_sensitivity_affine_projection")
    generic_z=[[F(-2),F(1,7),F(3,5)],[F(2,3),F(-1,5),F(4,7)],[F(2),F(7,3),F(-1)]]
    fitted=m.fit_anchor_decoder(generic_z,[-1,F(1,3),2],feature_id="artificial:generic",max_updates=60)
    gp=m.evaluate_extension(fitted,generic_z+[[F(-1,7),F(2,11),F(4,3)]],[-1,F(1,3),2,F(5,2)])
    witnesses.extend([verify_qp_receipt(fitted),verify_prediction_receipt(gp)])
    ok("generic_negative_non_simplex_3d_features_and_irrational_distances")
    fallback=m.fallback_prediction(fitted,generic_z,[-2,0,3],reason="artificial budget exhaustion")
    witnesses.append(verify_prediction_receipt(fallback))
    _require(_box(fallback["loss"])[1]==F(19,3),"fallback arbitrary-target loss differs")
    ok("full_predictor_range_fallback_with_unclipped_targets")
    zero=m.fit_anchor_decoder([[0],[F(1,100)]],[0,1],feature_id="artificial:zero-budget",max_updates=0)
    witnesses.append(verify_qp_receipt(zero))
    _require(zero["status"]=="VALID_CONSTANT_PRIMAL_ZERO_EDGE_DUAL_FALLBACK" and zero["work"]["coordinate_updates"]==0,"zero-budget fallback differs")
    ok("zero_update_budget_returns_valid_constant")
    limited=m.fit_anchor_decoder([[F(i,100)] for i in range(8)],[F(i%2) for i in range(8)],
        feature_id="artificial:hard-cap",max_updates=1,gap_sse=0)
    witnesses.append(verify_qp_receipt(limited))
    _require(limited["work"]["coordinate_updates"]<=1 and limited["work"]["initialization_coordinate_attempts"]==1,"initialization excluded from hard cap")
    ok("one_attempt_cap_includes_greedy_initialization")
    # Independent singleton optima are zero, but the common-class optimum on
    # the union is 1/4 because both rows have the identical feature vector.
    blocks=[m.class_risk_certificate([[0]],[y],feature_id="artificial:aggregate",ids=[str(i)]) for i,y in enumerate([0,1])]
    common=m.make_affine_head([0],F(1,2),feature_id="artificial:aggregate")
    common_receipt=m.evaluate_affine(common,[[0],[0]],[0,1],ids=["0","1"])
    agg=m.aggregate_class_risk(blocks,[common_receipt])
    _require(_box(agg["class_risk"])==(F(0),F(1,4)),"block optima incorrectly supplied a global upper")
    _require(agg["block_primals_averaged_for_union_upper"] is False and agg["diagnostic_assessment_constant_included"] is False,"aggregation upper source differs")
    diagnostic=m.aggregate_class_risk(blocks,include_diagnostic_constant=True)
    _require(_box(diagnostic["class_risk"])==(F(0),F(1,4)) and diagnostic["global_upper_candidates"][0]["training_only_head"] is False,"diagnostic constant role differs")
    ok("aggregation_counterexample_requires_common_frozen_head_upper")
    # Reject malformed and misbound premises.
    for name,call in [
        ("reject_nan",lambda:m.distance_table([[float('nan')]])),
        ("reject_infinite_target",lambda:m.class_risk_certificate([[0]],[float('inf')],feature_id="bad")),
        ("reject_boolean_coordinate",lambda:m.distance_table([[True]])),
        ("reject_ragged_features",lambda:m.distance_table([[1],[1,2]])),
        ("reject_more_than_32_qp_rows",lambda:m.class_risk_certificate([[0]]*33,[0]*33,feature_id="bad")),
        ("reject_negative_sensitivity",lambda:m.class_risk_certificate([[0]],[0],feature_id="bad",kappa=-1)),
        ("reject_duplicate_query_identifiers",lambda:m.distance_table([[0],[1]],ids=["x","x"])),
        ("reject_unprojected_oversize_affine",lambda:m.make_affine_head([3,4],0,feature_id="bad",kappa=2,project=False)),
        ("reject_partial_affine_center",lambda:m.make_affine_head([0],0,feature_id="bad",center_features=[0])),
        ("reject_malformed_precision",lambda:m.make_affine_head([0],0,feature_id="bad",bits=True)),
        ("reject_aggregate_without_global_head",lambda:m.aggregate_class_risk(blocks)),
        ("reject_duplicate_aggregate_rows",lambda:m.aggregate_class_risk([blocks[0],blocks[0]],[common_receipt])),
    ]:
        rejects(name,call)
    changed=deepcopy(fitted);changed["head"]["anchor_values"][0]=str(_f(changed["head"]["anchor_values"][0])+F(1,7))
    rejects("reject_altered_head_identity",lambda:m.evaluate_extension(changed,generic_z))
    changed=deepcopy(common_receipt);changed["query_ids"].reverse()
    rejects("reject_global_receipt_wrong_union_order",lambda:m.aggregate_class_risk(blocks,[changed]))
    changed=deepcopy(fitted);changed["dual"]["sse"]="12345"
    rejects("checker_rejects_corrupt_dual",lambda:verify_qp_receipt(changed))
    changed=deepcopy(gp);changed["rows"][0]["anchor_distances"][0]["squared_exact"]="1"
    rejects("checker_rejects_corrupt_prediction_distance",lambda:verify_prediction_receipt(changed))
    # Pure component calls must succeed while file opens and imports are
    # forbidden. Modules are already loaded by this driver at this point.
    original_open,original_import=builtins.open,builtins.__import__
    def blocked(*args,**kwargs):
        raise AssertionError("Pure arithmetic component attempted I/O or a dynamic import")
    try:
        builtins.open=blocked;builtins.__import__=blocked
        pure=m.fit_anchor_decoder([[0,0],[F(1,3),F(2,3)]],[F(-1,7),F(9,7)],feature_id="artificial:pure",max_updates=20)
        m.evaluate_extension(pure,[[F(2,5),F(-1,5)]],[F(11,9)])
    finally:
        builtins.open=original_open;builtins.__import__=original_import
    ok("pure_calls_with_file_open_and_dynamic_import_disabled")
    # Timings are artificial engineering estimates only. No actual data or
    # feature model appears in this deliberately tight geometry.
    benchmark_z=[[float(F(((i+1)*(j+7))%37,3700)) for j in range(8)] for i in range(32)]
    benchmark_y=[F((i*13)%41-5,31) for i in range(32)]
    benchmark={}
    for purpose,method,budget in [("class_block",m.class_risk_certificate,400),("training_head",m.fit_anchor_decoder,4000)]:
        start=time.perf_counter()
        record=method(benchmark_z,benchmark_y,feature_id="artificial:timing",max_updates=budget,gap_sse=F(1,10**7))
        elapsed=time.perf_counter()-start
        witnesses.append(verify_qp_receipt(record))
        benchmark[purpose]={"n":32,"dimension":8,"budget":budget,"seconds":elapsed,
            "coordinate_attempts":record["work"]["coordinate_updates"],"status":record["status"],
            "gap_sse":record["optimization"]["gap_sse"],"serialized_bytes":len(json.dumps(record))}
    ok("hard_32_row_artificial_timing_and_direct_witness_checks")
    return {"schema":"step35-artificial-finite-feature-preflight-v1","status":"PASS",
        "actual_public_data_rows_loaded":0,"native_feature_models_imported_or_fitted":0,
        "reserved_rows_opened":0,"tests":tests,"test_count":len(tests),"expected_rejections":rejections,
        "coordinate_optimum_cases":coordinate_cases,"direct_final_witness_checks":witnesses,
        "benchmark_artificial_only":benchmark,"optimization_trace_replay_claim":False,
        "assurance":"CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES",
        "native_transform_error_enclosed":False,"population_risk_claim":False,
        "artificial_witnesses":{"generic_fit":fitted,"generic_prediction":gp,"projected_affine":affine,
            "affine_prediction":ap,"range_fallback":fallback,"zero_budget_fit":zero,
            "one_attempt_fit":limited,"singleton_class_blocks":blocks,
            "common_head_prediction":common_receipt,"aggregate":agg},
        "example_api_result":{"fit_keys":list(fitted),"prediction_keys":list(gp),"class_keys":list(blocks[0]),
            "aggregate":agg,"fit_status":fitted["status"],"loss":gp["loss"]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--verify",type=Path,help="Directly check a retained QP, head or prediction JSON receipt")
    args=parser.parse_args()
    if args.verify is not None:
        record=json.loads(args.verify.read_text())
        schema=record["schema"]
        if schema=="step35-stored-feature-finite-qp-v1":
            result=verify_qp_receipt(record)
        elif schema=="step35-stored-feature-prediction-receipt-v1":
            result=verify_prediction_receipt(record)
        elif schema in ("step35-stored-feature-head-v1","step35-admissible-affine-head-v1"):
            result=verify_head(record)
        else:
            raise ValueError("Unsupported direct receipt schema")
    else:
        result=run_preflight()
        here=Path(__file__).resolve()
        result["source_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [here,here.with_name("finite_feature35.py")]}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    summary={k:result[k] for k in ["status","test_count","expected_rejections","coordinate_optimum_cases","benchmark_artificial_only","source_sha256"] if k in result}
    print(json.dumps(summary or result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
