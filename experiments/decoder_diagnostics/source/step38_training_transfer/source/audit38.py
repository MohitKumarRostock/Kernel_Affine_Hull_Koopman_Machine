"""Independent Step38 verification: no scientific producer imports.

The pure APIs verify original-label training witnesses, full-anchor head
membership, shared query geometry, exact arithmetic prediction enclosures and
losses. Data/protocol/execution adapters are added before the experiment seal.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from math import isfinite,isqrt,lcm
from pathlib import Path
import sys
import time
import traceback
import zipfile

import independent_math35 as old
import audit_math37 as graph_math

ROOT=Path(__file__).resolve().parents[1]
ASSURANCE="CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
HEAD_SCHEMA="step38-stored-feature-extension-head-v1"
RECIPES=("affine","extension32","qp32","envelope_all","qp_all")
INDEPENDENT_MATH_SHA256="3f440f427614ac230f4a09ac062d5bfc18eaa47babdaaff1c3c9b04b2c330308"
GRAPH_MATH_SHA256="9484154f36df51a9e084e102358bac45f0da6de89d8052dc50d2d761886b901c"
PRIMARY_SETTINGS=dict(graph_math.PRIMARY_SETTINGS)
BASE_HASHES={"source/independent_math35.py":INDEPENDENT_MATH_SHA256,
    "source/audit_math37.py":GRAPH_MATH_SHA256,
    "source/finite_feature35.py":"7a6d9478bb03293eb42f133a61ac6a31471c3ce95fcf819304f54e3d5635fc18",
    "source/full_graph37.py":"08a311f901a6f3143b85fb30b4f8df6def917b23175c3dd3d70b2e9d0dda2b92",
    "source/representations35.py":"8ee3acfd590b5ea1a6716f6d5da05ebed3f86dcc842dacea0242d8ddd80637b3",
    "source/models36.py":"e48d79f7539a18d24d2df68b0b29b97e5288f8e0929d5d881c59ae1f97463390"}
ARCHIVE_SHA256="139e9908f0a3544bb222386855c9ce107e96467306bb8e4ce936aab59e7baac4"
CSV_SHA256="373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"
BASELINES=("persistence","training_mean","hist_gradient_boosting")
PARTITIONS={"train":(datetime(2017,12,1),datetime(2018,8,1),242),
    "validation":(datetime(2018,8,1),datetime(2018,9,1),31),
    "test":(datetime(2018,9,1),datetime(2018,12,1),91)}
REP_CONFIG={"dimension":24,"scalings":["minmax01","standard"],"nystrom_widths":[8,24],
    "kahm_caps":[4,12,20],"kahm_prefixes":[1,2],"cluster_counts":[8,16],"max_fit":512,
    "max_references":24,"tau":1e-6,"omega":2.0,"probe_count":32,"nystrom_eigen_floor":1e-12,
    "temporal_selection":"equal_row_stratum_midpoints","metric_normalization":"reciprocal_median_positive_pair_distance"}

class AuditError(AssertionError):pass
def require(value,message):
    if not value:raise AuditError(message)
def q(value):return old.q(value)
def canon(value):return old.canon(value)
def digest(value):return old.digest(value)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return old.read_json(path)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(canon(value),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
def newline_digest(value):
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()
def interval(record,expected=None):return old.interval(record,expected)
def clip(v):return min(F(1),max(F(0),v))
def class_identity(feature_id,kappa="2"):return graph_math.class_identity(feature_id,kappa,("0","1"))

def exact_matrix(z):
    if hasattr(z,"tolist"):z=z.tolist()
    result=graph_math.exact_matrix(z)
    for row in result:
        for value in row:
            numeric=float(value)
            require(isfinite(numeric) and F.from_float(numeric)==value,
                    "Stored feature coordinate is not an exact finite binary64 value")
    return result

def exact_ids(ids,n):
    require(isinstance(ids,list) and len(ids)==n and len(set(ids))==n and
            all(isinstance(value,str) and value for value in ids),"Invalid unique identifier roster")
    return ids

def audit_fit(record,z,y,ids,expected=None):
    """Check a training/subset/post-assessment finite-class fit exactly."""
    expected=expected or {};z=exact_matrix(z);y=list(map(q,y));n=len(z)
    exact_ids(ids,n)
    require(record["schema"]=="step38-training-class-fit-v1" and record["assurance"]==ASSURANCE and
            record["role"] in ("train","assessment_diagnostic","artificial"),"Unknown fit schema or role")
    require(len(y)==n==record["n"] and record["dimension"]==len(z[0]) and record["ids"]==ids,
            "Fit cohort identity or dimensions differ")
    kap=q(record["kappa"])
    require(kap>=0 and record["output_interval"]==["0","1"] and
            record["class_id"]==class_identity(record["feature_id"],kap),"Fit class definition differs")
    require(record["feature_digest"]==digest(z) and record["target_digest"]==digest(y),
            "Fit original features or target digest differs")
    for name in ("role","feature_id","class_id","input_binding"):
        if name in expected:require(record[name]==expected[name],"Fit binding differs: "+name)
    if "kappa" in expected:require(kap==q(expected["kappa"]),"Fit sensitivity changed")
    require(record["head_promotion_allowed"] is (record["role"]=="train") and
            record["targets_clipped_in_loss"] is False and record["native_transform_error_enclosed"] is False and
            record["future_response_coverage_claim"] is False and record["population_risk_claim"] is False,
            "Fit information boundary or assurance scope differs")
    distance=record["distance"]
    require(distance["schema"]=="step37-complete-exact-distance-graph-v1" and distance["bits"]==80 and
            distance["pair_order"]=="lexicographic_i_then_j_i_lt_j" and
            distance["coordinates_interpreted_as_exact_binary64"] is True and
            distance["omitted_qp_pairs_are_redundant_given_output_interval"] is True,
            "Training distance convention differs")
    floors=[graph_math.exact_integer(v) for v in distance["floor_scaled"]]
    graph=graph_math.complete_distances(z,ids,floors,80)
    exponent=max(value.denominator.bit_length()-1 for row in z for value in row)
    exact=[i for i,e in enumerate(graph["pairs"]) if e["lower"]==e["upper"]]
    active=[i for i,e in enumerate(graph["pairs"]) if kap*e["lower"]<1]
    require(distance["pair_count"]==graph["pair_count"] and distance["common_binary_exponent"]==exponent and
            distance["exact_pair_indices"]==exact and distance["nonredundant_pair_indices"]==active and
            distance["zero_distance_pair_count"]==graph["exact_zero_pairs"] and
            int(distance["constraint_integer_scale"])==(1<<80)*kap.denominator and
            distance["kappa_numerator"]==kap.numerator,"Full-graph distance receipt differs")
    numerical=record["numerical"];settings=expected.get("settings",PRIMARY_SETTINGS)
    require(numerical["solver"]=="OSQP" and numerical["version"]=="1.0.5" and numerical["settings"]==settings and
            numerical["objective"]=="unhalved_SSE_without_constant_P2I_q_minus2y" and
            numerical["box_duals_used_in_certificate"] is False and
            numerical["constraint_lower_bounds_rounded_to_binary64_for_proposal_only"] is True,
            "Numerical proposal convention differs")
    rawx=numerical["raw_primal_binary64"];rawdual=numerical["raw_edge_dual_binary64"]
    if rawx is not None:rawx=graph_math.verify_binary64_list(rawx,n)
    if rawdual is not None:rawdual=graph_math.verify_binary64_list(rawdual,len(active))
    require(type(numerical["iterations"]) is int and numerical["iterations"]>=0 and
            isfinite(numerical["proposal_wall_seconds"]) and numerical["proposal_wall_seconds"]>=0,
            "Invalid numerical work record")
    if rawx is None or rawdual is None:
        require(isinstance(numerical["fallback_reason"],str) and numerical["fallback_reason"],
                "Missing numerical proposal has no retained fallback reason")
    candidates=record["primal_candidates"]
    order=["clipped_target_envelope"]+([] if rawx is None else ["solver_envelope"])+["clipped_mean_constant"]
    require([c["source"] for c in candidates]==order,"Declared candidate roster/order differs")
    checked=[]
    for candidate in candidates:
        raw_seed=y if candidate["source"]=="clipped_target_envelope" else rawx if candidate["source"]=="solver_envelope" else None
        checked.append(graph_math.verify_candidate(candidate,raw_seed,y,graph,kap))
    selected=min(candidates,key=lambda c:(q(c["sse"]),c["source"]))
    envelope=min((c for c in candidates if c["source"]!="solver_envelope"),key=lambda c:(q(c["sse"]),c["source"]))
    require(record["selected_qp_source"]==selected["source"] and
            record["selected_envelope_source"]==envelope["source"] and
            record["training_qp_no_worse_than_envelope"] is (q(selected["sse"])<=q(envelope["sse"])),
            "Training candidate selection differs from exact SSE/lexical rule")
    dual=record["dual"]
    require(dual["grid_bits"]==40 and dual["distance_endpoint"]=="upper" and
            dual["valid_for_any_finite_signed_multipliers"] is True,"Wrong true-class dual convention")
    expected_multipliers=[]
    if rawdual is not None:
        for index,value in zip(active,rawdual):
            lam=F(graph_math.quantized_integer(value,40),1<<40)
            if lam:expected_multipliers.append({"pair_index":index,"value":str(lam)})
    require(dual["signed_multipliers"]==expected_multipliers,"Retained signed dual quantization differs")
    edges=[(*graph["pairs"][r["pair_index"]]["indices"],r["value"]) for r in expected_multipliers]
    dual_check=graph_math.verify_dual(y,edges,graph,kap)
    for name,other in (("incidence","incidence"),("box_minimizers","box_minimizers"),("coordinate_terms","coordinate_minima")):
        require(list(map(q,dual[name]))==dual_check[other],"Dual exact reconstruction differs: "+name)
    require(q(dual["penalty"])==dual_check["penalty"] and q(dual["sse"])==dual_check["unclamped_sse"] and
            q(dual["mean"])==dual_check["unclamped_sse"]/n,"Dual objective normalization differs")
    floor=graph_math.original_target_floor(y);terms=[(clip(t)-t)**2 for t in y]
    require(list(map(q,record["range_floor"]["singleton_sse"]))==terms and
            q(record["range_floor"]["sse"])==floor["sse"] and q(record["range_floor"]["mean"])==floor["mean"],
            "Original-target range-floor arithmetic differs")
    low=dual_check["mean"];upper=q(selected["mean"])
    require(0<=low<=upper,"Training class bracket is inconsistent")
    interval(record["class_risk"],(low,upper))
    require(q(record["gap_target_mean"])==F(1,100000000) and record["gap_resolved"] is (upper-low<=F(1,100000000)),
            "Class-risk gap status differs")
    for value in record["timing"].values():require(isfinite(value) and value>=0,"Invalid fit runtime")
    return canon({"status":"PASS","role":record["role"],"n":n,"dimension":len(z[0]),
        "feature_id":record["feature_id"],"class_id":record["class_id"],"pair_distances_verified":graph["pair_count"],
        "primal_candidates_verified":len(candidates),"primal_pair_checks":len(candidates)*graph["pair_count"],
        "shortest_path_envelopes_verified":2*(len(candidates)-1),"signed_dual_edges_verified":len(edges),
        "class_risk":record["class_risk"],"selected_qp_source":selected["source"],
        "selected_envelope_source":envelope["source"],"gap_resolved":record["gap_resolved"],
        "outside_range_targets":floor["outside_count"],"duplicate_feature_pairs":graph["exact_zero_pairs"],
        "numerical_status_is_not_a_proof":True})

def audit_head(head,training_fit=None,z=None,y=None,training_ids=None,expected=None):
    """Check head membership once; new heads also bind the training-only fit."""
    expected=expected or {}
    if head.get("schema")!="step38-stored-feature-extension-head-v1":
        hid,kap,low,high=old.audit_head(head)
        require((low,high)==(F(0),F(1)),"Legacy prediction interval changed")
        if "feature_id" in expected:require(head["feature_id"]==expected["feature_id"],"Legacy feature identity differs")
        if "class_id" in expected:require(head["class_id"]==expected["class_id"],"Legacy class identity differs")
        return {"status":"PASS","head_id":hid,"class_id":head["class_id"],"kind":head["kind"],
                "anchor_count":0 if head["kind"]=="clipped_affine" else len(head["anchor_ids"])}
    require(training_fit is not None and z is not None and y is not None and training_ids is not None,
            "Full-anchor head requires independent training-source binding")
    features=exact_matrix(z);targets=list(map(q,y));n=len(features);exact_ids(training_ids,n)
    require(training_fit["role"]=="train" and training_fit["head_promotion_allowed"] is True and
            head["training_fit_role"]=="train" and head["kind"]=="symmetric_lipschitz_extension" and
            head["assurance"]==ASSURANCE and head["recipe"] in ("qp32","qp_all","envelope_all"),
            "Non-training witness was promoted or head schema differs")
    kap=q(head["kappa"])
    require(kap>=0 and head["output_interval"]==["0","1"] and
            head["class_id"]==class_identity(head["feature_id"],kap) and
            head["feature_id"]==training_fit["feature_id"] and head["class_id"]==training_fit["class_id"] and
            head["dimension"]==training_fit["dimension"]==len(features[0]) and
            q(training_fit["kappa"])==kap,"Full-anchor class identity differs")
    require(training_fit["feature_digest"]==digest(features) and training_fit["target_digest"]==digest(targets) and
            training_fit["ids"]==training_ids and training_fit["n"]==n==len(targets) and
            head["anchor_ids"]==training_ids and exact_matrix(head["anchor_features"])==features,
            "Full-anchor head is not bound to the original training geometry/targets")
    indices=head["anchor_indices"]
    require(isinstance(indices,list) and len(indices)==len(set(indices))==n and
            all(type(i) is int and i>=0 for i in indices),"Invalid training anchor indices")
    if head["recipe"] in ("qp_all","envelope_all"):
        require(indices==list(range(n)),"All-anchor recipe omitted/reordered a training anchor")
    else:require(n<=32,"QP32 exceeded its unchanged anchor count")
    key="selected_envelope_source" if head["recipe"]=="envelope_all" else "selected_qp_source"
    pool=[c for c in training_fit["primal_candidates"] if head["recipe"]!="envelope_all" or c["source"]!="solver_envelope"]
    chosen=min(pool,key=lambda c:(q(c["sse"]),c["source"]))
    values=list(map(q,head["anchor_values"]))
    require(training_fit[key]==chosen["source"]==head["selected_training_source"] and
            values==list(map(q,chosen["values"])) and len(values)==n and all(0<=v<=1 for v in values),
            "Promoted values do not follow the fixed training-only selection rule")
    den=lcm(*(x.denominator for row in features for x in row))
    int_features=[[x.numerator*(den//x.denominator) for x in row] for row in features]
    value_den=lcm(*(v.denominator for v in values));int_values=[v.numerator*(value_den//v.denominator) for v in values]
    for i,j in combinations(range(n),2):
        squared=sum((a-b)**2 for a,b in zip(int_features[i],int_features[j]))
        require((int_values[i]-int_values[j])**2*kap.denominator**2*den**2<=
                kap.numerator**2*squared*value_den**2,"Full-anchor exact interpolation feasibility failed")
    risk=sum(((v-t)**2 for v,t in zip(values,targets)),F(0))/n
    require(q(head["training_anchor_risk"])==risk==q(chosen["mean"]) and q(chosen["sse"])==risk*n,
            "Promoted head training risk differs")
    require(head["formula"]=="clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)",
            "Full-anchor extension/clipping formula differs")
    for key in ("recipe","feature_id","class_id","anchor_indices","training_fit_path","training_fit_sha256"):
        if key in expected:require(head[key]==expected[key],"Head external binding differs: "+key)
    hid="head38:"+digest(head)
    if "head_id" in expected:require(hid==expected["head_id"],"Head content identity differs")
    return canon({"status":"PASS","head_id":hid,"class_id":head["class_id"],"kind":head["kind"],
        "recipe":head["recipe"],"anchor_count":n,"pair_membership_checks":n*(n-1)//2,
        "training_anchor_risk":risk,"training_fit_role":"train","selected_source":chosen["source"]})

def audit_distances(record,anchor_z,query_z,anchor_ids,query_ids,expected=None):
    """Verify compact shared query geometry; return reusable internal cache."""
    expected=expected or {};anchors=exact_matrix(anchor_z);queries=exact_matrix(query_z)
    na,nq=len(anchors),len(queries);exact_ids(anchor_ids,na);exact_ids(query_ids,nq)
    require(record["schema"]=="step38-query-distances-v1" and record["anchor_ids"]==anchor_ids and
            record["query_ids"]==query_ids and record["n_anchor"]==na and record["n_query"]==nq and
            record["dimension"]==len(anchors[0])==len(queries[0]),"Shared query geometry identity/shape differs")
    require(record["anchor_feature_digest"]==digest(anchors) and record["query_feature_digest"]==digest(queries) and
            exact_matrix(record["features_exact"])==queries and record["bits"]==80 and
            record["pair_order"]=="query_then_anchor_rowmajor" and record["target_values_accessed"] is False,
            "Query exact feature binding or label-free role differs")
    for key in ("feature_id",):
        if key in expected:require(record[key]==expected[key],"Shared distance binding differs")
    exponent=max(x.denominator.bit_length()-1 for row in anchors+queries for x in row)
    require(record["common_binary_exponent"]==exponent and len(record["floor_scaled"])==na*nq,
            "Shared query integer scale or complete distance roster differs")
    den=lcm(*(x.denominator for row in anchors+queries for x in row))
    ai=[[x.numerator*(den//x.denominator) for x in row] for row in anchors]
    zi=[[x.numerator*(den//x.denominator) for x in row] for row in queries]
    floors=[graph_math.exact_integer(v) for v in record["floor_scaled"]]
    lows=[];highs=[];exact=[]
    for i,zrow in enumerate(zi):
        lo_row=[];hi_row=[]
        for j,arow in enumerate(ai):
            index=i*na+j;squared=F(sum((a-b)**2 for a,b in zip(zrow,arow)),den**2)
            lo,hi=graph_math.verify_sqrt_floor(squared,floors[index],80)
            if lo==hi:exact.append(index)
            lo_row.append(lo);hi_row.append(hi)
        lows.append(lo_row);highs.append(hi_row)
    require(record["exact_pair_indices"]==exact,"Exact query-distance mask differs")
    return {"summary":{"status":"PASS","feature_id":record["feature_id"],"queries":nq,"anchors":na,
                        "distance_pairs_verified":na*nq,"distance_digest":newline_digest(record)},
            "anchors":anchors,"queries":queries,"anchor_ids":anchor_ids,"query_ids":query_ids,
            "lower":lows,"upper":highs,"bits":80,"record_digest":newline_digest(record)}

def audit_prediction(record,head,query_z,query_ids,distance=None,distance_checked=None,expected=None):
    """Verify fixed-formula prediction; head membership is checked separately once."""
    expected=expected or {};z=exact_matrix(query_z);n=len(z);exact_ids(query_ids,n)
    hid=("head38:" if head.get("schema")==HEAD_SCHEMA else "head35:")+digest(head)
    kap=q(head["kappa"])
    require(record["schema"]=="step38-head-prediction-v1" and record["assurance"]==ASSURANCE and
            record["head_id"]==hid and record["feature_id"]==head["feature_id"] and
            record["class_id"]==head["class_id"]==class_identity(head["feature_id"],kap) and
            record["query_ids"]==query_ids and record["query_feature_digest"]==digest(z) and
            record["n"]==n==len(record["rows"])==len(record["predictions"]),
            "Prediction head/class/query identity differs")
    require(record["targets_accessed"] is False and record["future_response_coverage_claim"] is False and
            record["native_transform_error_enclosed"] is False,"Prediction information/assurance scope differs")
    require(kap>=0 and head["output_interval"]==["0","1"] and all(len(row)==head["dimension"] for row in z),
            "Prediction class parameters differ")
    for key in ("feature_id","class_id","head_id"):
        if key in expected:require(record[key]==expected[key],"Prediction external binding differs: "+key)
    kind=head["kind"];columns=None
    if kind=="clipped_affine":
        beta=list(map(q,head["coefficients"]));intercept=q(head["intercept"])
        require(len(beta)==len(z[0]) and sum((b*b for b in beta),F(0))<=kap*kap,
                "Affine prediction membership differs")
        require(record["distance_digest"] is None and record["anchor_columns"] is None,
                "Affine prediction used extension geometry")
    else:
        require(kind=="symmetric_lipschitz_extension" and distance is not None and distance_checked is not None,
                "Extension prediction requires independently verified shared geometry")
        require(distance_checked["record_digest"]==newline_digest(distance)==record["distance_digest"] and
                distance["feature_id"]==head["feature_id"] and distance_checked["queries"]==z and
                distance_checked["query_ids"]==query_ids,"Prediction uses another geometry receipt")
        lookup={name:i for i,name in enumerate(distance_checked["anchor_ids"])}
        try:columns=[lookup[name] for name in head["anchor_ids"]]
        except KeyError as error:raise AuditError("Head anchor missing from shared geometry") from error
        require(record["anchor_columns"]==columns and
                exact_matrix(head["anchor_features"])==[distance_checked["anchors"][i] for i in columns],
                "Prediction anchor columns/coordinates differ from immutable head")
        if head.get("schema")==HEAD_SCHEMA:require(head["anchor_indices"]==columns,"New head anchor subset differs")
        else:require(head.get("schema")=="step35-stored-feature-head-v1" and len(columns)<=32,"Legacy32 schema/cap changed")
        values=list(map(q,head["anchor_values"]))
        require(len(values)==len(columns) and all(0<=value<=1 for value in values),"Extension value roster/range differs")
    widths=[]
    for i,(row,prediction) in enumerate(zip(record["rows"],record["predictions"])):
        require(row["query_id"]==query_ids[i],"Prediction row ordering differs")
        if kind=="clipped_affine":
            raw=intercept+sum((b*x for b,x in zip(beta,z[i])),F(0));raw_box=(raw,raw)
        else:
            lower_cones=[(value-kap*distance_checked["upper"][i][j],value-kap*distance_checked["lower"][i][j])
                         for j,value in zip(columns,values)]
            upper_cones=[(value+kap*distance_checked["lower"][i][j],value+kap*distance_checked["upper"][i][j])
                         for j,value in zip(columns,values)]
            lower=tuple(max(v[k] for v in lower_cones) for k in (0,1))
            upper=tuple(min(v[k] for v in upper_cones) for k in (0,1))
            interval(row["lower_envelope"],lower);interval(row["upper_envelope"],upper)
            raw_box=tuple((lower[k]+upper[k])/2 for k in (0,1))
        interval(row["raw_prediction"],raw_box)
        expected_box=tuple(clip(v) for v in raw_box)
        interval(row["prediction"],expected_box);interval(prediction,expected_box)
        width=expected_box[1]-expected_box[0]
        require(0<=expected_box[0]<=expected_box[1]<=1 and width<=kap/F(1<<80),
                "Prediction enclosure range or exact arithmetic width failed")
        widths.append(width)
    return canon({"status":"PASS","head_id":hid,"class_id":head["class_id"],"n":n,
        "anchor_columns_used":0 if columns is None else len(columns),"maximum_prediction_width":max(widths),
        "prediction_enclosures_verified":n,"loss_targets_not_required":True})

def audit_baseline_prediction(record,query_ids,expected_values=None):
    require(record["schema"]=="step38-baseline-prediction-v1" and record["query_ids"]==query_ids and
            record["targets_accessed"] is False and record["class_membership_certified"] is False and
            record["assurance"]=="EXACT_LOSS_OF_STORED_BASELINE_PREDICTIONS" and
            len(record["predictions"])==len(query_ids),"Baseline prediction identity or scope differs")
    values=[]
    for pred in record["predictions"]:
        low,high=interval(pred);require(low==high,"Stored baseline prediction is not a point value");values.append(low)
    if expected_values is not None:require(values==list(map(q,expected_values)),"Baseline replay values differ")
    return {"status":"PASS","baseline":record["baseline"],"predictions_verified":len(values)}

def audit_loss(predictions,targets,loss_record):
    low,high=old.audit_loss(predictions,targets,loss_record)
    return {"lower":str(low),"upper":str(high),"width":str(high-low)}

def audit_practical_gate(new,legacy,ordinary,record):
    nu=interval(new)[1];ll=interval(legacy)[0];bl=interval(ordinary)[0]
    expected={"legacy_positive_risk_denominator":ll>0,
        "at_least_five_percent_mse_improvement":ll>0 and 20*nu<=19*ll,
        "no_worse_than_validation_selected_ordinary_baseline":nu<=bl,
        "numerical_practical_transfer_criterion":ll>0 and 20*nu<=19*ll and nu<=bl,
        "relative_gain_status":"DEFINED" if ll>0 else "UNDEFINED_RELATIVE_GAIN",
        "five_percent_refers_to":"MSE_not_RMSE","population_noninferiority_claim":False}
    require(record==expected,"Frozen practical-transfer gate arithmetic differs")
    return expected

def maps_roster():
    return sorted(["raw_"+s for s in ("minmax01","standard")]+
        ["nystrom_"+s+"_w"+str(w) for s in ("minmax01","standard") for w in (8,24)]+
        ["kahm_"+s+"_cap"+str(c)+"_p"+str(p) for s in ("minmax01","standard") for c in (4,12,20) for p in (1,2)])

def checked_path(relative):
    path=Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,"Unsafe artifact path")
    full=(ROOT/path).resolve();require(full.is_relative_to(ROOT.resolve()),"Artifact path escaped Step38")
    return full

def bound_reference(reference):
    path=checked_path(reference["path"])
    require(path.is_file() and sha(path)==reference["sha256"],"Bound artifact bytes differ: "+str(reference["path"]))
    return path

def verify_bound_seal(relative,schema):
    path=checked_path(relative);record=read(path)
    require(record["schema"]==schema and isinstance(record["bound_files"],dict) and record["bound_files"],"Seal schema/roster differs")
    for name,wanted in record["bound_files"].items():
        item=checked_path(name);require(item.is_file() and sha(item)==wanted,"Sealed file changed: "+name)
    return record

def audit_events():
    path=ROOT/"output/access_and_execution38.jsonl"
    if not path.exists():return []
    previous=None;previous_ns=None;previous_utc=None;events=[]
    for line in path.read_text().splitlines():
        record=json.loads(line);saved=record.pop("event_sha256");current=datetime.fromisoformat(record["utc"])
        require(newline_digest(record)==saved and record["previous_event_sha256"]==previous,"Access-event hash chain differs")
        require(type(record["monotonic_ns"]) is int and current.tzinfo is not None and
            (previous_ns is None or record["monotonic_ns"]>=previous_ns) and
            (previous_utc is None or current>=previous_utc),"Access event order/time regressed")
        require("failed" not in record["event"] and record["event"]!="phase_stopped_error","Execution retained a failure")
        record["event_sha256"]=saved;events.append(record);previous=saved
        previous_ns=record["monotonic_ns"];previous_utc=current
    return events

def append_event(name,**fields):
    events=audit_events();record={"event":name,"utc":datetime.now(timezone.utc).isoformat(),
        "monotonic_ns":time.monotonic_ns(),"previous_event_sha256":events[-1]["event_sha256"] if events else None,**fields}
    record["event_sha256"]=newline_digest(record)
    path=ROOT/"output/access_and_execution38.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as handle:handle.write(json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")

def require_once(events,name):
    indices=[i for i,e in enumerate(events) if e["event"]==name]
    require(len(indices)==1,"Expected exactly one event: "+name)
    return indices[0],events[indices[0]]

def iso(stamp):return stamp.isoformat(timespec="seconds")

def calendar_queries(name):
    begin,end,number=PARTITIONS[name];first=datetime(2017,12,1);out=[]
    while begin<end:
        target=begin.replace(hour=23);origin=target-timedelta(hours=24)
        if origin-timedelta(hours=23)>=first:
            out.append({"id":name+":"+iso(target),"target_timestamp":iso(target),"origin_timestamp":iso(origin)})
        begin+=timedelta(days=1)
    require(len(out)==number,"Independent calendar split count differs")
    return out

def lag_stamps(query):
    origin=datetime.fromisoformat(query["origin_timestamp"])
    return [iso(origin-timedelta(hours=23-i)) for i in range(24)]

def audit_source_metadata():
    """Independently decode approved calendar/operation tokens only, never counts."""
    source=read(ROOT/"inputs/public_source38.json");projection=read(ROOT/"inputs/calendar_projection38.json")
    archive=ROOT/"inputs/seoul_bike_uci560.zip"
    require(sha(archive)==ARCHIVE_SHA256==source["archive_sha256"]==projection["archive_sha256"],"New source archive binding differs")
    require(source["numerical_count_values_parsed"]==source["weather_values_parsed"]==0 and
        source["uci_id"]==560 and source["absolute_historical_never_seen_claim"] is False,"Source exposure scope differs")
    require(projection["parsed_data_columns"]==[0,2,13] and projection["count_values_decoded_or_parsed"]==0 and
        projection["weather_values_decoded_or_parsed"]==0 and projection["closure_rows_retained"] is True and
        projection["operation_flags_used_as_predictors"] is False,"Pre-freeze metadata scope differs")
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.namelist()==["SeoulBikeData.csv"],"Source archive member roster differs")
        raw=zipped.read("SeoulBikeData.csv")
    require(hashlib.sha256(raw).hexdigest()==CSV_SHA256==projection["csv_sha256"] and len(raw)==604166,
        "Opaque source member bytes differ")
    lines=raw.splitlines(keepends=True);require(len(lines)==8761,"Source hourly record count differs")
    require(lines[0].rstrip(b"\r\n").decode("latin1").split(",")==projection["headers"],"Source header metadata differs")
    rows=[];offset=len(lines[0]);flags={};first=datetime(2017,12,1)
    for index,line in enumerate(lines[1:]):
        cells=line.rstrip(b"\r\n").split(b",")
        require(len(cells)==14 and b'"' not in line,"Source record framing differs")
        # Only the explicitly approved fields are decoded. Count/weather byte
        # strings remain uninterpreted through this preexecution audit.
        date=datetime.strptime(cells[0].decode("ascii"),"%d/%m/%Y")
        hour=int(cells[2].decode("ascii"));flag=cells[13].decode("ascii")
        stamp=date.replace(hour=hour);require(stamp==first+timedelta(hours=index),"Source calendar has a gap or duplicate")
        require(flag in ("Yes","No"),"Unexpected operation flag")
        rows.append({"byte_start":offset,"byte_stop":offset+len(line),"csv_line":index+2,
            "functioning_day":flag,"timestamp":iso(stamp)})
        flags[flag]=flags.get(flag,0)+1;offset+=len(line)
    require(rows==projection["rows"] and flags==projection["operation_flag_counts"]=={"No":295,"Yes":8465},"Calendar byte projection differs")
    require(projection["projection_source_sha256"]==sha(ROOT/"source/prepare_data38.py"),"Calendar producer source changed")
    roster=read(ROOT/"inputs/calendar_roster38.json");row_by_stamp={row["timestamp"]:row for row in rows}
    require(roster["count_values_decoded_or_parsed"]==roster["weather_values_decoded_or_parsed"]==0 and
        roster["projection_sha256"]==sha(ROOT/"inputs/calendar_projection38.json") and
        roster["future_operation_flag_used_as_predictor_or_filter"] is False and
        roster["later_test_histories_may_include_earlier_test_targets"] is True,"Calendar cohort role differs")
    for name in PARTITIONS:
        expected=calendar_queries(name);r=roster["partitions"][name]
        require(r["queries"]==expected and r["query_ids_sha256"]==newline_digest([v["id"] for v in expected]) and
            r["timestamp_complete_targets"]==len(expected) and r["closure_filter_applied"] is False and
            r["measurement_missingness_not_examined"] is True and
            r["targets_flagged_nonfunctional"]==sum(row_by_stamp[v["target_timestamp"]]["functioning_day"]=="No" for v in expected),
            "Metadata-only split/closure roster differs")
    return {"status":"PASS","archive_sha256":ARCHIVE_SHA256,"csv_sha256":CSV_SHA256,
        "calendar_rows":len(rows),"partition_sizes":{n:PARTITIONS[n][2] for n in PARTITIONS},
        "count_values_decoded_or_parsed":0,"weather_values_decoded_or_parsed":0,
        "operation_rows_retained":True,"prior_exposure_scope_not_absolute":True}

def audit_protocol(p):
    require(p["schema"]=="step38-training-transfer-protocol-v1" and p["step"]==38 and p["seed"]==3801,
        "Protocol identity/seed differs")
    require(p["maps"]==maps_roster() and p["representations"]==REP_CONFIG,"Frozen 18-map roster/configuration differs")
    cls=p["class"]
    require(cls["kappa"]=="2" and cls["prediction_interval"]==["0","1"] and cls["assurance"]==ASSURANCE,
        "Certificate class differs")
    require(all(cls[k] is False for k in ("native_transform_error_enclosed","raw_input_lipschitz_claim","population_risk_claim","future_response_coverage_claim")),
        "Unsupported certificate assurance")
    data=p["data"]
    fixed={"uci_id":560,"target":"rented_bike_count","archive":"inputs/seoul_bike_uci560.zip","member":"SeoulBikeData.csv",
        "archive_sha256":ARCHIVE_SHA256,"lag_hours":24,"horizon_hours":24,"origin_hour":23,
        "expected_n":{n:PARTITIONS[n][2] for n in PARTITIONS},"partition_basis":"target_timestamp",
        "train_start":"2017-12-01","train_end":"2018-07-31","validation_start":"2018-08-01","validation_end":"2018-08-31",
        "test_start":"2018-09-01","test_end":"2018-11-30","targets_clipped":False,
        "source_timezone":"local_naive_Seoul_date_and_hour_no_UTC_reinterpretation",
        "target_normalization":"Exact(raw-training_target_min)/(training_target_max-training_target_min)",
        "degenerate_range_action":"STOP_RETAIN_FAILURE","previously_revealed_targets_may_be_later_history":True,
        "no_online_refit":True,"features":"Onlycausallyavailableowncounthistory;noweathercalendaroroperationflagpredictors",
        "operation_closures":"Keepallrecordedhoursincludingclosures;recordedrentalsnotlatentunconstraineddemand"}
    require(all(data.get(k)==v for k,v in fixed.items()),"Dataset, split, feature or access specification differs")
    heads=p["heads"]
    require(heads["recipes"]==list(RECIPES) and heads["expected_count"]==90 and heads["anchor_count"]==32 and
        heads["ridge_lambda"]=="1/1000000" and heads["legacy_extension_max_updates"]==4000 and
        heads["legacy_check_every"]==20 and heads["legacy_gap_sse"]=="1/10000000" and
        heads["distance_bits"]==80 and heads["grid_bits"]==40 and heads["solver_settings"]==PRIMARY_SETTINGS and
        heads["solver_version"]=="1.0.5" and heads["legacy32guard_unchanged"] is True and
        heads["original_training_labels_in_objective"] is True and heads["fallbacks_retained"] is True,
        "Frozen head/solver roster differs")
    require(heads["extension_formula"]=="clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)","Extension formula differs")
    baselines=p["baselines"]
    require(baselines["names"]==list(BASELINES) and baselines["class_membership_certified"] is False and
        baselines["clipped"] is False and baselines["hgb_parameters"]=={
        "loss":"squared_error","learning_rate":0.05,"max_iter":200,"max_leaf_nodes":15,"max_depth":None,
        "min_samples_leaf":20,"l2_regularization":1.0,"max_features":1.0,"max_bins":255,"categorical_features":None,
        "monotonic_cst":None,"interaction_cst":None,"warm_start":False,"early_stopping":False,"random_state":3801},
        "Ordinary baseline roster/parameters differ")
    selection=p["selection"]
    require(selection["objective"]=="minimum_validation_MSE_upper_then_deployment_payload_bytes_then_lexical_candidate_id" and
        selection["per_recipe_maps"]==18 and selection["legacybest_candidates"]==36 and selection["ordinarybest_candidates"]==3 and
        selection["eligibility_filter_used_for_primary_selection"] is False and selection["epsilon_factor"]=="19/20" and
        selection["no_validation_refit"] is True and selection["no_test_reselection"] is True,"Selection rule differs")
    ev=p["evaluation"];wanted=[m+"::"+r for m in maps_roster() for r in RECIPES]+["baseline::"+n for n in BASELINES]
    require(ev["prediction_ids"]==wanted and ev["all_panel_test_outcomes_retained"] is True and ev["primary_metric"]=="MSE",
        "All-roster evaluation differs")
    gate=ev["practical_gate"]
    require(gate["positive_legacy_denominator_required"] is True and gate["five_percent_MSE_gain"]=="20*new_test_upper<=19*legacybest_test_lower" and
        gate["ordinary_baseline_comparison"]=="new_test_upper<=ordinarybest_test_lower" and
        gate["undefined_zero_denominator"]=="UNDEFINED_RELATIVE_GAIN_notasuccess","Frozen practical criterion differs")
    secondary=ev["secondary_test_class_diagnosis"]
    require(secondary["enabled"] is True and secondary["maps"]==18 and secondary["after"]=="alltestpredictionandtargetrecordscomplete" and
        secondary["uses_test_labels"] is True and secondary["head_promotion_allowed"] is False and
        secondary["selection_changes_allowed"] is False and secondary["cost_reported_separately"] is True,"Secondary assessment role differs")
    execution=p["execution"]
    require(execution["single_primary_pass"] is True and execution["phase_order"]==["train","validate","test"] and
        execution["phase_budget_seconds"]=={"train":1800,"validate":1200,"test":2400} and execution["threads"]==1 and
        execution["native_representation_budget_seconds"]==600 and execution["no_outcome_driven_reruns"] is True and
        execution["reserved_patients_accessed"]==0 and execution["inherited_study_gates"]=={
        "G3":"NOT_ESTABLISHED","G4":"NOT_STARTED","natural_study":"NO_GO"},"Execution/retained-study boundaries differ")
    return {"status":"PASS","maps":18,"trained_heads":90,"ordinary_baselines":3,"per_query_predictions":93,
        "criterion":"20*qp_all_upper<=19*legacybest_lower AND qp_all_upper<=ordinarybest_lower; legacy lower positive",
        "secondary_classes":18,"selection_uses_validation_only":True,"test_reselection_forbidden":True}

def verify_sources():
    for name,wanted in BASE_HASHES.items():require(sha(ROOT/name)==wanted,"Inherited source changed: "+name)
    tree=ast.parse(Path(__file__).read_text());imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(n.name.split(".")[0] for n in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:imports.append(node.module.split(".")[0])
    require(not {"learners38","full_graph37","finite_feature35","run38","data38"}.intersection(imports),
        "Independent auditor imports a scientific producer")
    return {"status":"PASS","immutable_inherited_modules":len(BASE_HASHES),"scientific_producer_imports":0,
        "operational_map_replay_scope":"Pinned representation runtime only; proof math independently reconstructed"}

def independent_selfcheck():
    checks=dict(graph_math.independent_selfcheck());verify_sources()
    n=40;z=[[F(0),F(0)] for _ in range(n)];ids=["artificial-anchor:"+str(i) for i in range(n)]
    ys=[F(3,4)]*n;feature="artificial:constant40";cid=class_identity(feature)
    candidate={"source":"clipped_mean_constant","value":"3/4","values":["3/4"]*n,"sse":"0","mean":"0"}
    fit={"role":"train","head_promotion_allowed":True,"feature_id":feature,"class_id":cid,"kappa":"2",
        "dimension":2,"feature_digest":digest(z),"target_digest":digest(ys),"ids":ids,"n":n,
        "primal_candidates":[candidate],"selected_qp_source":candidate["source"],"selected_envelope_source":candidate["source"]}
    head={"schema":HEAD_SCHEMA,"kind":"symmetric_lipschitz_extension","recipe":"qp_all","feature_id":feature,
        "class_id":cid,"dimension":2,"kappa":"2","output_interval":["0","1"],"assurance":ASSURANCE,
        "anchor_ids":ids,"anchor_indices":list(range(n)),"anchor_features":canon(z),"anchor_values":["3/4"]*n,
        "training_fit_path":"artificial","training_fit_sha256":"0"*64,"training_fit_role":"train",
        "selected_training_source":candidate["source"],"training_anchor_risk":"0",
        "formula":"clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)"}
    require(audit_head(head,fit,z,ys,ids)["anchor_count"]==40,"Full-head guard still capped at32")
    checks["full_anchor_head_above32_with_exact_duplicates"]=True
    queries=[[F(0),F(0)],[F(1),F(0)],[F(1),F(1)]];qids=["artificial-query:"+str(i) for i in range(3)]
    floors=[0]*n+[1<<80]*n+[isqrt(2*(1<<160))]*n
    dist={"schema":"step38-query-distances-v1","feature_id":feature,"anchor_ids":ids,"query_ids":qids,
        "n_anchor":n,"n_query":3,"dimension":2,"anchor_feature_digest":digest(z),"query_feature_digest":digest(queries),
        "features_exact":canon(queries),"bits":80,"common_binary_exponent":0,"pair_order":"query_then_anchor_rowmajor",
        "floor_scaled":list(map(str,floors)),"exact_pair_indices":list(range(2*n)),"target_values_accessed":False}
    cache=audit_distances(dist,z,queries,ids,qids);rows=[]
    def bx(low,high):return {"lower":str(low),"upper":str(high),"width":str(high-low)}
    for i,name in enumerate(qids):
        lo,hi=cache["lower"][i][0],cache["upper"][i][0]
        lower=(F(3,4)-2*hi,F(3,4)-2*lo);upper=(F(3,4)+2*lo,F(3,4)+2*hi)
        # Analytic truth is constant3/4. At distance1 the raw envelopes are
        # -5/4 and11/4: clipping either before averaging would incorrectly give1/2.
        raw=(F(3,4)-(hi-lo),F(3,4)+(hi-lo))
        rows.append({"query_id":name,"lower_envelope":bx(*lower),"upper_envelope":bx(*upper),
            "raw_prediction":bx(*raw),"prediction":bx(*raw)})
    pred={"schema":"step38-head-prediction-v1","feature_id":feature,"class_id":cid,"head_id":"head38:"+digest(head),
        "query_ids":qids,"query_feature_digest":digest(queries),"n":3,"distance_digest":newline_digest(dist),
        "anchor_columns":list(range(n)),"rows":rows,"predictions":[r["prediction"] for r in rows],"targets_accessed":False,
        "assurance":ASSURANCE,"future_response_coverage_claim":False,"native_transform_error_enclosed":False}
    audit_prediction(pred,head,queries,qids,dist,cache)
    require(interval(rows[0]["prediction"])==(F(3,4),F(3,4)) and interval(rows[1]["prediction"])==(F(3,4),F(3,4)),
        "Anchor interpolation or clip-after-midpoint invariant failed")
    require(interval(rows[2]["prediction"])[1]-interval(rows[2]["prediction"])[0]==F(2,1<<80),"Irrational arithmetic width bound failed")
    checks["exact_anchor_interpolation_and_clip_after_midpoint"]=True
    checks["irrational_query_arithmetic_width_independent_of_anchor_count"]=True
    def reject(function):
        try:function()
        except (AuditError,old.AuditError,graph_math.AuditError):return
        raise AuditError("Adversarial artificial fixture was accepted")
    broken=deepcopy(dist);broken["floor_scaled"][-1]=str(int(broken["floor_scaled"][-1])+1)
    reject(lambda:audit_distances(broken,z,queries,ids,qids));checks["wrong_query_floor_rejected"]=True
    bad_head=deepcopy(head);bad_head["anchor_features"][0][0]="1"
    bad_pred=deepcopy(pred);bad_pred["head_id"]="head38:"+digest(bad_head)
    reject(lambda:audit_prediction(bad_pred,bad_head,queries,qids,dist,cache));checks["shared_anchor_geometry_substitution_rejected"]=True
    bad_fit=deepcopy(fit);bad_fit["role"]="assessment_diagnostic";bad_fit["head_promotion_allowed"]=False
    reject(lambda:audit_head(head,bad_fit,z,ys,ids));checks["assessment_witness_head_promotion_rejected"]=True
    bad_pred=deepcopy(pred);bad_pred["predictions"][1]=bx(F(1,2),F(1,2))
    reject(lambda:audit_prediction(bad_pred,head,queries,qids,dist,cache));checks["clip_before_midpoint_prediction_rejected"]=True
    new=bx(F(19,20),F(19,20));legacy=bx(1,1);ordinary=bx(1,1)
    expected={"legacy_positive_risk_denominator":True,"at_least_five_percent_mse_improvement":True,
        "no_worse_than_validation_selected_ordinary_baseline":True,"numerical_practical_transfer_criterion":True,
        "relative_gain_status":"DEFINED","five_percent_refers_to":"MSE_not_RMSE","population_noninferiority_claim":False}
    audit_practical_gate(new,legacy,ordinary,expected);checks["five_percent_MSE_gate_equality"]=True
    return checks

def artificial_preflight():
    started=time.perf_counter();checks=independent_selfcheck()
    result={"schema":"step38-independent-artificial-checks-v1","status":"PASS","checks":checks,
        "retained_producer_fixtures":audit_artificial_fixture(),
        "artificial_only":True,"scientific_count_values_decoded":0,"scientific_feature_or_label_arrays_read":False,
        "source_sha256":{name:sha(ROOT/name) for name in ("source/audit38.py","source/independent_math35.py","source/audit_math37.py")},
        "audit_wall_seconds":time.perf_counter()-started}
    write(ROOT/"tests/artificial_audit38.json",result);return result

def audit_artificial_fixture():
    path=ROOT/"tests/learner_fixture38.json";fixture=read(path)
    require(fixture["schema"]=="step38-artificial-learner-fixture-v1" and fixture["scientific_data_accessed"] is False,
        "Artificial integration fixture has an invalid scope")
    z,y,ids=fixture["training_z"],fixture["training_y"],fixture["training_ids"];indices=fixture["anchor_indices32"]
    fits=[]
    for fit,iz in ((fixture["full_fit"],list(range(len(z)))),(fixture["small_fit"],indices)):
        fits.append(audit_fit(fit,[z[i] for i in iz],[y[i] for i in iz],[ids[i] for i in iz]))
    old.audit_affine_record(fixture["affine_fit"]);old.audit_qp(fixture["legacy_fit"])
    query_z,query_ids,dist=fixture["query_z"],fixture["query_ids"],fixture["distances"]
    cache=audit_distances(dist,z,query_z,ids,query_ids);heads=[];predictions=[]
    require(set(fixture["heads"])==set(RECIPES)==set(fixture["predictions"])==set(fixture["losses"]),
        "Artificial recipe roster differs")
    for recipe in RECIPES:
        head=fixture["heads"][recipe]
        if recipe in ("qp32","qp_all","envelope_all"):
            ix=indices if recipe=="qp32" else list(range(len(z)));fit=fixture["small_fit"] if recipe=="qp32" else fixture["full_fit"]
            heads.append(audit_head(head,fit,[z[i] for i in ix],[y[i] for i in ix],[ids[i] for i in ix]))
        else:heads.append(audit_head(head))
        record=fixture["predictions"][recipe];predictions.append(audit_prediction(record,head,query_z,query_ids,dist,cache))
        audit_loss(record["predictions"],fixture["original_loss_targets"],fixture["losses"][recipe])
    return {"status":"PASS","fixture_sha256":sha(path),"new_training_fits":fits,"head_membership":heads,
        "predictions":predictions,"distance_geometry":cache["summary"],"original_label_losses_verified":5,
        "scientific_data_accessed":False}

def verify_protocol_seal():
    seal=verify_bound_seal("protocol_seal38.json","step38-protocol-source-input-seal-v1")
    require(seal["protocol_sha256"]==sha(ROOT/"protocol38.json") and seal["new_dataset_count_values_parsed_before_seal"]==0,
        "Protocol seal source/exposure binding differs")
    required={"protocol38.json",*BASE_HASHES,"source/audit38.py","source/data38.py","source/run38.py","source/learners38.py",
        "source/prepare_data38.py","source/protocol38.py","inputs/seoul_bike_uci560.zip","inputs/calendar_projection38.json",
        "inputs/calendar_roster38.json","inputs/public_source38.json","inputs/public_metadata38.json","inputs/prior_exposure38.json",
        "tests/artificial_audit38.json","tests/artificial_data38.json","tests/artificial_learners38.json"}
    require(required<=set(seal["bound_files"]),"Protocol seal omits an operational/audit/data dependency")
    for relative in ("tests/artificial_audit38.json","tests/artificial_data38.json","tests/artificial_learners38.json"):
        test=read(ROOT/relative);require(test["status"]=="PASS","Missing artificial PASS: "+relative)
        for path,wanted in test["source_sha256"].items():require(sha(checked_path(path))==wanted,"Stale artificial source binding: "+path)
    verify_sources();protocol=read(ROOT/"protocol38.json");audit_protocol(protocol)
    return protocol,seal

def preflight():
    started=time.perf_counter();protocol,seal=verify_protocol_seal();events=audit_events()
    require([e["event"] for e in events]==["protocol_sealed"],"Scientific access or repeated preflight occurred before preexecution PASS")
    require(events[0]["protocol_seal_sha256"]==sha(ROOT/"protocol_seal38.json"),"Seal event binding differs")
    require(not (ROOT/"output/primary_started38.json").exists() and not (ROOT/"output/normalization38.json").exists(),
        "Scientific run/data exists before independent preexecution audit")
    sources=audit_source_metadata();checks=independent_selfcheck()
    result={"schema":"step38-independent-preexecution-gate-v1","status":"PASS","protocol_seal_sha256":sha(ROOT/"protocol_seal38.json"),
        "protocol_sha256":sha(ROOT/"protocol38.json"),"protocol":audit_protocol(protocol),"metadata":sources,
        "independent_source":verify_sources(),"selfchecks":checks,"scientific_count_values_decoded":0,
        "scientific_feature_or_label_arrays_read":False,"reserved_patients_accessed":0,"audit_wall_seconds":time.perf_counter()-started}
    write(ROOT/"audit/preexecution_gate38.json",result)
    append_event("independent_preexecution_PASS",gate_sha256=sha(ROOT/"audit/preexecution_gate38.json"),
        protocol_seal_sha256=sha(ROOT/"protocol_seal38.json"))
    return result

def verify_gate(relative,event_name,**bindings):
    gate=read(ROOT/relative);require(gate["status"]=="PASS","Required independent gate is not PASS: "+relative)
    for name,wanted in bindings.items():require(gate[name]==wanted,"Independent gate binding differs: "+name)
    _,event=require_once(audit_events(),event_name)
    require(event["gate_sha256"]==sha(ROOT/relative),"Independent gate changed after its event")
    return gate

def authorized_source_counts(timestamps,role):
    """Decode only timestamps already exposed in a completed authorized phase."""
    require(role in PARTITIONS and timestamps and len(timestamps)==len(set(timestamps)),"Invalid audit count access request")
    events=audit_events();_,ready=require_once(events,"train_count_access_completed" if role=="train" else
        "validation_count_access_completed" if role=="validation" else "all_test_forecasts_and_scores_complete")
    if role=="train":
        require((ROOT/"output/fit_seal38.json").is_file(),"Training audit preceded fit seal")
        cutoff=datetime(2018,7,31,23)
    elif role=="validation":
        require((ROOT/"output/model_seal38.json").is_file(),"Validation audit preceded selection seal")
        cutoff=datetime(2018,8,31,23)
    else:
        require(len([e for e in events if e["event"]=="test_target_revealed"])==91 and ready["test_n"]==91,
            "Test audit preceded complete chronological target reveal")
        cutoff=datetime(2018,11,30,23)
    require(all(datetime(2017,12,1)<=datetime.fromisoformat(t)<=cutoff for t in timestamps),"Audit attempted a future count access")
    projection=read(ROOT/"inputs/calendar_projection38.json");lookup={row["timestamp"]:row for row in projection["rows"]}
    archive=ROOT/"inputs/seoul_bike_uci560.zip";require(sha(archive)==ARCHIVE_SHA256,"Source archive changed before authorized replay")
    with zipfile.ZipFile(archive) as zipped:raw=zipped.read("SeoulBikeData.csv")
    require(hashlib.sha256(raw).hexdigest()==CSV_SHA256,"Opaque source member changed before authorized replay")
    counts={}
    for stamp in timestamps:
        row=lookup[stamp];line=raw[row["byte_start"]:row["byte_stop"]].rstrip(b"\r\n");cells=line.split(b",")
        require(len(cells)==14 and b'"' not in line,"Source row framing changed")
        token=cells[1]
        require(token and token.isdigit() and (len(token)==1 or token[0]!=48),"Missing/noncanonical integer count")
        value=int(token);require(0<=value<=2**53,"Count cannot be stored exactly in binary64 history")
        counts[stamp]=value
    return counts

def audit_access_record(record,stage,timestamps,maximum):
    require(record["stage"]==stage and record["requested_timestamps"]==timestamps and
        record["count_token_decodes"]==len(timestamps) and record["unique_count_timestamps"]==len(set(timestamps)) and
        record["maximum_allowed_timestamp"]==maximum and record["maximum_decoded_timestamp"]==max(timestamps) and
        record["archive_sha256"]==ARCHIVE_SHA256 and record["weather_values_decoded_or_parsed"]==0 and
        record["future_operation_flag_used"] is False and all(stamp<=maximum for stamp in timestamps),
        "Causal count-access receipt differs")

def audit_development_cohort(role):
    import numpy as np
    require(role in ("train","validation"),"Bulk assessment audit requires completed test access")
    record=read(ROOT/("output/cohorts/"+role+".json"));normal=read(ROOT/"output/normalization38.json")
    queries=calendar_queries(role);ids=[r["id"] for r in queries]
    needed=sorted({stamp for item in queries for stamp in lag_stamps(item)+[item["target_timestamp"]]})
    counts=authorized_source_counts(needed,role)
    raw_targets=[counts[item["target_timestamp"]] for item in queries]
    raw_persistence=[counts[item["origin_timestamp"]] for item in queries]
    require(normal["schema"]=="step38-training-target-normalization-v1" and normal["fitted_partition"]=="train" and
        normal["target_values_clipped"] is False and normal["future_data_used"] is False and
        type(normal["target_min"]) is int and type(normal["target_max"]) is int and
        normal["target_max"]>normal["target_min"] and normal["span"]==normal["target_max"]-normal["target_min"] and
        normal["degenerate_range_action"]=="STOP_RETAIN_FAILURE","Training normalization role/range differs")
    if role=="train":
        require(normal["target_min"]==min(raw_targets) and normal["target_max"]==max(raw_targets) and
            normal["raw_train_targets_sha256"]==newline_digest(raw_targets) and normal["train_target_ids_sha256"]==newline_digest(ids),
            "Training target normalization was not fitted on the declared training labels")
    def normalized(v):return F(v-normal["target_min"],normal["span"])
    y=list(map(normalized,raw_targets));persistence=list(map(normalized,raw_persistence))
    expected_X=np.asarray([[counts[t] for t in lag_stamps(item)] for item in queries],dtype=np.float64)
    require(record["schema"]=="step38-causal-cohort-v1" and record["name"]==role and record["target"]=="rented_bike_count" and
        record["n"]==len(queries) and record["ids"]==ids and record["target_timestamps"]==[r["target_timestamp"] for r in queries] and
        record["origin_timestamps"]==[r["origin_timestamp"] for r in queries] and record["lag_timestamps"]==[lag_stamps(r) for r in queries],
        "Development cohort roster/chronology differs")
    require(record["raw_targets"]==raw_targets and record["raw_persistence"]==raw_persistence and
        list(map(q,record["y"]))==y and list(map(q,record["persistence"]))==persistence and
        record["normalization_sha256"]==sha(ROOT/"output/normalization38.json") and record["archive_sha256"]==ARCHIVE_SHA256 and
        record["all_recorded_operation_states_retained"] is True and record["weather_or_future_operation_features_used"] is False and
        record["output_range_target_clipping"] is False,"Original source labels/history/normalization differ")
    xpath=checked_path(record["X_path"]);require(sha(xpath)==record["X_sha256"],"Development history file changed")
    X=np.load(xpath,allow_pickle=False)
    require(X.dtype==np.float64 and np.array_equal(X,expected_X),"Development history is not the declared original source window")
    access=read(ROOT/("output/data_access_"+role+"38.json"));maximum="2018-07-31T23:00:00" if role=="train" else "2018-08-31T23:00:00"
    audit_access_record(access,role,needed,maximum)
    require(access["status"]=="PASS" and access["cohort_path"]=="output/cohorts/"+role+".json" and
        access["cohort_sha256"]==sha(ROOT/access["cohort_path"]) and access["normalization_sha256"]==record["normalization_sha256"],
        "Development count access cohort binding differs")
    _,event=require_once(audit_events(),role+"_count_access_completed")
    require(event["cohort_sha256"]==sha(ROOT/access["cohort_path"]) and event["access_sha256"]==sha(ROOT/("output/data_access_"+role+"38.json")),
        "Completed development access event binding differs")
    return {"record":record,"X":X,"y":y,"persistence":persistence,"ids":ids,"normalization":normal,
        "summary":{"status":"PASS","partition":role,"n":len(ids),"source_count_tokens_replayed":len(needed),
            "original_targets_outside_range":sum(v<0 or v>1 for v in y),"source_history_width":24,
            "all_recorded_operation_states_retained":True}}

def array_digest(array):
    import numpy as np
    value=np.ascontiguousarray(array);h=hashlib.sha256()
    h.update(str(value.shape).encode("ascii"));h.update(value.dtype.str.encode("ascii"));h.update(value.tobytes())
    return h.hexdigest()

def load_context():
    """Load already-sealed operational payloads; never fit a new scientific model."""
    import numpy as np
    import pickle
    import representations35 as replay
    manifest=read(ROOT/"output/fit_manifest38.json");maps={};train_features={};heads={}
    replay.initialize_runtime()
    for name,info in manifest["maps"].items():
        map_path=bound_reference({"path":info["map_path"],"sha256":info["map_sha256"]})
        maps[name]=replay.deserialize_map(map_path.read_bytes())
        train_features[name]=np.load(bound_reference(info["training_features"]),allow_pickle=False)
    for row in manifest["candidate_rows"]:heads[row["candidate_id"]]=read(bound_reference({"path":row["head_path"],"sha256":row["head_sha256"]}))
    baseline=manifest["baselines"]["hist_gradient_boosting"]
    hgb=pickle.loads(bound_reference({"path":baseline["payload_path"],"sha256":baseline["payload_sha256"]}).read_bytes())
    training=read(ROOT/"output/cohorts/train.json")
    return {"manifest":manifest,"maps":maps,"train_features":train_features,"heads":heads,"hgb":hgb,"training":training}

def audit_training(protocol,training):
    import numpy as np
    from scipy.spatial.distance import pdist
    import representations35 as replay
    context=load_context();manifest=context["manifest"];X=training["X"];y=training["y"];ids=training["ids"]
    rows=manifest["candidate_rows"];maps=maps_roster();wanted=[m+"::"+r for m in maps for r in RECIPES]
    require(manifest["schema"]=="step38-fit-manifest-v1" and [r["candidate_id"] for r in rows]==wanted and
        sorted(manifest["maps"])==maps and set(manifest["baselines"])==set(BASELINES) and manifest["training_n"]==len(ids) and
        manifest["all_fitting_precedes_validation_access"] is True and manifest["selection_available"] is False,
        "Training manifest omitted or changed a frozen candidate")
    plan=read(ROOT/"output/training_plan38.json");epsilon=F(19,20)*sum(((p-t)**2 for p,t in zip(training["persistence"],y)),F(0))/len(y)
    anchors=[((2*j+1)*len(y))//64 for j in range(32)]
    require(plan["schema"]=="step38-training-plan-v1" and plan["n"]==len(y) and q(plan["epsilon"])==epsilon and
        q(plan["epsilon_physical_mse"])==epsilon*training["normalization"]["span"]**2 and
        plan["anchor_indices32"]==anchors and plan["anchor_ids32"]==[ids[i] for i in anchors] and
        plan["source_role"]=="train_only" and manifest["training_plan_sha256"]==sha(ROOT/"output/training_plan38.json"),
        "Frozen training anchor/epsilon plan differs")
    metadata=read(ROOT/"output/representation_metadata38.json")
    require(metadata["schema"]=="step35-frozen-representations-v1" and metadata["config"]==REP_CONFIG and
        metadata["seed"]==3801 and metadata["training_shape"]==list(X.shape) and metadata["training_sha256"]==array_digest(X) and
        metadata["probe_indices"]==anchors and metadata["probe_sha256"]==array_digest(X[anchors]) and
        metadata["factory_source_sha256"]==sha(ROOT/"source/representations35.py") and
        metadata["forecast_labels_used"] is False and metadata["no_heldout_input_or_label_used"] is True and
        len(metadata["map_order"])==18 and set(metadata["map_order"])==set(maps),"Representation training binding differs")
    for mode,record in metadata["scalers"].items():
        low,high=X.min(axis=0),X.max(axis=0);constant=low==high
        offset=low if mode=="minmax01" else X.mean(axis=0)
        divisor=high-low if mode=="minmax01" else X.std(axis=0,ddof=0);divisor=np.where(constant,1.,divisor)
        require(np.array_equal(np.asarray(record["offset"]),offset) and np.array_equal(np.asarray(record["divisor"]),divisor) and
            record["constant_mask"]==constant.tolist() and record["training_rows"]==len(X) and record["heldout_clipping"] is False,
            "Input scaler was not fitted on declared training rows")
    meta={r["name"]:r for r in metadata["maps"]};require(set(meta)==set(maps),"Representation metadata map roster differs")
    fit_summaries=[];head_summaries=[];total_pairs=0;model_rows={r["candidate_id"]:r for r in rows}
    for name in maps:
        info=manifest["maps"][name];map_meta=meta[name];restored=context["maps"][name];z=context["train_features"][name]
        path=checked_path(info["map_path"])
        require(path.stat().st_size==info["map_payload_bytes"]==map_meta["payload_bytes"] and
            sha(path)==info["map_sha256"]==map_meta["payload_sha256"] and restored.name==name==info["name"] and
            restored.family==info["family"]==map_meta["family"] and restored.scaling==info["scaling"]==map_meta["scaling"] and
            replay.numeric_bytes(restored)==info["map_numeric_bytes"] and
            z.dtype==np.float64 and np.isfinite(z).all() and z.shape==(len(ids),info["dimension"]) and
            info["dimension"]==map_meta["actual_width"] and np.array_equal(z,replay.transform_one(restored,X)),
            "Stored deployment map or training feature replay differs: "+name)
        exact=exact_matrix(z);fref=info["training_features"]
        require(fref["feature_digest"]==digest(exact) and info["feature_id"]=="feature38:"+newline_digest({
            "map":name,"map_sha256":info["map_sha256"],"train_feature_sha256":fref["sha256"]}) and
            info["class_id"]==class_identity(info["feature_id"]),"Stored feature/class identity differs")
        probes=restored.core.transform(X[anchors]);distances=pdist(probes,metric="euclidean");positive=distances[distances>0]
        median=float(np.median(positive)) if len(positive) else 1.;metric=1./median if len(positive) else 1.
        require(float(restored.metric_scale)==metric==map_meta["metric_scale"] and map_meta["positive_probe_pair_count"]==len(positive),
            "Feature metric scalar did not use fixed training probes")
        if restored.family=="nystrom":
            width=map_meta["requested_width"];landmarks=[((2*j+1)*len(X))//(2*width) for j in range(width)]
            require(map_meta["landmark_indices"]==landmarks and
                np.array_equal(restored.core.landmarks,restored.core.scaler.transform(X[landmarks])),"Nystrom landmarks are not training-only")
        binding={"cohort_path":"output/cohorts/train.json","cohort_sha256":sha(ROOT/"output/cohorts/train.json"),
            "feature_path":fref["path"],"feature_sha256":fref["sha256"],"map_sha256":info["map_sha256"]}
        full_row=model_rows[name+"::qp_all"];full_fit=read(bound_reference({"path":full_row["fit_path"],"sha256":full_row["fit_sha256"]}))
        small_row=model_rows[name+"::qp32"];small_fit=read(bound_reference({"path":small_row["fit_path"],"sha256":small_row["fit_sha256"]}))
        for fit,zfit,yfit,idfit,indices in ((full_fit,z,y,ids,list(range(len(ids)))),
            (small_fit,z[anchors],[y[i] for i in anchors],[ids[i] for i in anchors],anchors)):
            result=audit_fit(fit,zfit,yfit,idfit,expected={"role":"train","feature_id":info["feature_id"],"class_id":info["class_id"],
                "input_binding":{**binding,"anchor_indices":indices},"kappa":"2","settings":protocol["heads"]["solver_settings"]})
            fit_summaries.append(result);total_pairs+=result["pair_distances_verified"]
        for recipe in RECIPES:
            row=model_rows[name+"::"+recipe];head=context["heads"][row["candidate_id"]]
            fit=read(bound_reference({"path":row["fit_path"],"sha256":row["fit_sha256"]}));hp=checked_path(row["head_path"])
            if recipe in ("qp_all","envelope_all"):
                require(row["fit_path"]==full_row["fit_path"] and row["fit_sha256"]==full_row["fit_sha256"],
                    "Full-anchor QP/envelope heads do not share the independently audited training core")
            require(row["map"]==name and row["recipe"]==recipe and row["feature_id"]==head["feature_id"]==info["feature_id"] and
                row["class_id"]==head["class_id"]==info["class_id"] and row["dimension"]==info["dimension"] and
                row["family"]==info["family"] and row["scaling"]==info["scaling"] and row["training_only"] is True and
                row["assurance"]==ASSURANCE and row["head_payload_bytes"]==hp.stat().st_size and
                row["map_payload_bytes"]==path.stat().st_size and row["deployment_payload_bytes"]==hp.stat().st_size+path.stat().st_size,
                "Training candidate source/size identity differs")
            expected={"feature_id":info["feature_id"],"class_id":info["class_id"],"head_id":row["head_id"]}
            if recipe=="affine":
                require(fit["head"]==head and fit["head_id"]==row["head_id"] and fit["training_binding"]==binding,"Legacy affine training binding differs")
                old.audit_affine_record(fit);result=audit_head(head,expected=expected)
                center=[sum((v[j] for v in exact),F(0))/len(y) for j in range(len(exact[0]))]
                require(list(map(q,fit["projection"]["center_features"]))==center and
                    q(fit["projection"]["center_target"])==sum(y,F(0))/len(y),"Affine recentering used other labels/coordinates")
                yf=np.asarray([float(v) for v in y]);xc=z-z.mean(axis=0);yc=yf-yf.mean()
                beta=np.linalg.solve(xc.T@xc+len(z)*1e-6*np.eye(z.shape[1]),xc.T@yc)
                require(list(map(q,fit["projection"]["source_coefficients"]))==[F.from_float(float(v)) for v in beta],
                    "Affine ridge proposal did not use fixed training-only recipe")
            elif recipe=="extension32":
                old.audit_qp(fit);result=audit_head(head,expected=expected)
                require(fit["kind"]=="training" and fit["head"]==head and fit["head_id"]==row["head_id"] and
                    fit["training_binding"]=={**binding,"anchor_indices":anchors} and
                    fit["query_ids"]==[ids[i] for i in anchors] and exact_matrix(fit["features_exact"])==[exact[i] for i in anchors] and
                    list(map(q,fit["targets_exact"]))==[y[i] for i in anchors] and fit["work"]["max_updates"]==4000 and
                    fit["work"]["grid_bits"]==40 and fit["distance_table"]["bits"]==80,
                    "Legacy extension32 anchor roster/recipe differs")
            else:
                indices=anchors if recipe=="qp32" else list(range(len(ids)))
                expected.update({"recipe":recipe,"anchor_indices":indices,"training_fit_path":row["fit_path"],"training_fit_sha256":row["fit_sha256"]})
                result=audit_head(head,fit,z[indices],[y[i] for i in indices],[ids[i] for i in indices],expected)
            require(result["head_id"]==row["head_id"] and row["anchor_count"]==result["anchor_count"],"Head content identity/anchor size differs")
            head_summaries.append(result)
    baselines=manifest["baselines"];mean=sum(y,F(0))/len(y)
    require(q(baselines["training_mean"]["value"])==mean and
        baselines["training_mean"]["deployment_payload_bytes"]==len((json.dumps({"value":str(mean)},sort_keys=True,separators=(",",":"))+"\n").encode()) and
        baselines["persistence"]["deployment_payload_bytes"]==len(b'{"rule":"last_observed_count"}\n'),"Ordinary mean/persistence artifact differs")
    hg=context["hgb"];params=hg["model"].get_params()
    require(all(params[k]==v for k,v in protocol["baselines"]["hgb_parameters"].items()) and hg["model"].n_iter_==200 and
        hg["model"].n_features_in_==24 and np.array_equal(hg["scaler"].offset,context["maps"]["raw_minmax01"].core.offset) and
        np.array_equal(hg["scaler"].divisor,context["maps"]["raw_minmax01"].core.divisor) and
        baselines["hist_gradient_boosting"]["deployment_payload_bytes"]==checked_path(baselines["hist_gradient_boosting"]["payload_path"]).stat().st_size,
        "HGB fixed parameters/training scaler/deployment size differs")
    require([r["map"] for r in manifest["group_timings"]]==maps and all(
        isfinite(v) and v>=0 for r in manifest["group_timings"] for k,v in r.items() if k!="map"),"Physical cost group roster/values differ")
    return {"status":"PASS","maps":18,"heads":90,"training_class_fits":36,"training_pair_distances_verified":total_pairs,
        "fits":fit_summaries,"head_membership":head_summaries,"epsilon":str(epsilon),"original_target_normalization_verified":True,
        "training_only_representations_and_heads_verified":True,"stored_features_replayed_bitwise":True,
        "new_scientific_model_fits_by_auditor":0,"complete_deployment_bytes_verified":True,
        "representation_scope":"Operational stored binary64 features; no native ideal-real enclosure"}

def prevalidation():
    started=time.perf_counter();protocol,_=verify_protocol_seal()
    verify_gate("audit/preexecution_gate38.json","independent_preexecution_PASS",protocol_seal_sha256=sha(ROOT/"protocol_seal38.json"))
    seal=verify_bound_seal("output/fit_seal38.json","step38-fit-seal-v1")
    require(seal["protocol_seal_sha256"]==sha(ROOT/"protocol_seal38.json") and seal["head_count"]==90 and
        seal["validation_values_parsed"]==seal["test_values_parsed"]==0 and
        seal["fit_manifest_sha256"]==sha(ROOT/"output/fit_manifest38.json"),"Fit seal ancestry/counts differ")
    require({"output/normalization38.json","output/cohorts/train.json","output/cohorts/train_X.npy",
        "output/fit_manifest38.json","output/training_plan38.json","output/representation_metadata38.json"}<=set(seal["bound_files"]),
        "Fit seal omitted a training input, normalization, or model manifest")
    events=audit_events();p,_=require_once(events,"independent_preexecution_PASS");s,_=require_once(events,"train_count_access_started")
    c,_=require_once(events,"train_count_access_completed");f,fe=require_once(events,"fit_sealed")
    require(p<s<c<f and fe["seal_sha256"]==sha(ROOT/"output/fit_seal38.json") and
        not any(e["event"].startswith("validation_count") or e["event"].startswith("test_") for e in events),
        "Validation/test access preceded independent fit audit")
    training=audit_development_cohort("train");model_audit=audit_training(protocol,training)
    result={"schema":"step38-independent-prevalidation-gate-v1","status":"PASS",
        "protocol_seal_sha256":sha(ROOT/"protocol_seal38.json"),"fit_seal_sha256":sha(ROOT/"output/fit_seal38.json"),
        "fit_manifest_sha256":sha(ROOT/"output/fit_manifest38.json"),"training_cohort":training["summary"],"training_models":model_audit,
        "validation_count_values_decoded":0,"test_count_values_decoded":0,"reserved_patients_accessed":0,
        "audit_wall_seconds":time.perf_counter()-started}
    write(ROOT/"audit/prevalidation_gate38.json",result)
    append_event("independent_prevalidation_PASS",gate_sha256=sha(ROOT/"audit/prevalidation_gate38.json"),fit_seal_sha256=sha(ROOT/"output/fit_seal38.json"))
    return result

def audit_prediction_bundle(record,context,X,ids,persistence,role):
    """Bind each formula to immutable heads, causally available X and stored maps."""
    import numpy as np
    import representations35 as replay
    manifest=context["manifest"];wanted=[name+"::"+recipe for name in maps_roster() for recipe in RECIPES]+["baseline::"+n for n in BASELINES]
    require(set(record["maps"])==set(maps_roster()) and set(record["predictions"])==set(wanted) and len(record["predictions"])==93,
        "Prediction panel map/head roster differs")
    totals={"maps":18,"heads":90,"baselines":3,"query_rows":len(ids),"query_anchor_distances_verified":0,"head_prediction_enclosures_verified":0}
    features={};largest=F(0)
    for name in maps_roster():
        info=manifest["maps"][name];item=record["maps"][name]
        path=bound_reference(item["features"]);z=np.load(path,allow_pickle=False)
        require(str(path.relative_to(ROOT))=="output/features/"+role+"/"+name+".npy" and z.dtype==np.float64 and
            z.shape==(len(ids),info["dimension"]) and np.isfinite(z).all() and
            np.array_equal(z,replay.transform_one(context["maps"][name],X)) and
            item["features"]["feature_digest"]==digest(exact_matrix(z)),"Query stored features do not replay from causal inputs: "+name)
        features[name]=z;distpath=bound_reference(item["distances"]);distance=read(distpath)
        require(str(distpath.relative_to(ROOT))=="output/distances/"+role+"/"+name+".json" and
            item["distance_digest"]==newline_digest(distance),"Shared query receipt location/digest differs")
        cache=audit_distances(distance,context["train_features"][name],z,context["training"]["ids"],ids,expected={"feature_id":info["feature_id"]})
        totals["query_anchor_distances_verified"]+=cache["summary"]["distance_pairs_verified"]
        for recipe in RECIPES:
            cid=name+"::"+recipe;checked=audit_prediction(record["predictions"][cid],context["heads"][cid],z,ids,distance,cache)
            totals["head_prediction_enclosures_verified"]+=checked["prediction_enclosures_verified"]
            largest=max(largest,q(checked["maximum_prediction_width"]))
    baseline_values={"persistence":persistence,"training_mean":[q(manifest["baselines"]["training_mean"]["value"])]*len(ids),
        "hist_gradient_boosting":context["hgb"]["model"].predict(context["hgb"]["scaler"].transform(X)).tolist()}
    for name in BASELINES:
        prediction=record["predictions"]["baseline::"+name]
        require(prediction["baseline"]==name,"Ordinary baseline identity differs")
        audit_baseline_prediction(prediction,ids,baseline_values[name])
    require(set(record["timing"])=={"shared_transform_seconds","shared_distance_seconds","all_head_interval_seconds"} and
        all(isfinite(v) and v>=0 for v in record["timing"].values()),"Shared query-work timing scope differs")
    return {"summary":{**totals,"status":"PASS","maximum_prediction_width":str(largest),
        "stored_query_features_replayed_bitwise":True,"baseline_predictions_replayed":True,
        "targets_used_in_prediction_formula_audit":False},"features":features}

def selected_by_validation(rows):
    chosen=min(rows,key=lambda r:(q(r["validation_risk"]["upper"]),r["deployment_payload_bytes"],r["candidate_id"]))
    return {"candidate_id":chosen["candidate_id"],"validation_risk":chosen["validation_risk"],
        "deployment_payload_bytes":chosen["deployment_payload_bytes"],"eligible":chosen["validation_eligible"]}

def audit_validation(protocol,cohort,context):
    predictions=read(ROOT/"output/validation_predictions38.json");selection=read(ROOT/"output/validation_selection38.json")
    require(predictions["schema"]=="step38-validation-predictions-v1" and predictions["fit_seal_sha256"]==sha(ROOT/"output/fit_seal38.json") and
        predictions["query_ids"]==cohort["ids"],"Validation predictions have another frozen-fit/cohort binding")
    checked=audit_prediction_bundle(predictions,context,cohort["X"],cohort["ids"],cohort["persistence"],"validation")
    require(selection["schema"]=="step38-validation-selection-v1" and selection["validation_n"]==31 and
        selection["fit_seal_sha256"]==predictions["fit_seal_sha256"] and selection["selection_uses_validation_only"] is True and
        selection["test_values_parsed"]==0 and selection["timing"]==predictions["timing"],"Selection stage/binding differs")
    epsilon=q(read(ROOT/"output/training_plan38.json")["epsilon"])
    require(q(selection["epsilon"])==epsilon,"Selection epsilon changed")
    rows=selection["candidate_rows"];baselines=selection["baseline_rows"]
    require(len(rows)==90 and [r["candidate_id"] for r in rows]==[r["candidate_id"] for r in context["manifest"]["candidate_rows"]] and
        [r["candidate_id"] for r in baselines]==["baseline::"+name for name in BASELINES],"Validation selection panel is incomplete")
    for fitted,row in zip(context["manifest"]["candidate_rows"],rows):
        require(all(row.get(k)==v for k,v in fitted.items()),"A fitted candidate changed during validation")
    for row in rows+baselines:
        cid=row["candidate_id"];prediction=predictions["predictions"][cid]
        audit_loss(prediction["predictions"],cohort["y"],row["validation_risk"])
        require(row["validation_prediction_digest"]==newline_digest(prediction) and
            row["validation_eligible"] is (q(row["validation_risk"]["upper"])<=epsilon),"Validation risk/eligibility binding differs")
        if cid.startswith("baseline::"):
            name=cid.split("::",1)[1]
            require(row["baseline"]==name and row["deployment_payload_bytes"]==context["manifest"]["baselines"][name]["deployment_payload_bytes"],
                "Validation ordinary baseline deployment size differs")
    expected={recipe:selected_by_validation([r for r in rows if r["recipe"]==recipe]) for recipe in RECIPES}
    expected["legacybest"]=selected_by_validation([r for r in rows if r["recipe"] in ("affine","extension32")])
    expected["ordinarybest"]=selected_by_validation(baselines)
    require(selection["selections"]==expected,"Validation selections differ from frozen MSE-upper/bytes/ID rule")
    return {"status":"PASS","predictions":checked["summary"],"original_label_losses_verified":93,
        "selection_count":7,"selected_ids":{k:v["candidate_id"] for k,v in expected.items()},
        "eligibility_filter_not_used_for_primary_selection":True,"validation_cohort":cohort["summary"]}

def verify_fit_and_model_gates():
    protocol,_=verify_protocol_seal()
    verify_gate("audit/preexecution_gate38.json","independent_preexecution_PASS",protocol_seal_sha256=sha(ROOT/"protocol_seal38.json"))
    fit=verify_bound_seal("output/fit_seal38.json","step38-fit-seal-v1")
    verify_gate("audit/prevalidation_gate38.json","independent_prevalidation_PASS",protocol_seal_sha256=sha(ROOT/"protocol_seal38.json"),
        fit_seal_sha256=sha(ROOT/"output/fit_seal38.json"))
    model=verify_bound_seal("output/model_seal38.json","step38-model-selection-seal-v1")
    require(fit["protocol_seal_sha256"]==model["protocol_seal_sha256"]==sha(ROOT/"protocol_seal38.json") and
        model["fit_seal_sha256"]==sha(ROOT/"output/fit_seal38.json") and model["test_values_parsed"]==0 and
        model["selection_sha256"]==sha(ROOT/"output/validation_selection38.json"),"Final model/selection seal ancestry differs")
    for name,wanted in fit["bound_files"].items():require(model["bound_files"].get(name)==wanted,"Final model seal omitted/changed a fit-sealed artifact")
    return protocol,fit,model

def pretest():
    started=time.perf_counter();protocol,fit,model=verify_fit_and_model_gates();events=audit_events()
    f,_=require_once(events,"independent_prevalidation_PASS");s,_=require_once(events,"validation_count_access_started")
    c,_=require_once(events,"validation_count_access_completed");m,me=require_once(events,"model_selection_sealed")
    require(f<s<c<m and me["seal_sha256"]==sha(ROOT/"output/model_seal38.json") and
        not any(e["event"].startswith("test_") for e in events) and not (ROOT/"output/test_access_state38.json").exists(),
        "Test access preceded independent final-selection audit")
    cohort=audit_development_cohort("validation");context=load_context();checked=audit_validation(protocol,cohort,context)
    require(model["selected_ids"]==checked["selected_ids"],"Final selection seal differs from independently verified choices")
    result={"schema":"step38-independent-pretest-gate-v1","status":"PASS","protocol_seal_sha256":sha(ROOT/"protocol_seal38.json"),
        "fit_seal_sha256":sha(ROOT/"output/fit_seal38.json"),"model_seal_sha256":sha(ROOT/"output/model_seal38.json"),
        "selection_sha256":sha(ROOT/"output/validation_selection38.json"),"validation_and_selection":checked,
        "fitted_model_bytes_unchanged_since_prevalidation":True,"test_count_values_decoded":0,"reserved_patients_accessed":0,
        "audit_wall_seconds":time.perf_counter()-started}
    write(ROOT/"audit/pretest_gate38.json",result)
    append_event("independent_pretest_PASS",gate_sha256=sha(ROOT/"audit/pretest_gate38.json"),model_seal_sha256=sha(ROOT/"output/model_seal38.json"))
    return result

def audit_test_forecasts(protocol,context):
    import numpy as np
    events=audit_events();gate_index,_=require_once(events,"independent_pretest_PASS")
    finished_index,finished=require_once(events,"all_test_forecasts_and_scores_complete")
    queries=calendar_queries("test");model_sha=sha(ROOT/"output/model_seal38.json");normal=read(ROOT/"output/normalization38.json")
    cohort=read(ROOT/"output/cohorts/test.json")
    require(cohort["schema"]=="step38-completed-test-cohort-v1" and cohort["n"]==91 and
        cohort["ids"]==[v["id"] for v in queries] and cohort["target_timestamps"]==[v["target_timestamp"] for v in queries] and
        cohort["origin_timestamps"]==[v["origin_timestamp"] for v in queries] and len(cohort["queries"])==91 and
        cohort["sequential_prediction_then_label_verified_by_loader"] is True,"Completed test cohort roster differs")
    needed=sorted({stamp for item in queries for stamp in lag_stamps(item)+[item["target_timestamp"]]})
    counts=authorized_source_counts(needed,"test")
    all_predictions={cid:[] for cid in protocol["evaluation"]["prediction_ids"]};all_features={name:[] for name in maps_roster()}
    targets=[];raw_targets=[];timings=[];summaries=[];previous_reveal=gate_index;overlap_total=0
    stage_names=("test_history_access_started","test_history_access_completed","test_prediction_receipt_written",
        "test_predictions_sealed","test_target_access_started","test_target_revealed")
    stage_events={name:{} for name in stage_names}
    for index,event in enumerate(events):
        if event["event"] in stage_events:
            name=event["event"];qid=event["query_id"]
            require(qid not in stage_events[name],"Repeated chronological test stage: "+name)
            stage_events[name][qid]=(index,event)
    for name,by_query in stage_events.items():
        require(set(by_query)==set(cohort["ids"]),"Missing/extra chronological test stage: "+name)
    for index,query in enumerate(queries):
        qid=query["id"];ordered=[stage_events[name][qid][0] for name in stage_names]
        require(previous_reveal<ordered[0] and ordered==sorted(set(ordered)) and ordered[-1]<finished_index,
            "Chronological forecast, prediction seal and target reveal ordering failed")
        previous_reveal=ordered[-1];ev={name:stage_events[name][qid][1] for name in stage_names}
        for name in stage_names:
            if name!="test_prediction_receipt_written":require(ev[name]["query_index"]==index,"Chronological query index differs")
        history_path=f"output/test_history/{index:03d}.json";history=read(ROOT/history_path)
        timestamps=lag_stamps(query);history_counts=[counts[t] for t in timestamps]
        raw_persistence=counts[query["origin_timestamp"]];persistence=F(raw_persistence-normal["target_min"],normal["span"])
        overlap=[t for t in timestamps if t in [r["target_timestamp"] for r in queries[:index]]];overlap_total+=len(overlap)
        require(history["schema"]=="step38-causal-test-history-v1" and history["query_id"]==qid and history["query_index"]==index and
            history["target_timestamp"]==query["target_timestamp"] and history["origin_timestamp"]==query["origin_timestamp"] and
            history["history_timestamps"]==timestamps and history["history_counts"]==history_counts and
            history["raw_persistence"]==raw_persistence and q(history["persistence"])==persistence and
            history["model_seal_sha256"]==model_sha and history["normalization_sha256"]==sha(ROOT/"output/normalization38.json") and
            history["previously_revealed_test_targets_in_history"]==overlap and history["current_target_count_decoded"] is False and
            history["weather_or_future_operation_features_used"] is False,"Test causal history/count/prefix binding differs")
        audit_access_record(history["access"],"test_history",timestamps,query["origin_timestamp"])
        xpath=checked_path(history["X_path"]);require(sha(xpath)==history["X_sha256"],"Test history array bytes changed")
        X=np.load(xpath,allow_pickle=False)
        require(X.dtype==np.float64 and X.shape==(1,24) and np.array_equal(X,np.asarray([history_counts],dtype=np.float64)),
            "Test source history array differs")
        require(ev["test_history_access_started"]["model_seal_sha256"]==model_sha and
            ev["test_history_access_started"]["maximum_allowed_timestamp"]==query["origin_timestamp"] and
            ev["test_history_access_completed"]["history_receipt_path"]==history_path and
            ev["test_history_access_completed"]["history_receipt_sha256"]==sha(ROOT/history_path) and
            ev["test_history_access_completed"]["current_target_count_decoded"] is False and
            ev["test_history_access_completed"]["previously_revealed_test_targets_in_history"]==overlap,
            "Test history event/receipt binding differs")
        seal_path=f"output/test_prediction_seals/{index:03d}.json";seal=read(ROOT/seal_path)
        require(seal["schema"]=="step38-test-prediction-byte-seal-v1" and seal["query_id"]==qid and seal["query_index"]==index and
            seal["model_seal_sha256"]==model_sha and seal["history_receipt_path"]==history_path and
            seal["history_receipt_sha256"]==sha(ROOT/history_path) and seal["prediction_ids"]==protocol["evaluation"]["prediction_ids"] and
            seal["prediction_count"]==93 and seal["current_target_count_decoded"] is False,"Per-query prediction byte seal differs")
        predpath=checked_path(seal["prediction_path"]);require(sha(predpath)==seal["prediction_sha256"],"Forecast bytes changed after seal")
        record=read(predpath)
        require(record["schema"]=="step38-sealed-test-predictions-v1" and record["query_id"]==qid and
            record["target_timestamp"]==query["target_timestamp"] and record["origin_timestamp"]==query["origin_timestamp"] and
            record["model_seal_sha256"]==model_sha and record["history_receipt_path"]==history_path and
            record["history_receipt_sha256"]==sha(ROOT/history_path) and record["sealed_before_current_target_reveal"] is True and
            record["future_target_used_in_prediction"] is False,"Current forecast info boundary differs")
        for name in ("test_prediction_receipt_written","test_predictions_sealed"):
            require(ev[name]["prediction_path"]==seal["prediction_path"] and ev[name]["prediction_sha256"]==seal["prediction_sha256"] and
                ev[name]["prediction_count"]==93,"Forecast file event binding differs")
        require(ev["test_predictions_sealed"]["prediction_seal_path"]==seal_path and
            ev["test_predictions_sealed"]["prediction_seal_sha256"]==sha(ROOT/seal_path) and
            ev["test_predictions_sealed"]["current_target_count_decoded"] is False and
            ev["test_target_access_started"]["prediction_seal_sha256"]==sha(ROOT/seal_path),"Prediction-before-target byte seal binding differs")
        checked=audit_prediction_bundle(record,context,X,[qid],[persistence],"test/q"+str(index).zfill(3));summaries.append(checked["summary"])
        for cid in all_predictions:all_predictions[cid].append(record["predictions"][cid]["predictions"][0])
        for name in all_features:all_features[name].append(checked["features"][name][0])
        timings.append(record["timing"])
        target_path=f"output/test_targets/{index:03d}.json";target=read(ROOT/target_path)
        raw=counts[query["target_timestamp"]];value=F(raw-normal["target_min"],normal["span"])
        require(target["schema"]=="step38-revealed-test-target-v1" and target["query_id"]==qid and target["query_index"]==index and
            target["target_timestamp"]==query["target_timestamp"] and target["origin_timestamp"]==query["origin_timestamp"] and
            target["raw_target"]==raw and q(target["y"])==value and target["normalization_sha256"]==sha(ROOT/"output/normalization38.json") and
            target["prediction_seal_path"]==seal_path and target["prediction_seal_sha256"]==sha(ROOT/seal_path) and
            target["model_seal_sha256"]==model_sha and target["target_value_clipped"] is False,"Original revealed test target differs")
        audit_access_record(target["access"],"test_target",[query["target_timestamp"]],query["target_timestamp"])
        require(ev["test_target_revealed"]["target_path"]==target_path and ev["test_target_revealed"]["target_sha256"]==sha(ROOT/target_path) and
            ev["test_target_revealed"]["prediction_seal_sha256"]==sha(ROOT/seal_path) and
            ev["test_target_revealed"]["next_query_index"]==index+1 and
            ev["test_target_revealed"]["chronological_test_status"]==("COMPLETE" if index==90 else "READY"),
            "Target reveal event/prefix differs")
        saved=cohort["queries"][index]
        expected_query={**query,"prediction_path":seal["prediction_path"],"prediction_sha256":seal["prediction_sha256"],
            "history_receipt_path":history_path,"history_receipt_sha256":sha(ROOT/history_path),
            "target_reveal":{**target,"path":target_path,"sha256":sha(ROOT/target_path)}}
        require(saved==expected_query,"Completed test query aggregation differs from retained receipt sequence")
        targets.append(value);raw_targets.append(raw)
    require(list(map(q,cohort["y"]))==targets and cohort["raw_targets"]==raw_targets,"Completed assessment labels differ")
    state=read(ROOT/"output/test_access_state38.json")
    require(state["schema"]=="step38-chronological-test-access-state-v1" and state["status"]=="COMPLETE" and state["next_index"]==91 and
        state["pending_query"] is None and state["previously_revealed_test_targets"]==[r["target_timestamp"] for r in queries] and
        state["model_seal_sha256"]==model_sha,"Final causal access state differs")
    summary={"status":"PASS","test_queries":91,"predictions_per_query":93,"prediction_byte_seals_verified":91,
        "query_anchor_distances_verified":sum(s["query_anchor_distances_verified"] for s in summaries),
        "head_prediction_enclosures_verified":sum(s["head_prediction_enclosures_verified"] for s in summaries),
        "ordinary_baseline_predictions_replayed":91*3,"maximum_prediction_width":str(max(q(s["maximum_prediction_width"]) for s in summaries)),
        "source_count_tokens_replayed_after_complete_test":len(needed),"original_test_targets_outside_range":sum(t<0 or t>1 for t in targets),
        "previously_revealed_test_targets_reused_as_lags":overlap_total,"each_current_target_revealed_after_own_predictions_sealed":True,
        "model_selection_and_fits_fixed_through_test":True,"all_test_predictions_before_all_test_labels_claim":False,
        "operation_closure_filter_applied":False,"population_or_future_response_coverage_claim":False}
    return {"summary":summary,"predictions":all_predictions,"features":all_features,"targets":targets,"raw_targets":raw_targets,
        "ids":cohort["ids"],"queries":queries,"timings":timings,"cohort":cohort}

def audit_result(protocol,context,test):
    import numpy as np
    primary=read(ROOT/"output/primary_forecast_result38.json");result=read(ROOT/"output/result38.json")
    selection=read(ROOT/"output/validation_selection38.json");epsilon=q(selection["epsilon"])
    require(primary["schema"]=="step38-primary-forecast-result-v1" and primary["test_n"]==91 and
        primary["model_seal_sha256"]==sha(ROOT/"output/model_seal38.json") and
        primary["selection_sha256"]==sha(ROOT/"output/validation_selection38.json") and primary["query_timing"]==test["timings"] and
        isfinite(primary["forecast_seconds"]) and primary["forecast_seconds"]>=0,"Primary forecast result provenance differs")
    rows=primary["candidate_rows"];validation_rows=selection["candidate_rows"]+selection["baseline_rows"]
    require(len(rows)==93 and [r["candidate_id"] for r in rows]==[r["candidate_id"] for r in validation_rows],
        "Fresh test result omitted/reordered frozen candidates")
    for row,prior in zip(rows,validation_rows):
        cid=row["candidate_id"];predictions=test["predictions"][cid]
        require(all(row.get(k)==v for k,v in prior.items()),"A validation-sealed candidate changed on test")
        audit_loss(predictions,test["targets"],row["test_risk"])
        require(row["test_predictions_digest"]==newline_digest(predictions) and
            row["test_eligible"] is (q(row["test_risk"]["upper"])<=epsilon) and
            set(row["test_monthly_risk"])=={"2018-09","2018-10","2018-11"},"Fresh test risk/eligibility/month aggregation differs")
        for month,risk in row["test_monthly_risk"].items():
            indices=[i for i,v in enumerate(test["queries"]) if v["target_timestamp"].startswith(month)]
            audit_loss([predictions[i] for i in indices],[test["targets"][i] for i in indices],risk)
    lookup={r["candidate_id"]:r for r in rows}
    selected={name:lookup[value["candidate_id"]] for name,value in selection["selections"].items()}
    require(primary["selected"]==selected,"Test results changed validation-only model selection")
    gate=audit_practical_gate(selected["qp_all"]["test_risk"],selected["legacybest"]["test_risk"],selected["ordinarybest"]["test_risk"],primary["practical_gate"])
    require(result["schema"]=="step38-training-transfer-result-v1" and result["step"]==38 and result["status"]=="COMPLETE" and
        result["primary"]==primary and result["protocol_sha256"]==sha(ROOT/"protocol38.json") and
        result["protocol_seal_sha256"]==sha(ROOT/"protocol_seal38.json") and result["fit_seal_sha256"]==sha(ROOT/"output/fit_seal38.json") and
        result["model_seal_sha256"]==sha(ROOT/"output/model_seal38.json") and result["pretest_gate_sha256"]==sha(ROOT/"audit/pretest_gate38.json") and
        result["normalization"]==read(ROOT/"output/normalization38.json") and result["training_plan"]==read(ROOT/"output/training_plan38.json") and
        (result["train_n"],result["validation_n"],result["test_n"],result["head_count"],result["baseline_count"])==(242,31,91,90,3),
        "Final experiment provenance/cohort/roster aggregation differs")
    require(result["model_selection_changed_after_test"] is False and result["reserved_patients_accessed"]==0 and
        all(result[k] is False for k in ("native_transform_error_enclosed","population_risk_claim","future_response_coverage_claim")),
        "Final result overstates information/assurance scope")
    secondary=result["secondary_classes"]
    require([r["map"] for r in secondary]==maps_roster() and isfinite(result["secondary_class_seconds"]) and result["secondary_class_seconds"]>=0,
        "Secondary complete-class roster/cost differs")
    class_checks=[];diagnoses={"ADEQUATE":0,"INADEQUATE":0,"UNRESOLVED":0};total_pairs=0
    for row in secondary:
        name=row["map"];info=context["manifest"]["maps"][name];fit=read(bound_reference(row["receipt"]))
        fpath=ROOT/("output/features/test_all/"+name+".npy");z=np.load(fpath,allow_pickle=False)
        require(z.dtype==np.float64 and np.array_equal(z,np.asarray(test["features"][name],dtype=np.float64)),
            "Post-assessment diagnostic geometry differs from sealed prediction features")
        binding={"feature_path":str(fpath.relative_to(ROOT)),"feature_sha256":sha(fpath),"cohort_path":"output/cohorts/test.json",
            "cohort_sha256":sha(ROOT/"output/cohorts/test.json"),"primary_forecasts_sha256":sha(ROOT/"output/primary_forecast_result38.json")}
        checked=audit_fit(fit,z,test["targets"],test["ids"],expected={"role":"assessment_diagnostic","feature_id":info["feature_id"],
            "class_id":info["class_id"],"input_binding":binding,"kappa":"2","settings":protocol["heads"]["solver_settings"]})
        class_checks.append(checked);total_pairs+=checked["pair_distances_verified"]
        head_rows=[r for r in rows if r.get("map")==name];require(len(head_rows)==5,"Secondary class head roster differs")
        lower=q(fit["class_risk"]["lower"]);upper=min(q(fit["class_risk"]["upper"]),*(q(r["test_risk"]["upper"]) for r in head_rows))
        require(0<=lower<=upper,"Post-assessment primal/lower/fixed-head class bracket conflicts")
        interval(row["class_risk"],(lower,upper));status="ADEQUATE" if upper<=epsilon else "INADEQUATE" if lower>epsilon else "UNRESOLVED"
        require(row["feature_id"]==info["feature_id"] and row["class_id"]==info["class_id"] and q(row["epsilon"])==epsilon and
            row["diagnosis"]==status and row["assessment_labels_used"] is True and row["head_promotion_allowed"] is False,
            "Post-assessment class diagnosis/role differs")
        require([v["candidate_id"] for v in row["head_excess"]]==[v["candidate_id"] for v in head_rows],"Secondary five-head excess roster differs")
        for excess,head in zip(row["head_excess"],head_rows):
            expected=(max(F(0),q(head["test_risk"]["lower"])-upper),max(F(0),q(head["test_risk"]["upper"])-lower))
            require(excess["recipe"]==head["recipe"],"Head excess recipe identity differs");interval(excess["excess"],expected)
        diagnoses[status]+=1
    return {"status":"PASS","test_original_label_losses_verified":93,"descriptive_monthly_losses_verified":279,
        "frozen_validation_selected_ids":{k:v["candidate_id"] for k,v in selected.items()},"practical_gate":gate,
        "secondary_class_diagnoses":diagnoses,"secondary_class_fits":class_checks,"secondary_pair_distances_verified":total_pairs,
        "secondary_head_excess_intervals_verified":90,"all_90_heads_and_3_baselines_retained":True,
        "secondary_fits_cannot_change_primary_models_or_selections":True}

def audit_phase_sequence(protocol):
    events=audit_events();names=[e["event"] for e in events]
    required=("protocol_sealed","independent_preexecution_PASS","train_count_access_started","train_count_access_completed",
        "fit_sealed","independent_prevalidation_PASS","validation_count_access_started","validation_count_access_completed",
        "model_selection_sealed","independent_pretest_PASS","all_test_forecasts_and_scores_complete")
    positions=[require_once(events,name)[0] for name in required]
    require(positions==sorted(positions),"Whole-experiment source/fit/selection/assessment gate sequence differs")
    starts=[e for e in events if e["event"]=="phase_started"];ends=[e for e in events if e["event"]=="phase_completed"]
    require([e["phase"] for e in starts]==["train","validate","test"]==[e["phase"] for e in ends],"Repeated/missing primary phase")
    for phase in ("train","validate","test"):
        record=read(ROOT/("output/"+phase+"_completed38.json"));event=next(e for e in ends if e["phase"]==phase)
        require(record["schema"]=="step38-phase-completion-v1" and record["phase"]==phase and record["status"]=="COMPLETE" and
            0<=record["wall_seconds"]<=protocol["execution"]["phase_budget_seconds"][phase] and
            event["receipt_sha256"]==sha(ROOT/("output/"+phase+"_completed38.json")),"Phase completion/budget binding differs")
    primary=read(ROOT/"output/primary_started38.json")
    require(primary["single_primary_pass"] is True and primary["protocol_seal_sha256"]==sha(ROOT/"protocol_seal38.json"),
        "Primary run marker differs")
    _,test_complete=require_once(events,"all_test_forecasts_and_scores_complete")
    require(test_complete["receipt_sha256"]==sha(ROOT/"output/primary_forecast_result38.json") and test_complete["test_n"]==91,
        "Completed forecast event differs")
    test_end=next(e for e in ends if e["phase"]=="test")
    require(test_end["result_sha256"]==sha(ROOT/"output/result38.json"),"Completed test result byte binding differs")
    return {"status":"PASS","event_count":len(events),"event_chain_sha256":sha(ROOT/"output/access_and_execution38.jsonl"),
        "single_primary_pass":True,"phase_and_access_gates_in_order":True,"source_fit_selection_seals_retained":True}

def postrun():
    started=time.perf_counter();protocol,fit,model=verify_fit_and_model_gates()
    verify_gate("audit/pretest_gate38.json","independent_pretest_PASS",protocol_seal_sha256=sha(ROOT/"protocol_seal38.json"),
        fit_seal_sha256=sha(ROOT/"output/fit_seal38.json"),model_seal_sha256=sha(ROOT/"output/model_seal38.json"))
    phase=audit_phase_sequence(protocol);context=load_context();test=audit_test_forecasts(protocol,context)
    scores=audit_result(protocol,context,test)
    # The numerical practical criterion is necessary but only becomes a valid
    # study success once this independent provenance, access and math audit PASSes.
    practical_success=scores["practical_gate"]["numerical_practical_transfer_criterion"]
    result={"schema":"step38-independent-postrun-audit-v1","status":"PASS","protocol_seal_sha256":sha(ROOT/"protocol_seal38.json"),
        "fit_seal_sha256":sha(ROOT/"output/fit_seal38.json"),"model_seal_sha256":sha(ROOT/"output/model_seal38.json"),
        "result_sha256":sha(ROOT/"output/result38.json"),"execution":phase,"test_forecasts":test["summary"],"result_arithmetic":scores,
        "certificate_and_access_validity":"PASS","practical_transfer_success":practical_success,
        "primary_status":"PRACTICAL_TRANSFER_DEMONSTRATED_ON_THIS_FRESH_COHORT" if practical_success else "PRACTICAL_TRANSFER_NOT_ESTABLISHED",
        "all_results_retained_regardless_of_outcome":True,"training_and_validation_proofs_reused_only_after_unchanged_byte_seal_checks":True,
        "new_scientific_model_fits_by_auditor":0,"reserved_patients_accessed":0,"native_transform_error_enclosed":False,
        "population_risk_claim":False,"future_response_coverage_claim":False,"audit_wall_seconds":time.perf_counter()-started}
    write(ROOT/"audit/postrun_audit38.json",result)
    append_event("independent_postrun_PASS",audit_sha256=sha(ROOT/"audit/postrun_audit38.json"),result_sha256=sha(ROOT/"output/result38.json"))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("selfcheck","artificial","preflight","prevalidation","pretest","postrun"));args=parser.parse_args()
    try:
        functions={"selfcheck":lambda:{"status":"PASS","selfchecks":independent_selfcheck(),"scientific_count_values_decoded":0},
            "artificial":artificial_preflight,"preflight":preflight,"prevalidation":prevalidation,"pretest":pretest,"postrun":postrun}
        output=functions[args.mode]()
        wanted=("status","schema","protocol_seal_sha256","fit_seal_sha256","model_seal_sha256","result_sha256",
            "primary_status","practical_transfer_success","scientific_count_values_decoded","audit_wall_seconds")
        summary={k:output[k] for k in wanted if k in output}
        if args.mode=="selfcheck":summary["selfchecks"]=output["selfchecks"]
        print(json.dumps(canon(summary),sort_keys=True),flush=True)
    except Exception as exc:
        failure={"status":"FAIL","mode":args.mode,"utc":datetime.now(timezone.utc).isoformat(),
            "error_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),"source_sha256":sha(Path(__file__))}
        write(ROOT/("audit/"+args.mode+"_audit_failure38.json"),failure)
        print(json.dumps(failure,sort_keys=True),flush=True);raise SystemExit(1)

if __name__=="__main__":main()

