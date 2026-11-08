"""Artificial-only integration checks; never opens scientific inputs."""
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import full_graph37 as g
import audit37 as a

ROOT=Path(__file__).resolve().parents[1]

def fixture(name,z,y,kappa="2",epsilon="1/5",settings=None,fail_solver=False):
    z=np.asarray(z,dtype=np.float64);y=list(map(F,y));n=len(y)
    feature="synthetic-feature:"+name;cid=g.ff.class_identity(feature,kappa,("0","1"))
    constant=g.clipped(sum(y,F(0))/n);risk=sum(((constant-t)**2 for t in y),F(0))/n
    heads=[{"candidate_id":name+"::"+recipe,"head_id":"synthetic-head:"+name+":"+recipe,
            "recipe":recipe,"feature_id":feature,"class_id":cid,"assurance":g.ASSURANCE,
            "decision_risk":{"lower":str(risk),"upper":str(risk),"width":"0"},
            "decision_prediction_sha256":"synthetic-constant-witness"} for recipe in ("affine","extension32")]
    ids=["artificial:"+str(i) for i in range(n)];kw=dict(ids=ids,feature_id=feature,class_id=cid,
        epsilon=epsilon,heads=heads,task="artificial",map_name=name,kappa=kappa,settings=settings)
    if fail_solver:
        with patch.object(g.osqp,"OSQP",side_effect=RuntimeError("artificial solver failure")):
            record=g.diagnose(z,y,**kw)
    else:record=g.diagnose(z,y,**kw)
    expected={"task":"artificial","map":name,"feature_id":feature,"class_id":cid,
              "epsilon":epsilon,"settings":dict(g.SOLVER_SETTINGS if settings is None else settings),"input_binding":{}}
    checked=a.audit_diagnostic(record,z.tolist(),y,ids,heads,expected=expected)
    payload={"name":name,"z_binary64":z.tolist(),"y":list(map(str,y)),"ids":ids,"heads":heads,
             "expected":expected,"record":record,"independent_check":checked}
    (ROOT/"tests"/("producer_fixture_"+name+".json")).write_bytes(g.canonical_bytes(payload))
    return record

def main():
    if (ROOT/"protocol_seal37.json").exists():raise RuntimeError("Artificial preparation already frozen")
    checks={};records=[]
    r=fixture("three_point",[[0.],[.25],[.5]],["0","1","0"],kappa="1",epsilon="1/8")
    assert F(r["class_risk"]["lower"])<=F(1,8)<=F(r["class_risk"]["upper"])
    assert F(r["class_risk"]["width"])<F(1,10**7)
    checks["nontrivial_three_point_analytic_optimum"]=True;records.append(r)
    r=fixture("duplicates",[[0.],[0.],[1.]],["-1/2","3/2","1/4"])
    assert r["distance"]["zero_distance_pair_count"]==1
    checks["duplicate_features_and_original_outside_targets"]=True;records.append(r)
    r=fixture("zero_kappa",[[0.,0.],[.5,1.],[1.,2.]],["0","1","0"],kappa="0")
    assert F(r["class_risk"]["upper"])==F(2,9)
    checks["zero_sensitivity_constant_class"]=True;records.append(r)
    r=fixture("tiny_distances",[[0.],[2.0**-100],[2.0**-90]],["0","1","1/3"])
    assert r["distance"]["common_binary_exponent"]>80
    checks["distances_below_dyadic_resolution"]=True;records.append(r)
    r=fixture("more_than_32",[[i/64.,(i%5)/32.] for i in range(35)],
        [str(F((i*7)%17,16)) for i in range(35)])
    assert r["distance"]["pair_count"]==595
    checks["full_graph_beyond_previous_32_anchor_head"]=True;records.append(r)
    r=fixture("single_outside",[[2.]], ["3/2"])
    assert F(r["class_risk"]["lower"])==F(r["class_risk"]["upper"])==F(1,4)
    checks["single_point_exact_range_floor"]=True;records.append(r)
    limited=dict(g.SOLVER_SETTINGS,max_iter=1,polishing=False)
    r=fixture("unfinished_solver",[[0.],[.01],[.02]], ["0","1","0"],settings=limited)
    assert r["numerical"]["iterations"]==1
    checks["unfinished_solver_still_yields_verified_bracket"]=True;records.append(r)
    r=fixture("solver_exception",[[0.],[.1],[1.]], ["0","1","1/2"],fail_solver=True)
    assert r["numerical"]["raw_primal_binary64"] is None and r["dual"]["signed_multipliers"]==[]
    assert len(r["primal_candidates"])==2
    checks["solver_exception_exact_fallback_without_retry"]=True;records.append(r)
    assert g.quantize(F(1,2**41))==0 and g.quantize(F(3,2**41))==2
    assert g.quantize(F(-1,2**41))==0 and g.quantize(F(-3,2**41))==-2
    checks["signed_nearest_ties_even_quantization"]=True
    assert a.three_way(F(1,8),F(1,8),F(1,8))=="CLASS_ADEQUATE"
    assert a.three_way(F(1,8),F(1,4),F(1,8))=="UNRESOLVED"
    checks["threshold_equality_is_not_inadequacy"]=True
    result={"schema":"step37-artificial-producer-integration-v1","status":"PASS","checks":checks,
        "artificial_only":True,"scientific_inputs_opened":0,"fixtures":len(records),
        "source_sha256":{str(p.relative_to(ROOT)):a.sha(p) for p in
            [ROOT/"source/full_graph37.py",ROOT/"source/check_producer37.py",ROOT/"source/audit37.py"]},
        "fixture_sha256":{str(p.relative_to(ROOT)):a.sha(p) for p in sorted((ROOT/"tests").glob("producer_fixture_*.json"))}}
    (ROOT/"tests/artificial_producer_preflight37.json").write_bytes(g.canonical_bytes(result))
    print(json.dumps({"status":"PASS","checks":len(checks),"fixtures":len(records),"scientific_inputs_opened":0}))

if __name__=="__main__":main()
