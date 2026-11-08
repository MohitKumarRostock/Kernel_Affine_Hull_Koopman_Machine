"""Artificial-only producer integration; no Seoul measurement access."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import finite_feature35 as ff
import learners38 as m
import representations35 as rep
import protocol38 as p

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(m.canonical_bytes(value))
def expect_reject(action):
    try:action()
    except (ValueError,ArithmeticError):return True
    raise AssertionError("Invalid artificial input was accepted")

def main():
    started=time.perf_counter();checks=[]
    z=np.asarray([[i/32,((i*7)%19)/64,((i*11)%23)/32] for i in range(40)],dtype=np.float64)
    z[-1]=z[0]
    y=[F((i*13)%37,36) for i in range(40)]
    ids=["artificial_train_"+str(i) for i in range(40)]
    anchors=rep.temporal_indices(len(z),32).tolist();feature_id="artificial_feature38"
    fit=m.fit_values(z,y,ids=ids,feature_id=feature_id,input_binding={"artificial":True})
    small=m.fit_values(z[anchors],[y[i] for i in anchors],ids=[ids[i] for i in anchors],feature_id=feature_id,input_binding={"artificial":True,"anchor_indices":anchors})
    assert F(fit["class_risk"]["lower"])<=F(fit["class_risk"]["upper"])
    assert fit["training_qp_no_worse_than_envelope"] is True
    checks.extend(["full_40_anchor_exact_feasibility_and_risk_bracket","duplicate_training_coordinates","qp_training_risk_no_worse_than_shared_envelope"])
    affine=m.fit_affine(z,y,feature_id=feature_id,training_binding={"artificial":True})
    legacy=ff.fit_anchor_decoder(z[anchors].tolist(),[y[i] for i in anchors],feature_id=feature_id,
        ids=[ids[i] for i in anchors],kappa="2",output_interval=("0","1"),distance_bits=80,
        max_updates=4000,check_every=20,gap_sse="1/10000000",grid_bits=40)
    heads={"affine":affine["head"],"extension32":legacy["head"],
        "qp32":m.make_extension(small,z[anchors],anchor_indices=anchors,recipe="qp32",fit_path="artificial/small.json",fit_sha256=m.digest(small)),
        "envelope_all":m.make_extension(fit,z,anchor_indices=list(range(40)),recipe="envelope_all",fit_path="artificial/full.json",fit_sha256=m.digest(fit)),
        "qp_all":m.make_extension(fit,z,anchor_indices=list(range(40)),recipe="qp_all",fit_path="artificial/full.json",fit_sha256=m.digest(fit))}
    queries=np.asarray([z[0],z[18],[.1,.2,.3],[-1.,-1.,-1.],[2.,2.,2.]],dtype=np.float64)
    qids=["artificial_query_"+str(i) for i in range(len(queries))]
    distance=m.query_distances(z,queries,anchor_ids=ids,query_ids=qids,feature_id=feature_id)
    predictions={r:m.evaluate_head(heads[r],queries,ids=qids,distance=distance) for r in m.RECIPES}
    for recipe,prediction in predictions.items():
        assert len(prediction["predictions"])==len(queries)
        for box in prediction["predictions"]:
            assert 0<=F(box["lower"])<=F(box["upper"])<=1
            assert F(box["width"])<=F(2,1<<80)
    for recipe in ("envelope_all","qp_all"):
        value=F(heads[recipe]["anchor_values"][0]);box=predictions[recipe]["predictions"][0]
        assert F(box["lower"])<=value<=F(box["upper"])
    checks.extend(["all_five_recipes_shared_query_geometry","full_anchor_interpolation","80_bit_fixed_prediction_enclosures"])
    outside=[F(-1,3),F(3,2),F(1,2),F(2),F(-1)]
    losses={r:m.score(v["predictions"],outside) for r,v in predictions.items()}
    assert all(F(loss["lower"])>0 for loss in losses.values())
    checks.append("unclipped_original_target_losses")
    forbidden=deepcopy(fit);forbidden["role"]="assessment_diagnostic";forbidden["head_promotion_allowed"]=False
    expect_reject(lambda:m.make_extension(forbidden,z,anchor_indices=list(range(40)),recipe="qp_all",fit_path="forbidden",fit_sha256="forbidden"))
    corrupted=deepcopy(heads["qp_all"]);corrupted["anchor_indices"]=list(reversed(range(40)))
    expect_reject(lambda:m.evaluate_head(corrupted,queries,ids=qids,distance=distance))
    corrupted=deepcopy(heads["extension32"]);corrupted["anchor_ids"]=ids[:33]
    expect_reject(lambda:m.evaluate_head(corrupted,queries,ids=qids,distance=distance))
    checks.extend(["assessment_witness_promotion_rejected","anchor_column_substitution_rejected","legacy_32_anchor_guard_preserved"])
    def risk(x):return {"lower":str(x),"upper":str(x),"width":"0"}
    assert m.practical_gate(risk(F(19,20)),risk(F(1)),risk(F(1)))["numerical_practical_transfer_criterion"] is True
    assert m.practical_gate(risk(F(0)),risk(F(0)),risk(F(0)))["relative_gain_status"]=="UNDEFINED_RELATIVE_GAIN"
    assert m.practical_gate(risk(F(19,20)),risk(F(1)),risk(F(9,10)))["numerical_practical_transfer_criterion"] is False
    rows=[{"candidate_id":name,"validation_risk":risk(F(1)),"deployment_payload_bytes":size,"validation_eligible":False} for name,size in (("z",3),("b",2),("a",2))]
    assert m.select(rows)["candidate_id"]=="a"
    checks.extend(["exact_practical_gate_boundary_and_positive_denominator","ordinary_baseline_conjunction","selection_bytes_and_lexical_tie_without_eligibility_filter"])
    # Synthetic raw inputs only: exercise all native dimensions/scalers and serialization.
    X=np.asarray([[((i+3)*(j+5)%97)+(i%7)*.125+(j%3)*.25 for j in range(24)] for i in range(64)],dtype=np.float64)
    maps,metadata=rep.fit_representations(X,rep.DEFAULT_CONFIG,3801)
    assert sorted(maps)==p.map_names()
    map_results=[]
    for name in sorted(maps):
        payload=rep.serialize_map(maps[name]);loaded=rep.deserialize_map(payload)
        first=rep.transform_one(maps[name],X[:3]);second=rep.transform_one(loaded,X[:3])
        assert np.array_equal(first,second) and np.isfinite(first).all()
        map_results.append({"name":name,"shape":list(first.shape),"payload_bytes":len(payload)})
    config=json.loads((ROOT/"protocol38.json").read_text())
    hgb=HistGradientBoostingRegressor(**config["baselines"]["hgb_parameters"])
    hgb.fit(maps["raw_minmax01"].core.transform(X),np.asarray([i/63 for i in range(64)]))
    assert np.isfinite(hgb.predict(maps["raw_minmax01"].core.transform(X[:3]))).all()
    checks.extend(["all_18_native_feature_maps_on_artificial_24_dimensional_inputs","map_serialization_prediction_roundtrip","frozen_HGB_parameters_runtime_compatibility"])
    fixture={"schema":"step38-artificial-learner-fixture-v1","scientific_data_accessed":False,
        "training_z":z.tolist(),"training_y":list(map(str,y)),"training_ids":ids,"anchor_indices32":anchors,
        "full_fit":fit,"small_fit":small,"affine_fit":affine,"legacy_fit":legacy,"heads":heads,
        "query_z":queries.tolist(),"query_ids":qids,"distances":distance,"predictions":predictions,
        "original_loss_targets":list(map(str,outside)),"losses":losses}
    fixture_path=ROOT/"tests/learner_fixture38.json";write(fixture_path,fixture)
    sources=["source/"+name for name in ("check_learners38.py","learners38.py","full_graph37.py","finite_feature35.py","representations35.py","protocol38.py","run38.py")]
    receipt={"schema":"step38-artificial-producer-check-v1","status":"PASS","checks":checks,"check_count":len(checks),
        "scientific_data_accessed":False,"artificial_only":True,"source_sha256":{name:sha(ROOT/name) for name in sources},
        "fixture_path":"tests/learner_fixture38.json","fixture_sha256":sha(fixture_path),
        "map_results":map_results,"wall_seconds":time.perf_counter()-started}
    write(ROOT/"tests/artificial_learners38.json",receipt)
    print(json.dumps({"status":"PASS","checks":len(checks),"seconds":receipt["wall_seconds"],"fixture_sha256":receipt["fixture_sha256"]}),flush=True)

if __name__=="__main__":main()
