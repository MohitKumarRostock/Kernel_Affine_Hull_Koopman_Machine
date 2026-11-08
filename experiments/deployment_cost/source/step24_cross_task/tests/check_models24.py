"""Manufactured-only checks for the Step 24 readout/compact-kernel implementation."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
             "OMP_THREAD_LIMIT"):
    os.environ[_key] = "1"

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import models24 as M


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejects(fn):
    try:
        fn()
    except (ValueError, TypeError, ArithmeticError):
        return True
    raise AssertionError("Manufactured invalid input was accepted")


def main():
    receipt_path = ROOT / "tests/models_preflight24.json"
    if receipt_path.exists():
        raise FileExistsError("Manufactured preflight receipt is write-once")
    # A separate manufactured namespace, never the HMM benchmark generator.
    rng = np.random.Generator(np.random.PCG64(2026110624002))
    checks, observations = [], []
    X = rng.normal(size=(192, 8))
    X[:, -1] = 2.75
    V = rng.normal(size=(61, 8))
    V[:, -1] = np.linspace(-20., 20., len(V))
    y_energy = 60. + 13. * X[:, 0] - 7. * X[:, 1] + .1 * rng.normal(size=len(X))
    y_hmm = np.tanh(X[:, 0] - .6 * X[:, 1] + .15 * X[:, 2])

    target = M.fit_target_map(y_energy)
    assert float(target["offset"]) == float(np.mean(y_energy))
    assert float(target["scale"]) == float(np.std(y_energy, ddof=0))
    fixed = M.fit_target_map(np.full(64, 7.5))
    assert fixed["offset"] == 7.5 and fixed["scale"] == 1.
    checks.append("Training target map is population mean/SD; constant target uses divisor1")

    scaler, constant = M.fit_feature_scaler(X)
    assert constant[-1] and scaler["offset"][-1] == 2.75 and scaler["scale"][-1] == 1.
    assert np.array_equal(M.standardize_features(X, scaler)[:, -1], np.zeros(len(X)))
    assert np.array_equal(M.standardize_features(V, scaler)[:, -1], V[:, -1] - 2.75)
    checks.append("Ridge head constant axes use divisor1 and retain future variation before readout")

    head, val, diag = M.fit_head({"family":"ridge", "alpha":.01}, X, y_energy, V, target, "appliances")
    assert set(head) == {"kind", "coef", "intercept", "output_bounds"}
    assert "head_scaler_evidence" in diag and diag["unclipped_export"]["export_passed"]
    assert np.all(val >= 0) and np.isfinite(val).all()
    replay = M.predict_head_reference(head, V, "appliances")
    M.audit_export(val, replay, target, "manufactured ridge independent replay")
    assert head["coef"][-1] == 0.
    checks.append("Ridge folds head standardizer and physical target inverse; independent ordered-sum replay passes")
    observations.append({"case":"ridge_appliances","diagnostics":diag})

    target_h = M.fit_target_map(y_hmm)
    head_h, pred_h, diag_h = M.fit_head({"family":"ridge", "alpha":1e-4}, X, y_hmm, V, target_h, "hmm_0")
    assert np.all((-1. <= pred_h) & (pred_h <= 1.))
    M.audit_export(pred_h, M.predict_head_reference(head_h, V, "hmm"), target_h)
    checks.append("HMM exported ridge uses the same fixed output map for task-kind and replication ids")
    observations.append({"case":"ridge_hmm","diagnostics":diag_h})

    fixture = {"kind":"ridge", "coef":np.array([1.]), "intercept":np.float64(0.),
               "output_bounds":M.output_bounds("hmm")}
    values = np.array([[-2.],[-.5],[0.],[2.]])
    assert np.array_equal(M.predict_head(fixture, values, "hmm"), np.array([-1.,-.5,0.,1.]))
    assert np.array_equal(M.predict_head(fixture, values, "hmm",clip=False),values[:,0])
    fixture["output_bounds"] = M.output_bounds("appliances")
    assert np.array_equal(M.predict_head(fixture, values, "appliances"), np.array([0.,0.,0.,2.]))
    checks.append("Physical-unit output maps clip HMM to[-1,1] and Appliances only below0")

    retained_tree_fields = None
    for leaves in (7,15,31):
        tree, predicted, evidence = M.fit_head({"family":"hgb","max_leaf_nodes":leaves},
                                              X,y_energy,V,target,"appliances")
        assert len(tree["trees"]) == evidence["tree_count"] == 200
        assert evidence["parameters"]["early_stopping"] is False
        assert evidence["parameters"]["random_state"] == 2026110623
        assert evidence["unclipped_export"]["max_abs_difference"] == 0.
        manual = M.predict_head_reference(tree,V,"appliances")
        assert np.array_equal(predicted,manual)
        node_bytes = sum(t["nodes"].nbytes for t in tree["trees"])
        inventory=M.numeric_inventory(tree)
        assert inventory["numeric_bytes"] >= node_bytes
        assert sum(c["retained_bytes"] for c in inventory["components"] if c["path"].endswith(".nodes")) == node_bytes
        retained_tree_fields = list(tree["trees"][0]["nodes"].dtype.names)
        assert "count" in retained_tree_fields and "gain" in retained_tree_fields and "depth" in retained_tree_fields
        observations.append({"case":f"hgb_leaves{leaves}","diagnostics":evidence,
                             "manual_replay_bit_identical":True,"full_node_bytes":node_bytes})
    checks.append("All three HGB menus fit200 iterations; exported/native and independent tree traversals agree exactly")
    checks.append("Full structured tree node fields, including count/gain/depth and padding, are charged")

    U = rng.normal(size=(512, 8)); U[:, -1] = 2.75; U[1] = U[0]
    for mode in ("minmax01","standard"):
        input_scaler = M.fit_input_scaler(U, mode)
        transformed = M.apply_input_scaler(V, input_scaler)
        assert np.array_equal(transformed[:, -1], np.zeros(len(V)))
        raw_extreme = U[:1].copy(); raw_extreme[:,0] = np.max(U[:,0]) + 10.
        if mode == "minmax01":
            assert M.apply_input_scaler(raw_extreme,input_scaler)[0,0] > 1.
    checks.append("Nystrom support scalers zero support-constant axes and never clip later variable axes")

    for count, mode, mult in [(32,"minmax01",.1),(128,"standard",1.),(512,"minmax01",10.)]:
        base, evidence = M.fit_nystrom_base(U, mode, count, mult)
        assert evidence["summary"]["floored_eigenvalue_count"] >= 1
        assert np.array_equal(evidence["arrays"]["W"],evidence["arrays"]["W"].T)
        assert np.array_equal(base["normalization"],base["normalization"].T)
        assert np.all(evidence["arrays"]["floored_eigenvalues"] >= evidence["summary"]["eigenvalue_floor"])
        F,Fv = M.nystrom_features(base,X),M.nystrom_features(base,V)
        ridge, prediction, diagnosis = M.fit_head({"family":"ridge","alpha":.01},F,y_hmm,Fv,target_h,"hmm")
        compact = M.collapse_nystrom(base,ridge)
        assert "normalization" not in compact and "W" not in compact
        assert compact["landmarks"].shape == (count,8)
        assert compact["coef"].shape == (count,)
        compact_prediction = M.predict_nystrom(compact,V,"hmm")
        raw_audit = M.audit_export(M.predict_head(ridge,Fv,"hmm",clip=False),
                                   M.predict_nystrom(compact,V,"hmm",clip=False),target_h,
                                   "manufactured Nystrom raw collapse")
        clipped_audit = M.audit_export(prediction,compact_prediction,target_h,
                                       "manufactured Nystrom output collapse")
        inventory=M.numeric_inventory(compact)
        # All deployed matrices are m-by-ambient, never an m-by-m normalization.
        assert all(c["shape"] != [count,count] for c in inventory["components"])
        expected_bytes = count*8*8 + count*8 + 8*8*2 + 8 + 8 + 8 + 16
        # input scaler: offset/divisor float64 vectors and8 Boolean mask bytes;
        # gamma/intercept scalar8 each, two HMM bound scalars16 bytes.
        assert inventory["numeric_bytes"] == expected_bytes
        observations.append({"case":f"nystrom_{mode}_{count}_{mult}",
                             "basis_summary":evidence["summary"],
                             "unclipped_collapse_audit":raw_audit,
                             "clipped_collapse_audit":clipped_audit,
                             "complete_numeric_bytes":inventory["numeric_bytes"]})
    checks.append("All Nystrom landmark counts use the prescribed positive-distance bandwidth and eigenvalue floor")
    checks.append("Compact Nystrom collapse agrees within frozen tolerance and omits dense normalization storage")

    owner=np.arange(100,dtype=np.float64)
    retained={"array":owner,"same_array":owner,"subset":owner[::3],
              "separate_copy":owner[:5].copy(),"scalar":np.float64(2.)}
    inventory=M.numeric_inventory(retained)
    assert inventory["numeric_bytes"] == owner.nbytes + 5*8 + 8
    assert sum("alias_of" in c for c in inventory["components"]) == 2
    only_view=M.numeric_inventory({"tail":owner[95:]})
    assert only_view["numeric_bytes"] == owner.nbytes
    checks.append("Inventory counts shared array owners once and includes full allocations retained through views")

    rejections=[
        lambda:M.fit_target_map(np.array([1.,np.nan])),
        lambda:M.fit_head({"family":"ridge","alpha":0.},X,y_energy,V,target,"appliances"),
        lambda:M.fit_head({"family":"hgb","max_leaf_nodes":2},X,y_energy,V,target,"appliances"),
        lambda:M.fit_head({"family":"hgb","max_leaf_nodes":7,"max_iter":5},X,y_energy,V,target,"appliances"),
        lambda:M.fit_nystrom_base(U[:511],"standard",32,1.),
        lambda:M.fit_nystrom_base(U,"standard",31,1.),
        lambda:M.fit_nystrom_base(U,"standard",32,3.),
        lambda:M.fit_nystrom_base(np.ones((512,2)),"standard",32,1.),
        lambda:M.numeric_inventory({"hidden":object()}),
        lambda:M.numeric_inventory({"object_array":np.array([object()],dtype=object)}),
        lambda:M.predict_head(head,V[:,:-1],"appliances"),
        lambda:M.predict_head(head,V,"hmm"),
        lambda:M.audit_export(np.array([0.]),np.array([1e-4]),target_h),
    ]
    assert all(rejects(fn) for fn in rejections)
    checks.append("Malformed/nonfinite inputs, changed menus, no-bandwidth case, hidden payloads and export drift fail closed")

    receipt={"schema":"step24-manufactured-model-preflight-v1",
        "completed_utc":datetime.now(timezone.utc).isoformat(),"passed":True,
        "checks":checks,"check_groups":len(checks),"invalid_cases_rejected":len(rejections),
        "manufactured_head_fits":8,"manufactured_nystrom_bases":3,
        "benchmark_model_fits":0,"benchmark_trajectories_generated":0,
        "real_dataset_rows_parsed":0,"ECG_or_reserve_access":False,
        "manufactured_seed":2026110624002,
        "retained_tree_node_fields":retained_tree_fields,
        "export_abs_factor":M.EXPORT_ABS_FACTOR,"export_relative_tolerance":M.EXPORT_REL_TOL,
        "observations":observations,
        "software":{key:importlib.metadata.version(key) for key in ["numpy","scipy","scikit-learn"]},
        "source_sha256":{
            "src/models24.py":sha(ROOT/"src/models24.py"),
            "tests/check_models24.py":sha(Path(__file__)),
            "../step23_cross_task/PROTOCOL.json":sha(ROOT.parent/"step23_cross_task/PROTOCOL.json")}}
    with receipt_path.open("x") as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in {"observations","source_sha256"}},indent=2))


if __name__=="__main__":main()
