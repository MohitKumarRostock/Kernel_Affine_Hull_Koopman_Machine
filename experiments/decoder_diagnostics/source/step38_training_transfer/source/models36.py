"""Training-only frozen head recipes and exact stored-feature evaluation.

Each call really fits/evaluates its requested head. There is no cache of fitted
heads and no access to the exhaustive comparator's outcomes. The orchestration
layer writes evidence only after the timed in-memory policy execution ends.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_key]="1"
from fractions import Fraction as F
import hashlib
import json
import pickle
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import finite_feature35 as ff
import representations35 as rep


def canonical_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def object_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def fit_evaluate(task, map_name, recipe, features_train, features_decision, train, decision,
                 map_info, anchors, protocol):
    started=time.perf_counter()
    feature_id=map_info["feature_id"]
    cfg=protocol["heads"]
    fit_started=time.perf_counter()
    if recipe=="affine":
        z=features_train
        y=np.asarray([float(v) for v in train["y"]],dtype=np.float64)
        xc=z-z.mean(axis=0);yc=y-y.mean()
        lam=float(F(cfg["ridge_lambda"]))
        beta=np.linalg.solve(xc.T@xc+len(z)*lam*np.eye(z.shape[1]),xc.T@yc)
        intercept=float(y.mean()-z.mean(axis=0)@beta)
        exact_center=[sum((F.from_float(float(v)) for v in z[:,j]),F(0))/len(z) for j in range(z.shape[1])]
        exact_y=sum(train["y"],F(0))/len(train["y"])
        fit=ff.make_affine_head(beta.tolist(),intercept,feature_id=feature_id,kappa=protocol["class"]["kappa"],
             output_interval=protocol["class"]["prediction_interval"],project=True,center_features=exact_center,
             center_target=exact_y,bits=protocol["certificate"]["distance_bits"])
        fit["training_binding"]={"task":task,"training_n":len(z),"training_cohort_path":f"output/cohorts/{task}/train.json",
                                 "training_feature_path":f"output/features/{task}/train/{map_name}.npy","ridge_lambda":cfg["ridge_lambda"]}
    elif recipe=="extension32":
        fit=ff.fit_anchor_decoder(features_train[anchors].tolist(),[train["y"][i] for i in anchors],
             feature_id=feature_id,ids=[train["ids"][i] for i in anchors],kappa=protocol["class"]["kappa"],
             output_interval=protocol["class"]["prediction_interval"],distance_bits=protocol["certificate"]["distance_bits"],
             max_updates=cfg["extension_max_coordinate_attempts"],check_every=cfg["check_every"],
             gap_sse=cfg["gap_sse"],grid_bits=cfg["multiplier_grid_bits"])
        fit["training_binding"]={"task":task,"training_cohort_path":f"output/cohorts/{task}/train.json",
                                 "training_feature_path":f"output/features/{task}/train/{map_name}.npy",
                                 "anchor_indices":[int(i) for i in anchors]}
    else:
        raise ValueError("Undeclared decoder recipe")
    fit_seconds=time.perf_counter()-fit_started
    head=fit["head"]
    evaluation_started=time.perf_counter()
    evaluator=ff.evaluate_affine if recipe=="affine" else ff.evaluate_extension
    prediction=evaluator(head,features_decision.tolist(),decision["y"],ids=decision["ids"])
    evaluation_seconds=time.perf_counter()-evaluation_started
    payload=canonical_bytes(head)
    row={"candidate_id":map_name+"::"+recipe,"task":task,"map":map_name,"recipe":recipe,
         "feature_id":feature_id,"class_id":fit["class_id"],"head_id":fit["head_id"],
         "feature_width":int(features_train.shape[1]),"family":map_name.split("_")[0],
         "scaling":"minmax01" if "minmax01" in map_name else "standard",
         "map_payload_bytes":map_info["payload_bytes"],"map_numeric_bytes":map_info["numeric_bytes"],
         "head_payload_bytes":len(payload),"deployment_payload_bytes":map_info["payload_bytes"]+len(payload),
         "decision_risk":{key:prediction["loss"][key] for key in ["lower","upper","width"]},
         "head_sha256":hashlib.sha256(payload).hexdigest(),"fit_sha256":object_sha(fit),
         "decision_prediction_sha256":object_sha(prediction),
         "fit_status":fit.get("status",fit["membership"]["status"]),
         "projection_applied":fit.get("projection",{}).get("projected",False),
         "fit_seconds":fit_seconds,"exact_decision_evaluation_seconds":evaluation_seconds,
         "fit_evaluate_and_identity_seconds":time.perf_counter()-started,
         "assurance":ff.ASSURANCE}
    return {"row":row,"head":head,"fit":fit,"decision":prediction}


def selection(rows,epsilon):
    epsilon=F(epsilon)
    eligible=[row for row in rows if F(row["decision_risk"]["upper"])<=epsilon]
    chosen=min(eligible,key=lambda row:(row["deployment_payload_bytes"],F(row["decision_risk"]["upper"]),row["candidate_id"]),default=None)
    return {"candidate_id":None if chosen is None else chosen["candidate_id"],"abstained":chosen is None,
            "eligible_evaluated_heads":len(eligible)}


def range_floor(targets,output_interval):
    low,high=map(F,output_interval)
    return sum(((max(low,min(high,y))-y)**2 for y in targets),F(0))/len(targets)


def baseline_receipt(name,predictions,cohort):
    values=[v if isinstance(v,F) else F.from_float(float(v)) for v in predictions]
    boxes=[{"lower":str(v),"upper":str(v),"width":"0"} for v in values]
    return {"schema":"step36-stored-baseline-predictions-v1","baseline":name,"query_ids":cohort["ids"],
            "targets_exact":[str(y) for y in cohort["y"]],"predictions_exact":[str(v) for v in values],
            "loss":ff.loss_bounds(boxes,cohort["y"]),"class_membership_certified":False,
            "assurance":"EXACT_LOSS_OF_STORED_BASELINE_PREDICTIONS","population_risk_claim":False}


def fit_baselines(train,decision,minmax_map,protocol):
    started=time.perf_counter()
    scaler=minmax_map.core
    params=protocol["baselines"]["hist_gradient_boosting_parameters"]
    model=HistGradientBoostingRegressor(**params)
    model.fit(scaler.transform(train["X"]),np.asarray([float(y) for y in train["y"]],dtype=np.float64))
    mean=sum(train["y"],F(0))/len(train["y"])
    payload=pickle.dumps({"model":model,"scaler":scaler},protocol=5)
    values={"persistence":decision["persistence"],"training_mean":[mean]*len(decision["y"]),
            "hist_gradient_boosting":model.predict(scaler.transform(decision["X"])).tolist()}
    receipts={name:baseline_receipt(name,prediction,decision) for name,prediction in values.items()}
    return {"model":model,"scaler":scaler,"training_mean":mean,"payload":payload,"decision_receipts":receipts,
            "fit_and_decision_seconds":time.perf_counter()-started}
