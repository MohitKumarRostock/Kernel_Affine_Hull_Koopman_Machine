"""Freeze the whole Step39 design before new numerical measurements are read."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
from sklearn.ensemble import HistGradientBoostingRegressor
import common39 as c
import representations35 as rep
import full_graph37 as graph

ROOT=c.ROOT

def maps():
    return sorted(["raw_"+s for s in ("minmax01","standard")]+
      ["nystrom_"+s+"_w"+str(w) for s in ("minmax01","standard") for w in (8,24)]+
      ["kahm_"+s+"_cap"+str(k)+"_p"+str(p) for s in ("minmax01","standard") for k in (4,12,20) for p in (1,2)])

def baselines():
    rows=[{"token":"persistence","kind":"persistence"},{"token":"training_mean","kind":"mean"}]
    for leaves,l2 in ((7,1),(15,1),(31,1),(15,10)):
        model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=leaves,
                  min_samples_leaf=20,l2_regularization=float(l2),early_stopping=False,random_state=3901)
        rows.append({"token":f"hgb_leaf{leaves}_l2_{l2}","kind":"hgb","parameters":model.get_params()})
    for token,lam in (("ridge_l0001","1/10000"),("ridge_l001","1/100"),("ridge_l1","1")):
        rows.append({"token":token,"kind":"ridge","ridge_lambda":lam,
                     "parameters":{"fit_intercept":True,"solver":"cholesky","alpha_rule":"n*lambda"}})
    return [{**r,"candidate_id":"baseline::"+r["token"]} for r in rows]

def protocol():
    import data39
    names=maps();base=baselines();configs=c.head_configs(names)
    return {"schema":"step39-temporal-selection-protocol-v1","step":39,"created_utc":c.utc(),"seed":3901,
      "question":"Do regularization and extension choice selected by chronological development improve certified prediction on two new domains?",
      "data":data39.data_contract(),"tasks":["tetouan_zone1","metro_traffic"],"stages":["f1","f2","f3","final"],
      "maps":names,"representations":rep.DEFAULT_CONFIG,
      "class":{"kappa":"2","output_interval":["0","1"],"metric":"Euclidean on training-only median-distance-normalized stored features",
        "assurance":"CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES","native_transform_error_enclosed":False,
        "raw_input_lipschitz_claim":False,"future_response_coverage_claim":False,"population_risk_claim":False},
      "heads":{"ridge_lambdas":["0","1/4","1"],"extension_weights":["0","1/2","1"],
        "regularizer":"mean squared distance of anchor values from exact training target mean",
        "regularized_objective":"mean((v-y)^2)+lambda*mean((v-training_mean)^2)",
        "effective_targets":"(y+lambda*training_mean)/(1+lambda)",
        "extension_formula":"clip_I((1-weight)*max_i(v_i-2*d_i)+weight*min_i(v_i+2*d_i))",
        "clipping_order":"AFTER_WEIGHTED_MIXTURE","all_training_anchors":True,
        "affine_ridge_lambda":"1/1000000","solver_settings":graph.SOLVER_SETTINGS,
        "distance_bits":80,"optimizer_sqrt_bits":80,"finite_valid_wide_brackets":"RETAIN_AND_REPORT_UNRESOLVED",
        "configurations":configs,"certified_heads_per_stage":198,"qp_fits_per_stage":54},
      "baselines":{"configurations":base,"clipped":False,"class_membership_certified":False,
        "same_lag_inputs":True,"fitting_scaler":"fold_training_raw_minmax01_scaler","count":9},
      "selection":{"folds":["f1","f2","f3"],"pooling":"Sum of physical-unit SSE divided by total held-out development query count",
        "normalization":"Every fold refits its own exact training-only target normalization; score pooling restores physical units",
        "tie_breakers":["exact physical MSE upper","f3 complete map+head payload bytes","lexical candidate_id"],
        "selectors":list(c.SELECTORS),"primary_selector":"joint_rolling","reference_selector":"default_qp_rolling",
        "ordinary_selector":"ordinary_rolling","selection_control":"joint_last_fold",
        "final_refit":"Same chosen configurations refit on all permitted pretest training rows through August31; all207 configurations retained",
        "test_reselection_allowed":False,"assessment_oracle_fits":False,
        "optimization_selection_stability":{"scope":"Same fold maps, training problems, lambdas, extension weights and fixed development labels",
          "intervals":"Arithmetic predictions expanded by certified optimizer discrepancy, intersected with [0,1], then exact original-label loss and physical pooling",
          "criterion":"Selected joint rolling candidate upper strictly below every other joint candidate lower",
          "otherwise":"UNRESOLVED_not_a_selection_change","future_performance_claim":False}},
      "evaluation":{"prediction_ids":[r["candidate_id"] for r in configs+base],"all_candidate_outcomes_retained":True,
        "primary_per_task":{"validity":"independent provenance, fitting, prediction, selection, chronology and loss checks PASS",
          "gain":"reference_lower>0 and 20*new_upper<=19*reference_lower",
          "ordinary":"new_upper<=ordinary_lower","metric":"physical MSE","both_conditions_required":True},
        "overall_success":"Both task-specific primary criteria and all independent checks pass",
        "months":["2017-09","2017-10","2017-11"],"monthly_role":"DESCRIPTIVE_NOT_INDEPENDENT_REPLICATIONS",
        "optimizer_refinement_consequences":"Predeclared deterministic comparisons for the same final fitted configurations",
        "population_significance_test":False,"future_response_coverage_claim":False},
      "execution":{"single_primary_pass":True,"scientific_reruns_allowed":False,"threads":1,
        "phase_budget_seconds":{"fit":1800,"validate":1800,"select":120,"test":3600,"aggregate":120},
        "native_representation_budget_seconds":300,"failure_policy":"STOP_RETAIN_FAILURE_NO_RETUNING",
        "audit_failure_policy":"STOP_RETAIN_EVIDENCE","gates":"protocol -> foldfit -> foldvalidation -> selection -> finalrefit -> sequential test"},
      "environment":{"python":platform.python_version(),"platform":platform.platform(),
        "packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy","osqp","scikit-learn","joblib","threadpoolctl")}},
      "prior_evidence":{"step38_result_sha256":c.sha(ROOT/"inputs/prior/result38.json"),
        "step38_prediction_corollary_sha256":c.sha(ROOT/"inputs/prior/optimization_prediction_corollary38.json"),
        "prior_steps_immutable":True,"reserved_patients_accessed":0}}

def seal():
    import data39
    if (ROOT/"protocol_seal39.json").exists(): raise RuntimeError("Cannot replace frozen scientific protocol")
    p=protocol();c.save(ROOT/"protocol39.json",p)
    files=[ROOT/"protocol39.json",ROOT/"analysis/interface39.txt",ROOT/"analysis/related_work39.json"]
    for folder in ("source","inputs","tests"):
        files += [x for x in (ROOT/folder).rglob("*") if x.is_file() and "__pycache__" not in x.parts and x.suffix!=".pyc"]
    record={"schema":"step39-protocol-source-seal-v1","sealed_utc":c.utc(),"protocol_sha256":c.sha(ROOT/"protocol39.json"),
            "bound_files":{c.relative(x):c.sha(x) for x in sorted(set(files))},"new_measurement_tokens_decoded":0,
            "single_primary_pass":True}
    ref=c.save(ROOT/"protocol_seal39.json",record)
    data39.event("protocol_frozen",protocol_seal_sha256=ref["sha256"],bound_file_count=len(record["bound_files"]))
    print(c.canonical_bytes({"status":"SEALED","protocol_seal_sha256":ref["sha256"],"bound_files":len(record["bound_files"])}).decode(),end="")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("operation",choices=["seal"]);parser.parse_args();seal()
