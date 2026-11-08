"""Freeze the whole Step40 design before new numerical measurements are read."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
from sklearn.ensemble import HistGradientBoostingRegressor
import common40 as c
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
    import data40
    names=maps();base=baselines();configs=c.head_configs(names)
    return {"schema":"step40-head-selection-protocol-v1","step":40,"created_utc":c.utc(),"seed":3901,
      "question":"Does joint affine/QP head and representation selection improve forecasting, and which fixed model decisions survive exact QP refinement?",
      "data":data40.data_contract(),"tasks":list(data40.task_ids()),"stages":["f1","f2","f3","final"],
      "maps":names,"representations":rep.DEFAULT_CONFIG,
      "class":{"kappa":"2","output_interval":["0","1"],"metric":"Euclidean on training-only median-distance-normalized stored features",
        "assurance":"CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES","native_transform_error_enclosed":False,
        "raw_input_lipschitz_claim":False,"future_response_coverage_claim":False,"population_risk_claim":False},
      "heads":{"ridge_lambdas":["0"],"extension_weights":["1/2"],
        "regularizer":"none: lambda=0; unchanged general inherited learner implementation",
        "regularized_objective":"mean((v-y)^2)","effective_targets":"y",
        "extension_formula":"clip_I((max_i(v_i-2*d_i)+min_i(v_i+2*d_i))/2)",
        "clipping_order":"AFTER_WEIGHTED_MIXTURE","all_training_anchors":True,
        "affine_ridge_lambda":"1/1000000","solver_settings":graph.SOLVER_SETTINGS,
        "distance_bits":80,"optimizer_sqrt_bits":80,"finite_valid_wide_brackets":"RETAIN_AND_REPORT_UNRESOLVED",
        "configurations":configs,"certified_heads_per_stage":36,"qp_fits_per_stage":18,
        "same_map_pairing":True,"affine_exact_training_optimizer_claim":False,
        "inherited_learner":"source/learners39.py"},
      "baselines":{"configurations":base,"clipped":False,"class_membership_certified":False,
        "same_lag_inputs":True,"fitting_scaler":"fold_training_raw_minmax01_scaler","count":9},
      "selection":{"folds":["f1","f2","f3"],"pooling":"Sum of physical-unit SSE divided by total held-out development query count",
        "normalization":"Each fold refits exact training-only target normalization; score pooling restores physical units",
        "tie_breakers":["exact physical MSE upper","f3 complete map+head payload bytes","lexical candidate_id"],
        "selectors":list(c.SELECTORS),"primary_selector":"joint_rolling","reference_selector":"qp_rolling",
        "ordinary_selector":"ordinary_rolling","affine_selector":"affine_rolling","selection_control":"joint_last_fold",
        "union_selector_logic":"With the same criterion and tie rule, a joint affine winner equals the affine-only winner and a joint QP winner equals the QP-only winner. The five-percent QP-relative endpoint therefore tests the benefit of admitting an affine winner, not superiority over affine-only selection.",
        "final_refit":"All45 fixed configurations refit on permitted pretest training rows through August31; selectors retain development IDs",
        "test_reselection_allowed":False,"assessment_oracle_fits":False,
        "mixed_comparator_decision":{"scope":"Fixed fitted affine coefficient payloads and exact QP anchor optima with midpoint extensions, same fold maps and fixed development labels",
          "intervals":"Affine arithmetic risks; QP arithmetic predictions expanded by certified optimizer discrepancy and clipped to[0,1]; exact label loss and physical pooling",
          "unique_winner":"Chosen upper strictly below every other mixed comparator lower",
          "survivors":"Candidates whose lower is at most the smallest comparator upper",
          "family_separation":"Minimum affine risk interval compared to minimum QP risk interval",
          "otherwise":"UNRESOLVED; no selection change inferred","affine_training_optimality_claim":False,
          "computational_saving_claim":False,"all_candidates_fitted":True,"future_performance_claim":False}},
      "evaluation":{"prediction_ids":[r["candidate_id"] for r in configs+base],"all_candidate_outcomes_retained":True,
        "primary_per_task":{"validity":"All independent provenance, fitting, prediction, selection, chronology and loss checks PASS",
          "gain":"qp_lower>0 and20*joint_upper<=19*qp_lower",
          "ordinary":"same final payload or joint_upper<=ordinary_lower",
          "affine":"same final payload or joint_upper<=affine_lower",
          "same_payload_fields":["candidate_id","head_id","head_sha256","test_predictions_digest"],
          "metric":"physical MSE","all_three_conditions_required":True,
          "same_joint_affine_is_not_joint_superiority":True},
        "overall_success":"Both task-specific primary criteria and all independent checks pass",
        "months_by_task":{task:[str(data40.prep.TASKS[task]["year"])+"-"+m for m in ("09","10","11")] for task in data40.task_ids()},
        "monthly_role":"DESCRIPTIVE_NOT_INDEPENDENT_REPLICATIONS",
        "same_map_comparisons":{"count_per_task":18,"signed_difference":"QP minus affine physical MSE; positive favors affine",
          "operational_and_exact_QP_comparators":True,"role":"DESCRIPTIVE_DEPENDENT_CONFIGURATIONS_NOT_INDEPENDENT_REPLICATIONS",
          "attribution":"Same stored map and data, whole head recipe changes; not pure capacity mechanism"},
        "optimizer_refinement_consequences":"Deterministic comparisons for the same fitted configurations, fixed data and stored features",
        "population_significance_test":False,"future_response_coverage_claim":False},
      "execution":{"single_primary_pass":True,"scientific_reruns_allowed":False,"threads":1,
        "phase_budget_seconds":{"fit":1800,"validate":1800,"select":120,"test":3600,"aggregate":120},
        "native_representation_budget_seconds":300,"failure_policy":"STOP_RETAIN_FAILURE_NO_RETUNING",
        "audit_failure_policy":"STOP_RETAIN_EVIDENCE","gates":"protocol -> foldfit -> foldvalidation -> selection -> finalrefit -> sequential test"},
      "environment":{"python":platform.python_version(),"platform":platform.platform(),
        "packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy","osqp","scikit-learn","joblib","threadpoolctl")}},
      "prior_evidence":{"step39_result_sha256":c.sha(ROOT/"inputs/prior/output__result39.json"),
        "step39_audit_sha256":c.sha(ROOT/"inputs/prior/audit__postrun_audit39.json"),
        "inherited_sources_sha256":c.sha(ROOT/"inputs/inherited_sources40.json"),
        "prior_steps_immutable":True,"reserved_patients_accessed":0,
        "motivation":"Step39 primary negative; predeclared affine selector promising but changed both map and head; Step40 is a fresh fixed paired comparison"}}

def seal():
    import data40
    if (ROOT/"protocol_seal40.json").exists(): raise RuntimeError("Cannot replace frozen scientific protocol")
    p=protocol();c.save(ROOT/"protocol40.json",p)
    files=[ROOT/"protocol40.json",ROOT/"analysis/interface40.txt",ROOT/"analysis/related_work40.json"]
    for folder in ("source","inputs","tests"):
        files += [x for x in (ROOT/folder).rglob("*") if x.is_file() and "__pycache__" not in x.parts and x.suffix!=".pyc"]
    record={"schema":"step40-protocol-source-seal-v1","sealed_utc":c.utc(),"protocol_sha256":c.sha(ROOT/"protocol40.json"),
            "bound_files":{c.relative(x):c.sha(x) for x in sorted(set(files))},"new_measurement_tokens_decoded":0,
            "zero_preseal_measurement_scope":"Selected2009Household/2016Jena cohorts in this study; public metadata includes unrelated illustrative2004Jena rows",
            "single_primary_pass":True}
    ref=c.save(ROOT/"protocol_seal40.json",record)
    data40.event("protocol_frozen",protocol_seal_sha256=ref["sha256"],bound_file_count=len(record["bound_files"]))
    print(c.canonical_bytes({"status":"SEALED","protocol_seal_sha256":ref["sha256"],"bound_files":len(record["bound_files"])}).decode(),end="")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("operation",choices=["seal"]);parser.parse_args();seal()
