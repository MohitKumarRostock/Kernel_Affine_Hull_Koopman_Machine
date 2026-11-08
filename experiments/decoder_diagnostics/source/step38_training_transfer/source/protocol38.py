"""Freeze the fresh-source training/validation/sequential-assessment experiment."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import full_graph37 as graph
import representations35 as rep
from learners38 import RECIPES

ROOT=Path(__file__).resolve().parents[1]
CHECKS=("tests/artificial_data38.json","tests/artificial_learners38.json","tests/artificial_audit38.json")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
def map_names():
    return sorted(["raw_"+s for s in ("minmax01","standard")]+
        ["nystrom_"+s+"_w"+str(w) for s in ("minmax01","standard") for w in (8,24)]+
        ["kahm_"+s+"_cap"+str(c)+"_p"+str(p) for s in ("minmax01","standard") for c in (4,12,20) for p in (1,2)])

def protocol():
    if (ROOT/"protocol_seal38.json").exists():raise RuntimeError("Protocol already sealed")
    previous=json.loads((ROOT/"inputs/prior/protocol37.json").read_text())
    source=json.loads((ROOT/"inputs/public_source38.json").read_text())
    maps=map_names();baseline=["persistence","training_mean","hist_gradient_boosting"]
    p={"schema":"step38-training-transfer-protocol-v1","step":38,
      "title":"Training-only full-anchor decoder transfer to a fresh public forecasting assessment",
      "created_utc":datetime.now(timezone.utc).isoformat(),"seed":3801,
      "primary_question":"Does validation-selected qp_all improve fresh test MSE by at least5percent over validation-selected legacybest while no worse than the validation-selected ordinary baseline?",
      "prior":{"step37_result_sha256":sha(ROOT/"inputs/prior/result37.json"),
          "step37_audit_sha256":sha(ROOT/"inputs/prior/postrun_audit37.json"),
          "step37_protocol_seal_sha256":sha(ROOT/"inputs/prior/protocol_seal37.json"),
          "finding":"All54classesadequate;39bothlegacyheadsfail;targetenvelope54/54adequate;strongerclasslowercannotexclude"},
      "data":{"dataset":"Seoul Bike Sharing Demand","uci_id":560,"doi":"10.24432/C5F62R","target":"rented_bike_count",
          "archive":"inputs/seoul_bike_uci560.zip","member":"SeoulBikeData.csv","archive_sha256":source["archive_sha256"],
          "source_timezone":"local_naive_Seoul_date_and_hour_no_UTC_reinterpretation",
          "train_start":"2017-12-01","train_end":"2018-07-31","validation_start":"2018-08-01","validation_end":"2018-08-31",
          "test_start":"2018-09-01","test_end":"2018-11-30","lag_hours":24,"horizon_hours":24,"origin_hour":23,
          "partition_basis":"target_timestamp","expected_n":{"train":242,"validation":31,"test":91},
          "input_order":"Own24consecutivehourlycountsendingatforecastorigin_oldestfirst",
          "features":"Onlycausallyavailableowncounthistory;noweathercalendaroroperationflagpredictors",
          "operation_closures":"Keepallrecordedhoursincludingclosures;recordedrentalsnotlatentunconstraineddemand",
          "eligibility":"Allrequiredtimestampspresentandcountsfiniteintegernonnegative;invalidmeasurementstopswithoutretuning",
          "target_normalization":"Exact(raw-training_target_min)/(training_target_max-training_target_min)",
          "degenerate_range_action":"STOP_RETAIN_FAILURE","targets_clipped":False,
          "training_numeric_gate":"independent_preexecution_PASS_and_protocol_source_archive_seal",
          "validation_numeric_gate":"allheadandbaselinefitseal_and_independent_prevalidation_PASS",
          "test_numeric_gate":"model_selectionseal_and_independent_pretest_PASS_then_causal_per_query_access",
          "test_order":"history_counts_at_or_before_origin_then_all93_prediction_receipt_seal_then_current_target_reveal",
          "previously_revealed_targets_may_be_later_history":True,"no_online_refit":True,
          "new_source_exposure_claim":"NoScopedauditretainedprioruse_and_nonewnumericalmeasurementaccessbeforefreeze;notabsolutehistoricalunfamiliarity"},
      "class":{"kappa":"2","prediction_interval":["0","1"],"metric":"Euclidean on fixed median-distance-normalized stored features",
          "assurance":graph.ASSURANCE,"native_transform_error_enclosed":False,"raw_input_lipschitz_claim":False,
          "population_risk_claim":False,"future_response_coverage_claim":False},
      "representations":dict(rep.DEFAULT_CONFIG),"maps":maps,
      "heads":{"recipes":list(RECIPES),"expected_count":90,"anchor_count":32,"anchor_rule":"equal_row_stratum_midpoints_on_training_rows",
          "ridge_lambda":"1/1000000","legacy_extension_max_updates":4000,"legacy_check_every":20,"legacy_gap_sse":"1/10000000",
          "distance_bits":80,"grid_bits":40,"solver_settings":dict(graph.SOLVER_SETTINGS),"solver_version":"1.0.5",
          "qp32":"Same32anchorsaslegacyextension;fixednewQPsolver_exactrepair_minall3candidates",
          "envelope_all":"Alltraininganchors_min_exactSSE_targetenvelope_and_constant_lexicalsource_tie",
          "qp_all":"Alltraininganchors_min_exactSSE_targetenvelope_solverenvelope_constant_lexicalsource_tie",
          "extension_formula":"clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)",
          "new_schema":"step38-stored-feature-extension-head-v1","legacy32guard_unchanged":True,
          "original_training_labels_in_objective":True,"numerical_status_not_a_proof":True,"fallbacks_retained":True,
          "cost_accounting":"Fullgraphfitshared_byqp_allandenvelope_all;sharedquerydistanceschargedonce;nounmeasuredstandalonelatencyclaim"},
      "baselines":{"names":baseline,"class_membership_certified":False,"clipped":False,
          "hgb_parameters":{"loss":"squared_error","learning_rate":0.05,"max_iter":200,"max_leaf_nodes":15,"max_depth":None,
            "min_samples_leaf":20,"l2_regularization":1.0,"max_features":1.0,"max_bins":255,"categorical_features":None,
            "monotonic_cst":None,"interaction_cst":None,"warm_start":False,"early_stopping":False,"random_state":3801}},
      "selection":{"objective":"minimum_validation_MSE_upper_then_deployment_payload_bytes_then_lexical_candidate_id",
          "per_recipe_maps":18,"legacybest_candidates":36,"ordinarybest_candidates":3,
          "eligibility_filter_used_for_primary_selection":False,"epsilon_factor":"19/20",
          "epsilon_recipe":"0.95timesexactnormalizedtrainingpersistenceMSE;diagnosticeligibilityonly",
          "no_validation_refit":True,"no_test_reselection":True},
      "evaluation":{"prediction_ids":[m+"::"+r for m in maps for r in RECIPES]+["baseline::"+b for b in baseline],
          "all_panel_test_outcomes_retained":True,"monthly_breakdown":"2018-09,2018-10,2018-11_descriptive_notindependentreplications",
          "primary_metric":"MSE","physical_unit_metric":"RMSE_of_hourly_rental_count",
          "exact_difference_interval":"[legacy_lower-new_upper,legacy_upper-new_lower]",
          "practical_gate":{"positive_legacy_denominator_required":True,"five_percent_MSE_gain":"20*new_test_upper<=19*legacybest_test_lower",
              "ordinary_baseline_comparison":"new_test_upper<=ordinarybest_test_lower",
              "validity":"Allindependentmodel_prediction_loss_selection_andaccesschecksPASS",
              "undefined_zero_denominator":"UNDEFINED_RELATIVE_GAIN_notasuccess",
              "interpretation":"Finitefreshassessmentcriterion_notpopulationnoninferiority_orfuturecoverage"},
          "secondary_test_class_diagnosis":{"enabled":True,"maps":18,"after":"alltestpredictionandtargetrecordscomplete",
              "uses_test_labels":True,"head_promotion_allowed":False,"selection_changes_allowed":False,
              "upper":"min_diagnostic_primal_and_all5_frozenhead_testuppers","head_excess":"[max(0,head_lower-class_upper),max(0,head_upper-class_lower)]",
              "cost_reported_separately":True}},
      "execution":{"single_primary_pass":True,"phase_order":["train","validate","test"],"phase_budget_seconds":{"train":1800,"validate":1200,"test":2400},
          "native_representation_budget_seconds":600,"no_outcome_driven_reruns":True,"failure_rule":"Stopretainpartialwithoutadaptivechanges",
          "threads":1,"scientific_public_data_reads_authorized":True,"reserved_patients_accessed":0,
          "inherited_study_gates":{"G3":"NOT_ESTABLISHED","G4":"NOT_STARTED","natural_study":"NO_GO"}},
      "reporting":{"all90heads_all3baselines_and_all18secondaryclasses":True,"primaryandsecondaryseparate":True,
          "new_fullanchor_recipe_not_newLipschitzregressiontheory":True,"monthlymetricsdescriptive":True,
          "new_dataset_count":1,"general_AI_significance_not_guaranteed":True,"priornegativefindingspreserved":True},
      "environment":{"python":platform.python_version(),"platform":platform.platform(),
          "packages":{n:importlib.metadata.version(n) for n in ("numpy","scipy","osqp","scikit-learn","joblib","threadpoolctl")}},
      "required_artificial_checks":list(CHECKS)}
    write(ROOT/"protocol38.json",p)
    print(json.dumps({"status":"WRITTEN_NOT_SEALED","protocol_sha256":sha(ROOT/"protocol38.json")}))

def seal():
    if (ROOT/"protocol_seal38.json").exists():raise RuntimeError("Cannot reseal")
    p=json.loads((ROOT/"protocol38.json").read_text())
    for relative in CHECKS:
        check=json.loads((ROOT/relative).read_text())
        if check.get("status")!="PASS":raise RuntimeError("Missing artificial PASS "+relative)
        for name,digest in check.get("source_sha256",{}).items():
            if sha(ROOT/name)!=digest:raise RuntimeError("Stale artificial source receipt "+name)
    paths=[ROOT/"protocol38.json"]
    paths += list((ROOT/"source").glob("*.py"))
    paths += [x for x in (ROOT/"inputs").rglob("*") if x.is_file() and "__pycache__" not in x.parts]
    paths += [x for x in (ROOT/"tests").glob("*.json")]
    for name in ("reporting/methods38.txt","analysis/methods_design_review38.json","analysis/implementation_review38.json"):
        if (ROOT/name).exists():paths.append(ROOT/name)
    record={"schema":"step38-protocol-source-input-seal-v1","sealed_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":sha(ROOT/"protocol38.json"),"bound_files":{str(x.relative_to(ROOT)):sha(x) for x in sorted(set(paths))},
        "new_dataset_count_values_parsed_before_seal":0,"prior_step37_outcomes_known":True}
    write(ROOT/"protocol_seal38.json",record)
    import data38
    data38.event("protocol_sealed",protocol_sha256=record["protocol_sha256"],protocol_seal_sha256=sha(ROOT/"protocol_seal38.json"))
    print(json.dumps({"status":"SEALED","protocol_seal_sha256":sha(ROOT/"protocol_seal38.json"),"bound_files":len(record["bound_files"])}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("write","seal"));a=parser.parse_args()
    protocol() if a.mode=="write" else seal()
