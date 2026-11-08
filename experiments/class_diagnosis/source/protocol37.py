"""Write and seal the complete Step37 diagnostic before gated numerical access."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import data37 as d
import full_graph37 as g
from prepare_inputs37 import TASKS,map_names

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_CHECKS=("tests/artificial_data_preflight37.json","tests/artificial_producer_preflight37.json",
                 "audit/artificial_audit_preflight37.json")

def write_protocol():
    if (ROOT/"protocol_seal37.json").exists():raise RuntimeError("Step37 already sealed")
    manifest=d.verify_input_manifest()
    protocol={"schema":"step37-prospective-diagnostic-protocol-v1","step":37,
      "title":"Full-class finite-cohort adequacy diagnosis after Step36",
      "created_utc":datetime.now(timezone.utc).isoformat(),"tasks":list(TASKS),"map_order":list(map_names()),
      "expected_classes":54,"inherited_heads":108,"cohort":"previously_opened_step36_decision",
      "input_manifest_sha256":d.sha_file(ROOT/d.MANIFEST_PATH),"prior":manifest["prior"],
      "scientific_scope":{"exploratory_analysis":True,"new_independent_validation":False,
          "features_and_targets_previously_opened_in_step36":True,
          "frozen_before_new_step37_numerical_diagnostic":True,"training_or_test_arrays_allowed":False,
          "raw_source_archive_allowed":False,"reserved_patient_arrays_allowed":False,
          "new_prediction_model_fits_allowed":False,"diagnostic_uses_decision_labels":True,
          "diagnostic_is_not_future_forecast_evidence":True,"native_transform_error_enclosed":False,
          "g3":"NOT_ESTABLISHED","g4":"NOT_STARTED","natural_study":"NO_GO"},
      "class":{"kappa":"2","prediction_interval":["0","1"],"assurance":g.ASSURANCE,
               "feature_ids_and_class_ids_unchanged":True,"full_pairwise_constraints":True,
               "norm":"Euclidean","loss":"original_unclipped_target_squared_error"},
      "thresholds":{task:{"epsilon":manifest["tasks"][task]["epsilon"],
          "epsilon_physical_mse":manifest["tasks"][task]["epsilon_physical_mse"],
          "rule":"unchanged_training_derived_step36_tolerance"} for task in TASKS},
      "representation_controls":{"both_scalings":["minmax01","standard"],
          "kahm_native_caps":[4,12,20],"kahm_realized_dimension_rule":"min(cap,D,N_local-1)",
          "kahm_prefixes":[1,2],"maps_refit":False,"maps_added_or_removed":False},
      "numerics":{"solver":"OSQP","version":"1.0.5","algebra":"builtin",
          "settings":dict(g.SOLVER_SETTINGS),"distance_bits":80,"proposal_and_dual_grid_bits":40,
          "objective":"unhalved_SSE_without_constant_P2I_q_minus2y",
          "qp_constraints":"box_I_plus_all_nonredundant_pair_rows",
          "omit_pair_only_if":"kappa_times_exact_lower_distance_ge_output_interval_width",
          "sqrt_receipt":"lexicographic_pair_floor80_checked_by_integer_squared_inequalities",
          "proposal_rounding":"nearest_dyadic_ties_even","solver_status_is_not_a_proof":True,
          "unfinished_finite_vectors_are_usable":True,"numerical_exception_policy":"record_then_fixed_exact_fallback",
          "adaptive_retries_allowed":False},
      "primal":{"candidate_order":["clipped_target_envelope","solver_envelope_if_finite","clipped_mean_constant"],
          "envelopes":"exact_complete_graph_multisource_Dijkstra_for_minorant_and_majorant",
          "repair":"exact_best_convex_blend_against_original_targets",
          "proposal_targets_clipped_only_for_range":True,"loss_targets_clipped":False,
          "select":"minimum_exact_SSE_then_lexicographic_source",
          "class_upper":"min_new_verified_primal_mean_and_best_inherited_admissible_head_upper"},
      "dual":{"signed_edges":"rounded_OSQP_edge_duals_box_duals_unused",
          "endpoint":"upper_outward_distance","scalar_minimizer":"clip(y-Btranspose_lambda/2,I)",
          "lower":"max_common_range_floor_and_signed_weak_dual_mean",
          "absent_nonfinite_fallback":"all_edge_multipliers_zero"},
      "decision":{"adequate":"upper_le_epsilon","inadequate":"lower_gt_epsilon","otherwise":"UNRESOLVED",
          "gap_target_mean":"1/100000000","gap_resolution_is_separate_from_threshold_decision":True,
          "head_excess":"[max(0,head_lower-class_upper),max(0,head_upper-class_lower)]",
          "head_inadequacy_requires":"head_lower_gt_epsilon",
          "new_finite_cohort_adequacy":"class_adequate_and_neither_inherited_head_verified_eligible"},
      "execution":{"single_primary_pass":True,"task_order":list(TASKS),"map_order":list(map_names()),
          "global_wall_budget_seconds":1800.0,"budget_checked_between_classes":True,
          "hard_arithmetic_or_provenance_failure":"stop_and_retain_partial_evidence_no_rerun",
          "numerical_failure":"retain_class_with_fixed_exact_fallback_no_retry",
          "budget_stop":"retain_all_completed_classes_and_explicit_missing_roster",
          "new_heldout_forecasts":0,"no_outcome_driven_setting_changes":True,
          "input_loader_requires_independent_preexecution_pass":True},
      "reporting":{"all_classes_and_heads_required":True,"missing_or_failed_classes_must_be_explicit":True,
          "publish_exact_rational_brackets":True,"publish_independent_receipt_audit":True,
          "report_finite_cohort_scope":True,"causal_optimizer_failure_claim":False,
          "out_of_sample_improvement_claim":False,"new_lipschitz_QP_novelty_claim":False,
          "new_exact_QP_certificate_paradigm_claim":False,
          "prior_step36_test_metrics_and_decisions_unchanged":True},
      "environment":{"python":platform.python_version(),"platform":platform.platform(),
          "packages":{name:importlib.metadata.version(name) for name in ("numpy","scipy","osqp")},
          "numerical_threads":1},
      "required_artificial_checks":list(REQUIRED_CHECKS)}
    if protocol["environment"]["packages"]["osqp"]!="1.0.5":raise RuntimeError("Unexpected frozen solver version")
    d.write_json(ROOT/"protocol37.json",protocol)
    print(json.dumps({"protocol_sha256":d.sha_file(ROOT/"protocol37.json"),"status":"WRITTEN_NOT_SEALED"}))

def seal():
    if (ROOT/"protocol_seal37.json").exists() or (ROOT/"output/primary_started37.json").exists():
        raise RuntimeError("Protocol can be sealed only once before primary execution")
    protocol=json.loads((ROOT/"protocol37.json").read_text());manifest=d.verify_input_manifest()
    for relative in REQUIRED_CHECKS:
        receipt=json.loads((ROOT/relative).read_text())
        if receipt.get("status")!="PASS":raise RuntimeError("Artificial gate is not PASS: "+relative)
        for name,sha in receipt.get("source_sha256",{}).items():
            if d.sha_file(ROOT/name)!=sha:raise RuntimeError("Artificial source check stale: "+name)
    selected={ROOT/"protocol37.json",ROOT/d.MANIFEST_PATH}
    selected.update(ROOT/item["path"] for item in manifest["files"])
    selected.update((ROOT/"source").glob("*.py"))
    selected.update((ROOT/"tests").glob("*.json"))
    selected.add(ROOT/"audit/artificial_audit_preflight37.json")
    for relative in ("reporting/methods37.txt","reporting/methods37_artificial_preflight.py",
                     "reporting/methods37_artificial_preflight_result.json","audit/producer_method_review37.json"):
        if (ROOT/relative).is_file():selected.add(ROOT/relative)
    bounds={str(p.relative_to(ROOT)):d.sha_file(p) for p in sorted(selected)}
    record={"schema":"step37-protocol-source-input-seal-v1","sealed_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":d.sha_file(ROOT/"protocol37.json"),"input_manifest_sha256":d.sha_file(ROOT/d.MANIFEST_PATH),
        "bound_files":bounds,"scientific_numerical_arrays_accessed_by_step37_before_seal":0,
        "all_prior_step36_outcomes_already_known":True,"exploratory_not_new_validation":True}
    d.write_json(ROOT/"protocol_seal37.json",record)
    d.event("protocol_sealed",protocol_sha256=record["protocol_sha256"],
            protocol_seal_sha256=d.sha_file(ROOT/"protocol_seal37.json"),bound_files=len(bounds))
    print(json.dumps({"status":"SEALED","protocol_seal_sha256":d.sha_file(ROOT/"protocol_seal37.json"),"bound_files":len(bounds)}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("write","seal"));args=parser.parse_args()
    write_protocol() if args.mode=="write" else seal()
