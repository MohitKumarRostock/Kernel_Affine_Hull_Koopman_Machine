"""Construct and seal the prospective Step36 external search-policy protocol."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from representations35 import DEFAULT_CONFIG

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")


def draft():
    return {
      "schema":"step36-triggered-search-protocol-v1","step":36,
      "title":"External test of certificate-triggered search beyond feasibility and cost pruning",
      "primary_question":"Does a triggered pair certificate avoid additional head fits and reduce actual matched search time relative to ordinary feasibility-plus-cost pruning while preserving the exhaustive threshold/cost choice?",
      "tasks":["casual","registered","cnt"],
      "data":{"dataset":"UCI Bike Sharing","uci_id":275,"doi":"10.24432/C5W894","license":"CC BY 4.0",
              "archive":"inputs/bike_sharing_uci275.zip","source_receipt":"inputs/public_source36.json",
              "member":"hour.csv","other_members_used_for_measurements":False,
              "origin_hour":23,"horizon_hours":24,"lag_hours":24,"partition_basis":"target timestamp",
              "train_start":"2011-01-01","train_end":"2011-12-31",
              "decision_start":"2012-01-01","decision_end":"2012-06-30",
              "test_start":"2012-07-01","test_end":"2012-12-31",
              "input_order":"Own target's24 consecutive hourly counts ending at origin, oldest to newest",
              "weather_calendar_or_other_count_features":False,"use_future_observed_weather":False,
              "eligibility":"Daily23:00 target calendar; all required lag timestamps and target timestamp must exist, with all3 count columns finite; no zero imputation",
              "common_roster_across_targets":True,"rolling_observed_history":True,"embargo_hours":0,
              "target_normalization":"Exact (raw-training_target_min)/(training_target_max-training_target_min), separately per target",
              "targets_clipped":False,"minimum_training_rows":32,"degenerate_range_action":"STOP_RETAIN_FAILURE",
              "target_relation":"cnt=casual+registered; these are correlated targets within one dataset, not independent replications",
              "test_barrier":"All Jul-Dec2012 count values and missingness remain uninterpreted until every model and policy is sealed"},
      "pre_freeze_exposure":{"programmatic_measurement_parsing":False,"public_model_fits":0,
              "incidental_training_snippet_seen":True,"scope":"A metadata-search snippet displayed a few January2011 training rows; values were not used to choose thresholds, origin, models or anticipated outcomes",
              "decision_or_test_measurements_seen":False,"broad_all_values_unseen_claim":False},
      "representations":DEFAULT_CONFIG,"seed":3601,
      "class":{"kappa":"2","prediction_interval":["0","1"],"metric":"Euclidean on frozen median-distance-normalized stored features",
               "assurance":"CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES","native_transform_error_enclosed":False,
               "population_risk_claim":False,"future_response_coverage_claim":False},
      "heads":{"recipes":["affine","extension32"],"ridge_lambda":"1/1000000","anchor_count":32,
               "anchor_rule":"equal-row-stratum midpoints in chronological training rows, shared across maps per target",
               "affine_rule":"All-training-row centered ridge; exact coefficient norm check/projection and exact stored-training-mean recentering; range clipping",
               "extension_rule":"Conservative lower-distance anchor QP, symmetric cone extension and range clipping",
               "extension_max_coordinate_attempts":4000,"multiplier_grid_bits":40,"check_every":20,"gap_sse":"1/10000000",
               "unfinished_rule":"Retain valid final witnesses; no outcome-driven extra iterations or replacement"},
      "selection":{"epsilon_factor":"19/20","epsilon_recipe":"Same as Step35:0.95 times exact normalized training persistence MSE, independently for each target",
                   "tolerance_not_tightened_from_step35_outcomes":True,
                   "eligibility":"achieved decision-risk upper endpoint<=epsilon",
                   "objective":"Minimum complete independently serialized representation+head parameter bytes among eligible heads",
                   "ties":["payload bytes ascending","decision risk upper ascending","candidate ID lexical ascending"],
                   "cost_excludes":"Shared source/runtime code and audit witnesses; includes encoder/reference/scaler/metric state and actual head state",
                   "no_eligible_action":"ABSTAIN"},
      "search":{"map_order":"Ascending frozen encoder pickle bytes, then map name lexical; available before all head fits",
                "head_order":["affine","extension32"],
                "modes":["exhaustive","ordinary","triggered"],
                "exhaustive":"Fit/evaluate every36-head candidate; no pruning; provides canonical proof/evaluation records",
                "ordinary":"Common range floor; known encoder+head-template lower payload bound; cheap affine first; update incumbent; recheck extension cost; fit all remaining competitive heads",
                "triggered":"Same ordinary rules; call pair certificate only when affine is not verified eligible, extension remains cost competitive, common range floor does not exclude and call budget remains",
                "cost_bound":"Exact canonical zero-string head template plus known encoder bytes; unknown rational parameter strings have length>=1",
                "cost_prune":"Strict cost lower bound>current eligible incumbent complete cost; equality never pruned",
                "affine_feasible_effect":"Disable class-inadequacy certificate only; extension still requires its own valid cost pruning condition",
                "range_floor":"Exact average squared distance of unchanged decision targets to I; if>epsilon both ordinary and triggered abstain before head fits",
                "class_prune":"Verified pair lower>epsilon; retains valid fixed-panel tolerance/cost choice",
                "no_retrospective_costs_or_head_cache":True},
      "certificate":{"type":"Disjoint-pair range-aware finite-cohort class lower bound",
                     "distance_bits":80,"neighbors":4,"max_pairs_per_call":64,"max_calls_per_target_run":6,
                     "pair_selection":"Full floating Euclidean pdist matrix; nearest4 candidates ordered by distance then neighbor query ID then index; unique unordered candidate edges; descending positive clipped-target Lipschitz violation then lexically sorted endpoint query IDs then indices; greedily disjoint",
                     "exact_pair_witness":"Analytic box-constrained two-point least-squares optimum using80bit distance UPPER endpoint",
                     "aggregation":"Sum disjoint pair SSE minima plus unmatched singleton range floors, divided by full decision cohort N; never double-count floor",
                     "pair_budget_scope":"At most384 exact selected-pair square-root enclosures per triggered target/run; does not bound floating matching work or wall time",
                     "approximate_distance_work_counted":True,
                     "no_pair_candidates_action":"Return range-only lower withzero exact pair distances",
                     "no_full_block_QP_escalation":True},
      "timing":{"ordinary_triggered_repetitions":5,"exhaustive_repetitions":1,
                "order":"For target index t and paired repetition r starting0, ordinary then triggered if(t+r)even; reversed ifodd",
                "exhaustive_executed_first_per_target":True,
                "canonical_results_unavailable_to_timed_policy":True,
                "measurement":"Whole in-memory policy execution including actual head refits, exact decision evaluation, payload serialization, hashes, policy bookkeeping and any pair certificate; disk evidence writes occur after clock stops",
                "shared_work":"Frozen representation construction, train/decision feature cache, and complete head-template byte-bound construction are performed once per target and reported separately",
                "independent_head_fits_each_execution":True,
                "outcome_identity":"All repeated head/prediction digests must match canonical exhaustive records after timed execution ends",
                "summary":"All5 paired times retained; report median paired ordinary-minus-triggered time and ranges; no iid speedup confidence claim",
                "no_deployment_latency_or_runtime_size_claim":True},
      "success":{"safety":"Every ordinary and triggered selection/abstention equals canonical exhaustive same-rule choice, on every target and repetition",
                 "per_target_incremental_utility":"Triggered avoids strictly more head fits than ordinary AND median paired ordinary-minus-triggered search time is strictly positive, with safety verified",
                 "consistent_panel_claim":"Per-target incremental criterion holds for all3 correlated targets; otherwise report mixed/negative results without broad utility claim",
                 "timing_noise_guard":"Positive measured time difference without additional avoided head fits is never certificate-specific utility",
                 "accuracy_claim":"Equal selected head identity gives equal achieved forecast and payload outcomes; any forecast comparison with other predictors remains descriptive"},
      "baselines":{"outside_search_panel":True,"names":["persistence","training_mean","hist_gradient_boosting"],
                   "ordinary_mse_selector":"Descriptive control from the exhaustive panel: minimum achieved decision-risk upper, then deployment bytes, then lexical ID; frozen before test and not a search-efficiency comparator",
                   "hist_gradient_boosting_parameters":{"loss":"squared_error","learning_rate":0.05,"max_iter":200,"max_leaf_nodes":15,"max_depth":None,
                       "min_samples_leaf":20,"l2_regularization":1.0,"max_features":1.0,"max_bins":255,"categorical_features":None,
                       "monotonic_cst":None,"interaction_cst":None,"warm_start":False,"early_stopping":False,"random_state":3601},
                   "data":"All eligible training rows, same24 causal inputs; training-only MinMax01 scaler",
                   "clipped":False,"class_membership_certified":False},
      "execution":{"primary_experimental_runs":1,"predeclared_timing_repetitions_are_not_adaptive_refits":True,
                   "overall_wall_budget_seconds":3600,"native_family_budget_seconds_per_target":600,"threads":1,
                   "failure_rule":"Stop and retain partial outcomes; no outcome-driven primary rerun",
                   "scientific_public_data_reads_authorized":True,"reserved_patient_arrays_read":False,"reserved_patients_accessed":0,
                   "reserved_patients_total":5557,"inherited_study_statuses":{"G3":"NOT_ESTABLISHED","G4":"NOT_STARTED","natural_study":"NO_GO"}},
      "report":{"all108_canonical_head_outcomes_retained":True,"test_no_reselection":True,"single_external_dataset_with_correlated_targets":True,
                "keep_step35_negative_result":True,"generic_safe_screening_novelty_claim":False,
                "required_outputs":["Full report","Manuscript insert","Exact results","Search traces and timing table","Scientific figure","Complete reproducibility package"]},
      "environment":{"python":platform.python_version(),"platform":platform.platform(),"packages":{n:importlib.metadata.version(n) for n in ["numpy","scipy","scikit-learn","pandas","joblib","threadpoolctl"]}},
      "prior":{"step35_result_sha256":"434fa12bc875de71da8f35adda85dc451b7a678d02f0ebc0ae99ef9b9615e05e",
               "step35_package_sha256":"64ac8ad5541583c788c3e0fd675bfa44ed01d1d72dc0b93f03fc42a580675b76",
               "step35_finding":"All18 affine heads feasible; zero possible class exclusions atfrozencriterion; ordinary encoder-cost pruning stronger than unnecessary class checks"}}


def main():
    p=argparse.ArgumentParser();p.add_argument("--seal",action="store_true");args=p.parse_args()
    seal=ROOT/"protocol_seal36.json";protocol=ROOT/"protocol36.json"
    if seal.exists():raise RuntimeError("Already sealed; do not replace frozen source/protocol")
    if not args.seal:
        write(protocol,draft());print(json.dumps({"status":"DRAFT_WRITTEN","protocol_sha256":sha(protocol)}));return
    required=[protocol,*sorted((ROOT/"source").glob("*.py")),*sorted((ROOT/"tests").glob("*.json")),ROOT/"analysis/finite_feature_preflight35.json"]
    required += sorted(p for p in (ROOT/"inputs").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    record={"schema":"step36-protocol-source-seal-v1","sealed_utc":datetime.now(timezone.utc).isoformat(),
            "programmatic_public_measurements_parsed_before_seal":False,"public_model_fits_before_seal":0,
            "incidental_training_snippet_disclosed":True,"decision_test_measurements_seen_before_seal":False,
            "bound_files":{str(path.relative_to(ROOT)):sha(path) for path in required}}
    write(seal,record);print(json.dumps({"status":"SEALED","files":len(required),"seal_sha256":sha(seal),"protocol_sha256":sha(protocol)}))


if __name__=="__main__":main()
