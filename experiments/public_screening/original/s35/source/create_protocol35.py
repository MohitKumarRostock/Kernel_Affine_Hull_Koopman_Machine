"""Write/review the Step35 protocol, or seal it before the sole primary run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from data35 import ROOT, sha_file, write_json
from representations35 import DEFAULT_CONFIG


def draft():
    return {
        "schema": "step35-public-forecasting-protocol-v1",
        "step": 35,
        "title": "Public forecasting pilot of finite-cohort class certificates and safe decoder screening",
        "primary_question": "Does exact class screening preserve the exhaustive tolerance/cost choice while avoiding enough decoder work to offset its certification overhead?",
        "status_at_drafting": "BEFORE_PUBLIC_NUMERIC_DATA_PARSING_AND_ALL_PUBLIC_MODEL_FITS",
        "scientific_scope": {
            "assurance": "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES",
            "stored_features": "Operational binary64 coordinates interpreted as exact dyadic numbers",
            "native_transform_error_enclosed": False,
            "native_pipeline_roundoff_certified": False,
            "population_risk_claim": False,
            "future_response_coverage_claim": False,
            "data_newness_claim": "New to the retained study record; UCI Appliances was already evaluated in Step24",
            "manufactured_fixture_only": False,
            "public_dataset_values_read_in_primary": True,
            "new_public_scientific_fits_authorized": True,
            "reserved_patient_arrays_read": False,
            "reserved_patients_total": 5557,
            "reserved_patients_accessed": 0,
            "inherited_study_statuses": {"G3": "NOT_ESTABLISHED", "G4": "NOT_STARTED", "natural_study": "NO_GO"},
        },
        "data": {
            "dataset": "UCI Beijing PM2.5", "uci_id": 381, "doi": "10.24432/C5JS49",
            "source_receipt": "inputs/public_source35.json", "license": "CC BY 4.0",
            "archive": "inputs/beijing_pm25_uci381.zip", "expected_hourly_rows": 43824,
            "series": "pm2.5", "other_measured_variables_used": False,
            "origin_hour": 23, "horizon_hours": 24, "lag_hours": 24,
            "input_order": "oldest to newest; origin minus23hours through origin inclusive",
            "time_basis": "Dataset-recorded calendar timestamps, without timezone conversion",
            "partition_basis": "Target timestamp calendar year",
            "train_target_years": [2010, 2011, 2012], "decision_target_years": [2013], "test_target_years": [2014],
            "eligibility": "Discard origin if any of its24 hourly PM lags or its target is absent or NA; no imputation",
            "common_roster_for_all_models": True,
            "embargo_hours": 0,
            "rolling_observed_history": True,
            "boundary_history": "Past observations from preceding years may be used; test histories may include earlier test-year observations available at each origin",
            "normalization": "Exact (PM-training_target_min)/(training_target_max-training_target_min)",
            "degenerate_training_range_action": "STOP_AND_RETAIN_FAILURE",
            "targets_clipped": False,
            "test_seal": "2014 PM values and missingness are uninterpreted until all models, thresholds, policies and deployment payloads are sealed",
        },
        "representations": DEFAULT_CONFIG,
        "seed": 3501,
        "class": {
            "metric": "Euclidean metric on stored training-normalized features",
            "kappa": "2", "prediction_interval": ["0", "1"],
            "metric_scale_fit": "Training-only reciprocal median positive Euclidean distance over32 temporal-stratum probes, fallback1",
            "class_definition": "All [0,1]-valued 2-Lipschitz decoders of each frozen representation",
            "range_only_floor": "mean squared distance of unchanged targets to [0,1]; combine with dual lower by max, never addition",
        },
        "heads": {
            "recipes_per_map": ["affine", "extension32"],
            "affine": {"objective": "mean squared error plus lambda*coefficient squared norm; unpenalized intercept",
                       "ridge_lambda": "1/1000000", "fit_rows": "all eligible training rows",
                       "solver": "numpy.linalg.solve on centered normal equations with n*lambda diagonal",
                       "membership": "Exact radial coefficient contraction to norm<=2, using80bit upward norm enclosure; intercept recentered at exact stored-feature training means",
                       "prediction": "Exact rational affine formula followed by clipping to [0,1]"},
            "extension32": {"anchor_count": 32,
                            "anchor_rule": "floor((2*j+1)*n_train/(2*32)) for j=0,...,31; training-label-independent and shared across maps",
                            "fit": "Conservative pairwise Lipschitz QP with distance lower endpoints and range-aware dual",
                            "prediction": "Clipped mean of McShane lower/upper envelopes; exact outward80bit distance arithmetic"},
            "training_labels_only": True,
            "incumbent": "kahm_minmax01_cap20_p2::affine",
        },
        "certificate": {
            "distance_bits": 80, "multiplier_grid_bits": 40, "check_every": 20,
            "gap_sse": "1/10000000", "head_max_coordinate_attempts": 4000,
            "class_max_coordinate_attempts_per_block": 400,
            "block_size": 32, "block_rule": "Consecutive disjoint eligible rows in chronological order, last block may be smaller",
            "decision_and_test_classes": "All18 frozen representation classes",
            "lower_aggregation": "Sum retained disjoint-block dual SSE lower bounds divided by total cohort rows",
            "upper_aggregation": "Minimum certified loss upper endpoint of the two actual common frozen training-only heads on the entire cohort",
            "average_block_primal_upper_permitted": False,
            "unfinished": "Retain valid final primal/dual witnesses and report unresolved gap; no favorable replacement or outcome-driven refit",
            "final_witness_independent_audit": True,
        },
        "policy": {
            "target": "Finite-panel tolerance/cost selection on the decision cohort",
            "epsilon_recipe": "19/20 times exact normalized training persistence MSE",
            "epsilon_factor": "19/20",
            "epsilon_is_population_or_clinical_tolerance": False,
            "screen": "Exclude class iff its decision-risk lower bound is strictly greater than epsilon",
            "fit_order": "All class screens first; all unexcluded heads next; freeze screened choice; shadow-fit excluded heads for exhaustive comparison; seal every model and choice before test parsing",
            "eligibility": "Achieved decision MSE upper endpoint <= epsilon",
            "cost": "Byte length of full independently serialized representation pickle plus canonical deployed-head JSON, including exact coefficients/anchors and metadata but excluding audit witnesses, shared code/runtime and fit data",
            "tie_breaking": ["deployment_payload_bytes ascending", "decision_risk_upper ascending", "candidate_id lexical ascending"],
            "abstention": "No candidate in the finite panel has a verified achieved upper <=epsilon",
            "control_same_rule": "Exhaustive same36-head threshold/cost selector with no class screening",
            "control_accuracy": "Ordinary minimum decision MSE upper endpoint among all36 heads; ties payload then lexicalID",
            "actions": {"KEEP": "selected incumbent", "REPAIR_DECODER": "selected extension32 on incumbent representation", "CHANGE_REPRESENTATION": "selected another frozen representation", "ABSTAIN": "no achieved candidate"},
            "soundness_goal": "Screened and exhaustive same-rule candidate IDs or abstentions must agree",
            "utility_goal": "Positive avoided decoder fit-plus-decision-evaluation time minus added decision-class certification time, with strict stage-accounting qualification",
            "no_test_selection": True,
            "policy_not_a_population_generalization_certificate": True,
        },
        "baselines": {
            "persistence": {"forecast": "Most recent observed PM value", "clipped": False},
            "training_mean": {"forecast": "Exact mean of eligible training targets", "clipped": False},
            "hist_gradient_boosting": {
                "fit_rows": "All eligible training inputs/labels; MinMax01 scaling fitted on training inputs only",
                "loss": "squared_error", "learning_rate": 0.05, "max_iter": 200,
                "max_leaf_nodes": 15, "max_depth": None, "min_samples_leaf": 20,
                "l2_regularization": 1.0, "max_features": 1.0, "max_bins": 255,
                "categorical_features": None, "monotonic_cst": None, "interaction_cst": None,
                "warm_start": False, "early_stopping": False, "random_state": 3501,
                "predictions_clipped_to_I": False, "class_membership_certified": False,
            },
            "baseline_in_threshold_cost_panel": False,
            "baseline_prediction_scope": "Squared losses of stored baseline predictions evaluated exactly; no claim of enclosing ideal-real training algorithms",
        },
        "execution": {
            "primary_run_count": 1, "overall_wall_budget_seconds": 3600,
            "native_family_wall_budget_seconds": 1800,
            "deadline_action": "Stop and preserve partial results/failure; do not silently rerun the primary",
            "threads": 1, "native_regrouped_controls_counted_in_physical_work": True,
            "stage_times_are_conditional_not_separately_run_pipeline_times": True,
            "artificial_preflights_allowed_before_freeze": True,
            "no_public_numeric_data_in_preflights": True,
            "all_frozen_panel_outcomes_reported": True,
        },
        "report": {
            "primary_metrics": ["Same-rule selection identity", "Classes excluded", "Decoder fits avoided", "Certification overhead", "Net conditional stage time saved", "Frozen selections'2014 MSE/RMSE", "Full deployment payload bytes"],
            "diagnostics": ["All36 head risks", "Per-class finite-cohort lower/upper bounds", "Range-only floor", "Excess-risk brackets", "Native requested/realized ranks", "Scaling and dimension sensitivity", "Missing-data exclusions", "Optimizer unfinished counts"],
            "uncertainty": "Exact arithmetic enclosures quantify numerical/class-witness uncertainty only; no iid confidence interval is asserted for dependent time-series rows",
            "single_dataset_pilot": True,
            "significance_claim_condition": "Do not claim actionable screening utility if no classes are excluded or added certificate cost exceeds avoided decoder work",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "pandas"]}},
        "prior": {
            "step34_protocol_sha256": "2daa99df814a36711b5f1fe1292818b0b6c5e1eec2bfcb7dc8c7d28dfc213829",
            "step34_result_sha256": "2c33826c813822896fff2524f3e8a19721e1a0f24d6bf4b239b9db03dd714bb8",
            "step34_package_sha256": "91628a2bfab6cfc8d0dc8d78b2cc1977328bf2e5beb1108435410296ec24fd61",
            "step34_conclusion": "Manufactured feature-covering anchor repair gained at the boundary but harmed the interior; overall benefit claim excluded",
            "replay_scope": "Step35 imports only pinned generic native source, not previous scientific arrays; prior complete proof archives are referenced rather than recursively copied",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    protocol_path = ROOT / "protocol35.json"
    seal_path = ROOT / "protocol_seal35.json"
    if seal_path.exists():
        raise RuntimeError("Protocol already sealed; refuse to replace it")
    if not args.seal:
        write_json(protocol_path, draft())
        print(json.dumps({"status": "DRAFT_WRITTEN", "sha256": sha_file(protocol_path)}))
        return
    if not protocol_path.exists():
        raise RuntimeError("Review the draft before sealing")
    paths = [protocol_path, ROOT / "inputs/public_source35.json", ROOT / "inputs/beijing_pm25_uci381.zip",
             ROOT / "inputs/native_source_receipt35.json"]
    paths += sorted((ROOT / "source").glob("*.py"))
    paths += sorted(p for p in (ROOT / "inputs/native").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths += sorted((ROOT / "tests").glob("*.json"))
    paths += [ROOT / "analysis/finite_feature_preflight35.json"]
    seal = {"schema": "step35-protocol-source-seal-v1", "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "public_numeric_values_parsed_before_seal": False,
            "public_model_fits_before_seal": 0,
            "bound_files": {str(p.relative_to(ROOT)): sha_file(p) for p in paths}}
    write_json(seal_path, seal)
    print(json.dumps({"status": "SEALED", "files": len(paths), "protocol_sha256": sha_file(protocol_path),
                      "seal_sha256": sha_file(seal_path)}))


if __name__ == "__main__":
    main()
