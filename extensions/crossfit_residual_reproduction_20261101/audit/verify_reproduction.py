#!/usr/bin/env python3
"""Independent audit of the separately declared computational reproduction.

This verifier requires the fresh source, fit, forecast and target-release
seals and the unchanged scientific analysis. It verifies equivalence to the
pinned original scientific source and same declared simulation cohort.
It does not authenticate the unavailable original completed-run records.
All 42 terminal arrays per task receive independent loss replay; selected
states receive coefficient and corrected native-PCA checks, with all saved
B candidate scores and correction alpha paths checked as before.
"""
from __future__ import annotations

import os
for _thread in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
SCALERS = ("standard", "minmax", "tanh_z_1")
CAPS = (1, 2, 4, 8, 12)
LEGACY_CAPS = (*CAPS, 20)
PAIRS = ("standard", "minmax", "tanh", "hybrid_tanh_minmax", "hybrid_minmax_tanh")
GEOMETRY = ("common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
LEGACY_METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", *GEOMETRY, "nonkahm_policy")
NONKAHM = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
CORRECTIONS = tuple(f"{mode}_{kind}" for mode in ("crossfit", "insample")
                    for kind in ("kahm", "rbf", "matched_rbf"))
NEW_METHODS = ("common", "shared_bias", "half_common", "half_shared_bias", *CORRECTIONS)
POLICY = tuple("legacy__" + x for x in NONKAHM) + (
    "crossfit__shared_bias", "crossfit__crossfit_rbf", "crossfit__insample_rbf")
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
PANELS = {"nonlinear": LAWS[1:], "linear": LAWS[:1], "all_descriptive": LAWS}
HORIZONS = (1, 4, 12)
PRIMARY = (
    ("crossfit_KAHM_vs_shared_bias", "nonlinear", "crossfit__calibrated__shared_bias", .97),
    ("crossfit_KAHM_vs_independent_RBF", "nonlinear", "crossfit__calibrated__crossfit_rbf", .97),
    ("crossfit_KAHM_vs_expanded_nonKAHM", "nonlinear", "policy__calibrated__expanded_nonkahm", .97),
    ("linear_crossfit_KAHM_vs_ridge", "linear", "legacy__calibrated__ridge", 1.05),
)
SOLVER_POLICY = "numpy-first-gesvd-on-linalgerror-v1"
SOLVER_TOLERANCES = {"relative_reconstruction_error": 1e-10,
                     "left_orthogonality_frobenius": 1e-8,
                     "right_orthogonality_frobenius": 1e-8,
                     "relative_normal_equation_residual": 1e-8}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.pca_cache = {}
        self.ranks = Counter()

    def require(self, valid, label):
        self.counts["assertions"] += 1
        if not bool(valid):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        self.require(np.array_equal(actual, expected), label)
        self.counts["exact_array_comparisons"] += 1

    def close(self, actual, expected, label, rtol=2e-9, atol=2e-10):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all()
                     and np.allclose(a, e, rtol=rtol, atol=atol), label)
        self.counts["numerical_values_compared"] += a.size


def forecast_keys():
    keys = {f"legacy__{mode}__{method}" for mode in ("calibrated", "gamma1") for method in LEGACY_METHODS}
    keys |= {f"crossfit__{mode}__{method}" for mode in ("calibrated", "eta1") for method in NEW_METHODS}
    keys |= {f"policy__{mode}__expanded_nonkahm" for mode in ("calibrated", "eta1")}
    return keys


def blend(base, raw, multiplier):
    out = base.copy()
    for h, value in enumerate(multiplier):
        if value != 0:
            out[:, h] = base[:, h] + value * (raw[:, h] - base[:, h])
    return out


def policy_key(identity, mode):
    family, method = identity.split("__", 1)
    actual = "calibrated" if mode == "calibrated" else ("gamma1" if family == "legacy" else "eta1")
    return f"{family}__{actual}__{method}"


def alpha_record(record, grid, audit, label):
    audit.exact([r["alpha"] for r in record["candidates"]], grid, label + " alpha grid")
    losses = np.asarray([r["mse_by_horizon"] for r in record["candidates"]])
    indices = np.argmin(losses, axis=0)
    audit.exact([r["candidate_index"] for r in record["selected"]], indices, label + " alpha first argmin")
    audit.exact([r["alpha"] for r in record["selected"]], np.asarray(grid)[indices], label + " selected alpha")
    audit.close([r["mse"] for r in record["selected"]], losses[indices, np.arange(3)], label + " selected alpha MSE")
    return np.asarray(grid)[indices]


def reproduction_provenance(root, source_root, cfg, freeze, abc_manifest, audit):
    """Authenticate this new execution against the pinned scientific source.

    This check does not authenticate unavailable original result bytes or
    original execution chronology. The declared cohort and its seeds are reused.
    """
    metadata = cfg["computational_reproduction"]
    audit.require("numerical_amendment" not in cfg,
                  "New execution does not claim historical exact-byte ABC reuse")
    audit.require(metadata["schema"] == "same-seed-computational-reproduction-v1"
                  and metadata["original_extension"] == "crossfit_residual_v2"
                  and metadata["original_source_commit"] == "f36e7c797f37d7d073dae42603b3770543d9fb43"
                  and metadata["source_commit"] == metadata["original_source_commit"],
                  "Pinned scientific source and explicitly separate reproduction")
    audit.require(metadata["same_cohort_not_independent"] is True
                  and metadata["historical_outputs_known_before_reproduction"] is True
                  and metadata["original_completed_run_authenticated"] is False
                  and metadata["additional_independent_repetitions"] == 0,
                  "No fresh-confirmation or original-execution authentication claim")
    audit.require(metadata["scientific_config_comparison"] ==
                  "remove numerical_amendment and computational_reproduction only",
                  "Only declared provenance fields may differ")
    original_root = source_root.parent / "crossfit_residual_v2"
    original_freeze_hash = "b118e88a1c788fac3a268682736d01c5a9219146f0db7bda1929159c059d3210"
    original_config_hash = "ecdf1dfa5a8b557967f0b0a1a1cb7b77bbd4aed67336ba6aae1f439c7685dc5e"
    audit.require(metadata["original_freeze_sha256"] == original_freeze_hash
                  == sha(original_root / "freeze.json"), "Pinned original source freeze")
    audit.require(metadata["original_config_sha256"] == original_config_hash
                  == sha(original_root / "config.json"), "Pinned original scientific configuration")
    original_freeze = read(original_root / "freeze.json")
    original_config = read(original_root / "config.json")
    for relative, expected in original_freeze["files"].items():
        audit.require(sha(original_root / relative) == expected,
                      "Original frozen source retained: " + relative)
    old_scientific, new_scientific = dict(original_config), dict(cfg)
    for value in (old_scientific, new_scientific):
        value.pop("numerical_amendment", None)
        value.pop("computational_reproduction", None)
    audit.require(old_scientific == new_scientific,
                  "Every scientific configuration field is unchanged")
    audit.require(metadata["original_numerical_amendment"] == original_config["numerical_amendment"],
                  "Historical numerical amendment retained as provenance only")
    required = {"code/crossfit_model.py", "code/pipeline.py", "code/numerical_solver.py",
                "analysis_code/analyze.py", "analysis_code/ANALYSIS_PROTOCOL.md",
                "audit/verify_crossfit.py", "audit/verify_pipeline.py", "requirements.txt",
                "readiness/prior_seed_exclusions.json"}
    files = metadata["original_source_files"]
    audit.require(set(files) == required, "Complete declared unchanged scientific-source roster")
    for relative, expected in files.items():
        audit.require(original_freeze["files"][relative] == expected
                      == sha(original_root / relative) == sha(source_root / relative),
                      "Unchanged original and reproduction source: " + relative)
        audit.require(freeze["files"][relative] == expected,
                      "Unchanged scientific source bound by new freeze: " + relative)
    for field in ("inherited_files", "previous_extension_files"):
        audit.require(freeze[field] == original_freeze[field],
                      "Every inherited source hash matches original freeze: " + field)
    documents = metadata["historical_documents"]
    audit.require(set(documents) == {"audit/original_protocol.md", "audit/original_numerical_amendment.md"},
                  "Complete historical-protocol record")
    for relative, record in documents.items():
        audit.require(record["original_path"] in ("PROTOCOL.md", "NUMERICAL_AMENDMENT.md")
                      and original_freeze["files"][record["original_path"]] == record["sha256"]
                      == sha(source_root / relative) == freeze["files"][relative],
                      "Historical protocol bytes preserved: " + relative)
    audit.require(metadata["original_source_manifest"] == "ORIGINAL_SOURCE.json",
                  "Fixed source-provenance manifest path")
    audit.require(sha(source_root / "ORIGINAL_SOURCE.json") == metadata["original_source_manifest_sha256"],
                  "Source-provenance manifest bound by configuration")
    manifest = read(source_root / "ORIGINAL_SOURCE.json")
    for field in ("original_source_commit", "original_extension", "original_freeze_sha256",
                  "original_config_sha256", "original_repository", "original_source_files",
                  "historical_documents"):
        audit.require(manifest[field] == metadata[field], "Source manifest agrees with configuration: " + field)
    audit.require(freeze["data_generated"] is False and freeze["D_generated"] is False
                  and freeze["numerical_amendment"] is None and freeze["public_preregistration"] is False,
                  "New source freeze explicitly precedes this reproduction's data generation")
    audit.require(freeze["files"]["audit/verify_reproduction.py"] == sha(__file__),
                  "Executing reproduction verifier is included in new source freeze")
    audit.require(abc_manifest["schema"] == "crossfit-residual-abc"
                  and abc_manifest["D_generated"] is False and "reuse" not in abc_manifest,
                  "New ABC preparation does not masquerade as restored original bytes")
    pipeline = sys.modules["crossfit_production_audit_pipeline"]
    expected_seeds = pipeline.seed_records(original_config)
    current_seeds = pipeline.seed_records(cfg)
    seed_manifest = read(root / "prepared/seed_manifest.json")
    audit.require(current_seeds == expected_seeds == seed_manifest["records"]
                  and seed_manifest["seed_count"] == len(expected_seeds) == 1272
                  and seed_manifest["D_streams_reserved_but_not_generated"] is True,
                  "Exactly the original declared 1272 seed streams; no new cohort")
    chronology = [("source_metadata", metadata["created_utc"]),
                  ("source_freeze", freeze["created_utc"]),
                  ("ABC_preparation", abc_manifest["created_utc"]),
                  ("fit_seal", read(root / "FIT_LOCK.json")["created_utc"]),
                  ("D_preparation", read(root / "prepared/d_manifest.json")["created_utc"]),
                  ("prediction_seal", read(root / "PREDICTION_LOCK.json")["created_utc"]),
                  ("target_release", read(root / "TARGET_RELEASE.json")["created_utc"])]
    moments = [datetime.fromisoformat(value) for _, value in chronology]
    audit.require(all(value.tzinfo is not None for value in moments)
                  and all(left <= right for left, right in zip(moments, moments[1:])),
                  "Fresh reproduction source, data, fit, forecast and release chronology")
    audit.counts["original_frozen_source_files_checked"] = len(original_freeze["files"])
    audit.counts["reproduction_identical_scientific_source_files_checked"] = len(files)
    return {"original_source_commit": metadata["original_source_commit"],
            "original_freeze_sha256": original_freeze_hash,
            "original_config_sha256": original_config_hash,
            "scientific_configuration_and_source_identical": True,
            "same_declared_seed_streams_verified": True,
            "additional_independent_repetitions": 0,
            "historical_outputs_known_before_reproduction": True,
            "original_completed_run_authenticated": False,
            "original_array_byte_identity_claimed": False,
            "original_execution_chronology_authenticated": False,
            "new_execution_chronology": dict(chronology),
            "chronology_scope": "Procedural timestamps and sealed artifacts from the new reproduction only",
            "source_manifest_sha256": sha(source_root / "ORIGINAL_SOURCE.json")}


def solver_events(folder, fit, source_root, freeze, cfg, audit, contexts):
    ledger = read(folder / "numerical_solver_events.json")
    installation = ledger["installation"]
    expected_sha = freeze["files"]["code/numerical_solver.py"]
    audit.require(installation["policy_id"] == SOLVER_POLICY
                  and installation["scope"] == "runtime._ridge_path only"
                  and installation["adapter_sha256"] == expected_sha == sha(source_root / "code/numerical_solver.py"),
                  "Uniform frozen numerical adapter provenance")
    audit.require(installation["tolerances"] == SOLVER_TOLERANCES, "Frozen numerical validity gates")
    inherited = source_root.parent / "dimension_forecasting"
    relative = "runtime/step1/forecasting_corrected/code/vendor_geometry/kahm_geometry/extensions/reconstruction_mechanism/models.py"
    # Producer absolute paths are provenance, not verifier filesystem locations.
    # Resolve only the trusted inherited module in the current checkout.
    local_runtime_sha = sha(inherited / relative)
    for file_key, hash_key in (("runtime_file", "runtime_sha256"),
                               ("original_function_file", "original_function_file_sha256")):
        audit.require(installation[file_key].replace("\\", "/").endswith("/" + relative),
                      "Recorded original ridge primitive has the trusted inherited module suffix")
        audit.require(freeze["inherited_files"][relative] == installation[hash_key] == local_runtime_sha,
                      "Original ridge primitive source hash")
    audit.require(fit["numerical_amendment"]["installation"] == installation
                  and fit["numerical_amendment"]["all_fallbacks_recovered"] is True,
                  "Completed-task numerical installation identity")
    events = ledger["events"]
    audit.require(ledger["fallback_count"] == len(events) == fit["numerical_amendment"]["fallback_count"],
                  "Fallback ledger count")
    recovered = sum(e["status"] == "recovered" and e["recovered"] is True for e in events)
    audit.require(ledger["recovered_count"] == recovered == len(events) and ledger["failed_count"] == 0,
                  "Every completed-run fallback recovered; no hidden failed attempt")
    for index, event in enumerate(events, 1):
        audit.require(event["event_index"] == index and event["policy_id"] == SOLVER_POLICY
                      and event["schema"] == "crossfit-ridge-svd-fallback-event-v1", "Recovery event identity/order")
        audit.require(event["provenance"] == installation and event["tolerances"] == SOLVER_TOLERANCES,
                      "Recovery event source and gate provenance")
        audit.require(event["solver_order"] == ["original_numpy_svd", "scipy_gesvd"]
                      and event["triggering_exception"]["type"] == "LinAlgError"
                      and event["fallback_exception"] is None, "Narrow fixed solver transition")
        audit.exact(event["alphas"], cfg["model"]["alpha_grid"], "Recovered original penalty grid")
        audit.require(event["features_shape"][0] == event["targets_shape"][0]
                      and event["features_dtype"] == event["targets_dtype"] == "float64", "Finite-design recovery allocation")
        for key in ("centered_array_sha256", "response_array_sha256"):
            audit.require(len(event[key]) == 64 and all(c in "0123456789abcdef" for c in event[key]), "Recovery array hash format")
        checks = event["checks"]
        for key in ("features_finite", "targets_finite", "feature_mean_finite", "target_mean_finite",
                    "centered_finite", "response_finite", "u_finite", "singular_values_finite", "vt_finite",
                    "factor_shapes_valid", "singular_values_nonnegative_and_descending", "projected_response_finite",
                    "response_rms_finite", "normal_equation_rhs_finite"):
            audit.require(checks[key] is True, "Recovery finite/shape check " + key)
        for key in ("relative_reconstruction_error", "left_orthogonality_frobenius", "right_orthogonality_frobenius"):
            audit.require(math.isfinite(checks[key]) and 0 <= checks[key] <= SOLVER_TOLERANCES[key], "Recovery factorization gate " + key)
        audit.exact([r["alpha"] for r in event["ridge_checks"]], event["alphas"], "Recovery per-penalty check coverage")
        for record in event["ridge_checks"]:
            value = record["relative_normal_equation_residual"]
            audit.require(record["coefficient_finite"] is True and record["normal_equation_residual_finite"] is True
                          and math.isfinite(value) and 0 <= value <= SOLVER_TOLERANCES["relative_normal_equation_residual"],
                          "Recovery ridge optimality gate")
        context = event["context"].get("ridge_prefix")
        if context is None:
            frame = next((r for r in event["context"]["frames"] if "mode" in r), {})
            context = "direct_ridge/" + str(frame.get("mode", "unspecified")) + "/" + str(frame.get("kind", "unspecified"))
        contexts[context] += 1
    audit.counts["numerical_solver_installed_tasks"] += 1
    audit.counts["numerical_fallback_attempts"] += len(events)
    audit.counts["numerical_fallback_recoveries"] += recovered
    audit.counts["numerical_fallback_failures_in_completed_run"] += ledger["failed_count"]


def scaler_metadata(tx, key, record, audit):
    arm = key.rsplit("__cap", 1)[0]
    mean, sd = tx.mean(axis=0), tx.std(axis=0, ddof=0)
    audit.exact(record["standard_mean"], mean, "Common-bank subset-only mean")
    audit.exact(record["standard_scale"], np.where(sd > 1e-12, sd, 1.0), "Common-bank subset-only scale")
    g = record["geometry_scaler"]
    if arm == "standard":
        audit.require(g is None, "Standard native history representation")
    elif arm == "minmax":
        audit.exact(g["minimum"], tx.min(axis=0), "Subset min-max minimum")
        audit.exact(g["maximum"], tx.max(axis=0), "Subset min-max maximum")
        audit.exact(g["range"], np.ptp(tx, axis=0), "Subset min-max range")
        audit.require(g["fit_rows"] == len(tx) and g["clip"] is False, "Subset min-max scope")
    else:
        audit.exact(g["mean"], mean, "Subset tanh location")
        audit.exact(g["raw_standard_deviation"], sd, "Subset tanh scale")
        audit.require(g["fit_rows"] == len(tx) and g["post_tanh_rescale"] is False
                      and g["multiplier"] == 1.0, "Subset tanh scope")
    bad = sum(bool(r.get("factor_error")) for r in record["bank"]["local_models"])
    audit.require(bad == 0, "Unexpected common-bank fitted factor failure")
    audit.counts["new_common_candidate_scaler_metadata_banks"] += 1
    audit.counts["new_common_fitted_factor_failures"] += bad


def development(abc, fit, pre, training, validation, cfg, audit):
    ax, ay, bx, by, groups = [abc[k] for k in ("A_X", "A_Y", "B_X", "B_Y", "A_groups")]
    scale = np.maximum(ay.var(axis=0, ddof=0), cfg["model"]["target_variance_floor"])
    alphas = cfg["model"]["alpha_grid"]
    audit.exact(abc["loss_scale"], scale, "Full-A loss variance")
    audit.require(ax.shape == (512, 12) and ay.shape == (512, 3) and bx.shape == (256, 12)
                  and by.shape == (256, 3), "Production fitting-array shapes")
    audit.exact(np.unique(groups), np.arange(64), "All original A episode identities")
    legacy, cross = fit["legacy"], fit["crossfit"]
    selected, diag = cross["selections"], cross["diagnostics"]
    audit.require(pre["C_loaded"] is False and pre["crossfit"] == selected, "Saved B decisions precede C")
    audit.require(pre["legacy"] == {k: legacy[k] for k in ("selections", "b_scores", "raw_rbf_selected_scaler")}, "Legacy B decisions unchanged by C")
    audit.require(pre["crossfit_frozen_state_sha256"] == cross["calibration"]["pre_C_hash"]
                  == cross["calibration"]["post_C_hash"], "Same-process cross-fit C state invariant")
    audit.require(legacy["calibration_records"]["pre_C_hash"] == legacy["calibration_records"]["post_C_hash"], "Same-process legacy C state invariant")
    actual = {k: array_hash(abc[k]) for k in ("A_X", "A_Y", "B_X", "B_Y")}
    audit.require(actual == legacy["diagnostics"]["input_hashes"] == diag["input_hashes"], "A/B input hashes")
    audit.require(diag["groups_A_sha256"] == array_hash(groups), "A group hash")
    order = [f"{pair}__cap{cap}" for cap in LEGACY_CAPS for pair in PAIRS]
    audit.require(set(legacy["b_scores"]) == set(order), "Complete legacy B candidate roster")
    for key in order:
        for method in GEOMETRY:
            pred = validation[f"legacy__{key}__{method}"]
            mse = np.mean((by - pred) ** 2, axis=0)
            audit.close(legacy["b_scores"][key][method], np.mean(mse / scale), "Legacy B score join")
            alpha_record(legacy["candidates"][key]["B"][method], alphas, audit, key + "/" + method)
            audit.counts["legacy_B_candidate_scores"] += 1
    for context, methods in legacy["selections"].items():
        for method, choice in methods.items():
            candidates = order
            if context in ("cap2", "cap20"):
                candidates = [k for k in candidates if k.endswith("__" + context)]
            elif context in PAIRS:
                candidates = [k for k in candidates if k.startswith(context + "__")]
            if method == "common" and context in ("selected", "cap2", "cap20"):
                candidates = [k for k in candidates if k.rsplit("__cap", 1)[0] in PAIRS[:3]]
            audit.require(choice == min(candidates, key=lambda k: legacy["b_scores"][k][method]), "Legacy B first-minimum context choice")
            audit.counts["legacy_B_context_choices"] += 1
    raw_scores = {}
    for scaler in SCALERS:
        record = legacy["diagnostics"]["raw_rbf"][scaler]
        losses = np.asarray([r["mse_by_horizon"] for r in record["candidates"]])
        indices = losses.argmin(axis=0)
        audit.exact([r["candidate_index"] for r in record["selected"]], indices, "Raw-RBF B penalty/bandwidth choice")
        raw_scores[scaler] = float(np.mean(losses[indices, np.arange(3)] / scale))
        audit.close(legacy["diagnostics"]["raw_rbf_scores"][scaler], raw_scores[scaler], "Raw-RBF scaler score")
    audit.require(legacy["raw_rbf_selected_scaler"] == min(SCALERS, key=raw_scores.__getitem__), "Raw-RBF scaler first argmin")
    audit.counts["raw_RBF_scaler_choices"] += 1
    for name, expected in (("A_groups", groups), ("A_fold_id", groups % 4),
                           ("A_oof_assignment_count", np.ones(len(groups), dtype=int))):
        audit.exact(training[name], expected, "Training array " + name)
    audit.exact(training["A_oof_residual"], ay - training["A_oof_common"], "OOF residual identity")
    audit.exact(training["A_insample_residual"], ay - training["A_full_common"], "In-sample residual identity")
    full = selected["full_common"]
    audit.require(full["selected_candidate"] == legacy["selections"]["selected"]["common"]
                  and full["common_gamma"] == 1 and full["selected_cap"] in CAPS, "Gamma-one full common identity")
    audit.exact(full["training_target_variance"], scale, "Full common target normalizer")
    audit.exact(training["B_common"], validation[f"legacy__{full['selected_candidate']}__common"], "Full common B forecast")
    audit.exact(selected["A_target_variance"], scale, "Departure target normalizer")
    bias = np.mean(by - training["B_common"], axis=0)
    half_bias = np.mean(by - training["B_half_common"], axis=0)
    audit.exact(selected["shared_bias"]["value"], bias, "Shared B mean bias")
    audit.exact(selected["half_common"]["shared_bias"], half_bias, "Half common B mean bias")
    audit.exact(training["B_shared_bias"], training["B_common"] + bias, "Shared-bias B forecast")
    audit.exact(training["B_half_shared_bias"], training["B_half_common"] + half_bias, "Half shared-bias B forecast")
    common_order = [f"{s}__cap{c}" for c in CAPS for s in SCALERS]
    fits = [(f"fold{k}", selected["folds"][k], np.flatnonzero(groups % 4 != k)) for k in range(4)]
    fits.append(("half", selected["half_common"], np.flatnonzero(groups < 32)))
    for tag, record, train in fits:
        audit.exact(record["training_row_indices"], train, tag + " training row IDs")
        audit.exact(record["training_episode_ids"], np.unique(groups[train]), tag + " training episodes")
        variance = np.maximum(ay[train].var(axis=0), cfg["model"]["target_variance_floor"])
        audit.exact(record["training_target_variance"], variance, tag + " complement-only normalizer")
        audit.require(record["common_gamma"] == 1 and record["candidate_order"] == common_order, tag + " common candidate order")
        if tag.startswith("fold"):
            k = int(tag[-1]); held = np.flatnonzero(groups % 4 == k)
            audit.exact(record["heldout_row_indices"], held, tag + " held-out rows")
            audit.exact(record["heldout_episode_ids"], np.unique(groups[held]), tag + " held-out episodes")
            audit.require(len(np.unique(groups[train])) == 48 and len(np.unique(groups[held])) == 16
                          and not set(groups[train]) & set(groups[held]), tag + " episode exclusion")
            audit.require(record["heldout_targets_accepted_by_fit"] is False
                          and record["state_before_OOF"] == record["state_after_OOF"], tag + " OOF fitting/query roles")
            for key, value in (("training_group_sha256", groups[train]), ("heldout_group_sha256", groups[held]),
                               ("training_rows_sha256", train), ("heldout_rows_sha256", held),
                               ("heldout_history_sha256", ax[held]), ("OOF_prediction_sha256", training["A_oof_common"][held])):
                audit.require(record[key] == array_hash(value), tag + " row and prediction hashes")
            audit.counts["whole_episode_folds_checked"] += 1
        scores = []
        for candidate in record["B_scores"]:
            key = candidate["candidate_id"]
            pred = validation[f"crossfit__{tag}__{key}__common"]
            mse = np.mean((by - pred) ** 2, axis=0)
            score = float(np.mean(mse / variance))
            audit.close(candidate["mse_by_horizon"], mse, tag + " common B MSE")
            audit.close(candidate["score"], score, tag + " common B score")
            audit.require(candidate["B_prediction_sha256"] == array_hash(pred), tag + " common B hash")
            chosen_alpha = alpha_record(candidate["head_selection"], alphas, audit, tag + "/" + key)
            audit.exact(candidate["selected_alpha_by_horizon"], chosen_alpha, tag + " common penalty")
            scaler_metadata(ax[train], key, diag["common_fits"][tag]["fit"]["banks"][key], audit)
            scores.append(score)
            audit.counts["new_common_B_candidate_scores"] += 1
        audit.require(record["selected_candidate"] == common_order[int(np.argmin(scores))], tag + " common B first argmin")
    for mode in ("crossfit", "insample"):
        residual = training["A_oof_residual" if mode == "crossfit" else "A_insample_residual"]
        for kind in ("kahm", "rbf"):
            record = selected["target_modes"][mode][kind]
            audit.require(record["candidate_order"] == common_order, "Correction candidate order")
            scores = []
            for candidate in record["candidates"]:
                key = "crossfit__" + candidate["state_key"]
                pred, path = validation[key], validation[key + "__alpha_paths"]
                audit.require(path.shape == (len(alphas), len(by), 3), "Complete correction alpha paths")
                losses = np.mean((by[None] - path) ** 2, axis=1)
                indices = losses.argmin(axis=0)
                audit.close(candidate["B_mse_by_alpha_horizon"], losses, "Correction alpha loss replay")
                audit.exact(candidate["selected_alpha_indices"], indices, "Correction alpha argmin")
                audit.exact(candidate["selected_alpha_by_horizon"], np.asarray(alphas)[indices], "Correction selected penalties")
                audit.close(pred, np.column_stack([path[int(i), :, h] for h, i in enumerate(indices)]), "Selected B forecast from alpha paths")
                mse = np.mean((by - pred) ** 2, axis=0); score = float(np.mean(mse / scale))
                audit.close(candidate["mse_by_horizon"], mse, "Correction B MSE")
                audit.close(candidate["score"], score, "Correction B normalized score")
                audit.require(candidate["B_prediction_sha256"] == array_hash(pred)
                              and candidate["B_alpha_prediction_sha256"] == array_hash(path)
                              and candidate["residual_target_sha256"] == array_hash(residual), "Correction B/residual hashes")
                audit.require(candidate["shared_baseline_sha256"] == diag["shared_baseline_sha256"], "All candidate heads share baseline")
                scores.append(score); audit.counts["correction_B_candidate_alpha_paths"] += 1
            audit.require(record["selected_candidate"] == common_order[int(np.argmin(scores))], "Correction geometry first argmin")
        match, kahm = [selected["target_modes"][mode][k] for k in ("matched_rbf", "kahm")]
        chosen_rbf = next(r for r in selected["target_modes"][mode]["rbf"]["candidates"] if r["candidate_id"] == kahm["selected_candidate"])
        audit.require(match["selected_candidate"] == kahm["selected_candidate"], "Matched RBF uses KAHM scaler/cap")
        audit.exact(match["selected_alpha_by_horizon"], chosen_rbf["selected_alpha_by_horizon"], "Matched RBF retains own penalties")
    audit.counts["development_tasks"] += 1


def selected_native_bank(tx, bank, records, oracle, audit, tag):
    arm = tag.rsplit("/", 1)[-1].rsplit("__cap", 1)[0]
    cap = int(tag.rsplit("__cap", 1)[-1])
    z = oracle.transformed(tx, arm)
    for local, record in zip(bank["models"], records):
        q, p = local["q"], local["p"]
        ids = np.asarray(record["reference_indices"], dtype=int)
        audit.require(np.all((ids >= 0) & (ids < len(tx))), "Selected bank training-only references")
        audit.exact(q, z[ids].astype(np.float32).astype(np.float64), "Selected bank transformed references")
        initial = min(cap, q.shape[1], len(q) - 1)
        # Native rank zero is determined before sample covariance exists.
        # In particular, one reference has N - 1 == 0 and requires no PCA.
        if initial == 0:
            expected = np.empty((q.shape[1], 0))
            audit.counts["selected_zero_admissible_rank_bases"] += 1
        else:
            cache_key = array_hash(q)
            if cache_key not in audit.pca_cache:
                centered = q.T - q.T.mean(axis=1, keepdims=True)
                values, vectors = np.linalg.eigh((1.0 / (len(q) - 1)) * (centered @ centered.T))
                order = np.argsort(-values)
                audit.pca_cache[cache_key] = vectors[:, order], int(np.count_nonzero(values[order] > np.finfo(float).eps))
                audit.counts["selected_reference_eigendecompositions"] += 1
            vectors, positive = audit.pca_cache[cache_key]
            rank = initial
            if positive == 0:
                expected = np.empty((q.shape[1], 0))
                audit.counts["selected_zero_variance_rank_bases"] += 1
            else:
                while True:
                    expected = vectors[:, :min(rank, positive)]
                    if rank <= 1 or np.ptp(expected.T @ q.T, axis=1).min() >= 1e-3:
                        break
                    rank -= 1
        audit.require(p.shape == expected.shape, "Selected actual native rank")
        audit.close(p @ p.T, expected @ expected.T, "Selected native covariance-PCA subspace", rtol=2e-8, atol=2e-8)
        dimension = local["dimension_audit"]
        audit.require(dimension["initial_admissible_rank"] == initial and dimension["effective_rank"] == p.shape[1]
                      and dimension["normal_direction_reserved"] is False, "Selected native dimension audit")
        audit.ranks[f"{arm}/cap{cap}/rank{p.shape[1]}"] += 1
        audit.counts["selected_local_bases_checked"] += 1
    audit.counts["selected_geometry_bank_occurrences_checked"] += 1


def numerical_states(bundle, abc, fit, training, validation, oracle, cfg, audit):
    suite, reference = bundle["crossfit"], bundle["legacy"]
    ax, ay, bx, by, groups = [abc[k] for k in ("A_X", "A_Y", "B_X", "B_Y", "A_groups")]
    model_cfg = cfg["model"]
    core = oracle.core
    before = core.fd.state_hash(bundle)
    for key, value in training.items():
        audit.exact(suite.training_arrays[key], value, "Stored-state/training-archive identity")
    audit.require(suite.selections == fit["crossfit"]["selections"], "Stored state selection identity")
    audit.exact(suite.bias, fit["crossfit"]["selections"]["shared_bias"]["value"], "Stored shared bias")
    model_fits = [("full", np.arange(len(ax)), suite.common)]
    model_fits += [(f"fold{k}", np.flatnonzero(groups % 4 != k), suite.fold_states[f"fold{k}"]) for k in range(4)]
    model_fits += [("half", np.flatnonzero(groups < 32), suite.half_common)]
    checked_full_banks = set()
    for tag, indices, common in model_fits:
        key = f"{common.scaler}__cap{common.cap}"
        if tag == "full":
            records = fit["legacy"]["banks"][key]["bank"]["local_models"]
            checked_full_banks.add(key)
        else:
            records = fit["crossfit"]["diagnostics"]["common_fits"][tag]["fit"]["banks"][key]["bank"]["local_models"]
        selected_native_bank(ax[indices], common.bank, records, oracle, audit, tag + "/" + key)
        oracle.common_oracle(common, ax[indices], ay[indices], bx, by, model_cfg["alpha_grid"],
                             model_cfg["feature_scale_floor"], audit, tag)
        if tag.startswith("fold"):
            held = np.flatnonzero(groups % 4 == int(tag[-1]))
            audit.exact(common.predict(ax[held])[0], training["A_oof_common"][held], "Stored OOF forecast replay")
        elif tag == "full":
            audit.exact(common.predict(ax)[0], training["A_full_common"], "Stored full-A common fitted values")
            audit.exact(common.predict(bx)[0], training["B_common"], "Stored full-A common B values")
        else:
            audit.exact(common.predict(bx)[0], training["B_half_common"], "Stored half-A common B values")
    feature_cache, checked_heads = {}, set()
    for method in CORRECTIONS:
        state = suite.departures[method]
        key = f"{state.scaler}__cap{state.cap}"
        head_key = f"{state.mode}__{key}__{state.kind}"
        audit.require(state is suite.departure_candidates[head_key], "Selected correction state alias")
        source_bank = reference.banks[key]
        audit.require(state.bank is source_bank.bank and state.feature_scaler is source_bank.scalers[state.kind], "Selected correction full-A feature sharing")
        if key not in checked_full_banks:
            selected_native_bank(ax, state.bank, fit["legacy"]["banks"][key]["bank"]["local_models"], oracle, audit, "full/" + key)
            checked_full_banks.add(key)
        if key not in feature_cache:
            feature_cache[key] = (oracle.raw_features(state.bank, ax), oracle.raw_features(state.bank, bx))
        if head_key not in checked_heads:
            ra, rb = feature_cache[key]
            fs = state.feature_scaler
            audit.exact(fs["mean"], ra[state.kind].mean(axis=0), "Selected correction full-A feature mean")
            audit.exact(fs["scale"], np.maximum(ra[state.kind].std(axis=0), model_cfg["feature_scale_floor"]), "Selected correction feature scale")
            fa, fb = [oracle.bounded_features(raw[state.kind], fs) for raw in (ra, rb)]
            residual = training["A_oof_residual" if state.mode == "crossfit" else "A_insample_residual"]
            pred, coefficients, mean, ymean = oracle.ridge_path(fa, residual, fb, model_cfg["alpha_grid"], intercept=False)
            paths = pred + training["B_shared_bias"][None]
            audit.close(paths, validation["crossfit__" + head_key + "__alpha_paths"], "Selected-head independent normal-equation paths")
            losses = np.mean((by[None] - paths) ** 2, axis=1)
            choice = losses.argmin(axis=0)
            audit.exact(state.head["alpha_by_horizon"], np.asarray(model_cfg["alpha_grid"])[choice], "Selected-head independent alpha argmin")
            beta = np.column_stack([coefficients[int(i)][:, h] for h, i in enumerate(choice)])
            audit.close(state.head["coef"], beta, "Selected correction independent slope coefficients")
            audit.close(state.head["mean"], mean, "Selected correction feature centering")
            audit.close(state.head["ymean"], ymean, "Selected correction response centering")
            audit.close(((fa - mean) @ beta).mean(axis=0), np.zeros(3), "Selected correction zero A mean")
            other_mode = "insample" if state.mode == "crossfit" else "crossfit"
            other = suite.departure_candidates[f"{other_mode}__{key}__{state.kind}"]
            audit.require(other.bank is state.bank and other.feature_scaler is state.feature_scaler, "Residual-mode feature alignment")
            audit.exact(other.head["mean"], state.head["mean"], "Residual-mode centered feature identity")
            checked_heads.add(head_key)
            audit.counts["unique_selected_correction_heads_independently_replayed"] += 1
        audit.counts["selected_correction_method_occurrences_checked"] += 1
    audit.require(core.fd.state_hash(bundle) == before, "Verification queries did not change loaded state")
    audit.counts["selected_state_numerical_tasks"] += 1


def identities(predictions, fit, audit, label):
    audit.require(set(predictions) == forecast_keys(), label + " exact 42-array roster")
    first = predictions["legacy__calibrated__ridge"]
    for value in predictions.values():
        audit.require(value.shape == first.shape and first.shape[1] == 3 and np.isfinite(value).all(), label + " finite compatible shapes")
    legacy, cross = fit["legacy"], fit["crossfit"]
    audit.exact(first, predictions["legacy__gamma1__ridge"], label + " ridge identity")
    for method in GEOMETRY:
        selected = legacy["selections"]["selected"][method]
        gamma = legacy["calibration"][selected][method]
        audit.close(predictions[f"legacy__calibrated__{method}"], blend(first, predictions[f"legacy__gamma1__{method}"], gamma), label + " legacy gamma")
    for method in ("raw_rbf", "raw_hgb", "persistence"):
        gamma = legacy["calibration_records"]["shared_records"][method]["gamma_by_horizon"]
        audit.close(predictions[f"legacy__calibrated__{method}"], blend(first, predictions[f"legacy__gamma1__{method}"], gamma), label + " raw gamma")
    common, shared = [predictions["crossfit__calibrated__" + m] for m in ("common", "shared_bias")]
    audit.exact(common, predictions["legacy__gamma1__common"], label + " gamma-one deployment common")
    audit.exact(shared, common + np.asarray(cross["selections"]["shared_bias"]["value"]), label + " shared bias")
    audit.exact(predictions["crossfit__calibrated__half_shared_bias"], predictions["crossfit__calibrated__half_common"] + np.asarray(cross["selections"]["half_common"]["shared_bias"]), label + " half bias")
    for method in NEW_METHODS[:4]:
        audit.exact(predictions[f"crossfit__calibrated__{method}"], predictions[f"crossfit__eta1__{method}"], label + " common mode invariance")
    for method in CORRECTIONS:
        eta = cross["calibration"]["choices"][method]["selected_eta"]
        audit.close(predictions[f"crossfit__calibrated__{method}"], blend(shared, predictions[f"crossfit__eta1__{method}"], eta), label + " correction eta")
        for h, value in enumerate(eta):
            if value == 0:
                audit.exact(predictions[f"crossfit__calibrated__{method}"][:, h], shared[:, h], label + " exact eta-zero recovery")
    old_choice = legacy["context_policies"]["selected"]
    for mode in ("calibrated", "gamma1"):
        audit.exact(predictions[f"legacy__{mode}__nonkahm_policy"], predictions[f"legacy__{mode}__{old_choice}"], label + " legacy policy identity")
    for mode in ("calibrated", "eta1"):
        audit.exact(predictions[f"policy__{mode}__expanded_nonkahm"], predictions[policy_key(fit["policy"]["selected_identity"], mode)], label + " expanded policy identity")


def calibration(abc, fit, predictions, cfg, audit):
    identities(predictions, fit, audit, "C")
    cy, scale = abc["C_Y"], abc["loss_scale"]
    grid = np.asarray(cfg["model"]["gamma_grid"])
    ridge, shared = predictions["legacy__calibrated__ridge"], predictions["crossfit__calibrated__shared_bias"]
    cases = []
    for method in GEOMETRY:
        choice = fit["legacy"]["selections"]["selected"][method]
        cases.append(("legacy", method, "gamma1", ridge, fit["legacy"]["calibration_records"]["candidate_records"][choice][method]))
    for method in ("raw_rbf", "raw_hgb", "persistence"):
        cases.append(("legacy", method, "gamma1", ridge, fit["legacy"]["calibration_records"]["shared_records"][method]))
    for method in CORRECTIONS:
        cases.append(("crossfit", method, "eta1", shared, fit["crossfit"]["calibration"]["choices"][method]))
    for family, method, unit, base, record in cases:
        raw = predictions[f"{family}__{unit}__{method}"]
        losses = np.array([np.mean((cy - blend(base, raw, np.full(3, g))) ** 2, axis=0) for g in grid])
        indices = losses.argmin(axis=0)
        audit.close([r["mse_by_horizon"] for r in record["candidates"]], losses, "Independent C calibration score paths")
        audit.exact(record["selected_indices"], indices, "Independent C multiplier argmin")
        audit.exact(record["gamma_by_horizon"], grid[indices], "Independent C chosen multiplier")
        audit.require(record["all_positive_candidates_failed"] is False, "No unexpected calibration failure")
        audit.counts[family + "_C_multiplier_choices"] += 1
    policy = fit["policy"]
    audit.require([r["identity"] for r in policy["candidates"]] == list(POLICY), "Expanded policy tie order")
    scores = []
    for record in policy["candidates"]:
        p = predictions[policy_key(record["identity"], "calibrated")]
        mse = np.mean((cy - p) ** 2, axis=0); score = float(np.mean(mse / scale))
        audit.close(record["mse_by_horizon"], mse, "Expanded C policy MSE")
        audit.close(record["score"], score, "Expanded C policy score")
        audit.require(record["C_prediction_sha256"] == array_hash(p), "Expanded C forecast hash")
        scores.append(score)
    audit.require(policy["selected_identity"] == POLICY[int(np.argmin(scores))], "Expanded C policy first minimum")
    legacy_scores = [float(np.mean(np.mean((cy - predictions[f"legacy__calibrated__{method}"]) ** 2, axis=0) / scale)) for method in NONKAHM]
    audit.require(fit["legacy"]["context_policies"]["selected"] == NONKAHM[int(np.argmin(legacy_scores))], "Legacy C policy first minimum")
    audit.counts["expanded_C_policy_choices"] += 1
    audit.counts["legacy_C_policy_choices"] += 1


def query_diagnostics(fit, query, audit):
    audit.require(query["legacy"]["state_before"] == query["legacy"]["state_after"], "Within-D legacy state invariant")
    cross = query["crossfit"]
    audit.require(cross["frozen_state_before"] == cross["frozen_state_after"], "Within-D cross-fit state invariant")
    # Pickle-based whole-state hashes are intentionally NOT compared across
    # serialization. Exact sealed state bytes and stable components connect C/D.
    for key in ("common_state_sha256", "bias_sha256", "shared_baseline_sha256"):
        audit.require(cross[key] == fit["crossfit"]["calibration"][key], "Stable C/D component identity " + key)
    groups = {"legacy": query["legacy"]["banks"],
              "crossfit_full": cross["geometry"], "crossfit_half": {"half": cross["half_common_geometry"]}}
    for family, banks in groups.items():
        for bank, record in banks.items():
            audit.counts[family + "_cached_query_bank_evaluations"] += 1
            for failure in ("coordinate", "kernel", "gate"):
                events = sum(len(x["rows"]) for x in record.get(failure + "_fallback_rows", []))
                audit.counts[f"{family}_{failure}_failure_local_row_events"] += events
                audit.require(events == 0, family + " unexpected " + failure + " query failure")
            failures = sum("factor_failure" in row for row in record["local_models"])
            audit.counts[family + "_query_factor_failures"] += failures
            audit.require(failures == 0, family + " unexpected factor failure")
            for kind in ("kahm", "rbf"):
                local = [r for r in record["local_models"] if r["kind"] == kind]
                union = sorted({int(row) for r in local for row in r["fallback_rows"]})
                audit.require(union == record[kind + "_fallback_rows"] and len(union) == record[kind + "_fallback_count"], "Reconstruction guard union accounting")
                audit.counts[f"{family}_{kind}_guard_local_model_row_events"] += sum(int(r["fallback_count"]) for r in local)
                audit.counts[f"{family}_{kind}_guard_union_rows_summed_over_banks"] += len(union)
                audit.counts[f"{family}_{kind}_rank_zero_local_query_models"] += sum(bool(r.get("rank_zero")) for r in local)
    for record in query["legacy"]["bounds"]:
        audit.require(record["coefficient_bound_violations"] == 0, "Legacy coefficient bound")
        audit.counts["legacy_coefficient_bound_violations"] += record["coefficient_bound_violations"]
        if record["correction_bound_by_horizon"] is not None:
            violations = np.count_nonzero(np.asarray(record["max_abs_correction"]) > np.asarray(record["correction_bound_by_horizon"]) * (1 + 1e-10) + 1e-12)
            audit.require(violations == 0, "Legacy bounded correction")
    for method, record in cross["bounds"].items():
        violations = np.count_nonzero(np.asarray(record["observed_max_abs_correction_by_horizon"]) > np.asarray(record["correction_bound_by_horizon"]) * (1 + 1e-10) + 1e-12)
        audit.require(violations == record["query_bound_violations"] == record["coefficient_bound_violations"] == 0, "New centered correction bound")
        audit.exact(cross["etas"][method], fit["crossfit"]["calibration"]["choices"][method]["selected_eta"], "Frozen D correction multiplier")
        audit.counts["crossfit_coefficient_bound_violations"] += record["coefficient_bound_violations"]
        audit.counts["crossfit_query_bound_violations"] += int(violations)


def raw_task_losses(predictions, target, scale, audit):
    audit.require(target.shape == (1664, 3) and np.isfinite(target).all(), "Complete finite D target matrix")
    result = {}
    for key, prediction in predictions.items():
        mse = np.mean((target - prediction) ** 2, axis=0)
        result[key] = {"mse": mse, "smse": mse / scale, "variance": np.asarray(scale)}
        audit.counts["raw_D_forecast_arrays_scored"] += 1
        audit.counts["raw_D_horizon_losses_scored"] += 3
    ridge = np.max(np.abs(target - predictions["legacy__calibrated__ridge"]))
    maximum = np.max(np.abs(target - predictions["crossfit__calibrated__crossfit_kahm"]))
    ratio = float(maximum / ridge) if ridge else (0.0 if maximum == 0 else math.inf)
    audit.require(np.isfinite(ratio), "Finite maximum-error ratio")
    return result, {"maximum_absolute_error": float(maximum), "ridge_maximum_absolute_error": float(ridge),
                    "maximum_error_ratio": ratio, "screen_pass": ratio <= 5.0}


def numerical_tables(directory, tasks, raw_losses, screens, audit, *, manufactured=False):
    directory = Path(directory)
    summary = read(directory / "analysis_summary.json")
    audit.require(summary["status"] == "completed" and summary["task_count"] == 192
                  and summary["forecast_arrays_per_task"] == 42, "Completed full-roster numerical tables")
    if manufactured:
        audit.require(summary["scope"].startswith("NONINFERENTIAL CONTRACT FIXTURE")
                      and summary["independent_repetitions"] == 0
                      and summary["nominal_repetition_slots"] == 24,
                      "Explicitly duplicated manufactured arithmetic fixture")
    else:
        audit.require(summary["independent_repetitions"] == 24, "Twenty-four independent production repetitions")
    tables = {}
    for name, record in summary["tables"].items():
        path = directory / name
        audit.require(sha(path) == record["sha256"], "Analysis table hash " + name)
        tables[name] = rows(path)
        audit.require(len(tables[name]) == record["rows"], "Analysis table row count " + name)
        audit.counts["analysis_table_hashes_checked"] += 1
    horizon_seen, task_seen = set(), set()
    for row in tables["task_horizon.csv"]:
        identity = (row["task_id"], row["forecast_key"], int(row["horizon"]))
        audit.require(identity not in horizon_seen, "Unique task/horizon output")
        horizon_seen.add(identity)
        tid, key, h = identity; j = HORIZONS.index(h)
        audit.require(tid in tasks and key in forecast_keys(), "Declared task/horizon identity")
        for value in ("mse", "smse"):
            audit.close(float(row[value]), raw_losses[tid][key][value][j], "Every raw D horizon loss joins output", rtol=1e-11, atol=1e-13)
        audit.close(float(row["A_target_variance"]), raw_losses[tid][key]["variance"][j], "A-only horizon denominator metadata", rtol=1e-12, atol=1e-14)
        audit.require(int(row["D_rows"]) == 1664 and int(row["repetition"]) == tasks[tid]["repetition"], "D row and repetition labels")
        audit.require(row["law"] == tasks[tid]["law"] and int(row["noise_index"]) == tasks[tid]["noise_index"], "Horizon law/noise labels")
        audit.counts["raw_D_horizon_table_joins"] += 1
    audit.require(len(horizon_seen) == 192 * 42 * 3, "Complete horizon table")
    for row in tables["task_losses.csv"]:
        identity = row["task_id"], row["forecast_key"]
        audit.require(identity not in task_seen, "Unique task output")
        task_seen.add(identity)
        for value in ("mse", "smse"):
            audit.close(float(row[value]), raw_losses[identity[0]][identity[1]][value].mean(), "Every raw D task mean joins output", rtol=1e-11, atol=1e-13)
        audit.counts["raw_D_task_loss_joins"] += 1
    audit.require(len(task_seen) == 192 * 42, "Complete task loss table")
    expected = {}
    for panel, laws in PANELS.items():
        for repetition in range(24):
            tids = [tid for tid, task in tasks.items() if task["repetition"] == repetition and task["law"] in laws]
            audit.require(len(tids) == 2 * len(laws), "Independent repetition condition count")
            for key in forecast_keys():
                expected[panel, repetition, key] = {value: float(np.mean([raw_losses[tid][key][value].mean() for tid in tids])) for value in ("mse", "smse")}
    seen = set()
    for row in tables["repetition_losses.csv"]:
        identity = row["panel"], int(row["repetition"]), row["forecast_key"]
        audit.require(identity in expected and identity not in seen, "Unique repetition/forecast aggregate")
        seen.add(identity)
        audit.require(int(row["n_conditions"]) == 2 * len(PANELS[identity[0]]), "Repetition inference condition count")
        for value in ("mse", "smse"):
            audit.close(float(row[value]), expected[identity][value], "Independent repetition aggregate", rtol=1e-11, atol=1e-13)
        audit.counts["repetition_aggregate_joins"] += 1
    audit.require(seen == set(expected), "All 24 by 42 by 3 repetition aggregates")
    def vector(panel, key):
        return np.asarray([expected[panel, r, key]["smse"] for r in range(24)])
    primary_rows = {r["contrast_id"]: r for r in tables["primary_contrasts.csv"]}
    summary_primary = {} if manufactured else {r["contrast_id"]: r for r in summary["primary_contrasts"]}
    detail_rows = {(r["contrast_id"], int(r["repetition"])): r for r in tables["primary_contrast_repetitions.csv"]}
    audit.require(len(primary_rows) == 4 and len(detail_rows) == 96, "Four primary tests and 96 paired gaps")
    primary = []
    critical = float(student_t.ppf(1 - .05 / 4, 23))
    for identifier, panel, comparator_key, factor in PRIMARY:
        candidate = vector(panel, "crossfit__calibrated__crossfit_kahm")
        comparator = vector(panel, comparator_key)
        gap = factor * comparator - candidate
        mean, sd = float(gap.mean()), float(gap.std(ddof=1)); se = sd / math.sqrt(24)
        lower = mean - critical * se
        p = float(student_t.sf(mean / se, 23)) if se else (0.0 if mean > 0 else (1.0 if mean < 0 else .5))
        values = {"candidate_mean_smse": candidate.mean(), "comparator_mean_smse": comparator.mean(),
                  "comparator_factor": factor, "margin_gap_mean": mean, "margin_gap_sample_sd": sd,
                  "margin_gap_se": se, "critical_t": critical, "simultaneous_one_sided_lower": lower,
                  "p_one_sided": p, "p_bonferroni": min(1., 4*p)}
        if comparator.mean() != 0:
            values["point_relative_gain_percent"] = 100 * (1 - candidate.mean() / comparator.mean())
        row = primary_rows[identifier]
        audit.require(int(row["n_repetitions"]) == 24 and int(row["df"]) == 23
                      and int(row["family_size"]) == 4 and float(row["family_alpha"]) == .05,
                      "Primary independent unit and multiplicity")
        for key, value in values.items():
            audit.close(float(row[key]), value, "Independent primary bound " + key, rtol=1e-10, atol=1e-12)
            if not manufactured:
                audit.close(summary_primary[identifier][key], value, "Primary summary/table join " + key, rtol=1e-10, atol=1e-12)
        audit.require(row["criterion_established"] == str(lower > 0), "Primary established flag")
        if not manufactured:
            audit.require(summary_primary[identifier]["criterion_established"] == bool(lower > 0), "Primary summary established flag")
        for repetition in range(24):
            detail = detail_rows[identifier, repetition]
            audit.close(float(detail["margin_gap"]), gap[repetition], "Independent paired primary gap")
            audit.close(float(detail["candidate_smse"]), candidate[repetition], "Primary candidate aggregate join")
            audit.close(float(detail["comparator_smse"]), comparator[repetition], "Primary comparator aggregate join")
        primary.append({"contrast_id": identifier, "candidate_mean_smse": float(candidate.mean()),
                        "comparator_mean_smse": float(comparator.mean()), "mean_margin_gap": mean,
                        "simultaneous_lower_margin_gap": lower, "criterion_established": lower > 0})
        audit.counts["primary_bounds_independently_recomputed"] += 1
    if not manufactured:
        audit.require(summary["primary_comparisons_established"] == sum(r["criterion_established"] for r in primary), "Primary summary count")
    for row in tables["panel_summary.csv"]:
        for value in ("mse", "smse"):
            v = np.asarray([expected[row["panel"], r, row["forecast_key"]][value] for r in range(24)])
            audit.close(float(row[value]), v.mean(), "Panel summary mean")
            audit.close(float(row[value + "_se"]), v.std(ddof=1) / math.sqrt(24), "Panel summary repetition standard error")
    for row in tables["descriptive_contrasts.csv"]:
        left, right = vector(row["panel"], row["left_key"]), vector(row["panel"], row["right_key"])
        difference = left - right; se = difference.std(ddof=1) / math.sqrt(24)
        margin = student_t.ppf(.975, 23) * se
        for key, value in (("mean_difference", difference.mean()), ("standard_error", se),
                           ("lower95", difference.mean() - margin), ("upper95", difference.mean() + margin)):
            audit.close(float(row[key]), value, "Descriptive paired interval replay")
        audit.counts["descriptive_intervals_recomputed"] += 1
    audit.require(len(tables["stability_screens.csv"]) == 192, "Stability screen task count")
    for row in tables["stability_screens.csv"]:
        expected_screen = screens[row["task_id"]]
        for key in ("maximum_absolute_error", "ridge_maximum_absolute_error", "maximum_error_ratio"):
            audit.close(float(row[key]), expected_screen[key], "Independent maximum-error screen")
        audit.require(row["screen_pass"] == str(expected_screen["screen_pass"]), "Independent screen flag")
    if not manufactured:
        audit.close(summary["maximum_selected_error_ratio"], max(s["maximum_error_ratio"] for s in screens.values()), "Maximum selected screen ratio")
        audit.require(summary["maximum_error_screens_passed"] == all(s["screen_pass"] for s in screens.values()), "Stability summary flag")
    return primary


def run(root, source_root, output, *, fixture=False):
    root, source_root, output = Path(root).resolve(), Path(source_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous audit attempts: " + str(output))
    started = time.perf_counter(); audit = Audit(); recovery_contexts = Counter()
    result = {"status": "failed", "started_utc": datetime.now(timezone.utc).isoformat(),
              "audit_source_sha256": sha(__file__), "mode": "manufactured_fixture" if fixture else "computational_reproduction"}
    try:
        # Check existence and declared release before the pipeline even hashes
        # terminal files. No target array or analysis outcome is read beforehand.
        release_path = root / "TARGET_RELEASE.json"
        audit.require(release_path.is_file() and read(release_path)["status"] == "released", "TARGET_RELEASE required before any terminal access")
        sys.path.insert(0, str(source_root / "code"))
        pipeline = load_module("crossfit_production_audit_pipeline", source_root / "code/pipeline.py")
        release = pipeline.verify_release(root)
        result["verified_release_before_target_or_outcome_access"] = True
        cfg = read(root / "config.json")
        freeze = read(root / "freeze.json")
        for name in ("code/crossfit_model.py", "code/pipeline.py", "code/numerical_solver.py", "audit/verify_crossfit.py"):
            audit.require(sha(source_root / name) == freeze["files"][name], "Executed auditor dependency matches frozen source " + name)
        abc_manifest = read(root / "prepared/abc_manifest.json")
        d_manifest = read(root / "prepared/d_manifest.json")
        tasks = {t["task_id"]: t for t in abc_manifest["tasks"]}
        terminals = {t["task_id"]: t for t in d_manifest["tasks"]}
        audit.require(len(tasks) == len(abc_manifest["tasks"]) and set(tasks) == set(terminals), "Complete matching unique task rosters")
        if fixture:
            audit.require(cfg["laws"] == ["manufactured_fixture"] and len(tasks) == 1
                          and abc_manifest["scientific_generator_invoked"] is False
                          and d_manifest["scientific_generator_invoked"] is False,
                          "Fixture mode limited to the explicitly manufactured integration fixture")
        else:
            result["computational_reproduction_provenance"] = reproduction_provenance(root, source_root, cfg, freeze, abc_manifest, audit)
            audit.require(cfg["master_seed"] == 2026110173 and cfg["repetitions"] == 24
                          and len(tasks) == 192 and tuple(cfg["laws"]) == LAWS,
                          "Frozen production cohort identity")
            factorial = {(rep, law, noise) for rep in range(24) for law in LAWS for noise in (0, 1)}
            audit.require({(t["repetition"], t["law"], t["noise_index"]) for t in tasks.values()} == factorial, "Full 192-task factorial")
            audit.require(read(root / "analysis/analysis_summary.json")["status"] == "completed", "Completed analysis required")
        oracle = load_module("crossfit_production_independent_oracle", source_root / "audit/verify_crossfit.py")
        oracle.core.fd.load_runtime()
        raw_losses, screens = {}, {}
        with threadpool_limits(limits=1):
            for index, (tid, task) in enumerate(sorted(tasks.items()), 1):
                terminal = terminals[tid]; folder = root / "results" / tid
                fit, pre = read(folder / "fit_summary.json"), read(folder / "pre_c_selection.json")
                audit.require(fit["status"] == "completed" and fit["task"] == task, "Fit task metadata")
                for field in ("law", "history", "noise_index", "noise_level", "repetition", "seed", "fit_seed"):
                    audit.require(task[field] == terminal[field], "ABC/D task metadata " + field)
                abc = arrays(root / task["input"])
                audit.require(not any(k.startswith("D_") for k in abc), "No D arrays in development archive")
                for key, value in abc.items():
                    audit.require(task["array_sha256"][key] == array_hash(value), "Development array digest")
                training = arrays(folder / "training_arrays.npz")
                validation = arrays(folder / "validation_predictions.npz")
                development(abc, fit, pre, training, validation, cfg, audit)
                solver_events(folder, fit, source_root, freeze, cfg, audit, recovery_contexts)
                bundle = pickle.loads(gzip.decompress((folder / "state.pkl.gz").read_bytes()))
                numerical_states(bundle, abc, fit, training, validation, oracle, cfg, audit)
                cp = arrays(folder / "calibration_predictions.npz")
                calibration(abc, fit, cp, cfg, audit)
                pred_folder = root / "predictions" / tid
                prediction = arrays(pred_folder / "predictions.npz")
                identities(prediction, fit, audit, "D")
                query_diagnostics(fit, read(pred_folder / "query_diagnostics.json"), audit)
                query_arrays = arrays(root / terminal["query"])
                audit.require(not any("Y" in key for key in query_arrays), "D query archive contains no target")
                audit.require(array_hash(query_arrays["D_X"]) == terminal["D_X_sha256"], "D query array digest")
                target_arrays = arrays(root / terminal["target"])
                audit.require(array_hash(target_arrays["D_Y"]) == terminal["D_Y_sha256"], "Released D target digest")
                for key in ("D_groups", "D_origins"):
                    audit.exact(target_arrays[key], query_arrays[key], "D target/query row alignment")
                audit.exact(np.unique(query_arrays["D_groups"], return_counts=True)[0], np.arange(128), "D episode roster")
                audit.exact(np.unique(query_arrays["D_groups"], return_counts=True)[1], np.full(128, 13), "Thirteen origins per D episode")
                raw_losses[tid], screens[tid] = raw_task_losses(prediction, target_arrays["D_Y"], abc["loss_scale"], audit)
                audit.pca_cache.clear()
                del bundle, abc, training, validation, cp, prediction, target_arrays, query_arrays, fit, pre
                audit.counts["complete_tasks_verified"] += 1
                if index % 8 == 0 or index == len(tasks):
                    print(json.dumps({"independent_audit_tasks": index, "total": len(tasks), "seconds": time.perf_counter() - started}), flush=True)
        primary = [] if fixture else numerical_tables(root / "analysis", tasks, raw_losses, screens, audit)
        result.update(status="passed", primary_comparisons=primary,
                      counts=dict(audit.counts), selected_rank_counts=dict(sorted(audit.ranks.items())),
                      numerical_recovery_contexts=dict(sorted(recovery_contexts.items())),
                      numerical_recovery_scope="Every sealed reproduction solver ledger and source/gate record. These recovery counts refer only to the newly executed computational reproduction. Original execution records are not authenticated. Selected-state coefficients receive independent numerical replay; numerical gates for other fallback paths are checked from their recorded evidence.",
                      raw_D_scope="Every retained forecast array in every audited task, all three horizons and every D row; no task subsampling.",
                      selected_state_scope="All four selected fold common models, full-A common, half-training common, and all six logical correction methods per task. Duplicate selected correction states are replayed once. Native PCA checks cover their selected geometry banks; independent normal-equation ridge checks cover selected common heads and selected correction alpha paths.",
                      selection_scope="All legacy and new common/correction B candidate scores; every saved correction alpha forecast path; all B argmins; raw-RBF scaler choices; selected-procedure C multipliers and legacy/expanded C policies.",
                      not_claimed=["Coefficient refits of every unselected candidate", "Independent reconstruction of the inherited raw KAHM feature-map implementation", "Independent regeneration of all D forecasts from coefficients", "A new scientific fit or selection by this auditor", "Natural-data adoption evidence", "Authentication of the original completed run", "Original input or forecast byte identity", "An additional independent simulation cohort"],
                      state_hash_scope="Whole-state pickle hashes compared only before/after the same in-memory stage. Cross-stage identity uses sealed serialized state bytes and stable common/bias/component identities.",
                      diagnostics_scope="All saved query banks; legacy, new full-A cache and half-common cache counted separately. Guard counts are local-model row events and within-bank union rows summed across banks; these are not unique D episodes.",
                      seals={name: sha(root / name) for name in ("freeze.json", "FIT_LOCK.json", "PREDICTION_LOCK.json", "TARGET_RELEASE.json")},
                      source_dependencies={name: sha(source_root / name) for name in ("code/crossfit_model.py", "code/pipeline.py", "code/numerical_solver.py", "audit/verify_crossfit.py")})
        if not fixture:
            result["analysis_summary_sha256"] = sha(root / "analysis/analysis_summary.json")
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc(), counts=dict(audit.counts))
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result.get("counts"), "error": result.get("error"), "receipt": str(output)}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.root, args.source_root or args.root, args.output, fixture=args.fixture))
