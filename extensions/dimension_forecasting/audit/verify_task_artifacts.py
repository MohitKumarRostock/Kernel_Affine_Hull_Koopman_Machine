#!/usr/bin/env python3
"""Independent reconstruction of retained ranks, B selection and C calibration.

No fitting module or primary analyzer is imported. The command-line entry is
restricted to the explicitly manufactured fixture. The functions are reused by
the separately gated post-lock production audit.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np


CANDIDATES = (("historical2", 2, "legacy_normal_departure"), ("native1", 1, "native"),
              ("native2", 2, "native"), ("native4", 4, "native"), ("native8", 8, "native"),
              ("native12", 12, "native"), ("native20", 20, "native"))
NATIVE = tuple(x[0] for x in CANDIDATES[1:])
METHODS = ("ridge", "common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
SHARED = ("ridge", "raw_rbf", "raw_hgb", "persistence")
NONKAHM = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
FULL = SHARED + METHODS[1:] + ("nonkahm_policy",)
CONTEXTS = ("historical2", "native20", "selected")
GEOMETRIC = METHODS[1:]
MODES = ("calibrated", "gamma1")
POLICY_SHA256 = "c03d9a1923b5d9c8ade8a4198fd06e2b24c6add9bf0e0992c469f90d7621efa2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.pca_cache = {}

    def require(self, condition, label, context=None):
        self.counts["assertions"] += 1
        if not bool(condition):
            raise AssertionError(label + (" :: " + str(context) if context is not None else ""))

    def equal(self, left, right, label, context=None):
        self.require(np.array_equal(np.asarray(left), np.asarray(right)), label, context)
        self.counts["exact_array_comparisons"] += 1

    def close(self, left, right, label, context=None):
        left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        self.require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
                     and np.allclose(left, right, rtol=1e-10, atol=1e-12), label, context)
        self.counts["numeric_comparisons"] += int(left.size)


def blend(base, candidate, gamma):
    result = base.copy()
    for h, weight in enumerate(gamma):
        if weight != 0:
            result[:, h] = base[:, h] + weight * (candidate[:, h] - base[:, h])
    return result


def independent_pca(references, cap, policy, audit):
    """Reconstruct the documented native algorithm from the stored references."""
    n, d = references.shape
    initial = min(cap, d if policy == "native" else d-1, n-1)
    if initial == 0:
        return np.empty((d, 0)), [], 0, initial
    key = array_hash(references)
    if key not in audit.pca_cache:
        values = references.T.astype(float)
        centered = values - values.mean(axis=1, keepdims=True)
        covariance = (1.0/(n-1)) * (centered @ centered.T)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, vectors = eigenvalues[order], vectors[:, order]
        positive = int(np.count_nonzero(eigenvalues > np.finfo(float).eps))
        audit.pca_cache[key] = (vectors, positive)
        audit.counts["independent_reference_eigendecompositions"] += 1
    vectors, positive = audit.pca_cache[key]
    if positive == 0:
        return np.empty((d, 0)), [], 0, initial
    requested, trace = initial, []
    while True:
        basis = vectors[:, :min(requested, positive)]
        projected = basis.T @ references.T.astype(float)
        smallest = float(np.ptp(projected, axis=1).min())
        trace.append({"requested_rank": requested, "returned_rank": int(basis.shape[1]),
                      "minimum_projected_range": smallest, "range_threshold_satisfied": smallest >= .001})
        if smallest >= .001 or requested <= 1:
            return basis, trace, positive, initial
        requested -= 1


def expected_prediction_keys():
    keys = {f"{c}__{mode}__{m}" for c, _, _ in CANDIDATES for mode in MODES for m in METHODS}
    keys |= {f"shared__{mode}__{m}" for mode in MODES for m in SHARED}
    keys |= {f"context_{c}__{mode}__{m}" for c in CONTEXTS for mode in MODES for m in FULL}
    keys |= {f"matched__{mode}__{m}" for mode in MODES for m in GEOMETRIC}
    assert len(keys) == 162
    return keys


def verify_views(values, fit, audit, split):
    """Every context is an alias of an already selected, frozen procedure."""
    tid = fit["task"]["task_id"]
    audit.require(set(values) == expected_prediction_keys(), "complete 162-view roster", [tid, split])
    for context in CONTEXTS:
        record = fit["contexts"][context]
        expected = {method: (fit["selections"][method]["selected_candidate_id"] if context == "selected" else context)
                    for method in GEOMETRIC}
        audit.require(record["method_to_candidate"] == expected,
                      "context binds to B-only cap choices", [tid, split, context])
        audit.require(record["role"] == ("primary" if context == "selected" else "secondary"),
                      "only selected context is primary", [tid, context])
        for mode in MODES:
            for method in SHARED + GEOMETRIC:
                source = "shared" if method in SHARED else expected[method]
                audit.equal(values[f"context_{context}__{mode}__{method}"], values[f"{source}__{mode}__{method}"],
                            "context prediction is exact frozen procedure alias", [tid, split, context, mode, method])
            winner = record["selected_policy"]
            audit.require(winner in NONKAHM, "policy candidate excludes both KAHM heads", [tid, context])
            audit.equal(values[f"context_{context}__{mode}__nonkahm_policy"],
                        values[f"context_{context}__{mode}__{winner}"],
                        "non-KAHM policy applies exactly its C-selected whole procedure", [tid, split, context, mode])
    matched = fit["selections"]["kahm_departure"]["selected_candidate_id"]
    audit.require(fit["matched_candidate_id"] == matched, "matched bank is bounded KAHM B-selected bank", tid)
    for mode in MODES:
        for method in GEOMETRIC:
            audit.equal(values[f"matched__{mode}__{method}"], values[f"{matched}__{mode}__{method}"],
                        "matched controls share actual KAHM-selected bank", [tid, split, mode, method])


def verify_shared_and_views(config, fit, pre, validation, calibration, abc, fit_directory, audit):
    tid = fit["task"]["task_id"]
    shared = fit["shared"]
    diag = shared["fit_diagnostics"]
    audit.require(diag["input_hashes"] == pre["input_hashes"] and diag["calibration_used"] is False
                  and diag["query_argument_in_original_raw_helpers"] == "B, not C"
                  and diag["raw_grid_failures"] == [], "shared competitors fit/select only A/B", tid)
    by, cy = abc["B_Y"], abc["C_Y"]
    scale = np.maximum(np.var(abc["A_Y"], axis=0), 1e-12)
    for method in SHARED:
        prediction = validation[f"shared__{method}"]
        audit.require(prediction.shape == by.shape and np.isfinite(prediction).all()
                      and array_hash(prediction) == diag["B_prediction_sha256"][method],
                      "shared B prediction identity", [tid, method])
        if method == "persistence":
            audit.equal(prediction, np.repeat(abc["B_X"][:, -1, None], 3, axis=1),
                        "persistence uses most recent B history value", tid)
            audit.require(diag["B"][method] == {"selection_split": None, "status": "fixed"},
                          "persistence has no hidden B tuning", tid)
            continue
        record = diag["B"][method]
        trials = record["candidates"]
        audit.require(record["fit_split"] == "A" and record["selection_split"] == "B"
                      and all(t["status"] == "ok" for t in trials),
                      "all declared raw competitors succeed on A/B", [tid, method])
        if method == "raw_rbf":
            expected = [(band, alpha) for band in config["model"]["raw_rbf_bandwidth_factors"]
                        for alpha in config["model"]["alpha_grid"]]
            audit.require([(r["bandwidth_factor"], r["alpha"]) for r in trials] == expected
                          and record["failures"] == [] and record["raw_plus_nystrom"] is True,
                          "complete original RBF grid retained in original order", tid)
            ids = np.asarray(record["landmark_indices"])
            audit.require(len(ids) == min(config["model"]["raw_rbf_landmarks"], len(abc["A_X"]))
                          and len(set(ids)) == len(ids) and np.all((ids >= 0) & (ids < len(abc["A_X"]))),
                          "raw RBF landmarks use A rows only", tid)
        elif method == "raw_hgb":
            expected = [(rate, leaves) for rate in config["model"]["raw_hgb_learning_rates"]
                        for leaves in config["model"]["raw_hgb_max_leaf_nodes"]]
            audit.require([(r["learning_rate"], r["max_leaf_nodes"]) for r in trials] == expected
                          and record["failures"] == []
                          and all(r["early_stopping"] is False
                                  and r["max_iter"] == config["model"]["raw_hgb_max_iter"]
                                  and r["l2_regularization"] == config["model"]["raw_hgb_l2_regularization"] for r in trials),
                          "complete original HGB grid retained without hidden validation", tid)
        else:
            audit.require([r["alpha"] for r in trials] == config["model"]["alpha_grid"],
                          "shared ridge retains original alpha grid", tid)
            audit.equal(prediction, validation["historical2__ridge"], "raw and geometry use same B ridge", tid)
        losses = np.asarray([r["mse_by_horizon"] for r in trials])
        selected = np.argmin(losses, axis=0)
        audit.require([r["candidate_index"] for r in record["selected"]] == selected.tolist(),
                      "raw per-horizon first-minimum B selection", [tid, method])
        actual = np.mean((by-prediction)**2, axis=0)
        audit.close(actual, losses[selected, np.arange(3)], "raw B predictions realize chosen candidate", [tid, method])
        audit.close(actual, [r["mse"] for r in record["selected"]], "raw B selected losses exported correctly", [tid, method])
        audit.counts["shared_B_choices_verified"] += 3
    cal = shared["calibration"]
    audit.require(cal["C_X_sha256"] == array_hash(abc["C_X"]) and cal["C_Y_sha256"] == array_hash(cy)
                  and cal["shared_pre_C_state_sha256"] == cal["shared_post_C_state_sha256"] == pre["shared_state_sha256"],
                  "shared C calibration preserves sealed A/B state", tid)
    raw = {method: calibration[f"shared__gamma1__{method}"] for method in SHARED}
    grid, gammas = np.asarray(config["model"]["gamma_grid"]), {}
    for method in SHARED:
        final = calibration[f"shared__calibrated__{method}"]
        audit.require(raw[method].shape == final.shape == cy.shape
                      and np.isfinite(raw[method]).all() and np.isfinite(final).all()
                      and array_hash(raw[method]) == cal["uncalibrated_prediction_sha256"][method]
                      and array_hash(final) == cal["calibrated_prediction_sha256"][method],
                      "shared C prediction identity", [tid, method])
        if method == "ridge":
            audit.equal(raw[method], final, "shared ridge is not recalibrated", tid)
            audit.equal(raw[method], calibration["historical2__gamma1__ridge"],
                        "raw and geometric C ridge identity", tid)
            continue
        if method == "persistence":
            audit.equal(raw[method], np.repeat(abc["C_X"][:, -1, None], 3, axis=1),
                        "persistence uses latest C history value", tid)
        losses = np.array([np.mean((cy-blend(raw["ridge"], raw[method], np.repeat(g,3)))**2, axis=0) for g in grid])
        indices = np.argmin(losses, axis=0)
        record = cal["C"][method]
        audit.require([r["gamma"] for r in record["candidates"]] == grid.tolist()
                      and all(r["status"] == "ok" for r in record["candidates"])
                      and record["selected_indices"] == indices.tolist() and record["selection_split"] == "C",
                      "raw C gamma complete-grid first minimum", [tid, method])
        audit.close(losses, [r["mse_by_horizon"] for r in record["candidates"]],
                    "independent raw C gamma loss reconstruction", [tid, method])
        gamma = grid[indices]
        audit.equal(gamma, record["gamma_by_horizon"], "raw C-only gamma selection", [tid, method])
        audit.equal(final, blend(raw["ridge"], raw[method], gamma), "raw exact C mixture", [tid, method])
        gammas[method] = gamma
        audit.counts["shared_C_gamma_choices_verified"] += 3
    state = shared["state"]
    packed = (fit_directory / state["path"]).read_bytes()
    unpacked = gzip.decompress(packed)
    audit.require(hashlib.sha256(packed).hexdigest() == state["sha256"] and len(packed) == state["compressed_bytes"]
                  and hashlib.sha256(unpacked).hexdigest() == state["uncompressed_state_sha256"]
                  and len(unpacked) == state["pickle_bytes"], "shared compressed and raw state identity", tid)
    verify_views(calibration, fit, audit, "C")
    for context in CONTEXTS:
        record = fit["contexts"][context]["policy"]
        scores = [float(np.mean(np.mean((cy-calibration[f"context_{context}__calibrated__{method}"])**2, axis=0)/scale))
                  for method in NONKAHM]
        winner = NONKAHM[int(np.argmin(scores))]
        audit.require(record["candidate_order"] == list(NONKAHM)
                      and [r["method"] for r in record["candidates"]] == list(NONKAHM)
                      and record["selected_method"] == fit["contexts"][context]["selected_policy"] == winner
                      and record["selection_split"] == "C" and record["cap_choices_already_frozen_on_B"] is True,
                      "non-KAHM C policy selects only seven already-defined procedures", [tid, context])
        audit.equal(record["A_target_variance"], scale, "C policy preserves A-only target variance", [tid, context])
        audit.close(scores, [r["normalized_mse"] for r in record["candidates"]],
                    "independent seven-procedure C score reconstruction", [tid, context])
        audit.counts["C_policy_choices_verified"] += 1
    return gammas


def verify_shared_prediction(predictions, summary, gammas, fit, x, audit):
    tid = fit["task"]["task_id"]
    identity = summary["state_identity"]["shared"]
    audit.require(identity["unchanged"] is True and identity["before_sha256"] == identity["after_sha256"],
                  "shared raw state unchanged by D query", tid)
    diag = summary["diagnostics"]["shared"]
    audit.require(diag["failures"] == [] and all(v == 0 for v in diag["recovered_prediction_cells"].values()),
                  "no raw query failures or recovered cells", tid)
    base = predictions["shared__gamma1__ridge"]
    audit.equal(base, predictions["historical2__gamma1__ridge"], "same ridge for all D competitors", tid)
    audit.equal(base, predictions["shared__calibrated__ridge"], "shared D ridge uncalibrated", tid)
    audit.equal(predictions["shared__gamma1__persistence"], np.repeat(x[:, -1, None], 3, axis=1),
                "D persistence uses latest history only", tid)
    for method in SHARED[1:]:
        audit.equal(predictions[f"shared__calibrated__{method}"],
                    blend(base, predictions[f"shared__gamma1__{method}"], gammas[method]),
                    "shared D predictions use fixed C-only mixing coefficients", [tid, method])
        audit.equal(diag["gamma_by_horizon"][method], gammas[method], "D reports actual fixed raw gamma", [tid, method])


def verify_task(config, abc_path, fit_directory, prediction_directory=None,
                query_path=None, *, audit=None, fixture=False):
    audit = Audit() if audit is None else audit
    fit_directory = Path(fit_directory)
    fit = read(fit_directory / "fit_summary.json")
    pre = read(fit_directory / "pre_c_selection.json")
    task = fit["task"]
    tid = task["task_id"]
    audit.require(bool(task.get("manufactured_only")) == fixture,
                  "manufactured and production audit roles cannot be interchanged", tid)
    audit.require(fit["status"] == "completed" and set(fit["candidates"]) == {x[0] for x in CANDIDATES},
                  "complete fitted candidate roster", tid)
    audit.require(pre["task"] == task and pre["C_arrays_loaded"] is False
                  and pre["D_generated_or_loaded"] is False and pre["selections"] == fit["selections"],
                  "cap selection was preserved across calibration", tid)
    audit.require(pre["candidate_order"] == [x[0] for x in CANDIDATES] and pre["native_cap_order"] == list(NATIVE),
                  "exact prespecified candidate and tie order", tid)
    audit.require(pre["completed_utc"] <= fit["completed_utc"], "pre-C record precedes completed calibration", tid)
    expected_files = {"validation_predictions.npz", "basis_diagnostics.npz", "pre_c_selection.json", "calibration_predictions.npz"}
    expected_files |= {f"states/{name}.pkl.gz" for name, _, _ in CANDIDATES}
    expected_files.add("states/shared.pkl.gz")
    audit.require(set(fit["files"]) == expected_files, "complete fitted artifact roster", tid)
    for relative, digest in fit["files"].items():
        path = (fit_directory / relative).resolve()
        audit.require(path.is_relative_to(fit_directory.resolve()) and sha(path) == digest,
                      "fitted artifact hash identity", [tid, relative])
        audit.counts["fit_files_verified"] += 1
    audit.require(fit["input_sha256"] == sha(abc_path), "fitted ABC file identity", tid)
    abc = arrays(abc_path)
    audit.require(not any(key.startswith("D_") for key in abc), "no D arrays in fitting archive", tid)
    ax, ay, bx, by, cx, cy = [abc[key] for key in ("A_X", "A_Y", "B_X", "B_Y", "C_X", "C_Y")]
    d = int(task["history"])
    audit.require(all(x.ndim == 2 and x.shape[1] == d for x in (ax, bx, cx))
                  and all(y.ndim == 2 and y.shape[1] == 3 for y in (ay, by, cy))
                  and all(len(x) == len(y) for x, y in ((ax, ay), (bx, by), (cx, cy)))
                  and all(np.isfinite(x).all() for x in (ax, ay, bx, by, cx, cy)), "finite ABC shapes", tid)
    if not fixture:
        audit.require([len(ax), len(bx), len(cx)] == [512, 256, 256], "production ABC row counts", tid)
    audit.require(fit["A_B_C_rows"] == [len(ax), len(bx), len(cx)], "reported ABC row counts", tid)
    hashes = {key: array_hash(abc[key]) for key in ("A_X", "A_Y", "B_X", "B_Y")}
    audit.require(pre["input_hashes"] == hashes, "pre-C fitting input hashes", tid)
    scale = np.maximum(np.var(ay, axis=0, ddof=0), 1e-12)
    audit.equal(scale, fit["A_target_variance"], "A-only variance normalization", tid)
    audit.equal(scale, pre["A_target_variance"], "sealed selection scale", tid)
    validation = arrays(fit_directory / "validation_predictions.npz")
    calibration = arrays(fit_directory / "calibration_predictions.npz")
    basis_arrays = arrays(fit_directory / "basis_diagnostics.npz")
    audit.require(set(validation) == {"A_target_variance"} | {f"{c}__{m}" for c, _, _ in CANDIDATES for m in METHODS} | {f"shared__{m}" for m in SHARED},
                  "complete B prediction array roster", tid)
    audit.require(set(calibration) == expected_prediction_keys(),
                  "complete C prediction array roster", tid)
    audit.equal(validation["A_target_variance"], scale, "stored B normalization scale", tid)
    expected_basis, rank_records = set(), []
    standardized = (ax - ax.mean(axis=0)) / np.where(ax.std(axis=0) > 1e-12, ax.std(axis=0), 1.)
    reference_anchor, gammas = None, {}

    for name, cap, policy in CANDIDATES:
        record = fit["candidates"][name]
        diag = record["fit_diagnostics"]
        audit.require(record["candidate_id"] == name and record["rank_cap"] == cap
                      and record["dimension_policy"] == policy, "candidate identity", [tid, name])
        audit.require(diag["input_hashes"] == hashes and diag["calibration_used"] is False
                      and diag["fit_and_alpha_selection_splits"] == ["A", "B"]
                      and diag["seed"] == task["fit_seed"], "A/B-only candidate fitting provenance", [tid, name])
        audit.equal(diag["A_target_variance"], scale, "candidate uses identical A-only loss scale", [tid,name])
        audit.require(diag["bank"]["requested_rank_cap"] == cap and diag["bank"]["dimension_policy"] == policy,
                      "bank-level requested policy metadata", [tid,name])
        locals_ = diag["bank"]["local_models"]
        audit.require(len(diag["local_dimensions"]) == len(locals_), "dimension export covers every actual local model", [tid,name])
        actual_refs = []
        effective_ranks = []
        for index, local in enumerate(locals_):
            q_key, p_key = f"{name}__model{index}__q", f"{name}__model{index}__p"
            expected_basis.update((q_key, p_key))
            q, p = basis_arrays[q_key], basis_arrays[p_key]
            a = local["dimension_audit"]
            ids = np.asarray(local["reference_indices"], dtype=int)
            audit.require(q.ndim == p.ndim == 2 and q.shape[1] == p.shape[0] == d
                          and len(q) == len(ids) <= config["model"]["max_references"]
                          and np.isfinite(q).all() and np.isfinite(p).all()
                          and np.all((ids >= 0) & (ids < len(ax))) and len(set(ids)) == len(ids),
                          "actual local reference/basis shapes and indices", [tid, name, index])
            audit.equal(q, standardized[ids].astype(np.float32).astype(np.float64),
                        "local references are A-standardized selected rows with native quantization", [tid, name, index])
            expected_p, trace, positive, initial = independent_pca(q, cap, policy, audit)
            audit.equal(p, expected_p, "actual basis equals independently reconstructed native PCA/range algorithm", [tid, name, index])
            audit.require(a["policy_source_sha256"] == POLICY_SHA256 and a["policy"] == policy
                          and a["requested_cap"] == cap and a["ambient_dimension"] == d
                          and a["local_reference_count"] == len(q)
                          and a["initial_admissible_rank"] == initial
                          and a["effective_rank"] == p.shape[1] == local["rank"] == local["effective_rank"]
                          and local["reference_count"] == len(q) and local["requested_rank"] == initial,
                          "rank log describes exact actual state", [tid, name, index])
            audit.require(a["reference_dtype"] == "float32" and a["pca_working_dtype"] == "float64"
                          and a["normal_direction_reserved"] == (policy == "legacy_normal_departure")
                          and a["pca_eigenvalue_threshold"] == np.finfo(float).eps
                          and a["range_threshold"] == .001 and a["diagnostic_eigendecomposition_count"] == 0,
                          "native precision and reduction conventions", [tid, name, index])
            limits = {"requested_cap":cap, "ambient_dimension":d if policy == "native" else d-1,
                      "local_sample_rank":len(q)-1}
            audit.require(a["initial_limits"] == limits
                          and a["initial_limit_reasons"] == [key for key,value in limits.items() if value == initial],
                          "initial rank-limiting constraints are correctly named", [tid,name,index])
            export = diag["local_dimensions"][index]
            audit.require(export == {**a,"model_index":index,"stored_basis_shape":list(p.shape),"stored_reference_shape":list(q.shape),
                                     "block_index":local["block"],"basis_sha256":array_hash(p),
                                     "reference_sha256":array_hash(q),"stored_reference_dtype":str(q.dtype)},
                          "exported dimension record is the exact actual-model audit", [tid,name,index])
            audit.require("factor_error" not in local, "no suppressed fitting factor error", [tid, name, index])
            audit.require(len(trace) == len(a["pca_trace"]), "actual range reduction trace length", [tid, name, index])
            for expected, stored in zip(trace, a["pca_trace"]):
                audit.require(expected["requested_rank"] == stored["requested_rank"]
                              and expected["returned_rank"] == stored["returned_rank"]
                              and expected["range_threshold_satisfied"] == stored["range_threshold_satisfied"],
                              "actual requested/effective rank trace", [tid, name, index])
                audit.close(expected["minimum_projected_range"], stored["minimum_projected_range"],
                            "actual projected range", [tid, name, index])
            reported_positive = a["empirical_positive_eigenvalue_count"]
            audit.require(a["empirical_positive_eigenvalue_count_exact"] == (reported_positive is not None)
                          and (reported_positive is None or reported_positive == positive)
                          and a["empirical_positive_eigenvalue_count_lower_bound"] <= positive,
                          "positive-eigenvalue count and uncertainty are truthful", [tid, name, index])
            if p.shape[1] == 0:
                audit.require(a["fallback_applied"] is True and a["status"] == "inherited_zero_rank_fallback"
                              and bool(a["native_failure"]), "rankzero is explicitly an inherited fallback", [tid, name, index])
                reasons = ["insufficient_local_references" if len(q) == 1 else "legacy_normal_dimension_zero"] if initial == 0 else ["no_positive_empirical_variance"]
            else:
                audit.require(a["fallback_applied"] is False and a["status"] == "fitted"
                              and a["native_failure"] is None and a["range_reduction_steps"] == len(trace)-1
                              and a["native_pca_calls"] == len(trace)
                              and a["final_requested_rank"] == trace[-1]["requested_rank"]
                              and a["range_threshold_satisfied"] == trace[-1]["range_threshold_satisfied"]
                              and a["range_floor_reached"] == (trace[-1]["requested_rank"] == 1 and not trace[-1]["range_threshold_satisfied"]),
                              "native range loop and floor outcomes are faithfully logged", [tid, name, index])
                reasons = (["positive_eigenvalue_limit"] if trace[0]["returned_rank"] < initial else [])
                if len(trace) > 1:
                    reasons.append("projected_range_reduction")
                if trace[-1]["requested_rank"] == 1 and not trace[-1]["range_threshold_satisfied"]:
                    reasons.append("range_threshold_unmet_at_rank_one")
            audit.require(a["rank_reduction_reasons"] == reasons, "actual native rank reduction reasons", [tid,name,index])
            actual_refs.append((q.copy(), ids.copy(), local["block"]))
            effective_ranks.append(p.shape[1])
            audit.counts["actual_local_rank_records"] += 1
        if reference_anchor is None:
            reference_anchor = actual_refs
        audit.require(len(reference_anchor) == len(actual_refs), "all caps share local model roster", [tid, name])
        for anchor, observed in zip(reference_anchor, actual_refs):
            audit.equal(anchor[0], observed[0], "all caps use exactly shared local references", [tid, name])
            audit.equal(anchor[1], observed[1], "all caps use exactly shared reference rows", [tid, name])
            audit.require(anchor[2] == observed[2], "all caps use same block mapping", [tid, name])
        common_width, departure_width = len(locals_) + sum(effective_ranks), len(locals_)*d
        widths = record["feature_widths"]
        audit.require(widths["common_geometry"] == common_width
                      and widths["kahm_departure_geometry"] == widths["rbf_departure_geometry"] == departure_width
                      and widths["full_readout"] == {"common": common_width, "kahm_departure": common_width+departure_width,
                                                     "rbf_departure": common_width+departure_width, "kahm_unbounded": common_width+departure_width,
                                                     "rbf_unbounded": common_width+departure_width},
                      "feature widths reflect actual retained bases and matched departure spaces", [tid, name])

        for method in METHODS:
            prediction = validation[f"{name}__{method}"]
            audit.require(prediction.shape == by.shape and np.isfinite(prediction).all(), "finite B prediction shape", [tid, name, method])
            audit.require(array_hash(prediction) == diag["B_prediction_sha256"][method], "B prediction array identity", [tid, name, method])
            b = diag["B"][method]
            trials = b["candidates"]
            audit.require([r["alpha"] for r in trials] == config["model"]["alpha_grid"]
                          and all(r["status"] == "ok" for r in trials), "complete B penalty candidate roster", [tid, name, method])
            losses = np.asarray([r["mse_by_horizon"] for r in trials])
            selected = np.argmin(losses, axis=0)
            audit.require([r["candidate_index"] for r in b["selected"]] == selected.tolist(), "first-minimum B alpha selection", [tid, name, method])
            audit.require([r["alpha"] for r in b["selected"]] == [config["model"]["alpha_grid"][i] for i in selected],
                          "reported selected alpha values match their indices", [tid,name,method])
            observed_loss = np.mean((by-prediction)**2, axis=0)
            audit.close(observed_loss, losses[selected, np.arange(3)], "selected B prediction realizes recorded alpha loss", [tid, name, method])
            audit.close(observed_loss, [r["mse"] for r in b["selected"]], "selected B loss export is correct", [tid,name,method])
            audit.counts["B_alpha_choices_verified"] += 3
        audit.equal(validation[f"{name}__ridge"], validation["historical2__ridge"], "shared B ridge is exactly invariant to cap", tid)
        cal = record["calibration"]
        audit.require(cal["C_X_sha256"] == array_hash(cx) and cal["C_Y_sha256"] == array_hash(cy)
                      and cal["candidate_pre_C_state_sha256"] == cal["candidate_post_C_state_sha256"] == pre["candidate_state_sha256"][name],
                      "C calibration uses declared arrays and preserves sealed A/B state", [tid, name])
        raw = {method: calibration[f"{name}__gamma1__{method}"] for method in METHODS}
        for method in METHODS:
            final = calibration[f"{name}__calibrated__{method}"]
            audit.require(raw[method].shape == final.shape == cy.shape and np.isfinite(raw[method]).all()
                          and np.isfinite(final).all(), "finite C prediction roster", [tid, name, method])
            audit.require(array_hash(raw[method]) == cal["uncalibrated_prediction_sha256"][method]
                          and array_hash(final) == cal["calibrated_prediction_sha256"][method], "C prediction array identity", [tid, name, method])
        audit.equal(raw["ridge"], calibration["historical2__gamma1__ridge"], "shared C ridge is exactly invariant to cap", tid)
        audit.equal(raw["ridge"], calibration[f"{name}__calibrated__ridge"], "ridge is not recalibrated", tid)
        gammas[name] = {}
        grid = np.asarray(config["model"]["gamma_grid"])
        for method in GEOMETRIC:
            losses = np.array([np.mean((cy-blend(raw["ridge"], raw[method], np.repeat(g,3)))**2, axis=0) for g in grid])
            indices = np.argmin(losses, axis=0)
            gamma = grid[indices]
            c = cal["C"][method]
            audit.require([r["gamma"] for r in c["candidates"]] == grid.tolist()
                          and all(r["status"] == "ok" for r in c["candidates"])
                          and c["selected_indices"] == indices.tolist(), "complete C grid and first-minimum selection", [tid, name, method])
            audit.close(losses, [r["mse_by_horizon"] for r in c["candidates"]], "independently reconstructed C gamma losses", [tid, name, method])
            audit.equal(gamma, c["gamma_by_horizon"], "C-only selected mixing coefficients", [tid, name, method])
            audit.equal(blend(raw["ridge"], raw[method], gamma), calibration[f"{name}__calibrated__{method}"],
                        "exact C mixture including gamma-zero identity", [tid, name, method])
            gammas[name][method] = gamma
            audit.counts["C_gamma_choices_verified"] += 3
        state = record["state"]
        payload = (fit_directory / state["path"]).read_bytes()
        audit.require(hashlib.sha256(payload).hexdigest() == state["sha256"] and len(payload) == state["compressed_bytes"],
                      "compressed fitted state identity", [tid, name])
        unpacked = gzip.decompress(payload)
        audit.require(hashlib.sha256(unpacked).hexdigest() == state["uncompressed_state_sha256"]
                      and len(unpacked) == state["pickle_bytes"], "uncompressed fitted state identity without executing pickle", [tid, name])
        rank_records.append({"candidate": name, "rank_cap": cap, "local_ranks": effective_ranks})
    audit.require(set(basis_arrays) == expected_basis, "complete inspectable reference/basis archive", tid)

    for method in GEOMETRIC:
        selection = pre["selections"][method]
        scores = [float(np.mean(np.mean((by-validation[f"{name}__{method}"])**2, axis=0)/scale)) for name in NATIVE]
        chosen = int(np.argmin(scores))
        audit.require(selection["candidate_order"] == list(NATIVE) and selection["selected_index"] == chosen
                      and selection["selected_candidate_id"] == NATIVE[chosen]
                      and selection["selected_cap"] == CANDIDATES[chosen+1][1]
                      and selection["selection_split"] == "B" and selection["calibration_used"] is False,
                      "independently reconstructed pre-C cap choice", [tid, method])
        audit.close(scores, [r["normalized_score"] for r in selection["candidates"]], "all normalized B cap scores", [tid, method])
        for name, trial in zip(NATIVE, selection["candidates"]):
            audit.require(trial["candidate_id"] == name and trial["pre_C_state_sha256"] == pre["candidate_state_sha256"][name]
                          and trial["B_prediction_sha256"] == array_hash(validation[f"{name}__{method}"]),
                          "cap scores bind to frozen candidate and predictions", [tid, method, name])
            audit.close(trial["mse_by_horizon"], np.mean((by-validation[f"{name}__{method}"])**2, axis=0),
                        "reported cap-selection horizon losses are independently reproduced", [tid,method,name])
        if d == 12:
            audit.require(selection["selected_cap"] != 20, "native12/native20 exact tie respects ascending order", [tid, method])
        audit.counts["B_cap_choices_verified"] += 1

    shared_gammas = verify_shared_and_views(config, fit, pre, validation, calibration, abc, fit_directory, audit)

    identity_pairs = [("historical2", "native2")] + ([("native12", "native20")] if d == 12 else [])
    for left, right in identity_pairs:
        for method in METHODS:
            audit.equal(validation[f"{left}__{method}"], validation[f"{right}__{method}"], "mandatory numerical identity on B", [tid, left, right, method])
            for mode in MODES:
                audit.equal(calibration[f"{left}__{mode}__{method}"], calibration[f"{right}__{mode}__{method}"],
                            "mandatory numerical identity on C", [tid, left, right, mode, method])
        for index in range(len(reference_anchor)):
            for key in ("q", "p"):
                audit.equal(basis_arrays[f"{left}__model{index}__{key}"], basis_arrays[f"{right}__model{index}__{key}"],
                            "mandatory local construction identity", [tid, left, right, index, key])

    predictions = None
    if prediction_directory is not None:
        prediction_directory = Path(prediction_directory)
        summary = read(prediction_directory / "predict_summary.json")
        audit.require(summary["status"] == "completed" and summary["task"] == task
                      and summary["D_target_file_opened"] is False
                      and summary["fit_summary_sha256"] == sha(fit_directory / "fit_summary.json"),
                      "prediction used exactly declared fitted task", tid)
        audit.require(summary["prediction_file_sha256"] == sha(prediction_directory / "predictions.npz"), "prediction file identity", tid)
        audit.require(set(summary["state_identity"]) == set(summary["diagnostics"]) == {c[0] for c in CANDIDATES} | {"shared"},
                      "complete D state/diagnostic roster", tid)
        query = arrays(query_path)
        audit.require(not any(k in query for k in ("D_Y", "Y", "targets")) and "D_X" in query, "target-free query archive", tid)
        x = query["D_X"]
        audit.require(x.ndim == 2 and x.shape[1] == d and np.isfinite(x).all()
                      and (fixture or len(x) == 1664), "D query shape and finiteness", tid)
        audit.require(summary["query_file_sha256"] == sha(query_path) and summary["D_X_sha256"] == array_hash(x), "actual D query identity", tid)
        predictions = arrays(prediction_directory / "predictions.npz")
        expected = expected_prediction_keys()
        audit.require(set(predictions) == expected == set(summary["prediction_array_sha256"]), "complete D prediction array roster", tid)
        for key, value in predictions.items():
            audit.require(value.shape == (len(x),3) and np.isfinite(value).all()
                          and array_hash(value) == summary["prediction_array_sha256"][key], "D prediction shape and identity", [tid,key])
        for name, _, _ in CANDIDATES:
            identity = summary["state_identity"][name]
            audit.require(identity["unchanged"] is True and identity["before_sha256"] == identity["after_sha256"], "fixed state unchanged by D queries", [tid,name])
            base = predictions[f"{name}__gamma1__ridge"]
            audit.equal(base, predictions["historical2__gamma1__ridge"], "shared D ridge is exactly invariant to cap", [tid,name])
            audit.equal(base, predictions[f"{name}__calibrated__ridge"], "D ridge remains uncalibrated", [tid,name])
            for method in GEOMETRIC:
                final = predictions[f"{name}__calibrated__{method}"]
                audit.equal(final, blend(base, predictions[f"{name}__gamma1__{method}"], gammas[name][method]),
                            "D mixture exactly uses C-only coefficients", [tid,name,method])
                bound = summary["diagnostics"][name]["bounds"][method]
                maximum = np.max(np.abs(final-base), axis=0)
                limit = None if bound["correction_bound_by_horizon"] is None else np.asarray(bound["correction_bound_by_horizon"])
                audit.close(maximum, bound["max_abs_correction_by_horizon"], "actual D correction diagnostic", [tid,name,method])
                if method.endswith("unbounded"):
                    audit.require(limit is None and not bound["global_bound"], "unbounded control makes no global correction-bound claim", [tid,name,method])
                else:
                    violations = int(np.count_nonzero(maximum > limit*(1+1e-10)+1e-12))
                    audit.require(bound["violations"] == violations == 0, "recorded correction bound holds on all D rows", [tid,name,method])
        for left, right in identity_pairs:
            for mode in MODES:
                for method in METHODS:
                    audit.equal(predictions[f"{left}__{mode}__{method}"], predictions[f"{right}__{mode}__{method}"],
                                "mandatory numerical identity on all D predictions", [tid,left,right,mode,method])
        verify_shared_prediction(predictions, summary, shared_gammas, fit, x, audit)
        verify_views(predictions, fit, audit, "D")
        audit.counts["prediction_tasks_verified"] += 1
    audit.counts["fitted_tasks_verified"] += 1
    return {"task": task, "scale": scale, "selections": pre["selections"], "gammas": gammas,
            "rank_records": rank_records, "predictions": predictions}


def fixture_main(root):
    config = read(root / "config.json")
    fixture = root / "verification/schema_fixture_final"
    audit = Audit()
    status, error = "passed", None
    try:
        result = verify_task(config, fixture / "abc.npz", fixture / "results", fixture / "predictions",
                             fixture / "queries.npz", audit=audit, fixture=True)
    except Exception as exc:
        status, error = "failed", repr(exc)
    receipt = {"schema": "kahm-full-forecast-independent-task-fixture-v1", "status": status,
               "completed_utc": datetime.now(timezone.utc).isoformat(), "fixture_only": True,
               "scope": "Manufactured artifact reconstruction; no production inputs, outcomes, fitting code or primary analyzer imported.",
               "audit_source_sha256": sha(__file__), "counts": dict(audit.counts), "error": error}
    (root / "audit/task_artifact_fixture_checks.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manufactured-fixture", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(fixture_main(args.root.resolve()))
