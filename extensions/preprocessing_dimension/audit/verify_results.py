#!/usr/bin/env python3
"""Independent, release-gated numerical and provenance audit.

This auditor does not import fitting routines or the primary analyzer. The
pipeline is imported solely for its global release guard. Raw D loss joins
cover the predeclared eight repetition-zero tasks; aggregate arithmetic,
B choices, and primary native bases cover all 192 tasks. Coverage is reported
explicitly rather than described as a replay of every raw forecast.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from scipy.stats import t as student_t

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCALERS = ("standard", "minmax", "tanh_z_1")
CAPS = (1, 2, 4, 8, 12, 20)
PAIRS = ("standard", "minmax", "tanh", "hybrid_tanh_minmax", "hybrid_minmax_tanh")
CONTEXTS = ("selected", "cap2", "cap20", *PAIRS)
GEOMETRY = ("common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
NONKAHM = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", *GEOMETRY, "nonkahm_policy")
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
MODES = ("calibrated", "gamma1")
HORIZONS = (1, 4, 12)
IDENTITY = ("family", "context", "pair", "cap", "view", "mode", "method")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.pca_cache = {}
        self.rank_counts = Counter()

    def require(self, valid, label, detail=""):
        self.counts["assertions"] += 1
        if not bool(valid):
            raise AssertionError(f"{label}: {detail}")

    def close(self, actual, expected, label, *, rtol=1e-10, atol=1e-12):
        left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
                     and np.allclose(left, right, rtol=rtol, atol=atol), label)
        self.counts["numerical_values_compared"] += left.size

    def exact(self, actual, expected, label):
        self.require(np.array_equal(actual, expected), label)
        self.counts["exact_array_comparisons"] += 1


def independent_history_transform(x, scaler):
    """Compute the declared history transform without any fitted-model import."""
    if scaler == "minmax":
        low, high = x.min(axis=0), x.max(axis=0)
        span = high - low
        transformed = 2.0 * ((x - low) / np.where(span == 0, 1.0, span)) - 1.0
        transformed[:, span == 0] = 0
        return transformed
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    z = (x - mean) / np.where(sd > 1e-12, sd, 1.0)
    return z if scaler == "standard" else np.tanh(z)


def independent_native_pca(q, cap, audit):
    n, dimension = q.shape
    initial = min(cap, dimension, n - 1)
    if initial == 0:
        return np.empty((dimension, 0)), initial, initial, 0, []
    key = array_hash(q)
    if key not in audit.pca_cache:
        x = q.T.astype(np.float64)
        centered = x - x.mean(axis=1, keepdims=True)
        covariance = (1.0 / (n - 1)) * (centered @ centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(-eigenvalues)
        positive = int(np.count_nonzero(eigenvalues[order] > np.finfo(float).eps))
        audit.pca_cache[key] = eigenvectors[:, order], positive
        audit.counts["independent_reference_eigendecompositions"] += 1
    vectors, positive = audit.pca_cache[key]
    if positive == 0:
        return np.empty((dimension, 0)), initial, initial, positive, []
    requested, trace = initial, []
    while True:
        basis = vectors[:, :min(requested, positive)]
        smallest = float(np.ptp(basis.T @ q.T, axis=1).min())
        trace.append((requested, basis.shape[1], smallest))
        if smallest >= 1e-3 or requested <= 1:
            return basis, initial, requested, positive, trace
        requested -= 1


def verify_banks(ax, banks, saved_basis, audit):
    expected = {f"{scaler}__cap{cap}" for scaler in SCALERS for cap in CAPS}
    audit.require(set(banks) == expected, "All eighteen primary banks retained")
    transformed = {scaler: independent_history_transform(ax, scaler) for scaler in SCALERS}
    reference_ids = {}
    expected_keys = set()
    for key, bank in banks.items():
        scaler, cap_text = key.rsplit("__cap", 1)
        cap = int(cap_text)
        source = bank["bank"]
        local = source["local_models"]
        dimensions = bank["local_dimensions"]
        audit.require(source["dimension_policy"] == "native" and source["requested_rank_cap"] == cap,
                      "Native cap metadata", key)
        audit.require(len(local) == len(dimensions) == sum(source["effective_blocks"]), "Local-model roster", key)
        audit.counts["A_only_geometry_transform_banks"] += 1
        for index, (record, dimension) in enumerate(zip(local, dimensions)):
            prefix = f"{key}__local{index}__"
            expected_keys.update(prefix + name for name in ("q", "p", "center"))
            q, basis, center = [saved_basis[prefix + name] for name in ("q", "p", "center")]
            ids = np.asarray(record["reference_indices"], dtype=int)
            audit.require(ids.ndim == 1 and len(ids) == len(q)
                          and np.all((ids >= 0) & (ids < len(ax))), "A-only reference indices", prefix)
            expected_q = transformed[scaler][ids].astype(np.float32).astype(np.float64)
            audit.exact(q, expected_q, "Stored references equal independently transformed A rows: " + prefix)
            audit.exact(center, q.mean(axis=0), "Local reference center: " + prefix)
            audit.require(dimension["reference_sha256"] == array_hash(q)
                          and dimension["basis_sha256"] == array_hash(basis), "Basis/reference array identities", prefix)
            audit.require(dimension["policy"] == "native" and not dimension["normal_direction_reserved"],
                          "No legacy normal-direction restriction", prefix)
            independent, initial, final_requested, positive, trace = independent_native_pca(q, cap, audit)
            audit.require(dimension["initial_admissible_rank"] == min(cap, ax.shape[1], len(q) - 1) == initial,
                          "Native initial dimension", prefix)
            audit.require(basis.shape == independent.shape
                          and basis.shape[1] == dimension["effective_rank"] == record["effective_rank"],
                          "Native effective dimension and actual basis shape", prefix)
            audit.close(basis @ basis.T, independent @ independent.T,
                        "Independent covariance-PCA subspace: " + prefix, rtol=2e-8, atol=2e-8)
            if basis.shape[1]:
                audit.close(basis.T @ basis, np.eye(basis.shape[1]), "Basis orthonormality: " + prefix)
                audit.require(dimension["final_requested_rank"] == final_requested
                              and dimension["range_reduction_steps"] == initial - final_requested,
                              "Native projected-range reduction count", prefix)
                audit.require(len(trace) == len(dimension["pca_trace"]), "Native range-loop trace length", prefix)
                for (rank, returned, span), observed in zip(trace, dimension["pca_trace"]):
                    audit.require(observed["requested_rank"] == rank and observed["returned_rank"] == returned,
                                  "Native range-loop requested/returned dimension", prefix)
                    audit.close(observed["minimum_projected_range"], span, "Native projected range: " + prefix,
                                rtol=2e-8, atol=2e-10)
            ref_key = (scaler, index)
            if ref_key in reference_ids:
                audit.exact(ids, reference_ids[ref_key], "References paired across caps: " + prefix)
            else:
                reference_ids[ref_key] = ids.copy()
            audit.counts["primary_local_bases_independently_reconstructed"] += 1
            audit.rank_counts[f"{scaler}/cap{cap}/rank{basis.shape[1]}"] += 1
    audit.require(set(saved_basis) == expected_keys, "Complete saved primary basis-array roster")


def verify_development(task, abc, fit, pre, validation, calibration, audit):
    primary = fit["primary"]
    scale = np.maximum(np.var(abc["A_Y"], axis=0, ddof=0), 1e-12)
    audit.exact(abc["loss_scale"], scale, "A-only evaluation normalization")
    audit.require(pre["C_loaded"] is False and pre["primary"]["selections"] == primary["selections"]
                  and pre["primary"]["b_scores"] == primary["b_scores"], "B choices durably saved before C")
    ordered = [f"{pair}__cap{cap}" for cap in CAPS for pair in PAIRS]
    audit.require(set(primary["b_scores"]) == set(ordered), "Complete primary B candidate roster")
    for candidate in ordered:
        for method in GEOMETRY:
            prediction = validation[f"{candidate}__{method}"]
            loss = np.mean((abc["B_Y"] - prediction) ** 2, axis=0)
            score = float(np.mean(loss / scale))
            audit.close(primary["b_scores"][candidate][method], score, "Independent B forecast-score join")
            audit.counts["primary_B_candidate_scores_recomputed"] += 1
    for context in CONTEXTS:
        for method in GEOMETRY:
            eligible = ordered
            if context in ("cap2", "cap20"):
                eligible = [key for key in eligible if key.endswith("__" + context)]
            elif context in PAIRS:
                eligible = [key for key in eligible if key.startswith(context + "__cap")]
            if method == "common" and context in ("selected", "cap2", "cap20"):
                eligible = [key for key in eligible if key.rsplit("__cap", 1)[0] in PAIRS[:3]]
            expected = min(eligible, key=lambda key: primary["b_scores"][key][method])
            audit.require(primary["selections"][context][method] == expected, "Independent B first-minimum selection")
            audit.counts["primary_B_context_choices_recomputed"] += 1
        scores = {}
        for method in NONKAHM:
            prediction = calibration[f"primary__{context}__calibrated__{method}"]
            scores[method] = float(np.mean(np.mean((abc["C_Y"] - prediction) ** 2, axis=0) / scale))
        selected = min(NONKAHM, key=scores.__getitem__)
        audit.require(primary["context_policies"][context] == selected, "Independent C non-KAHM policy argmin")
        for method in NONKAHM:
            audit.close(primary["calibration_records"]["policies"][context]["scores"][method], scores[method],
                        "C policy score reconstructed from saved forecasts")
        audit.counts["C_policy_choices_recomputed"] += 1
    raw_scores = {}
    for scaler in SCALERS:
        record = primary["diagnostics"]["raw_rbf"][scaler]
        audit.require(record["fit_split"] == "A" and record["selection_split"] == "B"
                      and not record["failures"], "Raw RBF A/B development roles")
        losses = np.asarray([candidate["mse_by_horizon"] for candidate in record["candidates"]])
        chosen = np.argmin(losses, axis=0)
        audit.require([row["candidate_index"] for row in record["selected"]] == chosen.tolist(),
                      "Raw RBF per-horizon B grid argmin")
        raw_scores[scaler] = float(np.mean(losses[chosen, np.arange(3)] / scale))
        audit.close(primary["diagnostics"]["raw_rbf_scores"][scaler], raw_scores[scaler], "Raw RBF scaler score")
    audit.require(primary["raw_rbf_selected_scaler"] == min(SCALERS, key=raw_scores.__getitem__),
                  "Raw RBF B-only input-scaler argmin")
    audit.counts["raw_RBF_scaler_choices_recomputed"] += 1
    secondary = fit["secondary"]
    diagnostic = secondary["diagnostics"]
    pids, rids = diagnostic["P_episode_ids"], diagnostic["R_episode_ids"]
    audit.require(len(pids) == len(rids) == 32 and set(pids).isdisjoint(rids), "P/R episode separation")
    pmask = np.isin(abc["A_groups"], pids)
    px, py = abc["A_X"][pmask], abc["A_Y"][pmask]
    pscale = np.maximum(np.var(py, axis=0, ddof=0), 1e-12)
    audit.close(secondary["selections"]["P_target_variance"], pscale, "P-only common/departure selection normalization")
    for key, record in diagnostic["banks"].items():
        scaler = key.rsplit("__cap", 1)[0]
        actual = record["geometry_scaler"]
        if scaler == "minmax":
            audit.exact(actual["minimum"], px.min(axis=0), "Secondary P-only min-max minimum")
            audit.exact(actual["maximum"], px.max(axis=0), "Secondary P-only min-max maximum")
            audit.require(actual["fit_rows"] == len(px) and actual["clip"] is False, "Secondary P-only min-max scope")
        elif scaler == "tanh_z_1":
            audit.exact(actual["mean"], px.mean(axis=0), "Secondary P-only tanh mean")
            audit.exact(actual["raw_standard_deviation"], px.std(axis=0, ddof=0), "Secondary P-only tanh standard deviation")
            audit.require(actual["fit_rows"] == len(px) and actual["post_tanh_rescale"] is False,
                          "Secondary tanh scope")
        else:
            audit.require(actual is None, "Secondary standard-scaler representation")
        audit.counts["secondary_history_scaler_metadata_banks_checked"] += 1
    for arm in SCALERS:
        record = diagnostic["common_rows"][arm]
        audit.require(record["bias_fit_split"] == "B" and not record["R_common_target_access"], "Fixed-common target roles")
        common = calibration[f"secondary__calibrated__common__{arm}"]
        shared = calibration[f"secondary__calibrated__shared_bias__{arm}"]
        audit.exact(shared, common + np.asarray(record["bias"]), "Shared fixed-common bias applied exactly")
    audit.counts["development_tasks_checked"] += 1


def verify_query_diagnostics(fit, query, audit):
    """Account for guards with explicit bank/local-row units, without D labels."""
    fit_banks = {
        "primary": fit["primary"]["banks"],
        "secondary": fit["secondary"]["diagnostics"]["banks"],
    }
    query_banks = {
        "primary": query["primary"]["banks"],
        "secondary": query["secondary"]["geometry"],
    }
    for architecture in ("primary", "secondary"):
        factor_failures = sum(bool(local.get("factor_error"))
            for bank in fit_banks[architecture].values() for local in bank["bank"]["local_models"])
        audit.counts[architecture + "_fitted_factor_failures"] += factor_failures
        audit.require(factor_failures == 0, architecture + " unexpected fitted factor failure")
        for bank, record in query_banks[architecture].items():
            audit.counts[architecture + "_unique_cached_query_bank_evaluations"] += 1
            for kind in ("coordinate", "kernel", "gate"):
                events = sum(len(entry["rows"]) for entry in record.get(kind + "_fallback_rows", []))
                audit.counts[f"{architecture}_{kind}_fallback_local_row_events"] += events
                audit.require(events == 0, architecture + " unexpected " + kind + " query failure", bank)
            query_factor_failures = sum("factor_failure" in local for local in record["local_models"])
            audit.counts[architecture + "_query_factor_failures"] += query_factor_failures
            audit.require(query_factor_failures == 0, architecture + " unexpected query factor failure", bank)
            for kind in ("kahm", "rbf"):
                local = [entry for entry in record["local_models"] if entry["kind"] == kind]
                local_events = sum(int(entry["fallback_count"]) for entry in local)
                union = sorted({int(row) for entry in local for row in entry["fallback_rows"]})
                audit.require(union == record[kind + "_fallback_rows"]
                              and len(union) == record[kind + "_fallback_count"],
                              "Reconstruction-guard union accounting", [architecture, bank, kind])
                audit.counts[f"{architecture}_{kind}_reconstruction_guard_local_model_row_events"] += local_events
                audit.counts[f"{architecture}_{kind}_reconstruction_guard_union_rows_summed_over_banks"] += len(union)
                audit.counts[f"{architecture}_{kind}_rank_zero_local_query_models"] += sum(bool(entry.get("rank_zero")) for entry in local)
    for bound in query["primary"]["bounds"]:
        violations = int(bound["coefficient_bound_violations"])
        audit.counts["primary_coefficient_bound_violations"] += violations
        audit.require(violations == 0, "Primary coefficient bound")
        limit = bound["correction_bound_by_horizon"]
        if limit is not None:
            actual = np.asarray(bound["max_abs_correction"])
            violations = int(np.count_nonzero(actual > np.asarray(limit) * (1 + 1e-10) + 1e-12))
            audit.counts["primary_D_correction_bound_violations"] += violations
            audit.require(violations == 0, "Primary D correction bound")
            audit.counts["primary_bounded_query_readouts_checked"] += 1
        else:
            audit.counts["primary_unbounded_query_readouts_not_claimed_bounded"] += 1
    for bound in query["secondary"]["bounds"].values():
        actual = np.asarray(bound["observed_max_abs_correction_by_horizon"])
        limit = np.asarray(bound["correction_bound_by_horizon"])
        violations = int(np.count_nonzero(actual > limit * (1 + 1e-10) + 1e-12))
        audit.require(violations == int(bound["query_bound_violations"]) == 0, "Secondary D correction bound")
        coefficient_violations = int(bound["coefficient_bound_violations"])
        audit.require(coefficient_violations == 0, "Secondary coefficient bound")
        audit.counts["secondary_D_correction_bound_violations"] += violations
        audit.counts["secondary_coefficient_bound_violations"] += coefficient_violations
        audit.counts["secondary_bounded_query_readouts_checked"] += 1


def expected_prediction_keys():
    keys = {f"primary__{context}__{mode}__{method}" for context in CONTEXTS for mode in MODES for method in METHODS}
    keys |= {f"candidate__{pair}__cap{cap}__{mode}__{method}" for pair in PAIRS for cap in CAPS for mode in MODES for method in GEOMETRY}
    keys |= {f"secondary__{mode}__{method}__{scaler}" for mode in MODES for method in ("common", "shared_bias") for scaler in SCALERS}
    keys |= {f"secondary__{mode}__{pair}__{method}" for mode in MODES for pair in ("ss", "mm", "tt", "tm", "mt") for method in ("kahm", "rbf", "matched_rbf")}
    assert len(keys) == 502
    return keys


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        yield from csv.DictReader(stream)


def audit_tables(root, tasks, d_tasks, audit):
    expected_keys = expected_prediction_keys()
    subset = {key for key, task in tasks.items() if task["repetition"] == 0}
    audit.require(len(subset) == 8, "Prespecified raw-loss subset has eight repetition-zero tasks")
    raw_losses = {}
    for tid in sorted(subset):
        abc = load_arrays(root / tasks[tid]["input"])
        outcomes = load_arrays(root / d_tasks[tid]["target"])["D_Y"]
        predictions = load_arrays(root / "predictions" / tid / "predictions.npz")
        audit.require(outcomes.shape == (1664, 3) and set(predictions) == expected_keys,
                      "Complete raw prediction/outcome roster in audit subset", tid)
        scale = np.maximum(np.var(abc["A_Y"], axis=0, ddof=0), 1e-12)
        for key, prediction in predictions.items():
            audit.require(prediction.shape == outcomes.shape and np.isfinite(prediction).all(), "Finite raw subset forecast")
            mse = np.mean((outcomes - prediction) ** 2, axis=0)
            raw_losses[tid, key] = mse, mse / scale
            audit.counts["raw_D_prediction_arrays_recomputed"] += 1
        audit.counts["raw_D_tasks_recomputed"] += 1
    raw_horizon_keys = set()
    for row in csv_rows(root / "analysis/task_horizon.csv"):
        key = row["task_id"], row["prediction_key"]
        if key not in raw_losses:
            continue
        horizon = int(row["horizon"])
        index = HORIZONS.index(horizon)
        audit.require((*key, horizon) not in raw_horizon_keys, "Unique raw horizon-table join")
        raw_horizon_keys.add((*key, horizon))
        audit.close([float(row["mse"]), float(row["smse"])],
                    [raw_losses[key][0][index], raw_losses[key][1][index]], "Raw D forecast-to-horizon-loss join")
        audit.counts["raw_D_horizon_loss_joins"] += 1
    audit.require(len(raw_horizon_keys) == 8 * 502 * 3, "Complete raw horizon-loss subset")
    groups = defaultdict(list)
    task_keys = defaultdict(set)
    for row in csv_rows(root / "analysis/task_losses.csv"):
        tid, key = row["task_id"], row["prediction_key"]
        audit.require(tid in tasks and key in expected_keys and key not in task_keys[tid], "Unique complete task-loss roster")
        task_keys[tid].add(key)
        task = tasks[tid]
        audit.require(int(row["repetition"]) == task["repetition"] and row["law"] == task["law"]
                      and int(row["noise_index"]) == task["noise_index"], "Task-loss metadata join")
        values = float(row["mse"]), float(row["smse"])
        audit.require(np.isfinite(values).all() and min(values) >= 0, "Finite nonnegative task losses")
        if (tid, key) in raw_losses:
            audit.close(values, [raw_losses[tid, key][0].mean(), raw_losses[tid, key][1].mean()], "Raw D forecast-to-task-loss join")
            audit.counts["raw_D_task_loss_joins"] += 1
        panels = ("all_descriptive", "linear" if task["law"] == "linear_ar" else "nonlinear")
        identity = tuple(row[field] for field in IDENTITY)
        for panel in panels:
            groups[panel, task["repetition"], identity].append((task["law"], task["noise_index"], values))
        audit.counts["task_loss_table_records_verified"] += 1
    audit.require(set(task_keys) == set(tasks) and all(keys == expected_keys for keys in task_keys.values()),
                  "All 192 by 502 task-loss records retained")
    repetition_vectors = {}
    seen = set()
    for row in csv_rows(root / "analysis/repetition_losses.csv"):
        identity = tuple(row[field] for field in IDENTITY)
        key = row["panel"], int(row["repetition"]), identity
        audit.require(key in groups and key not in seen, "Repetition table unique group join")
        seen.add(key)
        members = groups[key]
        expected_count = {"nonlinear": 6, "linear": 2, "all_descriptive": 8}[row["panel"]]
        audit.require(len(members) == expected_count and len({(m[0], m[1]) for m in members}) == expected_count,
                      "Independent-unit aggregation retains all law/noise cells")
        values = np.mean([member[2] for member in members], axis=0)
        audit.close([float(row["mse"]), float(row["smse"])], values, "Independent repetition arithmetic")
        vector_key = row["panel"], row["view"], row["mode"], row["method"], int(row["repetition"])
        audit.require(vector_key not in repetition_vectors, "Unique repetition forecast identity")
        repetition_vectors[vector_key] = float(values[1])
        audit.counts["repetition_aggregate_records_recomputed"] += 1
    audit.require(seen == set(groups) and len(seen) == 24 * 502 * 3, "Complete 24-unit repetition aggregation")
    del groups
    def vector(panel, context, method):
        return np.array([repetition_vectors[panel, "primary__" + context, "calibrated", method, rep] for rep in range(24)])
    cases = {
        "selected_KAHM_vs_nonKAHM_policy": ("nonlinear", "selected", "kahm_departure", "selected", "nonkahm_policy", .97),
        "selected_KAHM_vs_selected_RBF": ("nonlinear", "selected", "kahm_departure", "selected", "rbf_departure", .97),
        "tanh_KAHM_vs_standard_KAHM": ("nonlinear", "tanh", "kahm_departure", "standard", "kahm_departure", .97),
        "linear_selected_KAHM_vs_ridge": ("linear", "selected", "kahm_departure", "selected", "ridge", 1.05),
    }
    critical = float(student_t.ppf(1 - .05 / 4, 23))
    primary = {}
    for row in csv_rows(root / "analysis/primary_contrasts.csv"):
        identifier = row["contrast_id"]
        audit.require(identifier in cases and identifier not in primary, "Four frozen primary claims")
        panel, candidate_context, candidate_method, comparator_context, comparator_method, factor = cases[identifier]
        candidate = vector(panel, candidate_context, candidate_method)
        comparator = vector(panel, comparator_context, comparator_method)
        gap = factor * comparator - candidate
        mean, sd = float(gap.mean()), float(gap.std(ddof=1))
        se = sd / np.sqrt(24)
        lower = mean - critical * se
        numbers = {
            "candidate_mean_smse": float(candidate.mean()), "comparator_mean_smse": float(comparator.mean()),
            "point_relative_gain_percent": float(100 * (1 - candidate.mean() / comparator.mean())),
            "margin_gap_mean": mean, "margin_gap_sample_sd": sd, "margin_gap_se": float(se),
            "critical_t": critical, "simultaneous_one_sided_lower": float(lower),
        }
        for key, expected in numbers.items():
            audit.close(float(row[key]), expected, "Independent primary arithmetic: " + identifier + "/" + key)
        audit.require(int(row["n_repetitions"]) == 24 and int(row["df"]) == 23 and int(row["family_size"]) == 4
                      and float(row["family_alpha"]) == .05, "Primary independent units and family allocation")
        established = row["criterion_established"].lower() == "true"
        audit.require(established == (lower > 0), "Primary margin decision follows simultaneous lower gap")
        primary[identifier] = {**numbers, "criterion_established": established, "comparator_factor": factor,
                               "independent_unit": "repetition", "n": 24}
        audit.counts["primary_comparisons_independently_recomputed"] += 1
    audit.require(set(primary) == set(cases), "All four primary comparisons audited")
    summary = read(root / "analysis/analysis_summary.json")
    audit.require(summary["primary_comparisons_established"] == sum(row["criterion_established"] for row in primary.values()),
                  "Summary primary conclusion joins independently recomputed tests")
    for name, record in summary["tables"].items():
        audit.require(sha(root / "analysis" / name) == record["sha256"], "Reported table checksum", name)
        audit.counts["analysis_table_checksums_verified"] += 1
    return primary


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the earlier audit attempt: " + str(output))
    audit = Audit()
    result = {"status": "failed", "started_utc": datetime.now(timezone.utc).isoformat(),
              "audit_source_sha256": sha(Path(__file__))}
    try:
        sys.path.insert(0, str(root / "code"))
        from pipeline import verify_release
        # No terminal target, forecast, or outcome table is read before this
        # guard checks the complete global fit, prediction, and release seals.
        release = verify_release(root)
        result["global_release_verified_before_outcome_access"] = True
        analysis = read(root / "analysis/analysis_summary.json")
        audit.require(analysis["status"] == "completed" and analysis["task_count"] == 192, "Completed primary analysis required")
        abc_manifest, d_manifest = [read(root / "prepared" / name) for name in ("abc_manifest.json", "d_manifest.json")]
        tasks = {task["task_id"]: task for task in abc_manifest["tasks"]}
        d_tasks = {task["task_id"]: task for task in d_manifest["tasks"]}
        factorial = {(law, noise, rep) for law in LAWS for noise in (0, 1) for rep in range(24)}
        audit.require(len(tasks) == len(d_tasks) == 192 and set(tasks) == set(d_tasks)
                      and {(t["law"], t["noise_index"], t["repetition"]) for t in tasks.values()} == factorial,
                      "Complete frozen 192-task factorial")
        for index, (tid, task) in enumerate(sorted(tasks.items()), 1):
            abc = load_arrays(root / task["input"])
            folder = root / "results" / tid
            fit, pre = [read(folder / name) for name in ("fit_summary.json", "pre_c_selection.json")]
            basis = load_arrays(folder / "basis_diagnostics.npz")
            validation = load_arrays(folder / "validation_predictions.npz")
            # Only declared C forecasts are needed for policy replay. Loading
            # lazily avoids redundant copies of every diagnostic C view.
            with np.load(folder / "calibration_predictions.npz", allow_pickle=False) as calibration:
                verify_development(task, abc, fit, pre, validation, calibration, audit)
            verify_banks(abc["A_X"], fit["primary"]["banks"], basis, audit)
            query_diagnostics = read(root / "predictions" / tid / "query_diagnostics.json")
            verify_query_diagnostics(fit, query_diagnostics, audit)
            if index % 24 == 0:
                print(json.dumps({"independent_development_and_basis_tasks": index, "expected": 192}), flush=True)
        primary = audit_tables(root, tasks, d_tasks, audit)
        result.update(status="passed", counts=dict(audit.counts), rank_counts=dict(sorted(audit.rank_counts.items())),
                      primary_comparisons=primary,
                      raw_D_scope="Only repetition 0: all four laws and both noise levels, eight tasks, all 502 arrays and three horizons.",
                      aggregate_scope="All 192x502 task-loss rows joined to all 24x502x3 repetition-panel aggregates; all four primary margin calculations independently reproduced.",
                      construction_scope="Every primary bank and actual local basis independently reconstructed from saved references; references joined to independently transformed A-only histories. Secondary history-scaler metadata checked against P; not every secondary actual basis was independently reconstructed.",
                      selection_scope="Every primary B candidate score recomputed from saved B forecasts; all joint B context choices, raw-RBF scaler choices, and C non-KAHM policy decisions independently replayed.",
                      diagnostic_count_scope="All task query diagnostics, primary and secondary separately. Reconstruction guards retain inherited aggregate reasons; local-model row events and within-bank union rows summed over banks are distinct units. Repeated caps/banks can count the same input more than once. Secondary banks are counted once per cached query bank, not once per consuming correction head.",
                      audit_import_scope="No fitting or primary scoring functions used. The pipeline was imported only for global release verification.",
                      not_claimed=["A raw-forecast-to-loss replay for all 192 tasks", "A new model fit", "True risk-certificate coverage", "Natural-data adoption evidence", "Isolated per-method timing verification"],
                      seals={name: sha(root / name) for name in ("freeze.json", "FIT_LOCK.json", "PREDICTION_LOCK.json", "TARGET_RELEASE.json")},
                      analysis_summary_sha256=sha(root / "analysis/analysis_summary.json"))
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc(), counts=dict(audit.counts))
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result.get("counts"),
                      "error": result.get("error"), "receipt": str(output)}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.root, args.output or args.root / "audit/results_audit.json"))
