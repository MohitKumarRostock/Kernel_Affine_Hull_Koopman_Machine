#!/usr/bin/env python3
"""Independent manufactured-data audit of the cross-fitted procedure.

This program creates its own arrays. It never opens prepared production
data, production fitted states, terminal forecasts, or terminal targets.
The inherited raw geometry map is reused, but ridge fitting, preprocessing,
PCA/rank checks, selection arithmetic, blending, and policy joins below are
independent calculations. The receipt records this scope explicitly.
"""
from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_name] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.dont_write_bytecode = True
import crossfit_model as core
_pipeline_spec = importlib.util.spec_from_file_location("independent_crossfit_pipeline", ROOT / "code/pipeline.py")
pipeline = importlib.util.module_from_spec(_pipeline_spec)
sys.modules[_pipeline_spec.name] = pipeline
_pipeline_spec.loader.exec_module(pipeline)

SCALERS = ("standard", "minmax", "tanh_z_1")
CAPS = (1, 2, 4, 8, 12)
MODES = ("crossfit", "insample")
KINDS = ("kahm", "rbf")
SLOPES = tuple(f"{mode}_{kind}" for mode in MODES
               for kind in ("kahm", "rbf", "matched_rbf"))
NEW_METHODS = ("common", "shared_bias", "half_common", "half_shared_bias", *SLOPES)
LEGACY_METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common",
                  "kahm_departure", "rbf_departure", "kahm_unbounded",
                  "rbf_unbounded", "nonkahm_policy")
POLICY_ORDER = ("legacy__ridge", "legacy__raw_rbf", "legacy__raw_hgb",
                "legacy__persistence", "legacy__common", "legacy__rbf_departure",
                "legacy__rbf_unbounded", "crossfit__shared_bias",
                "crossfit__crossfit_rbf", "crossfit__insample_rbf")


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.pca_cache = {}

    def require(self, condition, label):
        self.counts["assertions"] += 1
        if not bool(condition):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        self.counts["assertions"] += 1
        np.testing.assert_array_equal(actual, expected, err_msg=label)

    def close(self, actual, expected, label, rtol=2e-9, atol=2e-10):
        self.counts["assertions"] += 1
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(a):
    a = np.ascontiguousarray(a)
    header = json.dumps({"dtype": str(a.dtype), "shape": list(a.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()


def transformed(x, scaler):
    if scaler == "minmax":
        low, span = x.min(axis=0), np.ptp(x, axis=0)
        z = 2.0 * (x - low) / np.where(span == 0.0, 1.0, span) - 1.0
        z[:, span == 0.0] = 0.0
        return z
    sd = x.std(axis=0, ddof=0)
    z = (x - x.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    return z if scaler == "standard" else np.tanh(z)


def bounded_features(raw, scaler):
    z = (raw - scaler["mean"]) / scaler["scale"]
    radius2 = z.shape[1]
    return z / np.sqrt(1.0 + np.sum(z * z, axis=1, keepdims=True) / radius2)


def raw_features(bank, x):
    # This audited extension changes fitting roles, not the inherited map.
    return core.fd.load_runtime()._raw_feature_maps(bank, x, {}, "independent_audit")[0]


def ridge_path(x, y, query, alphas, *, intercept):
    """Normal-equation oracle, independent of the implementation's SVD path."""
    mean, response_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mean, y - response_mean
    gram, rhs = xc.T @ xc, xc.T @ yc
    forecasts, coefficients = [], []
    for alpha in alphas:
        coefficient = np.linalg.solve(gram + len(x) * alpha * np.eye(x.shape[1]), rhs)
        forecast = (query - mean) @ coefficient
        if intercept:
            forecast = forecast + response_mean
        forecasts.append(forecast)
        coefficients.append(coefficient)
    return np.stack(forecasts), coefficients, mean, response_mean


def select_columns(values, choices):
    return np.column_stack([values[int(choice), :, h] for h, choice in enumerate(choices)])


def common_oracle(model, tx, ty, bx, by, alphas, floor, audit, tag):
    mean, scale = tx.mean(axis=0), np.maximum(tx.std(axis=0, ddof=0), floor)
    audit.exact(model.raw_scaler["mean"], mean, tag + " raw history mean")
    audit.exact(model.raw_scaler["scale"], scale, tag + " raw history scale")
    za, zb = (tx - mean) / scale, (bx - mean) / scale
    raw_b, raw_coefficients, raw_mean, raw_ymean = ridge_path(za, ty, zb, alphas, intercept=True)
    raw_losses = np.mean((by[None] - raw_b) ** 2, axis=1)
    raw_choice = np.argmin(raw_losses, axis=0)
    audit.exact(model.raw_head["alpha_by_horizon"], np.asarray(alphas)[raw_choice], tag + " raw alpha")
    raw_coef = np.column_stack([raw_coefficients[int(j)][:, h] for h, j in enumerate(raw_choice)])
    audit.close(model.raw_head["coef"], raw_coef, tag + " raw coefficient")
    baseline_a = raw_ymean + (za - raw_mean) @ raw_coef
    baseline_b = select_columns(raw_b, raw_choice)
    raw_a = raw_features(model.bank, tx)["common"]
    raw_b_features = raw_features(model.bank, bx)["common"]
    audit.exact(model.feature_scaler["mean"], raw_a.mean(axis=0), tag + " common feature mean")
    audit.exact(model.feature_scaler["scale"], np.maximum(raw_a.std(axis=0), floor), tag + " common feature scale")
    fa, fb = [bounded_features(value, model.feature_scaler) for value in (raw_a, raw_b_features)]
    path, coefficients, _, _ = ridge_path(fa, ty - baseline_a, fb, alphas, intercept=True)
    forecast_path = path + baseline_b[None]
    losses = np.mean((by[None] - forecast_path) ** 2, axis=1)
    choice = np.argmin(losses, axis=0)
    audit.exact(model.head["alpha_by_horizon"], np.asarray(alphas)[choice], tag + " common alpha")
    coef = np.column_stack([coefficients[int(j)][:, h] for h, j in enumerate(choice)])
    audit.close(model.head["coef"], coef, tag + " common coefficient")
    actual, _ = model.predict(bx)
    audit.close(actual, select_columns(forecast_path, choice), tag + " independent common forecast")
    audit.counts["common_models_with_independent_ridge_replay"] += 1


def native_basis(q, cap, audit):
    initial = min(cap, q.shape[1], len(q) - 1)
    key = array_digest(q)
    if key not in audit.pca_cache:
        centered = q.T - q.T.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh((centered @ centered.T) / (len(q) - 1))
        order = np.argsort(-values)
        positive = int(np.count_nonzero(values[order] > np.finfo(float).eps))
        audit.pca_cache[key] = vectors[:, order], positive
        audit.counts["independent_reference_eigendecompositions"] += 1
    vectors, positive = audit.pca_cache[key]
    if initial == 0 or positive == 0:
        return np.empty((q.shape[1], 0)), initial
    rank = initial
    while True:
        basis = vectors[:, :min(rank, positive)]
        if rank <= 1 or np.ptp(basis.T @ q.T, axis=1).min() >= 1e-3:
            return basis, initial
        rank -= 1


def audit_bank(tx, scaler, cap, records, basis_getter, audit, tag):
    z = transformed(tx, scaler)
    for i, record in enumerate(records):
        q, p, dimension = basis_getter(i)
        ids = np.asarray(record["reference_indices"], dtype=int)
        audit.require(np.all((ids >= 0) & (ids < len(tx))), tag + " reference rows belong to training subset")
        audit.exact(q, z[ids].astype(np.float32).astype(np.float64), tag + " subset-only transformed references")
        expected, initial = native_basis(q, cap, audit)
        audit.require(p.shape == expected.shape, tag + " actual native basis shape")
        audit.close(p @ p.T, expected @ expected.T, tag + " independent native PCA projector", rtol=2e-8, atol=2e-8)
        audit.require(dimension["initial_admissible_rank"] == initial, tag + " initial native dimension")
        audit.require(dimension["effective_rank"] == p.shape[1], tag + " stored native effective rank")
        audit.require(dimension["normal_direction_reserved"] is False, tag + " no normal reservation")
        audit.counts["subset_only_local_reference_and_basis_checks"] += 1
    audit.counts["subset_only_geometry_banks"] += 1


def common_selections(suite, groups, ax, ay, bx, by, model_config, audit):
    alphas = model_config["alpha_grid"]
    all_rows = np.arange(len(ax))
    fold = groups % 4
    audit.exact(suite.training_arrays["A_groups"], groups, "Original A group order retained")
    audit.exact(suite.training_arrays["A_fold_id"], fold, "Original episode IDs modulo four")
    audit.exact(suite.training_arrays["A_oof_assignment_count"], np.ones(len(ax), dtype=int), "OOF coverage exactly once")
    audit.exact(suite.training_arrays["A_oof_residual"], ay - suite.training_arrays["A_oof_common"], "OOF residual identity")
    audit.exact(suite.training_arrays["A_insample_residual"], ay - suite.training_arrays["A_full_common"], "In-sample residual identity")
    fits = []
    for k, row in enumerate(suite.selections["folds"]):
        train, held = all_rows[fold != k], all_rows[fold == k]
        audit.exact(row["training_row_indices"], train, "Complement row indices")
        audit.exact(row["heldout_row_indices"], held, "Held-out row indices")
        audit.exact(row["training_episode_ids"], np.unique(groups[train]), "Complement episode identities")
        audit.exact(row["heldout_episode_ids"], np.unique(groups[held]), "Held-out episode identities")
        audit.require(not set(groups[train]) & set(groups[held]), "No episode in both fold blocks")
        model = suite.fold_states[f"fold{k}"]
        forecast, _ = model.predict(ax[held])
        audit.exact(forecast, suite.training_arrays["A_oof_common"][held], "OOF prediction row-position join")
        fits.append((f"fold{k}", train, model, row))
        audit.counts["episode_folds"] += 1
    half_rows = all_rows[np.isin(groups, np.unique(groups)[:len(np.unique(groups)) // 2])]
    audit.exact(suite.selections["half_common"]["training_row_indices"], half_rows, "Half bridge is first half of episode IDs")
    fits.append(("half", half_rows, suite.half_common, suite.selections["half_common"]))
    for tag, rows, model, selection in fits:
        variance = np.maximum(ay[rows].var(axis=0), model_config["target_variance_floor"])
        audit.exact(selection["training_target_variance"], variance, tag + " subset-only target variance")
        audit.require(selection["common_gamma"] == 1.0, tag + " gamma-one common")
        scores = []
        expected_order = [f"{s}__cap{c}" for c in CAPS for s in SCALERS]
        audit.require(selection["candidate_order"] == expected_order, tag + " cap-first candidate ordering")
        for record in selection["B_scores"]:
            key = record["candidate_id"]
            pred = suite.validation_predictions[f"{tag}__{key}__common"]
            mse = np.mean((by - pred) ** 2, axis=0)
            score = float(np.mean(mse / variance))
            audit.close(record["mse_by_horizon"], mse, tag + " B MSE")
            audit.close(record["score"], score, tag + " B normalized score")
            losses = np.asarray([r["mse_by_horizon"] for r in record["head_selection"]["candidates"]])
            audit.exact(record["selected_alpha_by_horizon"], np.asarray(alphas)[losses.argmin(axis=0)], tag + " recorded alpha argmin")
            scores.append(score)
            diagnostic = suite.diagnostics["common_fits"][tag]["fit"]["banks"][key]
            dimensions = diagnostic["local_dimensions"]
            def get_basis(i):
                prefix = f"{tag}__{key}__local{i}"
                return (suite.basis_arrays[prefix + "__references"],
                        suite.basis_arrays[prefix + "__basis"], dimensions[i])
            audit_bank(ax[rows], record["scaler"], record["cap"],
                       diagnostic["bank"]["local_models"], get_basis, audit, tag + "/" + key)
            audit.counts["common_B_candidate_scores"] += 1
        choice = int(np.argmin(scores))
        audit.require(selection["selected_candidate"] == expected_order[choice], tag + " common B argmin")
        audit.require((model.scaler, model.cap) == (selection["selected_scaler"], selection["selected_cap"]), tag + " fitted selected identity")
        common_oracle(model, ax[rows], ay[rows], bx, by, alphas, model_config["feature_scale_floor"], audit, tag)
    common_oracle(suite.common, ax, ay, bx, by, alphas, model_config["feature_scale_floor"], audit, "full")


def departure_checks(reference, suite, ax, ay, bx, by, model_config, audit):
    alphas = np.asarray(model_config["alpha_grid"])
    full_common_a, _ = suite.common.predict(ax)
    full_common_b, _ = suite.common.predict(bx)
    audit.exact(suite.training_arrays["A_full_common"], full_common_a, "Full common training identity")
    audit.exact(suite.bias, np.mean(by - full_common_b, axis=0), "Shared bias is exact B mean residual")
    audit.exact(suite.training_arrays["B_common"], full_common_b, "B common identity")
    audit.exact(suite.training_arrays["B_shared_bias"], full_common_b + suite.bias, "B shared-bias identity")
    chosen = reference.candidates[reference.selections["selected"]["common"]]
    audit.exact(full_common_b, chosen.b_predictions["common"], "Deployment common is legacy gamma-one B selection")
    audit.require(suite.common.cap != 20, "Redundant cap-20 cannot win the tie")
    shared_b = full_common_b + suite.bias
    variance = np.maximum(ay.var(axis=0), model_config["target_variance_floor"])
    audit.exact(suite.selections["A_target_variance"], variance, "Departure full-A normalization")
    features = {}
    for cap in (1, 2, 4, 8, 12, 20):
        for scaler in SCALERS:
            key = f"{scaler}__cap{cap}"
            bank = reference.banks[key]
            def get_basis(i):
                local = bank.bank["models"][i]
                return local["q"], local["p"], local["dimension_audit"]
            audit_bank(ax, scaler, cap, bank.diagnostics["bank"]["local_models"], get_basis, audit, "full/" + key)
            if cap == 20:
                other = reference.banks[f"{scaler}__cap12"]
                for left, right in zip(bank.bank["models"], other.bank["models"]):
                    audit.exact(left["p"], right["p"], "Cap-12/cap-20 basis exact identity")
                    audit.exact(left["q"], right["q"], "Cap-12/cap-20 references exact identity")
                continue
            raw_a, raw_b = raw_features(bank.bank, ax), raw_features(bank.bank, bx)
            for kind in KINDS:
                fs = bank.scalers[kind]
                audit.exact(fs["mean"], raw_a[kind].mean(axis=0), "Full-A departure feature mean")
                audit.exact(fs["scale"], np.maximum(raw_a[kind].std(axis=0), model_config["feature_scale_floor"]), "Full-A departure feature scale")
                features[(scaler, cap, kind)] = tuple(bounded_features(r[kind], fs) for r in (raw_a, raw_b))
    expected_order = [f"{s}__cap{c}" for c in CAPS for s in SCALERS]
    audit.require(len(suite.departure_candidates) == 60, "Sixty target-mode/kind/geometry candidate heads")
    for mode in MODES:
        residual = ay - (suite.training_arrays["A_oof_common"] if mode == "crossfit" else full_common_a)
        for kind in KINDS:
            selection = suite.selections["target_modes"][mode][kind]
            audit.require(selection["candidate_order"] == expected_order, "Departure cap-first ordering")
            scores = []
            for record in selection["candidates"]:
                key = record["state_key"]
                state = suite.departure_candidates[key]
                primary_bank = reference.banks[f"{state.scaler}__cap{state.cap}"]
                audit.require(state.bank is primary_bank.bank, "Shared full-A geometry object")
                audit.require(state.feature_scaler is primary_bank.scalers[kind], "Shared full-A feature scaler object")
                fa, fb = features[(state.scaler, state.cap, kind)]
                predictions, coefficients, mean, ymean = ridge_path(fa, residual, fb, alphas, intercept=False)
                forecasts = predictions + shared_b[None]
                audit.close(suite.validation_predictions[key + "__alpha_paths"], forecasts, "Independent centered ridge paths")
                losses = np.mean((by[None] - forecasts) ** 2, axis=1)
                choice = losses.argmin(axis=0)
                audit.exact(state.head["alpha_by_horizon"], alphas[choice], "Independent departure alpha argmin")
                beta = np.column_stack([coefficients[int(j)][:, h] for h, j in enumerate(choice)])
                audit.close(state.head["coef"], beta, "Independent departure ridge coefficients")
                audit.close(state.head["mean"], mean, "Center bounded features after radial bound")
                audit.close(state.head["ymean"], ymean, "Centered residual response mean retained only as metadata")
                train_correction = (fa - mean) @ beta
                audit.close(train_correction.mean(axis=0), np.zeros(ay.shape[1]), "Slope-only correction has zero A mean")
                audit.close(state.mean_training_correction, train_correction.mean(axis=0), "Training correction diagnostic")
                selected_forecast = select_columns(forecasts, choice)
                audit.close(suite.validation_predictions[key], selected_forecast, "Selected slope forecast omits response intercept")
                mse = np.mean((by - selected_forecast) ** 2, axis=0)
                score = float(np.mean(mse / variance))
                audit.close(record["score"], score, "Departure normalized B score")
                audit.exact(record["selected_alpha_indices"], choice, "Saved departure alpha indices")
                scores.append(score)
                audit.counts["departure_candidates_independently_refitted"] += 1
            expected = expected_order[int(np.argmin(scores))]
            audit.require(selection["selected_candidate"] == expected, "Independent departure geometry argmin")
            audit.require(suite.departures[f"{mode}_{kind}"] is suite.departure_candidates[f"{mode}__{expected}__{kind}"], "Selected departure state join")
        kahm, matched = suite.departures[f"{mode}_kahm"], suite.departures[f"{mode}_matched_rbf"]
        audit.require((kahm.scaler, kahm.cap) == (matched.scaler, matched.cap), "Matched RBF geometry follows KAHM")
        audit.require(matched is suite.departure_candidates[f"{mode}__{kahm.scaler}__cap{kahm.cap}__rbf"], "Matched RBF keeps its own alpha")
    for cap in CAPS:
        for scaler in SCALERS:
            for kind in KINDS:
                left = suite.departure_candidates[f"crossfit__{scaler}__cap{cap}__{kind}"]
                right = suite.departure_candidates[f"insample__{scaler}__cap{cap}__{kind}"]
                audit.require(left.bank is right.bank and left.feature_scaler is right.feature_scaler,
                              "OOF and in-sample heads use identical fitted geometry and scaling")
                audit.exact(left.head["mean"], right.head["mean"], "OOF/in-sample bounded feature alignment")


def combine_arrays(legacy_arrays, new_result):
    arrays = {"legacy__" + key[len("primary__selected__"):]: value
              for key, value in legacy_arrays.items() if key.startswith("primary__selected__")}
    for mode, part in (("calibrated", "predictions"), ("eta1", "eta1_predictions")):
        arrays.update({f"crossfit__{mode}__{key}": value for key, value in new_result[part].items()})
    return arrays


def policy_array_key(identity, mode):
    family, method = identity.split("__", 1)
    actual = "calibrated" if mode == "calibrated" else ("gamma1" if family == "legacy" else "eta1")
    return f"{family}__{actual}__{method}"


def calibration_and_query(reference, suite, cx, cy, query, variance, cfg, audit):
    before = suite.frozen_hash()
    alternative = copy.deepcopy(suite)
    legacy_arrays, _ = reference.calibrate(cx, cy)
    audit.require(suite.frozen_hash() == before, "Legacy C calibration cannot change deployment common")
    result = suite.calibrate(cx, cy)
    alternative.calibrate(cx, cy * 31.0 + np.array([8.0, -9.0, 11.0]))
    audit.require(suite.frozen_hash() == alternative.frozen_hash() == before, "C targets only change correction etas")
    gamma = np.asarray(cfg["model"]["gamma_grid"])
    base = result["eta1_predictions"]["shared_bias"]
    for method in SLOPES:
        raw = result["eta1_predictions"][method]
        forecasts = base[None] + gamma[:, None, None] * (raw - base)[None]
        losses = np.mean((cy[None] - forecasts) ** 2, axis=1)
        choice = losses.argmin(axis=0)
        audit.exact(suite.etas[method], gamma[choice], "Independent C eta argmin")
        audit.close(result["predictions"][method], select_columns(forecasts, choice), "Independent C blended forecast")
        audit.counts["independent_C_eta_choices"] += 1
    arrays = combine_arrays(legacy_arrays, result)
    audit.require(tuple(cfg["policy_order"]) == POLICY_ORDER, "Frozen policy tie order")
    scores = [float(np.mean(np.mean((cy - arrays[policy_array_key(identity, "calibrated")]) ** 2, axis=0) / variance))
              for identity in POLICY_ORDER]
    policy = pipeline.choose_policy(arrays, cy, variance, list(POLICY_ORDER))
    audit.require(policy["selected_identity"] == POLICY_ORDER[int(np.argmin(scores))], "Expanded policy independent C argmin")
    for row, score in zip(policy["candidates"], scores):
        audit.close(row["score"], score, "Expanded policy candidate C score")
    pipeline.add_policy(arrays, policy)
    expected_keys = {f"legacy__{mode}__{method}" for mode in ("calibrated", "gamma1") for method in LEGACY_METHODS}
    expected_keys |= {f"crossfit__{mode}__{method}" for mode in ("calibrated", "eta1") for method in NEW_METHODS}
    expected_keys |= {f"policy__{mode}__expanded_nonkahm" for mode in ("calibrated", "eta1")}
    audit.require(set(arrays) == expected_keys == pipeline.expected_keys() and len(arrays) == 42, "Independent exact 42-array roster")
    for mode in ("calibrated", "eta1"):
        audit.exact(arrays[f"policy__{mode}__expanded_nonkahm"], arrays[policy_array_key(policy["selected_identity"], mode)], "Policy forwards same selected identity")
    # A deliberate tie and opposing eta-one ranking must not trigger reselection.
    fake = {}
    for index, identity in enumerate(POLICY_ORDER):
        fake[policy_array_key(identity, "calibrated")] = np.zeros_like(cy)
        fake[policy_array_key(identity, "eta1")] = np.full_like(cy, 100 - index)
    tie = pipeline.choose_policy(fake, np.zeros_like(cy), variance, list(POLICY_ORDER))
    audit.require(tie["selected_identity"] == POLICY_ORDER[0], "Policy first-minimum tie behavior")
    pipeline.add_policy(fake, tie)
    audit.exact(fake["policy__eta1__expanded_nonkahm"], fake[policy_array_key(POLICY_ORDER[0], "eta1")], "Eta-one cannot reselect policy")
    state_before = core.fd.state_hash(suite)
    predicted = suite.predict(query)
    audit.require(core.fd.state_hash(suite) == state_before, "Fixed-state query does not mutate suite")
    audit.require(list(inspect.signature(suite.predict).parameters) == ["x"], "Prediction API accepts only histories")
    audit.require(tuple(predicted["predictions"]) == NEW_METHODS, "Query method order")
    for value in predicted["predictions"].values():
        audit.require(value.shape == (len(query), 3) and np.isfinite(value).all(), "Finite query shape")
    zero = copy.deepcopy(suite)
    zero.etas = {method: np.zeros(3) for method in SLOPES}
    z = zero.predict(query)
    for method in SLOPES:
        audit.exact(z["predictions"][method], z["predictions"]["shared_bias"], "Eta-zero baseline exact identity")
        audit.require(predicted["diagnostics"]["bounds"][method]["query_bound_violations"] == 0, "Centered correction bound")
    audit.counts["retained_forecast_arrays"] = len(arrays)


def make_arrays():
    rng = np.random.default_rng(62810439)
    groups = np.repeat(np.arange(16, dtype=np.int64), 8)
    order = rng.permutation(len(groups))
    groups = groups[order]
    arrays = [rng.normal(size=(n, 12)) for n in (128, 64, 48, 31)]
    arrays = [a * np.linspace(0.4, 2.3, 12) + np.linspace(-1.0, 1.1, 12) for a in arrays]
    def target(x):
        return np.column_stack((0.4 * x[:, 0] + np.sin(x[:, 1]),
                                0.3 * x[:, 2] * x[:, 3] + 0.2 * x[:, 5],
                                np.cos(x[:, 4]) + 0.15 * x[:, 7] ** 2))
    ax, bx, cx, query = arrays
    ay, by, cy = [target(a) + 0.05 * rng.normal(size=(len(a), 3)) for a in (ax, bx, cx)]
    return ax, ay, groups, bx, by, cx, cy, query


def run():
    started = time.perf_counter()
    cfg = json.loads((ROOT / "config.json").read_text())
    model_config = cfg["model"]
    audit = Audit()
    audit.require(tuple(cfg["native_cap_order"]) == CAPS and tuple(cfg["scaler_order"]) == SCALERS, "Declared cap/scaler menu")
    audit.require(cfg["execution"]["output_array_count_per_task"] == 42, "Declared array count")
    ax, ay, groups, bx, by, cx, cy, query = make_arrays()
    seed = 815713
    with threadpool_limits(limits=1):
        reference = core.previous.fit_primary(ax, ay, bx, by, seed=seed, config=model_config)
        suite = core.fit_crossfit(ax, ay, groups, bx, by, seed=seed, config=model_config, reference_primary=reference)
        common_selections(suite, groups, ax, ay, bx, by, model_config, audit)
        departure_checks(reference, suite, ax, ay, bx, by, model_config, audit)
        # Refit the actual full procedure after changing only held-out fold 0.
        # Its own complement model, validation choices, and fixed-query output
        # must be identical even though the full-A deployment procedure changes.
        ax_changed, ay_changed = ax.copy(), ay.copy()
        mask = groups % 4 == 0
        ax_changed[mask] = ax_changed[mask] * 1.4 + 2.0
        ay_changed[mask] = ay_changed[mask] * -3.0 + np.array([7.0, -11.0, 4.0])
        changed_reference = core.previous.fit_primary(ax_changed, ay_changed, bx, by, seed=seed, config=model_config)
        changed = core.fit_crossfit(ax_changed, ay_changed, groups, bx, by, seed=seed, config=model_config, reference_primary=changed_reference)
        left, right = suite.fold_states["fold0"], changed.fold_states["fold0"]
        audit.require(left.state_hash() == right.state_hash(), "Excluded fold histories and outcomes cannot change complement state")
        audit.require(suite.selections["folds"][0]["B_scores"] == changed.selections["folds"][0]["B_scores"], "Excluded fold cannot change common candidate selection")
        audit.exact(left.predict(query)[0], right.predict(query)[0], "Excluded-fold perturbation leaves common predictions unchanged")
        audit.require(suite.common.state_hash() != changed.common.state_hash(), "Perturbation reaches full-A deployment fit (nonvacuous probe)")
        audit.counts["excluded_fold_perturbation_refits"] = 1
        variance = np.maximum(ay.var(axis=0), model_config["target_variance_floor"])
        calibration_and_query(reference, suite, cx, cy, query, variance, cfg, audit)
    paths = [ROOT / "code/crossfit_model.py", ROOT / "code/pipeline.py", ROOT / "config.json", Path(__file__)]
    return {
        "status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Manufactured arrays only; no production data, fits, forecasts, or D targets read",
        "fixture": {"seed": 62810439, "fit_seed": seed, "episodes": 16, "A_rows": len(ax),
                    "B_rows": len(bx), "C_rows": len(cx), "query_rows": len(query),
                    "history": 12, "horizons": 3, "production_model_grid_unchanged": True,
                    "group_rows_shuffled": True},
        "independence_scope": "Independent normal-equation ridge, preprocessing, native PCA/rank, selection, calibration and policy arithmetic; inherited raw feature map reused",
        "counts": dict(audit.counts),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "wall_seconds": time.perf_counter() - started,
        "checks": {"whole_episode_OOF_coverage": True, "excluded_fold_state_invariance": True,
                   "complement_only_target_normalizers": True, "independent_ridge_and_B_selection": True,
                   "subset_only_reference_geometry": True, "full_A_feature_alignment": True,
                   "gamma_one_deployment_common": True, "fixed_shared_B_bias": True,
                   "slope_only_centered_corrections": True, "C_state_isolation": True,
                   "expanded_policy_identity_and_ties": True, "eta_zero_exact_recovery": True,
                   "target_free_query_without_state_mutation": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness/independent_crossfit_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior audit receipts; choose a new output path: " + str(args.output))
    try:
        receipt = run()
    except Exception:
        receipt = {"status": "failed", "scope": "Manufactured-data-only independent audit",
                   "completed_utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
