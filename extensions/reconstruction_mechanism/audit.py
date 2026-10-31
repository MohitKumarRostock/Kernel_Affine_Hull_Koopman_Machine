"""Independent implementation audit of frozen inputs, states, and results.

This is internal verification, not external replication. No study fitting,
prediction, feature, input-preparation, or analysis function is imported or
called. Selected coefficients are checked by their normal equations without
refitting. Discarded B-candidate coefficients were not retained: their recorded
score argmins can be checked, but those discarded scores cannot be independently
replayed without a prohibited refit.

Run after the study completes, optionally with --output /tmp/audit.json to leave
the archived audit unchanged. --task-id verifies a labeled incomplete subset.
"""
from __future__ import annotations

import os
for _audit_thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_audit_thread_variable] = "1"

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np
from scipy.special import stdtr, stdtrit

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
SCHEMA = "reconstruction-mechanism-independent-audit-v1"
METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "kahm_departure",
           "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy")
GEOMETRY = ("common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
POLICY = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
SPLITS = ("A", "B", "C", "D")
# Independent direct-distance and tree evaluation need not reproduce every
# floating-point operation of the study code. Tolerances are fixed here before
# the full audit; actual maximum discrepancies are reported.
RTOL, ATOL = 1e-8, 2e-10


class InertState:
    """Only the saved attributes of the two study classes are deserialized."""


class StateReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "extensions.reconstruction_mechanism.models" and name in {
                "ForecastSuite", "ReconstructionFeatures"}:
            return InertState
        return super().find_class(module, name)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    prefix = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(prefix + b"\0" + value.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class Checks:
    def __init__(self, task):
        self.task = task
        self.count = 0
        self.maximum_abs_prediction_difference = 0.0
        self.maximum_scaled_normal_equation_residual = 0.0
        self.ridge_vectors = 0
        self.selectors = 0

    def require(self, condition, name):
        self.count += 1
        if not condition:
            raise AssertionError(f"{self.task}: {name}")

    def equal(self, actual, expected, name):
        self.require(np.array_equal(np.asarray(actual), np.asarray(expected)), name)

    def close(self, actual, expected, name, *, rtol=RTOL, atol=ATOL):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape, name + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), name + " finite")
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            discrepancy = float(np.max(np.abs(a - b)))
            raise AssertionError(f"{self.task}: {name}: maximum absolute difference {discrepancy:.17g}")
        self.count += 1


def seed(parts):
    return int(np.random.SeedSequence([int(value) for value in parts]).generate_state(1)[0])


def verify_windows(arrays, metadata, config, checks):
    history, horizons = config["history"], config["horizons"]
    allowed = []
    origin = history - 1
    while origin + max(horizons) < config["episode_length"]:
        allowed.append(origin)
        origin += config["stride"]
    total = 0
    amplitude = float(np.std(arrays["A_base_episodes"], ddof=0))
    checks.close(arrays["A_base_episode_std"], amplitude, "A-only noise amplitude")
    checks.close(arrays["noise_std"], amplitude * metadata["noise_level"], "A-only noise standard deviation")
    for split_index, split in enumerate(SPLITS):
        record = metadata["splits"][split]
        episodes = arrays[f"{split}_episodes"]
        base, noise = arrays[f"{split}_base_episodes"], arrays[f"{split}_unit_noise"]
        rows = arrays[f"{split}_row_indices"]
        checks.require(episodes.shape == (config["split_episodes"][split], config["episode_length"]),
                       f"{split} episode shape")
        checks.close(episodes, base + amplitude * metadata["noise_level"] * noise, f"{split} observation noise")
        expected_noise = np.random.default_rng(seed(record["unit_noise_seed_components"])).standard_normal(base.shape)
        checks.equal(noise, expected_noise, f"{split} unit noise replay")
        checks.require(array_sha(base) == record["base_episode_sha256"], f"{split} base checksum")
        checks.require(array_sha(noise) == record["unit_noise_sha256"], f"{split} noise checksum")
        full_rows = len(episodes) * len(allowed)
        cap = config["split_caps"][split]
        expected_rows = np.arange(full_rows, dtype=np.int64)
        if cap is not None and cap < full_rows:
            expected_rows = np.sort(np.random.default_rng(seed(record["cap_seed_components"])).choice(
                full_rows, cap, replace=False))
        checks.equal(rows, expected_rows, f"{split} row cap")
        checks.require(array_sha(rows) == record["retained_row_indices_sha256"], f"{split} retained checksum")
        expected_x = np.empty((len(rows), history))
        expected_y = np.empty((len(rows), len(horizons)))
        groups, origins = np.empty(len(rows), dtype=np.int64), np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            episode, local = divmod(int(row), len(allowed))
            origin = allowed[local]
            groups[i], origins[i] = episode, origin
            for lag in range(history):
                expected_x[i, lag] = episodes[episode, origin - history + 1 + lag]
            for h, offset in enumerate(horizons):
                expected_y[i, h] = episodes[episode, origin + offset]
        checks.equal(arrays[f"{split}_X"], expected_x, f"{split} scalar history indexing")
        checks.equal(arrays[f"{split}_Y"], expected_y, f"{split} scalar target indexing")
        checks.equal(arrays[f"{split}_groups"], groups, f"{split} scalar groups")
        checks.equal(arrays[f"{split}_origins"], origins, f"{split} scalar origins")
        total += len(rows)
    loss_scale = np.maximum(np.var(arrays["A_Y"], axis=0, ddof=0), config["analysis"]["variance_floor"])
    checks.close(arrays["loss_scale"], loss_scale, "A-only loss scale")
    checks.close(metadata["loss_scale"], loss_scale, "manifest loss scale")
    return total, loss_scale


def verify_stream_domains(tasks, config):
    """Check configured seed domains, including accidental 32-bit collisions."""
    streams = {}
    for task in tasks:
        if task["noise_index"] != 0:
            continue
        law_index = config["laws"].index(task["law"])
        for split_index, split in enumerate(SPLITS):
            record = task["splits"][split]
            for prefix, salt_name in (("generation", "generation"), ("unit_noise", "unit_noise"), ("cap", "caps")):
                parts = [task["seed"], law_index, split_index, config["seed_salts"][salt_name]]
                if record[prefix + "_seed_components"] != parts or record[prefix + "_seed"] != seed(parts):
                    raise AssertionError("Recorded random stream differs from its configured domain")
                streams[tuple(parts)] = seed(parts)
        parts = [task["seed"], law_index, config["seed_salts"]["fit"]]
        if task["fit_seed_components"] != parts or task["fit_seed"] != seed(parts):
            raise AssertionError("Fitting random stream differs from its configured domain")
        streams[tuple(parts)] = seed(parts)
    if len(set(streams.values())) != len(streams):
        raise AssertionError("Distinct configured stream domains share a generated seed")
    return len(streams)


def standardize(state, x):
    return (x - np.asarray(state["mean"])) / np.asarray(state["scale"])


def check_scaler(state, fitting, floor, checks, label, *, inherited=False):
    mean, deviation = np.mean(fitting, axis=0), np.std(fitting, axis=0, ddof=0)
    scale = np.where(deviation > 1e-12, deviation, 1.0) if inherited else np.maximum(deviation, floor)
    checks.close(state["mean"], mean, label + " A mean")
    checks.close(state["scale"], scale, label + " A scale")


def radial(v):
    # Inputs in this completed study are finite and moderate. Use the displayed
    # closed form, independently of the study's overflow-safe implementation.
    width = v.shape[1]
    return v / np.sqrt(1.0 + np.sum(v * v, axis=1, keepdims=True) / width)


def geometry_features(state, x, a_slice, config, checks):
    bank = state.bank
    check_scaler(bank, x[a_slice], 1e-12, checks, "bank history", inherited=True)
    z = standardize(bank, x)
    coordinates, affine, logits = [], [], []
    for model in bank["models"]:
        difference = z - model["center"]
        coordinate = difference @ model["p"]
        coordinates.append(coordinate)
        affine.append(model["center"] + coordinate @ model["p"].T)
        logits.append(-np.sum(difference * difference, axis=1) / (2.0 * model["radius"]))
    logits = np.column_stack(logits)
    gates = np.empty_like(logits)
    for left, right in bank["block_slices"]:
        values = logits[:, left:right]
        weights = np.exp(values - np.max(values, axis=1, keepdims=True))
        gates[:, left:right] = weights / np.sum(weights, axis=1, keepdims=True)
    common = np.column_stack([gates] + [gates[:, j:j + 1] * coordinate
                                      for j, coordinate in enumerate(coordinates)])
    pieces = {"kahm": [], "rbf": []}
    fallback_rows = {"kahm": set(), "rbf": set()}
    maximum_inverse_error = 0.0
    for j, model in enumerate(bank["models"]):
        if not model["rank"]:
            for kind in pieces:
                pieces[kind].append(np.zeros_like(affine[j]))
            continue
        if "whitening" not in model:
            for kind in pieces:
                pieces[kind].append(np.zeros_like(affine[j]))
                fallback_rows[kind].update(range(len(x)))
            continue
        query = z @ model["p"] @ model["whitening"].T
        references = model["q"] @ model["p"] @ model["whitening"].T
        checks.close(references, model["uq"], f"bank {j} projected references")
        covariance = np.atleast_2d(np.cov(model["q"] @ model["p"], rowvar=False, ddof=1))
        checks.close(model["whitening"] @ covariance @ model["whitening"].T,
                     np.eye(model["rank"]), f"bank {j} projected covariance metric")
        difference = query[:, None, :] - references[None, :, :]
        logs = -np.sum(difference * difference, axis=2) / (2 * model["rank"])
        kernel = np.exp(logs - logs.max(axis=1, keepdims=True))
        if "inverse" in model:
            differences = references[:, None, :] - references[None, :, :]
            gram = np.exp(-np.sum(differences * differences, axis=2) / (2 * model["rank"]))
            identity = (gram + model["lambda_original"] * np.eye(len(gram))) @ model["inverse"]
            checks.close(identity, np.eye(len(gram)), f"bank {j} regularized inverse")
            maximum_inverse_error = max(maximum_inverse_error, float(np.max(np.abs(identity - np.eye(len(gram))))))
        for kind in pieces:
            if kind == "kahm" and "inverse" not in model:
                pieces[kind].append(np.zeros_like(affine[j]))
                fallback_rows[kind].update(range(len(x)))
                continue
            weights = kernel @ model["inverse"] if kind == "kahm" else kernel
            denominator = weights.sum(axis=1)
            ratio = np.abs(denominator) / np.abs(weights).sum(axis=1)
            safe = np.isfinite(weights).all(axis=1) & (ratio > config["relative_denominator_tolerance"])
            estimate = affine[j].copy()
            proposal = (weights[safe] @ model["q"]) / denominator[safe, None]
            finite = np.isfinite(proposal).all(axis=1)
            safe_ids = np.flatnonzero(safe)
            safe[safe_ids[~finite]] = False
            estimate[safe_ids[finite]] = proposal[finite]
            fallback_rows[kind].update(np.flatnonzero(~safe).tolist())
            pieces[kind].append(gates[:, j:j + 1] * (estimate - affine[j]))
    raw = {"common": common, **{kind: np.column_stack(value) for kind, value in pieces.items()}}
    scaled, capped = {}, {}
    for kind, value in raw.items():
        check_scaler(state.scalers[kind], value[a_slice], config["feature_scale_floor"], checks, kind + " feature")
        scaled[kind] = standardize(state.scalers[kind], value)
        capped[kind] = radial(scaled[kind])
    features = {"common": capped["common"]}
    for kind in pieces:
        features[kind + "_departure"] = np.column_stack((capped["common"], capped[kind]))
        features[kind + "_unbounded"] = np.column_stack((capped["common"], scaled[kind]))
    for method, matrix in features.items():
        checks.require(np.isfinite(matrix).all(), method + " reconstructed finite features")
        checks.equal(matrix[:, :common.shape[1]], capped["common"], method + " identical common block")
        if not method.endswith("unbounded"):
            checks.require(np.max(np.linalg.norm(matrix, axis=1)) <= np.sqrt(matrix.shape[1]) * (1 + 1e-12),
                           method + " feature norm bound")
    return features, {"maximum_reference_inverse_identity_error": maximum_inverse_error,
                      "fallback_rows": {kind: sorted(value) for kind, value in fallback_rows.items()}}


def forward(head, feature):
    return np.asarray(head["ymean"]) + (feature - np.asarray(head["mean"])) @ np.asarray(head["coef"])


def check_head(head, features, targets, checks, label):
    mean, response_mean = features.mean(axis=0), targets.mean(axis=0)
    xc, yc = features - mean, targets - response_mean
    coefficient, alphas = np.asarray(head["coef"]), np.asarray(head["alpha_by_horizon"])
    checks.close(head["mean"], mean, label + " feature mean")
    checks.close(head["ymean"], response_mean, label + " unpenalized intercept")
    checks.require(np.all(alphas > 0), label + " positive penalties")
    gradient = xc.T @ (xc @ coefficient - yc) / len(features) + coefficient * alphas[None, :]
    normalization = 1.0 + np.linalg.norm(xc) * np.linalg.norm(yc) / len(features) + np.linalg.norm(coefficient * alphas)
    scaled_residual = float(np.linalg.norm(gradient) / normalization)
    checks.require(scaled_residual < 1e-9, label + " mean-SSE normal equations")
    checks.maximum_scaled_normal_equation_residual = max(checks.maximum_scaled_normal_equation_residual, scaled_residual)
    rms = np.sqrt(np.mean(yc * yc, axis=0))
    checks.close(head["response_rms"], rms, label + " centered-response RMS")
    coefficient_bound = rms / (2 * np.sqrt(alphas))
    checks.require(np.all(np.linalg.norm(coefficient, axis=0) <= coefficient_bound * (1 + 1e-9) + 1e-11),
                   label + " sharp coefficient bound")
    checks.ridge_vectors += targets.shape[1]


def rbf_features(state, z):
    differences = z[:, None, :] - state["landmarks"][None, :, :]
    kernel = np.exp(-np.sum(differences * differences, axis=2) / (2 * state["bandwidth_squared"]))
    raw = np.column_stack((z, kernel @ state["transform"]))
    return standardize(state["scaler"], raw), raw


def tree_forward(model, x, checks, label):
    """Direct numeric split traversal; no sklearn prediction API is called."""
    checks.require(model._preprocessor is None, label + " no hidden preprocessing")
    result = np.full(len(x), float(np.asarray(model._baseline_prediction).ravel()[0]))
    for iteration in model._predictors:
        checks.require(len(iteration) == 1, label + " scalar boosting tree")
        nodes = iteration[0].nodes
        checks.require(not np.any(nodes["is_categorical"]), label + " numeric tree splits")
        positions = np.zeros(len(x), dtype=np.int64)
        active = np.arange(len(x), dtype=np.int64)
        for _ in range(len(nodes)):
            if len(active) == 0:
                break
            index = positions[active]
            leaves = nodes["is_leaf"][index].astype(bool)
            if np.any(leaves):
                result[active[leaves]] += nodes["value"][index[leaves]]
            active, index = active[~leaves], index[~leaves]
            if len(active):
                values = x[active, nodes["feature_idx"][index]]
                left = values <= nodes["num_threshold"][index]
                positions[active] = np.where(left, nodes["left"][index], nodes["right"][index])
        checks.require(len(active) == 0, label + " terminating tree traversal")
    return result


def mse(truth, prediction):
    return np.mean((truth - prediction) ** 2, axis=0)


def blend(baseline, candidate, gamma):
    result = baseline.copy()
    for h, multiplier in enumerate(gamma):
        if multiplier != 0:
            result[:, h] = baseline[:, h] + multiplier * (candidate[:, h] - baseline[:, h])
    return result


def check_B(record, observed, checks, label, *, head=None, states=None):
    if record.get("status") == "failed":
        return
    scores = np.array([[np.inf if value is None else value for value in candidate["mse_by_horizon"]]
                       if candidate.get("mse_by_horizon") is not None else [np.inf] * len(observed)
                       for candidate in record["candidates"]])
    chosen = np.argmin(scores, axis=0)
    checks.equal([entry["candidate_index"] for entry in record["selected"]], chosen, label + " B table argmin")
    checks.close(observed, [scores[index, h] for h, index in enumerate(chosen)], label + " selected B loss replay")
    if head is not None:
        checks.close(head["alpha_by_horizon"], [record["candidates"][i]["alpha"] for i in chosen],
                     label + " selected penalty")
    if states is not None:
        checks.equal([value["candidate_index"] for value in states], chosen, label + " selected state identities")
    checks.selectors += len(chosen)


def task_audit(study, task, config):
    checks = Checks(task["task_id"])
    output = study / "results" / task["task_id"]
    summary = read_json(output / "summary.json")
    checks.require(summary["status"] == "completed", "completed task")
    checks.require(not summary["development_only"], "reserved scope")
    checks.require(summary["freeze_sha256"] == sha(study / "freeze.json"), "task freeze identity")
    checks.require(summary["input_sha256"] == task["sha256"], "input identity")
    input_path = study / task["input"]
    checks.require(sha(input_path) == task["sha256"], "prepared file checksum")
    for filename, expected in summary["files"].items():
        checks.require(sha(output / filename) == expected, filename + " checksum")
    with np.load(input_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    for name, expected in task["array_sha256"].items():
        checks.require(array_sha(arrays[name]) == expected, name + " array checksum")
    windows, loss_scale = verify_windows(arrays, task, config, checks)
    # These are trusted states produced locally by the frozen study. The two
    # study classes are replaced with inert containers at deserialization.
    suite = StateReader(io.BytesIO(gzip.decompress((output / "fitted_suite.pkl.gz").read_bytes()))).load()
    checks.require("extensions.reconstruction_mechanism.models" not in sys.modules, "study prediction module not imported")
    checks.require(suite.config == config["model"], "saved model configuration")
    diagnostics = summary["fit_diagnostics"]
    slices, left = {}, 0
    for split in SPLITS:
        slices[split] = slice(left, left + len(arrays[split + "_X"]))
        left = slices[split].stop
    x = np.vstack([arrays[split + "_X"] for split in SPLITS])
    a, b, c, d = [slices[split] for split in SPLITS]
    ya, yb, yc = [arrays[split + "_Y"] for split in ("A", "B", "C")]
    check_scaler(suite.input_scaler, x[a], config["model"]["feature_scale_floor"], checks, "raw input")
    z = standardize(suite.input_scaler, x)
    check_head(suite.baseline, z[a], ya, checks, "baseline")
    baseline = forward(suite.baseline, z)
    candidate = {"ridge": baseline, "persistence": np.repeat(x[:, -1:], ya.shape[1], axis=1)}
    check_B(diagnostics["B"]["ridge"], mse(yb, baseline[b]), checks, "ridge", head=suite.baseline)
    features, feature_record = {}, {}
    if suite.geometry is not None:
        features, feature_record = geometry_features(suite.geometry, x, a, config["model"], checks)
        for kind, combined_rows in feature_record["fallback_rows"].items():
            for split in SPLITS:
                bounds = slices[split]
                expected_rows = [row - bounds.start for row in combined_rows if bounds.start <= row < bounds.stop]
                recorded = (summary["prediction_diagnostics"] if split == "D"
                            else diagnostics["features_A_B_C"][split])
                checks.equal(recorded[kind + "_fallback_rows"], expected_rows, f"{kind} {split} fallback rows")
    for method in GEOMETRY:
        if method not in suite.heads or method not in features:
            checks.require(method in diagnostics["method_failures"], method + " declared fit failure")
            candidate[method] = baseline.copy()
            continue
        check_head(suite.heads[method], features[method][a], ya - baseline[a], checks, method)
        candidate[method] = baseline + forward(suite.heads[method], features[method])
        check_B(diagnostics["B"][method], mse(yb, candidate[method][b]), checks, method, head=suite.heads[method])
    if suite.raw_rbf is None:
        checks.require("raw_rbf" in diagnostics["method_failures"], "raw RBF declared failure")
        candidate["raw_rbf"] = baseline.copy()
    else:
        columns, cache = [], {}
        for h, selected in enumerate(suite.raw_rbf):
            state = selected["features"]
            key = state["bandwidth_factor"]
            if key not in cache:
                cache[key], raw = rbf_features(state, z)
                check_scaler(state["scaler"], raw[a], config["model"]["feature_scale_floor"], checks, f"RBF factor {key}")
                indices = diagnostics["B"]["raw_rbf"]["landmark_indices"]
                checks.equal(state["landmarks"], z[a][indices], f"RBF factor {key} A landmarks")
            feature = cache[key]
            check_head(selected["head"], feature[a], ya[:, h:h + 1], checks, f"raw RBF horizon {h}")
            columns.append(forward(selected["head"], feature)[:, 0])
            chosen = diagnostics["B"]["raw_rbf"]["selected"][h]
            checks.close(state["bandwidth_factor"], chosen["bandwidth_factor"], f"raw RBF horizon {h} factor")
            checks.close(selected["head"]["alpha_by_horizon"], [chosen["alpha"]], f"raw RBF horizon {h} alpha")
        candidate["raw_rbf"] = np.column_stack(columns)
        check_B(diagnostics["B"]["raw_rbf"], mse(yb, candidate["raw_rbf"][b]), checks, "raw_rbf", states=suite.raw_rbf)
    if suite.raw_hgb is None:
        checks.require("raw_hgb" in diagnostics["method_failures"], "HGB declared failure")
        candidate["raw_hgb"] = baseline.copy()
    else:
        columns = []
        for h, model in enumerate(suite.raw_hgb):
            selected = diagnostics["B"]["raw_hgb"]["selected"][h]
            checks.require(model.learning_rate == selected["learning_rate"] and model.max_leaf_nodes == selected["max_leaf_nodes"],
                           f"HGB horizon {h} selected configuration")
            checks.require(not model.early_stopping and len(model._predictors) == config["model"]["raw_hgb_max_iter"],
                           f"HGB horizon {h} fixed iterations")
            checks.close(np.asarray(model._baseline_prediction).ravel(), [ya[:, h].mean()], f"HGB horizon {h} initial constant")
            columns.append(tree_forward(model, x, checks, f"HGB horizon {h}"))
        candidate["raw_hgb"] = np.column_stack(columns)
        check_B(diagnostics["B"]["raw_hgb"], mse(yb, candidate["raw_hgb"][b]), checks, "raw_hgb")
    prediction = {"ridge": baseline}
    gamma_grid = np.asarray(config["model"]["gamma_grid"])
    for method in METHODS[1:-1]:
        score = np.array([mse(yc, blend(baseline[c], candidate[method][c], np.full(ya.shape[1], multiplier)))
                          for multiplier in gamma_grid])
        chosen = np.argmin(score, axis=0)
        record = diagnostics["C"][method]
        checks.close([item["mse_by_horizon"] for item in record["candidates"]], score, method + " C candidate losses")
        checks.equal(record["selected_indices"], chosen, method + " C gamma argmin")
        checks.equal(suite.gammas[method], gamma_grid[chosen], method + " deployed gamma")
        prediction[method] = blend(baseline, candidate[method], suite.gammas[method])
        checks.close(record["selected_mse_by_horizon"], mse(yc, prediction[method][c]), method + " C calibrated loss")
        checks.selectors += ya.shape[1]
    policy_scores = [float(np.mean(mse(yc, prediction[method][c]) / loss_scale)) for method in POLICY]
    checks.equal([row["method"] for row in diagnostics["policy"]["candidates"]], POLICY, "policy candidate order")
    checks.close([row["normalized_mse"] for row in diagnostics["policy"]["candidates"]], policy_scores, "policy C scores")
    selected_policy = POLICY[int(np.argmin(policy_scores))]
    checks.require(suite.selected_policy == selected_policy == diagnostics["policy"]["selected_method"], "C-only policy selection")
    prediction["nonkahm_policy"] = prediction[selected_policy].copy()
    checks.selectors += 1
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        checks.equal(saved["Y"], arrays["D_Y"], "saved target values")
        checks.require(array_sha(saved["Y"]) == summary["target_sha256"], "target checksum")
        checks.equal(saved["nonkahm_policy"], saved[selected_policy], "saved policy identity")
        losses, raw_mse = {}, {}
        for method in METHODS:
            checks.require(array_sha(saved[method]) == summary["prediction_array_sha256"][method], method + " saved checksum")
            checks.close(prediction[method][d], saved[method], method + " independent D prediction")
            difference = float(np.max(np.abs(prediction[method][d] - saved[method])))
            checks.maximum_abs_prediction_difference = max(checks.maximum_abs_prediction_difference, difference)
            raw_mse[method] = mse(saved["Y"], saved[method]).tolist()
            losses[method] = float(np.mean(np.asarray(raw_mse[method]) / loss_scale))
            if method in suite.heads and not method.endswith("unbounded"):
                head = suite.heads[method]
                response = ya - baseline[a]
                rms = np.sqrt(np.mean((response - response.mean(axis=0)) ** 2, axis=0))
                coefficient_bound = rms / (2 * np.sqrt(head["alpha_by_horizon"]))
                radius = np.sqrt(features[method].shape[1])
                bound = suite.gammas[method] * (np.abs(response.mean(axis=0)) +
                        (radius + np.linalg.norm(features[method][a].mean(axis=0))) * coefficient_bound)
                checks.close(summary["prediction_diagnostics"]["bounds"][method]["correction_bound_by_horizon"],
                             bound, method + " recorded correction bound")
                checks.require(np.all(np.max(np.abs(saved[method] - saved["ridge"]), axis=0) <= bound * (1 + 1e-8) + 1e-10),
                               method + " actual D correction bound")
        errors = {method: float(np.max(np.abs(saved["Y"] - saved[method]))) for method in ("ridge", "kahm_departure")}
    return {
        "task_id": task["task_id"], "seed": task["seed"], "law": task["law"], "noise_index": task["noise_index"],
        "checks": checks.count, "windows_verified": windows,
        "ridge_coefficient_vectors_verified": checks.ridge_vectors,
        "selectors_verified": checks.selectors, "method_predictions_verified": len(METHODS),
        "scored_cells_per_method": int(arrays["D_Y"].size),
        "maximum_abs_prediction_difference": checks.maximum_abs_prediction_difference,
        "maximum_scaled_normal_equation_residual": checks.maximum_scaled_normal_equation_residual,
        "normalized_losses": losses, "raw_mse_by_horizon": raw_mse,
        "linear_or_nonlinear_scope": "linear" if task["law"] == "linear_ar" else "nonlinear",
        "selected_policy": selected_policy, "maximum_absolute_errors": errors,
        "catastrophic_error_screen_passed": errors["kahm_departure"] <= config["analysis"]["max_error_ratio"] * errors["ridge"],
        "feature_checks": feature_record,
        "fitted_method_failures": diagnostics["method_failures"],
        "prediction_recovered_cells": summary["prediction_diagnostics"]["recovered_prediction_cells"],
        "result_files": summary["files"], "summary_sha256": sha(output / "summary.json"),
    }


def primary_statistics(records, config):
    repetition = []
    for repetition_seed in config["seeds"]:
        tasks = [record for record in records if record["seed"] == repetition_seed]
        row = {"seed": repetition_seed}
        for scope in ("nonlinear", "linear"):
            subset = [record for record in tasks if record["linear_or_nonlinear_scope"] == scope]
            expected = len(config["noise_levels"]) * (len(config["nonlinear_laws"]) if scope == "nonlinear" else 1)
            if len(subset) != expected:
                raise AssertionError("Incomplete independent repetition aggregate")
            row[scope] = {method: float(np.mean([record["normalized_losses"][method] for record in subset]))
                          for method in METHODS}
        repetition.append(row)
    n = len(repetition)
    family_alpha = config["analysis"]["family_alpha"]
    family_size = len(config["analysis"]["primary_comparators"]) + 1
    critical = float(stdtrit(n - 1, 1.0 - family_alpha / family_size))
    comparisons = [("nonlinear", method, 1.0 - config["analysis"]["superiority_margin"], "useful_superiority")
                   for method in config["analysis"]["primary_comparators"]]
    comparisons.append(("linear", "ridge", 1.0 + config["analysis"]["linear_noninferiority_margin"], "noninferiority"))
    contrasts = []
    for scope, method, multiplier, kind in comparisons:
        comparator = np.asarray([row[scope][method] for row in repetition])
        candidate = np.asarray([row[scope]["kahm_departure"] for row in repetition])
        gap = multiplier * comparator - candidate
        mean = float(np.mean(gap))
        deviation = float(np.std(gap, ddof=1))
        se = deviation / math.sqrt(n)
        lower = mean - critical * se
        p_one_sided = float(stdtr(n - 1, -mean / se)) if se > 0 else (0.0 if mean > 0 else 1.0 if mean < 0 else 0.5)
        contrasts.append({"scope": scope, "comparator": method, "kind": kind,
                          "candidate": "kahm_departure", "comparator_multiplier": multiplier,
                          "repetitions": n, "gap_values": gap.tolist(), "mean_gap": mean,
                          "sample_standard_deviation": deviation, "standard_error": se,
                          "simultaneous_one_sided_lower_bound": lower, "strictly_positive": lower > 0,
                          "candidate_mean_normalized_loss": float(candidate.mean()),
                          "comparator_mean_normalized_loss": float(comparator.mean()),
                          "point_relative_gain": float(1 - candidate.mean() / comparator.mean()),
                          "positive_margin_repetitions": int(np.count_nonzero(gap > 0)),
                          "p_one_sided": p_one_sided,
                          "p_bonferroni": min(1.0, family_size * p_one_sided),
                          "critical_t": critical, "degrees_of_freedom": n - 1,
                          "one_sided_family_alpha": family_alpha, "family_size": family_size})
    return repetition, contrasts


def compare_analysis(contrasts, path):
    """Read the published artifact only; no analyzer code is used."""
    decision = read_json(path)
    published = decision["primary_contrasts"]
    if len(published) != len(contrasts):
        raise AssertionError("Published primary-contrast count differs")
    mapping = {
        "n_repetitions": "repetitions", "df": "degrees_of_freedom",
        "comparator_factor": "comparator_multiplier",
        "candidate_mean_nmse": "candidate_mean_normalized_loss",
        "comparator_mean_nmse": "comparator_mean_normalized_loss",
        "point_relative_gain": "point_relative_gain", "margin_gap_mean": "mean_gap",
        "margin_gap_sample_sd": "sample_standard_deviation", "margin_gap_se": "standard_error",
        "critical_t": "critical_t", "simultaneous_one_sided_lower": "simultaneous_one_sided_lower_bound",
        "p_one_sided": "p_one_sided", "p_bonferroni": "p_bonferroni",
        "positive_margin_repetitions": "positive_margin_repetitions",
    }
    maximum = 0.0
    for computed in contrasts:
        matches = [row for row in published if row["comparator"] == computed["comparator"]]
        if len(matches) != 1:
            raise AssertionError("Published contrast is absent or ambiguous")
        reported = matches[0]
        if reported["candidate"] != computed["candidate"] or bool(reported["criterion_established"]) != computed["strictly_positive"]:
            raise AssertionError("Published candidate or primary criterion differs")
        for target, source in mapping.items():
            value, expected = float(reported[target]), float(computed[source])
            if not np.isclose(value, expected, rtol=1e-10, atol=1e-12):
                raise AssertionError(f"Published contrast {computed['comparator']} differs in {target}: {value} versus {expected}")
            maximum = max(maximum, abs(value - expected))
    return {"path": str(Path(path)), "sha256": sha(path), "contrasts_compared": len(contrasts),
            "maximum_absolute_statistic_difference": maximum,
            "scope": "Comparison against the published JSON values after independent recomputation from saved forecasts."}


def run(study, *, task_id=None, analysis_json=None):
    started = time.perf_counter()
    study = Path(study).resolve()
    config, manifest, frozen = [read_json(study / name) for name in
                                ("config.json", "prepared_inputs/manifest.json", "freeze.json")]
    for item in frozen["files"]:
        if sha(REPOSITORY / item["path"]) != item["sha256"]:
            raise AssertionError("Frozen file changed: " + item["path"])
    if config["methods"] != list(METHODS) or manifest["development_only"]:
        raise AssertionError("Unexpected roster or development scope")
    expected = {f"{law}__noise{noise}__seed{value}" for value in config["seeds"]
                for law in config["laws"] for noise in range(len(config["noise_levels"]))}
    if len(manifest["tasks"]) != len(expected) or {row["task_id"] for row in manifest["tasks"]} != expected:
        raise AssertionError("Manifest does not contain the full declared roster")
    streams = verify_stream_domains(manifest["tasks"], config)
    selected = [task for task in manifest["tasks"] if task_id is None or task["task_id"] == task_id]
    if not selected:
        raise ValueError("Unknown task identity")
    records = []
    for task in selected:
        records.append(task_audit(study, task, config))
        if len(records) == 1 or len(records) % 16 == 0:
            print(json.dumps({"audit_tasks_verified": len(records), "of": len(selected)}), flush=True)
    complete = task_id is None
    repetition, contrasts = primary_statistics(records, config) if complete else ([], [])
    agreement = compare_analysis(contrasts, analysis_json or study / "analysis/decision.json") if complete else None
    if complete:
        # Paired levels must share the exact base trajectories, unit noise, cap
        # indices, and fit seed. This compares generation metadata, not outcomes.
        for value in config["seeds"]:
            for law in config["laws"]:
                pair = [task for task in selected if task["seed"] == value and task["law"] == law]
                if len(pair) != 2 or pair[0]["fit_seed"] != pair[1]["fit_seed"]:
                    raise AssertionError("Noise-pair identity mismatch")
                for split in SPLITS:
                    for field in ("base_episode_sha256", "unit_noise_sha256", "retained_row_indices_sha256"):
                        if pair[0]["splits"][split][field] != pair[1]["splits"][split][field]:
                            raise AssertionError("Paired noise levels changed an underlying random draw")
    return {
        "schema_version": SCHEMA, "status": "passed", "complete_roster": complete,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Internal verification using independently implemented formulas and tree traversal; not external replication.",
        "no_refitting": True, "study_predictor_or_analyzer_functions_called": False,
        "limitations": [
            "Discarded B candidate coefficients are not retained. Their logged argmin ordering is checked, while only selected candidates' B losses can be independently replayed without refitting.",
            "The audit reconstructs windows and noisy observations from saved base episodes. It does not independently regenerate every nonlinear simulator trajectory or refit the unsupervised partitions and principal subspaces.",
            "Saved HGB trees are traversed independently; their training algorithm is not reimplemented or refitted.",
            "Verification uses the same published inputs and fitted states and is internal verification, not external empirical replication."
        ],
        "tasks_verified": len(records),
        "method_task_predictions_verified": sum(record["method_predictions_verified"] for record in records),
        "windows_verified": sum(record["windows_verified"] for record in records),
        "ridge_coefficient_vectors_verified": sum(record["ridge_coefficient_vectors_verified"] for record in records),
        "selectors_verified": sum(record["selectors_verified"] for record in records),
        "primary_contrasts_verified": len(contrasts),
        "scored_cells_per_method": sum(record["scored_cells_per_method"] for record in records),
        "maximum_abs_prediction_difference": max(record["maximum_abs_prediction_difference"] for record in records),
        "maximum_scaled_normal_equation_residual": max(record["maximum_scaled_normal_equation_residual"] for record in records),
        "independent_primary_statistical_gate_passed": all(row["strictly_positive"] for row in contrasts) if complete else None,
        "independent_catastrophic_error_screen_passed": all(row["catastrophic_error_screen_passed"] for row in records),
        "repetition_aggregates": repetition, "primary_contrasts": contrasts, "tasks": records,
        "rtol": RTOL, "atol": ATOL,
        "frozen_files_verified": len(frozen["files"]), "freeze_sha256": sha(study / "freeze.json"),
        "distinct_rng_streams_verified": streams,
        "published_analysis_agreement": agreement,
        "manifest_sha256": sha(study / "prepared_inputs/manifest.json"),
        "audit_source_sha256": sha(Path(__file__)), "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--analysis-json", type=Path,
                        help="Published decision JSON to compare; defaults to study-dir/analysis/decision.json")
    args = parser.parse_args()
    if args.task_id and args.output is None:
        parser.error("Partial verification requires an explicit --output path")
    output = args.output or args.study_dir / "results/audit.json"
    if args.output is None and output.exists():
        raise FileExistsError("Archived audit exists; use --output for a separate verification")
    try:
        result = run(args.study_dir, task_id=args.task_id, analysis_json=args.analysis_json)
    except Exception as exc:
        result = {"schema_version": SCHEMA, "status": "failed", "complete_roster": False,
                  "exception": repr(exc), "traceback": traceback.format_exc()}
        write_json(output, result)
        raise
    write_json(output, result)
    print(json.dumps({key: result[key] for key in ("status", "complete_roster", "tasks_verified",
                     "method_task_predictions_verified", "primary_contrasts_verified",
                     "maximum_abs_prediction_difference", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
