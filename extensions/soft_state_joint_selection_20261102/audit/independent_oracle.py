"""Independent mathematical replay of portable soft-state models.

This module imports neither the production model nor its numerical primitives.
It uses direct pairwise distances, extended-precision radial normalization,
and positive-definite normal-equation solves rather than the production SVD.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

ARMS = ("G", "P", "K", "R")
VIEWS = ("shared", "independent", "standard_shared", "standard_independent")
SCALERS = ("standard", "minmax", "tanh_z_1")
CAPS = (1, 2, 4, 8, 12)
BLOCKS = ("gates", "position", "kahm", "rbf")
# Declared before production. Independent computational order can differ from
# the producer, so numerical replay is distinct from exact artifact hashing.
REPLAY_RTOL = 2e-7
REPLAY_ATOL = 2e-9


class Auditor:
    def __init__(self):
        self.checks = 0
        self.max_absolute_difference = defaultdict(float)

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            raise AssertionError(label)

    def near(self, actual, expected, label, *, rtol=REPLAY_RTOL, atol=REPLAY_ATOL, exact=False):
        a, b = np.asarray(actual), np.asarray(expected)
        self.check(a.shape == b.shape, label + ": shape")
        self.check(np.isfinite(a).all() and np.isfinite(b).all(), label + ": finite")
        if a.size:
            difference = np.not_equal(a, b).astype(float) if a.dtype == bool or b.dtype == bool else np.abs(a - b)
            self.max_absolute_difference[label] = max(self.max_absolute_difference[label],
                                                       float(np.max(difference)))
        self.check(np.array_equal(a, b) if exact else np.allclose(a, b, rtol=rtol, atol=atol), label)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    description = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(description + b"\0" + value.tobytes()).hexdigest()


def read_portable(directory, audit):
    directory = Path(directory)
    document = json.loads((directory / "state.json").read_text())
    audit.check(document["schema"] == "kahm-soft-state-portable-v1", "Portable schema")
    audit.check(document["array_file"] == "state.npz", "Portable array filename")
    audit.check(file_hash(directory / "state.npz") == document["array_sha256"], "Portable archive hash")
    with np.load(directory / "state.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    visited = set()

    def decode(item):
        if isinstance(item, list):
            return [decode(child) for child in item]
        if not isinstance(item, dict):
            return item
        if "__ndarray__" in item:
            audit.check(set(item) == {"__ndarray__", "dtype", "shape", "sha256"}, "Array descriptor roster")
            key = item["__ndarray__"]
            audit.check(key in arrays and key not in visited, "Array descriptor uniqueness")
            visited.add(key)
            value = arrays[key]
            audit.check(str(value.dtype) == item["dtype"] and list(value.shape) == item["shape"],
                        "Portable array dtype/shape")
            audit.check(array_hash(value) == item["sha256"], "Portable array content hash")
            audit.check(not value.dtype.hasobject and np.isfinite(value).all(), "Portable finite numeric arrays")
            return value
        if "__tuple__" in item:
            audit.check(set(item) == {"__tuple__"}, "Tuple descriptor")
            return tuple(decode(child) for child in item["__tuple__"])
        return {key: decode(value) for key, value in item.items()}

    result = decode(document["tree"])
    audit.check(visited == set(arrays), "Portable arrays exhaustively referenced")
    canonical = (json.dumps(document["tree"], indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return result, hashlib.sha256(canonical).hexdigest()


def input_transform(bank, x, audit):
    """Independent application of frozen input geometry scaling."""
    scaler = bank.get("geometry_scaler")
    if scaler is None:
        z = (x - bank["mean"]) / bank["scale"]
    elif scaler["mode"] == "minmax":
        audit.check(scaler["clip"] is False, "No minmax query clipping")
        denominator = np.where(scaler["constant"], 1., scaler["range"])
        z = 2. * ((x - scaler["minimum"]) / denominator) - 1.
        z[:, scaler["constant"]] = 0.
    elif scaler["mode"] == "tanh_z_1":
        audit.check(scaler["multiplier"] == 1. and scaler["post_tanh_rescale"] is False,
                    "Declared tanh geometry map")
        z = np.tanh((x - scaler["mean"]) / scaler["scale"])
    else:
        raise AssertionError("Undeclared input geometry transformation")
    audit.check(np.isfinite(z).all(), "Finite independently transformed histories")
    return z


def validate_input_scaler(bank, ax, scaling, audit):
    """Recompute every fitted scaler statistic independently from A only."""
    audit.check(scaling in SCALERS, "Declared scaling mode")
    mean, sd = ax.mean(axis=0), ax.std(axis=0, ddof=0)
    scale = np.where(sd > 1e-12, sd, 1.)
    audit.near(bank["mean"], mean, "A geometry mean", exact=True)
    audit.near(bank["scale"], scale, "A geometry scale", exact=True)
    scaler = bank.get("geometry_scaler")
    if scaling == "standard":
        audit.check(scaler is None, "Standard geometry uses inherited scaler")
    else:
        audit.check(scaler is not None and scaler["mode"] == scaling and scaler["fit_rows"] == len(ax),
                    "Alternative geometry fitted on every A history")
        if scaling == "minmax":
            lower, upper = ax.min(axis=0), ax.max(axis=0)
            for key, expected in (("minimum", lower), ("maximum", upper), ("range", upper-lower),
                                  ("constant", upper == lower)):
                audit.near(scaler[key], expected, "A-only minmax " + key, exact=True)
            audit.check(scaler["clip"] is False, "Unclipped minmax map")
        else:
            for key, expected in (("mean", mean), ("raw_standard_deviation", sd), ("scale", scale)):
                audit.near(scaler[key], expected, "A-only tanh " + key, exact=True)
            audit.check(scaler["multiplier"] == 1. and scaler["ddof"] == 0 and
                        scaler["standard_deviation_floor"] == 1e-12 and
                        scaler["fitted_location_scale"] is True and scaler["post_tanh_rescale"] is False,
                        "Tanh fitting and no rescaling semantics")
    return input_transform(bank, ax, audit)


def validate_bank(bank, diagnostic, ax, cap, audit, *, scaling="standard"):
    """Check A references and native spectral/range semantics, not k-means replay."""
    z = validate_input_scaler(bank, ax, scaling, audit)
    audit.check(bank["config"]["dimension_policy"] == "native" and bank["config"]["rank"] == cap,
                "Native bank policy")
    expected_slices, offset = [], 0
    for count in diagnostic["bank"]["requested_blocks"]:
        expected_slices.append([offset, offset + count])
        offset += count
    audit.check([list(pair) for pair in bank["block_slices"]] == expected_slices,
                "Complete declared partition slices")
    audit.check(len(bank["models"]) == offset, "Declared local-model count")
    dimensions = []
    for j, local in enumerate(bank["models"]):
        q, p = local["q"], local["p"]
        ids = np.asarray(diagnostic["bank"]["local_models"][j]["reference_indices"], dtype=int)
        audit.check(len(ids) == len(q) and len(set(ids.tolist())) == len(ids) and
                    np.all((0 <= ids) & (ids < len(ax))), "Local reference row eligibility")
        audit.near(q, z[ids].astype(np.float32).astype(np.float64), "A source references", exact=True)
        audit.near(local["center"], q.mean(axis=0), "Reference center", exact=True)
        centered = q - q.mean(axis=0)
        radius = max(float(np.mean(np.sum(centered * centered, axis=1))), 1e-6)
        audit.near(local["radius"], radius, "Reference radius", rtol=1e-12, atol=1e-12)
        r = p.shape[1]
        record = local["dimension_audit"]
        admissible = min(cap, ax.shape[1], len(q) - 1)
        audit.check(local["rank"] == r == record["effective_rank"], "Stored/effective rank identity")
        audit.check(record["initial_admissible_rank"] == admissible and 0 <= r <= admissible,
                    "Native admissible dimension")
        audit.check(record["requested_cap"] == cap and record["policy"] == "native" and
                    record["normal_direction_reserved"] is False, "Native cap semantics")
        audit.near(p.T @ p, np.eye(r), "Basis orthogonality", rtol=1e-9, atol=1e-9)
        if len(q) == 1:
            audit.check(r == 0 and record["fallback_applied"], "Singleton zero-rank fallback")
            eigenvalues = np.zeros(ax.shape[1])
        else:
            # This is a covariance eigensystem check, independently derived
            # from the saved A references; the production PCA is not called.
            covariance = np.cov(q, rowvar=False, ddof=1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            positive = int(np.count_nonzero(eigenvalues > np.finfo(float).eps))
            if positive == 0:
                audit.check(r == 0 and record["fallback_applied"], "Constant zero-rank fallback")
            else:
                audit.check(r > 0, "Positive-variance local basis retained")
                audit.near(p.T @ covariance @ p, np.diag(eigenvalues[:r]), "Leading PCA spectrum",
                           rtol=5e-7, atol=5e-12)
                requested = admissible
                while True:
                    effective = min(requested, positive)
                    projection = q @ eigenvectors[:, :effective]
                    minimum_range = float(np.ptp(projection, axis=0).min())
                    if minimum_range >= 1e-3 or requested <= 1:
                        break
                    requested -= 1
                audit.check(r == effective, "Independent native rank/range loop")
                audit.check(record["final_requested_rank"] == requested and
                            record["range_reduction_steps"] == admissible - requested,
                            "Native rank reduction metadata")
                actual_min = float(np.ptp(q @ p, axis=0).min())
                audit.near(record["pca_trace"][-1]["minimum_projected_range"], actual_min,
                           "Final native projected range", rtol=2e-7, atol=2e-9)
        if r:
            projected = q @ p
            covariance = np.atleast_2d(np.cov(projected, rowvar=False, ddof=1))
            whitening = local["whitening"]
            audit.near(whitening @ covariance @ whitening.T, np.eye(r), "Projected whitening identity",
                       rtol=2e-6, atol=2e-6)
            audit.near(local["uq"], projected @ whitening.T, "Whitened reference coordinates")
            kernel = np.exp(-cdist(local["uq"], local["uq"], "sqeuclidean") / (2 * r))
            regularizer = float(local["lambda_original"])
            audit.check(regularizer > 0 and np.isfinite(regularizer), "Positive KAHM regularizer")
            audit.near((kernel + regularizer * np.eye(len(q))) @ local["inverse"], np.eye(len(q)),
                       "KAHM inverse identity", rtol=2e-7, atol=2e-7)
        dimensions.append({"model_index": j, "block_index": int(local["block"]), "cap": cap,
            "reference_count": len(q), "initial_admissible_rank": admissible, "effective_rank": r,
            "rank_zero": r == 0, "range_reduction_steps": record["range_reduction_steps"]})
    return dimensions


def raw_blocks(bank, x, audit):
    """Compute the declared geometry directly from saved local states."""
    z = input_transform(bank, x, audit)
    centers = np.stack([model["center"] for model in bank["models"]])
    radii = np.asarray([model["radius"] for model in bank["models"]])
    logits = -cdist(z, centers, "sqeuclidean") / (2 * radii)
    logits[~np.isfinite(logits)] = -np.inf
    gates = np.zeros_like(logits)
    gate_fallbacks = []
    for block, (start, stop) in enumerate(bank["block_slices"]):
        local_logits = logits[:, start:stop].copy()
        bad = ~np.isfinite(local_logits).any(axis=1)
        if bad.any():
            local_logits[bad] = 0.
            gate_fallbacks.append({"block": block, "rows": np.flatnonzero(bad).tolist()})
        weights = np.exp(local_logits - local_logits.max(axis=1, keepdims=True))
        gates[:, start:stop] = weights / weights.sum(axis=1, keepdims=True)
        audit.near(gates[:, start:stop].sum(axis=1), np.ones(len(x)), "Partition gate sums",
                   rtol=0, atol=1e-12)
    audit.check(np.all(gates >= 0) and np.all(gates <= 1), "Gate membership range")
    positions, kahm, rbf, records = [], [], [], []
    coordinate_fallbacks, kernel_fallbacks = [], []
    for j, local in enumerate(bank["models"]):
        p, q = local["p"], local["q"]
        coordinate = (z - local["center"]) @ p
        invalid_coordinate = ~np.isfinite(coordinate).all(axis=1)
        coordinate[invalid_coordinate] = 0
        if invalid_coordinate.any():
            coordinate_fallbacks.append({"model": j, "rows": np.flatnonzero(invalid_coordinate).tolist()})
        affine = local["center"] + coordinate @ p.T
        positions.append(gates[:, j:j + 1] * coordinate)
        rank = p.shape[1]
        if rank:
            query = (z @ p) @ local["whitening"].T
            logs = -cdist(query, local["uq"], "sqeuclidean") / (2 * rank)
            invalid_kernel = ~np.isfinite(logs).all(axis=1)
            if invalid_kernel.any():
                logs[invalid_kernel] = 0
                kernel_fallbacks.append({"model": j, "rows": np.flatnonzero(invalid_kernel).tolist()})
            kernel = np.exp(logs - logs.max(axis=1, keepdims=True))
        for name, pieces in (("kahm", kahm), ("rbf", rbf)):
            if rank == 0:
                pieces.append(np.zeros((len(x), x.shape[1])))
                records.append({"model": j, "kind": name, "fallback_rows": [], "rank_zero": True})
                continue
            weights = kernel @ local["inverse"] if name == "kahm" else kernel
            denominator, l1 = weights.sum(axis=1), np.abs(weights).sum(axis=1)
            relative = np.divide(np.abs(denominator), l1, out=np.zeros_like(l1), where=l1 != 0)
            safe = np.isfinite(weights).all(axis=1) & np.isfinite(relative)
            safe &= relative > bank["config"]["relative_denominator_tolerance"]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                reconstruction = (weights @ q) / denominator[:, None]
            safe &= np.isfinite(reconstruction).all(axis=1) & ~invalid_kernel
            reconstruction[~safe] = affine[~safe]
            departure = gates[:, j:j + 1] * (reconstruction - affine)
            bad = ~np.isfinite(departure).all(axis=1)
            departure[bad] = 0
            safe &= ~bad
            pieces.append(departure)
            records.append({"model": j, "kind": name,
                            "fallback_rows": np.flatnonzero(~safe).tolist(), "rank_zero": False})
    blocks = {"gates": gates, "position": np.concatenate(positions, axis=1),
              "kahm": np.concatenate(kahm, axis=1), "rbf": np.concatenate(rbf, axis=1)}
    audit.check(all(np.isfinite(v).all() for v in blocks.values()), "Finite independent raw features")
    return blocks, {"local_models": records, "gate_fallback_rows": gate_fallbacks,
                    "coordinate_fallback_rows": coordinate_fallbacks, "kernel_fallback_rows": kernel_fallbacks}


def transformed_features(state, raw, audit):
    blocks = {}
    for name in BLOCKS:
        values = raw[name]
        if values.shape[1] == 0:
            blocks[name] = np.zeros((len(values), 0))
            continue
        scaler = state["scalers"][name]
        z = (values - scaler["mean"]) / scaler["scale"]
        # Independent algebraic expression with a wider exponent range than
        # float64. This does not call or copy the producer's max-rescaling path.
        wide = z.astype(np.longdouble)
        denominator = np.sqrt(1 + np.sum(wide * wide, axis=1, keepdims=True) / values.shape[1])
        blocks[name] = np.asarray(wide / denominator, dtype=np.float64)
        audit.check(np.isfinite(blocks[name]).all(), "Finite independent bounded block")
        audit.check(np.all(np.linalg.norm(blocks[name], axis=1) <= math.sqrt(values.shape[1]) * (1 + 1e-12)),
                    "Block radial radius")
    result = {"G": blocks["gates"], "P": np.concatenate((blocks["gates"], blocks["position"]), axis=1)}
    result["K"] = np.concatenate((result["P"], blocks["kahm"]), axis=1)
    result["R"] = np.concatenate((result["P"], blocks["rbf"]), axis=1)
    return result


def check_diagnostics(saved, oracle, features, audit):
    for key in ("gate_fallback_rows", "coordinate_fallback_rows", "kernel_fallback_rows"):
        audit.check(saved[key] == oracle[key], "Independent " + key)
    expected = {(item["model"], item["kind"]): item for item in oracle["local_models"]}
    actual = {(item["model"], item["kind"]): item for item in saved["local_models"]}
    audit.check(set(expected) == set(actual), "Fallback local record roster")
    for identity, row in expected.items():
        audit.check(actual[identity]["fallback_rows"] == row["fallback_rows"], "Reconstruction fallback row identities")
        audit.check(actual[identity]["fallback_count"] == len(row["fallback_rows"]), "Reconstruction fallback count")
        audit.check(bool(actual[identity].get("rank_zero", False)) == row["rank_zero"], "Rank-zero query behavior")
    audit.check(all(saved["nesting_checks"].values()), "Producer exact nesting checks")
    audit.check(saved["whole_descriptor_radial_map"] is False, "No descriptor-wide radial renormalization")
    for arm in ARMS:
        audit.check(saved["method_widths"][arm] == features[arm].shape[1], "Independent feature width")
    prefix = saved["shared_prefix_sha256"]
    audit.check(prefix["P_gate_prefix"] == saved["method_feature_sha256"]["G"], "Exact G prefix hash in P")
    for key in ("K_P_prefix", "R_P_prefix"):
        audit.check(prefix[key] == saved["method_feature_sha256"]["P"], "Exact P prefix hash in enrichment")


def fit_ridge_path(features_a, targets_a, features_b, alphas):
    """Unique positive-penalty affine solution from Cholesky normal equations."""
    location, response_location = features_a.mean(axis=0), targets_a.mean(axis=0)
    x, y = features_a - location, targets_a - response_location
    gram, cross = x.T @ x, x.T @ y
    coefficients, predictions = [], []
    for alpha in alphas:
        positive = gram + len(x) * float(alpha) * np.eye(gram.shape[0])
        factor = cho_factor(positive, lower=True, check_finite=False)
        coefficient = cho_solve(factor, cross, check_finite=False)
        coefficients.append(coefficient)
        predictions.append(response_location + (features_b - location) @ coefficient)
    return np.stack(coefficients), np.stack(predictions), location, response_location
