"""Frozen-feature diagnostics, separate from fitting and confirmatory decisions.

All metric standardization and rank calculations use training features only.
Held-out labels are used solely for descriptive mismatch summaries.  These
diagnostics neither estimate optimal decoder risk nor demonstrate irreducible
information loss.  Branch reflection is applied to the *raw* input before the
full frozen scaler; a learned scaler need not commute with reflection.
"""
from __future__ import annotations

import copy
import hashlib
import pickle

import numpy as np
from scipy.spatial.distance import cdist


def _array(value, name, ndim=None):
    out = np.asarray(value)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if out.size == 0 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return out


def _digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode() + array.tobytes()).hexdigest()


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "q90": float(np.quantile(values, .90)), "q99": float(np.quantile(values, .99)),
            "maximum": float(values.max())}


def _groups(split, n):
    groups = np.asarray(split.get("groups", np.arange(n)))
    if groups.ndim != 1 or len(groups) != n:
        raise ValueError("Each observation must have one group identifier")
    return groups


def _group_mean(values, groups):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    totals = np.zeros((len(count), values.shape[1]))
    np.add.at(totals, inverse, values)
    return (totals / count[:, None]).mean(axis=0)


def _balanced_indices(groups, limit, rng):
    """Round-robin, randomized group sampling, fixed independently of features/Y."""
    n = len(groups)
    if limit is None or n <= limit:
        return np.arange(n)
    if int(limit) < 1:
        raise ValueError("Diagnostic subsample sizes must be positive")
    levels, inverse = np.unique(groups, return_inverse=True)
    order = rng.permutation(len(levels))
    queues = [rng.permutation(np.flatnonzero(inverse == k)) for k in order]
    chosen = []
    depth = 0
    while len(chosen) < min(n, int(limit)):
        for queue in queues:
            if depth < len(queue):
                chosen.append(int(queue[depth]))
                if len(chosen) == int(limit):
                    break
        depth += 1
    return np.sort(chosen)


def _feature_scaling(train):
    mean = train.mean(axis=0)
    standard_deviation = train.std(axis=0)
    active = standard_deviation > 1e-12
    scale = np.where(active, standard_deviation, 1.0)
    standardized = (train - mean) / scale
    singular = np.linalg.svd(standardized, full_matrices=False, compute_uv=False)
    energy = singular**2
    effective_rank = float(energy.sum()**2 / np.sum(energy**2)) if np.sum(energy**2) > 0 else 0.0
    cutoff = float(singular[0] * 1e-10) if len(singular) else 0.0
    numerical_rank = int(np.count_nonzero(singular > cutoff)) if cutoff > 0 else 0
    diagnostics = {
        "stored_dimension": int(train.shape[1]), "nonconstant_training_coordinates": int(active.sum()),
        "numerical_rank_relative_tolerance_1e_10": numerical_rank,
        "participation_effective_rank": effective_rank,
        "training_total_coordinate_variance": float(np.sum(standard_deviation**2)),
        "training_mean_sha256": _digest(mean), "training_scale_sha256": _digest(scale),
        "constant_coordinate_rule": "training standard deviation <= 1e-12 uses scale one; coordinate remains present",
    }
    return mean, scale, diagnostics


def _recovery(bank, features):
    result = {}
    for geometry in ("kahm", "affine", "centroid"):
        phi = features[f"{geometry}:phi"]
        logz = features[f"{geometry}:logz"]
        score = features[f"{geometry}:scores"]
        zeros = int(np.count_nonzero(phi == 0))
        if zeros:
            result[geometry] = {"zero_phi_coordinates": zeros, "maximum_absolute_score_error": None,
                                "within_5e_12": False, "note": "Cannot invert an emitted zero affinity using phi alone"}
            continue
        restored = np.empty_like(phi)
        for b, section in enumerate(bank.block_slices_):
            restored[:, section] = 1 + bank.tau - np.exp((np.log(phi[:, section]) + logz[:, b:b+1]) / bank.omega)
        error = float(np.max(np.abs(restored - score)))
        result[geometry] = {"zero_phi_coordinates": 0, "maximum_absolute_score_error": error,
                            "within_5e_12": bool(error <= 5e-12)}
    return result


def _nearest_neighbor_diagnostics(train_split, test_split, train_features, test_features,
                                  scaling, reference_indices, query_indices):
    features = ["raw", "kahm:phi", "affine:phi", "centroid:phi", "kahm:phi_logz", "affine:phi_logz",
                "kahm:local", "affine:local", "kahm:phi_raw", "affine:phi_raw"]
    features = [key for key in features if key in train_features]
    test_groups = _groups(test_split, len(test_split["X"]))[query_indices]
    raw_mean, raw_scale, _ = scaling["raw"]
    raw_train = (train_features["raw"][reference_indices] - raw_mean) / raw_scale
    raw_test = (test_features["raw"][query_indices] - raw_mean) / raw_scale
    raw_distances = cdist(raw_test, raw_train, metric="sqeuclidean")
    output = {}
    has_targets = "Y" in train_split and "Y" in test_split
    if has_targets:
        train_target = np.asarray(train_split["Y"], dtype=float)
        test_target = np.asarray(test_split["Y"], dtype=float)
        if train_target.ndim == 1:
            train_target = train_target[:, None]
        if test_target.ndim == 1:
            test_target = test_target[:, None]
        if (len(train_target) != len(train_split["X"]) or len(test_target) != len(test_split["X"])
                or train_target.shape[1] != test_target.shape[1]
                or not np.isfinite(train_target).all() or not np.isfinite(test_target).all()):
            raise ValueError("Training and held-out Y must have matching finite target coordinates")
    has_latents = all(key in train_split and key in test_split for key in ("S", "T"))
    row = np.arange(len(query_indices))
    for key in features:
        mean, scale, _ = scaling[key]
        reference = (train_features[key][reference_indices] - mean) / scale
        query = (test_features[key][query_indices] - mean) / scale
        distances = cdist(query, reference, metric="sqeuclidean")
        nearest = np.argmin(distances, axis=1)
        train_ids = reference_indices[nearest]
        selected_raw_distance = raw_distances[row, nearest]
        less = np.sum(raw_distances < selected_raw_distance[:, None], axis=1)
        tied = np.sum(raw_distances == selected_raw_distance[:, None], axis=1)
        raw_rank = 1 + less + 0.5 * (tied - 1)
        rank_fraction = (raw_rank - 1) / max(len(reference_indices) - 1, 1)
        nearest_distance = distances[row, nearest]
        result = {
            "standardized_coordinate_rms_neighbor_distance": _summary(np.sqrt(nearest_distance / reference.shape[1])),
            "selected_neighbor_raw_distance_rank_percentile": _summary(rank_fraction),
            "queries_with_exact_distance_ties_fraction": float(np.mean(np.sum(distances == nearest_distance[:, None], axis=1) > 1)),
            "selected_training_row_indices": train_ids.tolist(),
        }
        if has_targets:
            mismatch = (test_target[query_indices] - train_target[train_ids])**2
            mse = _group_mean(mismatch, test_groups)
            result["observed_target_mismatch"] = {
                "group_balanced_mse_by_target": mse.tolist(), "group_balanced_mean_mse": float(mse.mean()),
                "interpretation": "descriptive one-neighbor target mismatch; includes both samples' noise and finite-neighborhood error",
            }
        if has_latents:
            delta_s = np.asarray(test_split["S"])[query_indices] - np.asarray(train_split["S"])[train_ids]
            delta_t = np.asarray(test_split["T"])[query_indices] - np.asarray(train_split["T"])[train_ids]
            moments = _group_mean(np.column_stack((delta_s**2, 3 * delta_t**2,
                                                    2 * np.sqrt(3.0) * delta_s * delta_t)), test_groups)
            result["branch_noiseless_target_mismatch"] = {
                "A_branch_component": float(moments[0]), "B_longitudinal_component": float(moments[1]),
                "C_cross_component": float(moments[2]),
                "wrong_branch_fraction": float(_group_mean(delta_s != 0, test_groups)[0]),
                "formula_at_lambda": "(1-lambda)*A + lambda*B + sqrt(lambda*(1-lambda))*C",
                "interpretation": "exact noiseless target mismatch for these chosen neighbors; not optimal decoder risk",
            }
        output[key] = result
    return output


def geometry_diagnostics(bank, train_split, test_split, branch=False, *,
                         max_nn_train=1600, max_nn_test=512, diagnostics_seed=None):
    """Return JSON-serializable diagnostics for an already frozen GeometryBank.

    Both split dictionaries need X; groups and Y are optional.  The branch
    diagnostic additionally requires S, T, and branch-law metadata.  It never
    refits the bank, a decoder, or a hyperparameter.  Feature standardization is
    fitted on train_split only.  Nearest-neighbor rows are sampled by a fixed,
    group-balanced rule independent of their features or targets.

    No inferential confidence claims are returned.  Repeated windows and model
    fits are not reclassified as independent observations.
    """
    train_x = _array(train_split["X"], "training X", 2).astype(float, copy=False)
    test_x = _array(test_split["X"], "held-out X", 2).astype(float, copy=False)
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training and held-out input dimensions differ")
    if diagnostics_seed is None:
        diagnostics_seed = int(bank.seed) + 299983
    rng = np.random.default_rng(int(diagnostics_seed))
    train_groups = _groups(train_split, len(train_x))
    test_groups = _groups(test_split, len(test_x))
    reference_indices = _balanced_indices(train_groups, max_nn_train, rng)
    query_indices = _balanced_indices(test_groups, max_nn_test, rng)
    previous_diagnostics = copy.deepcopy(getattr(bank, "last_diagnostics_", None))
    had_previous_diagnostics = hasattr(bank, "last_diagnostics_")
    frozen_state = pickle.dumps(bank.state_dict(), protocol=5)
    try:
        train_features = bank.transform(train_x)
        test_features = bank.transform(test_x, diagnostics=True)
        test_reconstruction = copy.deepcopy(bank.last_diagnostics_)
        wanted = {"raw"}
        for geometry in ("kahm", "affine", "centroid"):
            wanted.update(f"{geometry}:{variant}" for variant in ("phi", "phi_logz", "scores", "radial_angle", "local", "phi_raw"))
        scaling = {key: _feature_scaling(train_features[key]) for key in sorted(wanted) if key in train_features}
        result = {
            "scope": "descriptive diagnostics of frozen representations; not irreducible information-loss estimates",
            "input_rows": {"training": len(train_x), "held_out": len(test_x)},
            "group_counts": {"training": int(len(np.unique(train_groups))), "held_out": int(len(np.unique(test_groups)))},
            "diagnostics_seed": int(diagnostics_seed),
            "feature_metrics": {key: record[2] for key, record in scaling.items()},
            "score_recovery_from_phi_plus_logz": {"training": _recovery(bank, train_features), "held_out": _recovery(bank, test_features)},
            "held_out_reconstruction_diagnostics": test_reconstruction,
            "nearest_neighbor_design": {
                "reference_training_row_indices": reference_indices.tolist(), "query_held_out_row_indices": query_indices.tolist(),
                "reference_count": len(reference_indices), "query_count": len(query_indices),
                "metric": "Euclidean distance after coordinate centering/scaling fit on all training features only",
                "selection": "fixed-seed randomized round-robin across input groups; same rows for every representation",
                "aggregation": "target mismatches average within selected held-out groups and then equally across those groups",
                "caveat": "feature scaling, dimension, neighborhood density, and finite training samples affect these descriptive values",
            },
            "nearest_neighbors": _nearest_neighbor_diagnostics(train_split, test_split, train_features, test_features,
                                                                  scaling, reference_indices, query_indices),
            "matched_reconstruction_scope": [
                "The scaler, fitted partitions, exact local reference observations, effective PCA basis, folding score, and affinity rule are shared.",
                "The operational reconstruction rule changes: normalized regularized kernel reconstruction versus local orthogonal affine projection.",
                "A common encoded PCA rank does not equate predictor capacity: the KAHM decoder combines full reference vectors and can leave the affine PCA plane.",
                "This holds locality and encoded subspace fixed; it does not compare independently optimized complete pipelines or equal computational budgets.",
                "local features are affinity-gated raw-query PCA coordinates, not reconstructed vectors.",
            ],
        }
        if branch:
            if test_x.shape[1] != 2 or not all(key in test_split for key in ("S", "T")):
                raise ValueError("Branch reflection requires two-dimensional raw X and known diagnostic S,T")
            t = _array(test_split["T"], "branch T", 1).astype(float)
            s = _array(test_split["S"], "branch S", 1).astype(float)
            if len(t) != len(test_x) or len(s) != len(test_x) or not np.allclose(test_x[:, 0], t, rtol=0, atol=1e-14):
                raise ValueError("Branch X[:,0] must equal diagnostic T")
            metadata = test_split.get("metadata", {})
            curvature, separation = metadata.get("curvature"), metadata.get("separation")
            if curvature is None or separation is None:
                raise ValueError("Branch law curvature and separation must be recorded")
            expected_second = float(curvature) * np.cos(np.pi * t) + float(separation) * s
            if not np.allclose(test_x[:, 1], expected_second, rtol=0, atol=1e-13):
                raise ValueError("Raw inputs do not agree with the declared branch law")
            reflected_x = test_x.copy()
            reflected_x[:, 0] *= -1
            reflected_features = bank.transform(reflected_x)
            reflection = {}
            for key, (mean, scale, _) in scaling.items():
                original = test_features[key]
                changed = reflected_features[key]
                delta = changed - original
                standardized_delta = delta / scale
                row_rms = np.sqrt(np.mean(standardized_delta**2, axis=1))
                train_variance = float(np.sum(np.var(train_features[key], axis=0)))
                reflected_energy = float(np.mean(np.sum(delta**2, axis=1)))
                reflection[key] = {
                    "stored_coordinate_rms": float(np.sqrt(np.mean(delta**2))),
                    "training_standardized_coordinate_rms": float(np.sqrt(np.mean(standardized_delta**2))),
                    "training_standardized_row_rms": _summary(row_rms),
                    "squared_separation_over_training_total_variance": reflected_energy / train_variance if train_variance > 0 else None,
                    "fraction_standardized_row_rms_below_1e_10": float(np.mean(row_rms < 1e-10)),
                }
            result["branch_reflection"] = {
                "raw_map": "(T, curvature*cos(pi*T)+separation*S) -> (-T, curvature*cos(pi*T)+separation*S)",
                "application_order": "reflect raw X, then apply the complete frozen geometry scaler and map",
                "distribution_preserving_under_declared_law": True,
                "reason": "T is uniform on [-1,1] independently of S and cosine is even",
                "noiseless_target_difference": "m(X)-m(reflected X) = 2*sqrt(3*lambda)*T",
                "empirical_squared_target_contrast_per_unit_lambda": float(np.mean(12 * t**2)),
                "population_squared_target_contrast_per_unit_lambda": 4.0,
                "empirical_longitudinal_signal_variance_about_zero_per_unit_lambda": float(np.mean(3 * t**2)),
                "metrics": reflection,
                "interpretation": "empirical sensitivity of the fixed representation to a known predictive direction; approximate invariance is not a proof of irreducible information loss",
                "reflected_input_sha256": _digest(reflected_x),
            }
        result["frozen_model_state_unchanged"] = pickle.dumps(bank.state_dict(), protocol=5) == frozen_state
        if not result["frozen_model_state_unchanged"]:
            raise RuntimeError("A diagnostic changed frozen fitted state")
        return result
    finally:
        if had_previous_diagnostics:
            bank.last_diagnostics_ = previous_diagnostics
        elif hasattr(bank, "last_diagnostics_"):
            del bank.last_diagnostics_
