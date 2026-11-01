"""A bounded raw-history Nyström residual family for the search control.

The common predictor and B-fitted bias are inherited unchanged.  A supplies
history transforms, Nyström features, feature statistics and centered residual
slopes; B selects alpha per horizon and one scaler/bandwidth across horizons.
C may select only an eta from the inherited grid.  Prediction accepts histories
only.  All bandwidth factors multiply the median *squared* landmark distance,
matching the historical raw-Nyström construction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

PREVIOUS = Path(__file__).resolve().parents[2] / "preprocessing_dimension"
if str(PREVIOUS / "code") not in sys.path:
    sys.path.insert(0, str(PREVIOUS / "code"))
import preprocessing_model as previous
import fixed_common_secondary as secondary

fd = previous.fd
SCALERS = ("standard", "minmax", "tanh_z_1")
BANDWIDTH_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
METHOD = "extra_rbf"


def _candidate_id(scaler, factor):
    return f"{scaler}__bw{float(factor):g}"


def _scaled(x, arm, audit, raw_scaler):
    m = fd.load_runtime()
    if arm == "standard":
        return m._standardize(raw_scaler, x)
    from geometry_adapter import scaled_histories
    return scaled_histories(x, audit)


@dataclass
class ExtraRBF:
    common: Any
    bias: np.ndarray
    raw_scaler: dict
    history_audit: dict
    feature_state: dict
    head: dict
    selections: dict
    validation_predictions: dict
    diagnostics: dict
    config: dict
    eta: np.ndarray | None = None
    calibration: dict = field(default_factory=dict)

    def frozen_state(self):
        """The A/B state; C calibration is intentionally kept outside it."""
        return {"common": self.common.predictive_state(), "bias": self.bias,
                "raw_scaler": self.raw_scaler, "history_audit": self.history_audit,
                "feature_state": self.feature_state, "head": self.head,
                "selections": self.selections, "config": self.config}

    def frozen_hash(self):
        return fd.state_hash(self.frozen_state())

    def _uncalibrated(self, x):
        m = fd.load_runtime()
        x = m._matrix(x, "extra RBF query histories")
        if x.shape[1] != len(self.raw_scaler["mean"]):
            raise ValueError("Extra RBF history width differs from its fitted state")
        common, common_diagnostics = self.common.predict(x)
        shared = common + self.bias
        z = _scaled(x, self.selections["selected_scaler"], self.history_audit,
                    self.raw_scaler)
        features = m.radial_bound(m._raw_rbf_features(self.feature_state, z))
        raw = shared + secondary._slope(self.head, features)
        if raw.shape != shared.shape or not np.isfinite(raw).all():
            raise FloatingPointError("Invalid extra RBF forecast")
        diagnostics = {"common_geometry": common_diagnostics,
                       "feature_bounds": m._feature_summary(features, np.sqrt(features.shape[1]))}
        return shared, raw, diagnostics

    def calibrate(self, cx, cy):
        """Select eta on C without altering any A/B state."""
        if self.eta is not None:
            raise RuntimeError("Extra RBF was already calibrated")
        m = fd.load_runtime()
        cx, cy = m._matrix(cx, "extra RBF C histories"), m._matrix(cy, "extra RBF C targets")
        if len(cx) != len(cy) or cy.shape[1] != len(self.bias):
            raise ValueError("Extra RBF C array shapes differ")
        before = self.frozen_hash()
        wall, cpu = time.perf_counter(), time.process_time()
        shared, raw, query = self._uncalibrated(cx)
        eta, prediction, record = m._select_gamma(cy, shared, raw, self.config["gamma_grid"])
        if self.frozen_hash() != before:
            raise AssertionError("Extra RBF calibration changed A/B state")
        self.eta = eta
        self.calibration = secondary.json_safe({
            "choice": record, "selected_eta": eta.tolist(), "selection_split": "C",
            "pre_C_hash": before, "post_C_hash": before,
            "C_X_sha256": fd.array_hash(cx), "C_Y_sha256": fd.array_hash(cy),
            "shared_baseline_sha256": fd.state_hash((self.common.predictive_state(), self.bias)),
            "calibrated_prediction_sha256": fd.array_hash(prediction),
            "eta1_prediction_sha256": fd.array_hash(raw), "query": query,
            "calibration_wall_seconds": time.perf_counter() - wall,
            "calibration_cpu_seconds": time.process_time() - cpu})
        return {"predictions": {METHOD: prediction}, "eta1_predictions": {METHOD: raw},
                "records": self.calibration}

    def predict(self, x):
        """Evaluate the fixed procedure using histories only."""
        if self.eta is None:
            raise RuntimeError("Calibrate extra RBF before prediction")
        m = fd.load_runtime()
        before = self.frozen_hash()
        eta_hash = fd.array_hash(self.eta)
        wall, cpu = time.perf_counter(), time.process_time()
        shared, raw, diagnostics = self._uncalibrated(x)
        prediction = m._blend(shared, raw, self.eta)
        bound = secondary._centered_bound(self.head, self.eta)
        observed = np.max(np.abs(prediction - shared), axis=0)
        limit = np.asarray(bound["correction_bound_by_horizon"])
        violations = int(np.count_nonzero(observed > limit * (1 + 1e-10) + 1e-12))
        if violations:
            raise AssertionError("Extra RBF prediction exceeded its centered correction bound")
        if self.frozen_hash() != before or fd.array_hash(self.eta) != eta_hash:
            raise AssertionError("Extra RBF prediction changed its fitted/calibrated state")
        diagnostics.update({
            "bounds": {**bound, "observed_max_abs_correction_by_horizon": observed.tolist(),
                       "query_bound_violations": violations},
            "frozen_state_before": before, "frozen_state_after": before,
            "eta_sha256": eta_hash, "eta": self.eta.tolist(),
            "shared_baseline_sha256": fd.state_hash((self.common.predictive_state(), self.bias)),
            "prediction_wall_seconds": time.perf_counter() - wall,
            "prediction_cpu_seconds": time.process_time() - cpu})
        return {"predictions": {METHOD: prediction}, "eta1_predictions": {METHOD: raw},
                "diagnostics": secondary.json_safe(diagnostics)}


def fit_extra_rbf(ax, ay, bx, by, *, crossfit_suite, seed, config):
    """Fit one extra non-KAHM family with fifteen feature configurations.

    The function deliberately accepts no C or D observations.  It retains
    selected numerical state only, together with all B candidate predictions
    and loss grids needed to audit the selection.
    """
    m = fd.load_runtime()
    ax, ay, bx, by = fd._fit_arrays(ax, ay, bx, by)
    effective = fd._config(20, "native", config)
    from geometry_adapter import fit_scaler_audit
    wall, cpu = time.perf_counter(), time.process_time()
    common = crossfit_suite.common
    bias = np.asarray(crossfit_suite.bias).copy()
    common_before = common.state_hash()
    common_a, _ = common.predict(ax)
    common_b, _ = common.predict(bx)
    np.testing.assert_array_equal(common_a, crossfit_suite.training_arrays["A_full_common"])
    np.testing.assert_array_equal(common_b, crossfit_suite.training_arrays["B_common"])
    residual = ay - common_a
    np.testing.assert_array_equal(residual, crossfit_suite.training_arrays["A_insample_residual"])
    shared_b = common_b + bias
    np.testing.assert_array_equal(shared_b, crossfit_suite.training_arrays["B_shared_bias"])
    np.testing.assert_array_equal(bias, np.mean(by - common_b, axis=0))
    variance = np.maximum(np.var(ay, axis=0, ddof=0), effective["target_variance_floor"])
    raw_scaler = m._scaler_fit(ax, effective["feature_scale_floor"])
    rng = np.random.default_rng(int(seed))
    ids = np.sort(rng.choice(len(ax), min(int(effective["raw_rbf_landmarks"]), len(ax)), replace=False))
    records, validation = [], {}
    best_score, selected = np.inf, None
    # Factor-first order parallels the KAHM cap-first order.  One scaler and
    # one bandwidth are selected jointly across horizons after alpha selection.
    transformed = {}
    for arm in SCALERS:
        audit = fit_scaler_audit(ax, arm)
        za, zb = (_scaled(x, arm, audit, raw_scaler) for x in (ax, bx))
        landmarks = za[ids].copy()
        pair = cdist(landmarks, landmarks, "sqeuclidean")
        positive = pair[pair > 0]
        median = float(np.median(positive)) if len(positive) else 1.0
        transformed[arm] = (audit, za, zb, landmarks, pair, median)
    for factor in BANDWIDTH_FACTORS:
        for arm in SCALERS:
            audit, za, zb, landmarks, pair, median = transformed[arm]
            bandwidth_squared = median * factor
            eigenvalues, eigenvectors = np.linalg.eigh(np.exp(-pair / (2 * bandwidth_squared)))
            keep = eigenvalues > max(1e-12, 1e-10 * float(np.max(eigenvalues)))
            if not np.any(keep):
                raise FloatingPointError("Extra RBF has no retained Nyström directions")
            transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
            raw_a = np.column_stack((za, np.exp(-cdist(za, landmarks, "sqeuclidean")
                                               / (2 * bandwidth_squared)) @ transform))
            state = {"landmarks": landmarks, "transform": transform,
                     "bandwidth_squared": bandwidth_squared,
                     "scaler": m._scaler_fit(raw_a, effective["feature_scale_floor"]),
                     "bandwidth_factor": float(factor)}
            fa = m.radial_bound(m._standardize(state["scaler"], raw_a))
            fb = m.radial_bound(m._raw_rbf_features(state, zb))
            path = m._ridge_path(fa, residual, effective["alpha_grid"])
            alpha_predictions = np.stack([shared_b + secondary._slope(head, fb) for head in path])
            losses = np.mean((by[None, :, :] - alpha_predictions) ** 2, axis=1)
            if not np.isfinite(losses).all():
                raise FloatingPointError("Extra RBF has nonfinite B losses")
            indices = np.argmin(losses, axis=0)
            head = m._combine_heads(path, indices)
            prediction = shared_b + secondary._slope(head, fb)
            mse = np.mean((by - prediction) ** 2, axis=0)
            score = float(np.mean(mse / variance))
            centered_mean = np.mean(secondary._slope(head, fa), axis=0)
            tolerance = 1e-10 * max(1.0, float(np.max(np.abs(residual))))
            if np.max(np.abs(centered_mean)) > tolerance:
                raise AssertionError("Extra RBF A correction has nonzero training mean")
            secondary._centered_bound(head, np.ones(ay.shape[1]))
            key = _candidate_id(arm, factor)
            validation[key] = prediction
            validation[key + "__alpha_paths"] = alpha_predictions
            records.append({
                "candidate_id": key, "scaler": arm, "bandwidth_factor": float(factor),
                "bandwidth_squared": float(bandwidth_squared), "score": score,
                "mse_by_horizon": mse.tolist(), "selected_alpha_indices": indices.tolist(),
                "selected_alpha_by_horizon": head["alpha_by_horizon"].tolist(),
                "B_mse_by_alpha_horizon": losses.tolist(),
                "mse_by_alpha_horizon": losses.tolist(),
                "B_prediction_sha256": fd.array_hash(prediction),
                "B_alpha_prediction_sha256": fd.array_hash(alpha_predictions),
                "feature_width": int(head["feature_width"]),
                "centered_feature_rank": int(head["feature_rank"]),
                "nystrom_rank": int(np.sum(keep)), "landmark_count": len(ids),
                "A_mean_centered_correction": centered_mean.tolist(),
                "A_feature_bound": m._feature_summary(fa, np.sqrt(fa.shape[1])),
                "B_feature_bound": m._feature_summary(fb, np.sqrt(fb.shape[1]))})
            if score < best_score:
                best_score = score
                selected = (len(records) - 1, audit, state, head)
    if selected is None or common.state_hash() != common_before:
        raise AssertionError("Extra RBF failed selection or changed the common predictor")
    index, audit, state, head = selected
    chosen = records[index]
    selections = {
        "selected_candidate": chosen["candidate_id"], "selected_index": index,
        "selected_scaler": chosen["scaler"],
        "selected_bandwidth_factor": chosen["bandwidth_factor"],
        "selected_alpha_by_horizon": chosen["selected_alpha_by_horizon"],
        "candidate_order": [row["candidate_id"] for row in records], "candidates": records,
        "selection_split": "B", "tie_rule": "First minimum: bandwidth factor ascending, then scaler order",
        "loss_scale": variance.tolist(), "residual_target_sha256": fd.array_hash(residual),
        "shared_baseline_sha256": fd.state_hash((common.predictive_state(), bias))}
    diagnostics = {
        "method": METHOD, "residual_mode": "insample", "response_intercept": "omitted",
        "history_fit_split": "A", "feature_fit_split": "A", "head_fit_split": "A",
        "family_selection_split": "B", "eta_selection_split": "C",
        "input_hashes": {key: fd.array_hash(value) for key, value in
                         (("A_X", ax), ("A_Y", ay), ("B_X", bx), ("B_Y", by))},
        "landmark_indices": ids.tolist(), "landmark_selection_seed": int(seed),
        "landmark_count": len(ids), "requested_landmarks": int(effective["raw_rbf_landmarks"]),
        "feature_configurations": len(records),
        "B_candidate_scores_per_horizon": len(records) * len(effective["alpha_grid"]),
        "alpha_grid": list(effective["alpha_grid"]), "scaler_order": list(SCALERS),
        "bandwidth_factors": list(BANDWIDTH_FACTORS),
        "bandwidth_convention": "median positive squared landmark distance times factor",
        "standard_scaler_convention": "Inherited raw-RBF population standardization with feature_scale_floor",
        "nystrom_eigenvalue_keep_rule": "lambda > max(1e-12, 1e-10 * lambda_max)",
        "raw_history_retained": True, "radial_radius_rule": "sqrt(feature width), before A centering",
        "selected_feature_width": int(head["feature_width"]),
        "selected_centered_feature_rank": int(head["feature_rank"]),
        "selected_nystrom_rank": chosen["nystrom_rank"],
        "selected_numerical_state_bytes": len(fd.state_bytes((raw_scaler, audit, state, head))),
        "complete_common_predictor_cost_included": False,
        "cost_scope": "Additional family fit/search, including validation baseline predictions; shared common construction excluded",
        "fit_wall_seconds": time.perf_counter() - wall,
        "fit_cpu_seconds": time.process_time() - cpu,
        "failures": []}
    return ExtraRBF(common, bias, raw_scaler, audit, state, head, selections,
                    validation, secondary.json_safe(diagnostics), copy.deepcopy(effective))


@dataclass
class _FixtureCommon:
    """A deterministic linear base used only by manufactured_fixture."""
    coef: np.ndarray

    def predictive_state(self):
        return {"coef": self.coef}

    def state_hash(self):
        return fd.state_hash(self.predictive_state())

    def predict(self, x):
        return np.asarray(x) @ self.coef, {}


def manufactured_fixture():
    """Check selection, centered ridge algebra and query invariance on made-up data."""
    from types import SimpleNamespace
    m = fd.load_runtime()
    rng = np.random.default_rng(491827)
    ax, bx, cx, dx = [rng.normal(size=(n, 12)) for n in (96, 48, 48, 31)]
    for x in (ax, bx, cx, dx):
        x[:, -1] = 2.0
    coefficient = rng.normal(size=(12, 3)) * 0.1
    common = _FixtureCommon(coefficient)

    def targets(x):
        signal = np.column_stack((np.sin(x[:, 0]), x[:, 1] * x[:, 2],
                                  np.tanh(x[:, 0] - x[:, 3])))
        return common.predict(x)[0] + 0.3 * signal + np.array([0.1, -0.2, 0.05])

    ay, by, cy = (targets(x) for x in (ax, bx, cx))
    bias = np.mean(by - common.predict(bx)[0], axis=0)
    reference = SimpleNamespace(common=common, bias=bias, training_arrays={
        "A_full_common": common.predict(ax)[0], "B_common": common.predict(bx)[0],
        "A_insample_residual": ay - common.predict(ax)[0],
        "B_shared_bias": common.predict(bx)[0] + bias})
    cfg = {k: copy.deepcopy(v) for k, v in m.DEFAULT_CONFIG.items()
           if k not in ("rank", "dimension_policy")}
    cfg.update(raw_rbf_landmarks=16, alpha_grid=[0.0001, 0.001, 0.01, 0.1, 1.0],
               gamma_grid=[0.0, 0.25, 0.5, 1.0])
    suite = fit_extra_rbf(ax, ay, bx, by, crossfit_suite=reference, seed=823, config=cfg)
    before = suite.frozen_hash()
    records = suite.selections["candidates"]
    assert len(records) == 15
    assert suite.diagnostics["B_candidate_scores_per_horizon"] == 75
    scores = []
    variance = np.maximum(np.var(ay, axis=0, ddof=0), cfg["target_variance_floor"])
    for record in records:
        paths = suite.validation_predictions[record["candidate_id"] + "__alpha_paths"]
        loss = np.mean((by[None] - paths) ** 2, axis=1)
        indices = np.argmin(loss, axis=0)
        np.testing.assert_array_equal(indices, record["selected_alpha_indices"])
        np.testing.assert_array_equal(loss, record["B_mse_by_alpha_horizon"])
        selected_b = np.column_stack([paths[indices[h], :, h] for h in range(3)])
        np.testing.assert_array_equal(selected_b, suite.validation_predictions[record["candidate_id"]])
        scores.append(float(np.mean(np.mean((by - selected_b) ** 2, axis=0) / variance)))
    assert int(np.argmin(scores)) == suite.selections["selected_index"]

    z = _scaled(ax, suite.selections["selected_scaler"], suite.history_audit, suite.raw_scaler)
    features = m.radial_bound(m._raw_rbf_features(suite.feature_state, z))
    centered = features - features.mean(axis=0)
    residual = reference.training_arrays["A_insample_residual"]
    response = residual - residual.mean(axis=0)
    for h, alpha in enumerate(suite.head["alpha_by_horizon"]):
        oracle = np.linalg.solve(centered.T @ centered + len(ax) * alpha * np.eye(centered.shape[1]),
                                 centered.T @ response[:, h])
        np.testing.assert_allclose(suite.head["coef"][:, h], oracle, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(np.mean(centered @ suite.head["coef"], axis=0), 0., atol=1e-12)

    calibrated = suite.calibrate(cx, cy)
    assert suite.frozen_hash() == before
    raw = calibrated["eta1_predictions"][METHOD]
    shared = common.predict(cx)[0] + bias
    eta_losses = np.stack([np.mean((cy - (shared + eta * (raw - shared))) ** 2, axis=0)
                           for eta in cfg["gamma_grid"]])
    np.testing.assert_array_equal(suite.eta, np.asarray(cfg["gamma_grid"])[np.argmin(eta_losses, axis=0)])
    result = suite.predict(dx)
    permutation = rng.permutation(len(dx))
    permuted = suite.predict(dx[permutation])["predictions"][METHOD]
    np.testing.assert_allclose(permuted, result["predictions"][METHOD][permutation], atol=1e-12, rtol=1e-12)
    appended = suite.predict(np.vstack((dx, 100 * dx[:3])))["predictions"][METHOD]
    np.testing.assert_allclose(appended[:len(dx)], result["predictions"][METHOD], atol=1e-12, rtol=1e-12)
    suite.eta = np.zeros(3)
    np.testing.assert_array_equal(suite.predict(dx)["predictions"][METHOD], common.predict(dx)[0] + bias)
    assert suite.frozen_hash() == before
    return {"status": "passed", "manufactured_data_only": True,
            "feature_configurations": len(records), "B_candidate_scores_per_horizon": 75,
            "selected_candidate": suite.selections["selected_candidate"],
            "independent_normal_equation_check": True, "all_B_selections_recomputed": True,
            "C_eta_selection_recomputed": True, "exact_zero_recovery": True,
            "query_order_and_batch_invariance": True, "A_B_state_unchanged": True,
            "query_bound_violations": result["diagnostics"]["bounds"]["query_bound_violations"]}
