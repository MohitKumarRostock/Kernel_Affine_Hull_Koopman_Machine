"""Baseline-preserving local-geometry residual forecasters.

This extension imports the archived natural-forecasting implementation read-only.
The raw model is its finite Gaussian Nyström-plus-linear predictor (or its raw
ridge control). A separately fitted, standardized local-coordinate ridge head
models the *in-sample* training residuals of that frozen raw predictor:

    prediction(x) = frozen_raw(x) + gamma * local_residual(x).

The local gate is KAHM, affine, or locality-aware affine, with an identical
constructor and the inherited stable KAHM query path. Correction features have
their own training-only standardization and ridge coefficient penalty. They do
not change raw kernel distances, landmarks, coefficients, or regularization.
The correction is linear in the local code; there is no correction Nyström
basis. Consequently a landmark-count comparison changes the raw backbone only.

Gamma zero returns the raw prediction without even evaluating local features.
This retains the actual fitted baseline, not merely its input information.
Validation selection and finite-sample generalization can still worsen test
error; no accuracy nondegradation, information creation, or risk certificate is
claimed. Residuals are not out-of-fold/cross-fitted residuals.

Fit/select/refit are distinct operations. ``fit_select`` keeps a training-only
fit and reports all alpha/gamma trials. A caller must construct a new model and
refit the selected settings on its declared pretest data, using a raw predictor
refitted on that same data. The caller owns cost accounting and selection of
any practical-improvement threshold.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from extensions.natural_forecasting.models import ForecastRegressor


GATES = ("kahm", "affine", "locality")
DEFAULT_ALPHAS = (0.01, 1.0, 100.0)
DEFAULT_GAMMAS = (0.0, 0.25, 0.5, 1.0)


def _matrix(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix with observations in rows")
    return value


def _targets(value: Any, rows: int, name: str = "target") -> tuple[np.ndarray, bool]:
    value = np.asarray(value, dtype=np.float64)
    single = value.ndim == 1
    matrix = _matrix(value[:, None] if single else value, name)
    if len(matrix) != rows:
        raise ValueError(f"Input and {name} have different row counts")
    return matrix, single


def _digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _gamma(value: Any) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    return value


class RawForecastRegressor(ForecastRegressor):
    """Inherited raw predictor with hashes of the exact fitting observations.

    Hashes allow an additive model to reject a baseline fitted on different
    observations, including an inadvertently refitted validation/test prefix.
    The numerical fitting/prediction implementation is inherited unchanged.
    ``fit_select`` is inherited and returns the complete alpha trial list.
    """

    def __init__(self, head="rbf", *, alpha=1.0, n_landmarks=64,
                 bandwidth_multiplier=1.0, seed=0, head_seed=1):
        super().__init__(representation="raw", head=head, alpha=alpha,
                         n_landmarks=n_landmarks,
                         bandwidth_multiplier=bandwidth_multiplier,
                         seed=seed, head_seed=head_seed, query_mode="stable")

    def _fit_system(self, x, y):
        x = _matrix(x, "training input")
        target, single = _targets(y, len(x))
        result = super()._fit_system(x, target[:, 0] if single else target)
        self.training_input_sha256_ = _digest(x)
        self.training_target_sha256_ = _digest(target)
        return result

    def metadata(self):
        result = super().metadata()
        result.update(training_input_sha256=self.training_input_sha256_,
                      training_target_sha256=self.training_target_sha256_)
        return result


def raw_model(head="rbf", **parameters) -> RawForecastRegressor:
    """Construct an unfitted reusable raw predictor."""
    return RawForecastRegressor(head=head, **parameters)


class AdditiveForecastRegressor:
    """Frozen raw predictor plus a separately regularized local ridge correction.

    Parameters
    ----------
    gate : {"kahm", "affine", "locality"}
        Inherited local-coordinate gate construction. Matched controls use
        identical seeds, geometry settings, and fitting observations.
    alpha : float
        Correction ridge penalty; independent of the raw model's penalty.
        The inherited head minimizes mean standardized-residual squared error
        plus ``alpha * ||coefficients||_F**2`` with an unpenalized intercept.
        Predictions are transformed back to original target units.
    gamma : float in [0, 1]
        Correction mixing weight. Zero is an explicit baseline candidate.
    seed : nonnegative integer
        Local-bank seed. No target labels enter reference selection or geometry.
    geometry_parameters : dict, optional
        Parameters of the inherited LocalFeatureBank; training-standardized
        inputs and stable Gaussian queries are fixed in this extension.

    Notes
    -----
    A supplied raw model is held by reference and is never refitted or modified
    by this estimator. The caller must not refit it after binding a correction.
    Cached raw predictions are optional computational reuse points: callers
    must supply predictions from that exact raw model on those exact rows.
    Shape/finiteness are checked; re-evaluation to verify cache contents would
    defeat the intended cost reuse.
    """

    def __init__(self, gate="kahm", *, alpha=1.0, gamma=1.0, seed=0,
                 geometry_parameters=None):
        if gate not in GATES:
            raise ValueError(f"gate must be in {GATES}")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.gate = gate
        self.alpha = _positive(alpha, "alpha")
        self.gamma = _gamma(gamma)
        self.seed = int(seed)
        self.geometry_parameters = dict(geometry_parameters or {})
        if self.geometry_parameters.get("scaling", "standard") != "standard":
            raise ValueError("This extension fixes training-standardized geometry inputs")

    def _bind_raw(self, x, y, raw_model):
        x = _matrix(x, "training input")
        target, single = _targets(y, len(x))
        if not isinstance(raw_model, RawForecastRegressor) or not hasattr(raw_model, "coefficients_"):
            raise ValueError("Supply a fitted RawForecastRegressor for the identical training observations")
        if (raw_model.training_input_sha256_ != _digest(x)
                or raw_model.training_target_sha256_ != _digest(target)
                or raw_model.single_target_ != single):
            raise ValueError("Raw model was fitted on different training inputs, targets, or target shape")
        self.raw_model_ = raw_model
        self.n_training_rows_, self.n_features_in_ = x.shape
        self.n_targets_ = target.shape[1]
        self.single_target_ = single
        self.training_input_sha256_ = _digest(x)
        self.training_target_sha256_ = _digest(target)
        self.raw_signature_ = (raw_model.training_input_sha256_,
                               raw_model.training_target_sha256_,
                               _digest(raw_model.coefficients_),
                               float(raw_model.alpha))
        return x, target

    def _require_fitted(self):
        if not hasattr(self, "correction_model_"):
            raise RuntimeError("Fit the additive model before prediction")
        current = (self.raw_model_.training_input_sha256_,
                   self.raw_model_.training_target_sha256_,
                   _digest(self.raw_model_.coefficients_),
                   float(self.raw_model_.alpha))
        if current != self.raw_signature_:
            raise RuntimeError("The shared raw model changed after fitting the correction; refit the correction")

    def _prediction_matrix(self, value, rows, name):
        value = np.asarray(value, dtype=np.float64)
        expected = (rows,) if self.single_target_ else (rows, self.n_targets_)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"{name} must have shape {expected} and contain finite predictions")
        return value[:, None] if self.single_target_ else value

    def _residual_target(self, x, target, raw_train_prediction):
        prediction = (self.raw_model_.predict(x) if raw_train_prediction is None
                      else raw_train_prediction)
        prediction = self._prediction_matrix(prediction, len(x), "raw training prediction")
        residual = target - prediction
        if not np.isfinite(residual).all():
            raise FloatingPointError("Nonfinite residual target")
        return residual[:, 0] if self.single_target_ else residual

    def _new_correction(self):
        return ForecastRegressor(
            representation=f"{self.gate}_local", head="ridge",
            alpha=self.alpha, seed=self.seed, head_seed=self.seed,
            geometry_parameters=self.geometry_parameters, query_mode="stable",
        )

    def _record_complexity(self):
        self.raw_basis_width_ = self.raw_model_.basis_width_
        self.raw_coefficient_count_ = int(self.raw_model_.coefficients_.size + self.n_targets_)
        correction = self.correction_model_
        self.correction_basis_width_ = 0 if correction is None else correction.basis_width_
        self.correction_coefficient_count_ = (0 if correction is None
                                            else int(correction.coefficients_.size + self.n_targets_))
        self.coefficient_count_ = self.raw_coefficient_count_ + self.correction_coefficient_count_
        self.stored_reference_rows_ = (0 if correction is None else
                                      sum(len(model.references) for model in correction.bank_.models_))
        self.fitted_kahm_kernels_ = (0 if correction is None else
                                    sum(model.ae is not None for model in correction.bank_.models_))

    def fit(self, x, y, *, raw_model, raw_train_prediction=None):
        """Fit a fixed correction on the same training rows as the raw model.

        For gamma zero, no local bank or correction head is fitted. This makes
        a final baseline-only refit avoid geometry cost. A later nonzero gamma
        override is rejected for that unfitted correction.
        """
        x, target = self._bind_raw(x, y, raw_model)
        if self.gamma == 0:
            self.correction_model_ = None
        else:
            residual = self._residual_target(x, target, raw_train_prediction)
            self.correction_model_ = self._new_correction().fit(x, residual)
        self.tuning_scope_ = "Fixed correction settings; fitting uses only supplied training observations."
        self._record_complexity()
        return self

    def fit_select(self, x, y, x_validation, y_validation, alphas=DEFAULT_ALPHAS,
                   gammas=DEFAULT_GAMMAS, *, raw_model,
                   raw_train_prediction=None, raw_validation_prediction=None):
        """Fit one bank/Gram system and select correction alpha/gamma by MSE.

        Returns one canonical gamma-zero trial, then positive-gamma trials in
        alpha-major, gamma-minor order. Ties prefer smaller gamma globally,
        followed by the supplied alpha order. The zero trial retains the first
        alpha for a numeric schema but marks it ``alpha_applies=False``.
        Raw selection must already have been performed by the caller. No raw
        fit or raw hyperparameter change occurs here.
        """
        alphas = tuple(_positive(a, "alpha") for a in alphas)
        gammas = tuple(_gamma(g) for g in gammas)
        if not alphas or not gammas or 0.0 not in gammas:
            raise ValueError("Use nonempty alpha/gamma grids including the gamma-zero baseline")
        x, target = self._bind_raw(x, y, raw_model)
        validation = _matrix(x_validation, "validation input")
        if validation.shape[1] != self.n_features_in_:
            raise ValueError("Validation input dimension differs from training")
        validation_target, single = _targets(y_validation, len(validation), "validation target")
        if single != self.single_target_ or validation_target.shape[1] != self.n_targets_:
            raise ValueError("Validation target shape differs from training")
        baseline = (self.raw_model_.predict(validation) if raw_validation_prediction is None
                    else raw_validation_prediction)
        baseline = self._prediction_matrix(baseline, len(validation), "raw validation prediction")
        residual = self._residual_target(x, target, raw_train_prediction)
        correction = self._new_correction()
        # Reuse the archived head's training-only construction and ridge solve.
        # These private numerical entry points are source-hashed by the runner.
        eigenvalues, eigenvectors, rhs = correction._fit_system(x, residual)
        features = correction._transform_features(validation)
        basis = correction.basis_.transform(features)
        design = (basis - correction.basis_mean_) / correction.basis_scale_
        baseline_loss = float(np.mean((baseline - validation_target) ** 2))
        if not np.isfinite(baseline_loss):
            raise FloatingPointError("Nonfinite baseline validation loss")
        trials = [{"alpha": alphas[0], "gamma": 0.0, "alpha_applies": False,
                   "validation_mse": baseline_loss}]
        best_key = (baseline_loss, 0.0, 0)
        best_alpha, best_gamma, best_coefficients = alphas[0], 0.0, None
        for alpha_index, alpha in enumerate(alphas):
            coefficients = correction._solve_alpha(eigenvalues, eigenvectors, rhs, alpha)
            if best_coefficients is None:
                # Keep a fitted correction for the search object's diagnostics;
                # a later fixed gamma-zero refit skips that correction entirely.
                best_coefficients = coefficients.copy()
            delta = ((design @ coefficients + correction.standardized_target_mean_)
                     * correction.target_scale_ + correction.target_mean_)
            for gamma in gammas:
                if gamma == 0:
                    continue
                prediction = baseline + gamma * delta
                loss = float(np.mean((prediction - validation_target) ** 2))
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss; no candidate was silently dropped")
                trials.append({"alpha": alpha, "gamma": gamma, "alpha_applies": True,
                               "validation_mse": loss})
                key = (loss, gamma, alpha_index)
                if key < best_key:
                    best_key = key
                    best_alpha, best_gamma = alpha, gamma
                    best_coefficients = coefficients.copy()
        correction.alpha = best_alpha
        correction.coefficients_ = best_coefficients
        correction.tuning_scope_ = "Correction alpha/gamma selected on validation; geometry and ridge system fit on training only."
        self.alpha, self.gamma = best_alpha, best_gamma
        self.correction_model_ = correction
        self.tuning_scope_ = ("Validation selection of correction alpha/gamma conditional on the supplied frozen raw model; "
                              "bank, scalers, residual target, and ridge system use training rows only. "
                              "All raw selection and correction searches must be charged by the caller.")
        self._record_complexity()
        return trials

    def predict_correction(self, x):
        """Return the unweighted correction in original target units."""
        self._require_fitted()
        if self.correction_model_ is None:
            raise RuntimeError("Gamma-zero fit omitted the correction; fit a correction before requesting it")
        return self.correction_model_.predict(x)

    def predict(self, x, *, gamma=None, raw_prediction=None):
        """Predict with optional reuse of this raw model's cached forecasts."""
        self._require_fitted()
        x = _matrix(x, "query input")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Query input dimension differs from training")
        mixing = self.gamma if gamma is None else _gamma(gamma)
        baseline = self.raw_model_.predict(x) if raw_prediction is None else raw_prediction
        baseline = self._prediction_matrix(baseline, len(x), "raw query prediction")
        if mixing == 0:
            prediction = baseline.copy()
        else:
            if self.correction_model_ is None:
                raise RuntimeError("Gamma-zero fit omitted the correction; refit before using a nonzero gamma")
            delta = self._prediction_matrix(self.correction_model_.predict(x), len(x), "correction prediction")
            prediction = baseline + mixing * delta
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite additive prediction")
        return prediction[:, 0] if self.single_target_ else prediction

    def metadata(self):
        self._require_fitted()
        raw = self.raw_model_.metadata()
        correction = None if self.correction_model_ is None else self.correction_model_.metadata()
        return {
            "architecture": "frozen_raw_plus_local_ridge_residual",
            "gate": self.gate, "alpha": self.alpha, "gamma": self.gamma, "seed": self.seed,
            "correction_alpha_applies": self.gamma != 0,
            "input_dimension": self.n_features_in_, "target_dimension": self.n_targets_,
            "n_training_rows": self.n_training_rows_,
            "training_input_sha256": self.training_input_sha256_,
            "training_target_sha256": self.training_target_sha256_,
            "raw": raw, "correction": correction,
            "raw_basis_width": raw["basis_width"],
            "correction_basis_width": 0 if correction is None else correction["basis_width"],
            "correction_has_nystrom_basis": False,
            "raw_coefficient_count": self.raw_coefficient_count_,
            "correction_coefficient_count": self.correction_coefficient_count_,
            "coefficient_count": self.coefficient_count_,
            "stored_reference_rows": self.stored_reference_rows_,
            "fitted_kahm_kernels": self.fitted_kahm_kernels_,
            "complexity_scope": "Stored fitted coefficient/intercept counts and reference rows (including repeated references across blocks); not equal statistical capacity, runtime, or total bytes.",
            "baseline_exact_at_gamma_zero": True,
            "residual_fit_scope": "In-sample residuals from the frozen raw predictor on the same training observations; not cross-fitted.",
            "tuning_scope": self.tuning_scope_,
            "guarantee_scope": "Preserves a fitted baseline at gamma zero; no test-risk nondegradation or risk certificate is claimed.",
        }
