"""Scratch-only KAHM preprocessing ablation; inherited sources stay untouched.

``standard`` is the exact inherited class. Other arms replace preprocessing
only for ``kind='kahm'``. The original geometry and decoder code is reused.
Min/max statistics use all training histories before the original seeded
geometry subsampling. The primary ``minmax`` arm does not clip held-out data;
``minmax_clip`` is a separate, explicitly nonlinear diagnostic.

Transform diagnostics count evaluated rows, including repeated evaluations
of the same rows for different candidate prefixes. They are not independent
sample counts or rates over unique observations.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import numpy as np

from certified_forecasting.features import NestedSoftFeatures, _matrix


MODES = ("standard", "minmax", "minmax_clip", "raw")
FITTED_BANKS = []
_PHASE = ContextVar("kahm_scaling_phase", default="unlabeled")


@contextmanager
def scaling_phase(label):
    """Label transform diagnostics without changing any fitted calculation."""
    token = _PHASE.set(str(label))
    try:
        yield
    finally:
        _PHASE.reset(token)


def reset_fitted_banks():
    FITTED_BANKS.clear()


class _ScalingSoftFeatures(NestedSoftFeatures):
    input_scaling_mode = "raw"

    def fit(self, X):
        X = _matrix(X)
        self.raw_train_min_ = X.min(axis=0)
        self.raw_train_max_ = X.max(axis=0)
        self.raw_train_mean_ = X.mean(axis=0)
        self.raw_train_std_ = X.std(axis=0)
        self.raw_train_range_ = self.raw_train_max_ - self.raw_train_min_
        if not np.isfinite(self.raw_train_range_).all():
            raise ValueError("Training coordinate ranges must be finite.")
        self.constant_features_ = self.raw_train_range_ == 0.0
        self.raw_train_n_ = int(len(X))
        self.scaling_diagnostics_ = {}
        self.effective_input_scaling_ = (
            self.input_scaling_mode if self.kind == "kahm"
            else "standard" if self.standardize else "raw"
        )
        self._input_scaler_fitted_ = True
        if self.kind != "kahm":
            # Preserve the original baseline path and numerical operation order.
            super().fit(X)
        else:
            # Scaling is fully external to the inherited geometry implementation.
            self.standardize = False
            Z = self._preprocess_kahm(X)
            self._record(X, Z, phase="fit_input")
            super().fit(Z)
        FITTED_BANKS.append(self)
        return self

    def _preprocess_kahm(self, X):
        if self.input_scaling_mode == "raw":
            return X
        span = np.where(self.constant_features_, 1.0, self.raw_train_range_)
        Z = 2.0 * ((X - self.raw_train_min_) / span) - 1.0
        # An unidentifiable training-constant coordinate contributes no geometry.
        Z[:, self.constant_features_] = 0.0
        if not np.isfinite(Z).all():
            raise ValueError("Scaled inputs must be finite.")
        return Z

    def _record(self, X, preclip_Z, phase=None):
        label = str(_PHASE.get() if phase is None else phase)
        outside = (preclip_Z < -1.0) | (preclip_Z > 1.0)
        outside_training = ((X < self.raw_train_min_) |
                            (X > self.raw_train_max_))
        rec = self.scaling_diagnostics_.setdefault(label, {
            "calls": 0, "evaluated_rows_including_repeats": 0,
            "evaluated_coordinates_including_repeats": 0,
            "outside_minus1_plus1_coordinates": 0,
            "rows_with_outside_minus1_plus1_coordinate": 0,
            "outside_training_extrema_coordinates": 0,
            "rows_outside_training_extrema": 0,
            "clipped_coordinates": 0,
            "preclip_min": float(preclip_Z.min()),
            "preclip_max": float(preclip_Z.max()),
        })
        rec["calls"] += 1
        rec["evaluated_rows_including_repeats"] += int(len(X))
        rec["evaluated_coordinates_including_repeats"] += int(X.size)
        rec["outside_minus1_plus1_coordinates"] += int(outside.sum())
        rec["rows_with_outside_minus1_plus1_coordinate"] += int(outside.any(axis=1).sum())
        rec["outside_training_extrema_coordinates"] += int(outside_training.sum())
        rec["rows_outside_training_extrema"] += int(outside_training.any(axis=1).sum())
        if self.input_scaling_mode == "minmax_clip":
            rec["clipped_coordinates"] += int(outside.sum())
        rec["preclip_min"] = min(rec["preclip_min"], float(preclip_Z.min()))
        rec["preclip_max"] = max(rec["preclip_max"], float(preclip_Z.max()))

    def preprocess_inputs(self, X, *, record=False, phase=None):
        """Return the actual geometry inputs without fitting or updating extrema."""
        X = _matrix(X)
        if not getattr(self, "_input_scaler_fitted_", False):
            raise ValueError("Fit the feature map before preprocessing inputs.")
        if X.shape[1] != len(self.raw_train_min_):
            raise ValueError("Input dimension differs from the fitted feature map.")
        if self.kind != "kahm":
            return (X - self.mean_) / self.scale_
        Z = self._preprocess_kahm(X)
        if record:
            self._record(X, Z, phase=phase)
        return np.clip(Z, -1.0, 1.0) if self.input_scaling_mode == "minmax_clip" else Z

    def transform(self, X, n_blocks=None):
        if self.kind != "kahm":
            return super().transform(X, n_blocks=n_blocks)
        Z = self.preprocess_inputs(X, record=True)
        # Inherited mean_/scale_ are exactly zero/one for these geometry banks.
        return super().transform(Z, n_blocks=n_blocks)

    def scaling_metadata(self):
        """JSON-safe scaler parameters and invocation-weighted diagnostics."""
        if not hasattr(self, "blocks_"):
            raise ValueError("Fit before requesting scaling metadata.")
        return {
            "requested_mode": self.input_scaling_mode,
            "effective_mode": self.effective_input_scaling_,
            "kind": self.kind, "seed": self.seed,
            "n_input_features": self.n_features_in_,
            "training_rows_for_scaler": self.raw_train_n_,
            "training_rows_for_geometry": int(len(self.fit_indices_)),
            "fit_indices_sha256": hashlib.sha256(
                np.asarray(self.fit_indices_, dtype="<i8").tobytes()).hexdigest(),
            "training_min": self.raw_train_min_.tolist(),
            "training_max": self.raw_train_max_.tolist(),
            "training_mean": self.raw_train_mean_.tolist(),
            "training_std": self.raw_train_std_.tolist(),
            "training_range": self.raw_train_range_.tolist(),
            "constant_features": self.constant_features_.tolist(),
            "constant_feature_policy": "map to zero in both minmax arms",
            "inner_mean": self.mean_.tolist(),
            "inner_scale": self.scale_.tolist(),
            "clip": self.kind == "kahm" and self.input_scaling_mode == "minmax_clip",
            "formula": ("2*(x-training_min)/training_range-1"
                        if self.kind == "kahm" and self.input_scaling_mode.startswith("minmax")
                        else "identity" if self.kind == "kahm"
                        else "inherited preprocessing"),
            "diagnostic_unit": "transform evaluations including repeated candidate-prefix evaluations",
            "diagnostics": self.scaling_diagnostics_,
        }


class MinMaxSoftFeatures(_ScalingSoftFeatures):
    input_scaling_mode = "minmax"


class ClippedMinMaxSoftFeatures(_ScalingSoftFeatures):
    input_scaling_mode = "minmax_clip"


class RawSoftFeatures(_ScalingSoftFeatures):
    input_scaling_mode = "raw"


def make_feature_class(mode):
    """Return a pickle-safe callable for run_forecasting.NestedSoftFeatures."""
    classes = {
        "standard": NestedSoftFeatures,
        "minmax": MinMaxSoftFeatures,
        "minmax_clip": ClippedMinMaxSoftFeatures,
        "raw": RawSoftFeatures,
    }
    if mode not in classes:
        raise ValueError(f"Unknown scaling mode {mode!r}; expected one of {MODES}.")
    return classes[mode]


def install_scaling(mode, runner=None):
    """Install one arm in the supplied runner module, default run_forecasting."""
    if runner is None:
        import run_forecasting as runner
    runner.NestedSoftFeatures = make_feature_class(mode)
    reset_fitted_banks()
    return runner.NestedSoftFeatures
