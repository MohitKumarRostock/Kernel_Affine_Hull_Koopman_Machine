"""Evaluate the original i.i.d.-pair certificate on frozen KAHM models.

This module performs no representation fitting and no random sampling.

Inputs
------
current_states:
    Physical current states with shape (D, M).

successor_states:
    Physical one-step successor states with shape (D, M).

The frozen KAHM map returns association arrays with shape (C, M). The
validated certificate-statistics implementation instead consumes row-major
simplex arrays with shape (M, C), so this module transposes explicitly.

The hard reference class is assigned independently of the soft coordinates
using ``nearest_stored_kmeans_center_v1``.

The fitted KAHM one-step predictor uses the orientation established by the
training implementation:

    Z_hat = B.T @ Phi

where Phi and Z_hat have shape (C, M).

The vector MSE is

    mean_i ||Z_hat_i - Z_i||_2^2

and vector RMSE is its square root. These are absolute association-space
errors, not the relative closure error used by some historical experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kahkm_certificate_statistics import (
    PairStatistics,
    statistics_from_pairs,
)
from kahkm_certificates import (
    CertificateResult,
    certificate_from_statistics,
)
from kahkm_reference_classes import (
    REFERENCE_CLASS_MAP_ID,
    reference_class_labels,
)
from kernel_affine_hull_koopman_machines import (
    kahm_associations,
)


DYNAMICAL_EVALUATION_ID = "frozen_kahkm_original_iid_evaluation_v1"


@dataclass(frozen=True)
class BudgetCertificate:
    """Certificate and fitted-predictor feasibility at one norm budget."""

    kappa: float
    frozen_predictor_within_budget: bool
    certificate: CertificateResult
    frozen_predictor_rmse_minus_bound: float


@dataclass(frozen=True)
class DynamicalEvaluationResult:
    """Complete deterministic evaluation of one retained pair dataset."""

    evaluation_id: str
    reference_class_map_id: str
    n_pairs: int
    n_classes: int
    state_dimension: int
    delta: float
    frozen_predictor_spectral_norm: float
    frozen_predictor_mse: float
    frozen_predictor_rmse: float
    statistics: PairStatistics
    budget_certificates: tuple[BudgetCertificate, ...]
    phi_rows: NDArray[np.float64]
    successor_rows: NDArray[np.float64]
    labels: tuple[int, ...]
    selected_center_squared_distances: tuple[float, ...]
    phi_coordinate_min: float
    phi_coordinate_max: float
    successor_coordinate_min: float
    successor_coordinate_max: float
    current_state_min: tuple[float, ...]
    current_state_max: tuple[float, ...]
    successor_state_min: tuple[float, ...]
    successor_state_max: tuple[float, ...]


def _finite_matrix(
    name: str,
    value: ArrayLike,
) -> NDArray[np.float64]:
    raw = np.asarray(
        value
    )

    if raw.dtype.kind not in "iuf":
        raise TypeError(
            f"{name} must contain real numeric values."
        )

    if raw.ndim != 2:
        raise ValueError(
            f"{name} must have shape (D, M)."
        )

    if (
        raw.shape[0] <= 0
        or raw.shape[1] <= 0
    ):
        raise ValueError(
            f"{name} must have nonzero dimensions."
        )

    array = np.asarray(
        raw,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _finite_real(
    name: str,
    value: Real,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            Real,
        )
    ):
        raise TypeError(
            f"{name} must be a real scalar, not boolean."
        )

    result = float(
        value
    )

    if not np.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _validated_kappas(
    values: Sequence[Real],
) -> tuple[float, ...]:
    if isinstance(
        values,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "kappas must be a non-string sequence."
        )

    if not isinstance(
        values,
        Sequence,
    ):
        raise TypeError(
            "kappas must be a sequence."
        )

    if len(values) == 0:
        raise ValueError(
            "kappas must be non-empty."
        )

    normalized = tuple(
        _finite_real(
            f"kappas[{index}]",
            value,
        )
        for index, value
        in enumerate(values)
    )

    if any(
        value < 0.0
        for value in normalized
    ):
        raise ValueError(
            "Every kappa must be nonnegative."
        )

    if len(
        set(normalized)
    ) != len(normalized):
        raise ValueError(
            "kappas must not contain duplicate binary64 values."
        )

    if tuple(
        sorted(
            normalized
        )
    ) != normalized:
        raise ValueError(
            "kappas must be sorted in ascending order."
        )

    return normalized


def _model_scalar(
    abstraction_model: Mapping[str, Any],
    key: str,
    supplied: float,
) -> None:
    if key not in abstraction_model:
        return

    model_value = _finite_real(
        f"abstraction_model[{key!r}]",
        abstraction_model[
            key
        ],
    )

    if model_value != supplied:
        raise ValueError(
            f"Frozen model {key}={model_value!r} "
            f"does not equal supplied value {supplied!r}."
        )


def evaluate_frozen_dynamical_pairs(
    *,
    abstraction_model: Mapping[str, Any],
    B: ArrayLike,
    omega: Real,
    tau: Real,
    current_states: ArrayLike,
    successor_states: ArrayLike,
    kappas: Sequence[Real],
    delta: Real,
    n_jobs: int = 1,
    batch_size: int | None = 256,
) -> DynamicalEvaluationResult:
    """Evaluate one already-generated independent-pair dataset.

    No data are sampled and no model is fitted.

    The same empirical f_hat and s_hat are reused for all supplied norm
    budgets. This is appropriate because the concentration event underlying
    the original certificate is independent of kappa.
    """
    if not isinstance(
        abstraction_model,
        Mapping,
    ):
        raise TypeError(
            "abstraction_model must be a mapping."
        )

    current = _finite_matrix(
        "current_states",
        current_states,
    )

    successor_state = _finite_matrix(
        "successor_states",
        successor_states,
    )

    if (
        current.shape
        != successor_state.shape
    ):
        raise ValueError(
            "current_states and successor_states "
            "must have the same (D, M) shape."
        )

    state_dimension, n_pairs = (
        current.shape
    )

    omega_value = _finite_real(
        "omega",
        omega,
    )

    tau_value = _finite_real(
        "tau",
        tau,
    )

    if omega_value <= 0.0:
        raise ValueError(
            "omega must be positive."
        )

    if tau_value <= 0.0:
        raise ValueError(
            "tau must be positive."
        )

    _model_scalar(
        abstraction_model,
        "kahkm_omega",
        omega_value,
    )

    _model_scalar(
        abstraction_model,
        "kahkm_tau",
        tau_value,
    )

    delta_value = _finite_real(
        "delta",
        delta,
    )

    if not (
        0.0
        < delta_value
        < 1.0
    ):
        raise ValueError(
            "delta must lie strictly between 0 and 1."
        )

    kappa_values = (
        _validated_kappas(
            kappas
        )
    )

    labels_result = (
        reference_class_labels(
            abstraction_model=dict(
                abstraction_model
            ),
            X=current,
        )
    )

    phi_columns = np.asarray(
        kahm_associations(
            dict(
                abstraction_model
            ),
            current,
            omega=omega_value,
            tau=tau_value,
            n_jobs=int(
                n_jobs
            ),
            batch_size=batch_size,
            show_progress=False,
        ),
        dtype=np.float64,
    )

    successor_columns = np.asarray(
        kahm_associations(
            dict(
                abstraction_model
            ),
            successor_state,
            omega=omega_value,
            tau=tau_value,
            n_jobs=int(
                n_jobs
            ),
            batch_size=batch_size,
            show_progress=False,
        ),
        dtype=np.float64,
    )

    if (
        phi_columns.ndim != 2
        or successor_columns.ndim != 2
    ):
        raise RuntimeError(
            "KAHM associations must be two-dimensional."
        )

    if (
        phi_columns.shape
        != successor_columns.shape
    ):
        raise RuntimeError(
            "Current and successor association shapes differ."
        )

    if (
        phi_columns.shape[1]
        != n_pairs
    ):
        raise RuntimeError(
            "KAHM association pair count differs from physical-state pair count."
        )

    n_classes = int(
        phi_columns.shape[0]
    )

    if n_classes <= 0:
        raise RuntimeError(
            "KAHM returned zero association coordinates."
        )

    if (
        labels_result.n_classes
        != n_classes
    ):
        raise RuntimeError(
            "Reference-class count differs from KAHM association dimension."
        )

    if (
        labels_result.n_states
        != n_pairs
    ):
        raise RuntimeError(
            "Reference-class label count differs from pair count."
        )

    if (
        not np.all(
            np.isfinite(
                phi_columns
            )
        )
        or not np.all(
            np.isfinite(
                successor_columns
            )
        )
    ):
        raise RuntimeError(
            "KAHM association evaluation produced nonfinite values."
        )

    # The validated statistics API consumes rows = pairs, columns = classes.
    phi_rows = np.asarray(
        phi_columns.T,
        dtype=np.float64,
    )

    successor_rows = np.asarray(
        successor_columns.T,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels_result.labels,
        dtype=np.int64,
    )

    statistics = statistics_from_pairs(
        phi=phi_rows,
        successors=successor_rows,
        labels=labels,
    )

    # Retain the exact already-computed association coordinates for campaign
    # evidence. Store them in the same row-major (M, C) convention consumed by
    # statistics_from_pairs, and make the returned copies read-only so callers
    # cannot silently mutate an evaluation result after certification.
    retained_phi_rows = np.array(
        phi_rows,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    retained_successor_rows = np.array(
        successor_rows,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    retained_phi_rows.setflags(
        write=False
    )

    retained_successor_rows.setflags(
        write=False
    )

    operator = _finite_matrix(
        "B",
        B,
    )

    if operator.shape != (
        n_classes,
        n_classes,
    ):
        raise ValueError(
            "B must have shape (C, C) matching "
            f"the frozen association dimension; got "
            f"{operator.shape}, expected "
            f"{(n_classes, n_classes)}."
        )

    spectral_norm = float(
        np.linalg.norm(
            operator,
            ord=2,
        )
    )

    if not np.isfinite(
        spectral_norm
    ):
        raise RuntimeError(
            "Frozen predictor spectral norm is nonfinite."
        )

    # Orientation is fixed by fit_kahkm: pred = B.T @ Phi.
    prediction_columns = (
        operator.T
        @ phi_columns
    )

    residual_columns = (
        prediction_columns
        - successor_columns
    )

    per_pair_squared_error = np.sum(
        residual_columns
        * residual_columns,
        axis=0,
        dtype=np.float64,
    )

    mse = float(
        np.mean(
            per_pair_squared_error,
            dtype=np.float64,
        )
    )

    if (
        not np.isfinite(
            mse
        )
        or mse < 0.0
    ):
        raise RuntimeError(
            "Frozen predictor MSE is invalid."
        )

    rmse = sqrt(
        mse
    )

    budget_results: list[
        BudgetCertificate
    ] = []

    for kappa_value in kappa_values:
        certificate = (
            certificate_from_statistics(
                f_hat=
                    statistics.f_hat,
                s_hat=
                    statistics.s_hat,
                n_pairs=
                    statistics.n_pairs,
                kappa=
                    kappa_value,
                delta=
                    delta_value,
            )
        )

        # Allow only a very small numerical tolerance when deciding whether
        # the already-fitted predictor itself belongs to the compared class.
        feasibility_atol = (
            64.0
            * np.finfo(
                np.float64
            ).eps
            * max(
                1.0,
                spectral_norm,
                kappa_value,
            )
        )

        within_budget = bool(
            spectral_norm
            <= kappa_value
            + feasibility_atol
        )

        budget_results.append(
            BudgetCertificate(
                kappa=
                    kappa_value,
                frozen_predictor_within_budget=
                    within_budget,
                certificate=
                    certificate,
                frozen_predictor_rmse_minus_bound=
                    float(
                        rmse
                        - certificate.L_kappa_delta
                    ),
            )
        )

    return DynamicalEvaluationResult(
        evaluation_id=
            DYNAMICAL_EVALUATION_ID,
        reference_class_map_id=
            REFERENCE_CLASS_MAP_ID,
        n_pairs=
            int(
                n_pairs
            ),
        n_classes=
            n_classes,
        state_dimension=
            int(
                state_dimension
            ),
        delta=
            delta_value,
        frozen_predictor_spectral_norm=
            spectral_norm,
        frozen_predictor_mse=
            mse,
        frozen_predictor_rmse=
            rmse,
        statistics=
            statistics,
        budget_certificates=
            tuple(
                budget_results
            ),
        phi_rows=
            retained_phi_rows,
        successor_rows=
            retained_successor_rows,
        labels=
            tuple(
                int(value)
                for value
                in labels
            ),
        selected_center_squared_distances=
            labels_result.selected_squared_distances,
        phi_coordinate_min=
            float(
                np.min(
                    phi_columns
                )
            ),
        phi_coordinate_max=
            float(
                np.max(
                    phi_columns
                )
            ),
        successor_coordinate_min=
            float(
                np.min(
                    successor_columns
                )
            ),
        successor_coordinate_max=
            float(
                np.max(
                    successor_columns
                )
            ),
        current_state_min=
            tuple(
                float(value)
                for value
                in np.min(
                    current,
                    axis=1,
                )
            ),
        current_state_max=
            tuple(
                float(value)
                for value
                in np.max(
                    current,
                    axis=1,
                )
            ),
        successor_state_min=
            tuple(
                float(value)
                for value
                in np.min(
                    successor_state,
                    axis=1,
                )
            ),
        successor_state_max=
            tuple(
                float(value)
                for value
                in np.max(
                    successor_state,
                    axis=1,
                )
            ),
    )
