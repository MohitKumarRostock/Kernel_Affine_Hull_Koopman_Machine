"""Uniform four-state reference from arXiv:2609.32652v1, Sec. 5.5.

State order: (a, b, d, e); current coordinates: (v, v, u, u), where
v=(p,1-p) and u=(1-p,p). Identity uses B=I. Conflicting successors are
(v,u,u,u), with conditional means (v+u)/2 and u. The irreducible MSE is
||v-u||^2/8 because the current representation is constant within classes.
The displayed B attains it and obeys ||B||_2 <= ||B||_F
= sqrt(p^2-p+3/2). The exact constrained optimum is returned only for
budgets covered by these sufficient norm bounds; smaller budgets are not
solved here. Coordinates at p=0.5 or p=1 are limiting test cases, not
claims of finite-sharpness KAHM constructions.

This module contains no simulation, fitting, confidence calculation, or I/O.
B follows the manuscript convention: column prediction B.T @ phi;
for data stored in rows, predictions are phi_rows @ B.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real

FOUR_STATE_REFERENCE_ID = "four_state_uniform_reference_v1"


@dataclass(frozen=True)
class FourStateReference:
    """Fixed model and analytical population quantities, not sample estimates."""

    reference_id: str
    dynamics: str
    p: float
    kappa: float
    state_names: tuple[str, ...]
    states: tuple[tuple[float, float], ...]
    probabilities: tuple[float, ...]
    labels: tuple[int, ...]
    phi: tuple[tuple[float, float], ...]
    successor_indices: tuple[int, ...]
    successors: tuple[tuple[float, float], ...]
    optimal_B: tuple[tuple[float, float], ...]
    sufficient_kappa: float
    assignment_loss: float
    sigma_res_sq: float
    exact_optimal_mse: float
    exact_optimal_rmse: float
    population_certificate_limit: float
    current_coordinate_variance: float


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and float-representable.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def four_state_reference(
    *, p: Real, dynamics: str, kappa: Real
) -> FourStateReference:
    """Return an analytical reference for a supported norm budget.

    Allowed dynamics: 'identity' and 'conflicting_successors'.
    Labels and successor indices are zero-based. State probabilities are
    fixed at 1/4; changing this law requires a separately validated reference.
    The sufficient budget is not claimed to be the smallest feasible one.
    Analytical formulas are evaluated in float64-compatible Python floats.
    """
    p_value = _finite_real("p", p)
    k_value = _finite_real("kappa", kappa)
    if not 0.5 <= p_value <= 1.0:
        raise ValueError("p must lie in [0.5, 1].")
    if k_value < 0.0:
        raise ValueError("kappa must be nonnegative.")
    if not isinstance(dynamics, str):
        raise TypeError("dynamics must be a string.")
    if dynamics not in ("identity", "conflicting_successors"):
        raise ValueError(
            "Unknown dynamics; use 'identity' or 'conflicting_successors'."
        )

    q = 1.0 - p_value
    contrast = 2.0 * p_value - 1.0
    v, u = (p_value, q), (q, p_value)
    coordinates = (v, v, u, u)

    if dynamics == "identity":
        indices = (0, 1, 2, 3)
        B = ((1.0, 0.0), (0.0, 1.0))
        sufficient_kappa = 1.0
        optimum_mse = 0.0
    else:
        indices = (0, 2, 2, 2)
        B = (
            (1.0 - p_value / 2.0, p_value / 2.0),
            (q / 2.0, (1.0 + p_value) / 2.0),
        )
        sufficient_kappa = sqrt(p_value * p_value - p_value + 1.5)
        optimum_mse = contrast * contrast / 4.0

    if k_value < sufficient_kappa:
        raise NotImplementedError(
            "This reference establishes the exact constrained optimum only for "
            f"kappa >= {sufficient_kappa:.17g} in this case. Smaller budgets "
            "require a separate constrained-risk calculation."
        )

    optimum_rmse = sqrt(optimum_mse)
    limit = max(0.0, optimum_rmse - k_value * (sqrt(2.0) * q))

    return FourStateReference(
        reference_id=FOUR_STATE_REFERENCE_ID,
        dynamics=dynamics,
        p=p_value,
        kappa=k_value,
        state_names=("a", "b", "d", "e"),
        states=((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)),
        probabilities=(0.25, 0.25, 0.25, 0.25),
        labels=(0, 0, 1, 1),
        phi=coordinates,
        successor_indices=indices,
        successors=tuple(coordinates[j] for j in indices),
        optimal_B=B,
        sufficient_kappa=sufficient_kappa,
        assignment_loss=q * q,
        sigma_res_sq=optimum_mse,
        exact_optimal_mse=optimum_mse,
        exact_optimal_rmse=optimum_rmse,
        population_certificate_limit=limit,
        current_coordinate_variance=contrast * contrast / 2.0,
    )
