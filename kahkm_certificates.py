"""Original i.i.d.-pair exclusion certificate.

Source: arXiv:2609.32652v1, Proposition 2 and Eq. (8), pp. 6-7.

Inputs are statistics for M independent, identically distributed evaluation
pairs. The representation and class map must be fixed independently of those
pairs. The guarantee is simultaneous over kappa, not over adaptively selected
representations or inspected sample sizes. Sampling assumptions cannot be
verified from summary statistics by this module.

f_hat = mean((1 - phi_i[c_i])**2).
s_hat = mean(||Z_i - mean(Z within current class c_i)||_2**2).
Use denominator M, not an unbiased variance correction. The returned bound
is vector RMSE, with no division by the number of coordinates.
"""

from dataclasses import dataclass
from math import isfinite, log, sqrt
from numbers import Integral, Real

CERTIFICATE_ID = "original_iid_eq8_v1"


@dataclass(frozen=True)
class CertificateResult:
    """Inputs and intermediate quantities, using the manuscript's notation."""

    n_pairs: int
    kappa: float
    delta: float
    f_hat: float
    s_hat: float
    r_delta: float
    F_delta: float
    V_delta: float
    L_kappa_delta: float


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite and float-representable."
        ) from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def certificate_from_statistics(
    *,
    f_hat: Real,
    s_hat: Real,
    n_pairs: Integral,
    kappa: Real,
    delta: Real,
) -> CertificateResult:
    """Evaluate Eq. (8); neither fit a predictor nor sample data.

    n_pairs is the number of i.i.d. evaluation pairs, not the number of
    correlated transitions. kappa is a spectral-norm budget and delta is
    the failure probability. Both empirical statistics must lie in [0, 1].
    Inputs are checked, not silently clipped into admissible ranges.
    """
    if isinstance(n_pairs, bool) or not isinstance(n_pairs, Integral):
        raise TypeError("n_pairs must be a positive integer, not a boolean.")
    M = int(n_pairs)
    if M <= 0:
        raise ValueError("n_pairs must be positive.")

    f = _finite_real("f_hat", f_hat)
    s = _finite_real("s_hat", s_hat)
    k = _finite_real("kappa", kappa)
    d = _finite_real("delta", delta)

    if not 0.0 <= f <= 1.0:
        raise ValueError("f_hat must lie in [0, 1].")
    if not 0.0 <= s <= 1.0:
        raise ValueError("s_hat must lie in [0, 1].")
    if k < 0.0:
        raise ValueError("kappa must be nonnegative.")
    if not 0.0 < d < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1.")

    # Avoid overflow in 2/delta when delta is very small.
    r = (log(2.0) - log(d)) / M

    F = min(1.0, f + r + sqrt(r * r + 2.0 * r * f))
    V = max(0.0, s - sqrt(2.0 * r))
    L = max(0.0, sqrt(V) - k * sqrt(2.0 * F))

    return CertificateResult(
        n_pairs=M,
        kappa=k,
        delta=d,
        f_hat=f,
        s_hat=s,
        r_delta=r,
        F_delta=F,
        V_delta=V,
        L_kappa_delta=L,
    )
