"""Compact empirical statistics for the fixed four-state reference.

For state counts (n_a,n_b,n_d,n_e), M is their sum and n_0=n_a+n_b.
Every assignment loss equals (1-p)^2, regardless of the sampled counts.
For conflicting successors, class 0 contributes
    SSE_0 = 2*(2*p-1)^2*n_a*n_b/n_0
when n_0>0; class 1 contributes zero. Identity has zero SSE in both classes.
s_hat = SSE_0/M. Empty classes contribute zero, not missing values.

These identities evaluate the empirical statistics, not their expectations.
Counts and the fixed model reconstruct the unordered evaluation sample.
Order is unnecessary for these statistics; floating-point summation can
differ from expanded row-by-row evaluation. No sampling or independence
verification is performed here. For an i.i.d. uniform-state experiment,
counts must be sampled from Multinomial(M; 1/4,1/4,1/4,1/4), not fixed
to balanced values.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from kahkm_certificate_statistics import PairStatistics, SIMPLEX_SUM_ATOL
from kahkm_four_state_reference import FourStateReference, four_state_reference

COUNT_STATISTICS_ID = "four_state_count_statistics_v1"
MAX_EXACT_PAIR_COUNT = 2**53


@dataclass(frozen=True)
class FourStateCountStatistics:
    """Retain the four input counts alongside the computed pair statistics."""

    statistics_id: str
    state_counts: tuple[int, ...]
    statistics: PairStatistics


def statistics_from_state_counts(
    *, reference: FourStateReference, state_counts: ArrayLike
) -> FourStateCountStatistics:
    """Compute without constructing M rows or altering the supplied counts.

    reference must be an unmodified output of four_state_reference.
    Counts must be four nonnegative integers in the order (a,b,d,e).
    Their sum must be positive and at most 2**53 so all counts and their
    total are exactly representable when converted to binary64.
    Booleans, floating-point counts, and masked arrays are rejected.

    Save this result together with the reference/configuration and source
    commit. Counts alone do not specify p, the dynamics, or the sampling law.
    """
    if not isinstance(reference, FourStateReference):
        raise TypeError("reference must be a FourStateReference.")
    canonical = four_state_reference(
        p=reference.p, dynamics=reference.dynamics, kappa=reference.kappa
    )
    if reference != canonical:
        raise ValueError("reference must be an unmodified factory result.")

    if np.ma.isMaskedArray(state_counts):
        raise TypeError("state_counts must not be a masked array.")
    try:
        values = tuple(state_counts)
    except TypeError as exc:
        raise TypeError("state_counts must contain four integers.") from exc
    if len(values) != 4:
        raise ValueError("state_counts must contain exactly four elements.")
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
        for x in values
    ):
        raise TypeError("Each state count must be an integer, not a boolean.")

    # Convert before arithmetic: NumPy fixed-width integer sums can overflow.
    counts = tuple(int(x) for x in values)
    if any(x < 0 for x in counts):
        raise ValueError("State counts must be nonnegative.")
    M = sum(counts)
    if not 1 <= M <= MAX_EXACT_PAIR_COUNT:
        raise ValueError("The total count must lie in [1, 2**53].")

    n_a, n_b, n_d, n_e = counts
    n_0, n_1 = n_a + n_b, n_d + n_e
    p, q = reference.p, 1.0 - reference.p
    v, u = (p, q), (q, p)
    empty_mean = (0.5, 0.5)
    mean_0 = empty_mean
    mean_1 = u if n_1 else empty_mean
    sse_0 = 0.0

    if n_0:
        if reference.dynamics == "identity":
            mean_0 = v
        else:
            weight_a, weight_b = n_a / n_0, n_b / n_0
            mean_0 = (
                weight_a * p + weight_b * q,
                weight_a * q + weight_b * p,
            )
            contrast = 2.0 * p - 1.0
            # Python integer multiplication is exact before true division.
            sse_0 = 2.0 * contrast * contrast * ((n_a * n_b) / n_0)

    # These errors describe the represented input rows, not the class means.
    occupied_states = [i for i, n in enumerate(counts) if n]
    current_error = max(
        abs(sum(reference.phi[i]) - 1.0) for i in occupied_states
    )
    successor_error = max(
        abs(sum(reference.successors[i]) - 1.0) for i in occupied_states
    )

    statistics = PairStatistics(
        n_pairs=M,
        n_classes=2,
        f_hat=q * q,
        s_hat=sse_0 / M,
        class_counts=(n_0, n_1),
        class_successor_means=(mean_0, mean_1),
        class_successor_sse=(sse_0, 0.0),
        current_max_row_sum_error=current_error,
        successor_max_row_sum_error=successor_error,
        simplex_sum_atol=SIMPLEX_SUM_ATOL,
    )
    return FourStateCountStatistics(
        statistics_id=COUNT_STATISTICS_ID,
        state_counts=counts,
        statistics=statistics,
    )
