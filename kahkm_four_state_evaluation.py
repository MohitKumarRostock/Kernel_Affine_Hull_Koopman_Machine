"""Evaluate retained four-state counts without drawing or redrawing samples.

Combine the validated count statistics, analytical reference, and original
certificate. This module performs no campaign execution, file I/O, fitting,
aggregation, confidence-interval calculation, or random sampling.

Coverage and exclusion flags describe one evaluation dataset, not estimated
probabilities. Repeated independent datasets are needed to estimate coverage
or exclusion power. The caller must establish the sampling and fixed-model
assumptions and retain configuration, seed metadata, and source provenance.

Comparisons use floating-point values of the analytical optimum. Report the
strict and tolerance-aware coverage checks separately; comparison_atol must
not change the certificate, strict coverage, or exclusion decisions.
The difference population_limit - finite_sample_bound can be negative.
Neither that difference nor optimum - finite_sample_bound is clipped.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from numpy.typing import ArrayLike

from kahkm_certificates import (
    CERTIFICATE_ID, CertificateResult, certificate_from_statistics,
)
from kahkm_four_state_counts import (
    FourStateCountStatistics, statistics_from_state_counts,
)
from kahkm_four_state_reference import FourStateReference

FOUR_STATE_EVALUATION_ID = "four_state_original_iid_evaluation_v1"


@dataclass(frozen=True)
class ExclusionDecision:
    """Strict exclusion at one tolerance and its analytical ground truth."""

    tolerance: float
    excluded: bool
    exact_error_exceeds_tolerance: bool
    erroneous_exclusion: bool


@dataclass(frozen=True)
class FourStateEvaluation:
    """One result; shared reference fields can be stored once per case."""

    evaluation_id: str
    reference: FourStateReference
    count_statistics: FourStateCountStatistics
    certificate_id: str
    certificate: CertificateResult
    comparison_atol: float
    strict_coverage: bool
    tolerance_aware_coverage: bool
    zero_certificate: bool
    gap_to_optimum: float
    population_bound_slack: float
    sampling_gap: float
    exclusions: tuple[ExclusionDecision, ...]


def _nonnegative_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and float-representable.") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def evaluate_four_state_counts(
    *, reference: FourStateReference, state_counts: ArrayLike, delta: Real,
    exclusion_tolerances: list[Real] | tuple[Real, ...], comparison_atol: Real,
) -> FourStateEvaluation:
    """Evaluate a fixed count vector and preserve any observed violations.

    Exclusion tolerances must be a nonempty ordered list or tuple of distinct,
    finite, nonnegative values. Their supplied order is preserved. A bound
    equal to a tolerance does NOT exclude that tolerance. All gaps are in
    vector-RMSE units, with no per-coordinate normalization.

    comparison_atol applies ONLY to L <= exact_optimal_rmse + comparison_atol.
    No counts, parameters, bounds, or signed gaps are adjusted. Exceptions
    propagate without a retry; the runner must retain the failed attempt
    and its already saved counts. This function never generates new counts.
    """
    atol = _nonnegative_real("comparison_atol", comparison_atol)
    if not isinstance(exclusion_tolerances, (list, tuple)):
        raise TypeError("exclusion_tolerances must be an ordered list or tuple.")
    if not exclusion_tolerances:
        raise ValueError("exclusion_tolerances must not be empty.")
    tolerances = tuple(
        _nonnegative_real(f"exclusion_tolerances[{index}]", value)
        for index, value in enumerate(exclusion_tolerances)
    )
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("exclusion_tolerances must be distinct.")

    counted = statistics_from_state_counts(
        reference=reference, state_counts=state_counts,
    )
    statistics = counted.statistics
    certificate = certificate_from_statistics(
        f_hat=statistics.f_hat, s_hat=statistics.s_hat,
        n_pairs=statistics.n_pairs, kappa=reference.kappa, delta=delta,
    )
    lower = certificate.L_kappa_delta
    optimum = reference.exact_optimal_rmse
    limit = reference.population_certificate_limit
    exclusions = tuple(
        ExclusionDecision(
            tolerance=tolerance,
            excluded=lower > tolerance,
            exact_error_exceeds_tolerance=optimum > tolerance,
            erroneous_exclusion=(lower > tolerance and optimum <= tolerance),
        )
        for tolerance in tolerances
    )

    return FourStateEvaluation(
        evaluation_id=FOUR_STATE_EVALUATION_ID,
        reference=reference,
        count_statistics=counted,
        certificate_id=CERTIFICATE_ID,
        certificate=certificate,
        comparison_atol=atol,
        strict_coverage=lower <= optimum,
        tolerance_aware_coverage=lower <= optimum + atol,
        zero_certificate=lower == 0.0,
        gap_to_optimum=optimum - lower,
        population_bound_slack=optimum - limit,
        sampling_gap=limit - lower,
        exclusions=exclusions,
    )
