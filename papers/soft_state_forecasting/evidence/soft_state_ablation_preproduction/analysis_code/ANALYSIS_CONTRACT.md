# Frozen analysis contract

This analysis evaluates the prespecified eight representation-only forecasts:
`shared__G`, `shared__P`, `shared__K`, `shared__R`, and the corresponding
`independent__` views. G contains transformed memberships, P adds transformed
weighted local position, K adds transformed KAHM departures to P, and R adds
transformed positive-kernel departures to P. The underlying membership block
is simplex valued separately within each partition; the transformed or enriched
readout vector need not be simplex valued.

For each horizon, compute mean squared error across all 1,664 D origins and
divide by the population variance (`ddof=0`) of that horizon's A targets,
floored at `1e-12`. Average the three horizons equally. The independent unit is
a complete repetition: average the six nonlinear law/noise task losses equally,
or the two linear-law/noise losses equally. No origin, horizon, noise condition,
local model, or task is counted as an independent replicate. The fixed cohort
contains 32 repetitions and 256 tasks. All tasks must be present.

The primary shared-cap view uses the cap selected by P on B. At that cap, each
arm has its own B-selected penalty per horizon. The independent-cap view uses
each arm's own B-selected cap and penalties and remains secondary. C has no
role in fitting, selection, calibration, or deciding which analysis to report.

## One primary family

For repetition r, define the margin-adjusted difference
`d_r = L_candidate,r - c * L_reference,r`.

| Contrast | Panel | c | Interpretation |
|---|---|---:|---|
| P versus G | Nonlinear | 0.97 | 3% superiority |
| K versus P | Nonlinear | 0.97 | 3% superiority |
| K versus R | Nonlinear | 0.97 | 3% superiority |
| K versus P | Linear | 1.05 | 5% noninferiority |

Each claim passes only if
`mean(d) + t.ppf(1 - 0.05/4, 31) * sd(d, ddof=1) / sqrt(32) < 0`.
All four results, including failures, are reported. The Student approximation
applies to the repetition aggregates; these are not distribution-free risk
certificates. The 3% threshold is an inherited research convention. Failure
does not establish equality or exclude a smaller benefit. A ratio of mean
losses is a descriptive percentage, not a ratio confidence bound.

## Secondary reporting and integrity

Report all eight method means, each law/noise condition, selected caps,
effective local ranks and feature widths, resource costs, numerical diagnostics,
and the fixed set of secondary paired differences. Secondary two-sided 95%
Student intervals are unadjusted and descriptive. They cannot replace a failed
primary claim or justify a new subset claim.

The analysis requires the source freeze, full fit seal, full prediction seal,
and terminal target release. It computes losses directly from the released
target and frozen prediction arrays, verifies their hashes and the full
factorial roster, and refuses to overwrite an existing result directory.
Independent auditing separately reconstructs gates, features, selected ridge
forecasts, all B selection losses, and primary statistics without importing
this analyzer or the production model.
