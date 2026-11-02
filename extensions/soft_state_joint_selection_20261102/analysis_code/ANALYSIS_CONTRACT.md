# Frozen joint-selection analysis contract

This follow-up evaluates 16 representation-only forecasts: G, P, K and R in
each of the `shared`, `independent`, `standard_shared`, and
`standard_independent` views. G contains transformed memberships, P adds
transformed weighted local position, K adds transformed KAHM departures to P,
and R adds transformed positive-kernel departures to P. Only the underlying
membership block is simplex valued separately within each partition; the
transformed or enriched readout vector need not be simplex valued.

The joint grid is the Cartesian product of input transformations
`standard`, `minmax`, `tanh_z_1`, in that order, with native dimension caps
`1,2,4,8,12` ascending within each transformation. Candidate identifiers are
`{input_scaling}_cap{cap}`. Exact ties select the first candidate in this order.
The model separately selects the ridge penalty for every candidate, arm and
horizon on B. Candidate choice uses the equal-horizon mean of these B losses,
each divided by that horizon's A-target population variance.

## Selection views and paired comparator

| View | Eligible candidates | Geometry selector |
| --- | --- | --- |
| `shared` | All 15 scaler/cap pairs | P's B score, shared by all four arms |
| `independent` | All 15 scaler/cap pairs | Each arm's own B score |
| `standard_shared` | The five standardization candidates | P's B score, shared by all four arms |
| `standard_independent` | The five standardization candidates | Each arm's own B score |

All views reuse the same candidate fits. The standard-only views are paired
comparators on the **same new cohort**, not reuses of historical outcomes.
They distinguish the effect of enabling scaling selection from a change in
evaluation data. For independently selected arms, and for P in the shared
view, the joint candidate superset guarantees no larger selected B loss.
It does not guarantee smaller D loss. The shared K/R comparison concerns
enrichment of P-selected geometry; it does not independently optimize K or R.

All fitting uses A. Only B selects penalties and geometry. C has no role in
fitting, selection, calibration, stopping, or deciding which analysis to report.
D is terminal evaluation after complete fit and prediction seals.

## Losses and independent unit

For each horizon, compute mean squared error across all 1,664 D origins and
divide by the population variance (`ddof=0`) of that horizon's A targets,
floored at `1e-12`. Average the three horizons equally. The independent unit is
a complete repetition: average the six nonlinear law/noise task losses equally,
or the two linear-law/noise losses equally. No origin, horizon, noise condition,
local model, or task is counted as an independent replicate. The fixed cohort
contains 32 repetitions and 256 tasks under master seed `2026110204`. All tasks
must be present. These observations are not pooled with earlier cohorts.

## One primary family of six contrasts

For repetition r, define the margin-adjusted difference
`d_r = L_candidate,r - c * L_reference,r`.

| Candidate | Reference | Panel | c | Interpretation |
| --- | --- | --- | ---: | --- |
| `shared__P` | `shared__G` | Nonlinear | 0.97 | 3% superiority |
| `shared__K` | `shared__P` | Nonlinear | 0.97 | 3% superiority |
| `shared__K` | `shared__R` | Nonlinear | 0.97 | 3% superiority |
| `shared__K` | `shared__P` | Linear | 1.05 | 5% noninferiority |
| `independent__K` | `standard_independent__K` | Nonlinear | 0.97 | 3% superiority from enabling joint selection |
| `independent__K` | `standard_independent__K` | Linear | 1.05 | 5% noninferiority from enabling joint selection |

Each claim passes only if
`mean(d) + t.ppf(1 - 0.05/6, 31) * sd(d, ddof=1) / sqrt(32) < 0`.
All six results, including failures, are reported as one Bonferroni-adjusted
family. The Student approximation applies to repetition aggregates; these are
not distribution-free risk certificates. The 3% and 5% thresholds are inherited
research conventions. Failure does not establish equality or exclude a smaller
benefit. A ratio of mean losses is a descriptive percentage, not a ratio
confidence bound. If the reference mean is exactly zero, the percentage is
reported as undefined while the paired difference remains defined.

The first four questions retain the representation interpretation. The final
two address the optimization question directly, with K choosing its own
geometry in both procedures. They are declared before new data generation.

## Secondary reporting

Report all 16 method means, every law/noise condition and horizon, selected
scaler/cap pairs, their marginal frequencies, effective local ranks, feature
widths, full B candidate scores, C diagnostics, resource costs and numerical
diagnostics. The fixed secondary comparison roster contains:

- P/G, K/P, K/R and R/P within each of the four views.
- Independent/shared for each arm within both joint and standard-only search.
- Joint/standard-only for every arm within both shared and independent views.

These 32 comparisons are shown separately for nonlinear and linear panels,
giving 64 descriptive rows. Secondary two-sided 95% Student intervals are
unadjusted and descriptive; even when they overlap a primary pair, the primary
decision uses the declared margin and six-comparison adjustment. They cannot
replace a failed primary criterion or justify a new subset claim. Independent
K/R superiority is not a primary claim in this follow-up.

The optimization tables additionally retain every paired task and repetition
loss, selected-geometry changes, B and D differences, and repetition-level
wins/ties/losses. Primary difference distributions report raw paired differences
and quantiles for transparency; these summaries do not create extra tests.

## Integrity and output

The analysis requires the source freeze and publication receipt, full fit seal,
full prediction seal, and terminal target release. It computes losses directly
from released targets and frozen prediction arrays, verifies their hashes and
the full factorial roster, checks selections against the recorded B first
minima, and refuses to overwrite an existing result directory. It checks that
the standard-only views use exactly the standard subset and that shared views
use one candidate for all arms.

The summary records the executing analyzer digest, source/configuration seals,
target-release digest, and row counts and hashes for every table. Independent
auditing separately reconstructs gates, features, selected ridge forecasts,
B selection losses and primary statistics without importing this analyzer or
the production model. Manufactured statistical checks exercise known positive,
negative and boundary contrasts, equal-cell aggregation, the six-claim family,
cross-context references and incomplete/duplicated/nonfinite data rejection.
