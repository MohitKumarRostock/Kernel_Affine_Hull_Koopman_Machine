# Local subspace dimension and risk certificates for retained KAHM representations

## Question and scope

This study examines how local KAHM subspace dimension affects forecasting error
and the utility of finite-sample risk certificates for retained representation
blocks. The compared policies are a fixed cap of two, the native cap of twenty,
and a cap selected on independent validation episodes. The initial request at
each local encoder is `min(cap, d, N_local - 1)`; native positive-eigenvalue and
projected-range rules determine the stored rank. Requested caps and actual
local ranks are recorded separately.

The study concerns the block-direct decoder on six bounded synthetic laws.
It does not evaluate every decoder, preprocessing variant, or natural-data
cohort in the broader empirical program. The two-dimensional learned-geometry
mechanism and its larger-block certificate experiments retain their original
scope: their ambient dimension already permits the declared rank of two.

## Episodes and development

Harmonic, quasiperiodic, nonlinear autoregressive, and logistic laws each have
five independently generated repetitions. Van der Pol and Duffing each have
three. Each repetition uses 128 training, 512 validation, 4096 calibration, and
2048 test episodes of length 72. The generators, numerical oscillator law, and
bounded sensor observable are the archived implementations. Their source bytes
are included in the runtime manifest.

The master seed is 2026110104. NumPy `SeedSequence` derives separate numerical
streams for each law, repetition, and split. The frozen schedule records the
resulting integer seeds before observations are generated. Histories of lengths
4 and 12 share the same thirteen forecast origins, beginning at index 11 with
stride 4. The horizons are 1, 4, and 12. Statistical scores average equally over
origins within each independent episode; overlapping origins are not treated as
independent observations.

Coordinate standardization is fitted on training histories and then fixed.
Each bank retains blocks with 2, 4, 8, and 16 coordinates, giving prefixes of
dimensions 2, 6, 14, and 30. Geometry uses at most 1600 training histories, with
at most 24 references per local KAHM fit. The affinity exponent is 16. A common
geometry seed within a history length gives the same partitions and reference
sampling across requested caps. The decoder uses ridge penalty 0.0001 and
sum-of-block Frobenius norm budget 4. The optimizer tolerance is 1e-7 and its
iteration limit is 20000. Feasible iterates remain eligible if that tolerance is
not reached; optimization gaps and budget residuals are retained.

For each KAHM policy, validation selects one complete `(history, cap)` bank by
minimizing the mean validation MSE of its best prefix over all three horizons.
Ties use fewer retained coordinates, shorter history, smaller requested cap,
and candidate name, in that order. The two fixed policies admit only cap 2 or
cap 20 respectively. The selected policy considers caps 1, 2, 4, 8, 12, and 20.
All four prefixes of the chosen bank are then retained for certification. The
validation-selected prefix is identified separately as the forecasting model.
Thus the four certified classes within an arm share one bank and remain nested.
The policy does not choose a different geometry independently at each prefix.

Distinct RBF and affine-hull banks use their archived definitions, including
the affine comparator's rank-two SVD normal-direction rule. Each selects one
history by the same best-prefix validation criterion and retains all four
prefixes. These are separate geometric comparators, not KAHMs with matched
local bases. Ridge autoregression and quadratic regression each select a
history on validation; persistence and the training-mean model complete the
family. There are 70 development candidates and 24 evaluated models per
repetition. Equal fitted candidates may share stored numerical arrays, but all
24 declared model aliases retain their allocated certificate statements.

## Ordering and independence

The protocol, schedule, configuration, and executable source hashes are sealed
before development. All 26 development fits, validation choices, fitted model
files, and candidate descriptions are then sealed before any calibration
episode is generated. Calibration evaluates the complete frozen family and all
planned comparisons. Every calibration result and decision is sealed before
test episodes are generated or scored. No calibration or test statistic enters
the selection of a cap, history, prefix, decoder, or representation.

The runtime rejects altered frozen inputs, incomplete prerequisite stages, and
overwrites of completed attempts. A failed numerical fit is preserved as a
failure rather than silently replaced by a different configuration. Only
manufactured fixtures are used for implementation checks before the study
freeze; they are not draws from the scheduled production streams.

## Endpoints and simultaneous allocation

Forecasts and targets lie in `[-1, 1]`, so episode-average squared loss is in
`[0, 4]`. An upper endpoint bounds the population loss of the fitted feasible
forecast. Disjoint episode pairs provide a lower witness for the entire
Lipschitz class with the frozen budget and feature metric. Its score is the
origin-average of

`0.5 * max(abs(Y - Y') - kappa * distance(F, F'), 0)**2`.

For retained block decoders, `distance` is the maximum Euclidean block distance.
The witness lies in `[0, 2]`. Ridge and quadratic direct models use their
Euclidean feature metric; persistence has budget one and the constant map has
budget zero. Feature-map choices, norm budgets, and clipping are fixed before
calibration. Clipping is nonexpansive.

Empirical-Bernstein calibration uses the sample variance across independent
episodes for fitted losses, and across independent disjoint episode pairs for
lower witnesses. Per repetition, 24 models and three horizons yield 144
designated one-sided risk endpoints: one fitted-loss upper and one witness
lower endpoint per model and horizon.

Nine adjacent-prefix fitted-loss comparisons concern the three KAHM banks.
Three further comparisons contrast the validation-selected forecasts:
cap 2 versus native 20, cap 2 versus selected cap, and native 20 versus selected
cap. Each has three horizon-specific lower endpoints for expected old loss
minus new loss. The shifted loss difference lies in `[0, 8]`. There are 36
designated one-sided improvement endpoints per repetition. These comparisons
concern fitted forecasts, not the difference between unknown class optima.

A single failure probability 0.05 covers the complete campaign. Allocation
0.04 over `26 * 144 = 3744` risk endpoints and 0.01 over `26 * 36 = 936`
improvement endpoints gives the same per-endpoint probability
`0.05 / 4680`. Only the designated tails belong to this event; unused opposite
tails are not reported as additional confidence claims. Equal models do not
reduce the frozen denominator. Endpoint maxima or tolerance decisions computed
from this simultaneous family require no additional allocation.

This campaign-wide scope differs from the per-repetition allocation of the
earlier broad forecasting study. Comparisons within this study use identical
allocation. Changes in width relative to earlier studies cannot be attributed
to dimension alone.

## Outcomes and interpretation

The primary forecasting outcomes are test MSE by law, policy, and horizon,
and the mean over the three horizons within each law. Results retain every
scheduled repetition, including unfavorable effects. Across-law averages are
not interpreted as a common physical risk. The analysis reports selected caps,
histories, prefixes, actual local ranks, optimization gaps, runtime, and all
prespecified pairwise contrasts.

Certificate outcomes include lower and upper endpoints, endpoint separation,
zero lower endpoints, upper endpoints reaching the loss bound, and decisions
at RMSE tolerances 0.2, 0.35, and 0.5. A model is accepted when every horizon's
upper endpoint is below the squared tolerance, and its class is excluded when
any horizon's lower endpoint exceeds that tolerance. Other outcomes remain
unresolved. If both conditions occur, the record is retained with the explicit
diagnostic `conflicting_bounds`; a finite-sample confidence event can fail, so
such a result is not silently discarded or converted to an execution failure.
A zero lower endpoint does not establish absence of a population obstruction.

The unknown class-optimal risks are not observed on these laws. Independent
test errors therefore measure forecasting utility and provide consistency
diagnostics; their containment in an interval is not an estimate of true
coverage for the class optimum. The unchanged controlled studies supply the
separate known-risk evidence. The campaign's confidence statement follows its
sampling assumptions and fixed-family allocation, rather than a finite test
sample acting as population truth.

All validation scores, calibration losses, disjoint-pair quantities, test
losses, fitted-model provenance, and rank records are archived. Standalone
array analysis recomputes selections, designated confidence endpoints,
comparisons, and decisions without importing the KAHM runtime or unpickling
models. Coefficients, intercepts, and block slices permit a separate norm-budget
check. Complete calibration and test predictions and targets are additionally
retained for the prespecified repetition numbered zero of each of the six laws.
These six full-array joins check prediction-to-loss wiring; the full campaign
has a score-level audit rather than an independent reconstruction of every
forecast. This storage rule is fixed before outcomes and does not change the
scientific evaluation roster. Every split retains episode and origin indices,
generation seeds, array hashes, and prediction-range diagnostics.
