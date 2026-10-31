# Independent protocol review: focused adaptive KAHM development study

Date: 2026-10-31. Review stage: before the reported accuracy run.

This review concerns the proposed bounded-adaptation prototype and development
decision. It is not an external registration, an independent experimental
replication, or an endorsement of an outcome that has not yet been observed.
No forecasting results from the new prototype were used in writing these notes.

## Overall assessment

The six-method comparison addresses the intended next question if every arm
uses the same causal observation stream and its complete service work is
measured. A favorable comparison with fixed KAHM or the retained baseline alone
would establish an adaptation benefit, but would not establish the value of
KAHM over matched adaptive geometry or a strong online kernel comparator.

The proposed advancement rule is appropriately conservative for deciding
whether to invest in a larger reserved-source study. It is not a statistical
test with a calibrated error rate. Failure to pass it means that this specific
prototype and recurrence hypothesis have not earned that next investment; it
does not disprove all possible KAHM forecasting constructions.

## Development data and interpretation

Use the four previously preserved real series: three Dominick weekly series and
one Sunspot daily series. Their selection is deliberately informed by prior
results and must be described as development. The current manuscript reports
KAHM gains over ridge on Dominick and essentially baseline-only behavior with
stronger raw RBF performance on Sunspot.

The preserved Dominick arrays have length 393, history length 52, offsets
`[1, 2, 4]`, and seasonal lag 52. The Sunspot array has length 4096, history
length 28, offsets `[1, 3, 7]`, and seasonal lag 1. Reconstruct the stream from
`raw_values`; the old A/B/C/D arrays include capped/subsampled rows and are not a
complete online replay, particularly for Sunspot.

Four paired generator seeds under stationary and recurring-regime schedules
provide eight synthetic streams. The paired generated stream, not each
overlapping forecast origin, is the synthetic comparison unit. Two chosen
natural collections provide descriptive source comparisons, not a basis for
broad source-population inference.

## Proposed procedure reviewed

- Six methods: baseline; fixed KAHM geometry with updated decoder; adaptive
  KAHM; matched adaptive affine; matched adaptive RBF-local; and a recursive
  sliding-window Gaussian kernel-ridge comparator.
- Two tiers: 64/128 total geometric references and 128/256 replay rows.
- Four local dictionaries, with at most four new references at a refresh of one
  local model every 16 observations; half of the initial references retained.
- Source-compatible normalized signed KAHM reconstruction and folding gates,
  including the declared numerical guards.
- Coherent current-window residual corrections using cross-statistics, with
  reconstruction of changed features when geometry changes.
- Initial fit before the 35% raw-index boundary; prequential calibration from
  35% to 50%; evaluation from 50% onward.
- Per-horizon correction weight selected from `[0, 0.25, 0.5, 1]` using only
  calibration predictions whose targets have matured by the stated cutoff.
- A 20 ms complete-observation service allowance on one numerical thread,
  treated as an engineering feasibility screen rather than an application
  deadline guarantee.

The kernel comparator has up to M support rows per horizon, whereas the local
geometric methods share M references and maintain additional decoder/replay
state. These are declared reference/replay tiers, not equal byte budgets.
Actual retained-state bytes must therefore accompany the results. Name the
kernel comparator by its implemented objective; do not call it KRLS-T unless
that published tracking algorithm is implemented.

## Clarifications required in the frozen protocol

### Boundary and maturity convention

Specify exact integer boundaries and inclusive/exclusive comparisons. A clean
convention is to fit only on labels with target index below the first
calibration origin, choose correction weights from issued calibration forecasts
with target index below the first evaluation origin, and then keep those
weights fixed throughout evaluation.

At arrival of y[t], score immutable forecasts previously issued for t, release
the newly matured supervised pairs, update according to the fixed rule, and
issue forecasts for future offsets. The sequence must be identical across
methods. Record origin and target index explicitly; model-state versions are
also useful. Never recompute a scored forecast after its target has been used
for an update.

It is legitimate to update on earlier evaluation-period observations. The
frozen object is the update algorithm, not the coefficient vector. Hyperparameter
changes prompted by evaluation-period loss inspection are a different matter
and must not be introduced into the reported run.

### Representation and residual coherence

Changing references or local principal directions changes the feature map.
Head sufficient statistics must be rebuilt or correctly transported. A
fixed-feature recursive least-squares identity does not justify reusing those
statistics across an arbitrary feature change.

Likewise, changing the baseline changes the residual target. Verify that the
cross-statistics implement the declared current-window correction objective
exactly, rather than silently mixing residuals from several baseline versions.
An independent batch comparison on a short stream is a meaningful check.

Reference insertion/deletion can permit efficient kernel factor updates only
while existing kernel entries remain unchanged. Changes to bandwidth, scaler,
covariance metric, principal directions, or regularization can require full
reconstruction. Charge that work. The prototype may use bounded local rebuilds;
it should not describe them as a rank-one update of every changed quantity.

### Model-state and selection parity

The local methods should receive the same reference-retention policy, refresh
schedule, replay availability, raw input information, and correction-weight
selection rule. Otherwise an improvement may belong to unequal adaptation or
selection opportunities rather than reconstruction geometry.

Calibration weights should be selected by strictly improving a first incumbent
of zero. Save every candidate loss and its eligible count. Different horizons
can have different calibration sets because their labels mature at different
times. Ensure that all candidates for the same horizon use exactly the same
issued predictions and targets.

### Scope of the 20 ms allowance

Complete-observation service time must include all model work needed at that
observation: matured-horizon updates, reference selection, any local rebuild,
feature recalculation, head updates, and newly issued forecasts. Separate
initialization and calibration costs from evaluation-period service costs.
Artifact writing and independent analysis can be excluded, but required model
bookkeeping cannot be moved outside the timer and disappear from the claim.

If overrun events are merely recorded, the experiment screens whether a method
meets this allowance on the measured host. It does not impose or guarantee a
hard deadline. Report mean, p95, p99, maximum, and overrun fraction, including
refresh events. Retained array/state bytes are not peak runtime resident memory.

Do not pool all service events into one percentile for the advancement gate:
the long Sunspot stream could hide a short stream's costly refreshes. Apply
the p99 allowance per evaluation stream at the proposed qualifying tier.

## Suggested unambiguous advancement rule

Evaluate the entire rule separately at each of the two tiers. A tier qualifies
only if all its accuracy, coverage, and resource conditions hold together.
Do not combine accuracy from the larger tier with latency from the smaller one.
Passing either complete tier can justify proceeding, since the experiment is
explicitly a development selection step. Report both tiers regardless.

For each comparator c, define the descriptive gain

`gain(c) = 100 * (loss(c) - loss(adaptive_KAHM)) / loss(c)`.

Use the frozen loss aggregation, not a mixture of ratios selected after seeing
results. The proposed numeric thresholds can be made precise as follows:

1. On recurring synthetic streams, gain is at least 3% against each of adaptive
   affine, adaptive RBF-local, and the kernel comparator; at least three of four
   paired seeds favor KAHM against each of these comparators.
2. On the same recurring aggregate, gain is at least 1% against both fixed
   KAHM and the baseline.
3. On the natural equal-collection aggregate, gain is at least 1% against each
   of the three adaptive/kernel challengers.
4. At least one same natural collection has at least 3% gain against all three
   challengers; no natural collection is worse by more than 5% against any of
   them. This avoids an ambiguous different favorable collection for each
   comparison.
5. Each evaluation stream at the qualifying tier has p99 service time at or
   below 20 ms; there are no unexpected exceptions or cap violations. Disclose
   numerical fallback events and KAHM activity even if fallbacks return a valid
   forecast. A numerical fallback must not be silently counted as an active
   KAHM success.

Possible labels:

- **PROCEED:** at least one tier passes the entire rule. This recommends a
  larger, genuinely reserved-source study; it does not establish the final
  publication claim.
- **REFINE:** the recurring synthetic mechanism conditions pass, but natural
  transfer or the resource conditions fail. State exactly which condition
  requires another development change.
- **DEFER:** the specified prototype does not show the recurring-regime
  mechanism signal. This is a decision about this proposed extension, not a
  universal statement about KAHM.

The thresholds are engineering/research decision choices, not empirically
calibrated significance cutoffs. Four seeds and two selected collections do not
support a strong probability claim about how often a future benchmark will
pass this gate. A benefit restricted to one previously favorable real source
should remain visible even when the aggregate is favorable.

## Minimum evidence for the independent audit

Preserve enough records to reconstruct, without importing model code:

- Every issued prediction: method, tier, stream, horizon, origin, target index,
  target value or later availability, baseline and unit-correction components
  where relevant, selected weight, and final forecast.
- Calibration candidate losses/counts and selected weights.
- Per-observation service timing and refresh/ordinary-step indicator.
- All numerical fallback, exception, missing-input, and unscorable-target
  records with their separate causes.
- Dictionary and replay size traces or maxima, plus retained-state byte
  accounting and its precise scope.
- Raw input and protocol/source hashes.

The audit should independently rebuild scored targets from raw arrays, check
origin/target relationships, recompute primary losses and aggregation, verify
weight selection using the allowed calibration set, and reproduce every
component of the development gate. It should not infer statistical
confirmation from a computational consistency check.

## Most valuable pre-run numerical checks

1. Perturb future values and require identical earlier issued forecasts.
2. Verify every supervised update has target index at or before current time.
3. Compare fixed-map recursive and independent batch decoder objectives.
4. Compare a refreshed geometry/head state with a fresh buffer reconstruction.
5. Check all reference, replay, and pending-label limits.
6. Verify zero correction returns the baseline exactly.
7. Reconstruct saved losses independently from immutable forecasts.

These checks address concrete risks in an adaptive forecasting implementation;
they are more informative than tests that simply restate its control flow.
