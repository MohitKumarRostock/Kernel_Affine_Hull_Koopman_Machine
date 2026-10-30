# Forecasting utility of KAHM: extension protocol v1

## Purpose and evidential status

This study asks when a fixed KAHM feature construction makes predictive structure
useful to a finite forecasting procedure, and whether its observed improvement
justifies its measured additional cost. It extends the unsubmitted prediction
manuscript. It does not modify or replace the original controlled, eight-source
natural, or scaling-sensitivity studies.

The design below and its executable configuration are fixed before the first
full extension fit or evaluation. `freeze.json` records execution-source,
configuration, protocol, dataset-metadata and selected-roster hashes. A public
execution checkpoint is published before full execution. This is a prospective
execution freeze, not an externally registered study. Engineering tests use toy
data and do not select the design from the declared study's outcomes. Any later
execution correction must be recorded, with its reason, affected runs and hashes;
no adverse or failed result is silently replaced.

## Questions and primary comparisons

1. Does an additive KAHM correction improve a fixed raw-history predictor?
2. Does preserving the raw prediction pathway change the outcome relative to
   the earlier concatenated raw-plus-KAHM procedure?
3. Can simpler affine or locality features explain the effect?
4. How do training sample size and finite raw-decoder capacity change the result?
5. Does a fixed conservative validation rule obtain a useful error-cost tradeoff
   on datasets outside the original eight-source study?

The primary synthetic contrasts are raw RBF minus additive KAHM and concatenated
KAHM minus additive KAHM within each prespecified system/data/capacity cell.
Positive loss differences favor the additive construction. Affine/locality,
ridge and naive controls remain in every reported table. Archive comparisons
are descriptive, with equal weight to each of the four source collections.

## Models and fitting

The methods comprise raw ridge, raw finite-RBF plus linear features, the original
concatenated raw-plus-KAHM finite-RBF architecture, and additive KAHM, affine and
locality corrections. A persistence/seasonal-naive control is included. The
executable configuration also includes a fixed histogram-gradient-boosting
control with 100 iterations, at most 15 leaves, minimum leaf size 5 and L2
regularization 0.01. This is a practical comparator, not exhaustive tree tuning.

For the additive procedure, first select the raw RBF ridge penalty using only
training and validation observations. Preserve that fitted raw predictor and
fit a ridge correction to its in-sample training residuals. This residual fit is
not cross-fitting. Each gate type uses the inherited local-coordinate features
and the same declared reference-selection/PCA configuration. Affine and locality
controls do not fit a KAHM kernel. The fitted prediction is

`raw_prediction + gamma * local_correction`.

The exact raw prediction remains available at `gamma = 0`; its query path skips
the local correction. This fact establishes baseline availability, not improved
generalization. The raw branch is not refitted while comparing correction
penalties or mixing coefficients. Train-only standardization is inherited from
the existing forecasting implementation. KAHM uses its stable, row-rescaled
Gaussian evaluation, including the documented signed-normalizer conventions.

Raw and correction ridge penalties are selected from `0.01, 1, 100`. Mixing
coefficients are `0, 0.25, 0.5, 1`. The raw penalty is chosen first; correction
penalty and mixing are then chosen by validation MSE. Ties prefer the simpler
prediction (`gamma=0`, then the smaller nonzero mixing coefficient, followed by
the fixed configuration order). The same validation opportunities apply to all
three correction geometries. The zero correction is counted once. Selection
opportunities differ from the three-penalty raw and concatenated controls and
are reported; equal feature counts do not imply equal capacity or computation.

Geometry uses blocks with 2, 4 and 8 groups, requested rank two, at most 24 local
references, a 512-row bank fitting cap, exponent 16 and offset `1e-6`. A common
seed and inputs give matched partitions and reference choices across geometries.
Each alternative deployed geometry is charged for its own fitting and query
cost. The raw fit can be reused within an additive procedure or a selector; a
shared computation is charged once, not zero times or repeatedly.

## Synthetic factorial study

The three inherited numerical laws are nonlinear autoregression, Van der Pol
and Duffing. Each supplies length-72 independent episodes, history length 12,
horizons 1, 4 and 12, and origin stride 4 (13 origins per episode). Origins within
an episode are not independent observations.

Training counts 32 and 128 episodes are crossed with 16 and 64 raw RBF landmarks.
Each cell has 12 complete fresh repetitions identified by seeds 9100--9111.
The executable seed derivation assigns disjoint training, validation and test
streams. Within a repetition, the smaller training collection is the first 32
episodes of the larger 128 collection; validation and test collections are
shared across the paired data-size/capacity comparisons. There are 64 validation
and 128 test episodes per repetition. This gives 144 task configurations.

All training origins are available to the raw and correction fits; only the
geometry bank has the explicit 512-row cap. Selected predictors are retained
from training-only fitting, without a train-plus-validation refit, so the stated
training-episode factor retains its meaning. Hyperparameters use validation
targets. Test outcomes cannot determine a model, a candidate grid or an analysis
choice.

The task loss averages squared error over target horizons and origins within
an episode and then over episodes. Per-horizon and per-episode losses and full
predictions are retained. Paired complete-repetition differences are the units
for synthetic summary intervals, with a declared family correction for the 24
primary system/data/capacity contrasts. Student intervals, if reported, depend
on a normal approximation for 12 complete-repetition differences and are not
distribution-free certificates. All cell means and unfavorable comparisons are
reported. Larger counts of windows or horizons do not enlarge the replication
count.

## Archive datasets and chronological evaluation

The new public Monash archive records are Weather (Zenodo 4654822), Solar Weekly
(4656151), COVID Deaths (4656009), and the original Car Parts data with missing
values (4656022). These records are absent from the original study's electricity,
NN5, tourism, FRED-MD, traffic, hospital, Saugeen and US-births roster. They were
chosen for provenance, varied forecasting structure and computational
feasibility, not observed extension model performance. Solar Weekly contains
aggregated simulated NREL solar series; the archive stage is therefore not a
study of four purely observational sources. The code phase name `natural` is
retained as a workflow label and does not override this provenance distinction. They are four selected
collections, not a random sample of future application domains. Their upstream
provenance, record licenses, byte checksums and download URLs are retained.

Each record contributes up to three series, ordered by SHA-256 using the fixed
salt `forecasting-utility-natural-v1` and source/series identifiers. Eligibility
uses series length and the training prefix only: at least 16 available complete
training windows and a positive finite training-only error scale. Selected
series are never replaced because of later validation/test availability or
forecasting error. Missing values are not interpolated. Unsupported selected
tasks remain explicit in the accounting.

Each series is split chronologically at 60% and 80% of its length. Complete
targets must lie within the assigned split. A context may use earlier observed
values. Daily Weather and COVID use history 28 and horizon 7; weekly Solar and
monthly Car Parts use history 6 and horizon 2 because these archives contain
shorter records. All horizon coordinates are predicted directly. Seasonal-naive
periods are 7 for daily data, 52 for weekly Solar and 12 for monthly Car Parts;
unavailable seasonal histories fall back to the last available observation.

Natural training fits use at most 512 evenly spaced available windows; validation
and test use at most 64 evenly spaced available origins. RBF landmark counts
16 and 64 are crossed with the fixed series roster and one declared fitting
seed, giving up to 24 task configurations. After validation selection, each
selected procedure is refitted on admissible pretest windows, capped at 512,
before the reserved test interval is scored. Recorded origins and targets are
common to all compared methods. Earlier test observations may enter a later
causal context; test targets never retune parameters.

MSE is reported in original units and divided by a fixed error scale computed
from the initial uncapped training prefix: mean squared seasonal difference,
with first-difference fallback when the seasonal scale is unavailable or zero.
No validation or test observation sets the scale. Source-balanced aggregation
weights the four collections equally, then their selected series and capacities
equally. Source-wise results remain visible. Related origins, capacities and
series do not support independent-window tests or broad population p-values.
Leave-one-source-out averages describe the sensitivity of the archive summary
to its four selected collections; they are not extra confirmation samples.

## Decisions, costs and interpretation

The conservative KAHM policy uses the selected additive forecast only if its
validation MSE is at least 5% below that of the selected raw RBF predictor;
otherwise it returns the raw predictor. A corresponding all-geometry policy may
choose the validation-best correction under the same threshold. The 5% margin
is a declared research sensitivity, not an established application requirement.
All declared rules are reported; confirmation results do not select the rule.

Resource measurements include raw search, feature preparation, correction search,
validation prediction, selected refit, and final prediction. Shared raw operations
are charged once in each additive or portfolio procedure. A policy pays for
examining a correction even when it rejects that correction. Wall time and CPU
time are recorded separately where available; serialization/model size and
batch prediction cost are distinct quantities. Imports, data download and
analysis/plotting are reported separately from model timing. Execution uses one
active timed task and one numerical-library thread. These are implementation-
and hardware-specific measurements, not intrinsic complexity guarantees.

The research question is evaluated through error differences, practical margins,
validation choices and complete measured cost profiles. No claim of universally
optimal complexity, population Bayes-risk estimation, probability calibration,
distribution-shift robustness, or class-level necessity follows from this study.
Appending deterministic KAHM features to a raw history adds no Bayes information;
any gain concerns the specified finite fitting procedure. Formal class-risk
certificates remain the subject of the separate manuscript.

A descriptive residual decomposition is computed from saved raw and additive
test predictions without any additional fitting or selection. Writing
`d = additive_prediction - raw_prediction`, the observed MSE improvement equals
`2 * mean((target - raw_prediction) * d) - mean(d**2)`. We report the residual
alignment and correction-energy terms, using the same training-only scale for
archive tasks, and verify this arithmetic identity. This describes the realized
forecast correction; it is not an estimator of representation information or
an additional confirmatory hypothesis test.

## Reproducibility and failure accounting

Every task records source and configuration hashes, data split/origin hashes,
all validation candidates, selected settings, warnings, failures, prediction
arrays, and timing scope. Failed attempts remain in the report. Output paths
must be new or resume an explicitly matched frozen run without overwriting
completed tasks. The independent analysis recomputes losses from predictions,
checks complete rosters and hashes, and generates manuscript tables and figures.
The original archive remains unchanged. New evidence is labeled as this
extension rather than a re-estimation of the original study.
