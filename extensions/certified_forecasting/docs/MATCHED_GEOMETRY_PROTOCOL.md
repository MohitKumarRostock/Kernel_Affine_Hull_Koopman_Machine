# Matched KAHM geometry study: design and statistical interpretation

The study design was fixed on 29 October 2026 before the follow-up fits. It
was informed by earlier experiments and was not externally preregistered.
The immutable [original protocol](../results/matched_geometry_study/protocol.md)
and run manifests preserve the design, creation time, source and configuration
hashes, software versions, data hash, and job schedule. The reference repository
head was `71efa873419e2b58ee89746dafd123967ff007b1`. This document explains that
design and the scope of its statistical conclusions, including the limitation
created by reuse of the PenDigits evaluation cohort.

## Scientific question

The study compares KAHM, radial basis function (RBF), and local affine-hull
features under the same prediction architectures and model-selection
opportunities. Crossing all three geometries with all three decoders distinguishes the
contribution of feature construction from the contribution of the decoder.
The task panel contains six synthetic systems and the PenDigits acquisition
domain.

## Design

| Factor | Values |
|---|---|
| Tasks | Harmonic, quasiperiodic, nonlinear AR, logistic, Van der Pol, Duffing, and original PenDigits trajectories |
| Fitting seeds | 5701, 5702, 5703, 5704, 5705 |
| Geometry | KAHM, RBF, local affine hull |
| Prediction architecture | Direct sum-of-block-Frobenius constraint; direct Frobenius constraint; affine feature-transition rollout |
| Retained prefixes | 2, 6, 14, 30 features from dictionaries of size 2, 4, 8, 16 |
| Preprocessing | `standard`, `minmax`, `minmax_clip`, `raw`, available to every geometry |
| Synthetic histories / horizons | Histories 4 and 12; horizons 1, 4, 12 |
| PenDigits histories / horizons | Histories 4 and 8; horizons 1, 2, 4; both coordinates |

Sample sizes, origin stride, generator parameters, regularization, geometry
subsampling, cluster-point limits, target ranges, and coefficient budgets
follow the original task configurations. Input-scaling statistics use all
training histories before geometry subsampling. The targets retain their
original units.

Each new synthetic seed generates separate training, validation, calibration,
and test streams using the inherited offsets. PenDigits uses new training-stroke
subsamples and geometry-fitting randomness within the same development and
held-out writer cohorts. Its repetitions measure fitting variation on a fixed
cohort; they do not supply additional independent writers.

For one scalar task and one preprocessing arm, each geometry has 24 candidates
(three architectures, two history lengths, four prefixes). Across four arms,
there are 96 candidates per geometry. Primary selection fixes the architecture
and compares 32 candidates per geometry (four preprocessing arms, two histories,
four prefixes). A secondary analysis can select from all 96.

The three matched geometry families contribute 72 candidates per scalar fit.
Twelve additional controls comprise the KAHM hard-assignment and duplicate-block
ablations and the existing autoregressive, polynomial, polynomial-rollout,
persistence, and constant models. Thus each scalar fit has 84 candidates.
The 120 synthetic jobs and four PenDigits jobs, each containing ten
coordinate--seed fits, produce 13,440 candidate fits in total.

Equal numerical coefficient budgets do not imply equal statistical capacity
across feature maps. The direct architectures also use different constraint
classes. Rollout sensitivity is computed from the fitted transition and output
decoder. Feature count measures representation size, rather than equal runtime,
storage, or kernel-evaluation cost.

## Selection and outcomes

Within each task, seed, geometry, and architecture, selection minimizes mean
validation MSE across horizons. Ties are resolved by feature count, history
length, preprocessing order (`standard`, `minmax`, `minmax_clip`, `raw`), and
candidate name. Calibration and test losses are absent from this ordering.
There are six primary comparisons per task and seed: KAHM against each of two
alternative geometries under each of three architectures.

The main descriptive outcome is test MSE averaged over horizons. Percentage
changes are `100 * (mean KAHM MSE / mean alternative MSE - 1)`, with means taken
over fitting seeds within a task. They are ratios of means, not means of
seed-specific ratios. Per-horizon results and every paired seed effect are
retained. Physical MSE values from different tasks are not pooled.

Fixed-dimension comparisons select only history length and preprocessing at
each dimension. They and the secondary `head_selected` comparisons are
descriptive. Both are identified in the `analysis` column of
[taskwise_comparisons.csv](../results/matched_geometry_study/analysis/taskwise_comparisons.csv).

PenDigits selection is coordinate-specific and weights validation writers
equally. After selection, losses are averaged over coordinates and horizons
within each held-out writer; the 13 eligible writers then receive equal weight.
The same writers appear in all fitting repetitions.

## Confidence accounting

For each synthetic task and seed, each preprocessing arm has failure budget
`delta=0.01`: `0.008` for its candidate risk bounds and `0.002` for its refinement
comparisons. A union bound over four arms uses `0.04`. The remaining `0.01` is
split among the six primary geometry comparisons, giving `0.01/6` per comparison.

For each comparison, alternative loss minus KAHM loss is averaged over horizons
within an independent calibration episode. Targets and predictions are scalar
and lie in `[-1,1]`, so this difference lies in `[-4,4]`. Its empirical Bernstein
radius therefore uses interval length `8`. A positive lower endpoint certifies
a reduction in expected all-horizon MSE for the selected fitted KAHM model.
Selection uses independent development data, so no additional multiplicity
factor is required for candidates rejected by validation.

The resulting 95% event is simultaneous within one complete synthetic task--seed
combination. It is not simultaneous across tasks, seeds, or earlier studies.
The same calibration episodes may enter several confidence calculations;
the union bound does not require independence between those calculations.
An inconclusive lower endpoint does not establish equality of risks.

PenDigits retains the numerical writer-level Hoeffding calculations with
`delta=0.0125` per preprocessing arm, allocated across candidates, coordinates,
five seeds, and horizons. The four allocations sum to `0.05`. A population
confidence interpretation would require both independently and identically
distributed evaluation writers and model-family specification independent of
their evaluation losses. Writer sampling is unverified, and the follow-up
design was informed by earlier results on the same cohort. These endpoints are
therefore nominal calculations rather than confirmatory population guarantees.
Paired writer comparisons describe the evaluated cohort. The synthetic
comparisons have the separate sampling design and confidence allocation stated
above.

## Evidence and reproducibility

The run manifest fixes the schedule, configuration bytes, source hashes, and
data hash before fitting. Reduced development runs use separate seeds and are
implementation checks rather than forecasting evidence. The complete schedule
contains 124 completed jobs and no failed jobs; all inconclusive certificates
are retained.

Completed score archives are written atomically and checked for complete ZIP
members, finite arrays, expected dimensions, and agreement with scalar summaries.
Historical configurations, source snapshots, and numerical records remain
unchanged. The prepared PenDigits source has SHA-256
`320aceb0f7f473f8eb21efe6f20bbf87a62f729e0523ba7a60885b073d1e45c2`.

The [study directory](../results/matched_geometry_study/) contains scalar results,
writer losses, the full-score archive index, and the analysis outputs. The
runners regenerate full synthetic score archives. A separate compact evidence
package retains the selected-model score arrays. The
[reproduction guide](REPRODUCIBILITY.md) distinguishes these materials and gives
the commands for reconstruction.
