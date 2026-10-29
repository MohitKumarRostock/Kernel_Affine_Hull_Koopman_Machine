# KAHM-focused manuscript revision

## Revised scientific framing

The manuscript is now titled **Forecasting with Kernel Affine Hull Machines:
Representation Refinement and Risk Certification**. KAHM is central to the
title, abstract, introduction, method explanation, experimental questions,
and discussion. The presentation explains how its local reconstruction models
produce soft associations, how retained dictionaries support refinement, and
how direct forecasting and independent calibration assess their predictive
value.

The reconstruction machinery is explicitly inherited from the authors' prior
KAHM work. The new forecasting task, pairwise obstruction, retained refinement,
and implemented decision procedure are distinguished from those earlier
components. The general risk and refinement results continue to apply to
other feature maps. This makes the contribution relevant beyond users of
KAHM and states the actual extension clearly.

The revision follows the [JMLR author guidance](https://www.jmlr.org/author-info.html)
on a clear advance, supported claims, and reproducible experiments, and the
[reviewer guidance](https://www.jmlr.org/reviewer-guide.html) on significance,
generality, and distinction from prior work. The title page uses five keywords,
a running title under 50 characters, and an abstract under 200 words.

## New evidence that supports the framing

The previous experiments had KAHM Frobenius and rollout models without the
corresponding RBF and affine-hull models. The new frozen follow-up supplies
all those controls at every retained prefix, gives the geometries identical
preprocessing choices, and selects models using validation data alone.
All seven existing tasks and five fresh fitting seeds are included. There
are 124 completed jobs and 13,440 fitted candidates, with no failed jobs.
The [frozen protocol](MATCHED_GEOMETRY_PROTOCOL.md) specifies the design and
the [retained analysis](../results/matched_geometry_study/analysis/) contains
every result.

The primary taskwise comparisons are below. Each cell reports percentage
change in KAHM MSE relative to RBF / affine-hull MSE within the stated head.
Negative values favor KAHM. Means pool the predefined horizons and five
fitting repetitions within each task; PenDigits additionally pools the
coordinate losses and gives the eligible writers equal weight. Available
history, size, and scaling choices are matched, while the validation-selected
settings can differ. Fixed-size comparisons are reported separately.

| Task | Block direct: RBF / affine | Frobenius direct: RBF / affine | Rollout: RBF / affine |
|---|---:|---:|---:|
| Harmonic | -2.4% / -5.1% | -1.5% / +6.1% | -0.2% / +1.2% |
| Quasiperiodic | -2.0% / -1.4% | -1.7% / +30.1% | -6.1% / +3.3% |
| Nonlinear AR | -2.7% / -4.4% | -2.7% / -4.4% | -5.0% / -4.9% |
| Logistic | -1.0% / -4.5% | -0.4% / -5.8% | +1.7% / -5.7% |
| Van der Pol | -14.4% / +9.5% | -14.5% / +102.7% | -17.1% / +92.8% |
| Duffing | +3.8% / +13.1% | +10.4% / +70.6% | +1.4% / +79.9% |
| PenDigits | -15.6% / -12.7% | -17.2% / -1.9% | -17.9% / -1.4% |

Exact values, all fitting-seed effects, and per-horizon comparisons are in
[taskwise_comparisons.csv](../results/matched_geometry_study/analysis/taskwise_comparisons.csv).
KAHM has lower mean MSE in 28 comparisons and higher MSE in 14. This is a
descriptive count of related task/head/alternative comparisons, not 42
independent statistical replications.

The substantive findings are:

- On nonlinear AR, all six primary geometry comparisons favor KAHM in every
  fitting repetition. The observed reductions are modest and repeat across
  the five fresh synthetic episode streams.
- On PenDigits, KAHM improves over RBF in all five fitting repetitions for
  each head. Its block-direct comparison also improves over affine geometry
  by 12.7% on average. The smaller Frobenius and rollout differences against
  affine geometry vary more across fitting repetitions.
- The new affine Frobenius and rollout controls perform substantially better
  than their KAHM counterparts on both oscillator tasks. The earlier favorable
  KAHM-head results therefore do not establish a unique KAHM geometry advantage.
- Strong ordinary controls remain relevant. In the new PenDigits schedule,
  quadratic direct MSE is 0.0010182 and AR MSE is 0.0018437, versus 0.0050449
  for KAHM with the Frobenius head. The geometric-control gains do not imply
  that KAHM is the most accurate forecaster overall.

These conclusions strengthen the empirical account of when the KAHM
construction helps and when the choice of decoder or alternative geometry
changes the result. They do not establish universal superiority, equal
statistical capacities, or a runtime advantage. Additional PenDigits fitting
seeds reuse the same 13 eligible held-out writers.

## Accuracy and certification are separate findings

One of the 180 prespecified primary calibration comparisons has a positive
improvement lower endpoint: Van der Pol, seed 5703, rollout, KAHM versus RBF.
Its lower endpoint is 0.00107100848 MSE. The other 179 comparisons remain
unresolved. The combined confidence allocation is joint within a complete
synthetic task/seed, not across all tasks and seeds. PenDigits comparative
effects remain descriptive on the fixed writer cohort.

The complete [calibration comparison table](../results/matched_geometry_study/analysis/calibration_comparisons.csv)
retains positive and unresolved outcomes. Its shifted bounded-mean fields
are accompanied by the unshifted `improvement_lower` and
`empirical_improvement` fields; those latter fields express the loss difference.

## Verification and preservation

All 45 tests passed, including exact default candidate descriptions and
predictions against the previously published commit. Twenty separate
development smoke jobs covered all routes and modes; their outcomes are not
accuracy evidence. The full analysis checked 30,240 synthetic arrays containing
247,726,080 float64 values and 131,040 PenDigits writer-loss values.

A separate standard-library implementation reconstructed all 360 primary
model choices from the raw JSON records and independently verified all 42
taskwise effects. Selected identities matched exactly, and the largest
numerical difference was below 4.3e-14, within the declared 1e-12 tolerance.
The independent check is reproducible with `scripts/verify_matched_summary.py`.

All historical numerical records, frozen replay sources, and original
manuscript tables and figures are preserved. New results and publication
assets are retained separately. Full synthetic score archives are indexed by
hash and reproducible from the frozen runner; the compact evidence package
contains the 270 primary selected synthetic models' 810 score arrays and
the complete scalar/writer records.
