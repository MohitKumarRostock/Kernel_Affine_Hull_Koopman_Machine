# Matched-geometry results and scientific interpretation

The manuscript **Forecasting with Kernel Affine Hull Machines: Representation
Refinement and Risk Certification** combines inherited KAHM reconstruction
features with constrained observed-output forecasting and episode-level risk
certification. Its pairwise obstruction and refinement results also apply to
other fixed representations. The [literature context](LITERATURE_AUDIT.md)
identifies the established ingredients and the scope of this extension.

## Matched comparison

The follow-up supplies RBF and affine-hull controls for all three prediction
architectures: block-direct prediction, Frobenius-direct prediction, and feature
rollout. All geometries receive the same history lengths, retained dimensions,
and preprocessing choices. Validation alone selects the candidate within each
geometry--architecture family. The study covers seven tasks and five new fitting
seeds, with 124 completed jobs, 13,440 candidate fits, and no failed jobs.
The [design and statistical interpretation](MATCHED_GEOMETRY_PROTOCOL.md)
distinguish new synthetic episodes from reuse of the PenDigits writer cohort.

Each primary family selects from 32 candidates: four preprocessing choices,
two history lengths, and four retained dimensions. Selected settings may differ
between geometries. Fixed-dimension comparisons are reported separately.

The table reports `100 * (mean KAHM MSE / mean alternative MSE - 1)` within each
prediction architecture. Negative values favor KAHM. MSE is averaged over the
specified horizons and five fitting repetitions within a task. PenDigits also
averages the coordinate losses and weights eligible writers equally.

| Task | Block direct: RBF / affine | Frobenius direct: RBF / affine | Rollout: RBF / affine |
|---|---:|---:|---:|
| Harmonic | -2.4% / -5.1% | -1.5% / +6.1% | -0.2% / +1.2% |
| Quasiperiodic | -2.0% / -1.4% | -1.7% / +30.1% | -6.1% / +3.3% |
| Nonlinear AR | -2.7% / -4.4% | -2.7% / -4.4% | -5.0% / -4.9% |
| Logistic | -1.0% / -4.5% | -0.4% / -5.8% | +1.7% / -5.7% |
| Van der Pol | -14.4% / +9.5% | -14.5% / +102.7% | -17.1% / +92.8% |
| Duffing | +3.8% / +13.1% | +10.4% / +70.6% | +1.4% / +79.9% |
| PenDigits | -15.6% / -12.7% | -17.2% / -1.9% | -17.9% / -1.4% |


Exact values, fitting-seed effects, and per-horizon comparisons are in
[taskwise_comparisons.csv](../results/matched_geometry_study/analysis/taskwise_comparisons.csv).
KAHM has lower mean MSE in 28 primary contrasts and higher mean MSE in 14.
These 42 contrasts share tasks and models; they are not independent statistical
replications.

On nonlinear AR, all six primary comparisons favor KAHM in every repetition.
On PenDigits, KAHM improves on RBF under every prediction architecture in all
five repetitions. The block-direct reduction relative to affine-hull features
is 12.7% on average; the smaller Frobenius-direct and rollout differences vary
more across repetitions. The same 13 eligible held-out writers are used in
all PenDigits repetitions.

Affine-hull features with Frobenius-direct and rollout predictors substantially
outperform their KAHM counterparts on both nonlinear oscillator tasks. Thus a
gain from changing a KAHM decoder does not establish an advantage of KAHM
geometry. Ordinary controls also remain competitive: on PenDigits, quadratic
direct MSE is 0.0010182 and autoregressive MSE is 0.0018437, compared with
0.0050449 for Frobenius-direct KAHM. The study supports task-dependent geometric
effects, rather than universal superiority or a runtime advantage.

## Calibration and empirical accuracy

One of 180 primary synthetic calibration comparisons has a positive improvement
lower endpoint: Van der Pol, seed 5703, KAHM rollout versus RBF rollout. Its
lower endpoint is 0.00107100848 MSE. The other 179 comparisons are inconclusive.
The combined 95% event is simultaneous within a synthetic task--seed combination,
not across tasks or seeds. New synthetic calibration episodes support this
interpretation even though the study design was informed by earlier experiments.

PenDigits comparisons remain descriptive on the previously examined writer
cohort. Its retained writer-level endpoints are nominal: a new population
confidence interpretation would require both iid writer sampling and a study
design independent of evaluation-writer outcomes. Neither requirement is
established for this follow-up. New fitting seeds do not create new evaluation
writers or remove dependence on previously examined cohort results.

The [calibration comparison table](../results/matched_geometry_study/analysis/calibration_comparisons.csv)
retains positive and inconclusive outcomes. The generic `mean`, `lower`, and
`upper` fields refer to the shifted nonnegative variable used by the
bounded-mean implementation. The unshifted `improvement_lower` and
`empirical_improvement` fields express alternative risk minus KAHM risk.

## Verification and reproducibility

The retained [implementation record](../results/matched_geometry_study/verification/implementation_verification.json)
documents 45 passing tests and 20 reduced development jobs at study completion.
Those reduced jobs are implementation checks, not forecasting evidence. The
[full analysis record](../results/matched_geometry_study/analysis/audit.json)
reports checks of 30,240 synthetic arrays containing 247,726,080 float64 values
and 131,040 PenDigits writer-loss values.

The independent standard-library verifier reconstructs all 360 primary model
choices from the raw scalar records, checks writer-level pooling, and reconstructs
all 42 taskwise primary effects. The check was repeated during the manuscript
revision: selected identities matched exactly, and the maximum numerical
discrepancy was 4.263256414560601e-14, below the declared 1e-12 tolerance. The
[retained verification](../results/matched_geometry_study/verification/independent_primary_verification.json)
and `scripts/verify_matched_summary.py` document the calculation.

Historical numerical records, source snapshots, and original figure data are
preserved. Full synthetic score archives are indexed by hash and reproducible
from the frozen runner. The separate compact evidence package contains 270
selected synthetic models' 810 score arrays and complete scalar/writer records.
The [study README](../results/matched_geometry_study/README.md) identifies each
available component and the commands for independent reconstruction.
