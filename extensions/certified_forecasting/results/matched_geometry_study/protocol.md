# Matched KAHM geometry study: prospective follow-up protocol

Design fixed on 29 October 2026, before the full follow-up fits. The machine
readable run manifest records the exact creation time, source and configuration
hashes, software versions, data hash, and complete job schedule. This is an
internally frozen follow-up motivated by the earlier results, not an external
preregistration. The reference repository head before these changes is
`71efa873419e2b58ee89746dafd123967ff007b1`.

## Scientific question

Does replacing RBF or local affine-hull soft features with KAHM soft features
improve forecasting when the readout, input information, retained dimension,
training data, and validation selection opportunities are matched? In the
earlier experiments, KAHM Frobenius and rollout models had no corresponding RBF
or affine-hull controls. This study supplies those controls across the complete
existing task panel. It does not introduce a new real acquisition domain.

## Frozen design

| Factor | Values |
|---|---|
| Tasks | Harmonic, quasiperiodic, nonlinear AR, logistic, Van der Pol, Duffing, and original PenDigits trajectories |
| Fitting seeds | 5701, 5702, 5703, 5704, 5705 |
| Geometry | KAHM, RBF, local affine hull |
| Head | Direct sum-of-block-Frobenius constraint; direct Frobenius constraint; affine feature-transition rollout |
| Retained prefixes | 2, 6, 14, 30 features from dictionaries of size 2, 4, 8, 16 |
| Preprocessing | `standard`, `minmax`, `minmax_clip`, `raw`, available equally to every geometry |
| Synthetic histories / horizons | Histories 4 and 12; horizons 1, 4, 12 |
| PenDigits histories / horizons | Histories 4 and 8; horizons 1, 2, 4; both coordinates |

Sample sizes, origin stride, generator parameters, regularization, geometry
subsampling, cluster point caps, known target ranges, and coefficient budgets
are inherited from the original task configurations. Scaling is fitted on
training histories before geometry subsampling; targets keep their original
units. The seeds regenerate all four synthetic episode streams, using the
inherited stream offsets. PenDigits uses new training-stroke subsamples and
geometry randomness with the same training, validation, and held-out writer
cohort. The latter is fitting variation on a fixed cohort, not independent
replication of a new writer population.

Within a scalar fit, each geometry has the same 24 head/history/prefix
candidates. The four preprocessing arms give each geometry 96 candidates.
The KAHM hard-assignment and duplicated-block ablations and the existing
AR, polynomial, polynomial-rollout, persistence, and constant controls are
retained separately. They do not enlarge the three matched geometry families.
There are 84 candidates per scalar fit, 120 synthetic jobs, and four PenDigits
jobs containing ten coordinate/seed fits each: 13,440 fitted candidates in all.

The same numerical norm budget is a controlled protocol choice, not a claim
that different feature maps induce identical statistical capacity. Likewise,
Frobenius and block-norm direct heads belong to different constraint classes;
rollout sensitivity is computed from its fitted transition and readout.
Feature count is not a claim of equal training time, storage, or kernel cost.

## Selection and outcomes

For each task, seed, geometry, and head, choose preprocessing, history, and
prefix using the mean validation MSE across the stated horizons. The
deterministic ordering is validation MSE, feature count, history, preprocessing
order (`standard`, `minmax`, `minmax_clip`, `raw`), and candidate name. This
produces six primary comparisons per task/seed: KAHM versus each alternative
within each of the three heads. All-horizon test MSE is the main descriptive
outcome, accompanied by per-horizon results and all five paired seed effects.
Percent changes use ratios of mean MSEs, with the exact direction labeled.
Do not pool physical MSE values across tasks or select a head using test error.

At each fixed dimension, additionally select history and preprocessing by
validation within each geometry/head. Report the complete dimension curves,
including regressions. These mechanistic comparisons are descriptive. A
secondary comparison may select the head by validation as well; it is labeled
separately and has no additional comparative confidence claim.

PenDigits selection is coordinate-specific. Average the selected coordinate
losses and horizons within each held-out writer, and then give each writer
equal weight. Retain all writer effects and the repeated fitting-seed effects.
The same writers across seeds must not be counted as new independent units.

## Confidence accounting

For each synthetic task/seed, each of the four preprocessing arms uses
`delta=0.01` for its complete frozen risk/refinement family, with the inherited
80/20 risk/refinement allocation. The union of these four families costs 0.04.
The remaining 0.01 is split equally over the six primary validation-selected
geometry comparisons. For each comparison, average the squared losses over
horizons within an independent calibration episode, then apply the existing
paired empirical-Bernstein procedure to alternative loss minus KAHM loss.
Each scalar prediction and target lies in [-1,1], so the individual loss lies
in [0,4] and its paired difference lies in [-4,4]. A positive lower endpoint
certifies improvement of these fitted predictors on the stated episode law.
Selection used only independent development data, so it does not require an
additional calibration multiplicity factor over rejected development models.

The resulting 0.05 event is joint within a complete synthetic task/seed. It is
not a simultaneous event over all tasks, fitting seeds, or historical studies.
The fixed-dimension and head-selected secondary comparisons have descriptive
test effects only. An interval failing to certify improvement is reported as
unresolved, not as proof of equal risk.

PenDigits retains its original writer-level Hoeffding upper bounds with
`delta=0.0125` per preprocessing arm, allocated across all candidates, both
coordinates, five seeds, and horizons. Their union over the four arms costs
0.05, conditional on the original unverified independent-writer sampling
assumption. PenDigits has no additional independent calibration cohort for
the new geometry comparisons; paired writer differences remain descriptive.

## Integrity and reporting

The runner freezes the complete schedule, configuration bytes, source hashes,
and pinned data hash before full fitting. Development smoke tests use separate
seeds and reduced configurations and are never manuscript accuracy evidence.
All scheduled results, errors, and unsuccessful certificates are retained.
Completed score archives are written atomically and checked for complete ZIP
members, finite numeric arrays, expected dimensions, and agreement with their
scalar summaries. Historical configurations, source snapshots, and results are
preserved unchanged.

The prepared PenDigits source used by this follow-up has SHA-256
`320aceb0f7f473f8eb21efe6f20bbf87a62f729e0523ba7a60885b073d1e45c2`.
The committed results retain every scalar candidate result, all writer losses,
the complete score-archive index, and the analysis outputs. The supplied
runners regenerate full synthetic score arrays; a compact evidence package
retains the selected-model arrays used in the matched comparisons.
