# Scientific scope and implementation consistency

The initial review on 28 October 2026 covered the original forecasting
campaign, exact-risk benchmark, PenDigits evaluation, and budget sensitivity.
The 29 October revision also examined input scaling and matched geometry,
using the retained configurations, scalar records, selection code, certificate
code, and primary references. The checks below distinguish mathematical
assumptions, reconstructed empirical quantities, and limits of interpretation.

## Method and theory

The prediction target is the observed future signal under the task's fixed
transformation. Direct multihorizon prediction and iterated feature-transition
baselines are evaluated on this same target. Neither a Markov assumption nor
exact Koopman closure is required for the episode-level certificate.

The retained representation is a product of probability simplices. Refinement
keeps the actual earlier coordinate functions, rather than requiring nested
cluster partitions. The sum of block Frobenius norms is paired with the maximum
Euclidean distance across blocks. This establishes nested feasible forecast
classes and invariance to exact duplicate blocks. It does not establish
monotone risk for separately fitted finite-sample predictors. The quadratic
ridge penalty can change under duplication; the unregularized numerical
comparison therefore uses zero ridge.

The pairwise lower bound and fitted-model upper bound concern the same target,
origin law, metric, and sensitivity budget. A positive lower endpoint excludes
only the stated restricted class. A zero endpoint is inconclusive. The
necessary metric-expansion inequality is not a sufficient learning rule, and
future noise may make an obstruction irreducible.

Optimization gaps concern the finite regularized training objective. They are
floating-point numerical checks, not interval-arithmetic proofs and not
population-risk certificates. Statistical validity requires the returned
predictor to satisfy its stated sensitivity budget, rather than to attain a
global training optimum.

## Independent units and calibration

Common forecast origins are used across history lengths. Overlapping training
windows may be dependent because the finite optimization objective does not
assume independent rows. Confidence calculations use one average loss per
independent episode and one obstruction score per disjoint episode pair.

A calibration family must be specified independently of its evaluation data.
A finite-family multiplicity allocation does not establish this independence.
The campaigns therefore have different inferential scopes:

| Campaign | Evaluation data | Interpretation |
|---|---|---|
| Original synthetic forecasting | Separate development, calibration, and test episode streams | Within-run risk and refinement guarantees under the specified episode law |
| Original budget sensitivity | New streams, shared across three prespecified budgets within a repetition | Allocation `0.05/3` per budget family; no simultaneous claim across all task--seed repetitions |
| Original-seed scaling replay | Previously used calibration and test streams | Matched descriptive comparison; new-arm endpoints are nominal without a separate independence or adaptive-validity argument |
| Synthetic scaling confirmation, seeds 4701--4703 | New development, calibration, and test episodes | Stated within-arm guarantees; no joint guarantee for searching across four arms |
| Matched synthetic geometry, seeds 5701--5705 | New development, calibration, and test episodes | Joint 95% event within one task--seed combination, including four arm families and six primary comparisons |
| PenDigits follow-ups | New fitting randomness, same eligible held-out writer cohort | Descriptive cohort comparisons and nominal writer-level endpoints; additional fitting seeds are not new population samples |

The conditional PenDigits formula additionally assumes iid writer sampling,
which the source dataset does not establish. For follow-ups informed by earlier
results on the same writers, independence of design from evaluation outcomes
is also unestablished. These qualifications are separate from whether the
arithmetic endpoints are reproduced exactly.

## Empirical counts and reconstruction

The original main campaign contains 40 candidates, three horizons, and 20
complete learning runs: 800 fits, 4800 one-sided risk endpoints, 480 KAHM
candidate--horizon lower statements, and 360 adjacent-prefix comparisons.
Within each run, the failure budget is 0.04 for 240 risk tails and 0.01 for
18 paired-improvement tails.

The matched follow-up contains 13,440 candidates in 124 completed jobs. Primary
selection uses 32 alternatives per geometry--architecture family, with the
same choices available to all three geometries. Percentage effects use ratios
of mean MSEs across fitting seeds. The independent standard-library verifier
reconstructs all 360 selected coordinate-level models and all 42 primary
contrasts from raw scalar records; the maximum discrepancy is below 4.3e-14.
The [verification record](../results/matched_geometry_study/verification/independent_primary_verification.json)
and [study README](../results/matched_geometry_study/README.md) give the evidence
and reproduction command.

An independent reconstruction of the scaling records checked 8,144 original
and 3,840 confirmation candidate records. It reproduced 3,264 original and
1,152 confirmation validation-selected models exactly, including 9,792 and
3,456 horizon rows. All 285 original and 105 confirmation taskwise mean-effect
rows agreed within 1.6e-13. This check verifies selection and mean aggregation;
it does not rerun fitting or reconstruct every full-score confidence interval.

The exact-risk benchmark compares against analytically known oracle and fitted
risks and retains the observed coverage violation. Its three confidence
directions share 0.05 within a trial. There is no asserted global event across
all Monte Carlo trials or benchmark conditions.

## PenDigits population and outcome definitions

PenDigits uses original acquisition coordinates and documents inferred writer
identifiers. The eligible cohort contains 13 held-out writers. Losses are
averaged within writers, then equally across writers; coordinate MSE is pooled
before taking its square root. Fitting-seed standard deviations measure
variation on that same cohort, not uncertainty across independently sampled
writer populations.

The length restriction defines a retrospective eligible-stroke population.
Eligibility of a newly started stroke is generally unknown before it reaches
the required length. The results therefore do not establish an unrestricted
online prediction guarantee. The writer identifier remains a documented
inference rather than a source field with independently established meaning.

## Contribution and evidence scope

The predecessor already supplies fitted-risk upper bounds and independent-
trajectory extensions. The [literature review](LITERATURE_AUDIT.md) credits
these ingredients and distinguishes the pairwise observed-output witness,
retained refinement geometry, and implemented finite-family selection.

Feature count is a representation-size proxy. The experiments do not establish
equal runtime, storage, or statistical capacity across all methods. Relative
KAHM gains against geometric controls coexist with stronger affine-hull results
on oscillators and stronger autoregressive/quadratic predictors on PenDigits.

Code and scalar summaries are distributed in the repository; some full score
archives are supplied separately or regenerated from frozen runners. The study
READMEs identify these components without treating an index of hashes as the
full archive. Historical protocols and numerical records remain unchanged;
current documentation states the revised interpretation of reused data.
