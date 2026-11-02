# Predictive contribution of geometric soft-state observables

## Question and inferential scope

This prospective computational experiment asks what memberships, within-region
position, and reconstruction departures contribute to direct forecasts of the
same future observation. It tests a representation hierarchy introduced in the
accompanying manuscript. Earlier forecasting results informed this question;
the design does not claim independent hypothesis discovery or external
preregistration. Source, configuration and analysis are frozen before production
data generation. Fresh repetitions under inherited laws supply new evaluation
units, not new application domains. Earlier terminal observations are not pooled.

The four representations are G (membership gates), P (gates and weighted local
principal coordinates), K (P plus KAHM departures), and R (P plus positive-kernel
RBF departures). P is the common representation. K and R are alternative
enrichments of it. The target is always the observed future y at the specified
horizon. The experiment does not estimate autonomous coordinate dynamics,
unrestricted information sufficiency, or Markov sufficiency.

## Data and fixed budget

The fixed design has 32 independent repetitions, four inherited numerical laws
(linear AR, nonlinear AR, Van der Pol, Duffing), and added-noise levels 0 and 0.1,
giving 256 task suites. Scalar histories have width 12, forecast horizons are 1, 4, 12,
and retained origins are separated by 4 steps. Each base episode has 72 stored
values. The original oscillator observations already include their inherited
sensor noise at added-noise level zero.

Independent episode sets A/B/C/D contain 64/32/32/128 episodes. Retained history
rows are capped at 512/256/256 on A/B/C; every one of the 1664 D rows is retained.
The original observation-law functions and window indexing are reused without
modification. Added Gaussian noise has amplitude equal to the declared noise
level times the standard deviation of original A episode values. Base episodes,
unit-noise draws, row selection and fitting streams are paired across the two
noise levels and all compared methods. The new master seed is 2026110203.

The complete stream roster is checked for intended-distinct collisions and
against the recorded predecessor exclusions before generation. These checks
establish deterministic stream identities, not cryptographic isolation.
The repetition count is fixed before production; no sample size extension,
candidate addition or stopping decision may depend on terminal results.

## Shared geometry and native dimension

At each cap, all four arms use the same A-fitted standardization, reference
partitions, centers, principal coordinates and locality gates. There are
partitions with 2, 4, 8 local models, at most 24 references per model, and the
inherited 512-row bank limit. The original Gaussian kernels, reconstruction
regularization rule and relative-denominator numerical checks are retained.

The native initial local rank is min(cap, ambient dimension, references-1),
followed by the native positive-variance and coordinate-range reductions.
Full ambient rank is permitted. The candidate caps are 1, 2, 4, 8, 12; cap 20 is an
alias of 12 at this history width and is checked on artificial data rather than
counted as another search opportunity. No historical rank-two or ambient-minus-one
restriction is reintroduced.

The gate vector is nonnegative and sums to one within each partition. Its
concatenation is a product-of-simplices representation. The coordinate and
departure blocks may be signed. KAHM and RBF share memberships; KAHM changes
the reconstruction observables through its regularized inverse-Gram formula.

## Block transformations and direct readouts

The raw common matrix is separated into its gate and local-position blocks.
Each of the gate, position, KAHM-departure and RBF-departure blocks is centered
and scaled using A alone, with feature-scale floor 1e-6. Each standardized block
of width p is separately transformed by v/sqrt(1+||v||^2/p). Empty blocks stay
empty. There is no subsequent standardization of a concatenated vector.

The transformed gate block is identical in G/P/K/R. The transformed position
block is identical in P/K/R. Thus adding observables does not rescale an existing
block. This differs from bounding the complete common vector in older studies
and defines a new procedure; earlier results are not relabeled as its outcomes.
K and R have the same nominal departure width. Equal widths and search counts
do not imply equal function spaces, conditioning, runtime or statistical capacity.

Every arm fits an affine ridge readout directly to the observed A targets,
separately by horizon. The intercept is unpenalized. The penalty grid is
1e-4, 1e-3, 1e-2, 1e-1, 1. B selects the penalty independently for each cap, arm and
horizon, with first-minimum tie resolution. There is no raw-history baseline,
residual target, correction multiplier or calibration. C predictions are saved
as diagnostics; C does not affect any fitted quantity or selection.

The inherited checked ridge solver is used unchanged: a NumPy SVD failure may
invoke the documented SciPy gesvd recovery, with the same objective and an
explicit numerical record. Unexpected nonfinite values or model failures stop
execution. No failing task is excluded from an apparently completed cohort.

## Primary shared-bank comparison and secondary selection

For the primary comparison, P selects one cap by its B loss averaged equally
over horizons after per-horizon penalty selection. Losses are divided by the
corresponding A-target population variance (ddof 0), floored at 1e-12. Exact ties
choose the first cap in ascending order. All four primary arms then use that
same selected bank, retaining their own B-selected penalties.

The primary estimand is therefore enrichment of a validation-selected common
representation. It is not each method's individually optimized bank. A separately
reported secondary context lets each arm select its own cap using the same B
objective and the same candidate fits. It answers a complete-procedure question
and cannot replace an unsuccessful primary contrast. Candidate cap curves,
selected ranks, feature widths, fallbacks and complete fitting/selection costs
are reported as diagnostics.

## Primary analysis

For each context and arm, squared errors are divided by A-target variance and
averaged over origins and horizons within a task. Nonlinear repetition losses
average the six nonlinear-law/noise cells equally. Linear repetition losses
average the two linear-law/noise cells. The 32 repetition aggregates are the
inferential units; overlapping windows are not treated as independent samples.

There is one family of four prespecified primary contrasts in the shared-bank
context:

1. P versus G on nonlinear laws: a 3% relative superiority margin.
2. K versus P on nonlinear laws: a 3% relative superiority margin.
3. K versus R on nonlinear laws: a 3% relative superiority margin.
4. K versus P on the linear law: a 5% relative noninferiority margin.

For each repetition define d=L(candidate)-c L(reference), with c=0.97 for
superiority and c=1.05 for linear noninferiority. The upper bound is mean(d)
plus t_(1-0.05/4, 31) times sd(d)/sqrt(32). A criterion is established only when
this one-sided Bonferroni-adjusted paired-t upper bound is strictly negative.
The t model is an inferential approximation for independent repetition means;
the adjustment controls the declared four-contrast family under that model.
The margins are inherited research conventions, not deployment-derived costs.

Reported percentage gains use 1-mean(candidate)/mean(reference), not an average
of per-repetition percentages. Independent-cap, law, noise, horizon and other
diagnostic comparisons are secondary and descriptive. Any unadjusted 95%
interval is labeled as such and cannot be promoted into the primary family.
The analysis reports all arms and conditions, including unfavorable outcomes.
Failure to establish a margin does not establish equivalence.

## Execution order and retained evidence

Artificial-data preflight checks precede the source freeze. They exercise shared
block identity, partition normalization, native cap aliases, zero-rank handling,
portable-state roundtrips, ridge selection and the independent audit. They do
not use production seeds or terminal outcomes.

The production order is: freeze source/configuration/analysis; publish and verify
the frozen source; write its commit-bound publication receipt; prepare A/B/C;
fit every task and seal the complete fitted-state roster; generate D queries
and separately retained targets; predict from queries only and seal the entire
forecast roster; release targets; analyze; independently audit. Fitted states,
selected and candidate records, data identities, forecasts and numerical logs
are retained. No fitting API accepts D targets. Hash seals document procedural
ordering and detect changes; they are not independent oversight or access control.

The audit independently recomputes forecast losses, repetition aggregates and
the four bounds, checks the canonical task/array roster, and verifies structural
and source identities. Its report specifies the extent of model replay and data
regeneration. Computational verification is distinct from independent scientific
replication.

## Interpretation and completion

This study is complete when the fixed cohort, analysis and audit are complete,
regardless of which primary contrasts are established. Useful local position,
useful reconstruction enrichment and a KAHM-specific increment are distinct
possible findings. Inconclusive or unfavorable results narrow the manuscript's
interpretation. They do not trigger a new tuning search on these terminal data.
