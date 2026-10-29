# Scientific extension: representation limits, cost decisions, and forecasting evidence

This account records the scientific extension relative to repository commit
`c15a7526f64b4b47ac09998548a6beee1f763c22`. The revised manuscript is titled
**Certifying Prediction Limits and Representation Refinement with Kernel
Affine Hull Machines**. The repository remains a reusable, venue-independent
research resource.

The central claim is a framework for assessing a fixed representation and
sensitivity budget against an expected-error tolerance. KAHM is one implemented
application. The evidence supports task-dependent forecasting comparisons and
specific certification decisions, rather than universal KAHM superiority or
uniform superiority of the pairwise lower bound.

## Scientific questions addressed

| Question | Revised analysis and evidence | Interpretation limit |
|---|---|---|
| Was the original centered comparator sufficiently strong? | The manuscript corrects the interpretation of the original zero-variance comparison and adds direct standard-deviation concentration with known ranges. A dual-certified empirical-minimum KL comparator targets the same full scalar Lipschitz class. | The earlier gap partly reflected unused structural information. No universal finite-sample dominance is claimed. |
| Can the pair witness be loose beyond two atoms? | An exact slack identity separates conditional residual bias from geometric slack. Continuous-support equality, rare-conflict, and affine-versus-Lipschitz examples identify distinct cases. | A positive affine minimum is not used to lower-bound a larger Lipschitz class. Zero witness does not establish adequacy. |
| Can conditional pairing improve the bound without changing the target risk? | A stratified theorem weights conditional witnesses by population masses and accounts for their estimation, random counts, and sparse strata. Compatible conditional optima give sharpness cases. | Partitions can discard useful cross-stratum conflicts and increase finite-sample uncertainty. Conditional optimizers must be globally compatible. |
| Do lower certificates change a substantive decision? | Joint upper and lower bounds yield an interval for minimum feasible representation cost among declared classes. A controlled positive-tolerance study demonstrates cost certification and all-class exclusion. | The accepted fitted predictor can remain identical to upper-only selection. Feature-count cost is not runtime, and optimality is only within the prescribed family. |
| Is the empirical assessment broader than a small pen-writing cohort? | A separate speech-energy study uses documented recording identifiers, a frozen group split, 978 calibration groups, eight model families, and independent numerical reconstruction. | Group-record iid sampling remains an explicit assumption; released-clip eligibility is retrospective. Conventional regressors outperform KAHM on this task. |
| Are the contribution and predecessors distinguished? | The paper credits the paired Lipschitz argument of Akash et al., scalar Lipschitz optimization of Gottlieb et al., established group norms, and classical concentration and risk-control tools. A population comparison with the inverse-contrastive bound is derived. | The empirical-minimum and KL comparators are established tools used to strengthen evaluation, not claimed new concentration results. |
| Can a reader assess uncertainty and absolute effect size? | The primary forecasting table now includes sample SD across fits. The matched table places absolute MSE, fitting SD, relative change, and paired seed counts together. Notation and sampling conventions are explicit. | Repeated fits on one cohort do not supply independent population cohorts; descriptive differences are not calibrated superiority statements. |

## New lower-certificate comparison

The fixed protocol evaluates eight finite joint laws at calibration sizes
64, 256, 1024, and 4096, with 300 repetitions per law and size. It retains all
9,600 trials and 76,800 primary and secondary endpoints. Each primary method
has its own one-sided 0.05 failure budget; the four methods do not jointly
constitute one 95% event.

The benchmark compares the original pair empirical-Bernstein certificate,
a range-aware centered certificate, a certified empirical-minimum KL lower
bound, and stratified empirical-Bernstein calibration with estimated masses.
Population and empirical scalar Lipschitz programs have feasible primal and
dual values. The maximum empirical gap is approximately `4.0e-11`; floating-point
guards do not constitute formal interval arithmetic.

The results include both advantages and limitations:

- In partial collapse, the minimum MSE is 0.2533. Stratification raises the
  population witness from 0.0867 to 0.2533. At `n=4096`, mean lower endpoints
  are 0.0646 for ordinary pairing, 0.1772 for primary stratification, and
  0.2178 for empirical-minimum calibration.
- At RMSE tolerance 0.25 in that law, exclusion succeeds in 211/300 ordinary
  pair trials and 300/300 stratified and empirical-minimum trials.
- In the rare-conflict law, the true minimum is 0.025. Primary stratification
  still does not exclude tolerance 0.10 at `n=4096`; empirical-minimum
  calibration does so in 300/300 trials. Support-aware stratification succeeds
  in 284/300, and additionally using known masses succeeds in 300/300.
- The three-point V law is an adverse stratification case: its population
  witness decreases from 0.2222 to 0.1667.

All primary methods have zero observed violations on this grid. This is a
Monte Carlo observation, not proof of uniform coverage. Each cell retains
Monte Carlo intervals. Timing starts after finite-support score-table
precomputation and includes empirical optimization; it is not a general
continuous-support or deployment benchmark.

The original two-atom archive remains intact, including its single violation
in 28,800 trials. In its collapsed case at `n=4096` and tail `0.05/3`, using
known zero input variance with the old target correction already gives an
analytic expected-endpoint lower bound of 0.9030575. This is not a newly
simulated mean. The original centered mean was 0.5327 and the pair mean 0.9177;
the associated fourth-root penalty was implementation-specific.

## Decision value at a positive tolerance

The controlled mechanism has history `(Z,S)` and future
`Y=0.8*S*Z+0.1*N` in the repairable regime, or `Y=0.8*S*R+0.1*N` in the
irreducible regime. Independent signs and a Bernoulli state give exact finite
population risks. Three retained representation prefixes have costs 1, 3,
and 5, with common sensitivity budget 2 and RMSE tolerance 0.30.

The experiment retains 100 independent repetitions per regime and five nested
calibration sizes up to 65,536. One 0.05 event per regime/repetition covers
all prefixes, sample sizes, upper bounds, and both stratification partitions.

Upper calibration accepts the repaired cost-5 forecast in every repetition
from `n=1024`. Stratified lower bounds additionally establish minimum cost 5
in 100/100 repetitions at `n=16384` and `65536`. Ordinary pairing never closes
that cost interval on the prescribed schedule. In the irreducible regime,
stratification excludes all classes in 1/100 repetitions at 16,384 and
100/100 at 65,536. Small-sample disadvantages remain: at 4,096 observations,
ordinary pairing excludes the constant class in all repetitions while
stratification excludes none. These substantial sample requirements are
reported without extrapolating frequency of success to natural forecasting.

An audit of the earlier forecasting archives shows why the distinction matters:
all 186 horizon-specific lower endpoints for full-family validation winners
were zero. The new study establishes a concrete added class-level conclusion,
rather than implying that positive endpoints already changed those earlier
selected models.

## Speech-energy evidence and protocol accounting

The new data source is the official Mini Speech Commands archive, attributed
to Pete Warden and TensorFlow under CC BY 4.0. The fixed format restriction
retains 7,178 clips from 1,630 recording identifiers. The train/validation/
calibration/test split has 244/163/978/245 identifiers; an identifier never
crosses splits. Source identifiers can denote web recording sessions and do
not prove unique human identity.

The target is a causal 10 ms frame-power observable, floored at -60 dB relative
to full scale and mapped to [0,1]. There is no utterance-specific normalization
using future statistics. Histories and horizons are fixed, and losses average
within recording groups before equal group weighting. Three complete fits
select from 486 candidates each before any held-out losses are evaluated.
The 72 reported upper bounds share a total failure probability of 0.05 under
the stated iid group-record assumption.

All six learned families certify the prespecified 18 dB requirement at every
horizon in all three fits. No family certifies all-horizon 6 or 12 dB.
Persistence and the constant predictor do not pass 18 dB across all horizons.
At 120 ms, mean test RMSEs are 12.73 dB for KAHM, 12.51 for ridge AR, 12.47
for quadratic regression, and 12.39 for Gaussian-kernel regression. These
accuracy comparisons are descriptive. The tolerances are explicit observable
resolution levels, not requirements supplied by an audio application.

The first execution interleaved calibration and test scoring after every
model had already been frozen. A deterministic replay corrects the order,
finishing and hashing all calibration calculations before test scoring.
Every fitted parameter and geometry value, group loss, and non-timing result
matches exactly. Both executions and the deviation are retained; they are one
study, not independent replications. Witness-analysis reporting corrections
are similarly separated from immutable samples and endpoints.

## Verification and presentation

The current verification includes:

- 54 standard code tests and 7 mathematical benchmark tests, all passing.
- Reconstruction of 76,800 new lower endpoints, maximum discrepancy
  `1.11e-16`, and checks of all finite population optima and dual directions.
- Independent verification of 600 decision-study fits, 3,000 endpoint records,
  6,000 mass linear programs, and the saved random streams.
- Independent reconstruction of 7,178 speech observables, 88,056 group-level
  losses, 24 validation selections, and all 72 KL upper endpoints.
- Recalculation of 5,388 original lower/upper certificate pairs and 792 paired
  improvement bounds from saved moments, with zero arithmetic discrepancy.
  This is not a new reconstruction of unavailable original synthetic arrays.
- Independent checks of the revised tables, including 63 matched MSE means,
  63 fitting SDs, 42 relative effects and paired win counts, and every speech
  table entry.

The complete manuscript uses the standard 11-point article/preprint layout,
Times text, and unchanged six-inch by 8.5-inch text area. The abstract has
185 whitespace-delimited words, or 195 with hyphenated terms separated.
The revision consolidates repeated experimental description and reproduction
instructions. All theorem proofs, primary comparison rows, adverse findings,
and sampling limitations remain in the PDF. Secondary fixed-dimension plots
and detailed scaling tables remain in the cited, reproducible repository
archive. The layout verification records the final page count and PDF hash.

These checks support the stated mathematics and numerical evidence. They do
not establish an exhaustive novelty search, formal proof verification, iid
sampling of public recording groups, or editorial acceptance. Scientific
responsibility and publication declarations remain with the authors.
