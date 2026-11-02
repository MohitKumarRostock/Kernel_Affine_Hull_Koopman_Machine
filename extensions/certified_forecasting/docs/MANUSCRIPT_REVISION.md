# Scientific overview: representation limits and calibrated forecasting

The study asks whether a fixed representation can support a declared prediction
accuracy, whether a fitted predictor achieves it, and whether a more costly
representation is necessary within a predefined family. The target is the
observed signal under a fixed episode and forecast-origin law. The theoretical
results apply to fixed representations beyond Kernel Affine Hull Machines
(KAHMs); KAHM forecasting is the principal application.

## Contribution and relation to existing work

The pair witness measures target separation that a decoder with a specified
Lipschitz sensitivity cannot reproduce. The analysis separates its population
slack from sampling uncertainty, characterizes equality, and gives examples
where the witness is arbitrarily small relative to the optimal risk.
Stratified pairing can recover conflicts that unconditional pairs rarely
encounter, while reweighting strata preserves the original marginal risk.
The finite-sample certificate accounts for random stratum counts, sparse
strata and estimated probability masses. A partition can also weaken a witness;
there is no general dominance assertion.

Simultaneous class lower bounds and fitted-risk upper bounds support different
conclusions. Upper calibration can accept a fitted forecast. Lower bounds can
exclude every permitted decoder of an inadequate representation. If all cheaper
classes are excluded and one fitted forecast is accepted, the resulting cost
interval certifies the least feasible cost among the specified alternatives.
Here cost counts retained feature coordinates; it is not a runtime measure.

These results use established ingredients. Pairwise Lipschitz regression
arguments are a close antecedent in [Akash et al. (2021)](https://arxiv.org/abs/2102.08343).
The prior [KAHM prediction-limit study](https://arxiv.org/abs/2609.32652) already
includes reconstruction features, centered lower certificates, empirical-risk
alternatives, upper-risk bounds and independent-trajectory evaluation.
Finite-sample Lipschitz regression has a standard constrained optimization
formulation; the new comparison uses that full class rather than substituting
a narrower affine decoder. Empirical Bernstein and standard-deviation
concentration, bounded-loss Chernoff bounds, simultaneous testing and group
norms are established tools. The study does not claim these tools, independent
episode calibration, or the basic paired-residual argument as new. The
[related-work section](../paper/sections/related_work.tex) and
[literature context](LITERATURE_AUDIT.md) identify the relevant sources and
contribution boundaries.

## Evidence and statistical units

| Study | Independent unit and design | Supported interpretation |
|---|---|---|
| Finite-law witness comparison | 300 independent calibration samples at each of four sizes for each of eight fixed laws | Tightness, positive-tolerance exclusion and computation for known finite supports; every method uses the same Lipschitz class |
| Controlled decision experiment | One forecast per independent episode; 100 fitting/calibration repetitions in each of two specified regimes | Additional minimum-cost and infeasibility conclusions, including their substantial calibration requirements |
| Original and matched forecasting | Independently initialized synthetic episodes; new synthetic streams for the confirmatory comparisons | Task-specific forecasting and calibration under the stated numerical laws |
| Pen trajectories | Thirteen eligible held-out writer groups, inferred from source metadata | Descriptive forecasting on that cohort; repeated fitting seeds do not create new writers |
| Speech energy | Disjoint recording-identifier bundles; 978 calibration and 245 test bundles | Upper-risk calibration under an explicit iid common-population bundle assumption |

### Lower-bound comparison

The [frozen protocol](../experiments/witness_comparison/protocol.md) specifies
eight laws with deterministic or noisy targets, partial collapse, rare
conflicts and an affine-versus-Lipschitz mismatch. Calibration sizes are 64,
256, 1,024 and 4,096. All 9,600 trials and 76,800 primary/secondary endpoints
are retained. Each method separately receives one-sided failure probability
0.05; there is no joint 95% claim across methods or simulated conditions.

The primary comparators are disjoint-pair empirical Bernstein, a range-aware
centered bound, a dual-certified empirical-minimum bound with Kullback–Leibler
inversion, and stratification with estimated masses. The centered comparator
uses direct standard-deviation concentration and available support information.
The empirical-minimum comparator targets the same full Lipschitz class. Its
lower objective comes from nonnegative dual multipliers, rather than treating
a primal minimization value as a lower bound.

The results are mixed. On partial collapse, minimum MSE is 0.2533 and the
unconditional population witness is 0.0867; stratification recovers the full
0.2533 at population level. At 4,096 observations, the mean calibrated
endpoints are 0.0646 for ordinary pairing, 0.1772 for primary stratification,
and 0.2178 for empirical minimization. At RMSE tolerance 0.25, their exclusion
counts are respectively 211, 300 and 300 out of 300 trials. On the three-point
V law, stratification instead reduces the population witness from 0.2222 to
0.1667.

For rare conflict, the minimum is 0.025 and ordinary pairing remains
uninformative at the reported sizes. Empirical-minimum calibration excludes
RMSE tolerance 0.10 in all 300 trials at size 4,096; primary stratification
excludes it in none. A secondary support-aware stratified bound succeeds in
284 trials, and supplying known stratum masses raises this to 300. These
comparisons distinguish the value of pairing from additional prior information.
The [complete summaries](../results/witness_comparison/analysis/summary.csv)
and [exclusion counts](../results/witness_comparison/analysis/exclusion_power.csv)
retain every law, size, method and tolerance.

All primary methods have zero observed coverage violations in the fixed grid;
that observation does not prove coverage on other distributions. Numerical
reconstruction agrees within 1.12e-16, and the maximum empirical primal–dual
gap is 4.01e-11. The [source and audit record](../results/witness_comparison/SOURCE_AND_AUDIT.md)
distinguishes the frozen simulation from later reporting-only corrections.
Timings are descriptive prototype measurements after representation lookup,
using a precomputed finite-support pair-score table. They are not complexity
or deployment-speed claims for continuous, high-dimensional representations.

### Decisions enabled by lower calibration

The [decision study](../results/decision_value/README.md) uses three retained
representation costs, 1, 3 and 5, at the fixed RMSE tolerance 0.30. Its exact
class-optimal MSEs are `(0.138, 0.138, 0.01)` in a repairable regime and
`(0.138, 0.138, 0.138)` in an irreducible regime. These are controlled
representations for the general certificate, not a KAHM performance result.

Upper calibration accepts the repaired cost-5 predictor from 1,024 episodes
onward in all 100 repetitions. Stratified lower calibration additionally
excludes both cheaper classes in all repetitions at 16,384 and 65,536
episodes. The selected predictor is unchanged; the added conclusion is that
its representation cost is minimal within the prescribed family. In the
irreducible regime, all three classes are excluded in one repetition at
16,384 episodes and all 100 at 65,536. Ordinary lower calibration does not
establish either complete conclusion at the tested sizes. At 4,096 episodes,
however, ordinary pairing excludes the constant class in every repetition
while stratified pairing excludes no class. Both the added decision value and
the cost of conditional estimation are therefore visible.

One failure budget 0.05 covers all five sample sizes, three prefixes and
reported bounds within one regime/repetition; it does not cover all 200
repetitions simultaneously. The [independent verifier](../results/decision_value/verification.json)
reconstructs 600 fitted models, 3,000 endpoint records and 6,000 mass linear
programs. The experiment establishes a mechanism, not how often such a
minimum-cost conclusion will occur in a natural application.

### Forecasting and the new speech application

The earlier matched geometry–decoder comparison includes 13,440 candidates
and all 42 primary task–architecture–alternative contrasts. KAHM has lower
mean MSE in 28 and higher mean MSE in 14; these contrasts share tasks and
models. Affine-hull features are substantially stronger under Frobenius-direct
and rollout heads on the oscillator tasks. On PenDigits, conventional
quadratic and autoregressive forecasts remain more accurate than KAHM.
Only one of 180 fresh synthetic paired geometry comparisons has a positive
improvement certificate. The other comparisons are inconclusive, rather than
proofs of equal risk. The [matched study records](../results/matched_geometry_study/README.md)
retain all favorable and unfavorable effects.

The new [speech protocol](../experiments/speech_forecasting/protocol.md) uses
TensorFlow Mini Speech Commands to forecast a causal, floored acoustic energy
observable. The retained 7,178 exact-one-second clips have 1,630 recording
identifiers, split into 244 training, 163 validation, 978 calibration and 245
test groups. Each statistical unit is an identifier's complete clip bundle.
The protocol averages origins, then clips, and then weights identifiers
equally. The observable represents frame power floored at -60 dBFS on [0,1];
its guarantees do not concern the raw waveform or unclipped power.

Across three fits on the same split, KAHM test RMSE averages approximately
4.00, 7.58 and 12.73 dB at 10, 40 and 120 ms. Linear autoregression gives
3.49, 7.22 and 12.51 dB; the Nyström Gaussian-kernel comparator gives
3.51, 7.17 and 12.39 dB. KAHM improves over persistence at the longer horizons
but does not establish superiority over the conventional predictors.

One budget 0.05 covers all 72 selected-family/seed/horizon upper bounds.
All six learned nonconstant families certify the all-horizon 18 dB tolerance
in all three fits; persistence and the constant model do not. No family
certifies an all-horizon 6 or 12 dB requirement. These are transparent
prespecified resolution levels, not requirements supplied by an operational
audio application. The [summary](../results/speech_forecasting_v1_replay/analysis/summary.json)
and [verification](../results/speech_forecasting_v1_replay/verification.json)
retain the complete results.

The recording identifiers are documented session/group proxies, not biometric
proof that every identifier belongs to a different person. Independent,
identically distributed bundles from a common population remain an explicit
sampling assumption. Provider trimming and the exact-duration filter restrict
the target population to eligible released clips. Fitting repetitions share
these groups. An initial scoring-order deviation interleaved calibration and
test scoring for already-frozen models; a strict-order replay completes all
calibration calculations before any test scoring. All 88,056 recorded losses,
72 endpoints and saved predictor values agree exactly. Both the deviation
and the replay are documented.

## Interpretation and limits

The strongest supported conclusion is a framework for comparing representation
limits, fitted accuracy and predefined costs under one fixed target law. The
experiments do not establish universal KAHM superiority, universal tightness
of a pair witness, or an adaptive representation-learning algorithm. The
empirical-minimum comparator is often tighter on the small finite-support
problems, and conditional estimation can require many observations.

Population guarantees depend on the stated target bounds and independent
sampling units. Reusing a previously examined cohort for an adapted follow-up
is not made confirmatory by new fitting seeds. The PenDigits follow-ups remain
descriptive. The speech result supplies a larger group-based application under
an explicit sampling assumption, not a distribution-shift or continuous-stream
guarantee. Numerical audits, independent reconstruction and deterministic
replays check the calculations; they do not establish the sampling assumptions
or replace the mathematical arguments. Reproduction instructions and the
separate [scientific verification record](SCIENTIFIC_VERIFICATION.md) document
those checks without treating them as evidence of publication readiness.
