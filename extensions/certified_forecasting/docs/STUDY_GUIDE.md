# Scientific study guide

The manuscript **Certifying Prediction Limits and Representation Refinement
with Kernel Affine Hull Machines** asks three questions about forecasting:
can a fitted predictor achieve a specified error tolerance, can any decoder of
a given representation achieve it, and what is the least feasible
representation cost among a predefined set of alternatives? This guide
explains the quantities being assessed and the role of each experiment.
The [reproduction guide](REPRODUCIBILITY.md) provides executable commands;
the [implementation guide](../IMPLEMENTATION_NOTES.md) defines the feature and
decoder calculations.

## Prediction task and risk

A causal history contains only observations available at the forecast origin.
A representation maps this history to features, and a decoder maps the
features to future observed outputs. Kernel Affine Hull Machines (KAHMs)
provide the principal representation: their coordinates describe relative
agreement with local reconstruction models. The statistical results also
apply to other fixed representations.

Let $X$ denote the observed history, $Y\in\mathbb R^q$ its target, and
$f(X)\in\mathbb R^q$ a prediction. The dimension $q$ is the number of target
coordinates being scored. Population mean squared error (MSE) is
$R(f)=\mathbb E\|Y-f(X)\|_2^2$, where $\|\cdot\|_2$ is the Euclidean norm
and the expectation uses the specified episode and forecast-origin law.
Root-mean-square error (RMSE) is $\sqrt{R(f)}$. A change of representation
is evaluated with the target and its sampling law held fixed. Studies that
average scalar errors over coordinates or horizons specify that normalization
separately.

For a representation $\Gamma$ and distance $d$ on its feature space, the
admissible decoder class has a declared sensitivity budget $\kappa\geq0$.
A decoder $g$ in that class satisfies
$\|g(u)-g(v)\|_2\leq\kappa d(u,v)$ for all feature values $u,v$ in its
domain. This Lipschitz constraint limits how much predictions can change as
features change. The optimal class risk is
$R^*=\inf_g R(g\circ\Gamma)$, where the infimum is over that stated class.

## Meaning of the certificates

An upper endpoint $U$ bounds the population risk of a particular fitted
predictor. A lower endpoint $L$ bounds the minimum risk of an entire decoder
class. On their simultaneous confidence event, an RMSE tolerance
$\epsilon\geq0$ has the following interpretations:

| Condition | Conclusion |
|---|---|
| $U\leq\epsilon^2$ | The fitted predictor achieves the requested tolerance. |
| $L>\epsilon^2$ | No decoder in the stated class achieves the tolerance. |
| Neither condition | The available bounds leave the decision unresolved. |

The lower endpoint does not assess unrestricted predictors using arbitrary
information. It applies to the stated representation, metric, decoder class,
and sensitivity budget. A zero lower endpoint is inconclusive, as is an upper
endpoint above the tolerance. The conclusions concern expected error under
the evaluation law; they do not give a prediction interval for every history.

The pairwise lower bound compares target separation with the separation that
the decoder can produce. Write $U_0=\Gamma(X)$ and let $(U'_0,Y')$ be an
independent copy of $(U_0,Y)$. With $(z)_+=\max\{z,0\}$, the unconditional
population witness is

$$
W_\kappa=\frac12\mathbb E
\left(\|Y-Y'\|_2-\kappa d(U_0,U'_0)\right)_+^2.
$$

The name *witness* refers to a measurable obstruction to small prediction
error: targets separated by more than the allowed prediction separation cannot
both be predicted accurately. The manuscript proves $W_\kappa\leq R^*$,
characterizes equality, and gives examples where the inequality is arbitrarily
loose. Its episode implementation compares different episodes at matching
origins and averages over the prescribed origin law.

Stratified pairing makes comparisons within groups defined from histories
before calibration. Weighting each conditional witness by the corresponding
population group probability preserves the original risk. When these
probabilities are unknown, their estimation uncertainty also enters the
lower endpoint. Stratification can expose rare conflicts but can also discard
useful comparisons across groups or incur larger finite-sample penalties.

## Representation refinement and cost

The KAHM construction appends fitted feature blocks while retaining all earlier
coordinate functions. A decoder is constrained by the sum of the Frobenius
norms of its coefficient blocks, and the associated feature distance is the
maximum Euclidean distance across blocks. Earlier decoders remain feasible by
assigning zero coefficients to new blocks. Exact copies of an existing block
do not enlarge the feasible class, because coefficients on identical blocks
can be merged without increasing the sum of their norms.

This invariance concerns the feasible predictor class. A quadratic ridge
penalty can change when coefficients are distributed across duplicates, so
separately fitted penalized models need not be identical. Refitting an entire
feature map also does not establish nesting unless the earlier coordinate
functions are retained.

For classes with predefined costs, simultaneous lower endpoints can exclude
cheaper alternatives while an upper endpoint certifies a feasible fitted
predictor. If every strictly cheaper class is excluded, the accepted cost is
minimal within that family. If all classes are excluded, the requested
tolerance is infeasible within the family. In these studies, cost is the
number of retained feature coordinates. It is not measured fitting time,
inference time, or memory usage.

## Experiments and independent sampling units

| Study | Design and independent unit | Scientific role |
|---|---|---|
| [Two-atom calibration](../configs/known_risk_v1.json) | 28,800 simulated calibration trials with analytically known risks | Checks empirical certificate behavior against exact population quantities. |
| [Finite-law witness comparison](../experiments/witness_comparison/) | Eight laws, four sample sizes, 300 independent calibration samples per law and size | Separates population looseness from finite-sample calibration penalties for the same scalar Lipschitz class. |
| [Representation-cost decisions](../experiments/decision_value/) | Independent complete episodes; 100 repetitions in each of two controlled regimes | Tests minimum-cost certification and exclusion of every prescribed class. |
| [Synthetic forecasting](REPRODUCIBILITY.md#independent-episode-campaign) | Independently initialized episodes from four main laws and two oscillator laws | Evaluates forecasting and risk certification under specified bounded observation processes. |
| [Input scaling](../results/input_scaling_study/) | Paired preprocessing arms; fresh synthetic confirmation and repeated fits on a fixed pen-writing cohort | Measures the dependence of KAHM performance on the input transformation. |
| [Matched geometry and decoders](MATCHED_GEOMETRY_PROTOCOL.md) | Three geometries, three decoders, seven tasks, five fitting seeds | Distinguishes representation effects from decoder and model-selection effects. |
| [Pen trajectories](PENDIGITS_PROVENANCE.md) | Thirteen eligible held-out writer groups, inferred from source metadata | Describes forecasting on the evaluated writer cohort and the limits of small-group calibration. |
| [Speech energy](../experiments/speech_forecasting/) | Disjoint recording-identifier bundles; 978 calibration and 245 test bundles | Evaluates upper-risk calibration under an explicit independent, identically distributed group-record assumption. |

Overlapping windows within an episode are not independent calibration samples.
Repeated fits on the same real-data cohort do not create new independent
writers or recording groups. The confidence allocations therefore use episodes
or groups and state explicitly which candidate families, horizons, and sample
sizes share an event.

### Comparing lower bounds

The eight-law study compares four primary methods: the disjoint-pair witness
with an empirical Bernstein correction, a range-aware centered bound, a
certified empirical-minimum bound with Kullback--Leibler inversion, and
stratified calibration with estimated group probabilities. The
[fixed protocol](../experiments/witness_comparison/protocol.md) specifies all
laws, sample sizes, confidence allocations, and secondary comparisons.
Each method receives its own one-sided failure probability 0.05; the comparison
does not assert a simultaneous 95% event across methods and simulated trials.

The empirical-minimum comparator optimizes over the same full scalar
Lipschitz class. A feasible primal minimization value is an upper bound on the
minimum; it cannot by itself provide the required lower bound. The implementation
instead evaluates a feasible dual objective and uses that value to calibrate
a lower confidence endpoint. The numerical brackets use floating-point guards
and feasibility checks, rather than formal interval arithmetic.

The results identify both successes and limitations. For partial collapse,
the population minimum MSE is 0.2533, the ordinary witness is 0.0867, and the
stratified witness is 0.2533. At 4,096 observations, the mean calibrated lower
endpoints are 0.0646 for ordinary pairing, 0.1772 for stratification, and
0.2178 for empirical minimization. For the three-point V law, stratification
instead reduces the population witness from 0.2222 to 0.1667.

In the rare-conflict example, the population minimum is 0.025. At 4,096
observations and RMSE tolerance 0.10, empirical-minimum calibration excludes
the class in all 300 trials, while the primary stratified method excludes it
in none. Secondary methods that use additional support information or known
stratum probabilities obtain stronger results. The
[endpoint summaries](../results/witness_comparison/analysis/summary.csv) and
[exclusion frequencies](../results/witness_comparison/analysis/exclusion_power.csv)
report every method, law, sample size, and tolerance.

### Decisions enabled by lower calibration

The controlled decision study uses costs 1, 3, and 5 and RMSE tolerance 0.30.
The exact class minima are $(0.138,0.138,0.01)$ in the repairable regime and
$(0.138,0.138,0.138)$ in the irreducible regime. These explicit representations
isolate the behavior of the general certificate; they do not measure KAHM
accuracy.

Upper calibration accepts the cost-5 predictor from 1,024 episodes onward in
all 100 repairable repetitions. Stratified lower calibration certifies its
minimum cost in all repetitions at 16,384 and 65,536 episodes by excluding
both cheaper classes. In the irreducible regime, it excludes every class in
one repetition at 16,384 episodes and all 100 at 65,536. At 4,096 episodes,
however, ordinary pairing excludes the constant class in all repetitions while
stratification excludes none. These outcomes show that the stronger class
conclusion can require substantially more data than fitted-model acceptance.
The [complete outcome table](../results/decision_value/README.md) includes all
sample sizes and both regimes.

### Forecasting comparisons

The matched study contains 13,440 candidate fits and 42 descriptive
comparisons of KAHM with another geometry under the same decoder. KAHM has
lower mean MSE in 28 and higher mean MSE in 14. These comparisons share tasks
and models and are not independent replications. Affine-hull features perform
better on the oscillator tasks under the overall-Frobenius and rollout
decoders. On PenDigits, autoregressive and quadratic predictors remain more
accurate than KAHM. Only one of 180 synthetic paired geometry comparisons
has a positive calibrated improvement endpoint; the others are inconclusive.

Speech forecasting uses a causal frame-energy observable, floored at -60 dB
relative to full scale and mapped to $[0,1]$. It evaluates 7,178 eligible
one-second clips from 1,630 recording identifiers. Across three fits, six
learned families meet the all-horizon 18 dB RMSE upper criterion; no family
meets the all-horizon 6 or 12 dB criterion. Conventional regressors have lower
test errors than KAHM. All 72 upper bounds share one failure probability 0.05.
The [speech study](../experiments/speech_forecasting/README.md) describes the
observable, group assumption, complete outcomes, and evaluation-order
provenance.

## Statistical interpretation and reproducibility

The guarantees require known target bounds and defensible independent
sampling units. Clipping a prediction or observing a finite sample range does
not establish a population target bound. A chronological split of one
arbitrarily dependent series does not establish independent episodes.

The PenDigits comparisons reuse a small fixed writer cohort, and the dataset
does not establish independent, identically distributed writer sampling.
Studies designed after examining that cohort remain descriptive even when
fitting seeds change. The speech identifiers are recording-group proxies;
they do not prove that every group represents a distinct person or that the
groups are independent population samples. Provider trimming and eligibility
filters also restrict both real-data tasks to retrospective released-data
populations.

Independent numerical reconstruction, source hashes, and deterministic
replays assess calculations and provenance. They do not establish sampling
assumptions or replace the mathematical arguments. The
[archive index](ARCHIVE_INDEX.md) identifies the version-specific protocols and
verification records without treating historical revision accounts as current
method descriptions. The manuscript's
[related-work section](../paper/sections/related_work.tex) identifies the
existing concentration inequalities, pairwise Lipschitz arguments, and
optimization tools on which the analysis builds.
