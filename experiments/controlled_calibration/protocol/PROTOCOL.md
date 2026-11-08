# Step 17: matched-computation calibration of a learned geometry hierarchy

## Scientific question and inheritance

The experiment compares two uses of exactly the same disjoint-block signed
dual SSE values. The manuscript's block hierarchy calibrates each centered
score, SSE/(b-1), as the mean of independent blocks. The matched-computation
comparator sums these valid block dual values, divides by the total episode
count n, and treats the result as a lower value for one full-sample empirical
Lipschitz risk minimum. Taking the maximum of valid lower values for this
same minimum needs one empirical-minimum confidence event and no new QP.

The common upper endpoint concerns a fixed fitted predictor under the same
ordinary squared loss. The question is whether the block hierarchy adds
decision resolution beyond this cheap comparator, given identical solves,
features, observations and fitted upper bounds. This is not a claim that
residual identities, KL concentration, or the current manuscript's empirical
minimum theorem are new.

The two smooth-product mechanisms and their target units, tolerance and norm
budget are inherited from the earlier learned-KAHM decision study. All
training, support, validation, calibration and terminal episodes are new,
and every KAHM bank is fitted anew using the validated current native engine.
Old fitted banks and scientific observations are not used in this experiment.
All earlier files and evidence remain immutable.

## Fixed task and independent units

There are 12 complete independent learning repetitions in each of two regimes.
The two input coordinates are independently X_j=S_j(0.9+U_j), where S_j is
uniform on {-1,+1} and U_j is uniform on [-0.1,0.1]. In the repairable regime,
Y=0.8 X_1 X_2+E. In the irreducible regime, Y=0.8 V_1 V_2+E, where V is an
independent hidden draw from the input law. In both regimes E is independently
uniform on [-0.05,0.05]. The target lies in [-0.85,0.85]. A row is one complete
independent episode at one forecast origin. There is no within-trajectory
pseudoreplication.

Each learning case has five independent pools:

| Pool | Episodes | Allowed use |
|---|---:|---|
| Geometry support | 256 | Input scaling, KMeans partitions and native references; no target is generated |
| Head training | 512 | Fit the constrained ridge head |
| Validation | 256 | Choose one common geometry and then each prefix's ridge |
| Calibration | 4096 | Fixed-class lower and fixed-head upper endpoints; leading prefixes of size 256, 1024 and 4096 |
| Terminal | 1024 | Independent descriptive forecast evaluation after calibration audit |

The loss is ordinary scalar squared error, bounded by 1.7^2=2.89 because each
fitted prediction is clipped to the declared target interval. Losses are not
capped. The MSE tolerance is 0.1. Target bounds, tolerance and kappa=2 are
inherited mechanism-study choices, fixed before new data; they are not chosen
after observing a lower endpoint.

The irreducible target variance is known and the constant zero predictor is
in every declared class, so its population class minimum equals that variance.
For repairable, 0.05^2/3 is a Bayes floor, not an asserted exact risk minimum of
a learned representation. A terminal average of conditional risk over sampled
inputs is not relabeled an exact integrated fitted population risk.

## Geometry and nested classes

The native geometry consists of two retained soft blocks, with 2 and 4 states.
The two retained prefixes therefore have 2 and 6 soft coordinates. Physical
inputs are not appended to the feature vector.

The common geometry menu is the Cartesian product of MinMax01/standard input
scaling, local caps 1/2, and concentrations 0.5/4/16/64, giving 16 geometries.
The native default min(20,input_dimension,local_reference_count-1) is at most
two here; cap one is the smaller alternative. Scaling is fitted on support
inputs only. Constant axes map to zero and query inputs are never clipped.
Native partitions and retained references are shared across local-cap variants.
At most 24 reference columns are used per local encoder. Soft power smoothing
is tau=1e-6. The geometry routine also fits unused reference-regrouped controls;
their cost is charged, and their outputs never choose the model.

The feature metric is max_s ||Phi_s(x)-Phi_s(x')||_2 over retained blocks. A
scalar decoder has an unpenalized intercept and coefficient budget
sum_s ||w_s||_2 <=2, followed by clipping to the known target interval. Such a
decoder belongs to the corresponding bounded 2-Lipschitz class. The lower
certificates cover this full Lipschitz class, which includes the fitted linear
heads. Retaining old blocks and appending zero head coefficients preserves
feasibility. Duplicate blocks do not create more coefficient budget.

The ridge menu is 0, 1e-4, 1e-2. All 96 head candidates per case are retained.
One shared geometry is chosen by the smallest largest-prefix validation MSE
over all geometry/ridge combinations. Ties use geometry index then ridge index;
geometry order is MinMax01 before standard, cap one before two, concentration
0.5/4/16/64. Within this chosen common geometry, each prefix chooses the ridge
with smallest validation MSE, ties by ridge index. Choosing geometry separately
at each prefix is forbidden because it would destroy the selected nesting.

Constrained ridge uses the inherited BlockBoundedRidge implementation, tolerance
1e-7 and at most 20000 iterations. A finite nonconverged candidate is retained
with its gap. The final coefficient vector is moved inward only if needed to
have norm sum at most 2-1e-12; the training-mean intercept, objective and
Frank-Wolfe gap are then recomputed. Every raw and guarded coefficient is saved.

## Matched lower endpoints and common upper endpoint

For each frozen case and prefix, use consecutive disjoint blocks with sizes
2, 4, 8, 16 and 32. Solve once on all 4096 calibration observations and reuse
the corresponding leading blocks at n=256,1024,4096. The native block solver
is an unchanged copy of the prior signed-edge FISTA solver, with step 2/b,
at most 4000 iterations, tolerance 1e-7 and checks every 20 iterations.

All signed multipliers, primal candidates, guarded dual SSEs, upper SSEs,
gaps, iterations, convergence flags and feasibility residuals are retained.
Stopping-limit outcomes are retained. A nonfinite or malformed numerical
failure aborts the stage; no row, model, seed or failed run is silently replaced.

For each class and sample prefix:

1. Calibrate each of the five block-score means with its sharp scalar range
   D^2 floor(b^2/4)/(b(b-1)) and n/b independent blocks. The hierarchy endpoint
   is the maximum of these five simultaneous endpoints.
2. Divide each partition's sum of the same guarded dual SSEs by n. Each value
   is below the same full-sample empirical Lipschitz risk minimum. Maximize
   these values and use the existing empirical-minimum KL lower result with
   sample size n and ordinary-loss range D^2. Per-b recycled values and their
   maximum share one confidence event.
3. Compute one common bounded-mean KL upper endpoint from the already fixed
   fitted model's squared losses.
4. Report the maximum of hierarchy and recycled endpoints as an already
   jointly covered combined endpoint. It adds no solve or confidence event.

There are 1008 one-sided events: 24 cases x2 classes x3 sample sizes x
(5 block lower +1 recycled lower +1 fitted upper). Every endpoint receives
alpha=0.05/1008. Thus the five-member block family receives five times the
aggregate failure allocation of the single recycled endpoint. The union
guarantee is simultaneous across the whole declared experiment, conditional
on construction and selection data. Calibration cannot choose a new geometry,
predictor, target, sensitivity budget, sample-size menu, tolerance or partition.

## Primary decision comparison

The primary calibration size is n=1024. For repairable, the prespecified
resolution question is L_2>0.1 and U_6<=0.1: the cheaper class is excluded and
the refined fitted predictor is adequate. For irreducible, it is L_2>0.1 and
L_6>0.1: both classes are excluded. Count paired outcomes both, hierarchy-only,
recycled-only, or neither over the 12 independent learning cases per regime.
These are descriptive procedure counts; no population superiority inference
over new training datasets is asserted. All decisions and endpoints at n256
and n4096 are secondary and remain reported.

For each method, also report the cheapest lower-compatible prefix and cheapest
upper-accepted prefix. Equal finite values certify the minimum feasible prefix
among the two declared representation classes. Upper-only analysis establishes
fitted adequacy but cannot exclude an entire class. Failure of an upper bound
to pass, or failure of a lower bound to exceed the tolerance, is unresolved.

## Computation and numerical scope

Charge complete support geometry construction, unused regrouped native fits,
all shared development encoding, all candidate heads and validation. Record
selected numerical deployment payloads, including retained encoder arrays,
scalers, centers and heads, separately from Python/runtime overhead. The
payload must increase from prefix two to prefix six.

During terminal evaluation, time a separate uncached 1024-input query for
each retained prefix in a warm process, with feature and head time separated.
No fit-skipping benefit is claimed: the common geometry search is eager.
Certificate costs distinguish the shared block solves from upper, block-KL
and recycled-KL postprocessing. Maximal-n solve timings are not reported as
measured smaller-n runtimes.

Native folding scores use the inherited interval engine with precision ladder
80/128/192 bits, score error budget1e-10 and canonical-coordinate budget1e-9.
Power-normalized states use the inherited MPFR implementation with output
budget1e-9 and per-block simplex residual at most2e-13. Every required query
must pass; no per-query geometry or numerical target is changed to obtain a
favorable result. Statistical bounds concern the mathematical functions and
the declared sampling law. Float64 decoder fits, guarded block duals and KL
roots are replayed independently, but do not constitute an end-to-end interval
certificate of population prediction risk.

## Stream and evidence discipline

Use uint64 SeedSequence([2026110617,1195724617,regime_index,repetition,pool_index])
for the pool order support,train,validation,cal,test,geometry. Case identifiers
are12*regime_index+repetition, with repairable before irreducible. All144 streams
must be unique and disjoint from the full inherited reservations and every
Steps11-16 scientific/manufactured stream. Manufactured preflight seeds
2026110617000 through2026110617099 are reserved separately.

The ordered stages are source/protocol/stream freeze, all fits, independent
fit audit, all calibration, independent calibration audit, terminal evaluation,
and independent terminal audit. Each stage is write-once and hash-sealed.
The controller checks literal passing gates before generating any later data.
Every selected bank/head identity is recorded in calibration and terminal
generation receipts. Full saved evidence, all 96 candidate heads per case,
and all candidate validation losses allow independent replay without importing
the producer's prediction or aggregation functions.

Machine-readable constants and precise choices are in protocol17.json. Frozen
sources, stream roster and environment are in run/freeze.json. Cases use fit,
cal and test subdirectories r0 through r23. Each stage summary has explicit
by_case records. No main experimental sample may be generated before the
complete runner, certificate and independent-auditor manufactured preflights
are passing and root authorizes the single prospective controller run.
