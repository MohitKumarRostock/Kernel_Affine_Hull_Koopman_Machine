# Reconstruction departures as predictive bases: fixed mechanism study

## Status and purpose

This protocol specifies one bounded mechanism experiment before its reserved
evaluation forecasts are computed. The executable configuration, source hashes,
prepared-input hashes and local freeze timestamp form the authoritative audit
record. This is a **local prospective freeze**, not a claim of public
preregistration. The final research report must identify the freeze and any
subsequent technical corrections. An experimental branch or a protocol is not
the publication deliverable: the manuscript and public repository must report
the completed experiment, including an unfavorable or unresolved result.

The question is whether a KAHM reconstruction provides useful functions beyond a
shared local affine basis for a restricted decoder, and whether that value
survives a bounded, explicitly controlled reconstruction-to-prediction map.
All reconstructions are deterministic functions of the observed history; they
cannot add statistical information beyond that history. This study concerns
basis value under a restricted, regularized decoder. It is not a search over new
forecasting niches. The three nonlinear laws
have been investigated in earlier development, so independent new draws support
inference conditional on these known laws; they do not provide an unseen-domain
generalization test. All earlier results remain separate and keep their original
protocols and normalizations.

The primary candidate is fixed as `kahm_departure`. Neither the best method on D,
the best law, the best noise level nor the best horizon may replace it. The
unbounded feature ablations are secondary diagnostics. A positive result advances
the research to genuinely reserved application evidence. A negative or unresolved
result at the fixed budget stops this branch of KAHM forecasting-performance
development and supports a manuscript about the conditions under which geometric
reconstructions do, or do not, support prediction.

## Mechanism and matched comparisons

All geometric methods use one A-fitted history standardization and one A-fitted
bank. The fixed bank has block sizes 2, 4 and 8, local rank 2, at most 24 reference
histories per local model and at most 512 A rows. References, principal subspaces,
centers, scales and locality gates are shared between methods. No target enters
this unsupervised bank construction. All numerical conventions, regularization
and denominator guards are fixed by `config.json` and the frozen source.

For a standardized history z and local model j, let A_aff,j(z) be its affine
projection, q_ji the reference histories, and k_j(z) their common projected
Gaussian kernel vector. Let K_j be the corresponding reference Gram matrix and
C_j=(K_j+lambda_j I)^(-1), with the inherited KAHM reference-energy regularization.
The two reconstruction maps are

    A_K,j(z) = [k_j(z) C_j Q_j] / [k_j(z) C_j 1],
    A_R,j(z) = [k_j(z) Q_j] / [k_j(z) 1].

For numerical evaluation, each log-kernel row is shifted by its maximum before
exponentiation. This common positive rescaling leaves either ratio unchanged.
KAHM denominator guard events and the resulting affine fallback must be counted;
they are part of the evaluated procedure, not silently discarded observations.

The shared gate g_j is the blockwise softmax of minus one half the squared
standardized center distance divided by the fitted local radius (the inherited
mean squared local distance). The common feature vector
contains these gates and their gated principal coordinates. The extra features
are the full-coordinate departures

    delta_K,j(z) = g_j(z) [A_K,j(z) - A_aff,j(z)],
    delta_R,j(z) = g_j(z) [A_R,j(z) - A_aff,j(z)].

All extra coordinates are retained, including those outside the local principal
subspace. Using the affine map in this expression gives the identically zero
departure. Consequently `common` is the common-only, nested basis comparison;
it is not claimed to match the extra statistical capacity. The matched RBF arm
has the same references, Gaussian distance, gate, coordinate width and selection
budget as the KAHM departure arm. Equal coordinate width and grids do not imply
equal effective complexity or computation time, which are reported separately.

Feature means and standard deviations are fitted on A alone, with each feature
standard deviation floored at 1e-6. The inherited shared-bank history scaler
retains a history standard deviation above 1e-12 and otherwise substitutes 1;
all geometry arms use that same scaler. The raw ridge/RBF input scalers instead
floor their A standard deviations at 1e-6. After feature
standardization, the common and departure blocks are bounded **separately** by

    T_B(v) = v / sqrt(1 + ||v||_2^2 / B^2),  B = sqrt(block width).

There is no second standardization after this map. Separate maps preserve exactly
the same common block in `common`, `kahm_departure` and `rbf_departure`; appending
a departure cannot rescale the shared coordinates. The two unbounded ablations
retain the bounded common block and omit only the departure bound. The algebraic
analysis must state which part of the departure is distinct from the common
features and characterize the deterministic feature/correction stability. A
bounded correction feature does not by itself bound the raw ridge baseline or
prove a globally continuous input-to-forecast map at denominator fallback
boundaries.

## Observations and replication

The frozen roster contains four laws, two added observation-noise levels and 24
independent repetitions: **192 task suites**. The repetition seeds are the
integers 946100 through 946123, inclusive. These seeds are disjoint from prior
development and evaluation seeds. Seed 975001 is reserved for development-only
checks and must not enter the reported roster. Independent random streams are
used for episode generation, added noise, row capping and model fitting. Their
derivation and hashes are recorded by the input preparation code.

The four laws are:

1. **Linear AR negative control:** y_t=.65 y_(t-1)-.15 y_(t-2)+epsilon_t,
   epsilon_t iid Normal(0,.1^2), two initial zero values, 256 discarded stored
   values and 72 retained observations beginning at y_256. This is a linear
   Gaussian base law with A-scaled independent Gaussian observation noise.
   Conditional on the realized noise amplitude, the D law is Gaussian with a
   linear conditional mean; across repetitions the random A-estimated amplitude
   induces a variance mixture. Neither exact stationarity nor equivalence of
   finite-sample ridge to an oracle with known noise amplitude is assumed.
2. **Nonlinear AR:**
   y_t=tanh(1.25 y_(t-1)-.65 y_(t-2)+.25 sin(3 y_(t-1))+u_t), with independent
   u_t uniform on [-.10,.10], two initial values uniform on [-.8,.8], and the
   first 80 stored values discarded. This is the inherited, unchanged generator.
3. **Van der Pol:** dx/dt=v and dv/dt=(1-x^2)v-x.
4. **Duffing:** dx/dt=v and dv/dt=-.2v+x-x^3+.3 cos(1.2t+phase).

The last two laws use the inherited float64 RK4 generator, step .025, observation
interval .1, burn time 10, independent initial x,v uniform on [-2,2] and phase
uniform on [0,2pi]. Their observed values are .95 tanh(x)+u, with independent u
uniform on [-.02,.02]. They are fixed numerical observation laws, not exact
continuous-ODE samples or noise-free latent states.

For each law and repetition, the four split episode sets are generated once.
The extra noise is independent standard Gaussian noise multiplied by eta times
the population standard deviation (`ddof=0`) of **all A unaugmented observed
episode values**. The levels are eta=0 and eta=.1. The same underlying episodes,
standard-normal arrays, cap indices and model seed are paired across these two
levels. The noise affects both histories and observed future targets. Models do
not receive latent trajectories, the noise array or the noise level as an input.
The eta=0 oscillator observations retain their inherited sensor noise. The new
Gaussian noise and the linear AR process are not bounded observations; previous
bounded-loss guarantees must not be carried over without a new argument.

Each episode contains 72 observations. Inputs contain the latest 12 observations
in chronological order and targets are horizons 1, 4 and 12 from the same origin.
Origin stride 4 gives 13 origins per episode, at indices 11,15,...,59. The splits
contain 64/32/32/128 independent episodes for A/B/C/D. Deterministic caps retain
512/256/256/all rows, respectively, giving 1,664 D origins and 4,992 D target
cells per task. Different origins within an episode overlap and are not
independent replicates. Across noise levels, tasks share episodes deliberately.

The 192 tasks therefore contain 958,464 scored target cells for each method.
This count is an accounting total, **not** the inferential sample size. The
primary inferential sample size is 24 independent repetition aggregates.

## Fitting and selection without D

The order is fixed:

| Split | Permitted use |
| --- | --- |
| A | Fit all scalers, bank parameters, reference kernels, decoder coefficients and raw models |
| B | Select hyperparameters separately for each horizon |
| C | Select correction strength separately for each horizon, then select one complete non-KAHM procedure |
| D | Reserved evaluation only; never supplied to a fitting or selection API |

The ten reported procedures, in deterministic order, are `ridge`, `raw_rbf`,
`raw_hgb`, `persistence`, `common`, `kahm_departure`, `rbf_departure`,
`kahm_unbounded`, `rbf_unbounded`, and `nonkahm_policy`.

The baseline is a raw-history ridge predictor. Linear decoders have free
intercepts and minimize mean squared residual error plus alpha times the squared
coefficient norm. Centered SVD/QR solves are used; an implementation must not
accidentally use the previous sum-SSE convention with the same numeric alpha.
The fixed alpha grid is [1e-4,.001,.01,.1,1]. Geometric arms fit corrections to
the A-fitted baseline residuals. They differ only in the specified feature map.
Hyperparameters are selected by B error separately for each horizon.

The raw RBF control uses 64 Nyström landmarks and bandwidth factors [.5,1,2]. It
concatenates the standardized raw histories and Nyström columns, scales the
combined columns using A alone, and selects bandwidth/regularization on B
separately by horizon. Histogram gradient boosting uses 100 iterations, leaf counts [7,15]
and learning rates [.05,.1], with configurations selected separately by horizon.
Persistence uses the latest observed value at every horizon. Other fixed raw
model settings are included in the executable configuration.

Every non-ridge arm has correction strength gamma selected from [0,.25,.5,1] on C
separately by horizon. Raw RBF, boosting and persistence are represented as
baseline+gamma*(raw_candidate-baseline). Geometric arms are represented as
baseline+gamma*geometric_correction. Ties follow the frozen candidate order; gamma
zero is a genuine baseline fallback. No procedure is re-fitted on B or C after
selection.

The non-KAHM policy selects one complete, already C-calibrated procedure using the
mean of its three C horizon errors normalized by A target variances. Its candidate
order is `ridge`, `raw_rbf`, `raw_hgb`, `persistence`, `common`, `rbf_departure`,
`rbf_unbounded`. Both KAHM arms are excluded. This policy is a genuine selection
procedure, not the D oracle minimum; its D predictions must equal those of the
stored C-selected method. All B scores, C gamma scores and final selections must
be saved.

## Loss, primary comparisons and fixed decision

For task t and horizon h, let v_th=max(var(A_Y[:,h],ddof=0),1e-12), where A_Y
contains the retained A target rows. The variance floor is fixed before
evaluation and its activation is reported. For method m,

    L_tm = mean_h [ mean_i (Y_D,ih - prediction_m,ih)^2 / v_th ].

Raw horizon MSEs and these normalized values are both retained. Normalization
scales are recomputed independently from the prepared A arrays and checked
against the manifest. All horizons receive equal weight; all six nonlinear
law/noise conditions receive equal weight within a repetition. The linear-control
aggregate averages its two noise levels within the repetition.

There are five primary useful-superiority contrasts. For each repetition r and
each comparator m in `common`, `rbf_departure`, `raw_rbf`, `raw_hgb`,
`nonkahm_policy`, define

    d_rm = .97 L_r,m - L_r,kahm_departure,

where L_r is the six-condition nonlinear aggregate. Thus a positive population
mean gap corresponds to at least a 3% improvement in the ratio of expected
normalized losses. The 3% margin is a chosen research-allocation threshold, not a
known application-derived minimum effect or universal definition of effectiveness.

The sixth primary contrast is the linear-control noninferiority comparison:

    d_r,linear = 1.05 L_r,ridge - L_r,kahm_departure.

The 5% tolerance is likewise a declared research screen. It is not a guarantee
that each individual realization or every linear process will avoid degradation.

For each of the six contrasts, calculate the mean paired gap, sample standard
deviation s, standard error s/sqrt(24), and one-sided lower confidence bound

    lower = mean(d) - t_(1-.05/6, df=23) s/sqrt(24).

All six lower bounds must be **strictly positive**. Bonferroni controls the
simultaneous one-sided family at alpha .05 under the paired-mean t assumptions.
This is an approximate finite-sample analysis relying on independent repetition
aggregates and suitably behaved contrast distributions; 24 is a fixed resource
budget, not a demonstrated power calculation. Dependence between methods,
conditions or noise levels within a repetition is retained by pairing. No
independence assumption is made about individual forecasting windows. This
family does not retroactively correct all previous development decisions or
provide a familywise claim across earlier studies.

Report the full 24 gap values per contrast, both component means, the point
relative gain, the chosen margin and the simultaneous lower bounds. Do not label
the corresponding symmetric intervals simultaneous 95% two-sided intervals.
Per-law, per-noise and per-horizon results, unbounded ablations and other method
comparisons are descriptive secondary evidence; no selected secondary finding
can reverse the primary decision.

The complete gate also requires finite forecasts and the documented
bounded-feature, coefficient and correction checks for `common`,
`kahm_departure` and `rbf_departure`. For the centered mean-SSE ridge solve,
the coefficient norm is at most s_h/(2 sqrt(alpha_h)), where s_h is the A
centered residual-response RMS. If the concatenated feature radius is R, then
the magnitude of the deployed correction is at most

    gamma_h * [abs(A residual mean_h)
               + (R + ||A feature mean||_2) * s_h/(2 sqrt(alpha_h))].

The analyzer verifies the reported fitted-state bound calculations and recomputes
the actual D correction as saved prediction minus saved ridge prediction. These
are implementation checks of a deterministic bound, not new prediction stress
experiments or empirical guarantees for arbitrary input distributions. In
addition, the following catastrophic-error engineering screen must pass in
**every** one of the 192 tasks:

    max_abs_error(kahm_departure) <= 5 * max_abs_error(ridge).

The maximum is over D origins and horizons. A zero ridge maximum requires a zero
candidate maximum. This factor is a deliberately coarse engineering screen and
is not calibrated to an application safety requirement. All triggered fallbacks
and diagnostics remain visible regardless of whether the screen passes.

Individual failed hyperparameter candidates are logged and the remaining
declared candidates remain eligible. No undeclared numerical jitter, extra
hyperparameter candidates or hidden refitting is permitted. Local denominator
fallback is part of the fixed reconstruction mechanism. A wholly failed method
or any final prediction-cell recovery to the baseline is instead a technical
block: failed comparators must not silently weaken an apparent superiority
claim. All original outputs and failure records are retained.

The only decisions are:

- **ADVANCE_TO_APPLICATION:** every statistical contrast and stability screen
  passes on the complete frozen roster. The evidence then warrants a separate,
  genuinely reserved application study; it does not yet justify broad efficacy
  or online-adaptation claims.
- **STOP_FORECASTING_PERFORMANCE_BRANCH:** the complete experiment does not pass
  every gate. A lower bound at or below zero includes unresolved precision as
  well as an unfavorable point effect. It does not prove equivalence or universal
  impossibility. No more methods, seeds, horizons or favorable subsets are added
  to this experiment to change the decision.
- **TECHNICAL_BLOCK:** required tasks, targets, forecasts, frozen input identities
  or essential diagnostics are incomplete or inconsistent. The missing or invalid
  roster is reported; no successful-subset efficacy aggregate is substituted.
  Any necessary repair must preserve the protocol, be documented, and rerun
  affected tasks from the unchanged inputs. A substantive post-D scientific
  change would be a new study, not a repair of this one.

Gamma activation is reported by condition and horizon; it is **not** required to
exceed an arbitrary prevalence. The five superiority conditions already prevent
credit for universal baseline recovery. Training, selection, prediction and
serialized-state costs are reported with their accounting boundaries. There is
no arbitrary runtime-ratio gate without a defined application requirement.

## Reproducibility and reporting contract

`prepared_inputs/manifest.json` identifies every task, split, random stream,
retained row and input checksum. Each task's result folder contains
`predictions.npz` with `Y` and all ten method arrays of shape (1664,3), a
`summary.json` with completed status, selections, diagnostics and resource
measurements, and sufficient fitted state for replay. Task inputs contain A_Y and
D_Y. Forecast targets must match the prepared D_Y byte-for-byte in value; saved
policy forecasts must match its stored C-selected arm. Machine environment and
source hashes accompany the results.

`analyze.py` imports no forecasting model. It recomputes all losses from saved
predictions and prepared targets, verifies the full roster and normalization,
and writes task/horizon/condition tables, independent-repetition aggregates, all
primary contrasts, stability and activation diagnostics, resource measurements
and the fixed decision. Hashes of analyzed inputs and generated outputs are
retained. An independent implementation can reproduce the six primary contrasts
using the published repetition table alone.

The final manuscript must explain the algebraic mechanism, state the scope of
its stability results, disclose all six comparisons and failures, and distinguish
new independent draws from new application evidence. A failed gate is a completed
research result. Neither a method roadmap nor an unsupported statement that
KAHM is an effective general forecasting solution is an acceptable substitute.
