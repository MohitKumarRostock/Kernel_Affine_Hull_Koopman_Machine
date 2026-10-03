# Confirmatory frozen-abstraction dynamical certificate protocol

Status: FROZEN DESIGN TO BE COMMITTED BEFORE CONFIRMATORY SAMPLING.

This protocol defines a fresh confirmatory dynamical study of the original
finite-sample i.i.d.-pair exclusion certificate. The design was selected only
after completion of the separately archived dynamical pilot. Pilot evaluation
pairs are planning data only and must not be reused as confirmatory evidence.

## Scientific question

For the prespecified Van der Pol KAHM abstraction described below, after
sharpening the association map with a training-selected fixed omega and
refitting the linear closure using training data only, does the original
finite-sample certificate give a non-vacuous lower bound on one-step vector
RMSE at a norm budget that contains the fitted predictor?

A positive certificate rules out every linear predictor satisfying the
corresponding spectral-norm budget. A zero certificate is a valid
inconclusive result and is not an execution failure.

## Selection provenance

The completed dynamical pilot used:

- Duffing with omega = 2;
- Van der Pol with omega = 4;
- nearest stored K-means center as the fixed hard reference-class map;
- genuinely independent current/successor evaluation pairs;
- sample sizes 128, 512, and 2048;
- eight replicate datasets per fixed system/mode/sample-size cell.

The pilot showed that the original frozen representations were structurally
vacuous at the fitted predictor norm: increasing evaluation sample size alone
could not make the asymptotic plug-in lower bound positive.

Post-pilot exploratory planning then varied only the association sharpness
parameter omega while reusing the already-frozen KAHM abstraction and
refitting the NLMS operator from the original training trajectories only.

For Van der Pol, omega = 128 produced a positive asymptotic plug-in lower
bound at the fitted-predictor norm in every retained M = 2048 planning
replicate in both evaluation modes. Plug-in finite-sample planning was
positive in every planning replicate at M = 4096 and had a substantially
larger lower-bound margin by M = 16384.

Duffing remained structurally vacuous throughout the exploratory high-omega
screen through omega = 2048. At high omega, exact-zero association
coordinates also began to appear in binary64 arithmetic. Duffing is therefore
outside the scope of this confirmatory campaign.

These exploratory calculations justify the prospective design below but are
not themselves confirmatory certificate evidence.

## Source frozen abstraction

The confirmatory study reuses the independently archived Van der Pol
state-regime abstraction from the completed frozen-representation build.

Source identities:

- representation build execution commit:
  `1add19b94372257d16118f60c69e1e8afa668259`;
- independent representation verifier commit:
  `65db148f865c94f988676381c6b5f12222356b31`;
- frozen-representation archive SHA-256:
  `04a5ad3d4dae3a8fafce2576512fdce06d6c738907ced7d3ad9fde600d8c0e2a`;
- source pilot configuration SHA-256:
  `3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1`.

The following fitted abstraction components are reused without refitting:

- state-space K-means centers;
- KAHM/OTFL autoencoder material;
- effective class count C = 25.

The state-space clustering and autoencoders must not be retrained or selected
using confirmatory evaluation data.

## Training-only association and operator refit

The confirmatory soft association map uses:

    omega = 128
    tau   = 1e-6

The same frozen abstraction is evaluated at this omega on the original
training snapshot pairs. The linear closure matrix B is then refit from
training data only.

Training settings remain:

- system: Van der Pol oscillator;
- dt: 0.02;
- mu: 1.0;
- training seeds: 0, 1, 2;
- 1200 transitions per training trajectory;
- three concatenated training trajectories;
- beta: 0.1;
- NLMS epochs: 20;
- shuffle: false;
- initial NLMS matrix: identity;
- fitted predictor orientation: `B.T @ Phi`;
- no stochastic projection of B.

For the fixed frozen abstraction,

    Phi_train = Phi_128(X_train)
    Chi_train = Phi_128(X_train_plus)

and B is learned by the existing manuscript NLMS recursion without using any
confirmatory evaluation pair.

The resulting abstraction-plus-omega-plus-B artifact must be constructed,
verified, and committed before confirmatory pair sampling begins.

## Fixed hard reference-class map

For every independently evaluated current state x,

    c(x) = argmin_c ||x - m_c||_2,

where m_c is the corresponding stored state-space K-means center from the
frozen source abstraction.

Implementation identifier:

    nearest_stored_kmeans_center_v1

Ties are resolved by the smallest zero-based class index.

The hard reference class is not defined by argmax Phi(x). The state-space
reference partition remains distinct from the soft association coordinates.

## Independent evaluation-pair law

Every retained pair must come from its own independent simulated trajectory
draw. Serially adjacent transitions from a common trajectory must not be
counted as independent observations.

For each retained pair:

1. independently draw an initial state from the benchmark initial-condition
   distribution;
2. draw T uniformly from {0, ..., 1199};
3. evolve the system for T steps;
4. call the resulting state X;
5. take exactly one additional RK4 step to obtain X_plus;
6. retain only (X, X_plus).

The implementation identifier remains:

    independent_uniform_time_pair_v1

## Evaluation modes

Two modes are prespecified and reported separately.

### matched_stochastic

Use the existing Van der Pol benchmark transition mechanism, including
additive Gaussian state noise with standard deviation 1e-5 per coordinate
after each RK4 step.

### deterministic

Use the same initial-condition law, vector field, dt, time-index law, and
frozen representation, but set transition process noise exactly to zero.

## Confirmatory design

System:

- Van der Pol only.

Evaluation modes:

- matched_stochastic;
- deterministic.

Independent-pair sample size per replicate:

    M = 16384

Replicate datasets per evaluation mode:

    32

Therefore the complete prespecified campaign contains:

    1 system
    x 2 evaluation modes
    x 1 sample size
    x 32 replicates
    = 64 replicate datasets
    = 1,048,576 retained independent pairs.

All 64 scheduled attempts must be represented in the retained execution
record. Failed attempts must not be silently replaced.

## Certificate statistics

For each replicate, with M independent evaluation pairs,

    phi_i = Phi_128(X_i)
    Z_i   = Phi_128(X_i_plus)

and fixed current-state hard labels c_i = c(X_i), compute

    f_hat = (1/M) sum_i (1 - phi_i[c_i])^2

and

    s_hat = (1/M) sum_i
            ||Z_i - mean(Z_j : c_j = c_i)||_2^2.

Empty sampled classes contribute zero to the empirical within-class residual
sum exactly as in the validated original statistics implementation.

With delta = 0.05,

    r_delta = (log(2) - log(delta)) / M

    F_delta = min(
        1,
        f_hat + r_delta
        + sqrt(r_delta^2 + 2 r_delta f_hat)
    )

    V_delta = max(
        0,
        s_hat - sqrt(2 r_delta)
    )

and for every prespecified spectral-norm budget kappa,

    L_kappa_delta = max(
        0,
        sqrt(V_delta) - kappa sqrt(2 F_delta)
    ).

## Spectral-norm budgets

After the training-only B refit is frozen, evaluate the unique sorted set

    {1, sqrt(2), ||B||_2, 2 ||B||_2}.

The fitted-predictor feasibility interpretation must always use the actual
frozen spectral norm ||B||_2. A bound at kappa < ||B||_2 does not apply to
the fitted B.

Norm budgets are determined before confirmatory evaluation outcomes are
available.

## Fresh RNG namespace

Confirmatory evaluation pairs must use a seed namespace disjoint from the
completed pilot.

The seed construction remains based on:

- `numpy.random.SeedSequence`;
- PCG64;
- root seed 20261003;
- full immutable seed material including campaign, system, mode, sample
  size, replicate, pair, stream, and root components.

The completed pilot used campaign key 3.

The confirmatory campaign uses:

    campaign key = 4

Van der Pol system key remains:

    system key = 2

Mode keys remain:

    matched_stochastic = 1
    deterministic      = 2

Stream keys remain:

    trajectory_seed = 1
    time_seed       = 2

Before execution, the complete confirmatory seed schedule must be checked for
uniqueness and for disjointness from the completed pilot and the original
training seeds.

## Retained evidence

Retain enough data to reproduce every reported value, including at minimum:

- complete scheduled-attempt metadata;
- complete RNG/seed material;
- retained current and successor physical states;
- retained time indices;
- Phi_128(X);
- Phi_128(X_plus);
- fixed hard-class labels;
- selected-center squared distances;
- class counts;
- f_hat;
- s_hat;
- F_delta;
- V_delta;
- every prespecified L_kappa_delta;
- each kappa;
- frozen refitted-B spectral norm;
- frozen refitted-B one-step vector MSE and RMSE;
- simplex diagnostics;
- state-range diagnostics;
- source/configuration/artifact identities;
- environment and dependency provenance;
- execution runtime.

The fitted predictor error is an attainable error for that fixed fitted
predictor. It is not the exact constrained optimum and must not be reported
as such.

## Reporting

Report matched-stochastic and deterministic modes separately.

For each mode, report at minimum:

- certificate q05, median, and q95 across the 32 independent replicate
  datasets;
- zero-certificate frequency;
- F_delta and V_delta summaries;
- frozen-predictor RMSE summary;
- signed gap between frozen-predictor RMSE and the lower certificate;
- class occupancy diagnostics;
- runtime diagnostics.

Replicate quantiles are descriptive statistics, not confidence intervals.

Any reported interval for a replicate-level probability must clearly state
whether it is pointwise or simultaneous and which replicate datasets are the
Bernoulli trials.

The lower-bound/predictor-error gap must not be interpreted as exact
certificate tightness because the constrained optimal dynamical prediction
risk is unknown.

## Confirmatory safeguards

- The omega value, system scope, sample size, replicate count, evaluation
  modes, norm-budget rule, reference-class map, pair law, and RNG namespace
  are frozen before confirmatory sampling.
- Pilot evaluation pairs must not appear in confirmatory certificate
  estimates.
- Confirmatory evaluation pairs must not tune omega, tau, C, training seeds,
  clustering, autoencoders, NLMS settings, norm budgets, or sample size.
- The training-only refitted representation must be frozen and independently
  verified before evaluation-pair generation.
- The complete 64-attempt schedule must be persisted before sampling.
- Sample evidence should be persisted before certificate evaluation.
- Sampling or evaluation failure must not trigger a replacement seed.
- Every scheduled failure remains part of the execution record.
- A zero certificate remains a valid confirmatory outcome.
- All confirmatory numerical results must undergo an independent audit from
  retained raw pairs and the committed frozen representation.
- The final evidence package must be archived with checksums.
- Execution-source, verification-source, and later archival commits must
  retain their distinct provenance roles.

## Relationship to the controlled study

The controlled four-state campaign remains the validation study for exact
coverage, exclusion power, and tightness against a known optimal population
risk.

This dynamical confirmatory study answers a different question: whether the
original certificate can become non-vacuous on a real frozen KAHM
representation under a fully prospective independent-pair evaluation
protocol after the representation design has been fixed.
