# Frozen-representation dynamical certificate protocol

Status: DRAFT PROTOCOL FROZEN BEFORE CERTIFICATE EVALUATION.

This protocol defines the next certificate-validation study on the Duffing
and Van der Pol benchmarks. No certificate result may be used to select the
representation, class map, evaluation distribution, norm-budget rule, or
pilot sample-size grid described here.

## Scientific question

For a KAHM representation selected previously for forecasting diagnostics,
does the original finite-sample exclusion certificate give a useful lower
bound on one-step vector RMSE when evaluated on genuinely independent
current/successor pairs?

The experiment separates:

1. representation construction;
2. independent certificate evaluation;
3. fitted-predictor performance.

A positive certificate rules out every linear predictor satisfying the
specified spectral-norm budget. A zero certificate is inconclusive.

## Frozen representations

The representation settings are inherited from the retained tuned external
baseline comparison and are not selected using certificate outcomes.

### Duffing

- system: unforced damped Duffing oscillator
- dt: 0.03
- training seeds: 0, 1, 2
- training steps per trajectory: 1200
- C: 10
- omega: 2.0
- tau: 1e-6
- subspace_dim: 4
- Nb: 100
- beta: 0.1
- NLMS epochs: 20
- K-means kind: full
- maximum training points per cluster: none
- representation random state: 0
- fitted closure is not projected to a stochastic matrix

### Van der Pol

- system: Van der Pol oscillator
- mu: 1.0
- dt: 0.02
- training seeds: 0, 1, 2
- training steps per trajectory: 1200
- C: 25
- omega: 4.0
- tau: 1e-6
- subspace_dim: 4
- Nb: 100
- beta: 0.1
- NLMS epochs: 20
- K-means kind: full
- maximum training points per cluster: none
- representation random state: 0
- fitted closure is not projected to a stochastic matrix

Training uses the existing Experiment 15 trajectory generators and
`fit_kahkm` implementation. The fitted abstraction is frozen before any
certificate-evaluation pairs are generated.

The final execution must retain enough fitted-model material and provenance
to reproduce Phi(x), the stored state-space cluster centers, and the learned
NLMS matrix B.

## Fixed hard reference-class map

For any independently evaluated current state x,

    c(x) = argmin_c ||x - m_c||_2,

where m_c is the corresponding stored K-means state-space cluster center.

Implementation identifier:

    nearest_stored_kmeans_center_v1

Ties are resolved by the smallest zero-based class index.

The hard class is NOT defined as argmax Phi(x). The hard reference partition
and the soft KAHM coordinate map are deliberately kept distinct.

## Independent evaluation-pair law

The certificate study must not count serially adjacent transitions from one
trajectory as independent observations.

Each evaluation pair is generated from its own independent draw under the
following law.

For each pair:

1. draw an initial state from the same initial-condition distribution used
   by the benchmark simulator;
2. draw T uniformly from {0, ..., 1199};
3. evolve the system from the initial state for T steps;
4. call the state after those T steps X;
5. take exactly one additional RK4 step to obtain X_plus;
6. retain only the pair (X, X_plus).

Thus the current-state distribution is a uniform-in-time mixture over the
same 1200-step horizon used to construct each training trajectory, while
different retained pairs do not share a simulated trajectory.

No evaluation seed may overlap any seed material used in representation
training.

## Two evaluation dynamics

Two prespecified evaluation modes are required.

### matched_stochastic

Use the process-noise convention of the existing benchmark generators:

- Duffing: additive Gaussian state noise with standard deviation 1e-4 per
  coordinate after each RK4 step;
- Van der Pol: additive Gaussian state noise with standard deviation 1e-5
  per coordinate after each RK4 step.

This evaluates the certificate under the same stochastic transition mechanism
used by the existing benchmark simulations.

### deterministic

Use the same initial-condition distribution, dt, vector field, uniform time
index, and frozen representation, but set transition process noise exactly
to zero.

This mode helps distinguish representation/dynamics ambiguity from
irreducible successor randomness.

The deterministic mode is a separate evaluation law and must be reported
separately rather than pooled with the stochastic mode.

## Soft current and successor coordinates

For every retained independent pair:

    phi_i = Phi(X_i)
    Z_i   = Phi(X_i_plus),

using the frozen KAHM abstraction and its frozen omega and tau.

The existing `kahm_associations` implementation is used without refitting.

## Original certificate statistics

For M independent evaluation pairs, retain and compute:

    f_hat = (1/M) sum_i (1 - phi_i[c_i])^2

and

    s_hat = (1/M) sum_i
            ||Z_i - mean(Z_j : c_j = c_i)||_2^2.

Empty sampled classes contribute zero to the empirical within-class residual
sum, exactly as in the validated statistics implementation.

The original finite-sample certificate is then evaluated without fitting a
new predictor to the evaluation data.

## Spectral-norm budgets

Norm-budget choices are determined only from the frozen training fit and fixed
constants, never from certificate-evaluation outcomes.

For each frozen representation, evaluate the unique sorted set

    {1, sqrt(2), ||B_NLMS||_2, 2 ||B_NLMS||_2}.

`||B_NLMS||_2` is computed from the matrix fitted using training data only.

The event underlying the original certificate does not depend on kappa, so
these budgets reuse the same independently evaluated statistics. All reported
bounds remain explicitly associated with their kappa value.

## Pilot design

Before any larger dynamical campaign, run a separately labeled pilot.

Systems:

- Duffing
- Van der Pol

Evaluation modes:

- matched_stochastic
- deterministic

Independent-pair sample sizes:

- 128
- 512
- 2048

Replicate datasets per fixed system/mode/sample-size cell:

- 8

The pilot is for detecting implementation problems, assessing computational
cost, and determining whether the original certificate is numerically
non-vacuous. It is not final manuscript evidence.

The pilot may not be silently promoted to a final campaign.

## Quantities retained per replicate

Retain at minimum:

- system;
- evaluation mode;
- replicate identifier;
- sample size M;
- complete RNG/seed metadata;
- state-pair sufficient data or a losslessly reproducible representation;
- hard-class counts;
- selected-center squared-distance diagnostics;
- f_hat;
- s_hat;
- F_delta;
- V_delta;
- every L_kappa_delta;
- each kappa;
- frozen NLMS spectral norm;
- actual frozen-NLMS one-step vector RMSE on the same pairs;
- actual frozen-NLMS one-step squared error;
- simplex-coordinate range and row/column sum diagnostics;
- current-state and successor-state coordinate ranges;
- execution source commit;
- frozen-model identifier and hashes;
- configuration hash;
- dependency/environment record.

The NLMS evaluation error is an attainable error for that fitted predictor.
It is not the exact constrained optimum and must not be reported as such.

## Pilot reporting

For each fixed system, evaluation mode, sample size, and kappa, report:

- certificate median and descriptive 5th/95th percentiles;
- zero-certificate frequency;
- F_delta and V_delta summaries;
- gap between the frozen predictor RMSE and the lower certificate;
- class occupancy diagnostics;
- evaluation runtime.

Do not interpret the lower/upper gap as exact certificate tightness because
the constrained optimal risk is unknown on these dynamical benchmarks.

The controlled four-state study remains the experiment for exact coverage,
power, and tightness against known optimal population risk.

## Independence and selection safeguards

- Representation fitting and all representation choices precede certificate
  evaluation.
- Evaluation pairs are not used to tune C, omega, tau, training seeds,
  clustering, or KAHM construction.
- Each retained pair is an independent unit generated from its own independent
  simulated trajectory draw.
- No transition sequence from a single trajectory is treated as M independent
  pairs.
- Pilot outcomes must not be used to retrospectively alter the pilot
  configuration.
- Any later final dynamical campaign must be frozen in a new configuration
  after the pilot and before final evaluation.
- Multiple representations must not be searched and then reported using the
  same certificate-evaluation data without a separately justified selection
  correction or fresh final evaluation data.

## Reproducibility

The execution workflow must follow the repository's established pattern:

1. commit code, tests, and pilot configuration before execution;
2. require a clean execution source commit;
3. save the complete schedule before sampling;
4. preserve every scheduled attempt and failure;
5. save generated evaluation data before certificate evaluation where
   practical;
6. never silently retry with replacement seed material;
7. retain raw sufficient statistics and outputs;
8. independently verify archived numerical results;
9. archive evidence with checksums;
10. generate manuscript tables and figures only from retained evidence.

Historical experiment outputs are not modified by this study.
