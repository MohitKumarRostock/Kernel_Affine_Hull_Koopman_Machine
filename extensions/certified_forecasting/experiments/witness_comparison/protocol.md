# Prespecified finite-support witness comparison

## Scope and freeze

This protocol is written before drawing any benchmark calibration sample. It
compares lower confidence bounds for the same minimum population squared error
over all scalar 1-Lipschitz functions of a one-dimensional representation,
clipped to [-1,1]. It is a diagnostic study of certificate tightness and cost,
not a fitted KAHM accuracy comparison. The finite joint support is known to all
methods; its probabilities are used only by the simulator and population
oracle, except for an explicitly labelled secondary known-mass method.

The primary run uses NumPy SeedSequence entropy 202610290731, law indices in
the order below, calibration sizes 64, 256, 1024, 4096, and 300 independent
repetitions per law and size. A separate child stream identified by
[entropy, law_index, sample_size, repetition] generates each sample. Samples
are independent across sizes, rather than nested. Every scheduled trial,
solver diagnostic and numerical failure is retained. No law, threshold,
method, or repetition is selected by the resulting performance. The protocol,
source files and configuration are hashed into the run manifest before the
first sample is drawn. Output directories must not already exist.

## Eight joint laws

All laws use metric |u-u'| and sensitivity budget kappa=1. Entries below are
(u,y,probability). Repeated u values represent target variation that the
representation cannot distinguish. A stratum is a fixed function of u.

1. `collapsed_binary`: (0,-1,1/2), (0,1,1/2); one stratum.
2. `two_atom_partial`: (-0.6/sqrt(2),-1,1/2),
   (0.6/sqrt(2),1,1/2); one stratum. This is an isometric scalar version of
   the earlier two-coordinate simplex example at separation a=0.6.
3. `three_point_v`: (-1,1,1/3), (0,-1,1/3), (1,1,1/3);
   strata {-1,0} and {1}.
4. `five_point_zigzag`: u=(-1,-1/2,0,1/2,1),
   y=(-1,1,-1,1,-1), all probabilities 1/5; strata {-1,-1/2}
   and {0,1/2,1}.
5. `partial_collapse`: (-1,-1), (-1,0.2), (0,-0.2), (0,0.2),
   (1,-0.2), (1,1), all probabilities 1/6; one stratum per distinct u.
6. `noisy_affine`: (-1,-1), (-1,0), (0,-0.5), (0,0.5),
   (1,0), (1,1), all probabilities 1/6; one stratum per distinct u.
   The conditional mean is u/2 and independent sign noise has magnitude 1/2.
7. `rare_conflict`: (3,0,0.9), (0,-1,0.05), (1,1,0.05);
   strata {3} and {0,1}. The conflict is rare under unconditional pairing.
8. `affine_mismatch`: (-1,-0.5,1/3), (0,0.5,1/3),
   (1,-0.5,1/3); one stratum. A 1-Lipschitz interpolant has zero risk,
   although no affine predictor interpolates all three target values.

The fixed RMSE tolerances are 0.10, 0.25 and 0.50 for every law. We report
exclusion power only when the oracle minimum is strictly greater than the
squared tolerance; otherwise we report false-exclusion frequency. The
thresholds therefore test positive risk requirements on the shared [-1,1]
target scale, including thresholds on both sides of the population optimum.
They are not thresholds chosen to make a method succeed.

## Confidence procedures

Every method receives its own one-sided failure probability eta=0.05. Coverage
is evaluated separately for each method. There is no claim of a joint 95%
event across methods, laws, sample sizes or Monte Carlo repetitions. All
methods use the same iid sample within a trial. No prediction fitting data
are required; empirical-risk minimization below is a certificate calculation.

The four primary comparisons are:

* `pair_eb`: the original disjoint-pair witness with empirical Bernstein
  correction and the generic score range [0,2].
* `centered_range`: the centered population bound
  (sd(Y)-kappa sd(U))_+^2. For nonconstant U, allocate eta/3 to a direct
  target-SD lower bound, a target-variance lower bound, and an input-SD
  upper bound. The SD bounds use Maurer and Pontil (2009), Theorem 10,
  scaled by the known coordinate range. The target-variance bound uses
  bounded differences for the variance with denominator n. Take the maximum
  of the two valid target-SD lower bounds and cap the input-SD upper bound
  by half its known support range. If U is structurally constant, its SD
  is exactly zero and the two target bounds each receive eta/2.
* `erm_kl`: minimize empirical squared loss over the same full scalar
  Lipschitz class. Use a dual-feasible lower objective d_emp, not a raw
  primal solver objective. Put q=max(0,d_emp)/4. The lower endpoint is 4p,
  where p<=q solves n*kl(q||p)=log(1/eta); p=0 when q=0. This is valid
  because the empirical minimum is at most the empirical loss of a fixed
  population minimizer, and the bounded-loss Chernoff bound applies to that
  fixed function. Its target is the class minimum; it does not supply an
  upper-risk or excess-risk guarantee for the empirical fitted model.
* `stratified_eb`: within each fixed U-based stratum, form disjoint pairs
  in sample order and apply the same generic [0,2] score range. For S
  strata, allocate eta/(2S) to each conditional witness lower bound and
  eta/2 to simultaneous mass intervals. If fewer than two pairs are
  available, use lower endpoint zero. The mass radius is
  sqrt(log(4S/eta)/(2n)). Minimize the weighted conditional lower endpoints
  over the probability simplex intersected with those clipped mass intervals.
  The method does not receive the true stratum probabilities.

Prespecified secondary comparisons are `pair_support_eb` and
`stratified_support_eb`, replacing the generic witness range by its exact
maximum over the known support; `stratified_knownmass_support_eb`, which
additionally uses true stratum masses and allocates eta/S to the conditional
scores; and `erm_hoeffding`, which subtracts
4*sqrt(log(1/eta)/(2n)) from the same certified empirical minimum.
These diagnose which gains depend on support information or mass information.
They are retained for every trial, not substituted after seeing outcomes.

For the historical collapsed two-atom setting n=4096, eta=0.05/3, we also
reproduce the analytical calculation
1-1/n-4*sqrt(log(2/eta)/(2n))=0.903057515... . This is a lower bound on the
expected original centered endpoint after using the known zero input
variance. It is not a Monte Carlo mean or the new direct-SD bound.

## Oracle and numerical safeguards

For each law, aggregate repeated representation values. The scalar global
Lipschitz class restricted to ordered support points has adjacent constraints
|f[j+1]-f[j]|<=u[j+1]-u[j] and box constraints |f[j]|<=1. These constraints
are sufficient for a global 1-Lipschitz extension, followed by clipping.
Population and empirical objectives are convex quadratic programs. Empirical
support values absent from a sample may be omitted: Lipschitz extension fills
those values without increasing empirical risk.

A feasible primal vector supplies an upper risk bound. Nonnegative Lagrange
multipliers and an analytically minimized quadratic Lagrangian supply a lower
bound. The code reports both, their gap and feasibility residual. It applies
a downward floating-point guard to every dual objective and an upward guard
to every population upper objective. Required population/empirical brackets
are <=1e-8; failures trigger a documented deterministic fallback, and unresolved
failures abort the run rather than silently omit observations. These are
checked floating-point brackets, not formal interval-arithmetic proofs.
Analytically solvable laws and independent feasible-grid checks test the
implementation before the benchmark. An affine-only minimum is reported as
a separate diagnostic and is never substituted for the Lipschitz oracle.

## Reported quantities and retained evidence

Report each law's population optimum bracket, unconditional witness,
stratified witness, centered population bound and affine-only diagnostic.
This separates population slack from finite-sample estimation. For every
method/law/size report lower-endpoint mean, median and quantiles; true-risk
coverage and Monte Carlo binomial uncertainty; exclusion counts at all three
positive tolerances; mean signed difference from the corresponding population
bound; and certificate-only timing. Mean signed differences may be negative
in individual draws because an empirical endpoint can exceed its population
witness while remaining below the population optimum.

Timing begins after sample generation and representation lookup. It includes
each certificate's statistics, confidence correction and, for ERM, numerical
optimization. Record each timing and summarize the median; population oracle,
feature learning, data generation and plotting are excluded. Shared numerical
work is timed once and identified, not credited as an unfair zero-cost ERM
comparator. Timings are descriptive measurements on the recorded machine.

Retain all joint-support counts, unconditional pair-count matrices,
within-stratum pair-count matrices, solver certificates, method endpoints,
and timing records in compressed JSON Lines. These sufficient statistics
reconstruct every reported endpoint without retaining millions of repeated
atom identifiers. The independent per-trial random seed reconstructs the
ordered samples if needed. Save a CSV summary, full population diagnostics,
source/environment manifests, aggregate hashes, and a machine-checkable audit.
The analyzer generates all eight sample-size panels and compact tables; it
must not hide unfavorable laws. No model or full forecasting campaign is rerun.
