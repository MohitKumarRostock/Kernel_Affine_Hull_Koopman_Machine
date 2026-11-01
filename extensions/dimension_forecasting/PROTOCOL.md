# Essential forecasting rerun after correction of the KAHM dimension policy

## Scope and source

This study repeats the principal ten-procedure forecasting comparison after
correcting the local KAHM dimension policy. It does not revise manuscript prose,
evaluate risk certificates, or introduce natural-data claims. The observation
laws and numerical forecasting primitives are preserved from repository commit
`7dda356138675934aab5e327ec667bde0a5eda79`. The corrected construction uses the
verified Step 1 policy with SHA-256
`c03d9a1923b5d9c8ade8a4198fd06e2b24c6add9bf0e0992c469f90d7621efa2`.
The policy requests `min(cap, ambient dimension, local reference count - 1)`
and retains the native positive-eigenvalue and projected-range rules. The
historical comparator explicitly uses cap 2 and the former normal-departure
restriction. Requested and realized local dimensions are recorded separately.

## Episodes and independent units

The design has 24 fresh paired repetitions, four original laws (`linear_ar`,
`nonlinear_ar`, `vanderpol`, `duffing`), and added-noise levels 0 and 0.1, giving
192 tasks. History length is 12, episode length 72, horizons are 1, 4, and 12,
and stride is 4. There are 64/32/32/128 independent A/B/C/D episodes, with
512/256/256/all 1664 retained rows. The 13 origins per episode are dependent;
all primary uncertainty calculations therefore use repetition aggregates.

The original generators, initial-condition laws, integrations, burns and
observation maps are unchanged. Noise is `eta * s_A * Z`, where `s_A` is the
population standard deviation of all unaugmented A episode values. The same
base episodes, unit-noise arrays, retained rows and fitting seed are paired
across noise levels and candidate dimensions. Both histories and targets are
augmented. Scaling and normalization are fitted on A only.

Master seed 2026110103 uses the exact seed-derivation contract in `config.json`.
All 1272 intended-distinct derived uint32 seeds are checked for collisions
before generation, including against 2173 distinct seeds from the original
principal experiment (including development) and Step 2. Prior seeds and their
derivations are retained in `readiness/prior_seed_exclusions.json`. No prior
fitted state, calibration episode, or terminal forecast is reused.

## Procedures and B-only dimension selection

The complete original method roster is ridge, raw Nyström RBF, histogram
gradient boosting, persistence, common geometry, bounded KAHM departure,
bounded RBF departure, unbounded KAHM departure, unbounded RBF departure, and
the C-selected non-KAHM policy. All original hyperparameter grids remain
unchanged. Raw competitors and ridge are fitted once per task because their
definitions do not depend on the local KAHM cap.

Seven banks are fitted independently: historical cap 2 with the explicit
legacy restriction, and native caps 1, 2, 4, 8, 12, and 20. They use the same
A scaling, partitions and ordered reference samples. Every bank fits all five
geometry heads. Ridge penalties are selected per horizon on B using the
original mean row squared-error objective. Each geometry procedure then
selects one native cap across all horizons by minimizing the mean of its
uncalibrated B MSE divided by the corresponding A target population variance,
floored at 1e-12. Exact ties choose the first cap in ascending order. All five
choices and the complete B trial records are durably saved before C arrays
are opened for calibration.

The original raw RBF grid uses 64 A landmarks, bandwidth factors 0.5, 1, and 2,
and five ridge penalties. The HGB grid uses learning rates 0.05 and 0.1, maximum
leaf counts 7 and 15, 100 iterations, L2 penalty 1, and no early stopping. Raw
RBF bandwidth/penalty and HGB configuration are chosen per horizon on B.
Persistence is fixed. The shared raw fitting helpers are unchanged; their
query argument is B during this stage and cannot expose C to model selection.

## C calibration and comparison contexts

C selects a blend multiplier from 0, 0.25, 0.5, and 1 per horizon for every
nonridge procedure. A zero multiplier returns the same shared ridge forecast
exactly. Three complete comparison contexts are retained:

1. `selected`: each of the five geometry procedures uses its own B-selected
   native cap. This is the sole primary context.
2. `historical2`: all five geometry procedures use the historical cap-2 bank.
3. `native20`: all five geometry procedures use the corrected cap-20 bank.

Within each context, the original non-KAHM policy selects on C among ridge,
raw RBF, HGB, persistence, common geometry, bounded RBF and unbounded RBF.
Its score is the mean horizon C MSE divided by A target variance. It selects a
complete procedure after its cap is frozen on B; C never receives a menu of
caps for the same procedure. Ties use the original seven-method order.

Secondary matched controls use the bounded KAHM procedure's selected bank for
all five geometry heads. These controls support within-bank mechanism
comparisons and are distinct from independently selected competitors. A
comparison of independently selected KAHM and RBF procedures is a comparison
of complete tuning procedures and does not isolate reconstruction geometry.

Both calibrated and gamma-one forecasts are retained. In a gamma-one context,
the policy forwards the gamma-one forecast of its already C-selected method;
the policy is not selected again under the ablation.

## Frozen stages and numerical integrity

Manufactured-input parity and independent preflight must pass before source,
configuration and analysis are frozen. ABC episodes are generated only after
that freeze. All models, B selections, C multipliers and C policy choices are
globally sealed before any D episode is generated. D query archives contain
histories only and are distinct from reserved target archives. Prediction
APIs receive histories only, and fitted-state hashes must be unchanged after
querying. All forecasts are sealed before any terminal target is opened for
scoring. Later D histories may contain earlier within-episode target values;
queries perform no cross-row fitting, statistics or selection.

The inherited local rank-zero and guarded reconstruction rules remain
explicitly logged. Unexpected bank-factor, fitting, candidate-grid or query
errors stop the study; tasks or candidates are not silently removed. Every
task verifies exact historical2/native2 numerical identity, native12/native20
identity at ambient dimension 12, and identical reference sampling across
caps. Metadata intentionally distinguishes the dimension policies even when
the numerical maps coincide. The cap-20 candidate cannot win an exact tie
with cap 12 in the selected context.

## Primary claim family and descriptive outcomes

The primary family restores the original six simultaneous claims for the
selected, calibrated bounded KAHM procedure: five nonlinear 3% superiority
comparisons against common geometry, bounded RBF, raw RBF, HGB and the
non-KAHM policy, plus 5% noninferiority to ridge on the linear negative
control. Equal horizon, noise and law averages are computed within each
independent repetition. For a nonlinear comparator, the gap is
`0.97 * comparator SMSE - KAHM SMSE`; the linear gap is
`1.05 * ridge SMSE - KAHM SMSE`. Each lower bound is the sample mean gap minus
`t.ppf(1 - 0.05/6, 23) * sample_sd / sqrt(24)`. A claim passes only when its
lower bound is positive; the complete original gate requires all six.

SMSE divides each horizon MSE by the population variance of retained A
targets, floored at 1e-12. Historical2/native20 contexts, gamma-one results,
matched controls, individual laws/noise/horizons and additional pairwise
contrasts are secondary. Their descriptive intervals do not expand the
primary claim family. The report also retains original maximum-error-ratio
screens, calibration activation, bounded-feature/readout checks, all guard
counts, effective ranks, reference identities, selection frequencies and
failure records. Neither local rank 2 nor a selected cap is interpreted as
the global stacked feature rank.

## Resource interpretation

One thread is used per numerical worker, with at most three workers on
distinct CPUs. Timings are descriptive shared-host measurements, not
isolated speed benchmarks. Shared raw competitors are fitted once. Selected
geometry costs include all six native candidate banks and the method's
heads, B selection and selected C calibration; policy construction is charged
the union of required nodes without double-counting shared work. Both joint
study-bundle costs and source-defined per-procedure retained-state sizes are
reported. Serialized study checkpoints include tuning evidence and are not
called deployment state or peak memory.
