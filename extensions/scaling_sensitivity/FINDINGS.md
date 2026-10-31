# Geometry-scaling sensitivity: findings and implications

## Main finding

Training-only, unclipped min-max geometry scaling changes some KAHM errors
and a secondary controlled ranking, but **does not reverse the integrated
paper's main natural or KAHM-local controlled comparisons**. The natural
raw-plus-KAHM aggregate is nearly unchanged; the largest KAHM-local controlled
improvement occurs on Duffing, where simpler predictors still have lower error.

These are descriptive paired results on already inspected sources and draws.
They are not a fresh source-confirmation study, a statistical equivalence test,
or evidence that KAHM is invariant to scaling.

## What was run

The fixed protocol and execution code were published before the new fits at
`c5b8817a9157de685fd1c70a22d83542a7bdf766`. The intervention replaced the scaler
inside the geometry pipeline, where it affects partitions, local directions,
reconstruction and gates. The standardized raw-input skip/reference, targets,
splits, seeds, decoder budgets, query conventions, and validation grids were
preserved. Fitting statistics came only from the appropriate training set;
future queries were not clipped.

- **Natural:** all 208 original configurations, three raw-augmented geometric
  methods under both scalers, and two fixed raw references: **1,664 standalone
  method runs**.
- **Controlled:** 24 complete draws per law, three laws, 64/256 landmarks,
  both scalers, and five representations: **72 paired jobs / 1,440 head fits**.
  The locality-coordinate representation is an explicitly added matched
  control. The other four representations were in the original capacity study.

No scaler was selected from test outcomes. The original benefit diagnostic
was not refitted or reevaluated for min-max geometry. Both phases completed
without failures, serially with one benchmark worker and one BLAS thread.

## Natural results

The following table uses the original equal-source hierarchy on the **four
formerly reserved source families**, covering 80 configurations. Lower scaled
MSE is better. Raw references retain their original standardization.

| Method | Standard geometry | Min-max geometry | Min-max versus standard |
|---|---:|---:|---:|
| Raw ridge, fixed reference | 0.7798112 | 0.7798112 | Fixed |
| Raw RBF, fixed reference | 0.8075209 | 0.8075209 | Fixed |
| Raw + affine | 0.8458120 | 0.8595516 | +1.6244% |
| Raw + locality | 0.8461821 | 0.8487775 | +0.3067% |
| Raw + KAHM | 0.8419228 | 0.8421913 | **+0.0319%** |

Min-max raw-plus-KAHM remains **4.2934% above raw RBF** and **7.9994% above
raw ridge**. Raw ridge has lower loss on all four source means. KAHM beats raw
RBF only on Saugeen, and leaving out any one family still leaves the min-max
KAHM aggregate above raw RBF.

| Source | KAHM min-max versus KAHM standard |
|---|---:|
| Traffic | −0.15% |
| Hospital | −0.86% |
| Saugeen | +0.90% |
| US births | +0.35% |

The small-capacity aggregate worsens by **0.99%**, while the medium-capacity
aggregate improves by **0.96%**. Those settings jointly vary sample and model
size caps, so the difference does not isolate either factor.

The original development pool remains separate: KAHM changes from
**2.1768361 to 2.3443567**, an increase of **7.6956%**. The source effects are
Electricity −1.99%, NN5 +2.06%, Tourism −5.20%, and FRED-MD +19.54%.
Excluding FRED-MD changes the aggregate scaler effect to a 3.62% reduction.
This heterogeneity is retained rather than used to select favorable sources.
Raw ridge remains the lowest-error development aggregate; min-max affine
also remains substantially better than KAHM there.

## Controlled results

Errors below are multiplied by **1,000**. Every cell averages the same 24
complete data/fit draws. The raw reference remains standardized in both arms.

| Law | Landmarks | Raw RBF | KAHM local, standard | KAHM local, min-max | KAHM change |
|---|---:|---:|---:|---:|---:|
| Nonlinear AR | 64 | 4.4238 | 4.8795 | 4.8678 | −0.24% |
| Nonlinear AR | 256 | 4.6958 | 5.1749 | 5.1584 | −0.32% |
| Van der Pol | 64 | 2.8586 | 1.0266 | 1.0248 | −0.18% |
| Van der Pol | 256 | 1.6541 | 0.8720 | 0.8740 | +0.23% |
| Duffing | 64 | 3.4597 | 4.2154 | 4.0951 | −2.85% |
| Duffing | 256 | 2.8364 | 4.0736 | 3.7741 | **−7.35%** |

Raw RBF retains lower mean loss than KAHM local on autoregression and Duffing.
KAHM local remains better than raw, affine-local, and locality-local on Van der
Pol at both capacities. On Duffing at 256 landmarks, even the improved
min-max KAHM error **3.7741** remains above raw **2.8364**, min-max affine
**2.7742**, and min-max locality **2.6921**.

**Not every secondary ranking is stable.** KAHM's score-only representation
worsens by 6.03% and 4.78% on Van der Pol at 64/256 landmarks. At 256 landmarks,
standardized KAHM scores (**1.0499**) beat standardized affine-local
(**1.0780**), whereas min-max KAHM scores (**1.1001**) trail min-max
affine-local (**1.0812**), in the same units. The paper explicitly distinguishes
this reversal from the main KAHM-local comparisons.

## Costs and input-range diagnostics

The new source-balanced mean warm method times on the formerly reserved
natural sources are:

| Method | Seconds per task |
|---|---:|
| Raw RBF | 0.176205 |
| Raw ridge | 0.186285 |
| Raw + KAHM, standard geometry | 0.380373 |
| Raw + KAHM, min-max geometry | 0.556100 |

The min-max KAHM time is **3.1560 times** the contemporaneous raw RBF time
and **1.4620 times** its standard-geometry counterpart. These measured totals
cover search, selection cleanup, selected refit, forecast and serialization.
Data loading, launch-to-result time and extra audit work are separately
recorded; the source preparation is reused. Launch-to-result is not isolated
startup overhead, and the inherited import timer is not a complete cold-start
measurement.

All timing observations are retained, including the 6.2914-second Saugeen
min-max KAHM configuration. A single execution does not identify an intrinsic
min-max computational penalty or separate algorithmic changes from runtime
variation. New times are compared only with references measured in this run.

On the formerly reserved sources, 3.77% of test input coordinates, or 33.08%
of test histories, exceed the final fitting range under the original hierarchy.
The largest absolute min-max coordinate is **3.1990**. The corresponding
development figures are 21.45%, 49.26%, and **1.8638**. There are no constant
input coordinates in the final natural refits. These diagnostics confirm that
future observations outside the fitted range were not clipped by the transformation.

## Verification and interpretation

- All **1,040** standard natural/raw-reference prediction arrays are bit-for-bit
  equal to their archived versions, with identical selected alpha values.
- All **1,664** new natural losses and scaler states were reconstructed by
  separate analysis code.
- All **1,440** controlled prediction arrays were checked against episode losses
  before compact export; **576** historical standard method comparisons and
  **144** raw cross-arm comparisons passed.
- The complete original scientific checkpoints and all **609** dependencies in
  the new execution freeze remain unchanged.
- The public compact archive retains all natural forecasts, controlled episode
  losses, unchanged execution records, and the full controlled-prediction audit
  with digests. Replaying compact losses is distinguished from regenerating and
  checking omitted controlled predictions.

The tested scaling alternative does not account for the unfavorable aggregate natural-data comparison. The results support conclusions conditional on preprocessing and the evaluated procedure. Because the data had already been inspected, they do not validate selection of a new default scaler or a general rule for KAHM adoption.

Exact results are in [analysis/summary.json](analysis/summary.json),
[analysis/AUDIT.json](analysis/AUDIT.json), and the adjacent CSV files.
The [reproduction guide](README.md) gives restoration, audit and full-run
commands. The integrated paper and anonymous review package use these same
audited values.
