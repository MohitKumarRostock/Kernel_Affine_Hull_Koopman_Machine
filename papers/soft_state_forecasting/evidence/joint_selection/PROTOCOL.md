# Joint scaling and dimension selection for soft-state forecasts

## Scope and relationship to earlier evidence

This prospective follow-up evaluates joint selection of input transformation,
native local dimension, and ridge regularization in the current representation-only
soft-state architecture. It preserves the completed `soft_state_ablation_20261102`
study and its results. Earlier results motivated this question; they are not pooled
with the new repetitions, and this is not independent hypothesis discovery or an
external preregistration. The source, configuration, analysis, and audit are frozen
and published before production data generation.

G contains membership gates. P adds gated local principal coordinates. K adds
KAHM reconstruction departures to P. R adds matched positive-kernel/RBF departures
to P. All predict the same observed future response directly using horizon-specific
affine ridge heads. This study introduces no residual target, raw-history baseline,
correction multiplier, calibration, online adaptation, or autonomous state dynamics.

## Fixed cohort and pairing

There are 32 new independent repetitions, four inherited observation laws
(linear AR, nonlinear AR, Van der Pol, Duffing), and added-noise levels 0 and 0.1:
256 task suites in total. The history width is 12, horizons are 1, 4, and 12, the
origin stride is 4, and each episode has 72 stored observations. The oscillators'
inherited observation noise remains present at added-noise level zero.

Independent episode sets A/B/C/D contain 64/32/32/128 episodes. Retained history
rows are capped at 512/256/256 on A/B/C; all 1,664 D origins per task are scored.
Added Gaussian noise is scaled by the standard deviation of original A episode
values. Base episodes, unit-noise draws, retained-row choices, and fitting streams
are paired across noise levels, arms, input transformations, caps, and selection
views. The existing observation functions and window indexing remain unchanged.

The new master seed is 2026110204. Before generation, the complete 1,696-stream
roster is checked for unintended internal collisions and against the predecessor
exclusions, including every declared stream in the completed soft-state study.
These checks establish deterministic stream identities, not cryptographic
independence. The cohort size and candidate menu are fixed before production.
No stopping, extension, candidate addition, or selection change may depend on D.

## Geometry candidates and training-only transformations

The candidate grid is the Cartesian product of the ordered input scalers
`standard`, `minmax`, `tanh_z_1` and ascending native caps `1,2,4,8,12`.
Scaler order is the outer loop and cap order is the inner loop. A candidate is
identified as `SCALER_capCAP`, for example `tanh_z_1_cap4`. Exact ties choose the
first candidate in that order. Cap 20 aliases cap 12 at ambient width 12 and is
checked only on manufactured data, not counted as a search opportunity.

Each input transform is fitted on the entire designated A history matrix before
bank subsampling. Standardization uses A population means and standard deviations,
with the inherited constant-coordinate rule. Min-max uses A extrema and maps a
nonconstant coordinate to `2*(x-min_A)/(max_A-min_A)-1`; queries are not clipped.
An exactly constant A coordinate maps to zero for every query. `tanh_z_1` applies
`tanh((x-mean_A)/scale_A)` with the inherited population-scale threshold rule and
no subsequent restandardization of the transformed input histories. These maps
need not preserve the same distances or numerical information.

Within each scaler/cap candidate, G/P/K/R share the same A-fitted input transform,
reference partitions, centers, bases, and locality gates. Partitions have 2, 4,
and 8 local models, at most 24 references per model, and the inherited 512-row bank
limit. The projected Gaussian kernels, reconstruction regularization, and relative
denominator checks remain unchanged. Native initial rank is
`min(cap, ambient_dimension, references-1)`, followed by the native variance/range
reductions. Full ambient rank is permitted. Different scalers may change partitions
and references; within a scaler, candidates use the same fitting seed.

The gate vector is nonnegative and sums to one within each partition. Only that
block is simplex-valued. Coordinates and departures can be signed. The KAHM
inverse-Gram coefficients are distinct from the nonnegative locality gates.

## Feature blocks and direct ridge readouts

For every candidate, the gate, local-position, KAHM-departure, and RBF-departure
blocks are separately centered and scaled on A only, with feature-scale floor
1e-6. A standardized block v of width p is mapped to
`v/sqrt(1+||v||^2/p)`; an empty block remains empty. There is no further
standardization or radial normalization of a concatenated descriptor.

G is exactly the gate prefix of P; P is exactly the shared prefix of K and R.
Adding departures does not rescale common coordinates. K and R retain equal
nominal departure widths. Equal widths and candidate counts do not establish
equal function spaces, effective complexity, conditioning, or computation.

All ridge coefficients and unpenalized intercepts are fitted using A. The fixed
penalty grid is `1e-4,1e-3,1e-2,1e-1,1`. B chooses a penalty separately for every
candidate, arm, and horizon; the first minimum resolves ties. Candidate score is
the equal-horizon mean of selected B MSE divided by the corresponding A-target
population variance, floored at 1e-12. The selected A-fitted models are kept:
there is no refit on A+B or C. C predictions are retained as diagnostics and
cannot affect fitting, candidate selection, or penalty selection.

The inherited checked ridge solver is unchanged. Its documented NumPy SVD
failure recovery uses SciPy gesvd for the same objective and records the event.
Unexpected nonfinite values or model failures stop the study. Failed tasks are
not silently excluded or replaced by another estimator.

## Four paired selection views

Each task retains 16 logical forecast arrays, using the same 15 candidate fits:

| View | Candidate selection | Question |
| --- | --- | --- |
| `shared` | P selects a joint scaler/cap pair on B; all arms use that bank and their own penalties | Enrichment of a selected common representation |
| `independent` | Each arm selects its own joint scaler/cap pair on B | Performance of each complete tuned procedure |
| `standard_shared` | P selects a cap from the cached standard-only candidates | Same-cohort structural reference without scaling search |
| `standard_independent` | Each arm selects a cap from its cached standard-only candidates | Same-cohort complete-procedure reference without scaling search |

Standard-only forecasts reuse exact fitted candidates, with no additional fitting.
The joint search contains the standard-only menu, so its minimum B score cannot
be higher. This is a validation identity, not a guarantee of lower D loss. The
shared comparison deliberately selects geometry for P; it does not separately
optimize K. Independent selection is therefore reported explicitly. Comparisons
across differently selected banks are never described as matched-bank ablations.

## Prespecified primary family

For each arm/view, losses are standardized using A-target variance and averaged
over origins and horizons within a task. Each nonlinear repetition loss averages
the six nonlinear-law/noise cells equally; each linear loss averages the two
linear-law/noise cells. The 32 independent repetition aggregates are the
inferential units. Overlapping windows, task cells, and horizons are not treated
as independent experimental replications.

There is one family of six one-sided primary contrasts:

| Candidate | Reference | Population | Criterion |
| --- | --- | --- | --- |
| `shared__P` | `shared__G` | Nonlinear | 3% superiority |
| `shared__K` | `shared__P` | Nonlinear | 3% superiority |
| `shared__K` | `shared__R` | Nonlinear | 3% superiority |
| `shared__K` | `shared__P` | Linear | 5% noninferiority |
| `independent__K` | `standard_independent__K` | Nonlinear | 3% superiority |
| `independent__K` | `standard_independent__K` | Linear | 5% noninferiority |

For repetition r let `d_r = L_candidate,r - c*L_reference,r`, where c is 0.97
for superiority and 1.05 for noninferiority. A criterion is established only if
`mean(d) + t_(1-0.05/6,31)*sd(d)/sqrt(32) < 0`. The Bonferroni family adjustment
is applied to all six contrasts. Student bounds use an approximate model for
independent repetition means. Margins are inherited research conventions, not
deployment-derived utility thresholds.

Percentage gains are `100*(1-mean(candidate)/mean(reference))`, not an average
of individual percentage gains. Independent-selection K/R and K/P comparisons,
standard-only structural comparisons, law/noise/horizon breakdowns, selected
scaler/cap frequencies, candidate curves, and other paired 95% intervals are
descriptive. They cannot replace a failed primary criterion. All arms and all
conditions, including unfavorable results, are reported. Failure to establish a
margin does not establish equivalence.

## Execution order, reproducibility, and audit

Manufactured-only preflight checks precede the source freeze. They verify
standard-only agreement with the published architecture, training-only scalers,
shared feature prefixes, native cap aliases, constant/zero-rank handling,
penalty and joint/subset choices, serialization, state immutability, and the
full staged artifact chain. Artificial fixtures use separate test streams;
their outputs are not scientific results.

Production order is: freeze source/configuration/analysis/audit; publish and
verify the frozen source; record its commit-bound publication receipt; prepare
A/B/C; fit every task; seal all fitted states; generate D queries and separately
retained targets; predict using queries only; seal every forecast; release D
targets; analyze; independently audit. No fitting API accepts C or D targets.
Candidate B predictions and scores, all selected models, input identities,
fitted portable states, predictions, numerical events, and measured costs are
retained. Hash seals document order and detect changes; they are not independent
oversight or access controls.

The independent audit checks the complete task and forecast roster, source and
artifact links, candidate/subset choices, losses, repetition aggregation, and
all six bounds. Its report records the exact scope of feature/prediction replay,
data regeneration, and any sampling. Computational agreement is not independent
scientific replication. Resource totals include rejected candidates and shared
computations; they are not isolated inference costs of one deployed arm.

The follow-up is complete only when its entire fixed cohort, analysis, and audit
are complete, irrespective of the outcome. Joint tuning is selection within a
finite declared grid, not global optimization of every KAHM parameter. The
experiment remains a controlled simulation study and does not establish
autonomous soft-state dynamics, Koopman closure, or performance in new domains.
