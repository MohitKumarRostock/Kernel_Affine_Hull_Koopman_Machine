# Joint input scaling and native dimension selection: results

**Production analysis and the independent production audit are complete.**
The audit passed 7,422,883 checks across the full cohort. All
32 fresh independent repetitions, 256 tasks, 4,096 terminal forecast arrays,
and six prespecified comparisons are present. The frozen analysis establishes
four of the six criteria. It establishes the nonlinear contributions of local
position and KAHM enrichment relative to their prescribed shared-bank controls,
and both linear noninferiority criteria. It does not establish either the
KAHM-over-RBF superiority margin or the joint-tuning superiority margin.

The source was published as commit
[`8e0250279074aefa39079fa33205d3c74f835848`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/8e0250279074aefa39079fa33205d3c74f835848)
before production preparation, on branch
`research/kahm-soft-state-joint-selection-2026-11-02`.
The [publication receipt](PUBLICATION.json) identifies the actual public source.
The [analysis summary](analysis/analysis_summary.json) has SHA-256
`1345429169d0bd263d17a54fb469718a768a0be218db18059b0084176d2421e9`.
The [independent audit summary](verification/independent_audit_summary.json)
records the completed computational coverage. The full receipt has SHA-256
`8ba3931c3e705838d9f3ec924f374b8cb9e492aa6f90803684412d8151a9c89c`;
its [lossless compressed copy](verification/independent_audit.json.gz) restores
the exact original JSON. The audit's [implementation and scope](audit/README.md)
are separate from the manufactured-data preflight checks.

## Design and interpretation

The study retains the representation-only G/P/K/R architecture. G contains
locality gates, P adds gate-weighted local principal coordinates, K adds KAHM
departures to P, and R adds matched positive-kernel RBF departures to P. All
arms use horizon-specific affine ridge heads to predict the same observed
future response. Raw gates sum to one within each partition; signed coordinates
and departures do not make the full representation simplex-valued.

The fresh cohort contains four inherited laws, two paired added-noise levels,
history width 12, and horizons 1, 4, and 12. A/B/C/D contain 64/32/32/128
episodes. A fits input transforms, geometry, feature transforms, and readouts;
B selects penalties and geometry; C supplies diagnostics; D supplies final
evaluation. Every task scores all 1,664 D origins. The new master seed is
2026110204. The [protocol](PROTOCOL.md) and [configuration](config.json) retain
the full allocation and procedural order.

Joint selection searches 15 candidates: A-fitted `standard`, `minmax`, and
`tanh_z_1` input transforms crossed with native caps 1, 2, 4, 8, and 12. Each
candidate uses the same five ridge penalties. The four views reuse these same
fits:

| View | Geometry selection on B |
| --- | --- |
| `shared` | P selects a joint scaler/cap pair; G/P/K/R share that bank |
| `independent` | Each arm selects its own joint scaler/cap pair |
| `standard_shared` | P selects among the cached standard-only candidates |
| `standard_independent` | Each arm selects among its cached standard-only candidates |

The standard-only comparator uses the same fresh observations and exact
standard candidate fits as joint selection. It is not the earlier study's
reported mean. The predecessor remains at
[`soft_state_ablation_20261102`](../soft_state_ablation_20261102/RESULTS.md), with
its own 32 repetitions and four-comparison family. No observations are pooled.

## All six primary comparisons

For repetition r, the tested difference is
`d_r = loss_candidate,r - c * loss_reference,r`, with c = 0.97 for superiority
and c = 1.05 for noninferiority. A criterion is established only when
`U = mean(d) + 2.530925845218133 * sd(d) / sqrt(32)` is strictly negative.
The critical value applies a one-sided Bonferroni adjustment across all six
comparisons, with family alpha 0.05 and 31 degrees of freedom. Windows,
horizons, and law/noise cells are not independent inference units.

Relative gain is `100 * (1 - mean(candidate) / mean(reference))`; a negative
gain denotes higher loss. U is in standardized-loss units and already includes
the prescribed margin. These quantities are different scales.

| Primary comparison | Panel | Candidate SMSE | Reference SMSE | Relative gain | Margin-adjusted U | Criterion established |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Shared P versus G | Nonlinear | 0.02935955 | 0.09377669 | 68.69% | -0.05878519 | Yes: 3% superiority |
| Shared K versus P | Nonlinear | 0.02675882 | 0.02935955 | 8.86% | -0.00144862 | Yes: 3% superiority |
| Shared K versus R | Nonlinear | 0.02675882 | 0.02651524 | -0.9186% | 0.00119423 | No: 3% superiority |
| Shared K versus P | Linear | 0.92708684 | 0.91556375 | -1.26% | -0.03053738 | Yes: 5% noninferiority |
| Independent joint K versus standard-only K | Nonlinear | 0.02621031 | 0.02622938 | 0.0727% | 0.00090112 | No: 3% superiority |
| Independent joint K versus standard-only K | Linear | 0.92088119 | 0.91859298 | -0.2491% | -0.04175207 | Yes: 5% noninferiority |

Sources: [primary contrasts](analysis/primary_contrasts.csv),
[paired repetition records](analysis/primary_paired_repetitions.csv), and
[analysis summary](analysis/analysis_summary.json). Rounded displays do not
replace the retained numerical precision.

Local position reduces shared-bank nonlinear loss by 68.69%, and adding KAHM
departures reduces P's loss by 8.86%. K nevertheless has 0.9186% higher loss
than matched R in that context. Joint selection gives independent K only a
0.0727% nonlinear mean gain over standard-only selection. Its adjusted bound
does not establish the prescribed 3% improvement. Linear K has 1.26% higher
loss than shared P; independent joint K has 0.2491% higher loss than its
standard-only comparator. Both remain within their prespecified 5%
noninferiority criteria. Noninferiority does not assert an improvement, and
failure to establish a superiority margin does not establish equivalence.

## Complete panel losses

These are equal means over the appropriate law/noise cells within each
repetition and then over 32 repetitions. Each task first averages its three
horizons. The nonlinear panel contains nonlinear AR, Van der Pol, and Duffing;
the linear panel contains linear AR.

| Search | Geometry | Arm | Nonlinear SMSE | Linear SMSE |
| --- | --- | --- | ---: | ---: |
| Joint | P-shared | G | 0.09377669 | 0.94069450 |
| Joint | P-shared | P | 0.02935955 | 0.91556375 |
| Joint | P-shared | K | 0.02675882 | 0.92708684 |
| Joint | P-shared | R | 0.02651524 | 0.91984061 |
| Joint | Own | G | 0.09216419 | 0.93732333 |
| Joint | Own | P | 0.02935955 | 0.91556375 |
| Joint | Own | K | 0.02621031 | 0.92088119 |
| Joint | Own | R | 0.02599175 | 0.92044205 |
| Standard only | P-shared | G | 0.09661958 | 0.94396547 |
| Standard only | P-shared | P | 0.02978903 | 0.91478885 |
| Standard only | P-shared | K | 0.02669383 | 0.92246216 |
| Standard only | P-shared | R | 0.02664788 | 0.91945607 |
| Standard only | Own | G | 0.09661958 | 0.94396547 |
| Standard only | Own | P | 0.02978903 | 0.91478885 |
| Standard only | Own | K | 0.02622938 | 0.91859298 |
| Standard only | Own | R | 0.02609260 | 0.91871374 |

Source: [panel summary](analysis/panel_summary.csv). Independent joint K has
nonlinear mean loss 0.02621031 versus R's 0.02599175, or 0.8409% higher loss.
This independent K/R comparison is descriptive and is not a primary test.

## Per-law tuning effects: descriptive

The following means average both noise conditions equally across all 32
repetitions. Both gain columns use independent selection. The final column
compares joint K with joint R; it does not estimate the effect of scaling alone.

| Law | Standard-only K SMSE | Joint K SMSE | Joint R SMSE | Joint K gain over standard K | Joint K gain over joint R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear AR | 0.02772543 | 0.02786083 | 0.02655174 | -0.4884% | -4.93% |
| Van der Pol | 0.02447644 | 0.02442091 | 0.02494795 | 0.2269% | 2.11% |
| Duffing | 0.02648627 | 0.02634919 | 0.02647557 | 0.5176% | 0.4773% |
| Linear AR | 0.91859298 | 0.92088119 | 0.92044205 | -0.2491% | -0.0477% |

Source: equal-noise aggregation of
[condition means](analysis/condition_summary.csv). No per-law primary
inference was declared. Joint tuning has favorable mean effects on the two
oscillators and unfavorable effects on both AR laws in this cohort. These
small, mixed effects do not establish a broadly useful scaling policy.

For nonlinear independent K, joint selection wins in 15 repetition aggregates
and loses in 17, with no ties. For linear K, it wins in none, ties in 18, and
loses in 14. The unadjusted nonlinear mean paired loss difference is
-0.0000190713, with a descriptive 95% interval from -0.0001284791 to
0.0000903366. This unadjusted interval and the win counts do not replace the
six-comparison primary bound. See the
[optimization summary](analysis/optimization_summary.csv) and
[paired repetition distribution](analysis/optimization_repetition_pairs.csv).

## Selected scalers and caps

Independent K selected the following scalers on B. Each law has 64 task cells
(32 repetitions times two noise conditions), so these are selection frequencies,
not 64 independent inference units.

| Law | Standard | Min-max | Tanh | Joint median cap | Standard-only median cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear AR | 34 | 13 | 17 | 4 | 4 |
| Van der Pol | 30 | 30 | 4 | 1 | 1 |
| Duffing | 27 | 21 | 16 | 2 | 3 |
| Linear AR | 48 | 9 | 7 | 8 | 8 |

The median of selected caps can fall between available values: the
standard-only Duffing median of 3 is the average of central caps 2 and 4.
A selected cap is not the realized rank of every local model.

| Law | Joint K cap 1 | Cap 2 | Cap 4 | Cap 8 | Cap 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear AR | 13 | 4 | 16 | 31 | 0 |
| Van der Pol | 45 | 19 | 0 | 0 | 0 |
| Duffing | 3 | 30 | 31 | 0 | 0 |
| Linear AR | 0 | 0 | 13 | 30 | 21 |

Across nonlinear tasks, independent K chooses standard/min-max/tanh on
91/64/37 of 192 tasks (47.4%/33.3%/19.3%). The corresponding counts are
78/71/43 for P and 82/69/41 for R. On linear tasks, K chooses 48/9/7 of
64 (75.0%/14.1%/10.9%), compared with 31/28/5 for P and 32/28/4 for R.

Sources: [task selections](analysis/selection_by_task.csv),
[scaler frequencies](analysis/scaling_selection_frequencies.csv),
[cap frequencies](analysis/cap_selection_frequencies.csv), and
[joint frequencies](analysis/joint_selection_frequencies.csv).
The larger candidate menu cannot worsen the selected B objective for its
selector because it contains every standard-only candidate. That property
does not guarantee lower D loss. Frequencies describe the validation procedure;
they do not establish that one scaler is intrinsically superior.

## Retained evidence, resources, and limits

The analysis retains 23 CSV tables, including all candidate B scores, C
diagnostics, D losses, selections, effective local dimensions, feature widths,
paired differences, and resources. All unfavorable arms and conditions remain
in those tables. The complete [analysis table index](analysis/analysis_summary.json)
records row counts and SHA-256 identities.

Fitting all scaler/cap candidates and four arms consumed 1,177.70 CPU seconds
and a sum of 1,177.83 task wall seconds, with median 4.56 seconds per task.
The terminal prediction stage summed to 760.07 wall seconds. These are measured
stage costs from [resource records](analysis/resource_summary.csv), not elapsed
parallel-job duration or an isolated deployment cost for K. Shared computations,
rejected candidates, and reuse across views prevent direct attribution to one arm.

The source freeze and fit/prediction/target-release seals document the declared
order. They are not public preregistration, independent oversight, or
cryptographic target isolation. Fresh realizations of known simulation laws
provide new sampling evidence under those laws, not new application domains.
The finite declared grid does not globally optimize geometry or forecasting
performance. The research margins are inherited conventions; they were not
derived from deployment costs. Direct observed-output prediction does not
establish sufficient Markov states or autonomous soft-state dynamics.

### Completed independent computational audit

The independent audit passed all 256 tasks and 7,422,883 checks. It independently
refitted and replayed all 76,800 B penalty-path arrays, replayed all 4,096 C and
4,096 D forecast arrays, and recomputed the six primary bounds and their four
established criteria. It checked every A-fitted input scaler, reconstructed
histories and targets from retained episodes and reserved streams, and checked
local references, native dimensions, and kernel identities. No numerical
solver recoveries occurred.

Maximum absolute forecast differences were 1.127675730572264e-10 on B penalty
paths, 2.4236168627567167e-12 on C, and 6.276451580689013e-12 on D. Replay uses
the frozen relative tolerance 2e-7 and absolute tolerance 2e-9. Byte and prefix
hash checks are exact; numerical replay is agreement within those tolerances.

Physical-law integration and partition k-means fitting are not independently
reimplemented. This is an internal computational audit, not external scientific
replication. The [audit summary](verification/independent_audit_summary.json)
and [full compressed receipt](verification/independent_audit.json.gz) preserve
its coverage and numerical differences. The
[reporting and restoration guide](reporting/README.md) distinguishes manuscript
builds, display regeneration, and complete numerical replay.
