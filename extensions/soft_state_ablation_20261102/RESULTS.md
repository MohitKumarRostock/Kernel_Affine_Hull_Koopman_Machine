# Matched representation results

## Design and inferential unit

The experiment contains 32 independent repetitions of four simulation laws at two added-noise levels, giving 256 task suites. G uses memberships; P adds local principal coordinates; K adds KAHM reconstruction departures to P; R adds matched positive-kernel RBF departures. All arms use direct affine ridge readouts of the same future observations. The primary procedure uses P's validation-selected bank for every arm. Independently selected banks are secondary.

Loss is squared observed-response error divided by retained-A target population variance, averaged across D origins and horizons 1, 4, and 12. Each nonlinear repetition averages six law/noise cells equally, and each linear repetition averages two. The inferential sample size is 32, not the number of origins, horizons, or tasks.

## Complete primary family

For each repetition, let `d = candidate loss - c * reference loss`, where `c = 0.97` for nonlinear superiority and `c = 1.05` for linear noninferiority. The upper endpoint is `mean(d) + t_(0.9875,31) * sd(d)/sqrt(32)`, with critical value 2.35556828216. A criterion is established only when this endpoint is negative. The four comparisons form one Bonferroni family with nominal error level 0.05 under the paired-Student approximation.

| Population | Contrast | Criterion | Gain (%) | Mean margin difference | Adjusted upper endpoint | Established |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Nonlinear | P versus G | 3% superiority | 69.201371 | -0.065453523934 | -0.062533784780 | Yes |
| Nonlinear | K versus P | 3% superiority | 10.458540 | -0.002271178553 | -0.001987333416 | Yes |
| Nonlinear | K versus R | 3% superiority | 0.029216 | 0.000810251024 | 0.000981017934 | No |
| Linear | K versus P | 5% noninferiority | -1.111369 | -0.035422357199 | -0.031341020701 | Yes |

Gain is `100 * (1 - mean(candidate)/mean(reference))`; it is not an average of repetition-wise percentages. Margin differences and upper endpoints are in absolute standardized-loss units, not percentages. Failure of the K/R criterion does not establish equivalence. Linear noninferiority permits deterioration within the specified tolerance and does not establish superiority.

Exact values are in [primary_contrasts.csv](analysis/primary_contrasts.csv), [primary_paired_repetitions.csv](analysis/primary_paired_repetitions.csv), and [repetition_losses.csv](analysis/repetition_losses.csv). The margins are research conventions rather than deployment-derived costs.

## Aggregate loss and independent selection

| Cap rule | Arm | Nonlinear mean SMSE | Linear mean SMSE |
| --- | --- | ---: | ---: |
| Shared | G | 0.098870344704 | 0.93904211322 |
| Shared | P | 0.030450710429 | 0.91092102224 |
| Shared | K | 0.027266010562 | 0.92104471615 |
| Shared | R | 0.027273978906 | 0.91586573775 |
| Independent | G | 0.098870344704 | 0.93904211322 |
| Independent | P | 0.030450710429 | 0.91092102224 |
| Independent | K | 0.026754569714 | 0.91493610488 |
| Independent | R | 0.026750705751 | 0.91589004526 |

Independent selection slightly favors R over K in nonlinear mean loss. The ordinary shared-cap K-minus-R loss difference is -0.00000796834, with descriptive unadjusted 95% paired interval [-0.00015892602, 0.00014298933]. With independent caps, the difference is 0.00000386396 and the interval is [-0.00013086953, 0.00013859745]. Neither comparison establishes equivalence. All 24 secondary comparisons are in [secondary_descriptive.csv](analysis/secondary_descriptive.csv); repetition standard deviations are in [panel_summary.csv](analysis/panel_summary.csv).

## Heterogeneity and selection

K worsens P on nonlinear autoregression at both noise levels and improves it on both oscillator laws. Its ordering against R changes across law/noise conditions. The complete means, including unfavorable cells, are in [condition_summary.csv](analysis/condition_summary.csv).

| Horizon | Shared P nonlinear SMSE | Shared K nonlinear SMSE | Shared R nonlinear SMSE |
| --- | ---: | ---: | ---: |
| 1 | 0.012255109 | 0.012638634 | 0.012042172 |
| 4 | 0.019254072 | 0.019227180 | 0.018675706 |
| 12 | 0.059842950 | 0.049932217 | 0.051104059 |

The aggregate K/P improvement is concentrated at horizon 12. These are descriptive results for separately fitted direct readouts, not recursive rollout steps or additional independent tests. All horizon results are in [horizon_summary.csv](analysis/horizon_summary.csv).

P selects caps 1, 2, 4, 8, and 12 in 31, 104, 75, 32, and 14 tasks. G's gates are identical across caps, so its independent choice of cap 1 is a tie-breaking outcome. Shared feature widths are 14 for G, 28–182 for P, and 196–350 for K/R. The experiment does not demonstrate compression of the 12-dimensional history. Full counts and native dimensions are retained in [cap_selection_frequencies.csv](analysis/cap_selection_frequencies.csv), [selection_by_task.csv](analysis/selection_by_task.csv), and [basis_dimensions.csv.gz](analysis/basis_dimensions.csv.gz).

## Computational scope and verification

Complete-search fitting uses 931.88 summed task wall seconds; D prediction uses 671.84 summed task wall seconds. Maximum worker peak resident memory is 156.99 MiB. These are not isolated K/R costs, elapsed parallel-stage times, or aggregate concurrent memory. All-candidate NPZ states occupy 177.93 MiB. [resource_summary.csv](analysis/resource_summary.csv) retains the measurements.

The retained audit reports 2,563,814 checks, refitting of 25,600 B penalty-path arrays with Cholesky calculations rather than production SVD, and replay of 2,048 arrays for each of C and D. Maximum absolute discrepancies are 1.20e-11 for B, 2.43e-12 for C, and 5.59e-12 for D under relative/absolute tolerances 2e-7/2e-9. There are no recorded solver recoveries. The [audit summary](verification/independent_audit_summary.json) and [losslessly compressed full receipt](verification/independent_audit.json.gz) define the complete coverage.

Partition clustering and physical-law integration are not independently reimplemented. This is computational verification, not external scientific replication. Stage seals establish recorded execution order, not cryptographic isolation of targets.

## Interpretation and reproducibility

The evidence supports local position and conditional reconstruction enrichment for the tested affine readouts. It does not establish KAHM-specific superiority over matched RBF, uniform benefit across laws, or an autonomous state representation. Earlier studies informed the design; the 32 repetitions are fresh samples under known laws, not new application domains. Earlier observations are not pooled into this family.

The [protocol](PROTOCOL.md), [configuration](config.json), [analysis summary](analysis/analysis_summary.json), and [source freeze](freeze.json) preserve the statistical contract. Scientific source commit `2b7d19b5b47f46b5b21b28d24960688f7969d96f` precedes production preparation; results release `65ecf679796cc38b2342e2a4c12fa0226dc9ac78` has a separate identity. [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) distinguishes document builds, saved-loss checks, compact restoration, full model audit, and simulation regeneration, including the availability limits of separately indexed numerical archives.
