# Completed soft-state ablation results

The fixed study completed on 2 November 2026 with all **32 independent
repetitions, 256 tasks, four primary comparisons, and the passed independent
computational audit**. Three of the four prespecified criteria were established.
In the nonlinear population, adding local position to memberships improved
mean loss by **69.20%**, and adding KAHM departures to that common representation
improved it by **10.46%**. The KAHM-over-RBF point gain was only **0.03%** and
did not establish the required 3% margin. KAHM had **1.11% higher linear loss**
than the common representation while satisfying the 5% noninferiority criterion.

These results support the predictive contribution of local position and
reconstruction enrichment under the declared procedure. They do not establish
a general KAHM-specific advantage over matched positive-kernel enrichment.
The complete numerical values and identities are in the
[analysis summary](analysis/analysis_summary.json) and
[independent audit summary](verification/independent_audit_summary.json).

## Representation and comparison scope

G contains transformed membership gates. P adds gate-weighted local principal
coordinates. K adds KAHM departures to P, and R adds positive-kernel RBF
departures to the same P. Raw memberships sum to one within each partition,
so their concatenation lies in a product of simplices. Separately transformed
blocks and signed enrichments need not be simplex-valued. The transforms retain
exact common G/P prefixes when extra blocks are appended.

Each arm uses a direct affine ridge readout of its declared representation to
predict the same observed future response. A fits geometry and coefficients;
B selects the penalty and cap; C supplies diagnostics only; D supplies final
evaluation. There is no retained raw-history forecast, residual-target head,
or correction multiplier. The primary comparison uses P's B-selected cap for
all four arms. This is deliberately conditional on a validation-selected
common bank. Independent per-arm cap selection is secondary.
[Sources: protocol](PROTOCOL.md), [configuration](config.json).

## All four primary comparisons

Errors are divided by the corresponding A-target population variance and
averaged over retained D origins and horizons 1, 4, and 12 within a task.
Each nonlinear repetition averages its six nonlinear-law/noise cells equally;
each linear repetition averages its two noise cells. The **32 complete
repetition aggregates** are the inferential units. Overlapping windows,
horizons, and the 256 task conditions are not additional independent units.

For each repetition, the tested difference is
`d = candidate loss - c * reference loss`, with `c = 0.97` for nonlinear
superiority and `c = 1.05` for linear noninferiority. The upper bound is
`mean(d) + t_(0.9875,31) * sd(d)/sqrt(32)`. The four one-sided paired Student
bounds use Bonferroni family alpha 0.05. A criterion is established only when
the bound is strictly negative. The critical value is 2.35556828216; the
Student model is an approximation for independent repetition means. The
3%/5% margins are inherited research conventions, not deployment-derived costs.

| Population | Contrast | Criterion | Point gain | Mean margin difference | Simultaneous upper bound | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Nonlinear | P versus G | 3% superiority | 69.201371% | -0.065453523934 | -0.06253378478 | Established |
| Nonlinear | K versus P | 3% superiority | 10.458540% | -0.0022711785534 | -0.0019873334161 | Established |
| Nonlinear | K versus R | 3% superiority | 0.029216% | 0.00081025102409 | 0.00098101793433 | **Not established** |
| Linear | K versus P | 5% noninferiority | -1.111369% | -0.035422357199 | -0.031341020701 | Established |

Point gain is `100 * (1 - mean(candidate)/mean(reference))`, not an average
of per-repetition percentages. Margin differences and bounds are absolute
standardized-loss values, not percentage bounds. The linear criterion permits
a limited deterioration; it is not linear superiority. Failure of the K-versus-R
margin is not an equivalence result or proof of zero benefit.
[Sources: primary table](analysis/primary_contrasts.csv),
[paired repetition values](analysis/primary_paired_repetitions.csv),
[protocol](PROTOCOL.md).

## Mean loss for every arm

The shared-cap rows are the primary view. The independent-cap rows describe a
different, complete selection procedure and cannot replace an unsuccessful
primary result. Each displayed value is mean D standardized MSE across 32
complete repetitions.

| Cap selection | Arm | Nonlinear mean SMSE | Linear mean SMSE |
| --- | --- | --- | --- |
| Shared | G | 0.098870344704 | 0.93904211322 |
| Shared | P | 0.030450710429 | 0.91092102224 |
| Shared | K | 0.027266010562 | 0.92104471615 |
| Shared | R | 0.027273978906 | 0.91586573775 |
| Independent | G | 0.098870344704 | 0.93904211322 |
| Independent | P | 0.030450710429 | 0.91092102224 |
| Independent | K | 0.026754569714 | 0.91493610488 |
| Independent | R | 0.026750705751 | 0.91589004526 |

The ordinary shared-cap nonlinear K-minus-R difference is
-7.9683430761e-06, with a **descriptive, unadjusted paired 95%
interval** [-0.00015892601679, 0.00014298933064].
With independent caps, the K-minus-R difference is
3.863962726e-06, interval
[-0.00013086952675, 0.0001385974522],
and the K-over-R point gain is -0.014444%.
The small change in point ordering does not establish either method's
superiority or equivalence. All 24 planned secondary contrasts are retained.
[Sources: panel summary](analysis/panel_summary.csv),
[secondary descriptive comparisons](analysis/secondary_descriptive.csv).

## Descriptive heterogeneity

The following shared-cap condition means average the three horizons within
each of the 32 repetitions. The added-noise setting is eta; zero added noise
does not remove the inherited oscillator sensor noise. Positive gains favor
the first named arm. These are descriptive comparisons, with no additional
condition-specific confirmatory claims.

| Law | eta | G | P | K | R | K versus P gain | K versus R gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nonlinear AR | 0 | 0.02442813 | 0.02144490 | 0.02261477 | 0.02174993 | -5.455% | -3.976% |
| Nonlinear AR | 0.1 | 0.03371238 | 0.03296585 | 0.03510191 | 0.03364418 | -6.480% | -4.333% |
| Van der Pol | 0 | 0.14803730 | 0.01516777 | 0.00328116 | 0.00467759 | 78.368% | 29.854% |
| Van der Pol | 0.1 | 0.14930766 | 0.05114168 | 0.04719829 | 0.04778715 | 7.711% | 1.232% |
| Duffing | 0 | 0.11893512 | 0.01587385 | 0.01333026 | 0.01387444 | 16.024% | 3.922% |
| Duffing | 0.1 | 0.11880148 | 0.04611023 | 0.04206968 | 0.04191058 | 8.763% | -0.380% |
| Linear AR | 0 | 0.93764818 | 0.90951392 | 0.91895053 | 0.91456597 | -1.038% | -0.479% |
| Linear AR | 0.1 | 0.94043604 | 0.91232812 | 0.92313890 | 0.91716551 | -1.185% | -0.651% |

K improves P in both oscillator laws at both noise settings but worsens P in
nonlinear autoregression at both settings. Its K-over-R advantage is large
in Van der Pol without added noise and changes sign across other conditions.
Thus the aggregate nonlinear result does not imply a uniform benefit.
[Source: condition summary](analysis/condition_summary.csv).

The nonlinear horizon decomposition also shows where the aggregate
K-over-P reduction arises:

| Horizon | P mean SMSE | K mean SMSE | R mean SMSE | K versus P gain |
| --- | --- | --- | --- | --- |
| 1 | 0.012255109 | 0.012638634 | 0.012042172 | -3.130% |
| 4 | 0.019254072 | 0.019227180 | 0.018675706 | 0.140% |
| 12 | 0.059842950 | 0.049932217 | 0.051104059 | 16.561% |

These horizon rows use the same repetitions and their declared A-target
variance normalization; they are not independent tests or additional sampling
units. [Source: horizon summary](analysis/horizon_summary.csv).

## Selected caps and native ranks

The five native caps are 1, 2, 4, 8, and 12. The following counts cover all 256
task conditions; they are descriptive, not independent inferential units.

| Selection rule | Cap 1 | Cap 2 | Cap 4 | Cap 8 | Cap 12 |
| --- | --- | --- | --- | --- | --- |
| Shared cap selected by P | 31 | 104 | 75 | 32 | 14 |
| Independent G | 256 | 0 | 0 | 0 | 0 |
| Independent P | 31 | 104 | 75 | 32 | 14 |
| Independent K | 67 | 57 | 73 | 42 | 17 |
| Independent R | 59 | 75 | 89 | 16 | 17 |

The shared row applies to all four primary arms. Independent G selects cap 1
by the declared first-minimum tie rule because its gate representation is
identical across caps; these are not five different gate function spaces.
Cap 20 is an alias of 12 at history width 12 and is not an extra candidate.

The audit checked all 17,920 native local-model/cap records. One singleton
local model has rank zero under all five caps; one two-reference model has
rank one at caps 2, 4, 8, and 12. All other local ranks equal their requested
caps. These are declared native eligibility outcomes. There were no
coordinate-range reduction steps and no query-denominator fallback calls.
[Sources: cap frequencies](analysis/cap_selection_frequencies.csv),
[task selections](analysis/selection_by_task.csv),
[compressed dimension records](analysis/basis_dimensions.csv.gz),
[compressed full audit](verification/independent_audit.json.gz).

## Measured computation and storage

These are complete-search costs for all five caps and all four arms with
shared geometry; they are not isolated method costs or evidence of equal
computation across arms. C diagnostics are included in whole-fit totals.
The study used three workers with one numerical thread per worker.

| Measured quantity | Value | Unit |
| --- | --- | --- |
| Whole fit: sum of task wall times | 931.880 | s |
| Whole fit: sum of task CPU times | 929.017 | s |
| D prediction: sum of task wall times | 671.839 | s |
| D prediction: sum of task CPU times | 670.600 | s |
| Maximum worker peak RSS | 156.992 | MiB |
| All-candidate portable NPZ states | 177.934 | MiB |
| All-candidate portable JSON states | 88.239 | MiB |
| Fit-stage parallel elapsed wall time | 363.525 | s |
| Prediction-stage parallel elapsed wall time | 280.233 | s |

Summed task wall time differs from parallel stage elapsed time; peak worker
RSS is not simultaneous total memory. Stored states include every candidate,
not only a minimally deployed predictor. The process-stage receipts also
include worker-launch and orchestration time. Analysis, independent audit,
report generation, packaging, and compression are outside the listed fit and
prediction task costs.
[Source: resource summary](analysis/resource_summary.csv);
the sealed fit/prediction stage receipts and all resource records are preserved
in the separate numerical archives.

## Independent computational verification

The full audit passed **2,563,814 checks** across all 256 tasks:
**25,600 B penalty-path arrays**,
**2,048 selected C arrays**, and
**2,048 selected D arrays**. The checker
reconstructs windows, targets, noise and row identities from retained physical
episodes; independently constructs the four feature maps; refits B ridge paths
using Cholesky normal equations rather than the production SVD; replays selected
C/D predictions; and recomputes losses, repetition aggregates and all four bounds.
It also checks native dimensions, kernel identities, partition normalization,
shared-prefix hashes, selections, and source/stage/seed identities.

Maximum absolute forecast discrepancies were 1.20209398e-11
for B, 2.42889042e-12 for C, and
5.58575408e-12 for D, under the frozen replay
tolerances `rtol=2e-07`, `atol=2e-09`.
Maximum absolute primary-upper-bound discrepancy was
2.08166817e-17.
There were **0 numerical solver recoveries**.
All 1,696 declared streams were independently rederived, were distinct where
required, and had no overlap with the recorded historical exclusions.

Partition k-means fitting and physical-law integration were not independently
reimplemented. This is a separate computational audit, not an independent
scientific replication by another group. Source publication and stage hashes
establish recorded workflow order, not cryptographic target isolation.
[Sources: audit summary](verification/independent_audit_summary.json),
[complete audit compressed without alteration](verification/independent_audit.json.gz).

## Source identity and recorded stage order

The prospectively published scientific source commit is
[`2b7d19b5b47f46b5b21b28d24960688f7969d96f`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2b7d19b5b47f46b5b21b28d24960688f7969d96f).
It was verified before production preparation. Earlier simulations informed
the research question, and public preregistration is not claimed. The commit
carrying this completed results report has its own repository identity; it
must not be substituted for the scientific source commit.

All of the following times are UTC on 2 November 2026:

| Event | UTC time |
| --- | --- |
| Source freeze | 08:33:38 |
| Source publication receipt | 09:20:24 |
| A/B/C preparation completed | 09:20:57 |
| Complete fit seal | 09:28:00 |
| D preparation completed | 09:28:33 |
| Complete prediction seal | 09:34:57 |
| Target release | 09:35:11 |
| Analysis completed | 09:35:18 |
| Independent audit completed | 09:37:28 |

The fit and prediction seals cover the complete 256-task population. The D
manifest binds the fit seal and A/B/C manifest; target release binds both seals
and the D manifest; the analysis binds the release; the audit binds the exact
analysis summary. These links were independently verified.
[Sources: freeze](freeze.json), [publication](PUBLICATION.json),
[fit seal](FIT_LOCK.json), [prediction seal](PREDICTION_LOCK.json),
[target release](TARGET_RELEASE.json), [analysis](analysis/analysis_summary.json).

Full audit SHA256:

```text
72ed918ecb2a5a60b11de415e04f876a6187842785da7d2c6af21c20346a6398
```

Analysis summary SHA256:

```text
d8de5e0f88fb9b986c6faa2ff8d762abf3e5c260edc91c739bb930d207d125b6
```

## Compact release and reproduction

The largest B-score table, complete dimension table and full audit receipt are
published as `.gz` copies. Their original bytes, line endings, sizes and hashes
are preserved. All other analysis CSVs and the analysis summary are plain files.
The [compact release manifest](reporting/compact_release_manifest.json) binds
both byte representations. The [restoration utility](reporting/restore_compact_records.py)
restores only those three files, accepts identical existing originals, and
rejects conflicting ones. It performs no scientific calculations.

The supplied manuscript tables and figures already compile directly. Full
asset regeneration can first restore the three records; a complete computational
replay additionally requires the separately indexed numerical archives with
prepared episodes, fitted states, all B/C paths, predictions and execution logs.
The [numerical archive summary](reporting/numerical_archive_summary.json)
records archive/index identities and the availability limit: they are retained
as separate author deliverables, with no public download URL asserted.
See [README.md](README.md) and [reporting instructions](reporting/README.md).
No numerical observations were pooled with earlier studies, omitted after
inspection, or used to extend the fixed repetition budget.

The evidence concerns four inherited laws, fresh repetitions, a selected
geometry and affine-readout family. It does not evaluate new application
domains, autonomous soft-coordinate forecasting or Koopman rollouts, establish
Markov sufficiency or closure, or demonstrate universal forecasting superiority.
Earlier dimension, residual and policy studies retain their original methods,
estimands and comparison families.
