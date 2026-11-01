# Cross-fitted residual correction with a full-training common predictor

## Scope and principal finding

Episode-wise cross-fitted KAHM corrections reduced nonlinear-panel standardized
mean squared error by 9.1968% relative to the identical full-training common
predictor with a shared bias adjustment. The prescribed 3% superiority criterion
was established for this comparison. The reduction against independently
selected cross-fitted RBF corrections was 2.8662%, which did not establish the
prescribed 3% margin. Error was 5.1246% higher than that of the expanded non-KAHM
selection policy. On the linear panel, KAHM error was 1.1314% higher than raw
ridge, while satisfying the prescribed 5% noninferiority criterion. Thus two
of the four primary criteria were established. These findings support residual
correction of the shared common forecast within this experiment, while the
comparisons with the independently selected RBF and expanded policy limit a
broader claim of KAHM superiority. The numerical results and decisions are
reported in [the primary comparison table](published_tables/primary_contrasts.csv).

The results arise from a computational reproduction of the original declared
simulation cohort. The published original source was recoverable at commit
`f36e7c797f37d7d073dae42603b3770543d9fb43`, but the completed original execution's
arrays, fitted states, forecasts, analysis tables, and seals were unavailable.
The reproduction source was published at
`77553bdcb6cc16fe22862b0727068f86be1d7cde` before regenerating data. Model,
numerical-solver, pipeline, and scientific-analyzer bytes match the original
source; all scientific configuration values and declared seed streams are
unchanged. Prior reported aggregate findings were known before this
reproduction. Reusing these streams adds zero independent repetitions and
does not authenticate the unavailable original execution or establish byte
identity with its records. The current results and verification concern the
newly retained execution. Source identity and this scope are documented in
[the reproduction protocol](../PROTOCOL.md),
[the source-publication receipt](../PUBLISHED_SOURCE.json), and
[the independent audit](../verification/reproduction_audit.json).

## Experimental procedure and estimands

The design contains 24 repetition units, four observation laws, two added-noise
levels, history width 12, and forecast horizons 1, 4, and 12. The laws are
linear autoregression, nonlinear autoregression, Van der Pol, and Duffing.
The A/B/C/D episode allocation is 64/32/32/128, yielding 192 task suites. Each
task retains 42 forecast arrays, for 8,064 arrays in total. These arrays,
tasks, horizons, and dependent origins are not additional inferential units.

For each task and horizon, terminal mean squared error is divided by the
population variance of the retained A targets, floored at `1e-12`. The nonlinear
panel equally averages its three laws, two noise levels, and three horizons
within each repetition; the linear panel equally averages two noise levels and
three horizons. Reported panel SMSE is the mean of the 24 repetition-level
values. A percentage reduction is calculated from the two panel means,
`100 * (comparator_mean - candidate_mean) / comparator_mean`. It is not an
average of condition-specific percentage reductions. The definitions and
inference rules are given in the unchanged
[analysis protocol](../analysis_code/ANALYSIS_PROTOCOL.md).

Whole A episodes are assigned to four folds. Each common predictor forming an
out-of-fold residual uses the other 48 A episodes and independently selects its
settings on B. The deployed common predictor is fitted separately on all 64 A
episodes. Both residual-target procedures use that same deployment predictor,
the same B-fitted residual-mean bias, the same A correction sample, and the same
candidate feature representations. One correction uses out-of-fold residuals;
the other uses residuals of the full-A predictor evaluated on A. Their final
correction settings are selected separately on B and C.

The new common and correction searches use standardization, min-max scaling,
and standardized tanh scaling, with native caps 1, 2, 4, 8, and 12. Local
construction starts at `min(cap, ambient_dimension, local_reference_count - 1)`
and retains the original variance and range reductions. The inherited cap-20
candidate is equivalent to cap 12 at ambient dimension 12. C chooses correction
multipliers from `{0, 0.25, 0.5, 1}` and selects the expanded non-KAHM policy.
The independently selected RBF correction is a primary comparator. The matched
RBF diagnostic uses KAHM's selected scaler and cap while retaining its own
penalty and multiplier selection. It does not replace the independently
selected primary comparator. Complete selection frequencies are retained in
[the B-selection table](published_tables/B_selection_counts.csv) and
[the C-choice table](published_tables/C_choice_counts.csv).

## Primary comparisons

For a nonlinear comparison, the repetition-level margin gap is
`g = 0.97 * L_comparator - L_KAHM`; for linear noninferiority it is
`g = 1.05 * L_ridge - L_KAHM`. Each criterion is established when
`mean(g) - t_(1 - 0.05/4, 23) * sd(g)/sqrt(24)` is positive. The critical value
is 2.397875. These are approximate Bonferroni-adjusted, one-sided paired-t
bounds across four prespecified claims. The displayed lower bound is in
standardized margin-gap units; it is not a confidence interval for a risk ratio.

| Panel and comparator | KAHM SMSE | Comparator SMSE | Relative reduction | Required margin | Simultaneous lower gap | Criterion |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Nonlinear: identical common plus shared bias | 0.02857496 | 0.03146912 | 9.1968% | 3% superiority | 0.00177828 | Established |
| Nonlinear: independently selected cross-fit RBF | 0.02857496 | 0.02941815 | 2.8662% | 3% superiority | -0.00017209 | Not established |
| Nonlinear: expanded non-KAHM policy | 0.02857496 | 0.02718199 | -5.1246% | 3% superiority | -0.00246267 | Not established |
| Linear: raw ridge | 0.91880268 | 0.90852404 | -1.1314% | 5% noninferiority | 0.03194972 | Established |

Positive percentage reductions favor KAHM. Negative values indicate higher
KAHM error. All numerical entries come from
[the primary comparison table](published_tables/primary_contrasts.csv).

The independent RBF comparison illustrates the distinction between a lower
observed error and evidence for the specified minimum improvement. KAHM had
lower nonlinear repetition-level error than independently selected RBF in all
24 repetitions, but its mean reduction was below 3%, and the simultaneous lower
margin bound was negative. Against the expanded non-KAHM policy, KAHM had higher
error in every nonlinear repetition aggregate. The favorable shared-baseline
and linear decisions do not substitute for either failed primary superiority
claim. Repetition signs and unadjusted paired differences are supplied in
[the descriptive contrasts](published_tables/descriptive_contrasts.csv).

## Residual-target and geometry diagnostics

The in-sample KAHM correction had nonlinear SMSE 0.02755879, compared with
0.02857496 for the cross-fitted correction. Cross-fitted KAHM therefore had
3.6873% higher error relative to the in-sample procedure, with the same ordering
in all 24 nonlinear repetition aggregates. In-sample KAHM reduced error by
12.4259% against the shared baseline and by 3.9643% against independently
selected in-sample RBF. Cross-fitted KAHM reduced error by 4.1731% against the
matched cross-fit RBF diagnostic. These are secondary descriptive comparisons;
their intervals are unadjusted and do not establish additional simultaneous
primary claims.

| Descriptive contrast, left minus right | Mean SMSE difference | Unadjusted 95% paired interval | Relative comparison |
| --- | ---: | --- | --- |
| Cross-fit KAHM minus in-sample KAHM | 0.00101617 | [0.00087832, 0.00115403] | Cross-fit error 3.6873% higher |
| In-sample KAHM minus shared baseline | -0.00391033 | [-0.00404109, -0.00377958] | In-sample error 12.4259% lower |
| In-sample KAHM minus independent in-sample RBF | -0.00113762 | [-0.00129754, -0.00097771] | In-sample KAHM error 3.9643% lower |
| Cross-fit KAHM minus matched cross-fit RBF | -0.00124439 | [-0.00135008, -0.00113870] | Cross-fit KAHM error 4.1731% lower |

The intervals concern absolute standardized-loss differences. The percentage
columns use the corresponding observed comparator mean and are not ratio
intervals. Values are drawn from
[the descriptive contrast table](published_tables/descriptive_contrasts.csv).

An additional post hoc descriptive comparison of the saved panel means gives
in-sample KAHM error 1.3862% higher than the expanded non-KAHM policy
(0.02755879 versus 0.02718199). This arithmetic comparison is not a new
prespecified test and receives no additional primary inference. It indicates
that the favorable in-sample comparisons against the common baseline and RBF
do not establish dominance over the expanded policy. The underlying values
appear in [the panel summary](published_tables/panel_summary.csv).

The observed disadvantage of cross-fitted targets is compatible with a mismatch
between the residual-generating predictor and the predictor used at deployment.
The former is fitted on 48 episodes and the latter on 64; their independently
selected settings may also differ. Consequently, exclusion of an episode from
its residual-generating fit does not ensure that the resulting residual is an
unbiased error of the separately fitted deployment predictor. This is a
possible explanation, not an identified causal mechanism. Because each
correction procedure makes its own B/C selections, this experiment compares
complete procedures rather than isolating a universal effect of cross-fitting.

## Variation across the nonlinear conditions

The aggregate improvement over the shared baseline varies substantially across
the declared laws and noise conditions. The following values average horizons
within each task and then repetitions. They are descriptive condition summaries.

| Law | Added noise | Shared baseline SMSE | Cross-fit KAHM SMSE | In-sample KAHM SMSE | Expanded policy SMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear AR | 0 | 0.01977491 | 0.01979642 | 0.01978103 | 0.01984176 |
| Nonlinear AR | 0.1 | 0.03167890 | 0.03169705 | 0.03166453 | 0.03172935 |
| Van der Pol | 0 | 0.01971742 | 0.00905255 | 0.00757194 | 0.00399967 |
| Van der Pol | 0.1 | 0.05794773 | 0.05365489 | 0.05173976 | 0.05276262 |
| Duffing | 0 | 0.01376919 | 0.01231622 | 0.01109927 | 0.01132927 |
| Duffing | 0.1 | 0.04592658 | 0.04493264 | 0.04349620 | 0.04342927 |

Source: [condition summary](published_tables/condition_summary.csv).

For Van der Pol without added noise, cross-fitted KAHM reduced error by 54.0886%
relative to the shared baseline, while the expanded policy achieved lower
error still. In both nonlinear AR conditions the cross-fit correction slightly
increased error relative to the shared baseline. Thus the panel-level benefit
does not imply an improvement for every law or condition. The condition
comparisons do not alter the original aggregation or primary decisions.

## Numerical verification

The independent reproduction audit passed after verifying all 192 tasks,
8,064 retained D forecast arrays, and 24,192 horizon losses. It checked 21,770
selected local bases, independently replayed 1,152 selected common models and
1,040 distinct selected correction heads, recomputed all four primary bounds,
and checked 141 descriptive intervals. The receipt records 946,138 assertions
and explicit handling of two selected local bases with zero admissible rank.
These counts identify the verification scope; they are not extra observations
or research replications.

The completed reproduction recorded four ridge-SVD recovery attempts, all
successful under the fixed checked recovery rule. The verifier reported no
failed fallback in this completed execution, no recorded query-geometry
failure, and no coefficient- or query-bound violation. The declared
maximum-error screen passed in all tasks; its largest observed ratio to the
ridge maximum absolute error was 1.8873, below the screen threshold of 5.
This screen is a numerical diagnostic and does not establish a distribution-free
forecast guarantee. The earlier unhandled SVD failure and the historical
singleton-PCA verifier defect remain separate events and are not erased by
successful verification of this reproduction. Evidence is supplied in
[the full audit receipt](../verification/reproduction_audit.json) and the
retained analysis summary in the analysis archive.

The audit independently checks selected coefficients and native PCA subspaces,
all saved B-selection evidence and correction alpha paths, C calibration and
policy choices, and the joins from retained D arrays to reported losses. It
does not independently implement the inherited raw KAHM feature map, refit
every unselected candidate, or regenerate every terminal forecast from its
coefficients. Procedural seals and timestamps document this execution; they
do not constitute access-control isolation or authenticate the unavailable
original run.

## Interpretation and evidence distribution

On this fixed synthetic cohort and finite tuning menu, cross-fitted KAHM
provides a useful correction to the specified full-training common baseline.
The independently selected RBF and expanded-policy results constrain the
strength of the geometry claim, and the in-sample ablation supplies evidence
against assuming that the evaluated cross-fitting procedure improves terminal
forecasting. The experiment establishes neither a general preference for
in-sample residuals nor a universal limitation of cross-fitting. It provides
no natural-series adoption result, risk-certificate validation, or basis for
changing the original manuscript's broader theoretical claims.

The repository release includes primary, panel, descriptive-contrast, condition,
B-selection-count, and C-choice-count tables under `reporting/published_tables/`,
with the independent audit retained under `verification/`. One complete analysis
archive contains all analysis CSV/JSON files and verification receipts. Sixteen
standalone numerical-record archives preserve development and terminal arrays,
fitted states, forecasts, execution records, and stage provenance. Together,
the 17 verified ZIPs contain 3,311 files. The
[archive manifest](ARCHIVE_MANIFEST.json) indexes every retained numerical file
and archive by size and SHA-256. Final destination checks confirmed every ZIP's
size and hash against that unchanged manifest. Exact-byte reconstruction of one
initially inconsistent delivery artifact and the unresolved cause of that
discrepancy are documented in the separate
[delivery verification receipt](archive_delivery_verification.json); scientific
evidence and the original archive inventory were unchanged. Source code remains
available from the published repository. These records preserve the distinction
between the verified computational reproduction and the unavailable original
execution.
