# Evidence on geometry-induced forecasting representations

The [current manuscript](papers/soft_state_forecasting/) studies geometry-induced soft-state abstractions and their predictive enrichment. This guide separates the new representation ablation from the previously completed forecasting studies. The [research index](docs/RESEARCH_INDEX.md) maps their files; the [reproducibility guide](REPRODUCIBILITY.md) explains their computational evidence.

**Status: all 256 tasks, the frozen analysis, and the independent production audit are complete.** The scientific source is [commit `2b7d19b5`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2b7d19b5b47f46b5b21b28d24960688f7969d96f). The [audit receipt](extensions/soft_state_ablation_20261102/verification/independent_audit_summary.json) records verification coverage and limits. Three of four primary criteria are established; the KAHM-versus-RBF superiority margin is not.

A deterministic transformation cannot add information to its input, although it can make a predictive relationship easier for a restricted decoder to express. The completed studies support conditional feature and residual-correction value on the studied synthetic laws. A general KAHM-specific advantage over independently tuned alternatives or a complete forecasting selection policy has not been established.

## Current question: memberships, position, and reconstruction enrichment

The evaluated locality gates are nonnegative and sum to one within each reference partition. Their collection lies in a product of simplices. Gate-weighted principal coordinates retain within-region position; KAHM or matched RBF departures add signed observables. The enriched descriptor is not simplex-valued. Shared memberships do not make the two reconstruction maps identical, and distinct raw feature spans do not establish forecasting superiority.

The new [soft-state ablation](extensions/soft_state_ablation_20261102/) has completed its fixed budget of 32 fresh repetition units and 256 tasks under the same four simulation laws, with horizons 1, 4, and 12. Four arms compare memberships (G), memberships plus local position (P), P plus KAHM departures (K), and P plus matched RBF departures (R), predicting the same external observations. The frozen analysis is complete, and the independent production audit has passed. Primary P-selected shared-bank comparisons have their own four-criterion family; independently selected banks are secondary.

This design addresses a gap in the earlier studies: their common feature block already included local position, so it was not a membership-only control. The new study can distinguish the contribution of position from the incremental contribution of reconstruction. A successful readout would still not establish a sufficient Markov state or an autonomous soft-coordinate evolution model. The companion score-normalized KAHM abstraction uses a different membership map.

### Four-arm primary results

The shared-bank nonlinear mean SMSEs are G **0.09887034**, P **0.03045071**, K **0.02726601**, and R **0.02727398**. Relative gains use the ratio of the corresponding mean losses.

| Primary comparison | Panel | Relative gain | Prescribed criterion | Established |
| --- | --- | ---: | --- | --- |
| P versus G | Nonlinear | 69.201% | 3% superiority | Yes |
| K versus P | Nonlinear | 10.459% | 3% superiority | Yes |
| K versus R | Nonlinear | 0.0292% | 3% superiority | No |
| K versus P | Linear | -1.111% | 5% noninferiority | Yes |

Three of four primary criteria are established. Local position materially improves the memberships-only readout, and KAHM departures improve the common-coordinate readout under the prescribed procedure. The KAHM-specific superiority margin over matched RBF remains unestablished; the small mean difference is not an equivalence finding. Linear KAHM loss is 1.111% higher than P while satisfying the 5% noninferiority criterion. These are direct observed-output forecasts, not evidence of sufficient Markov states or autonomous soft-state dynamics.

The confidence family uses one-sided Bonferroni-adjusted paired Student bounds across 32 independent repetition aggregates, with 31 degrees of freedom; windows and horizons are not independent inference units. The 3% and 5% thresholds are inherited research criteria, not deployment-derived utility thresholds.

In the secondary independently selected context, nonlinear K and R means are 0.02675457 and 0.02675071. K improves P by 12.138% but is 0.0144% worse than R. These descriptive comparisons do not replace or reverse the primary family. The observed value of richer observables should therefore be separated from a unique advantage for the KAHM reconstruction rule. [Primary contrasts](extensions/soft_state_ablation_20261102/analysis/primary_contrasts.csv), [panel losses](extensions/soft_state_ablation_20261102/analysis/panel_summary.csv), and [secondary comparisons](extensions/soft_state_ablation_20261102/analysis/secondary_descriptive.csv) retain the complete numbers.

The exact-closure proposition concerns a specified deterministic evolution and a fixed observable space. Equal augmented raw spans have the same invariance status. This standard implication does not show that either fitted representation closes, and it does not equate norm-constrained or ridge-penalized predictions after a general coordinate change.

## Earlier dimension-selected experiment and follow-up evidence

**Feature distinctness depends on the reference geometry.** Direct transfer of normalized KAHM weights to reference targets is a ratio of two kernel-ridge predictions. Some symmetric reference sets give the same augmented raw feature span as matched RBF reconstruction. An irregular three-reference construction demonstrates that a KAHM reconstruction departure can also lie outside that span. Equality or inequality of raw spans alone does not determine the predictions obtained after feature normalization and regularization.

**A bounded residual predictor controls correction magnitude.** The principal experiment uses a radial bound on each feature block and a centered ridge residual predictor. The resulting bound applies uniformly to the correction under the stated assumptions. It does not bound the baseline prediction or guarantee a small forecasting error.

**Dimension selection changes predictive behavior.** The principal comparison selects one local cap per geometric procedure on validation data. KAHM improves the nonlinear aggregate by 3.26% relative to a fixed cap of two, while native cap twenty increases error by 8.39%. Three of the six primary criteria are established: improvements over common features and raw-input RBF, and linear-control noninferiority. The tuned RBF and boosting margins are not established, and the selected non-KAHM procedure has lower mean error. Matched-bank comparisons show that the dimension-selection gain accompanies improved common-feature loss and a weaker additional KAHM benefit. See the [principal findings](extensions/dimension_forecasting/FINDINGS.md).

**Preprocessing and residual-target studies sharpen the conditional result.** The preprocessing study retains its own four-comparison primary family; only linear-control noninferiority is established. Tanh/tanh has 6.486% higher nonlinear error than standard/standard. Jointly selected KAHM has 4.022% higher error than the selected non-KAHM policy and 0.281% higher error than independently selected bounded RBF. Its separate fixed-common experiment gives positive secondary evidence: centered KAHM corrections improve over their shared baseline by 10.780–14.934% and independently selected RBF by 2.372–4.140%. These smaller-common-fit comparisons are not pooled with the primary complete-procedure ranking. See the [preprocessing findings](extensions/preprocessing_dimension/FINDINGS.md).

**A full-training common predictor does not establish the complete KAHM advantage.** The cross-fit computational reproduction establishes two of four primary criteria. Cross-fitted KAHM improves over the shared common-plus-bias baseline by 9.1968%, while its 2.8662% gain over independently selected OOF RBF misses the prescribed 3% margin. Its error is 5.1246% higher than the expanded non-KAHM policy. Linear-control noninferiority is established. See the [cross-fit results](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md).

| Evidence layer | Preserved primary family | Main conclusion |
| --- | --- | --- |
| Principal dimension-selected study | Six comparisons; three established | Useful gains over common features and raw-input RBF; the complete comparison criterion is not established. |
| Preprocessing follow-up | Four comparisons; one established | Tanh and joint preprocessing selection do not establish a practical complete-procedure advantage; disjoint-episode fixed-common corrections are positive secondary evidence. |
| Cross-fit computational reproduction | Original four comparisons; two established | A shared-baseline gain survives with a full-training common predictor; the independent-RBF margin and policy superiority are not established. |
| Fresh policy augmentation | Four comparisons; two linear safeguards established | Adding KAHM gives a 0.816% nonlinear mean gain over P0, while neither 3% superiority criterion is established; 24 genuinely new repetitions. |
| Separate natural policy evaluation | Descriptive external benchmark-development evidence | On 48 reserved M4 training-series histories, adding KAHM increases overall policy SMSE by 2.273%; no independent-source inference. |

These are separate comparison families, not a pooled success count or a replacement of earlier failed claims. The cross-fit reproduction uses the original 24 simulation repetition units and contributes zero additional independent repetitions.

## Interpreting the residual-correction mechanism

For a fixed fitted baseline and correction, let `e = Y - f0(X)` and let the forecast add `eta * d(X)`. The squared-error change on the target distribution is exactly

\[
\mathbb{E}[(e-\eta d)^2]-\mathbb{E}[e^2]
=\eta^2\mathbb{E}[d^2]-2\eta\mathbb{E}[ed].
\]

A nonzero correction helps when its alignment with the predictable baseline residual is large enough relative to its squared magnitude. Bounding the correction controls magnitude; it does not guarantee favorable alignment. Reconstruction accuracy, an enlarged feature space, and an available zero multiplier therefore do not by themselves ensure better held-out prediction.

The secondary cross-fit comparison is particularly informative. Cross-fitted KAHM has 3.6873% higher error than in-sample KAHM. The in-sample procedure improves over the shared baseline by 12.4259% and independently selected in-sample RBF by 3.9643%. Its mean error nevertheless remains 1.3862% above the expanded non-KAHM policy, a post hoc descriptive comparison without a new primary test. Fold predictors use 48 episodes, whereas the deployed common predictor uses 64. Residual-target alignment is a plausible explanation for the ordering, but the experiment does not isolate training size or establish that cross-fitting is generally inferior.

The subsequent [policy-augmentation study](extensions/policy_augmentation_20261101/RESULTS.md) tests whether a selector can exploit a KAHM correction's complementary forecasts. It compares the unchanged P0 menu with P1, which adds the existing in-sample KAHM correction, and P0plus, which adds a new raw-history RBF correction with the same number of additional validation opportunities. On 24 fresh repetitions, P1 reduces nonlinear mean SMSE by 0.816% versus P0 and 0.658% versus P0plus. Neither prespecified 3% superiority criterion is established; both linear noninferiority criteria are established. Equal search counts do not match feature width or runtime. These results show a small observed incremental benefit under the studied laws, without establishing the declared practical margin.

The [separate natural evaluation](extensions/policy_augmentation_natural_20261101/FINDINGS.md) uses 24 Weekly and 24 Hourly M4 training histories outside the recorded prior pilot. P1 has 2.273% higher overall SMSE than P0 and 2.718% lower than P0plus. C selects KAHM on three series, producing one win and two losses versus P0; on the remaining 45, P1 equals P0. Both enlarged menus have higher overall error than P0. This chronological evaluation uses reserved training-file segments and no official Test data. The natural menu adds causal seasonal-naive and drift controls and omits the episode cross-fit family. Its scores remain descriptive and are not pooled with the synthetic inference.

## Supporting studies and interpretation

| Study | Scientific question | Main interpretation |
| --- | --- | --- |
| [Controlled representations](extensions/predictive_geometry/README.md) | Which features retain predictive position, and can a finite decoder use them? | Local coordinates recover information lost by reconstruction scores; benefits depend on the generating law and comparator. |
| [Natural forecasting](extensions/natural_forecasting/README.md) | Does the representation improve prediction on reserved data sources? | Raw-plus-KAHM has higher aggregate error than raw-plus-RBF on the four confirmation families. |
| [Scaling sensitivity](extensions/scaling_sensitivity/README.md) | Does a change of input scale alter the comparison? | The unfavorable aggregate ordering persists; the analysis reuses previously evaluated data. |
| [Additive corrections](extensions/forecasting_utility/README.md) | Does retaining a raw-input baseline yield useful geometric corrections? | Some synthetic improvements occur, but fit failures prevent a complete natural-data aggregate. |
| [Target-adapted geometry](extensions/target_geometry/README.md) | Do a supervised metric and donor residuals improve KAHM prediction? | Selected gains over ridge do not establish the incremental value of the learned metric or a KAHM-specific advantage. |
| [Direct reconstruction training](extensions/direct_reconstruction/README.md) | Does optimizing local bandwidths for forecasting outperform equally trained controls? | Two of 18 adjusted synthetic comparisons favor KAHM, six favor a comparator, and ten remain unresolved. No favorable comparison is against an equally trained non-KAHM local control. |
| [Online adaptation](extensions/adaptive_development/REPRODUCING.md) | Do maintained geometry and residual predictors justify their cost? | Both capacity tiers fail the prescribed accuracy criteria; severe extrapolation remains after a numerical solver repair. |
| [Historical fixed-cap reconstruction departures](extensions/reconstruction_mechanism/README.md) | Do distinct, bounded reconstruction features improve a matched forecasting procedure? | The common-feature comparison and linear-control noninferiority criterion are satisfied; the four other useful-improvement criteria are not established. |

The studies use different procedures, populations, normalization conventions, and selection budgets. Their numerical scores should be interpreted within the corresponding protocol, and their observations are not pooled as independent evidence for a single test. The current paper includes the policy-augmentation and natural-series evidence. The older document edition under `extensions/tmlr_integration/` predates those evaluations and remains available as a historical release.

## Statistical and computational scope

The earlier dimension study has 24 independent repetition aggregates. Each repetition combines four simulation laws and two paired noise levels; the primary nonlinear result averages the three nonlinear laws and both noise levels equally. Forecast windows, horizons, and noise conditions within a repetition are dependent. Its six simultaneous lower bounds use a Bonferroni-adjusted paired Student t calculation with 23 degrees of freedom. The preprocessing and cross-fit studies each retain their own four-comparison family and repetition-level inference. Failure to establish a 3% margin does not establish equality between methods.

The nonlinear laws were known during method development. Fresh random realizations provide independent sampling under those laws, not independent application domains. The dimension and preprocessing protocols were frozen locally before production; this was not public preregistration. The cross-fit reproduction has a different chronology: earlier reported outcomes were known, the original completed records were unavailable, and the same seeds were regenerated under recovered scientific code. New seals and the independent audit verify that retained execution without authenticating the missing original bytes or historical seals. It is not an independent confirmation sample.

Audit coverage also differs. The dimension-study audit covers all 192 tasks and its 27 tables. The preprocessing audit reconstructs all aggregate arithmetic and the four primary comparisons, while its raw forecast-to-loss replay covers the prespecified eight repetition-zero tasks. The cross-fit audit checks all 192 tasks, 8,064 retained forecast arrays and 24,192 horizon losses. Each receipt defines its own limits; computational consistency is not external scientific replication.

Each study retains its original decision rule. Passing a baseline comparison, noninferiority control or stability screen does not replace a failed superiority claim. The conclusions concern the evaluated procedures and budgets and do not rule out different representations, decoders or forecasting populations.

## Access and manuscript provenance

- [Current paper and editable source](papers/soft_state_forecasting/)
- [New G/P/K/R study and execution status](extensions/soft_state_ablation_20261102/)
- [Historical coordinated manuscript edition](extensions/tmlr_integration/README.md)
- [Mathematical definitions and derivations](extensions/reconstruction_mechanism/MATHEMATICAL_BACKGROUND.md)
- [Complete principal results](extensions/dimension_forecasting/FINDINGS.md)
- [Principal evidence restoration, analysis, and replay](extensions/dimension_forecasting/README.md)
- [Preprocessing results and numerical package](extensions/preprocessing_dimension/RESULTS_PACKAGE.md)
- [Cross-fit results and exact evidence links](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md)
- [Fresh policy-augmentation results and primary criteria](extensions/policy_augmentation_20261101/RESULTS.md)
- [Separate natural policy findings and selected cases](extensions/policy_augmentation_natural_20261101/FINDINGS.md)
- [Complete policy-augmentation numerical archive index](extensions/policy_augmentation_20261101/reporting/ARCHIVE_INDEX.json)
- [Historical coordinated document release](extensions/tmlr_integration/review/COORDINATED_MANUSCRIPT_RELEASE.json)
- [Earlier Koopman result provenance](reproduction/reported_results.tsv)

The canonical paper directory holds the current main manuscript, supplementary studies, individual editable files and build instructions. The revision places the completed four-arm analysis alongside retained earlier findings, whose full protocols and comparison families remain in their study records and supplementary material. Generated displays have separate study namespaces and retain their evidence identities. The new ablation has its own complete cohort and analysis; its independent production audit has passed. Reorganizing the paper does not modify the frozen procedures or outcomes of previous experiments.

The dimension-study records and historical fixed-cap saved-forecast archive have distinct scopes, as specified in their study guides. The editable source preserves required author-identifying license attribution and is distributed separately from the anonymous review files. The documents' TMLR formatting does not claim submission or acceptance.

KAHM is the local reconstruction used in the forecasting representations. KAHKM is the earlier Koopman architecture. The distinction is maintained throughout the result documentation.

The completed four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from the frozen scientific source and manuscript edition.
