# Evidence on KAHM representations for time-series forecasting

The research examines whether local reconstruction gives a finite forecasting decoder useful functions of the observed history. The distinction between information, function representation, and fitted prediction is central: a deterministic transformation cannot add information to its input, although it can make a predictive relationship easier for a restricted model to express.

## Principal findings

**Feature distinctness depends on the reference geometry.** Direct transfer of normalized KAHM weights to reference targets is a ratio of two kernel-ridge predictions. Some symmetric reference sets give the same augmented raw feature span as matched RBF reconstruction. An irregular three-reference construction demonstrates that a KAHM reconstruction departure can also lie outside that span. Equality or inequality of raw spans alone does not determine the predictions obtained after feature normalization and regularization.

**A bounded residual predictor controls correction magnitude.** The principal experiment uses a radial bound on each feature block and a centered ridge residual predictor. The resulting bound applies uniformly to the correction under the stated assumptions. It does not bound the baseline prediction or guarantee a small forecasting error.

**The empirical benefit is conditional.** In the principal experiment, KAHM reduces aggregate nonlinear error by 12.58% relative to the common gate and local-coordinate features. Its gains over matched RBF departures and raw-history RBF features do not establish the prescribed 3% improvement under the simultaneous analysis. Boosted trees and the validation-selected non-KAHM procedure perform better in aggregate. The evidence therefore supports the usefulness of reconstruction features in one comparison without establishing a distinctive practical KAHM advantage.

## Studies and interpretation

| Study | Scientific question | Main interpretation |
| --- | --- | --- |
| [Controlled representations](extensions/predictive_geometry/README.md) | Which features retain predictive position, and can a finite decoder use them? | Local coordinates recover information lost by reconstruction scores; benefits depend on the generating law and comparator. |
| [Natural forecasting](extensions/natural_forecasting/README.md) | Does the representation improve prediction on reserved data sources? | Raw-plus-KAHM has higher aggregate error than raw-plus-RBF on the four confirmation families. |
| [Scaling sensitivity](extensions/scaling_sensitivity/README.md) | Does a change of input scale alter the comparison? | The unfavorable aggregate ordering persists; the analysis reuses previously evaluated data. |
| [Additive corrections](extensions/forecasting_utility/README.md) | Does retaining a raw-input baseline yield useful geometric corrections? | Some synthetic improvements occur, but fit failures prevent a complete natural-data aggregate. |
| [Target-adapted geometry](extensions/target_geometry/README.md) | Do a supervised metric and donor residuals improve KAHM prediction? | Selected gains over ridge do not establish the incremental value of the learned metric or a KAHM-specific advantage. |
| [Direct reconstruction training](extensions/direct_reconstruction/README.md) | Does optimizing local bandwidths for forecasting outperform equally trained controls? | Two of 18 adjusted synthetic comparisons favor KAHM, six favor a comparator, and ten remain unresolved. No favorable comparison is against an equally trained non-KAHM local control. |
| [Online adaptation](extensions/adaptive_development/REPRODUCING.md) | Do maintained geometry and residual predictors justify their cost? | Both capacity tiers fail the prescribed accuracy criteria; severe extrapolation remains after a numerical solver repair. |
| [Reconstruction departures](extensions/reconstruction_mechanism/README.md) | Do distinct, bounded reconstruction features improve a matched forecasting procedure? | The common-feature comparison and linear-control noninferiority criterion are satisfied; the four other useful-improvement criteria are not established. |

The studies use different procedures, populations, normalization conventions, and selection budgets. Their numerical scores should be interpreted within the corresponding protocol, and their observations are not pooled as independent evidence for a single test.

## Statistical and computational scope

The principal study has 24 independent repetition aggregates. Each repetition combines four simulation laws and two paired noise levels; the primary nonlinear result averages the three nonlinear laws and both noise levels equally. Forecast windows, horizons, and noise conditions within a repetition are dependent. The six simultaneous lower bounds use a Bonferroni-adjusted paired Student t calculation with 23 degrees of freedom. Failure to establish a 3% margin does not establish equality between methods.

The nonlinear laws were known during method development. Fresh random realizations provide independent sampling under those laws, not independent application domains. The locally recorded source and input checkpoint preceded held-out prediction, but was not a public preregistration. Selected-model replay and independent-code checks support computational consistency within the study; they do not constitute external replication.

The predefined study decision required every accuracy condition and stability check to be satisfied. That combined criterion was not met, and no further experiment was used to replace the result. This conclusion applies to the evaluated procedures and budgets. It does not rule out different geometric representations, decoders, or forecasting settings.

## Access

- [Focused paper, Supporting Studies, and editable source](extensions/tmlr_integration/README.md)
- [Mathematical definitions and derivations](extensions/reconstruction_mechanism/MATHEMATICAL_BACKGROUND.md)
- [Complete principal results](extensions/reconstruction_mechanism/FINDINGS.md)
- [Evidence restoration, analysis, and replay](extensions/reconstruction_mechanism/README.md)
- [Earlier Koopman result provenance](reproduction/reported_results.tsv)

The focused manuscript contains the core proofs and principal experimental protocol. The separate, self-contained *Supporting Studies* document retains the complementary methods, results, and technical appendices. The review supplement combines that PDF with the unchanged principal saved-forecast archive. The editable source preserves required author-identifying license attribution and is distributed separately from the anonymous review files.

KAHM is the local reconstruction used in the forecasting representations. KAHKM is the earlier Koopman architecture. The distinction is maintained throughout the result documentation.
