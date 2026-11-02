# Current forecasting manuscript and representation study

**Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines** is the current manuscript. Its canonical directory is [papers/soft_state_forecasting/](papers/soft_state_forecasting/), containing the paper, companion supplementary material, editable source, and document build instructions as the revision is assembled.

**Status: all 256 tasks, the frozen analysis, and the independent production audit are complete.** The scientific source is [commit `2b7d19b5`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2b7d19b5b47f46b5b21b28d24960688f7969d96f). The [audit receipt](extensions/soft_state_ablation_20261102/verification/independent_audit_summary.json) records verification coverage and limits. Three of four primary criteria are established; the KAHM-versus-RBF superiority margin is not.

- [Main manuscript PDF](papers/soft_state_forecasting/main.pdf)
- [Supplementary studies PDF](papers/soft_state_forecasting/supplement.pdf)
- [Individual editable source files](papers/soft_state_forecasting/)
- [Current build instructions](papers/soft_state_forecasting/README_BUILD.md)
- [Representation-ablation study](extensions/soft_state_ablation_20261102/) and [primary results](extensions/soft_state_ablation_20261102/analysis/primary_contrasts.csv)
- [Research index](docs/RESEARCH_INDEX.md)
- [Findings and limitations](TIME_SERIES_RESEARCH.md)
- [Scientific reproduction guide](REPRODUCIBILITY.md)

## Scope of the contribution

The paper organizes a forecasting representation into locality memberships, within-region principal coordinates, and reconstruction departures. Memberships sum to one within each reference partition; their collection lies in a product of simplices. KAHM and matched positive-kernel RBF reconstructions share those memberships. The signed enriched descriptor is not itself simplex-valued and is not assumed to be a sufficient or Markov state.

The mathematical results distinguish normalized target transfer, equality or distinctness of augmented raw feature spans, bounded correction magnitude, and conditional exact dynamical closure. The connection between an invariant observable span and exact linear coordinate evolution is a standard function-space implication, applied to the feature-span comparisons here. It does not demonstrate that the fitted forecasting representation closes or that KAHM outperforms alternatives.

The experiments predict common future observations with direct readouts. They do not evaluate autonomous soft-coordinate propagation. The companion reconstruction-score abstraction is a different representation, so its successor-coordinate results do not validate this pipeline's dynamics.

## New study and retained evidence

The G/P/K/R ablation under `extensions/soft_state_ablation_20261102/` has completed 32 fresh repetitions and 256 tasks; the frozen analysis is complete and the independent production audit has passed. Primary comparisons use P-selected shared geometry. P improves nonlinear loss over G by 69.201%, and K improves P by 10.459%, establishing their 3% margins. K improves matched RBF by only 0.0292%, leaving the KAHM-specific 3% margin unestablished. Linear K is 1.111% worse than P but satisfies the 5% noninferiority criterion. Three of four primary criteria are established. Independent-cap comparisons remain secondary; their nonlinear K loss is 0.0144% higher than R. These results support predictive enrichment under the specified readouts, without establishing KAHM-specific superiority or autonomous soft-state dynamics.

The revised paper retains the dimension-selected, preprocessing, residual-target, and policy-augmentation findings with their original scopes. In particular, the 24-repetition policy study does not establish either prescribed nonlinear superiority criterion, and the separate 48-series natural panel is unfavorable relative to the base policy. Reframing the representation does not convert these outcomes into successful claims or pool earlier observations with the new cohort.

## Editions and review materials

The previous coordinated paper, **Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans, Bounded Corrections, and Matched Evaluation**, remains under [extensions/tmlr_integration/](extensions/tmlr_integration/). Its [release record](extensions/tmlr_integration/review/COORDINATED_MANUSCRIPT_RELEASE.json) identifies that earlier document edition. Frozen scientific procedures and evidence stay in their original study directories.

The current PDFs use TMLR review formatting and disclose AI assistance. Editable source retains license attribution and should be checked separately before an anonymous upload. Compiling the manuscript, reconstructing reported statistics, replaying saved models, and refitting the scientific procedure have different evidential scopes.

The [TMLR author guidelines](https://jmlr.org/tmlr/author-guide.html) describe submission formatting and review requirements. These materials do not assert journal submission, acceptance, or endorsement.

The completed four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from the frozen scientific source and manuscript edition.
