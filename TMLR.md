# Current forecasting manuscript and representation study

**Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines** is the current manuscript. Its canonical directory is [papers/soft_state_forecasting/](papers/soft_state_forecasting/), containing the paper, companion supplementary material, editable source, and document build instructions as the revision is assembled.

**Joint-selection status: all 256 production tasks, the frozen analysis, and the independent production audit are complete.** The scientific source is [commit `8e025027`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/8e0250279074aefa39079fa33205d3c74f835848), on `research/kahm-soft-state-joint-selection-2026-11-02`. The [results report](extensions/soft_state_joint_selection_20261102/RESULTS.md) records four of six criteria established by the analysis and independent audit. The KAHM-versus-RBF and joint-versus-standard 3% superiority margins remain unestablished. The [build guide](papers/soft_state_forecasting/README_BUILD.md) records document build status.

- [Main manuscript PDF](papers/soft_state_forecasting/main.pdf)
- [Supplementary studies PDF](papers/soft_state_forecasting/supplement.pdf)
- [Individual editable source files](papers/soft_state_forecasting/)
- [Current build instructions](papers/soft_state_forecasting/README_BUILD.md)
- [Joint-selection study](extensions/soft_state_joint_selection_20261102/) and [six primary results](extensions/soft_state_joint_selection_20261102/analysis/primary_contrasts.csv)
- [Passed independent joint-selection audit](extensions/soft_state_joint_selection_20261102/verification/independent_audit_summary.json)
- [Preserved standard-only representation ablation](extensions/soft_state_ablation_20261102/RESULTS.md)
- [Research index](docs/RESEARCH_INDEX.md)
- [Findings and limitations](TIME_SERIES_RESEARCH.md)
- [Scientific reproduction guide](REPRODUCIBILITY.md)

## Scope of the contribution

The paper organizes a forecasting representation into locality memberships, within-region principal coordinates, and reconstruction departures. Memberships sum to one within each reference partition; their collection lies in a product of simplices. KAHM and matched positive-kernel RBF reconstructions share those memberships. The signed enriched descriptor is not itself simplex-valued and is not assumed to be a sufficient or Markov state.

The mathematical results distinguish normalized target transfer, equality or distinctness of augmented raw feature spans, bounded correction magnitude, and conditional exact dynamical closure. The connection between an invariant observable span and exact linear coordinate evolution is a standard function-space implication, applied to the feature-span comparisons here. It does not demonstrate that the fitted forecasting representation closes or that KAHM outperforms alternatives.

The experiments predict common future observations with direct readouts. They do not evaluate autonomous soft-coordinate propagation. The companion reconstruction-score abstraction is a different representation, so its successor-coordinate results do not validate this pipeline's dynamics.

## New study and retained evidence

The new study under `extensions/soft_state_joint_selection_20261102/` has completed 32 fresh repetitions and 256 tasks. It searches three A-fitted input scalers and five native caps using B, with standard-only controls reusing the same candidate fits on the same cohort. Four shared-bank structural contrasts and two independent joint-versus-standard K contrasts form one six-comparison family. The independent production audit passed 7,422,883 checks.

In the frozen analysis, shared P improves nonlinear mean loss over G by 68.69%, and K improves P by 8.86%; both establish their 3% margins. Shared K has 0.9186% higher loss than matched R, leaving its superiority margin unestablished. Independent joint K has only a 0.0727% nonlinear gain over standard-only K, also leaving the prescribed 3% improvement unestablished. Shared linear K has 1.26% higher loss than P, and independent joint K has 0.2491% higher linear loss than standard-only K; both satisfy 5% noninferiority. Thus four of six criteria are established and independently verified. Other independently tuned comparisons remain descriptive, including nonlinear K/R means of 0.02621031/0.02599175. See the [complete results](extensions/soft_state_joint_selection_20261102/RESULTS.md).

The predecessor standard-only G/P/K/R ablation remains unchanged, with its passed audit and separate four-comparison family. It established three of four criteria: nonlinear P/G and K/P gains of 69.201% and 10.459%, plus linear noninferiority. Its 0.0292% K/R gain did not establish a 3% margin. These earlier means are not the same-cohort standard-only controls for the new tuning question.

The revised paper retains the dimension-selected, preprocessing, residual-target, and policy-augmentation findings with their original scopes. In particular, the 24-repetition policy study does not establish either prescribed nonlinear superiority criterion, and the separate 48-series natural panel is unfavorable relative to the base policy. Reframing the representation does not convert these outcomes into successful claims or pool earlier observations with the new cohort.

## Editions and review materials

The previous coordinated paper, **Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans, Bounded Corrections, and Matched Evaluation**, remains under [extensions/tmlr_integration/](extensions/tmlr_integration/). Its [release record](extensions/tmlr_integration/review/COORDINATED_MANUSCRIPT_RELEASE.json) identifies that earlier document edition. Frozen scientific procedures and evidence stay in their original study directories.

The current PDFs use TMLR review formatting and disclose AI assistance. Editable source retains license attribution and should be checked separately before an anonymous upload. Compiling the manuscript, reconstructing reported statistics, replaying saved models, and refitting the scientific procedure have different evidential scopes.

The [TMLR author guidelines](https://jmlr.org/tmlr/author-guide.html) describe submission formatting and review requirements. These materials do not assert journal submission, acceptance, or endorsement.

The predecessor four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from that study's frozen scientific source and manuscript edition.
