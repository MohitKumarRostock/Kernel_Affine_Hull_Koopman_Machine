# KAHM forecasting manuscript and scientific evidence

**Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans,
Bounded Corrections, and Matched Evaluation** studies local reconstruction
features and their role in a finite forecasting procedure.

- [Main manuscript](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Supporting Studies](extensions/tmlr_integration/review/KAHM_Supplementary_Studies.pdf)
- [Editable source](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Build and evidence guide](extensions/tmlr_integration/README.md)
- [Principal experimental findings](extensions/dimension_forecasting/FINDINGS.md)
- [Reproduction instructions](extensions/dimension_forecasting/README.md)

## Claims and evidence

The paper develops identities for normalized KAHM target transfer, sufficient
conditions for feature-span equivalence with matched RBF reconstruction, an
explicit distinctness example, and a uniform correction-magnitude bound. These
mathematical results do not imply an accuracy advantage.

The principal experiment compares ten complete procedures in 192 task suites.
Local caps are selected independently for the geometric procedures on
validation data. KAHM establishes the declared nonlinear margins over common
features and raw-input RBF and the linear-control noninferiority criterion.
The margins over tuned RBF departures, boosted trees and the selected non-KAHM
procedure are not established. The latter has lower mean error. Secondary
matched-bank comparisons account for the dimension-selection gain mainly
through improved common-feature predictor loss.

The complete source and analysis are in `extensions/dimension_forecasting`.
Its frozen protocol, 27 analysis tables and independent numerical audit have
separate identities from the compiled manuscript. The historical fixed-cap
saved-forecast package and its original verifier remain available under their
original experimental scope; they do not reproduce the new principal results.

## Presentation and reproducibility

The forecasting PDFs retain the official TMLR style in anonymous review mode
and an explicit disclosure of AI assistance. The editable source retains
licenses and attribution. The accompanying document guide describes the
current PDFs, source dependencies and numerical archives. Local source freezes
are not public preregistrations, and internal independent-code audits are not
external replications.

The [TMLR author guidelines](https://jmlr.org/tmlr/author-guide.html) describe
submission formatting and review requirements. These research materials do
not assert journal submission, acceptance or endorsement.
