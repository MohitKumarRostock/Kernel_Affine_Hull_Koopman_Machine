# KAHM forecasting: feature spans, bounded corrections, and dimension selection

**Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans,
Bounded Corrections, and Matched Evaluation** studies which functions local
reconstructions add to a forecasting representation, how large a fitted
residual correction can become, and whether the resulting complete forecasting
procedure improves on competing procedures.

- [Main manuscript](review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Supporting Studies](review/KAHM_Supplementary_Studies.pdf)
- [Editable manuscript sources](review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Main LaTeX entry point](manuscript/main.tex) and [supporting-study entry point](manuscript/supplement.tex)
- [Principal forecasting experiment](../dimension_forecasting/README.md)
- [Complete scientific findings](../dimension_forecasting/FINDINGS.md)

## Scientific scope

The main manuscript develops conditional feature-span equivalence and
distinctness results for KAHM and RBF reconstructions, together with a uniform
bound on the ridge residual correction including its free intercept. Its
principal experiment compares ten complete procedures on four numerical laws,
two added-noise levels and 24 independent repetitions. The local dimension cap
is selected on validation data separately for each geometric procedure.
Calibration subsequently chooses blending coefficients and one complete
non-KAHM procedure; terminal forecasts are fixed before target release.

Validation-selected KAHM reduces mean nonlinear standardized MSE by 3.26%
relative to the historical cap-two construction, while native cap twenty
increases it by 8.39%. Three of the six primary margin criteria are established:
the nonlinear comparisons with common features and raw-input RBF, and
noninferiority to ridge on the linear control. The full comparison criterion
is not established. The selected non-KAHM procedure has lower aggregate error,
and the KAHM advantage over an independently tuned RBF departure is small.

Separate matched-bank comparisons account for the dimension-selection gain
through improved common-feature predictor loss and a weaker additional KAHM
departure benefit. Local dimension, stacked-basis rank and predictive accuracy
are distinct quantities. A fixed cap is not an accuracy guarantee.

The self-contained *Supporting Studies* document retains the complementary
controlled, natural-data, scaling, additive-correction, target-geometry,
direct-training and adaptation studies. Their historical construction choices,
sampling units and uncertainty statements remain specific to those studies.
Their outcomes are not pooled with the principal experiment.

## Compile the documents

Install TeX Live with `latexmk`, pdfLaTeX and BibTeX. From `manuscript/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

All display assets, bibliography files and licensed TMLR style dependencies
needed for compilation are included. Compilation does not fit forecasting
models or require the full numerical-record archives. Both PDFs retain
anonymous TMLR review mode. The editable source retains scientific licenses
and attribution; it is not itself an anonymized submission package.
The [manuscript build guide](manuscript/README_BUILD.md) describes the exact
source and evidence scope.

## Reproduce the principal results

The [dimension-study guide](../dimension_forecasting/README.md) supplies
restoration, fitting, prediction, analysis and independent-audit commands.
The complete numerical records are distributed as six disjoint archives,
`KAHM_Forecasting_Dimension_Records_01.zip` through `_06.zip`. They restore
under `extensions/dimension_forecasting/`. Their
[archive manifest](../dimension_forecasting/documentation/KAHM_Forecasting_Dimension_Records_manifest.json)
records every member and checksum. All 27 compact analysis tables and the
independent final audit accompany the source extension.

The primary inference uses 24 independent repetition aggregates and six
approximate simultaneous paired-t bounds. It does not treat overlapping
forecast origins as independent observations and does not supply the
bounded-loss certificates studied in the separate risk manuscript. Fixed-cap
sensitivity results and matched-bank mechanism comparisons remain secondary.

## Historical evidence and package utilities

The [fixed-cap reconstruction study](../reconstruction_mechanism/README.md)
and its saved-forecast archive retain the original experiment and verification
scope. `package_tmlr_review.py`, `verify_saved_forecasts.py` and the original
`package_submission.py` refer to that fixed-cap saved-forecast schema. Their
192-task archive and eight recomputed tables must not be used as evidence for
the dimension-selected principal experiment.

Similarly, `build_mechanism_assets.py` regenerates historical reconstruction
assets. The dimension-selected manuscript uses its own display namespace and
the source and table identities recorded with that study. Historical result
files and verification receipts retain their original meanings.

Scientific protocols, numerical records and manuscript editions have separate
identities. A successful document build verifies compilation; the independent
study audits verify the recorded numerical evidence. The repository makes no
claim of journal submission or acceptance.
