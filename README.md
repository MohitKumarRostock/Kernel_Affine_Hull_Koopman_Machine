# KAHM representations for time-series forecasting

This repository contains the manuscript, implementation, and experimental evidence for **Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans, Bounded Corrections, and Matched Evaluation**. The study examines how a Kernel Affine Hull Machine (KAHM) reconstruction contributes features to a finite forecasting model, when those features differ from simpler alternatives, and whether the resulting procedure improves prediction.

## Manuscript and principal findings

- [Manuscript PDF](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Supplementary studies](extensions/tmlr_integration/review/KAHM_Supplementary_Studies.pdf)
- [Editable LaTeX source](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Mathematical background](extensions/reconstruction_mechanism/MATHEMATICAL_BACKGROUND.md)
- [Experimental findings](extensions/dimension_forecasting/FINDINGS.md)
- [Reproduction instructions](extensions/dimension_forecasting/README.md)
- [Guide to the studies](TIME_SERIES_RESEARCH.md)

The theoretical analysis distinguishes three properties. A deterministic reconstruction cannot add information to an observed history, but it can change the functions available to a restricted decoder. The KAHM reconstruction sometimes gives the same augmented feature span as a matched radial-basis-function (RBF) reconstruction; an explicit irregular reference set also shows that the spans can differ. Separately, radial feature normalization and a ridge residual predictor provide a uniform bound on the magnitude of the forecast correction. This bound does not guarantee accuracy.

The principal experiment compares ten procedures in 192 task suites: four simulation laws, two added-noise levels, and 24 independent repetitions. Each geometric procedure selects its own local dimension cap on validation data. Bounded KAHM reduces mean nonlinear standardized MSE by **9.01% relative to common local-coordinate features** and **8.01% relative to raw-input RBF regression**, establishing the declared 3% margins. Its **0.72% gain over the independently tuned RBF departure** and **2.39% gain over boosted trees** do not establish that margin. KAHM has **4.08% higher error than the selected non-KAHM procedure**. Linear-control noninferiority and all maximum-error screens are satisfied; three of six primary criteria are established.

These results support a conditional contribution of reconstruction features to a finite decoder. They do not establish a distinctive practical forecasting advantage for KAHM over the complete set of evaluated alternatives. The inferential units are the 24 repetition aggregates. The 9,584,640 individual forecast values are dependent measurements within those repetitions.

The main paper contains the central definitions and theoretical statements, complete proofs, the matched forecasting experiment, and its full protocol. A concise synthesis keeps the material supporting findings and limitations visible. The companion supplementary studies provide the complete controlled, natural-data, scaling, additive, learned-geometry, direct-training, and adaptive investigations with their own notation and comparison families.

## Reproduce the principal experiment

Restore all six `KAHM_Forecasting_Dimension_Records_*.zip` archives at the
repository root, as specified in the [study guide](extensions/dimension_forecasting/README.md).
The source extension contains the frozen protocol, complete analysis tables
and independent-audit records. To verify the source and replay the analysis:

```bash
python -m pip install -r extensions/dimension_forecasting/requirements.txt
cd extensions/dimension_forecasting
python code/study_seals.py verify --root .
python analysis_code/analyze_forecasting.py --root . --output verification/reanalysis
```

Use a new analysis output directory. Full fitting and target-release replay
commands are supplied in the study guide. The independent audit checks all
192 tasks, 31,104 forecast arrays, 18,816 local PCA bases and 27 analysis tables.
This checks the recorded computation; it is not an external application replication.
Protocols were frozen locally before production, without public preregistration.
The historical [fixed-cap experiment](extensions/reconstruction_mechanism/README.md)
retains its original evidence and saved-forecast schema.

## Cross-fitted residual correction with a full-training common predictor

An additional synthetic study evaluates episode-wise out-of-fold residuals
around a common predictor fitted on all designated training episodes. Its
computational reproduction establishes two of the four original primary
criteria. Cross-fitted KAHM reduces nonlinear standardized MSE by **9.1968%**
relative to the identical common-plus-bias baseline. Its **2.8662%** reduction
against independently selected cross-fitted RBF does not establish the
prescribed 3% margin, and its error is **5.1246% higher** than the expanded
non-KAHM policy. Linear performance satisfies the prescribed 5% noninferiority
criterion. In the secondary residual-target comparison, cross-fitted KAHM has
3.6873% higher error than the in-sample correction.

The independent audit covers all 192 task suites and 8,064 retained forecast
arrays. The reproduction reuses the original 24 simulation repetition units
and adds no independent repetitions. It verifies its newly retained execution
without authenticating the unavailable original completed-run records. The
numerical package contains one analysis ZIP and sixteen standalone records
ZIPs, with 3,311 retained files and an exact size/SHA-256 index.

[Results](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md)
· [Protocol](extensions/crossfit_residual_reproduction_20261101/PROTOCOL.md)
· [Primary comparisons](extensions/crossfit_residual_reproduction_20261101/reporting/published_tables/primary_contrasts.csv)
· [Independent audit](extensions/crossfit_residual_reproduction_20261101/verification/reproduction_audit.json)
· [Archive manifest](extensions/crossfit_residual_reproduction_20261101/reporting/ARCHIVE_MANIFEST.json)
· [Archive delivery verification](extensions/crossfit_residual_reproduction_20261101/reporting/archive_delivery_verification.json)

## Repository organization

| Location | Contents |
| --- | --- |
| `extensions/tmlr_integration/` | Main paper and supporting-study source, generated displays, compiled PDFs, and submission-package instructions |
| `extensions/dimension_forecasting/` | Principal dimension-selected comparison, source and phase records, complete analysis, and independent audit |
| `extensions/crossfit_residual_reproduction_20261101/` | Cross-fitted and in-sample residual correction study, same-cohort computational reproduction, audited results, and complete numerical archive index |
| `extensions/reconstruction_mechanism/` | Historical fixed-cap comparison, algebraic examples, saved evidence, and verification tools |
| `extensions/predictive_geometry/` | Controlled representation and decoder experiments |
| `extensions/natural_forecasting/` | Natural-data evaluation and dataset provenance |
| `extensions/scaling_sensitivity/` | Sensitivity to input scaling on previously evaluated data |
| `extensions/forecasting_utility/` | Additive corrections to a retained raw-input baseline |
| `extensions/target_geometry/` | Supervised metric learning and donor-residual experiments |
| `extensions/direct_reconstruction/` | Forecast-trained local bandwidths and gate temperatures |
| `extensions/adaptive_development/` | Online adaptation with bounded reference and replay storage |
| `reproduction/` | Provenance and result mapping for the earlier Koopman architecture |

KAHM denotes the local geometric reconstruction. KAHKM denotes the earlier Kernel Affine Hull Koopman Machine architecture. Their prediction targets and decoders differ, so results for one construction do not automatically transfer to the other. The [reproducibility guide](REPRODUCIBILITY.md) separates their computational environments and evidence.

## Build the manuscript

The LaTeX source includes the bibliography, official TMLR style, figures, and tables. With TeX Live, `latexmk`, and Latin Modern fonts installed:

```bash
cd extensions/tmlr_integration/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

Compilation does not fit forecasting models. Both PDFs use anonymous TMLR review formatting. The [document guide](extensions/tmlr_integration/README.md) distinguishes the current manuscript and dimension-study evidence from historical saved-forecast packages. The editable source retains required author-identifying license attribution and is not an anonymous review upload. The public repository does not imply that the manuscript has been submitted or accepted.

## Licensing and citation

Original software is distributed under the [MIT License](LICENSE). Unless otherwise indicated, original generated research data and non-code research outputs are distributed under [CC BY 4.0](DATA_LICENSE). Third-party data, dependencies, and template files retain their stated licenses and attribution.

Repository citation metadata appears in [CITATION.cff](CITATION.cff). A reproducibility citation should identify the particular manuscript version and repository commit used.
