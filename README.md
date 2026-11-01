# KAHM representations for time-series forecasting

This repository contains the manuscript, implementation, and experimental evidence for **Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans, Bounded Corrections, and Matched Evaluation**. The study examines how a Kernel Affine Hull Machine (KAHM) reconstruction contributes features to a finite forecasting model, when those features differ from simpler alternatives, and whether the resulting procedure improves prediction.

## Manuscript and principal findings

- [Manuscript PDF](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Supplementary studies](extensions/tmlr_integration/review/KAHM_Supplementary_Studies.pdf)
- [Editable LaTeX source](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Mathematical background](extensions/reconstruction_mechanism/MATHEMATICAL_BACKGROUND.md)
- [Principal experimental findings](extensions/dimension_forecasting/FINDINGS.md)
- [Preprocessing findings](extensions/preprocessing_dimension/FINDINGS.md)
- [Cross-fit reproduction findings](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md)
- [Fresh policy-augmentation findings](extensions/policy_augmentation_20261101/RESULTS.md)
- [Policy-augmentation research report](extensions/policy_augmentation_20261101/reporting/KAHM_Policy_Augmentation_Research_Report.pdf)
- [Separate 48-series natural policy evaluation](extensions/policy_augmentation_natural_20261101/FINDINGS.md)
- [Principal reproduction instructions](extensions/dimension_forecasting/README.md)
- [Guide to the studies](TIME_SERIES_RESEARCH.md)

The theoretical analysis distinguishes three properties. A deterministic reconstruction cannot add information to an observed history, but it can change the functions available to a restricted decoder. The KAHM reconstruction sometimes gives the same augmented feature span as a matched radial-basis-function (RBF) reconstruction; an explicit irregular reference set also shows that the spans can differ. Separately, radial feature normalization and a ridge residual predictor provide a uniform bound on the magnitude of the forecast correction. This bound does not guarantee accuracy.

The principal experiment compares ten procedures in 192 task suites: four simulation laws, two added-noise levels, and 24 independent repetitions. Each geometric procedure selects its own local dimension cap on validation data. Bounded KAHM reduces mean nonlinear standardized MSE by **9.01% relative to common local-coordinate features** and **8.01% relative to raw-input RBF regression**, establishing the declared 3% margins. Its **0.72% gain over the independently tuned RBF departure** and **2.39% gain over boosted trees** do not establish that margin. KAHM has **4.08% higher error than the selected non-KAHM procedure**. Linear-control noninferiority and all maximum-error screens are satisfied; three of six primary criteria are established.

These results support a conditional contribution of reconstruction features to a finite decoder. They do not establish a distinctive practical forecasting advantage for KAHM over the complete set of evaluated alternatives. The inferential units are the 24 repetition aggregates. The 9,584,640 individual forecast values are dependent measurements within those repetitions.

The coordinated manuscript presents the principal study and two follow-ups as separate experiments:

| Study | Primary comparison family | Established criteria | Evidence scope |
| --- | --- | --- | --- |
| Validation-selected local dimension | Six comparisons | Three: nonlinear gains over common features and raw-input RBF, and linear noninferiority | Principal controlled simulation experiment, 24 repetition units |
| Preprocessing and local dimension | Four comparisons | One: linear noninferiority | Separate controlled experiment; fixed-common residual corrections are secondary |
| Cross-fitted residual correction | Four comparisons | Two: nonlinear gain over the shared baseline, and linear noninferiority | Same-seed computational reproduction of 24 original cross-fit units; zero additional independent repetitions |

The strongest demonstrated role is a bounded feature family that can improve residual correction around a specified forecast. An advantage over independently tuned alternatives and the complete non-KAHM selection policy must be assessed separately. A subsequent [fresh policy-augmentation study](extensions/policy_augmentation_20261101/RESULTS.md) now evaluates this distinction directly: adding KAHM reduces nonlinear policy loss by 0.816%, while neither prespecified 3% superiority criterion is established. Its separate natural-series policy evaluation has 2.273% higher mean error than the base policy.

The main paper contains the central definitions and theoretical statements, complete proofs, the principal forecasting experiment and its protocol, and the preprocessing and residual-target follow-ups. The companion supplementary studies retain the complete controlled, natural-data, scaling, additive, learned-geometry, direct-training, adaptive, and follow-up investigations with their own notation and comparison families. That coordinated document edition predates the fresh policy-augmentation study; the subsequent synthetic and natural findings are reported in the separate study records linked below.

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

## Preprocessing and corrections around a fixed common forecast

The preprocessing study preserves a separate four-comparison primary family.
Tanh/tanh preprocessing has **6.486% higher nonlinear error** than the
standard/standard procedure. Jointly selected KAHM has **4.022% higher error**
than the selected non-KAHM policy and **0.281% higher error** than independently
selected bounded RBF. Only linear-control noninferiority is established.

A secondary experiment fits each common predictor on 32 episodes and fits
centered corrections on 32 disjoint episodes. Within each of its five settings,
KAHM and RBF share the same common forecast and bias. KAHM reduces nonlinear
error by **10.780–14.934% relative to that shared baseline** and by
**2.372–4.140% relative to independently selected RBF**. These secondary
comparisons support a conditional residual-correction mechanism; they do not
replace the primary complete-procedure results.

[Findings](extensions/preprocessing_dimension/FINDINGS.md)
· [Frozen protocol](extensions/preprocessing_dimension/PROTOCOL.md)
· [Results and restoration guide](extensions/preprocessing_dimension/RESULTS_PACKAGE.md)
· [Independent audit](extensions/preprocessing_dimension/audit/final_results_audit.json)
· [Numerical archive index](extensions/preprocessing_dimension/reporting/archive_index.json)

## Cross-fitted residual correction with a full-training common predictor

An additional synthetic study evaluates episode-wise out-of-fold residuals
around a common predictor fitted on all designated training episodes. Its
computational reproduction establishes two of the four original primary
criteria. Cross-fitted KAHM reduces nonlinear standardized MSE by **9.1968%**
relative to the identical common-plus-bias baseline. Its **2.8662%** reduction
against independently selected cross-fitted RBF does not establish the
prescribed 3% margin, and its error is **5.1246% higher** than the expanded
non-KAHM policy. Linear performance satisfies the prescribed 5% noninferiority
criterion.

In the secondary residual-target comparison, cross-fitted KAHM has **3.6873%
higher error** than in-sample KAHM. The in-sample correction reduces error by
**12.4259% relative to the shared baseline** and **3.9643% relative to
independently selected in-sample RBF**. Its mean error remains **1.3862% higher
than the expanded non-KAHM policy**, a post hoc descriptive comparison rather
than an additional primary test. The difference between 48-episode fold
predictors and the 64-episode deployed common predictor is a possible
residual-alignment explanation; this experiment does not isolate that cause.

The independent audit covers all 192 task suites and 8,064 retained forecast
arrays. The reproduction reuses the original 24 simulation repetition units
and adds no independent repetitions. Earlier reported outcomes were known
before regeneration. It verifies its newly retained execution without
authenticating the unavailable original completed-run records. The numerical
package contains one analysis ZIP and sixteen standalone records ZIPs, with
3,311 retained files and an exact size/SHA-256 index.

[Results](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md)
· [Protocol](extensions/crossfit_residual_reproduction_20261101/PROTOCOL.md)
· [Primary comparisons](extensions/crossfit_residual_reproduction_20261101/reporting/published_tables/primary_contrasts.csv)
· [Independent audit](extensions/crossfit_residual_reproduction_20261101/verification/reproduction_audit.json)
· [Archive manifest](extensions/crossfit_residual_reproduction_20261101/reporting/ARCHIVE_MANIFEST.json)
· [Archive delivery verification](extensions/crossfit_residual_reproduction_20261101/reporting/archive_delivery_verification.json)

## Fresh policy augmentation and separate natural evaluation

The new synthetic experiment uses **24 fresh repetitions and 192 task suites**.
P0 is the established ten-family menu, P1 adds the existing in-sample KAHM
residual correction, and P0plus adds a new raw-history Nyström RBF correction.
The two additions receive the same 75 B-validation scores per horizon and
one additional C-selection slot. Native local dimension and preprocessing
remain validation-selected; equal search counts do not imply equal capacity
or runtime.

P1 has **0.816% lower nonlinear SMSE than P0** and **0.658% lower than
P0plus**. Neither declared 3% superiority criterion is established; both
linear 5% noninferiority criteria are established. The standalone KAHM
correction improves the shared baseline by 12.497%, a separate descriptive
comparison that does not establish complete-policy superiority.

The chronological natural evaluation uses **24 Weekly and 24 Hourly M4
training-series histories**, excluding the previously evaluated pilot IDs.
It reserves terminal training segments, uses no official M4 Test data, and
does not claim an official M4 ranking or independent-source inference.
P1 has **2.273% higher overall SMSE than P0** and **2.718% lower than P0plus**.
KAHM is selected on 3 of 48 series, with one win and two losses versus P0;
the other 45 P1 forecasts are identical to P0.

[Synthetic results](extensions/policy_augmentation_20261101/RESULTS.md)
· [Frozen synthetic protocol](extensions/policy_augmentation_20261101/PROTOCOL.md)
· [Synthetic independent audit](extensions/policy_augmentation_20261101/verification/policy_audit.json)
· [Natural findings](extensions/policy_augmentation_natural_20261101/FINDINGS.md)
· [Natural reproduction](extensions/policy_augmentation_natural_20261101/REPRODUCTION.md)
· [Natural independent audit](extensions/policy_augmentation_natural_20261101/audit/verification.json)
· [Numerical archive index](extensions/policy_augmentation_20261101/reporting/ARCHIVE_INDEX.json)

## Repository organization

| Location | Contents |
| --- | --- |
| `extensions/tmlr_integration/` | Coordinated main paper and supporting-study source, generated displays, compiled PDFs, and document release records |
| `extensions/dimension_forecasting/` | Principal dimension-selected comparison, source and phase records, complete analysis, and independent audit |
| `extensions/preprocessing_dimension/` | Four-comparison preprocessing study, secondary disjoint-episode residual corrections, audited results, and numerical archive index |
| `extensions/crossfit_residual_reproduction_20261101/` | Cross-fitted and in-sample residual correction study, same-cohort computational reproduction, audited results, and complete numerical archive index |
| `extensions/policy_augmentation_20261101/` | Fresh 24-repetition policy augmentation, matched added-search control, complete analysis, audits, report builder and numerical archive index |
| `extensions/policy_augmentation_natural_20261101/` | Separate chronological 48-series M4 training-history evaluation, deterministic roster, audited descriptive results and reproduction guide |
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

Compilation uses the supplied generated displays and does not fit forecasting
models or require the numerical archives. The [document guide](extensions/tmlr_integration/README.md)
and [build guide](extensions/tmlr_integration/manuscript/README_BUILD.md)
describe the coordinated source, including `core_followup_experiments.tex`
and the separate preprocessing and cross-fit display namespaces. The
[coordinated release record](extensions/tmlr_integration/review/COORDINATED_MANUSCRIPT_RELEASE.json)
identifies the delivered document edition. Reporting and manuscript-source
changes leave the frozen scientific protocols, fitting code, analysis
definitions, numerical evidence and stage seals unchanged.

Both PDFs use anonymous TMLR review formatting. The editable source retains
required author-identifying license attribution and is not an anonymous review
upload. Historical saved-forecast packaging scripts retain their original
fixed-cap scope. The public repository does not imply that the manuscript has
been submitted or accepted.

## Licensing and citation

Original software is distributed under the [MIT License](LICENSE). Unless otherwise indicated, original generated research data and non-code research outputs are distributed under [CC BY 4.0](DATA_LICENSE). Third-party data, dependencies, and template files retain their stated licenses and attribution.

Repository citation metadata appears in [CITATION.cff](CITATION.cff). A reproducibility citation should identify the particular manuscript version and repository commit used.
