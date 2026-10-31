# KAHM representations for time-series forecasting

This repository contains the manuscript, implementation, and experimental evidence for **Kernel Affine Hull Representations for Time-Series Forecasting: Distinctness, Stability, and Predictive Limits**. The study examines how a Kernel Affine Hull Machine (KAHM) reconstruction contributes features to a finite forecasting model, when those features differ from simpler alternatives, and whether the resulting procedure improves prediction.

## Manuscript and principal findings

- [Manuscript PDF](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Editable LaTeX source](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Mathematical background](extensions/reconstruction_mechanism/MATHEMATICAL_BACKGROUND.md)
- [Experimental findings](extensions/reconstruction_mechanism/FINDINGS.md)
- [Reproduction instructions](extensions/reconstruction_mechanism/README.md)
- [Guide to the studies](TIME_SERIES_RESEARCH.md)

The theoretical analysis distinguishes three properties. A deterministic reconstruction cannot add information to an observed history, but it can change the functions available to a restricted decoder. The KAHM reconstruction sometimes gives the same augmented feature span as a matched radial-basis-function (RBF) reconstruction; an explicit irregular reference set also shows that the spans can differ. Separately, radial feature normalization and a ridge residual predictor provide a uniform bound on the magnitude of the forecast correction. This bound does not guarantee accuracy.

The principal experiment compares ten procedures in 192 task suites, comprising four simulation laws, two observation-noise levels, and 24 independent repetitions. On the three nonlinear laws, the bounded KAHM procedure reduces aggregate normalized mean squared error by **12.58% relative to the common local-coordinate features**. The gains over matched RBF departures and raw-history RBF features are **1.17%** and **3.92%**, respectively; neither comparison establishes the prespecified 3% improvement under the simultaneous analysis. KAHM has **1.77% higher error than boosted trees** and **6.04% higher error than the validation-selected non-KAHM procedure**. The linear-control noninferiority criterion and all numerical stability checks are satisfied.

These results support a conditional contribution of reconstruction features to a finite decoder. They do not establish a distinctive practical forecasting advantage for KAHM over the complete set of evaluated alternatives. The inferential units are the 24 repetition aggregates. The 9,584,640 individual forecast values are dependent measurements within those repetitions.

## Reproduce the principal experiment

The public repository stores large evidence files in lossless, hash-verified parts. From the repository root, use Python 3.12 and run:

```bash
python -m pip install -r extensions/reconstruction_mechanism/requirements.txt
python -m extensions.reconstruction_mechanism.evidence unpack
python -m extensions.reconstruction_mechanism.run verify
python -m extensions.reconstruction_mechanism.analyze --output /tmp/kahm-analysis
python -m extensions.reconstruction_mechanism.audit \
  --analysis-json /tmp/kahm-analysis/decision.json \
  --output /tmp/kahm-audit.json
```

The [study guide](extensions/reconstruction_mechanism/README.md) gives the commands for selected-model replay and a new execution of the fixed experiment. The saved evidence includes all 1,920 task–procedure prediction arrays, selected model states, selection records, and measured costs. An internal audit evaluates the saved models using independent code and recomputes the primary statistical comparisons. This verifies computational consistency; it is not an external replication.

The source and inputs were fixed locally before the held-out forecasts were computed. That checkpoint is documented as a prospective local freeze, not a public preregistration. Each earlier study has its own design, data scope, and statistical analysis. Their observations are not pooled with the principal experiment.

## Repository organization

| Location | Contents |
| --- | --- |
| `extensions/tmlr_integration/` | Manuscript source, generated displays, compiled paper, and build instructions |
| `extensions/reconstruction_mechanism/` | Reconstruction-departure experiment, algebraic examples, saved evidence, and verification tools |
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
```

Compilation does not fit forecasting models. The paper is prepared in anonymous TMLR review format. The public repository contains author-identifying information and is separate from any anonymous submission supplement. The repository does not imply that the manuscript has been submitted or accepted.

## Licensing and citation

Original software is distributed under the [MIT License](LICENSE). Unless otherwise indicated, original generated research data and non-code research outputs are distributed under [CC BY 4.0](DATA_LICENSE). Third-party data, dependencies, and template files retain their stated licenses and attribution.

Repository citation metadata appears in [CITATION.cff](CITATION.cff). A reproducibility citation should identify the particular manuscript version and repository commit used.
