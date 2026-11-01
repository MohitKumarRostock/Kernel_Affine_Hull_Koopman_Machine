# Kernel Affine Hull Koopman Machine

## KAHM representations for time-series forecasting

**Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans,
Bounded Corrections, and Matched Evaluation** studies reconstruction-derived
feature spaces, bounded residual corrections, and their forecasting value.
The manuscript specifies the native local-dimension policy and evaluates
procedure-specific validation selection before calibration and test.
It uses anonymous TMLR review formatting.

- [Main manuscript PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-manuscript-dimension-revision-2026-11-01/extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Supporting studies PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-manuscript-dimension-revision-2026-11-01/extensions/tmlr_integration/review/KAHM_Supplementary_Studies.pdf)
- [Complete editable LaTeX source archive](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-manuscript-dimension-revision-2026-11-01/extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Scientific findings and all six primary comparisons](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-manuscript-dimension-revision-2026-11-01/extensions/dimension_forecasting/FINDINGS.md)
- [Study code, protocol, and reproduction commands](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-manuscript-dimension-revision-2026-11-01/extensions/dimension_forecasting/README.md)
- [Manuscript build instructions and verification records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-manuscript-dimension-revision-2026-11-01/extensions/tmlr_integration/manuscript/README_BUILD.md)

The principal experiment compares ten complete procedures on **192 task
suites**, with **24 independent repetition aggregates** as the inferential
units. The dimension-selected bounded KAHM procedure reduces nonlinear
aggregate normalized MSE by **9.01%** relative to independently selected common
features and by **8.01%** relative to raw-input RBF; the prespecified margins
are established for these comparisons. Together with linear-control
noninferiority, **three of six primary criteria** are established. The complete
joint criterion remains unmet, and KAHM has **4.08% higher** aggregate error
than the selected non-KAHM policy.

Selecting the cap improves nonlinear aggregate error by **3.26%** relative to
the historical cap-two procedure, while native cap twenty increases it by
**8.39%**. Dimension selection is therefore part of the forecasting procedure;
these results do not establish a universal advantage for larger subspaces or
for KAHM. Earlier supporting studies retain their original configurations and
evidential scope.

## Risk certificates for forecasting representations

**A Hierarchy of Risk Certificates for Forecasting Representations** develops
normalized block-risk lower bounds and calibrated decisions for declared
representation-decoder classes. Learned KAHM dictionaries provide a concrete
implementation. The complete main paper contains the proofs; the experimental
supplement contains the detailed protocols and secondary results.

- [Main manuscript PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-manuscript-dimension-revision-2026-11-01/extensions/certified_forecasting/paper/main.pdf)
- [Experimental supplement PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-manuscript-dimension-revision-2026-11-01/extensions/certified_forecasting/paper/supplement.pdf)
- [Complete editable LaTeX source archive](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-manuscript-dimension-revision-2026-11-01/extensions/certified_forecasting/review/Risk_Certificates_Manuscript_Source.zip)
- [Manuscript source and build instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-manuscript-dimension-revision-2026-11-01/extensions/certified_forecasting/paper/README_BUILD.txt)
- [Dimension-policy findings](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-manuscript-dimension-revision-2026-11-01/extensions/dimension_certificates/FINDINGS.md)
- [Study code, protocol, and reproduction commands](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-manuscript-dimension-revision-2026-11-01/extensions/dimension_certificates/README.md)

The targeted dimension-policy study uses **26 fitting repetitions** and a
study-wide allocation over **4,680 designated one-sided statements**.
Validation-selected caps improve descriptive test error in **18 repetitions**
and tie in eight relative to cap two. The selected-forecast certificate
decisions remain unchanged. Improved fitted accuracy is distinct from improved
class-exclusion resolution; the manuscript retains that distinction and the
separate scope of the historical studies.

## Research materials and availability

The [forecasting branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/kahm-manuscript-dimension-revision-2026-11-01) and
[certificate branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-manuscript-dimension-revision-2026-11-01) contain the reviewed manuscripts,
editable sources, computational code, protocols, analysis tables, and audit
receipts. The large numerical-records archives are separately supplied and
are not included in these Git branches. Replaying saved numerical records
requires those archives; the study guides identify their checksums and the
commands for fresh computation.

The original Koopman-machine implementation and its documentation follow below.

## Original Koopman-machine software

Research software and reproducibility artifacts for Kernel Affine Hull Koopman Machine (KAHKM) experiments.

The repository contains the KAHKM implementation, experiment runners, hyperparameter-search scripts, archived numerical outputs, figure and table generators, and provenance records used to reproduce the reported results.

## Reproducibility

The authoritative mapping from each reported result to its generator, experiment runner, configuration, seeds, raw or per-run output, and aggregate artifact is:

`reproduction/reported_results.tsv`

Entries marked `PASS` correspond to retained reported results with an identified reproducibility chain. Entries marked `REMOVE` identify legacy outputs that are not part of the canonical reported result set.

### Reference environment

The reference environment is native Apple Silicon (`arm64`) macOS.

Key files:

- `requirements.txt` — direct Python dependencies.
- `requirements-lock-arm64.txt` — frozen reference dependency environment.
- `scripts/bootstrap_arm64.sh` — native Apple Silicon environment bootstrap.
- `scripts/verify_environment.py` — architecture, dependency, and import verification.
- `provenance/reference_arm64_environment/` — captured hardware, operating-system, Python, NumPy, and package metadata.

Create the reference environment with:

```bash
bash scripts/bootstrap_arm64.sh
```

Verify it with:

```bash
.venv-arm64/bin/python scripts/verify_environment.py
```

### Canonical derived-result generators

Composite or newly canonicalized reported artifacts can be regenerated with:

```bash
.venv-arm64/bin/python scripts/generate_selection_table.py
.venv-arm64/bin/python scripts/generate_behavioral_summary.py
.venv-arm64/bin/python scripts/generate_search_protocol.py
.venv-arm64/bin/python scripts/generate_spectral_figures.py
```

Other reported tables and figures use the corresponding `step_*` generators and experiment runners identified in `reproduction/reported_results.tsv`.

## Reproducibility evidence

For a reviewer-facing summary of the current evidence state, see:

- `reproduction/REPRODUCIBILITY_EVIDENCE.md` — consolidated evidence summary and claim boundaries.
- `reproduction/reproducibility_evidence.json` — machine-readable form of the same summary.
- `reproduction/raw_data/DATASET_CARD.md` — canonical raw scientific input package.
- `reproduction/run_accounting/README.md` — historical run-accounting audit.
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` — prospective 27-job Duffing reference rerun.

The historical audit and prospective gap-closing campaign are intentionally
kept separate. The historical run-accounting audit preserves the original
Duffing reference-search gap, while the prospective campaign records all 27
scheduled attempts and their exit statuses from a frozen clean commit.

The repository does not claim that every failed attempt from every historical
campaign was preserved when the available historical evidence does not support
that stronger statement.

## Repository organization

- `experiment_*.py` — primary experiment implementations.
- `run_*.py` — experiment and hyperparameter-search runners.
- `step_*.py` — table and figure generation workflows.
- `scripts/` — environment and canonical reproducibility utilities.
- `reproduction/` — result-provenance matrix and canonical generated artifacts.
- `provenance/` — environment and dependency provenance.
- `figures/` — generated figures used by the associated research work.
- `kahkm_*` directories and archives — experiment outputs and derived artifacts.

## Licensing

Original source code in this repository is licensed under the MIT License. See `LICENSE`.

Unless otherwise indicated, original generated research data and non-code research outputs are licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). See `DATA_LICENSE`.

Third-party dependencies and any separately identified third-party material remain under their respective licenses.

## Citation

Citation metadata is provided in `CITATION.cff`.

A persistent archival identifier such as a DOI should be cited once an immutable archival release is available. Until then, cite the repository and associated research work using the metadata in `CITATION.cff`.

## Repository

https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine
