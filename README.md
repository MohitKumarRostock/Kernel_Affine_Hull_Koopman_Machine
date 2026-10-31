# Kernel Affine Hull Koopman Machine

## KAHM representations for time-series forecasting

**Kernel Affine Hull Representations for Time-Series Forecasting: Distinctness,
Stability, and Predictive Limits** studies reconstruction-derived feature spaces,
bounded residual corrections, and their empirical forecasting value. The
manuscript uses anonymous TMLR review formatting.

- [Manuscript PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Editable LaTeX source](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Scientific findings and the six primary comparisons](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/reconstruction_mechanism/FINDINGS.md)
- [Mathematical definitions and derivations](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/reconstruction_mechanism/MATHEMATICAL_BACKGROUND.md)
- [Implementation, evidence, and reproduction commands](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/REPRODUCIBILITY.md)
- [Manuscript build and anonymous numerical supplement](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/tmlr_integration/README.md)
- [Compilation and evidence-verification results](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/tmlr_integration/TMLR_SUBMISSION_VALIDATION.json)

The principal experiment compares ten procedures on **192 task suites**, using
**24 independent repetition aggregates** for inference. Bounded KAHM reconstruction
departures reduce nonlinear aggregate normalized mean squared error by **12.58%**
relative to the shared local-coordinate basis. The **1.17%** gain over matched
bounded RBF reconstruction does not establish the prespecified 3% improvement
margin. The **3.92%** point gain over raw-input RBF forecasting has an unresolved
simultaneous margin bound. Boosted trees and the validation-selected non-KAHM
procedure have lower aggregate error.

The correction bounds, numerical stability checks, and linear-control
noninferiority criterion are satisfied. Two of the six required comparisons
are established, so the overall improvement criterion is not satisfied.
The results distinguish useful reconstruction features from a practical
forecasting advantage specific to KAHM.

The evidence contains **1,920 prediction arrays and 9,584,640 forecast values**.
The source and input record was specified locally before evaluation and was
not a public preregistration. The anonymous saved-forecast supplement independently
reconstructs all eight numerical analysis tables and the six primary confidence
bounds without fitting models or loading serialized model objects. Full fitting,
selected-state replay, numerical diagnostics, and the earlier studies have
separate reproduction instructions.

Use the [research branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/kahm-reconstruction-mechanism)
for the manuscript and its computational evidence. The original Koopman-machine
implementation and its documentation follow below.

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
