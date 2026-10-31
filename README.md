# Kernel Affine Hull Koopman Machine

## Completed KAHM reconstruction-to-prediction research

**Kernel Affine Hull Representations for Time-Series Forecasting: Distinctness,
Stability, and Predictive Limits** reports the completed mechanism study,
its algebra and stability results, and the earlier preserved experiments.

- [Current manuscript PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Editable LaTeX source ZIP](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Completed findings and six fixed comparisons](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/reconstruction_mechanism/FINDINGS.md)
- [Study code, complete evidence and reproduction commands](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/extensions/reconstruction_mechanism/README.md)
- [Research evidence and interpretation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-reconstruction-mechanism/TIME_SERIES_RESEARCH.md)

The final fixed experiment contains **192 complete task suites**, ten
procedures and **24 independent repetition aggregates**. Bounded KAHM
reconstruction departures reduce nonlinear aggregate normalized MSE by
**12.58% against the common gate-and-coordinate basis**. Their **1.17% gain
against matched bounded RBF** does not establish the chosen 3% useful
margin. The **3.92% raw-RBF point gain** has an unresolved simultaneous
useful-margin bound. KAHM error is **1.77% higher than boosted trees** and
**6.04% higher than the validation-selected non-KAHM policy**.

All declared stability screens and linear-control noninferiority pass, but
only two of the six required comparisons pass overall. The fixed decision
is **STOP_FORECASTING_PERFORMANCE_BRANCH**. The study establishes conditional
basis distinctness, bounded corrections and useful value beyond the common
basis; it does not establish the required KAHM-specific practical forecasting
advantage. This is a result for the declared procedures and known laws,
not a universal impossibility claim.

The completed evidence includes **1,920 prediction arrays and 9,584,640 scored
values**. The source/input checkpoint was prospective and local; its later
publication is not public preregistration. Exact saved-state replay and an
independent-code internal audit accompany the results. Lossless evidence
parts preserve the original bytes and the earlier adaptive study.

Use the [current research branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/kahm-reconstruction-mechanism)
for the complete paper, code and audit commands. The manuscript is a
**research preprint, not submitted for peer review**. Earlier research
branches and their original evidence remain available. The original
Koopman implementation and its documentation follow below.

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
