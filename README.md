# Kernel Affine Hull Koopman Machine

## KAHM time-series research

The current study develops **KAHM as an explicit time-series forecasting
method**, training local reconstruction bandwidths and gate temperatures
against the final forecast and horizon. Affine, locality and RBF-local
controls receive matched supervision and optimization ceilings, alongside
raw ridge, raw RBF, gradient boosting and a validation-selected policy.

- [Current manuscript PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-direct-reconstruction/extensions/tmlr_integration/review/KAHM_Forecast_Trained_Manuscript.pdf)
- [Editable manuscript source ZIP](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-direct-reconstruction/extensions/tmlr_integration/review/KAHM_Forecast_Trained_Manuscript_Source.zip)
- [Study, code and reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-direct-reconstruction/extensions/direct_reconstruction/README.md)
- [Complete findings and limitations](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-direct-reconstruction/extensions/direct_reconstruction/FINDINGS.md)
- [Research map and remaining questions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/kahm-direct-reconstruction/TIME_SERIES_RESEARCH.md)
- [Review the experiment and manuscript revision: PR #8](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/pull/8)

The [public scientific freeze](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/39174d06dffc7e28fed73dbbf866e539560dd021)
preceded execution. All 43 feasible task suites completed, preserving 731
selected model and prediction records. Three planned rideshare tasks lacked
geometry-training windows and remain documented as unavailable; the full
four-collection natural accuracy aggregate is undefined.

The 18 adjusted synthetic comparisons give two KAHM improvements, six
disadvantages and ten unresolved differences. Reconstruction training lowers
Van der Pol KAHM error by 41.43% relative to gate-only training, but matched
and practical alternatives remain stronger in important comparisons. The
seven-series natural subset improves ridge by 3.55%, chiefly through one
collection, while raw RBF and boosting are more accurate in aggregate and
faster in the measured implementation. The manuscript establishes a
reproducible forecasting construction without claiming broad superiority.

The complete current study is on
[`research/kahm-direct-reconstruction`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/kahm-direct-reconstruction).
Use that branch for its audit and reproduction commands. This main branch
preserves the original Koopman implementation below. The research manuscript
remains an **unsubmitted draft**; earlier studies and failed fits are retained.

Earlier resources remain available: the
[forecasting-utility release](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/releases/tag/forecasting-utility-v0.1.0)
and the separate [KAHM classification entry point](https://github.com/MohitKumarRostock/KAHM).

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
