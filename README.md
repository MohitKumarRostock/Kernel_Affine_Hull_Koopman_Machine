# KAHM reconstruction and time-series prediction

**Completed research:** KAHM reconstruction departures can add useful basis functions to a finite forecast decoder, and their corrections can be bounded. The fixed study does **not** establish the required KAHM-specific practical forecasting advantage, so its decision is to **stop this forecasting-performance branch**.

## Current manuscript and results

**Kernel Affine Hull Representations for Time-Series Forecasting: Distinctness, Stability, and Predictive Limits** is a research preprint reporting completed theory and experiments.

- [Manuscript PDF](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Editable LaTeX source ZIP](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Completed mechanism findings](extensions/reconstruction_mechanism/FINDINGS.md)
- [Code, evidence restoration and audit commands](extensions/reconstruction_mechanism/README.md)
- [All six fixed comparisons](extensions/reconstruction_mechanism/analysis/decision.json)
- [Research evidence and interpretation](TIME_SERIES_RESEARCH.md)
- [Manuscript build and package guide](extensions/tmlr_integration/README.md)

The final experiment contains **192 complete task suites**, ten procedures and **24 independent repetitions**, with a local source/input freeze before reserved forecasts. KAHM lowers nonlinear aggregate normalized MSE by **12.58% against the common gate-and-coordinate basis**. Its **1.17% gain against matched RBF departure** does not meet the fixed 3% useful-effect criterion. The **3.92% point gain over raw RBF** has an unresolved simultaneous useful-margin bound. KAHM error is **1.77% higher than boosted trees** and **6.04% higher than the C-selected non-KAHM policy**. All stability screens and linear-control noninferiority pass; only two of the six required comparisons pass overall.

The algebra proves conditional raw-span distinctness and identifies exact redundancy and normalized-KRR identities. A radial feature map and centered-ridge bound control correction magnitude. These properties and the useful common-basis result are retained alongside the adverse practical comparison. The conclusion concerns the declared procedures and known laws; it is not a theorem that every KAHM or geometric predictor must fail.

The complete evidence contains **1,920 saved prediction arrays and 9,584,640 scored values**. Selected-state replay is exact on the recorded environment; an independent-code internal audit reconstructs predictions and all six contrast statistics. The public tree uses hash-verified lossless evidence parts. Restore them using the guide above before running audits. The local freeze is not claimed as public preregistration.

The earlier [direct-training study](extensions/direct_reconstruction/FINDINGS.md) and [adaptive development study](extensions/adaptive_development/FINDINGS.md) remain separate. Both adaptive capacity tiers failed their declared accuracy screens after a documented numerical amendment. Previous protocols, unsuccessful fits and limitations remain preserved. Anonymous author metadata is retained from the supplied manuscript; the preprint has not been submitted for peer review.

## Preserved KAHKM and historical study evidence

The following reference environment and result map concern the original KAHKM
experiments. Later forecasting extensions each declare their own environment,
protocol, freeze and evaluation scope.

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
