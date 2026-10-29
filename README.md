# Kernel Affine Hull Koopman Machine

Research software and reproducibility artifacts for Kernel Affine Hull Koopman Machine (KAHKM) experiments.

The repository contains the KAHKM implementation, experiment runners, hyperparameter-search scripts, archived numerical outputs, figure and table generators, and provenance records used to reproduce the reported results.

The software and reproducibility records are maintained independently of the
publication venue of any accompanying manuscript.

## KAHM forecasting and risk certification

The [certified forecasting extension](extensions/certified_forecasting/)
develops KAHM soft representations for direct prediction of future observed
outputs, retained representation refinement, and prediction-risk certification.
Its [editable manuscript](extensions/certified_forecasting/paper/) is titled
**Certifying Prediction Limits and Representation Refinement with Kernel
Affine Hull Machines**. The extension includes the original studies, a complete
input-scaling investigation, and a matched geometry--decoder study designed to
identify when the KAHM representation contributes to forecasting performance.
It also includes general lower-certificate comparisons, stratified
minimum-cost decisions, and recording-group speech-energy forecasting.
The reported accuracy comparisons include stronger alternatives and unresolved
certification outcomes.

Use the extension's own [reproduction guide](extensions/certified_forecasting/docs/REPRODUCIBILITY.md)
for its Linux/Python environment and experiment commands. The parent KAHKM
study and its reference environment below retain their original provenance.

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

The reproducibility evidence is documented in:

- `reproduction/REPRODUCIBILITY_EVIDENCE.md` — consolidated evidence and its scope.
- `reproduction/reproducibility_evidence.json` — machine-readable form of the same summary.
- `reproduction/raw_data/DATASET_CARD.md` — canonical raw scientific input package.
- `reproduction/run_accounting/README.md` — historical run-accounting audit.
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` — prospective 27-job Duffing reference rerun.

The historical Duffing reference archive contains 18 of the 27 scheduled jobs.
A separate prospective campaign records all 27 attempts and their exit statuses
from a fixed source commit. These later results establish the reproducibility
of the complete grid; the historical archive remains incomplete. Execution
ledgers covering every failed attempt are available for the prospective campaign
but not for all historical campaigns.

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

`CITATION.cff` identifies the repository and associated research work. It does
not currently specify an archival DOI; the commit identifier distinguishes the
software revision used in a study.

## Repository

https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine
