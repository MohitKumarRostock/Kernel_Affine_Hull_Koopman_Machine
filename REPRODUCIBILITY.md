# Reproducibility

This repository provides a reproducible reference environment and experiment workflow for the Kernel Affine Hull Koopman Machine (KAHKM).

This guide describes the parent KAHKM study. The forecasting extension has a
separate [reproduction guide](extensions/certified_forecasting/docs/REPRODUCIBILITY.md)
for its Linux environment, episode-based experiments, and manuscript.

## Reference platform

The reference computational environment is:

- Apple Silicon (`arm64`)
- Apple M1
- macOS
- Python 3.14
- native ARM64 execution
- Homebrew under `/opt/homebrew`

The captured reference-platform metadata are stored under:

`provenance/reference_arm64_environment/`

## Python dependencies

`requirements.txt` contains the direct Python dependencies used by the project.

`requirements-lock-arm64.txt` records the complete resolved Python package environment used for the reference Apple Silicon configuration.

Local virtual-environment directories are intentionally excluded from version control.

## Creating the reference environment on Apple Silicon

Install native Homebrew Python 3.14 if necessary:

```bash
/opt/homebrew/bin/brew install python@3.14
```

Create the reference environment from a fresh clone:

```bash
bash scripts/bootstrap_arm64.sh
```

Activate it:

```bash
source .venv-arm64/bin/activate
```

Verify the environment and core project imports:

```bash
python scripts/verify_environment.py
```

A successful verification terminates with:

```text
ENVIRONMENT VERIFICATION PASSED
```

## Reference dependency versions

The principal direct dependencies are:

- NumPy 2.4.4
- SciPy 1.17.1
- scikit-learn 1.8.0
- Matplotlib 3.10.9
- Gymnasium 1.3.0
- joblib 1.5.3
- Numba 0.67.0
- tqdm 4.70.1

The lock file is authoritative for the complete resolved environment.

## Environment provenance

The `provenance/` directory records the software and computational environment used for the reference reproduction workflow.

It intentionally excludes machine serial numbers, hardware UUIDs, credentials, local editor configuration, and virtual-environment directories.

## Reproduction principles

The repository follows these principles for reproducible experimental results:

1. Experiment parameters are stored in version-controlled configuration or manifest files.
2. Random seeds are explicit and recorded for every stochastic run.
3. Training, tuning, and final evaluation data are separated explicitly.
4. Prospective reproducibility campaigns use failure-preserving run ledgers that record every scheduled attempt, including failures.
5. Raw unaggregated experimental outputs used as reproducibility evidence are retained where available, with historical gaps reported explicitly rather than inferred away.
6. Aggregated results are derived programmatically from raw outputs.
7. Tables and figures are generated programmatically.
8. Every reported result is traceable to the runs that produced it.
9. The exact source commit and execution environment are recorded.
10. Distributed evidence packages include cryptographic checksums. An immutable
    archival release and its DOI are not recorded in the current citation metadata.

## Experimental reproduction

Environment reproduction and scientific-result reproduction are treated as separate stages.

The environment bootstrap establishes a validated computational environment. Experiment manifests, raw-data capture, run ledgers, result indexing, and full reproduction commands are maintained separately so that each stage can be audited independently.

## Evidence map

The consolidated reproducibility evidence is in:

`reproduction/REPRODUCIBILITY_EVIDENCE.md`

Its machine-readable counterpart is:

`reproduction/reproducibility_evidence.json`

The principal evidence layers are:

- `reproduction/raw_data/DATASET_CARD.md` for canonical raw scientific inputs;
- `reproduction/run_accounting/README.md` for the historical run-accounting audit;
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` for the prospective Duffing reference campaign;
- `reproduction/reported_results.tsv` for result-level provenance and retention status.

The historical Duffing reference-search archive contains 18 of 27 scheduled
jobs. The prospective campaign reruns the complete grid from a fixed source
commit and records every attempt, exit status, log, and output checksum.
Complete execution accounting therefore applies to that prospective campaign.
Equivalent ledgers are unavailable for all historical campaigns.

## Historical environments

Development may have occurred under earlier or different Python environments. Such historical environments are not presented as the reference reproduction environment unless their provenance can be established independently.

The reference environment documented here is the validated native Apple Silicon environment used for the reproducibility workflow.
