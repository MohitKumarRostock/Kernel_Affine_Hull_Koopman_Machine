# Reproducibility

This repository provides a reproducible reference environment and experiment workflow for the Kernel Affine Hull Koopman Machine (KAHKM).

The repository is intentionally independent of any particular journal, conference, or publication venue.

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
4. The 27-job reference experiment records every scheduled attempt, exit status and log, including failures.
5. Unaggregated experimental outputs are supplied where available, and missing outputs are identified.
6. Aggregated results are derived programmatically from raw outputs.
7. Tables and figures are generated programmatically.
8. Every reported result is traceable to the runs that produced it.
9. The exact source commit and execution environment are recorded.
10. Distributed files have cryptographic checksums.

## Experimental reproduction

Environment reproduction and scientific-result reproduction are treated as separate stages.

The environment bootstrap establishes a validated computational environment. Experiment manifests, raw-data capture, run ledgers, result indexing, and full reproduction commands are maintained separately so that each stage can be audited independently.

## Evidence map

The evidence summary is:

`reproduction/REPRODUCIBILITY_EVIDENCE.md`

Its machine-readable counterpart is:

`reproduction/reproducibility_evidence.json`

The principal evidence layers are:

- `reproduction/raw_data/DATASET_CARD.md` for raw scientific inputs;
- `reproduction/run_accounting/README.md` for coverage checks of archived runs;
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` for the separate 27-job Duffing reference experiment;
- `reproduction/reported_results.tsv` for the mapping from reported results to source code and data.

The archived Duffing reference search contains records for 18 of its 27 scheduled jobs. The separate 27-job reference experiment records every attempt, exit status, log and output checksum. These are records of different executions.

The execution log covers every attempt in the 27-job experiment. The other
archives do not establish that every failed attempt was saved.

## Environment scope

The reference environment is the documented native Apple Silicon configuration. Results produced with another environment require the corresponding software and platform records.

