# Canonical raw-data package

## Scope

This package contains the canonical raw scientific inputs used by the retained
empirical results R01--R22.

R23 is a documentary hyperparameter-search protocol and therefore has no
standalone raw scientific input dataset.

The package was generated from the frozen Git commit:

`ac531aefc66066c3fc320d94c0b362fb88f94967`

Every family capture started from a clean Git working tree.

## Package structure

The raw inputs are stored in seven compressed NumPy family archives. Each family
archive has a JSON sidecar describing its logical subdatasets, generation
parameters, array shapes and dtypes, environment, Git provenance, and both
archive-level and canonical array-content SHA-256 hashes.

| Family | Logical subdatasets | Purpose |
| --- | ---: | --- |
| `duffing_clean` | 11 | Clean Duffing trajectories used by retained experiments |
| `duffing_tuning_process_noise` | 3 | Expanded Duffing tuning trajectories with `process_noise=1e-5` |
| `vanderpol_clean` | 35 | Clean Van der Pol trajectories, including generalization pools |
| `noisy_training` | 480 | Training-noise realizations for robustness and noise-aware tuning |
| `classic_control_exp16` | 14 | CartPole/MountainCar fixed-policy training and test trajectories |
| `classic_control_exp17` | 4 | CartPole/MountainCar interpretability trajectories |
| `acrobot_exp18` | 6 | Three-seed Acrobot training/test trajectory pairs |

Total logical raw-input subdatasets: **553**.

## Environment

Reference capture environment:

- architecture: Apple Silicon `arm64`
- Python: 3.14.7
- NumPy: 2.4.4
- reference lockfile: `requirements-lock-arm64.txt`

The full environment provenance is also stored in each JSON sidecar.

## Important generation distinctions

### Duffing expanded tuning

The expanded Duffing tuning search used:

`process_noise = 1e-5`

and is therefore archived separately from the clean Duffing trajectories.

### Noise-aware Van der Pol tuning

The training-noise random seed depends on:

- model/data seed;
- cluster count `C`;
- kernel parameter `omega`;
- noise level.

Consequently, the noise-aware sweep contains a distinct raw noisy training input
for every `(seed, C, omega, noise_level)` job.

The frozen noise-seed expression is recorded in:

`reproduction/raw_data/canonical_input_plan.json`

### Sequential-decision tasks

CartPole, MountainCar, and Acrobot archives preserve transition arrays together
with the available action, reward, terminal, episode, and step-level fields
required by their respective experiment collectors.

## Integrity

`canonical/manifest.json` verifies all seven family archives and records the
frozen capture commit.

`canonical/SHA256SUMS` contains SHA-256 hashes for every distributed file in the
canonical package except the checksum file itself.

`canonical/dataset_index.tsv` is a compact index of family counts, archive
hashes, content hashes, metadata hashes, and capture provenance.

To verify distributed-file integrity from inside `reproduction/raw_data/canonical`:

```bash
shasum -a 256 -c SHA256SUMS
```

To verify semantic array contents and sidecars:

```bash
.venv-arm64/bin/python scripts/build_raw_data_manifest.py \
  --raw-dir reproduction/raw_data/canonical \
  --require-nonempty \
  --require-clean-captures
```

## Licensing

These original generated research data and non-code research outputs are covered
by the repository `DATA_LICENSE` unless otherwise indicated.

## Relationship to run-level results

The repository also preserves extensive unaggregated run-level metric CSV files.
Those files are not relabeled as raw scientific data. This package contains the
underlying generated trajectory, transition, and training-noise inputs needed to
support reproducibility of the retained empirical results.
