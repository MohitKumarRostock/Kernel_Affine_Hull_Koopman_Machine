# Soft state forecasting: computational reproduction and data

This guide specifies the source code, inputs, environments and verification criteria for the four-state prediction-risk study, independent-pair Van der Pol study, and supplementary forecasting experiments. The mathematical definitions and experimental designs are given in [Prediction-error certificates](SOFT_STATE_CERTIFICATES.md) and [KAHKM](KAHKM.md). The archived protocols and inventories are retained as source records; the instructions here specify the calculations directly.

## 1. Source and data identities

| Computational record | Git commit |
| --- | --- |
| Primary experiment code, four-state counts, frozen Van der Pol representation, compact dynamical results and forecasting data | [`0bbb2bafa107892e32e221e882326890b14fe36e`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e) |
| Four-method certificate comparison, deterministic derived CSV files and reproduction tests | [`78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2) |

These commit identifiers address immutable Git objects. An experiment execution commit identifies the code that generated its data; an evidence commit identifies distributed files and may also contain independently implemented verification procedures.

Use separate checkouts or detached worktrees for the respective sources. For example, the four-method comparison can be obtained using:

```bash
git clone --filter=blob:none --no-checkout https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine.git kahkm-evidence
cd kahkm-evidence
git fetch origin 78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2
git checkout --detach 78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2
```

For the primary experiment package, substitute `0bbb2bafa107892e32e221e882326890b14fe36e` for the comparison commit. Commands in the following sections are relative to the applicable checkout root, unless stated otherwise. The default branch does not contain the numerical evidence archives.

## 2. Environment

The primary reference environment is native Apple Silicon ARM64 with Python 3.14.7, NumPy 2.4.4, SciPy 1.17.1 and scikit-learn 1.8.0. The complete package specification and platform records are in [requirements-lock-arm64.txt](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/requirements-lock-arm64.txt) and [provenance/reference_arm64_environment](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e/provenance/reference_arm64_environment).

On a compatible native ARM64 macOS installation with Homebrew Python 3.14:

```bash
bash scripts/bootstrap_arm64.sh
.venv-arm64/bin/python scripts/verify_environment.py
```

The environment verifier checks architecture, dependencies and imports; it does not independently validate numerical results. Exact numerical identity across alternative platforms or dependency versions is not asserted.

The four-method count analysis uses Python 3.13 and NumPy 2.3.5; its commands below do not require fitting the KAHM model.

## 3. Controlled four-state data

Primary source directory: [four_state_original_iid_final_v1_run001](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/prospective_campaigns/four_state_original_iid_final_v1_run001).

| File | Contents |
| --- | --- |
| `reproduction/certificates/configs/four_state_final_v1.json` | Fixed case, sample-size and replicate design |
| `evidence.tar.gz` | State counts, per-dataset certificates, analytical reference values, schedule and numerical verification |
| `ARCHIVE.json`, `SHA256SUMS` | Archive member and distributed-file identities |
| `summary_v1/cell_summary.csv` | 117 case/sample-size summaries |
| `summary_v1/exclusion_summary.csv` | 702 case/sample-size/tolerance summaries |
| `summary_v1/numerical_audit.json` | Independent numerical recomputation |

The compressed evidence archive has SHA-256 `4776942edbd659f7a1c02ec554507f7c0259ca410449e2e5fd66f5353bf638d5`. Verify the archive distribution and inspect its members:

```bash
d=reproduction/prospective_campaigns/four_state_original_iid_final_v1_run001
(cd "$d" && shasum -a 256 -c SHA256SUMS)
tar -tzf "$d/evidence.tar.gz"
out=$(mktemp -d)
tar -xzf "$d/evidence.tar.gz" -C "$out"
```

Identify the extracted campaign directory containing the original `samples.jsonl` and `results.jsonl` records. With the reference environment and clean verifier source, run:

```bash
.venv-arm64/bin/python scripts/verify_four_state_final_numerics.py \
  --evidence /absolute/path/to/extracted/campaign-directory
```

The verifier uses the retained counts and rational/high-precision reference arithmetic. The archived report records 117,000 verified replicates, 49,271 zero label-based bounds, zero strict coverage violations and zero raw erroneous exclusions. This procedure recomputes the statistics and certificates from the archived observations, not the random count generation. The primary sampling entry point is [experiment_21_certificate_four_state_final.py](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/experiment_21_certificate_four_state_final.py); consult its command-line help and the fixed configuration for a separate simulation.

### Four-method comparison

Use the comparison commit and a Python 3.13 environment:

```bash
python3.13 -m venv .venv-erm
source .venv-erm/bin/activate
python -m pip install numpy==2.3.5
work=$(mktemp -d)
python reproduction/posthoc_four_state_erm/scripts/build_source_zip.py \
  --repo-root . --out "$work/four_state_frozen.zip"
python reproduction/posthoc_four_state_erm/scripts/reanalyze_historical_four_state.py \
  --archive "$work/four_state_frozen.zip" --out "$work/comparison"
(cd "$work/comparison" && shasum -a 256 \
  cell_summary_joint.csv cell_summary_pointwise.csv \
  replicates_joint.csv replicates_pointwise.csv)
```

Expected SHA-256 for the four **uncompressed** CSV files:

| File | SHA-256 |
| --- | --- |
| `cell_summary_joint.csv` | `6ffea4bfe2e8e1e545ddb8aecf241fc1341a2c3b7a934767907c96d7e2bfac37` |
| `cell_summary_pointwise.csv` | `eefda7da73d7dc9024f36e33664b121e92cedbb6b305cb19c961ad8c298b35a6` |
| `replicates_joint.csv` | `380476b3ce1f01129580ebfa0932828be7e2eae1c495d90bd980335278b859b9` |
| `replicates_pointwise.csv` | `927d48bfd80707abf17ff319f712282c948b0a8162c77a86ab161d5aa4519373` |

The [distributed outputs](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2/reproduction/posthoc_four_state_erm/outputs/historical_117k) contain replicate-level CSV files in gzip form. Verify their uncompressed contents against the corresponding digests. Run-specific manifests can differ in timestamp and packaging metadata. The [tests](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2/reproduction/posthoc_four_state_erm/tests) and [CI workflow](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2/.github/workflows/four_state_erm_reanalysis.yml) provide the associated automated checks.

The analysis operates on existing count records and computes the exact constrained empirical optimum analytically. Joint results use $\delta=0.0125$ per method, while pointwise results use $\delta=0.05$ per method. Neither analysis generates new random observations.

## 4. Independent-pair Van der Pol data

The fixed [representation archive](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/prospective_campaigns/dynamical_confirmatory_representation_v1_run001) contains the 25 reconstruction-model shards, centers, training records, matrix and independent verification records. Its `evidence.tar.gz` SHA-256 is `17fe746da6eddcc194da9fe19dedf7a80d3d887da397b103ae5e5f1f4718bcbc`.

The [evaluation archive](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/prospective_campaigns/dynamical_original_iid_confirmatory_v1_run001) provides `manifest_capsule.tar.gz`, `ARCHIVE.json`, checksums, the fixed configuration, and all 64 replicate-level primary results with aggregate summaries. The capsule contains 13 regular files and identifies the complete full-data evidence through its inventory. It is not the full physical-state or coordinate-array dataset.

| Public file under `summary_v1/` | Contents |
| --- | --- |
| `primary_replicates.csv` | One row per replicate at the fitted-norm budget |
| `mode_summary.csv` | Results separately for each transition law |
| `budget_summary.csv` | Results for every prespecified norm limit |
| `numerical_audit.json` | Recorded verification of retained arrays, random streams, statistics and bounds |
| `SHA256SUMS` | Summary-file checksums |

Check the distributed inventories using `shasum -a 256 -c SHA256SUMS` from the directory containing each checksum file. The full evaluation arrays are retained separately; data-access inquiries may be addressed to the corresponding author at [mohit.kumar@uni-rostock.de](mailto:mohit.kumar@uni-rostock.de). The full-data inventory fingerprint is `df9b1b7abd4a4c38b5bf8d0a5f43d7f6cb299ed23c070960b506f52641243221`.

### Public-data calculation

The following standard-library Python calculation reproduces each primary certificate from its published sufficient statistics and the two principal mean summaries. It does not require the omitted evaluation arrays.

```python
import csv
import math
from collections import defaultdict
from pathlib import Path

root = Path("reproduction/prospective_campaigns/"
            "dynamical_original_iid_confirmatory_v1_run001/summary_v1")
groups = defaultdict(list)
with (root / "primary_replicates.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        M = int(row["sample_size"])
        delta = float(row["delta"])
        kappa = float(row["kappa"])
        f_hat = float(row["f_hat"])
        s_hat = float(row["s_hat"])
        r = math.log(2 / delta) / M
        F = min(1, f_hat + r + math.sqrt(r*r + 2*r*f_hat))
        V = max(0, s_hat - math.sqrt(2*r))
        L = max(0, math.sqrt(V) - kappa*math.sqrt(2*F))
        assert math.isclose(L, float(row["L_kappa_delta"]),
                            rel_tol=0, abs_tol=1e-12)
        groups[row["evaluation_mode"]].append(
            (L, float(row["frozen_predictor_rmse"])))

assert len(groups) == 2 and all(len(g) == 32 for g in groups.values())
for mode, records in sorted(groups.items()):
    n = len(records)
    print(mode, sum(L > 0 for L, _ in records), n,
          sum(L for L, _ in records)/n,
          sum(rmse for _, rmse in records)/n)
```

Expected positive counts and means:

| Law | Positive / total | Mean certificate | Mean fitted RMSE |
| --- | ---: | ---: | ---: |
| Deterministic | 32/32 | 0.0844455 | 0.2759568 |
| Matched stochastic | 32/32 | 0.0845681 | 0.2763065 |

Recomputation from the full arrays requires access to the separate complete evidence directory. The [independent verifier](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/scripts/verify_dynamical_confirmatory_numerics.py) accepts `--evidence PATH --output-dir NEW_PATH`; the [summary generator](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/scripts/summarize_dynamical_confirmatory.py) accepts `--evidence PATH --output NEW_PATH`. The latter invokes numerical verification before aggregation. Supply the full evidence, not the compact capsule. The [experiment runner](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/experiment_23_certificate_dynamical_confirmatory.py), [configuration](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/certificates/configs/dynamical_confirmatory_v1.json), and [sampling protocol](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/certificates/DYNAMICAL_CONFIRMATORY_PROTOCOL.md) specify the independent simulation procedure. Public sufficient-statistic recomputation is not a verification of the omitted arrays.

## 5. Forecasting benchmarks

The forecasting experiments evaluate association construction, fitting, multistep errors, selection and numerical spectral diagnostics on simulated Duffing and Van der Pol systems and fixed-policy classic-control trajectories.

The [canonical input directory](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/raw_data/canonical) contains seven NumPy archives and 553 logical input datasets:

| Input family | Datasets |
| --- | ---: |
| Clean Duffing | 11 |
| Duffing tuning with process noise | 3 |
| Clean Van der Pol | 35 |
| Noisy training | 480 |
| CartPole/MountainCar forecasting | 14 |
| CartPole/MountainCar interpretation | 4 |
| Acrobot | 6 |

The manifest, sidecars, index and checksum file specify array shapes, dtypes, generation parameters and source identities. Verify file integrity:

```bash
(cd reproduction/raw_data/canonical && shasum -a 256 -c SHA256SUMS)
.venv-arm64/bin/python scripts/build_raw_data_manifest.py \
  --raw-dir reproduction/raw_data/canonical \
  --require-nonempty --require-clean-captures
```

The [result index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/reported_results.tsv) identifies generators, runners, configurations, seeds and output files. Its retained entries are marked `PASS`. Example generators for existing result records include:

```bash
.venv-arm64/bin/python scripts/generate_selection_table.py
.venv-arm64/bin/python scripts/generate_behavioral_summary.py
.venv-arm64/bin/python scripts/generate_search_protocol.py
.venv-arm64/bin/python scripts/generate_spectral_figures.py
```

Use the relevant experiment-specific runner and configuration for a fresh fit. Selection on later segments of training trajectories must not be described as independent final testing. Separate trajectory evaluations, fixed-policy control observations and spectral residual diagnostics have different statistical roles from independent-pair certificates.

The original Duffing reference-search record contains 18 of 27 grid jobs; the nine absent jobs have no original run-level evidence. The [separate complete 27-job experiment](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/prospective_campaigns/duffing_reference_27) contains an execution ledger and output checksums for all 27 jobs. It is independent evidence of reproducibility, not a replacement for missing original observations.

## 6. Scientific verification criteria

File and archive-member checksums establish data identity. Statistical recomputation checks reported quantities conditional on the retained data. A fresh simulation and refit test the computational procedure under a specified environment. These operations support distinct claims.

Monte Carlo intervals in the controlled and independent-pair studies are pointwise exact binomial intervals for replicate-level rates; empirical quantiles are descriptive. No observed violation in a finite set of replicates establishes zero theoretical failure probability. The sampling assumptions are properties of the experimental design, not consequences of matching checksums.

Cite the applicable source commit, configuration and evidence identity. Refer to [LICENSE](../LICENSE) and [DATA_LICENSE](../DATA_LICENSE) for use and redistribution.
