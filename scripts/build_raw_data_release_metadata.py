#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = ROOT / "reproduction" / "raw_data"
CANONICAL = RAW_ROOT / "canonical"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = CANONICAL / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_counts = {
        "duffing_clean": 11,
        "duffing_tuning_process_noise": 3,
        "vanderpol_clean": 35,
        "noisy_training": 480,
        "classic_control_exp16": 14,
        "classic_control_exp17": 4,
        "acrobot_exp18": 6,
    }

    if manifest.get("dataset_count") != 7:
        raise RuntimeError(f"Expected 7 family archives, got {manifest.get('dataset_count')}")
    if manifest.get("all_datasets_verified") is not True:
        raise RuntimeError("Manifest does not report all datasets verified.")
    if manifest.get("all_captures_started_clean") is not True:
        raise RuntimeError("Manifest does not report all captures started clean.")

    index_rows: list[dict[str, object]] = []

    for row in manifest["datasets"]:
        family = str(row["dataset_id"])

        if family not in expected_counts:
            raise RuntimeError(f"Unexpected family: {family}")

        metadata_path = CANONICAL / str(row["metadata_path"])
        archive_path = CANONICAL / str(row["archive_path"])

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        logical = metadata["extra_metadata"]["logical_subdatasets"]

        if len(logical) != expected_counts[family]:
            raise RuntimeError(
                f"{family}: expected {expected_counts[family]} logical subdatasets, "
                f"got {len(logical)}"
            )

        if sha256(archive_path) != row["archive_sha256"]:
            raise RuntimeError(f"Archive checksum mismatch: {archive_path}")

        if sha256(metadata_path) != row["metadata_sha256"]:
            raise RuntimeError(f"Metadata checksum mismatch: {metadata_path}")

        index_rows.append(
            {
                "family": family,
                "logical_subdatasets": len(logical),
                "archive": archive_path.name,
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": row["archive_sha256"],
                "content_sha256": row["content_sha256"],
                "metadata": metadata_path.name,
                "metadata_sha256": row["metadata_sha256"],
                "capture_git_commit": row["git_commit"],
                "capture_started_clean": not row["tree_dirty_before_capture"],
            }
        )

    total = sum(int(row["logical_subdatasets"]) for row in index_rows)
    if total != 553:
        raise RuntimeError(f"Expected 553 logical subdatasets, got {total}")

    index_rows.sort(key=lambda row: str(row["family"]))

    index_path = CANONICAL / "dataset_index.tsv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(index_rows[0].keys()),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(index_rows)

    checksum_targets = sorted(
        [
            path
            for path in CANONICAL.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        ],
        key=lambda path: path.name,
    )

    checksum_path = CANONICAL / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in checksum_targets
        ),
        encoding="utf-8",
    )

    card = """# Canonical raw-data package

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
.venv-arm64/bin/python scripts/build_raw_data_manifest.py \\
  --raw-dir reproduction/raw_data/canonical \\
  --require-nonempty \\
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
"""

    dataset_card_path = RAW_ROOT / "DATASET_CARD.md"
    dataset_card_path.write_text(card, encoding="utf-8")

    print(f"Wrote {dataset_card_path}")
    print(f"Wrote {index_path}")
    print(f"Wrote {checksum_path}")
    print("PASS: 7 family archives indexed")
    print("PASS: 553 logical raw-input subdatasets indexed")
    print("PASS: all archive and metadata checksums match manifest")


if __name__ == "__main__":
    main()
