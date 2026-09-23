#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reproduction" / "data_inventory"

DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".npz",
    ".npy",
    ".zip",
}

RAW_ARRAY_KEYS = {
    "X0",
    "X1",
    "X0_train",
    "X1_train",
    "X0_test",
    "X1_test",
    "states",
    "actions",
    "terminals",
    "episode_ids",
    "step_ids",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    return [
        ROOT / line
        for line in result.stdout.splitlines()
        if Path(line).suffix.lower() in DATA_SUFFIXES
        and not line.startswith(".vscode/")
        and not line.startswith("reproduction/data_inventory/")
    ]

def inspect_npz(path: Path) -> tuple[str, str]:
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)

        details = []

        for key in keys:
            arr = np.asarray(data[key])

            details.append(
                f"{key}:shape={list(arr.shape)},dtype={arr.dtype}"
            )

        if RAW_ARRAY_KEYS.intersection(keys):
            category = "raw_scientific_input"
        elif "indices" in path.name.lower():
            category = "selection_indices"
        else:
            category = "numeric_array_artifact"

    return category, "; ".join(details)

def classify(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    name = path.name.lower()
    rel = str(path.relative_to(ROOT))

    if suffix == ".npz":
        return inspect_npz(path)

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return (
            "numeric_array_artifact",
            f"shape={list(arr.shape)},dtype={arr.dtype}",
        )

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                members = zf.namelist()
        except Exception as exc:
            return "archive", f"unreadable archive: {exc}"

        return (
            "archive",
            f"{len(members)} members",
        )

    if suffix == ".json":
        return "metadata_or_configuration", ""

    if (
        "summary" in name
        or "table" in name
        or "plot_data" in name
        or rel.startswith("reproduction/generated/")
    ):
        return "aggregate_or_derived", ""

    if (
        "raw" in name
        or "one_step" in name
        or "multistep" in name
        or "regime_statistics" in name
        or "transition_matrix" in name
        or "mode_residual" in name
        or "/runs/" in rel
    ):
        return "unaggregated_run_level_result", ""

    return "tabular_or_other_data", ""

def source_capture_audit() -> list[dict[str, Any]]:
    rows = []

    for path in sorted(ROOT.glob("experiment_*.py")):
        text = path.read_text(encoding="utf-8")

        rows.append(
            {
                "script": path.name,
                "uses_np_save": (
                    "np.save(" in text
                    or "np.savez(" in text
                    or "np.savez_compressed(" in text
                ),
                "generates_X0_X1": (
                    "X0" in text and "X1" in text
                ),
                "mentions_trajectory_or_episode": (
                    "trajectory" in text.lower()
                    or "episode" in text.lower()
                ),
            }
        )

    return rows

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in tracked_files():
        if not path.is_file():
            continue

        category, details = classify(path)

        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "category": category,
                "details": details or "-",
            }
        )

    csv_path = OUT / "data_artifact_inventory.tsv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "suffix",
                "bytes",
                "sha256",
                "category",
                "details",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}

    for row in rows:
        counts[row["category"]] = (
            counts.get(row["category"], 0) + 1
        )

    capture = source_capture_audit()

    capture_path = OUT / "source_data_capture_audit.json"
    capture_path.write_text(
        json.dumps(capture, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip(),
        "artifact_counts_by_category": counts,
        "raw_scientific_input_files": [
            row["path"]
            for row in rows
            if row["category"] == "raw_scientific_input"
        ],
        "selection_index_files": [
            row["path"]
            for row in rows
            if row["category"] == "selection_indices"
        ],
        "experiments_with_numpy_array_save_calls": [
            row["script"]
            for row in capture
            if row["uses_np_save"]
        ],
        "assessment": (
            "The repository preserves extensive unaggregated run-level "
            "results, but raw scientific input trajectories/snapshot arrays "
            "are not yet systematically persisted for all retained results. "
            "A final reproduction campaign must capture deterministic raw "
            "inputs together with seeds, shapes, checksums, configuration, "
            "environment, and Git commit."
        ),
    }

    summary_path = OUT / "data_inventory_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {capture_path}")
    print(f"Wrote {summary_path}")

    print()
    print("=== CATEGORY COUNTS ===")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")

    print()
    print("=== RAW SCIENTIFIC INPUT FILES ===")
    for item in summary["raw_scientific_input_files"]:
        print(item)

    print()
    print("=== EXPERIMENTS WITH ARRAY-SAVE CALLS ===")
    for item in summary["experiments_with_numpy_array_save_calls"]:
        print(item)

if __name__ == "__main__":
    main()
