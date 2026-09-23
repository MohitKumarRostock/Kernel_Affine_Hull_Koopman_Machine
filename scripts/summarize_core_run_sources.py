#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reproduction" / "run_accounting" / "source_files.tsv"

TARGET_IDS = {
    "R01", "R02", "R03", "R04", "R05", "R06", "R07",
    "R08", "R09", "R10", "R11", "R12", "R13", "R14",
    "R15", "R16", "R21", "R22",
}

KEYS_OF_INTEREST = (
    "seed",
    "random_state",
    "train_seed",
    "test_seed",
    "data_seed",
    "replicate",
    "system",
    "task",
    "method",
    "operator",
    "n_clusters",
    "C",
    "omega",
    "noise_level",
    "train_count",
    "train_fraction",
    "horizon",
)


def sorted_values(values: set[str]) -> list[str]:
    def key(value: str):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)
    return sorted(values, key=key)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    groups: dict[tuple[str, str], dict[str, object]] = {}

    for row in rows:
        ids = {
            item.strip()
            for item in row["supporting_result_ids"].split(",")
            if item.strip()
        }

        if not (ids & TARGET_IDS):
            continue

        path = Path(row["path"])
        top = path.parts[0] if path.parts else ""
        basename = path.name
        group_key = (top, basename)

        if group_key not in groups:
            groups[group_key] = {
                "top": top,
                "basename": basename,
                "result_ids": set(),
                "file_count": 0,
                "total_rows": 0,
                "key_values": defaultdict(set),
                "sample_paths": [],
            }

        group = groups[group_key]
        group["result_ids"].update(ids & TARGET_IDS)
        group["file_count"] += 1

        row_count = str(row.get("row_count", "")).strip()
        if row_count:
            group["total_rows"] += int(row_count)

        run_keys = json.loads(row.get("run_key_values_json") or "{}")
        for key in KEYS_OF_INTEREST:
            for value in run_keys.get(key, []):
                group["key_values"][key].add(str(value))

        if len(group["sample_paths"]) < 3:
            group["sample_paths"].append(row["path"])

    def sort_key(item):
        group = item[1]
        ids = sorted(group["result_ids"])
        first_num = min(int(x[1:]) for x in ids) if ids else 999
        return (first_num, group["top"], group["basename"])

    shown = 0

    for _, group in sorted(groups.items(), key=sort_key):
        key_values = {
            key: sorted_values(values)
            for key, values in group["key_values"].items()
            if values
        }

        basename_lower = group["basename"].lower()
        interesting_name = any(
            token in basename_lower
            for token in (
                "raw",
                "result",
                "summary",
                "tuning",
                "search",
                "sweep",
                "general",
                "noise",
                "residual",
                "mode",
                "metric",
                "score",
            )
        )

        if not key_values and not interesting_name:
            continue

        shown += 1

        print()
        print(
            ",".join(sorted(group["result_ids"]))
            + " | "
            + group["top"]
            + " | "
            + group["basename"]
        )
        print(
            f"files={group['file_count']} "
            f"rows={group['total_rows']}"
        )

        if key_values:
            for key, values in key_values.items():
                print(f"  {key}: {values}")

        for sample in group["sample_paths"]:
            print(f"  sample: {sample}")

    print()
    print(f"Candidate source-family groups shown: {shown}")


if __name__ == "__main__":
    main()
