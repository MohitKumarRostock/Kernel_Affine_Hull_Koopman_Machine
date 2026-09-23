#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reproduction" / "run_accounting" / "source_files.tsv"
OUTPUT = ROOT / "reproduction" / "run_accounting" / "run_grid_candidates.tsv"

KEYWORDS = (
    "tuning",
    "search",
    "sweep",
    "generalization",
    "noise",
    "sensitivity",
    "acrobot",
    "residual_verified_spectra",
)

GRID_KEYS = (
    "seed",
    "random_state",
    "train_seed",
    "test_seed",
    "data_seed",
    "replicate",
    "system",
    "task",
    "n_clusters",
    "C",
    "omega",
    "noise_level",
    "train_count",
    "train_fraction",
    "horizon",
)

SEED_RE = re.compile(r"(?:^|[/_-])seed[_-]?(\d+)(?:[/_.-]|$)", re.I)
C_RE = re.compile(r"(?:^|[/_-])C(\d+)(?:[/_.-]|$)", re.I)
OMEGA_RE = re.compile(r"(?:^|[/_-])omega([0-9]+(?:p[0-9]+)?)(?:[/_.-]|$)", re.I)


def decode_omega(value: str) -> str:
    return value.replace("p", ".")


def sort_values(values: set[str]) -> list[str]:
    def key(value: str):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)
    return sorted(values, key=key)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    groups: dict[tuple[str, str, str], dict[str, object]] = {}

    for row in rows:
        path = row["path"]
        lowered = path.lower()
        keys = json.loads(row.get("run_key_values_json") or "{}")

        path_has_grid = any(
            regex.search(path)
            for regex in (SEED_RE, C_RE, OMEGA_RE)
        )
        key_has_grid = any(key in keys for key in GRID_KEYS)
        keyword_hit = any(word in lowered for word in KEYWORDS)

        if not (keyword_hit and (path_has_grid or key_has_grid)):
            continue

        p = Path(path)
        top = p.parts[0] if p.parts else ""
        basename = p.name
        result_ids = row["supporting_result_ids"]
        group_key = (result_ids, top, basename)

        if group_key not in groups:
            groups[group_key] = {
                "supporting_result_ids": result_ids,
                "top_level": top,
                "basename": basename,
                "file_count": 0,
                "total_rows": 0,
                "path_C": set(),
                "path_omega": set(),
                "path_seed": set(),
                "key_values": defaultdict(set),
                "sample_path": path,
            }

        group = groups[group_key]
        group["file_count"] = int(group["file_count"]) + 1

        raw_count = str(row.get("row_count", "")).strip()
        if raw_count:
            group["total_rows"] = int(group["total_rows"]) + int(raw_count)

        for match in C_RE.finditer(path):
            group["path_C"].add(match.group(1))
        for match in OMEGA_RE.finditer(path):
            group["path_omega"].add(decode_omega(match.group(1)))
        for match in SEED_RE.finditer(path):
            group["path_seed"].add(match.group(1))

        key_values = group["key_values"]
        for key in GRID_KEYS:
            for value in keys.get(key, []):
                key_values[key].add(str(value))

    output_rows: list[dict[str, object]] = []

    for _, group in sorted(groups.items()):
        key_values = {
            key: sort_values(values)
            for key, values in group["key_values"].items()
            if values
        }

        output_rows.append(
            {
                "supporting_result_ids": group["supporting_result_ids"],
                "top_level": group["top_level"],
                "basename": group["basename"],
                "file_count": group["file_count"],
                "total_rows": group["total_rows"],
                "path_C": ",".join(sort_values(group["path_C"])),
                "path_omega": ",".join(sort_values(group["path_omega"])),
                "path_seed": ",".join(sort_values(group["path_seed"])),
                "row_key_values_json": json.dumps(
                    key_values,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "sample_path": group["sample_path"],
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "supporting_result_ids",
        "top_level",
        "basename",
        "file_count",
        "total_rows",
        "path_C",
        "path_omega",
        "path_seed",
        "row_key_values_json",
        "sample_path",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {OUTPUT}")
    print(f"Candidate groups: {len(output_rows)}")
    print()

    for row in output_rows:
        print(
            f"{row['supporting_result_ids']} | "
            f"{row['top_level']} | "
            f"{row['basename']} | "
            f"files={row['file_count']} | rows={row['total_rows']}"
        )

        path_dims = []
        if row["path_C"]:
            path_dims.append(f"path_C=[{row['path_C']}]")
        if row["path_omega"]:
            path_dims.append(f"path_omega=[{row['path_omega']}]")
        if row["path_seed"]:
            path_dims.append(f"path_seed=[{row['path_seed']}]")
        if path_dims:
            print("  " + " ".join(path_dims))

        values = json.loads(row["row_key_values_json"])
        if values:
            compact = "; ".join(
                f"{key}={vals}"
                for key, vals in values.items()
            )
            print("  row_keys: " + compact)

        print("  sample: " + str(row["sample_path"]))


if __name__ == "__main__":
    main()
