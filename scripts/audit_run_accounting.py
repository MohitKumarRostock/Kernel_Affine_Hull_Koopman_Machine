#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "reproduction" / "reported_results.tsv"
OUT = ROOT / "reproduction" / "run_accounting"

DATA_SUFFIXES = {".csv", ".tsv", ".json", ".npz", ".npy"}

RUN_KEY_CANDIDATES = (
    "seed",
    "random_state",
    "random_states",
    "train_seed",
    "train_seeds",
    "train_seeds_used",
    "test_seed",
    "test_seeds",
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

_BRACE_RE = re.compile(r"\{([^{}]+)\}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_braces(text: str) -> list[str]:
    match = _BRACE_RE.search(text)
    if match is None:
        return [text]

    choices = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if not choices:
        return [text]

    out: list[str] = []
    for choice in choices:
        replaced = text[: match.start()] + choice + text[match.end() :]
        out.extend(expand_braces(replaced))
    return out


def declared_raw_references(value: str) -> list[str]:
    references: list[str] = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        lowered = part.lower()
        if (
            lowered.startswith("not applicable")
            or lowered.startswith("protocol generated")
            or lowered == "n/a"
        ):
            continue
        references.extend(expand_braces(part))
    return references


def files_for_reference(reference: str) -> tuple[list[Path], str | None]:
    path = ROOT / reference

    if path.is_file():
        return [path], None

    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in DATA_SUFFIXES
        )
        if not files:
            return [], f"directory contains no supported data files: {reference}"
        return files, None

    return [], f"missing declared raw reference: {reference}"


def inspect_table(path: Path) -> tuple[int | None, list[str], dict[str, list[str]]]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return None, [], {}

    delimiter = "\t" if suffix == ".tsv" else ","

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    distinct: dict[str, list[str]] = {}

    for key in RUN_KEY_CANDIDATES:
        if key not in columns:
            continue

        values = sorted(
            {
                str(row.get(key, "")).strip()
                for row in rows
                if str(row.get(key, "")).strip() != ""
            }
        )

        if len(values) <= 40:
            distinct[key] = values
        else:
            distinct[key] = [
                f"<{len(values)} distinct values>"
            ]

    return len(rows), columns, distinct


def main() -> None:
    if not MATRIX.is_file():
        raise FileNotFoundError(MATRIX)

    with MATRIX.open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle, delimiter="\t"))

    pass_results = [row for row in results if row.get("status") == "PASS"]

    file_to_results: dict[Path, set[str]] = defaultdict(set)
    coverage_rows: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []

    for row in pass_results:
        result_id = row["result_id"]
        raw_value = row.get("raw_output", "")
        references = declared_raw_references(raw_value)

        if not references:
            coverage_rows.append(
                {
                    "result_id": result_id,
                    "declared_reference": "",
                    "resolved_files": 0,
                    "status": "NO_FILE_REFERENCE",
                    "note": raw_value,
                }
            )
            continue

        for reference in references:
            files, error = files_for_reference(reference)

            if error is not None:
                missing.append(
                    {
                        "result_id": result_id,
                        "reference": reference,
                        "error": error,
                    }
                )
                coverage_rows.append(
                    {
                        "result_id": result_id,
                        "declared_reference": reference,
                        "resolved_files": 0,
                        "status": "MISSING",
                        "note": error,
                    }
                )
                continue

            for path in files:
                file_to_results[path].add(result_id)

            coverage_rows.append(
                {
                    "result_id": result_id,
                    "declared_reference": reference,
                    "resolved_files": len(files),
                    "status": "FOUND",
                    "note": "-",
                }
            )

    source_rows: list[dict[str, object]] = []
    total_tabular_rows = 0

    for path in sorted(file_to_results):
        row_count, columns, distinct = inspect_table(path)
        if row_count is not None:
            total_tabular_rows += row_count

        source_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "supporting_result_ids": ",".join(sorted(file_to_results[path])),
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "row_count": "" if row_count is None else row_count,
                "columns": ",".join(columns),
                "run_key_values_json": json.dumps(
                    distinct,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)

    source_path = OUT / "source_files.tsv"
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "path",
            "supporting_result_ids",
            "suffix",
            "bytes",
            "sha256",
            "row_count",
            "columns",
            "run_key_values_json",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(source_rows)

    coverage_path = OUT / "result_coverage.tsv"
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "result_id",
            "declared_reference",
            "resolved_files",
            "status",
            "note",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    summary = {
        "matrix": str(MATRIX.relative_to(ROOT)),
        "pass_result_count": len(pass_results),
        "unique_resolved_source_file_count": len(source_rows),
        "tabular_source_row_count": total_tabular_rows,
        "missing_reference_count": len(missing),
        "missing_references": missing,
        "purpose": (
            "Inventory every raw/per-run source declared by retained results "
            "without selecting or filtering individual rows. This is the "
            "foundation for expected-run completeness and no-cherry-picking "
            "verification."
        ),
    }

    summary_path = OUT / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {source_path}")
    print(f"Wrote {coverage_path}")
    print(f"Wrote {summary_path}")
    print(f"PASS results: {len(pass_results)}")
    print(f"Unique resolved source files: {len(source_rows)}")
    print(f"Tabular source rows: {total_tabular_rows}")
    print(f"Missing references: {len(missing)}")

    if missing:
        print()
        print("=== MISSING REFERENCES ===")
        for item in missing:
            print(
                item["result_id"],
                item["reference"],
                item["error"],
                sep="\t",
            )


if __name__ == "__main__":
    main()
