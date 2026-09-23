#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_C = {15, 25, 35}
EXPECTED_OMEGA = {4.0, 8.0, 12.0}
EXPECTED_SEED = {0, 1, 2}

ALIASES = {
    "C": ("n_clusters", "c", "clusters", "cluster_count", "num_clusters"),
    "omega": ("omega",),
    "seed": ("seed", "random_state", "train_seed", "data_seed", "replicate"),
}

PATH_PATTERNS = {
    "C": re.compile(r"(?:^|[/_-])C(?P<v>\d+)(?=[/_.-]|$)", re.I),
    "omega": re.compile(
        r"(?:^|[/_-])omega(?P<v>\d+(?:p\d+)?)(?=[/_.-]|$)",
        re.I,
    ),
    "seed": re.compile(
        r"(?:^|[/_-])seed[_-]?(?P<v>\d+)(?=[/_.-]|$)",
        re.I,
    ),
}


def tracked_files(pattern: str) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", pattern],
        cwd=ROOT,
        text=True,
    )
    return [
        ROOT / line
        for line in raw.splitlines()
        if line.strip() and (ROOT / line).is_file()
    ]


def normalize_float(value: str) -> float:
    return float(value.strip().replace("p", "."))


def path_value(rel: str, dim: str):
    match = PATH_PATTERNS[dim].search(rel)
    if match is None:
        return None
    value = match.group("v")
    if dim in {"C", "seed"}:
        return int(round(float(value.replace("p", "."))))
    return normalize_float(value)


def header_map(fieldnames: list[str]) -> dict[str, str]:
    by_lower = {name.strip().lower(): name for name in fieldnames}
    found = {}
    for dim, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in by_lower:
                found[dim] = by_lower[alias.lower()]
                break
    return found


def scan_csvs():
    groups = defaultdict(Counter)
    examples = defaultdict(list)

    for path in tracked_files("*.csv"):
        rel = str(path.relative_to(ROOT))
        top = Path(rel).parts[0]

        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                hmap = header_map(fields)

                for raw in reader:
                    values = {}

                    for dim in ("C", "omega", "seed"):
                        column = hmap.get(dim)
                        if column is not None:
                            text = str(raw.get(column, "")).strip()
                            if text:
                                try:
                                    values[dim] = (
                                        int(round(float(text)))
                                        if dim in {"C", "seed"}
                                        else float(text)
                                    )
                                except ValueError:
                                    pass

                        if dim not in values:
                            pv = path_value(rel, dim)
                            if pv is not None:
                                values[dim] = pv

                    if set(values) != {"C", "omega", "seed"}:
                        continue

                    combo = (
                        int(values["C"]),
                        round(float(values["omega"]), 12),
                        int(values["seed"]),
                    )

                    if (
                        combo[0] in EXPECTED_C
                        and combo[1] in EXPECTED_OMEGA
                        and combo[2] in EXPECTED_SEED
                    ):
                        groups[top][combo] += 1
                        if len(examples[top]) < 5:
                            examples[top].append(rel)

        except (UnicodeDecodeError, csv.Error):
            continue

    print("=== CSV CANDIDATES FOR REFERENCE GRID ===")
    expected = {
        (c, o, s)
        for c in EXPECTED_C
        for o in EXPECTED_OMEGA
        for s in EXPECTED_SEED
    }

    ranked = []
    for top, counter in groups.items():
        observed = set(counter)
        covered = expected & observed
        if not covered:
            continue
        ranked.append(
            (
                len(covered),
                top,
                min(counter[c] for c in covered),
                max(counter[c] for c in covered),
                sorted(expected - observed),
            )
        )

    ranked.sort(reverse=True)

    for covered, top, min_mult, max_mult, missing in ranked[:20]:
        has_c35 = any(combo[0] == 35 for combo in groups[top])
        print(
            f"{top}: coverage={covered}/27 "
            f"multiplicity={min_mult}..{max_mult} "
            f"contains_C35={has_c35}"
        )
        for sample in examples[top][:3]:
            print("  sample:", sample)
        if missing:
            print("  missing:", missing[:12])

    if not ranked:
        print("No tracked CSV group exposes all of C, omega, and seed for the reference grid.")


def walk_json(obj, path="$"):
    if isinstance(obj, dict):
        yield path, obj
        for key, value in obj.items():
            yield from walk_json(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from walk_json(value, f"{path}[{i}]")


def numeric_set(value):
    if not isinstance(value, list):
        return None
    out = set()
    for item in value:
        if isinstance(item, (int, float)):
            out.add(float(item))
        elif isinstance(item, str):
            try:
                out.add(float(item))
            except ValueError:
                return None
        else:
            return None
    return out


def scan_jsons():
    print()
    print("=== JSON GRID/METADATA CANDIDATES ===")

    hits = 0
    for path in tracked_files("*.json"):
        rel = str(path.relative_to(ROOT))
        if "duffing" not in rel.lower() and "tuning" not in rel.lower():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        file_hits = []

        for object_path, obj in walk_json(data):
            if not isinstance(obj, dict):
                continue

            lower = {str(k).lower(): v for k, v in obj.items()}

            c_value = None
            for key in ("n_clusters", "clusters", "c_grid", "n_clusters_grid"):
                if key in lower:
                    c_value = numeric_set(lower[key])
                    if c_value is not None:
                        break

            omega_value = None
            for key in ("omega", "omega_grid"):
                if key in lower:
                    omega_value = numeric_set(lower[key])
                    if omega_value is not None:
                        break

            seed_value = None
            for key in ("seeds", "seed_grid", "train_seeds"):
                if key in lower:
                    seed_value = numeric_set(lower[key])
                    if seed_value is not None:
                        break

            if (
                c_value == {15.0, 25.0, 35.0}
                or (
                    c_value is not None
                    and 35.0 in c_value
                    and omega_value is not None
                    and {4.0, 8.0, 12.0}.issubset(omega_value)
                )
            ):
                file_hits.append(
                    (
                        object_path,
                        sorted(c_value) if c_value is not None else None,
                        sorted(omega_value) if omega_value is not None else None,
                        sorted(seed_value) if seed_value is not None else None,
                    )
                )

        if file_hits:
            hits += 1
            print(rel)
            for hit in file_hits[:10]:
                print(
                    "  path=", hit[0],
                    " C=", hit[1],
                    " omega=", hit[2],
                    " seeds=", hit[3],
                    sep="",
                )

    if hits == 0:
        print("No matching tracked JSON grid metadata found.")


def scan_source_text():
    print()
    print("=== SOURCE/TEXT REFERENCES TO 15,25,35 ===")

    patterns = [
        re.compile(r"15\s*,\s*25\s*,\s*35"),
        re.compile(r"35\s*,\s*25\s*,\s*15"),
    ]

    hits = []
    for pattern in ("*.py", "*.md", "*.txt", "*.tsv"):
        for path in tracked_files(pattern):
            rel = str(path.relative_to(ROOT))
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(lines, start=1):
                if any(rx.search(line) for rx in patterns):
                    hits.append((rel, line_no, line.strip()))

    for rel, line_no, line in hits[:100]:
        print(f"{rel}:{line_no}: {line}")

    if not hits:
        print("No exact 15,25,35 sequence found in tracked text/source files.")


def main():
    scan_csvs()
    scan_jsons()
    scan_source_text()


if __name__ == "__main__":
    main()
