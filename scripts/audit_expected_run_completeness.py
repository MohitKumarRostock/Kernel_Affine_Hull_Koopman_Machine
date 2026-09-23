#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reproduction" / "run_accounting"

ALIASES = {
    "C": (
        "n_clusters",
        "c",
        "clusters",
        "cluster_count",
        "num_clusters",
    ),
    "omega": ("omega",),
    "seed": (
        "seed",
        "random_state",
        "train_seed",
        "data_seed",
        "replicate",
    ),
    "noise_level": (
        "noise_level",
        "training_noise",
        "noise",
        "sigma",
    ),
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

CAMPAIGNS = {
    "duffing_reference_search": {
        "dims": {
            "C": [15, 25, 35],
            "omega": [4.0, 8.0, 12.0],
            "seed": [0, 1, 2],
        },
        "name_hints": ("duffing",),
    },
    "duffing_expanded_search": {
        "dims": {
            "C": [8, 10, 12, 15, 20, 25, 30, 40],
            "omega": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
            "seed": [0, 1, 2],
        },
        "name_hints": ("duffing", "tuning"),
    },
    "vanderpol_clean_search": {
        "dims": {
            "C": [10, 15, 20, 25, 30, 40, 50],
            "omega": [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
            "seed": [0, 1, 2],
        },
        "name_hints": ("vanderpol",),
    },
    "vanderpol_noise_aware_search": {
        "dims": {
            "C": [10, 15, 20, 25, 30, 40],
            "omega": [0.25, 0.5, 1.0, 2.0, 4.0],
            "noise_level": [0.0, 0.005, 0.01, 0.02, 0.05],
            "seed": [0, 1, 2],
        },
        "name_hints": ("vanderpol", "noise"),
    },
    "acrobot_search": {
        "dims": {
            "C": [20, 40, 50, 75, 100, 150],
            "omega": [0.5, 1.0, 2.0, 4.0, 8.0],
            "seed": [0],
        },
        "name_hints": ("acrobot",),
    },
}

R15_TASKS = ("CartPole-v1", "MountainCar-v0")
R15_C = (10, 15, 20, 30, 50, 80, 100)
R15_OMEGA = (2.0, 4.0, 8.0, 12.0)
R15_HORIZONS = {"1", "5", "10", "20", "50"}

R16_SEEDS = (0, 1, 2)
R16_HORIZONS = {"1", "5", "10", "20", "50"}


def tracked_csvs() -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "*.csv"],
        cwd=ROOT,
        text=True,
    )
    paths = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        path = ROOT / line
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def canonical_header_map(fieldnames: list[str]) -> dict[str, str]:
    by_lower = {name.strip().lower(): name for name in fieldnames}
    found: dict[str, str] = {}
    for dim, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in by_lower:
                found[dim] = by_lower[alias.lower()]
                break
    return found


def parse_number(value: str, dim: str):
    value = value.strip()
    if value == "":
        raise ValueError
    if dim in {"C", "seed"}:
        return int(round(float(value)))
    return float(value)


def path_value(rel: str, dim: str):
    pattern = PATH_PATTERNS.get(dim)
    if pattern is None:
        return None
    match = pattern.search(rel)
    if match is None:
        return None
    value = match.group("v").replace("p", ".")
    return parse_number(value, dim)


def load_projected_rows(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    rel = str(path.relative_to(ROOT))
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            hmap = canonical_header_map(fieldnames)
            rows: list[dict[str, object]] = []
            for raw in reader:
                projected: dict[str, object] = {}
                for dim in ALIASES:
                    column = hmap.get(dim)
                    if column is not None:
                        try:
                            projected[dim] = parse_number(
                                str(raw.get(column, "")),
                                dim,
                            )
                        except (ValueError, TypeError):
                            pass
                    if dim not in projected:
                        pv = path_value(rel, dim)
                        if pv is not None:
                            projected[dim] = pv
                rows.append(projected)
            return rows, fieldnames
    except (UnicodeDecodeError, csv.Error):
        return [], []


def combo_product(dims: dict[str, list[object]]) -> set[tuple[object, ...]]:
    keys = list(dims)
    return {
        tuple(values)
        for values in itertools.product(*(dims[key] for key in keys))
    }


def normalize_tuple(values: tuple[object, ...]) -> tuple[object, ...]:
    out = []
    for value in values:
        if isinstance(value, float):
            out.append(round(value, 12))
        else:
            out.append(value)
    return tuple(out)


def group_observations(
    csv_paths: list[Path],
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}

    for path in csv_paths:
        rel = str(path.relative_to(ROOT))
        top = Path(rel).parts[0]
        rows, fieldnames = load_projected_rows(path)

        group = groups.setdefault(
            top,
            {
                "paths": [],
                "rows": [],
                "fieldnames": set(),
            },
        )
        group["paths"].append(rel)
        group["rows"].extend(rows)
        group["fieldnames"].update(fieldnames)

    return groups


def campaign_match(
    name: str,
    spec: dict[str, object],
    groups: dict[str, dict[str, object]],
) -> dict[str, object]:
    dims = spec["dims"]
    keys = list(dims)
    expected = {
        normalize_tuple(item)
        for item in combo_product(dims)
    }

    candidates = []

    for top, group in groups.items():
        observed_counter: Counter[tuple[object, ...]] = Counter()

        for row in group["rows"]:
            if not all(key in row for key in keys):
                continue
            combo = normalize_tuple(tuple(row[key] for key in keys))
            observed_counter[combo] += 1

        observed = set(observed_counter)
        covered = expected & observed
        missing = expected - observed

        if not covered:
            continue

        relevant_extra = {
            combo
            for combo in observed - expected
            if all(
                combo[i] in {
                    normalize_tuple((v,))[0]
                    for v in dims[key]
                }
                for i, key in enumerate(keys)
            )
        }

        hints = tuple(str(x).lower() for x in spec["name_hints"])
        top_lower = top.lower()
        hint_score = sum(1 for hint in hints if hint in top_lower)

        multiplicities = [
            observed_counter[combo]
            for combo in covered
        ]

        candidates.append(
            {
                "group": top,
                "covered": len(covered),
                "expected": len(expected),
                "missing": len(missing),
                "relevant_extra": len(relevant_extra),
                "coverage_fraction": len(covered) / len(expected),
                "hint_score": hint_score,
                "min_multiplicity": min(multiplicities),
                "max_multiplicity": max(multiplicities),
                "sample_missing": [
                    list(item)
                    for item in sorted(missing, key=str)[:10]
                ],
            }
        )

    candidates.sort(
        key=lambda item: (
            item["covered"],
            item["hint_score"],
            -item["relevant_extra"],
        ),
        reverse=True,
    )

    best = candidates[0] if candidates else None
    status = "PASS" if best and best["missing"] == 0 else "GAP"

    return {
        "campaign": name,
        "status": status,
        "dimensions": keys,
        "expected_jobs": len(expected),
        "best_candidate": best,
        "top_candidates": candidates[:5],
    }


def inspect_csv(path: Path) -> tuple[int, list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return len(rows), list(reader.fieldnames or []), rows


def check_r15() -> dict[str, object]:
    base = ROOT / "kahkm_experiment_16a_sensitivity_outputs" / "runs"
    expected_configs = {
        (task, C, omega)
        for task in R15_TASKS
        for C in R15_C
        for omega in R15_OMEGA
    }

    seen_multistep = set()
    seen_one_step = set()
    problems = []

    if not base.is_dir():
        return {
            "campaign": "r15_classic_control_sensitivity",
            "status": "GAP",
            "expected_configs": len(expected_configs),
            "problems": [f"missing directory: {base.relative_to(ROOT)}"],
        }

    rx = re.compile(
        r"^(?P<task>.+)_C(?P<C>\d+)_omega(?P<omega>\d+(?:p\d+)?)$"
    )

    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        match = rx.match(directory.name)
        if match is None:
            continue
        combo = (
            match.group("task"),
            int(match.group("C")),
            float(match.group("omega").replace("p", ".")),
        )

        multistep = directory / "experiment_16_multistep_summary.csv"
        one_step = directory / "experiment_16_one_step_results.csv"

        if multistep.is_file():
            n, fields, rows = inspect_csv(multistep)
            horizons = {str(row.get("horizon", "")).strip() for row in rows}
            methods = {str(row.get("method", "")).strip() for row in rows}
            if n != 40 or horizons != R15_HORIZONS or len(methods) != 8:
                problems.append(
                    f"{directory.name}: multistep rows={n}, "
                    f"horizons={sorted(horizons)}, methods={len(methods)}"
                )
            seen_multistep.add(combo)

        if one_step.is_file():
            n, fields, rows = inspect_csv(one_step)
            methods = {str(row.get("method", "")).strip() for row in rows}
            if n != 7 or len(methods) != 7:
                problems.append(
                    f"{directory.name}: one-step rows={n}, methods={len(methods)}"
                )
            seen_one_step.add(combo)

    missing_multistep = sorted(expected_configs - seen_multistep, key=str)
    missing_one_step = sorted(expected_configs - seen_one_step, key=str)
    extra_multistep = sorted(seen_multistep - expected_configs, key=str)
    extra_one_step = sorted(seen_one_step - expected_configs, key=str)

    status = "PASS"
    if (
        missing_multistep
        or missing_one_step
        or extra_multistep
        or extra_one_step
        or problems
    ):
        status = "GAP"

    return {
        "campaign": "r15_classic_control_sensitivity",
        "status": status,
        "expected_configs": len(expected_configs),
        "multistep_configs_found": len(seen_multistep),
        "one_step_configs_found": len(seen_one_step),
        "missing_multistep": [list(x) for x in missing_multistep],
        "missing_one_step": [list(x) for x in missing_one_step],
        "extra_multistep": [list(x) for x in extra_multistep],
        "extra_one_step": [list(x) for x in extra_one_step],
        "problems": problems,
    }


def check_r16() -> dict[str, object]:
    problems = []
    found_seeds = []

    for seed in R16_SEEDS:
        directory = ROOT / f"kahkm_exp18_acrobot_selected_seed{seed}"
        multistep = directory / "experiment_18_multistep_summary.csv"
        one_step = directory / "experiment_18_one_step_results.csv"

        if not directory.is_dir():
            problems.append(f"missing directory: {directory.relative_to(ROOT)}")
            continue

        if not multistep.is_file():
            problems.append(f"missing {multistep.relative_to(ROOT)}")
            continue
        if not one_step.is_file():
            problems.append(f"missing {one_step.relative_to(ROOT)}")
            continue

        n_multi, _, multi_rows = inspect_csv(multistep)
        n_one, _, one_rows = inspect_csv(one_step)

        horizons = {
            str(row.get("horizon", "")).strip()
            for row in multi_rows
        }
        multi_methods = {
            str(row.get("method", "")).strip()
            for row in multi_rows
        }
        one_methods = {
            str(row.get("method", "")).strip()
            for row in one_rows
        }
        C_values = {
            str(row.get("n_clusters", "")).strip()
            for row in multi_rows + one_rows
        }
        omega_values = {
            str(row.get("omega", "")).strip()
            for row in multi_rows + one_rows
        }

        if n_multi != 40:
            problems.append(f"seed {seed}: multistep rows={n_multi}, expected 40")
        if horizons != R16_HORIZONS:
            problems.append(
                f"seed {seed}: horizons={sorted(horizons)}, "
                f"expected={sorted(R16_HORIZONS)}"
            )
        if len(multi_methods) != 8:
            problems.append(
                f"seed {seed}: multistep methods={len(multi_methods)}, expected 8"
            )
        if n_one != 8 or len(one_methods) != 8:
            problems.append(
                f"seed {seed}: one-step rows/methods={n_one}/{len(one_methods)}, "
                "expected 8/8"
            )
        if C_values != {"150"}:
            problems.append(f"seed {seed}: C values={sorted(C_values)}")
        if omega_values != {"0.5"}:
            problems.append(f"seed {seed}: omega values={sorted(omega_values)}")

        found_seeds.append(seed)

    return {
        "campaign": "r16_acrobot_selected_validation",
        "status": "PASS" if not problems and found_seeds == [0, 1, 2] else "GAP",
        "expected_seeds": [0, 1, 2],
        "found_seeds": found_seeds,
        "problems": problems,
    }


def check_spectral() -> dict[str, object]:
    path = (
        ROOT
        / "kahkm_experiment_19_residual_verified_spectra"
        / "experiment_19_mode_residuals.csv"
    )

    if not path.is_file():
        return {
            "campaign": "r21_r22_spectral_diagnostics",
            "status": "GAP",
            "problems": [f"missing {path.relative_to(ROOT)}"],
        }

    n, _, rows = inspect_csv(path)
    systems = {str(row.get("system", "")).strip() for row in rows}
    operators = {str(row.get("operator", "")).strip() for row in rows}

    problems = []
    if n != 40:
        problems.append(f"rows={n}, expected 40")
    if systems != {"duffing", "vanderpol"}:
        problems.append(f"systems={sorted(systems)}")
    if operators != {"nlms", "ridge_ls"}:
        problems.append(f"operators={sorted(operators)}")

    return {
        "campaign": "r21_r22_spectral_diagnostics",
        "status": "PASS" if not problems else "GAP",
        "rows": n,
        "systems": sorted(systems),
        "operators": sorted(operators),
        "problems": problems,
    }


def main() -> None:
    csv_paths = tracked_csvs()
    groups = group_observations(csv_paths)

    checks = []

    for name, spec in CAMPAIGNS.items():
        checks.append(campaign_match(name, spec, groups))

    checks.append(check_r15())
    checks.append(check_r16())
    checks.append(check_spectral())

    OUT.mkdir(parents=True, exist_ok=True)

    json_path = OUT / "expected_run_completeness.json"
    json_path.write_text(
        json.dumps(
            {
                "tracked_csv_count": len(csv_paths),
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    tsv_path = OUT / "expected_run_completeness.tsv"
    fields = [
        "campaign",
        "status",
        "expected",
        "observed_or_covered",
        "best_group",
        "detail",
    ]

    rows = []
    for check in checks:
        if "expected_jobs" in check:
            best = check.get("best_candidate") or {}
            rows.append(
                {
                    "campaign": check["campaign"],
                    "status": check["status"],
                    "expected": check["expected_jobs"],
                    "observed_or_covered": best.get("covered", 0),
                    "best_group": best.get("group", ""),
                    "detail": json.dumps(
                        best,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        elif check["campaign"] == "r15_classic_control_sensitivity":
            rows.append(
                {
                    "campaign": check["campaign"],
                    "status": check["status"],
                    "expected": check["expected_configs"],
                    "observed_or_covered": min(
                        check.get("multistep_configs_found", 0),
                        check.get("one_step_configs_found", 0),
                    ),
                    "best_group": "kahkm_experiment_16a_sensitivity_outputs",
                    "detail": json.dumps(
                        check,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        elif check["campaign"] == "r16_acrobot_selected_validation":
            rows.append(
                {
                    "campaign": check["campaign"],
                    "status": check["status"],
                    "expected": 3,
                    "observed_or_covered": len(check["found_seeds"]),
                    "best_group": "kahkm_exp18_acrobot_selected_seed{0,1,2}",
                    "detail": json.dumps(
                        check,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        else:
            rows.append(
                {
                    "campaign": check["campaign"],
                    "status": check["status"],
                    "expected": 40,
                    "observed_or_covered": check.get("rows", 0),
                    "best_group": "kahkm_experiment_19_residual_verified_spectra",
                    "detail": json.dumps(
                        check,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {tsv_path}")
    print()
    print("=== EXPECTED-RUN COMPLETENESS ===")

    for check in checks:
        if "expected_jobs" in check:
            best = check.get("best_candidate")
            if best is None:
                print(
                    f"{check['status']}: {check['campaign']} "
                    f"0/{check['expected_jobs']} "
                    "(no candidate group)"
                )
            else:
                print(
                    f"{check['status']}: {check['campaign']} "
                    f"{best['covered']}/{check['expected_jobs']} "
                    f"group={best['group']} "
                    f"multiplicity={best['min_multiplicity']}.."
                    f"{best['max_multiplicity']}"
                )
                if best["missing"]:
                    print(
                        "  sample_missing="
                        + json.dumps(best["sample_missing"])
                    )
        elif check["campaign"] == "r15_classic_control_sensitivity":
            print(
                f"{check['status']}: {check['campaign']} "
                f"multistep={check.get('multistep_configs_found', 0)}/"
                f"{check['expected_configs']} "
                f"one_step={check.get('one_step_configs_found', 0)}/"
                f"{check['expected_configs']}"
            )
            for problem in check.get("problems", [])[:10]:
                print("  " + problem)
        elif check["campaign"] == "r16_acrobot_selected_validation":
            print(
                f"{check['status']}: {check['campaign']} "
                f"seeds={check['found_seeds']}"
            )
            for problem in check.get("problems", [])[:10]:
                print("  " + problem)
        else:
            print(
                f"{check['status']}: {check['campaign']} "
                f"rows={check.get('rows', 0)}"
            )
            for problem in check.get("problems", [])[:10]:
                print("  " + problem)

    passed = sum(check["status"] == "PASS" for check in checks)
    print()
    print(f"PASS checks: {passed}/{len(checks)}")
    print(
        "NOTE: this validates archived schedule coverage; it does not by itself "
        "prove that failed executions were preserved. The final frozen campaign "
        "ledger must record every attempted run and exit status."
    )


if __name__ == "__main__":
    main()
