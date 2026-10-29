#!/usr/bin/env python3
"""Independently reconstruct the absolute-error and fitting-variability tables.

The verifier uses retained selected-model records and the standard library;
it does not import either table generator. Episode-score and speech-loss
verification provide separate checks of the underlying records.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("harmonic", "quasiperiodic", "nonlinear_ar", "logistic", "vanderpol", "duffing", "pendigits")
LABELS = {"Harmonic": "harmonic", "Quasiperiodic": "quasiperiodic", "Nonlinear AR": "nonlinear_ar",
          "Logistic": "logistic", "Van der Pol": "vanderpol", "Duffing": "duffing", "PenDigits": "pendigits"}
HEADS = {"Block": "nested", "Frobenius": "frobenius", "Rollout": "koopman"}
SPEECH = {"KAHM blocks": "kahm", "RBF blocks": "rbf", "Affine-hull blocks": "affine", "Ridge AR": "ar",
          "Quadratic": "quadratic", "Gaussian kernel": "kernel", "Persistence": "persistence", "Constant": "constant"}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def tex_rows(path):
    return [[x.strip() for x in re.sub(r"\\\\\s*$", "", line).split("&")]
            for line in path.read_text().splitlines() if "&" in line and not line.lstrip().startswith("\\")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new verification output file.")
    root = args.root.resolve()
    pooled_path = root / "results/matched_geometry_study/analysis/selected_pooled.csv"
    speech_path = root / "results/speech_forecasting_v1_replay/results.json"
    matched_table = root / "paper/tables/matched_geometry_combined.tex"
    speech_table = root / "paper/tables/speech_forecasting.tex"
    groups = defaultdict(list)
    for row in rows(pooled_path):
        if row["analysis"] == "primary" and row["horizon"] == "all":
            groups[(row["task"], row["head"], row["geometry"])].append((int(row["seed"]), float(row["test_mse"])))
    assert len(groups) == 63 and all(len(x) == 5 for x in groups.values())
    count, task = 0, None
    for row in tex_rows(matched_table):
        if row[0] in LABELS:
            task = LABELS[row[0]]
        if row[1] not in HEADS:
            continue
        head = HEADS[row[1]]
        means, cells = {}, []
        for geometry in ("kahm", "rbf", "affine"):
            values = [x[1] for x in sorted(groups[(task, head, geometry)])]
            means[geometry] = statistics.fmean(values)
            cells.append(f"{1000*means[geometry]:.2f} ({1000*statistics.stdev(values):.2f})")
        for geometry in ("rbf", "affine"):
            left, right = sorted(groups[(task, head, "kahm")]), sorted(groups[(task, head, geometry)])
            assert [x[0] for x in left] == [x[0] for x in right]
            wins = sum(a[1] < b[1] for a, b in zip(left, right))
            effect = 100 * (means["kahm"] / means[geometry] - 1)
            cells.append(f"{effect:+.1f}\\% ({wins})")
        assert row[2:] == cells, (task, head, row[2:], cells)
        count += 1
    assert count == 21
    speech = json.loads(speech_path.read_text())["results"]
    speech_count = 0
    for row in tex_rows(speech_table):
        if row[0] not in SPEECH:
            continue
        records = [r for r in speech if r["family"] == SPEECH[row[0]]]
        assert len(records) == 3
        cells = []
        for h in range(3):
            values = [60*math.sqrt(r["test_mse"][h]) for r in records]
            cells.append(f"{statistics.fmean(values):.2f} ({statistics.stdev(values):.2f})")
        cells += [f"{max(60*math.sqrt(r['mse_upper'][h]) for r in records):.2f}" for h in range(3)]
        cells += [f"{sum(max(r['mse_upper']) <= .3**2 for r in records)}/3"]
        assert row[1:] == cells, (row[0], row[1:], cells)
        speech_count += 1
    assert speech_count == 8
    inputs = [pooled_path, speech_path, matched_table, speech_table]
    report = {"status": "passed", "matched_rows": count, "matched_means": 63,
              "matched_sample_standard_deviations": 63, "matched_relative_effects_and_paired_win_counts": 42,
              "speech_rows": speech_count, "speech_mean_sd_pairs": 24,
              "speech_maximum_upper_endpoints": 24,
              "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Aggregation and table presentation from frozen selected-model records; no new fits or population superiority tests."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
