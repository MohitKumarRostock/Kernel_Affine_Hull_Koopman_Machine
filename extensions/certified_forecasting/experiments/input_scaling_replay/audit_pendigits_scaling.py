#!/usr/bin/env python3
"""Independently check PenDigits scaling run accounting and cohort arithmetic."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "kahm-study/extensions/certified_forecasting/results/pendigits_v1b"
MODES = ["standard", "minmax", "minmax_clip", "raw"]
FIELDS = ["validation_mse", "cohort_writer_mean_mse", "conditional_iid_writer_mse_upper"]


def key(record):
    return record["seed"], record["coordinate"], record["name"]


def main():
    original = json.loads((ORIGINAL / "results.json").read_text())
    baseline = json.loads((HERE / "results/standard/pendigits_v1/results.json").read_text())
    reference = {key(record): record for record in original["results"]}
    base_records = {key(record): record for record in baseline["results"]}
    exact_entries = 0
    for identity, record in base_records.items():
        expected = reference[identity]
        for field in FIELDS:
            assert np.array_equal(expected[field], record[field]), (identity, field)
            exact_entries += len(record[field])
        assert expected["selected_before_test"] == record["selected_before_test"]
    report = {"baseline_vs_retained_exact_scalar_entries": exact_entries,
              "baseline_selection_flags_identical": True, "modes": {}}
    for mode in MODES:
        folder = HERE / "results" / mode / "pendigits_v1"
        result = json.loads((folder / "results.json").read_text())
        frozen = json.loads((folder / "frozen_candidates.json").read_text())
        with (folder / "writer_losses.csv").open() as stream:
            losses = list(csv.DictReader(stream))
        assert len(frozen) == 6 and all(x["n_candidates"] == 40 for x in frozen)
        assert len(result["results"]) == 240 and len(losses) == 9360
        assert result["n_held_out_writers"] == 13
        assert result["n_held_out_eligible_digits"] == 1751
        assert result["held_out_writer_ids_with_no_eligible_digits"] == [5]
        assert result["origins"] == [7, 11, 15, 19, 23, 27]
        assert result["n_upper_bound_statements"] == 720
        groups = {}
        for row in losses:
            group = (int(row["seed"]), int(row["coordinate"]), row["candidate"], int(row["horizon"]))
            value = float(row["writer_mean_squared_loss"])
            assert np.isfinite(value) and 0 <= value <= 4
            groups.setdefault(group, {})[int(row["writer"])] = value
        assert len(groups) == 720 and all(len(v) == 13 for v in groups.values())
        max_discrepancy, identical_baseline_scalars = 0.0, 0
        max_gap, max_norm = 0.0, 0.0
        for record in result["results"]:
            identity = key(record)
            for index, horizon in enumerate(result["configuration"]["horizons"]):
                writer_values = groups[identity + (horizon,)]
                cohort_mean = float(np.mean([writer_values[w] for w in sorted(writer_values)]))
                max_discrepancy = max(max_discrepancy, abs(cohort_mean - record["cohort_writer_mean_mse"][index]))
            if not record["family"].startswith("kahm"):
                for field in FIELDS:
                    assert np.array_equal(record[field], base_records[identity][field]), (mode, identity, field)
                    identical_baseline_scalars += len(record[field])
                assert record["selected_before_test"] == base_records[identity]["selected_before_test"]
            if "optimization_gap" in record:
                max_gap = max(max_gap, record["optimization_gap"])
            if "norm_sum" in record:
                max_norm = max(max_norm, record["norm_sum"])
                assert record["norm_sum"] <= 4 + 1e-9
            assert record.get("converged", True)
        assert max_discrepancy < 1e-14
        report["modes"][mode] = {
            "n_candidates": len(result["results"]), "frozen_coordinate_seed_families": len(frozen),
            "n_writer_loss_rows": len(losses), "n_writers": 13,
            "non_kahm_scalars_bitwise_identical_to_baseline": identical_baseline_scalars,
            "non_kahm_selection_flags_identical": True,
            "maximum_cohort_mse_reconstruction_difference": max_discrepancy,
            "maximum_block_norm_sum": max_norm, "maximum_optimization_gap": max_gap,
            "all_reported_solvers_converged": True,
        }
    target = HERE / "results/pendigits_scaling_verification.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
