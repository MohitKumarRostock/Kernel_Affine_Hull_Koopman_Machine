#!/usr/bin/env python3
"""Check new-fitting-seed PenDigits confirmation without treating it as a new cohort."""
from __future__ import annotations

import csv
import hashlib
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
    config_path = HERE / "confirmation_configs/pendigits_v1.json"
    protocol_path = HERE / "pendigits_confirmation_protocol.json"
    cfg = json.loads(config_path.read_text())
    inherited_config = json.loads((ORIGINAL / "config.json").read_text())
    assert cfg["seeds"] == [4701, 4702, 4703]
    assert {k: v for k, v in cfg.items() if k not in {"seeds", "campaign"}} == {k: v for k, v in inherited_config.items() if k not in {"seeds", "campaign"}}
    baseline = json.loads((HERE / "confirmation_results/standard/pendigits_v1/results.json").read_text())
    baseline_frozen = json.loads((HERE / "confirmation_results/standard/pendigits_v1/frozen_candidates.json").read_text())
    original = json.loads((ORIGINAL / "results.json").read_text())
    assert baseline["held_out_record_ids"] == original["held_out_record_ids"]
    base_records = {key(record): record for record in baseline["results"]}
    sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {"cohort_scope": "New fitting seeds on the same held-out writers; not a new-cohort replication.",
              "seeds": cfg["seeds"], "only_config_changes": ["campaign", "seeds"],
              "held_out_record_ids_identical_to_original": True,
              "protocol_sha256": sha256(protocol_path), "modes": {}}
    for mode in MODES:
        folder = HERE / "confirmation_results" / mode / "pendigits_v1"
        result = json.loads((folder / "results.json").read_text())
        frozen = json.loads((folder / "frozen_candidates.json").read_text())
        assert result["configuration"] == cfg
        assert result["held_out_record_ids"] == baseline["held_out_record_ids"]
        assert [r["training_record_ids"] for r in frozen] == [r["training_record_ids"] for r in baseline_frozen]
        manifest = json.loads((folder / "intervention_manifest.json").read_text())
        assert manifest["protocol_sha256"] == sha256(protocol_path)
        assert manifest["confirmation_config_sha256"] == sha256(config_path)
        assert manifest["adapter_sha256"] == sha256(HERE / "scaling_features.py")
        for name, expected in manifest["original_source_sha256"].items():
            assert sha256(HERE.parent / "kahm-study" / name) == expected, name
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
            "matching_training_and_held_out_records_across_arms": True,
            "source_config_protocol_hashes_verified": True,
        }
    target = HERE / "confirmation_results/pendigits_confirmation_verification.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
