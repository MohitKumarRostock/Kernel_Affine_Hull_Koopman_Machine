"""Manufactured-only validation of the prospective Appliances index/metadata plan."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from appliances_metadata23 import (
    SPEC, array_sha256, calendar_features, eligible_anchors, feature_names,
    make_split_plan, planned_summary, specification, validate_source_metadata,
    verify_split_plan,
)


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rejects(fn) -> bool:
    try:
        fn()
    except (ValueError, KeyError):
        return True
    raise AssertionError("Manufactured corruption was accepted")


def main() -> None:
    receipt_path = PROJECT / "tests/appliances_metadata_preflight23_v2.json"
    arrays_path = PROJECT / "tests/appliances_planned_indices23_v2.npz"
    if receipt_path.exists() or arrays_path.exists():
        raise FileExistsError("Prospective metadata receipts are write-once")
    spec = specification()
    plan = make_split_plan(spec)
    verify_split_plan(plan, spec)
    assert len(feature_names(spec)) == 167
    assert np.array_equal(plan["reference_priority_order"], make_split_plan(spec)["reference_priority_order"])
    assert len(plan["test"]) // 144 == 26 and len(plan["test"]) % 144 == 24
    assert len(eligible_anchors(0, 691, 144, 36)) == 512
    checks = [
        "Exact outer-segment and common-anchor counts",
        "512 target-independent reference anchors and11150 matched head anchors",
        "Reference/head anchor partition; permitted within-training footprint overlap",
        "No outer train/validation/test footprint crosses its boundary",
        "Every seasonal-naive observation is in every model's allowed input history",
        "167 explicit feature coordinates; interpolated airport weather/random columns excluded",
        "Repeated metadata-only reference-bank generation is bit-identical",
        "26 complete daily-origin blocks and24 remaining origins",
        "Off-by-one check:691 raw rows would supply512 full-footprint anchors",
    ]
    corruptions = []
    for name, fn in [
        ("row count mutation", lambda: make_split_plan(dict(spec, source=dict(spec["source"], advertised_rows=19734)))),
        ("history mutation", lambda: make_split_plan(dict(spec, forecast=dict(spec["forecast"], lookback_records=143)))),
        ("horizon mutation", lambda: make_split_plan(dict(spec, forecast=dict(spec["forecast"], horizons_records=[1,6,37])))),
    ]:
        assert rejects(fn); corruptions.append(name)
    bad = {k:v.copy() for k,v in plan.items()}
    bad["test"][0] = 15930
    assert rejects(lambda: verify_split_plan(bad, spec)); corruptions.append("test history boundary crossing")
    bad = {k:v.copy() for k,v in plan.items()}
    bad["head_train"][0] = bad["reference_sorted"][0]
    assert rejects(lambda: verify_split_plan(bad, spec)); corruptions.append("reference label anchor in head training")

    # These dates are manufactured metadata, not dates read from the dataset.
    synthetic_start = datetime(2000, 1, 3, 0, 0, 0)
    timestamps = [synthetic_start + timedelta(minutes=10*i) for i in range(19735)]
    header = spec["source"]["expected_csv_columns"]
    validate_source_metadata(header, timestamps, spec)
    checks.append("Manufactured29-column/19735-timestamp metadata passes regular600-second cadence")
    assert rejects(lambda: validate_source_metadata(header[:-1], timestamps, spec)); corruptions.append("missing header")
    assert rejects(lambda: validate_source_metadata(header, timestamps[:-1], spec)); corruptions.append("missing timestamp")
    bad_dates = timestamps.copy(); bad_dates[500] = bad_dates[499]
    assert rejects(lambda: validate_source_metadata(header, bad_dates, spec)); corruptions.append("duplicate timestamp")
    bad_dates = timestamps.copy(); bad_dates[500] += timedelta(minutes=1)
    assert rejects(lambda: validate_source_metadata(header, bad_dates, spec)); corruptions.append("irregular timestamp")
    bad_dates = timestamps.copy(); bad_dates[0] = bad_dates[0].replace(tzinfo=timezone.utc)
    assert rejects(lambda: validate_source_metadata(header, bad_dates, spec)); corruptions.append("implicit timezone conversion")
    c = calendar_features([synthetic_start, synthetic_start + timedelta(days=1), synthetic_start + timedelta(days=7)])
    assert np.isfinite(c).all() and np.max(np.abs(c)) <= 1.0
    assert np.array_equal(c[0,:2],c[1,:2])
    assert np.array_equal(c[0],c[2])
    checks.append("Manufactured calendar features obey daily/weekly periodicity and finite unit bounds")

    source_lock_path = PROJECT / "data/source_archive/source_lock23.json"
    source_lock = json.loads(source_lock_path.read_text())
    assert source_lock["passed"] and source_lock["csv_decompressed"] is False
    assert source_lock["csv_parsed"] is False and source_lock["numeric_values_inspected"] is False
    archive = ROOT / source_lock["local_archive"]
    assert sha(archive) == source_lock["archive_sha256"]
    checks.append("Opaque archive hash matches source lock; CSV remains unopened")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open("xb") as f:
        np.savez_compressed(f, **plan)
    pinned = [Path(__file__).resolve(), PROJECT / "data/appliances_metadata23.py", SPEC, source_lock_path]
    receipt = {
        "schema": "step23-appliances-metadata-manufactured-preflight-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "numpy_version": np.__version__,
        "checks": checks,
        "deliberate_corruptions_rejected": corruptions,
        "source_csv_decompressed": False,
        "source_csv_rows_parsed": 0,
        "source_numeric_values_inspected": False,
        "benchmark_fits": 0,
        "planned_indices_summary": planned_summary(plan, spec),
        "planned_indices_file": str(arrays_path.relative_to(ROOT)),
        "planned_indices_file_sha256": sha(arrays_path),
        "source_sha256": {str(p.relative_to(ROOT)):sha(p) for p in pinned},
        "opaque_source_archive_sha256": source_lock["archive_sha256"],
        "scope": "Prospective metadata/index/calendar checks and synthetic corruptions only; no scientific forecast result",
    }
    with receipt_path.open("x") as f:
        json.dump(receipt, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"passed": True, "checks":len(checks), "corruptions_rejected":len(corruptions),
                      "feature_dimension":len(feature_names(spec)), "head_training_anchors":len(plan["head_train"]),
                      "validation_anchors":len(plan["validation"]),"test_anchors":len(plan["test"]),
                      "source_csv_rows_parsed":0,"benchmark_fits":0},indent=2))


if __name__ == "__main__":
    main()
