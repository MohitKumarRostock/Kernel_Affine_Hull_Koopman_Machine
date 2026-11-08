"""Prospective Appliances benchmark metadata and index plan; never reads data values.

This module opens only the adjacent JSON protocol when called without a spec.
The source CSV remains compressed and opaque during Step 23. Actual ingestion,
fitting, and held-out evaluation belong to a later separately gated stage.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = HERE / "appliances_protocol23.json"


def specification() -> dict:
    return json.loads(SPEC.read_text())


def eligible_anchors(start: int, stop: int, lookback: int, max_horizon: int) -> np.ndarray:
    """Half-open raw segment [start,stop); include full history/target footprint."""
    if not (0 <= start < stop and lookback >= 1 and max_horizon >= 1):
        raise ValueError("Invalid segment, lookback, or horizon")
    first = start + lookback - 1
    last_exclusive = stop - max_horizon
    if first >= last_exclusive:
        raise ValueError("Segment has no eligible common-horizon anchors")
    return np.arange(first, last_exclusive, dtype=np.int64)


def array_sha256(array: np.ndarray) -> str:
    """Portable byte identity for planned index arrays, without archive timestamps."""
    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes(order="C")).hexdigest()


def make_split_plan(spec: dict | None = None) -> dict[str, np.ndarray]:
    spec = specification() if spec is None else spec
    if spec["source"]["advertised_rows"] != 19735:
        raise ValueError("Unexpected published row count: no adaptive repartition")
    if spec["forecast"]["lookback_records"] != 144:
        raise ValueError("History length changed")
    if spec["forecast"]["horizons_records"] != [1, 6, 36]:
        raise ValueError("Forecast horizon roster changed")
    if spec["partitions"]["raw_segments_zero_based_half_open"] != {
            "outer_train": [0, 11841], "validation": [11841, 15788],
            "test": [15788, 19735]}:
        raise ValueError("Outer time partition changed")
    plan: dict[str, np.ndarray] = {}
    for name, (start, stop) in spec["partitions"]["raw_segments_zero_based_half_open"].items():
        plan[name] = eligible_anchors(start, stop, 144, 36)
    bank_spec = spec["partitions"]["reference_bank"]
    if bank_spec["size"] != 512 or bank_spec["generator"] != "numpy.random.PCG64":
        raise ValueError("Reference bank policy changed")
    rng = np.random.Generator(np.random.PCG64(int(bank_spec["seed"])))
    plan["reference_priority_order"] = rng.permutation(plan["outer_train"])[:512].copy()
    plan["reference_sorted"] = np.sort(plan["reference_priority_order"])
    plan["head_train"] = np.setdiff1d(plan["outer_train"], plan["reference_sorted"], assume_unique=True)
    verify_split_plan(plan, spec)
    return plan


def verify_split_plan(plan: dict[str, np.ndarray], spec: dict | None = None) -> None:
    spec = specification() if spec is None else spec
    counts = {"outer_train": 11662, "reference_priority_order": 512,
              "reference_sorted": 512, "head_train": 11150,
              "validation": 3768, "test": 3768}
    for name, count in counts.items():
        a = np.asarray(plan[name])
        if a.dtype.kind not in "iu" or a.shape != (count,) or len(np.unique(a)) != count:
            raise ValueError("Invalid planned index array: " + name)
    if not np.array_equal(np.sort(plan["reference_priority_order"]), plan["reference_sorted"]):
        raise ValueError("Reference priority order and sorted set differ")
    if np.intersect1d(plan["head_train"], plan["reference_sorted"]).size:
        raise ValueError("Reference anchors are included as head-training label anchors")
    if not np.array_equal(np.sort(np.concatenate((plan["head_train"], plan["reference_sorted"]))),
                          plan["outer_train"]):
        raise ValueError("Reference/head anchors do not partition outer-training anchors")
    for name, (start, stop) in spec["partitions"]["raw_segments_zero_based_half_open"].items():
        a = plan[name]
        if int(a.min()) - 143 < start or int(a.max()) + 36 >= stop:
            raise ValueError("History or forecast target crosses outer partition: " + name)
        for horizon in spec["forecast"]["horizons_records"]:
            daily_source = a + horizon - 144
            if np.any(daily_source < a - 143) or np.any(daily_source > a):
                raise ValueError("Seasonal baseline accesses information outside the input history")


def feature_names(spec: dict | None = None) -> list[str]:
    spec = specification() if spec is None else spec
    names = [f"Appliances_lag{lag:03d}" for lag in range(144)]
    names += spec["features"]["current_local_channels"]
    names += spec["features"]["calendar_channels"]
    if len(names) != 167 or len(set(names)) != 167:
        raise ValueError("Feature roster is not the declared 167 distinct coordinates")
    return names


def validate_source_metadata(header: Iterable[str], timestamps: Iterable[datetime],
                             spec: dict | None = None) -> None:
    """Checks a supplied header/timestamp stream, without opening or reading any CSV.

    Used only on manufactured metadata during Step 23. A future ingestion stage
    must provide the actual metadata and separately enforce numeric integrity.
    """
    spec = specification() if spec is None else spec
    if list(header) != spec["source"]["expected_csv_columns"]:
        raise ValueError("Source column names/order differ from the declared schema")
    ts = list(timestamps)
    if len(ts) != spec["source"]["advertised_rows"]:
        raise ValueError("Source row count differs from the locked metadata")
    for i, t in enumerate(ts):
        if not isinstance(t, datetime) or t.tzinfo is not None:
            raise ValueError("Require source-naive datetime stamps without implicit timezone conversion")
        if t.second != 0 or t.microsecond != 0 or t.minute % 10:
            raise ValueError("Timestamp is not on the ten-minute wall-clock grid")
        if i and (t - ts[i - 1]).total_seconds() != 600:
            raise ValueError("Missing, duplicate, reordered, or irregular timestamp")


def calendar_features(timestamps: Iterable[datetime]) -> np.ndarray:
    """Four causal calendar coordinates at the origin, Monday weekday index zero."""
    result = []
    for t in timestamps:
        if not isinstance(t, datetime) or t.tzinfo is not None:
            raise ValueError("Expected a source-naive datetime")
        day_fraction = (3600 * t.hour + 60 * t.minute + t.second) / 86400.0
        day_angle = 2.0 * np.pi * day_fraction
        week_angle = 2.0 * np.pi * (t.weekday() + day_fraction) / 7.0
        result.append((np.sin(day_angle), np.cos(day_angle), np.sin(week_angle), np.cos(week_angle)))
    return np.asarray(result, dtype=np.float64).reshape(-1, 4)


def planned_summary(plan: dict[str, np.ndarray], spec: dict | None = None) -> dict:
    spec = specification() if spec is None else spec
    verify_split_plan(plan, spec)
    return {
        "source_rows_parsed": 0,
        "numeric_source_values_inspected": False,
        "benchmark_fits": 0,
        "feature_dimension": len(feature_names(spec)),
        "arrays": {name: {"count": int(len(a)), "min": int(np.min(a)), "max": int(np.max(a)),
                           "canonical_little_endian_int64_sha256": array_sha256(a)}
                   for name, a in plan.items()},
        "outer_partition_footprints_disjoint": True,
        "reference_and_head_raw_histories_may_overlap_within_outer_train": True,
        "reference_and_head_anchor_sets_disjoint": True,
        "daily_seasonal_source_within_input_history": True,
        "selection_used_only_indices": True,
    }
