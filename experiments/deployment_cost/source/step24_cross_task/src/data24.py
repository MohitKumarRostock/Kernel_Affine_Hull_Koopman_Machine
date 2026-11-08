"""Sealed, role-specific data access for the Step 24 forecasting panel.

Importing this module reads no data and constructs no RNG. Public development
loaders require the implementation seal; test loaders additionally require the
complete six-task fit seal. They return access receipts in ``metadata`` and
perform no file writes. Source CSV fields outside the permitted roles are
never converted to numbers or used to construct predictive features.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
from types import ModuleType
from typing import Callable, Iterable, Mapping
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASE = PROJECT.parent
STEP23 = BASE / "step23_cross_task"
TASK_IDS = tuple([f"hmm_{i}" for i in range(5)] + ["appliances"])
PROTOCOL_PATH = STEP23 / "PROTOCOL.json"
PRIOR_SEAL_PATH = STEP23 / "protocol/protocol_seal23.json"
ARCHIVE_PATH = STEP23 / "data/source_archive/uci374_appliances_energy_prediction.zip"
ARCHIVE_SHA256 = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
PROTOCOL_SHA256 = "e7109b924bc0899aa0d97c9091828c29068bd0f2b67714e754e59d594df588ef"
NEEDED_SOURCES = (
    HERE / "data24.py",
    PROTOCOL_PATH,
    PRIOR_SEAL_PATH,
    STEP23 / "controlled/hmm23.py",
    STEP23 / "controlled/hmm_spec23.json",
    STEP23 / "data/appliances_metadata23.py",
    STEP23 / "data/appliances_protocol23.json",
    STEP23 / "data/source_archive/source_lock23.json",
)
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\Z")


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path | str) -> dict:
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("A JSON object is required")
    return result


def source_bindings() -> dict[str, str]:
    """Current source digests for the parent's seal; no CSV/RNG access."""
    return {str(p.relative_to(BASE)): file_sha256(p) for p in NEEDED_SOURCES}


def _verify_prior_sources(bindings: Mapping[str, str]) -> None:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("The frozen Step 23 protocol changed")
    prior = _read_json(PRIOR_SEAL_PATH)
    if prior.get("passed") is not True or prior.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("The Step 23 source seal is invalid")
    pinned = {row["path"]: row["sha256"] for row in prior["files"]}
    for path in NEEDED_SOURCES:
        if path == PRIOR_SEAL_PATH or STEP23 not in path.parents:
            continue
        relative = str(path.relative_to(STEP23))
        if pinned.get(relative) != bindings[str(path.relative_to(BASE))]:
            raise RuntimeError("A frozen benchmark source changed: " + relative)


def _validate_implementation_seal(seal: dict, bindings: Mapping[str, str]) -> None:
    if seal.get("schema") != "step24-implementation-seal-v1" or seal.get("passed") is not True:
        raise PermissionError("Development data require a passed implementation seal")
    if seal.get("protocol_sha256") != PROTOCOL_SHA256:
        raise PermissionError("Implementation seal refers to another protocol")
    if seal.get("archive_sha256") != ARCHIVE_SHA256:
        raise PermissionError("Implementation seal refers to another source archive")
    listed = seal.get("source_sha256")
    if not isinstance(listed, dict):
        raise PermissionError("Implementation seal has no source binding")
    for name, digest in bindings.items():
        if listed.get(name) != digest:
            raise PermissionError("Implementation seal does not bind the current source: " + name)


def _check_gate(stage: str, task_id: str, gate_path: Path | str | None = None) -> dict:
    if task_id not in TASK_IDS or stage not in ("development", "test"):
        raise ValueError("Unknown fixed task or data-access stage")
    default_name = "implementation_seal.json" if stage == "development" else "fit_seal.json"
    path = PROJECT / "run" / default_name if gate_path is None else Path(gate_path)
    if not path.is_file():
        raise PermissionError("Required data-access seal is absent: " + path.name)
    seal = _read_json(path)
    bindings = source_bindings()
    _verify_prior_sources(bindings)
    if stage == "development":
        _validate_implementation_seal(seal, bindings)
        implementation_path = path
    else:
        if seal.get("schema") != "step24-fit-seal-v1" or seal.get("passed") is not True:
            raise PermissionError("Test data require the completed panel fit seal")
        if seal.get("protocol_sha256") != PROTOCOL_SHA256:
            raise PermissionError("Fit seal refers to another protocol")
        if seal.get("archive_sha256") != ARCHIVE_SHA256:
            raise PermissionError("Fit seal refers to another source archive")
        if seal.get("all_task_ids") != list(TASK_IDS):
            raise PermissionError("All six task procedures must be sealed before any test data")
        implementation_path = path.parent / "implementation_seal.json"
        if not implementation_path.is_file():
            raise PermissionError("Fit seal is missing its parent implementation seal")
        if seal.get("implementation_seal_sha256") != file_sha256(implementation_path):
            raise PermissionError("Fit seal does not bind its current implementation seal")
        implementation = _read_json(implementation_path)
        _validate_implementation_seal(implementation, bindings)
        if seal.get("source_sha256") != implementation.get("source_sha256"):
            raise PermissionError("Fit and implementation source bindings differ")
    return {
        "stage": stage,
        "task_id": task_id,
        "gate_file": str(path.resolve()),
        "gate_sha256": file_sha256(path),
        "implementation_seal_sha256": file_sha256(implementation_path),
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": dict(bindings),
        "archive_sha256": ARCHIVE_SHA256,
    }


@lru_cache(maxsize=None)
def _source_module(name: str, relative: str) -> ModuleType:
    """Load pinned local source without writing an import-cache file."""
    path = STEP23 / relative
    qualified_name = "_step24_data_" + name
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise ImportError("Conflicting inherited data module")
        return existing
    module = ModuleType(qualified_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[qualified_name] = module
    try:
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def hmm_module() -> ModuleType:
    return _source_module("hmm23", "controlled/hmm23.py")


def _metadata_module() -> ModuleType:
    return _source_module("appliances_metadata23", "data/appliances_metadata23.py")


def _array_record(array: np.ndarray) -> dict:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(json.dumps({"shape": list(a.shape), "dtype": a.dtype.str}, sort_keys=True).encode())
    h.update(a.tobytes(order="C"))
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": h.hexdigest()}


def _finite_matrix(value: object, shape: tuple[int, ...]) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError("Generated input/target shape or finite-value mismatch")
    return np.ascontiguousarray(a)


def _hmm_roles(replication: int, roles: tuple[str, ...],
               provider: Callable[[int, str], dict]) -> dict:
    """Private packaging seam; manufactured tests supply a scripted provider."""
    h = hmm_module()
    loaded = {}
    for role in roles:
        record = provider(replication, role)
        if set(record) != {"X", "Y", "metadata"}:
            raise ValueError("Unexpected HMM data fields; latent/oracle inputs are forbidden")
        n = h.ROLE_COUNTS[role]
        x = _finite_matrix(record["X"], (n, h.HISTORY_LENGTH))
        if not np.isin(x, h.SYMBOL_VALUES).all():
            raise ValueError("HMM inputs are not raw observation symbols")
        if role == "support":
            if record["Y"] is not None:
                raise ValueError("Support future labels must not be generated or supplied")
            y = None
        else:
            y = _finite_matrix(record["Y"], (n, len(h.HORIZONS)))
            if not np.isin(y, h.SYMBOL_VALUES).all():
                raise ValueError("HMM targets are not raw observation symbols")
        metadata = dict(record["metadata"])
        if metadata.get("role") != role or metadata.get("replication") != replication:
            raise ValueError("HMM stream metadata do not match the requested role")
        if metadata.get("rows") != n or metadata.get("history_length") != h.HISTORY_LENGTH:
            raise ValueError("HMM stream metadata dimensions changed")
        if metadata.get("horizons") != ([] if role == "support" else list(h.HORIZONS)):
            raise ValueError("HMM target horizon order changed")
        if metadata.get("seed_entropy") != list(h.seed_entropy(replication, role)):
            raise ValueError("HMM stream identity differs from the fixed role seed")
        if (metadata.get("bit_generator") != "PCG64DXSM"
                or metadata.get("one_example_per_independent_trajectory") is not True
                or metadata.get("hidden_states_or_oracles_in_inputs") is not False):
            raise ValueError("HMM independence/information contract changed")
        if metadata.get("support_future_targets_generated") is not False:
            raise ValueError("HMM generator violates the support-target contract")
        ids = h.ROLE_CODES[role] * 1_000_000 + np.arange(n, dtype=np.int64)
        loaded[role] = {"X": x, "Y": y, "anchor_ids": ids, "metadata": metadata}
    return loaded


def _load_hmm(task_id: str, stage: str, gate: dict) -> dict:
    h = hmm_module()
    replication = int(task_id[4:])
    roles = ("support", "train", "validation") if stage == "development" else ("test",)
    started = time.perf_counter_ns()
    loaded = _hmm_roles(replication, roles, h.generate_role)
    generated_ns = time.perf_counter_ns() - started
    assembly_start = time.perf_counter_ns()
    result = {"horizons": list(h.HORIZONS), "primary_column": list(h.HORIZONS).index(h.PRIMARY_HORIZON)}
    if stage == "development":
        result.update({"support": loaded["support"]["X"],
                       "train_X": loaded["train"]["X"], "train_Y": loaded["train"]["Y"],
                       "validation_X": loaded["validation"]["X"], "validation_Y": loaded["validation"]["Y"],
                       "train_anchor_ids": loaded["train"]["anchor_ids"],
                       "validation_anchor_ids": loaded["validation"]["anchor_ids"]})
    else:
        result.update({"test_X": loaded["test"]["X"], "test_Y": loaded["test"]["Y"],
                       "test_anchor_ids": loaded["test"]["anchor_ids"]})
    assembly_ns = time.perf_counter_ns() - assembly_start
    metadata = {
        "schema": "step24-role-specific-data-access-v1", "task_id": task_id, "stage": stage,
        "gate": gate, "roles": {role: loaded[role]["metadata"] for role in roles},
        "anchor_id_definition": "Within each task, role_code * 1000000 + within-role row index",
        "history_order": "oldest first; persistence uses the last coordinate",
        "support_labels_provided": False, "oracle_or_latent_variables_in_inputs": False,
        "benchmark_trajectories_generated": sum(h.ROLE_COUNTS[role] for role in roles),
        "test_trajectories_generated": h.ROLE_COUNTS["test"] if stage == "test" else 0,
        "source_csv_rows_parsed": 0, "source_numeric_values_converted": 0,
        "test_source_numeric_values_converted": 0, "ECG_or_reserve_access": False,
        "generation_nanoseconds": int(generated_ns), "feature_assembly_nanoseconds": int(assembly_ns),
        "arrays": {key: _array_record(value) for key, value in result.items() if isinstance(value, np.ndarray)},
    }
    if stage == "development":
        metadata["support_anchor_ids"] = _array_record(loaded["support"]["anchor_ids"])
    result["metadata"] = metadata
    return result


def _record_metadata(line: str) -> tuple[str, int]:
    """Return only the first field and CSV column count, without numeric tokens.

    The state machine scans record syntax. It never splits or returns numeric
    fields. It accepts ordinary quoting/doubled quotes and rejects multiline or
    malformed records. Source records contain a timestamp followed by scalars.
    """
    if not isinstance(line, str):
        raise TypeError("CSV text records are required")
    line = line.rstrip("\r\n")
    if not line or "\n" in line or "\r" in line or "\x00" in line:
        raise ValueError("Empty, multiline, or invalid CSV record")
    columns, first_end = 1, None
    in_quotes, field_start, after_quote = False, True, False
    i = 0
    while i < len(line):
        char = line[i]
        if in_quotes:
            if char == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    i += 2
                    continue
                in_quotes, after_quote = False, True
            i += 1
            continue
        if char == ",":
            if first_end is None:
                first_end = i
            columns += 1
            field_start, after_quote = True, False
        elif after_quote:
            raise ValueError("Non-delimiter after a quoted CSV field")
        elif char == '"':
            if not field_start:
                raise ValueError("Quote inside an unquoted CSV field")
            in_quotes, field_start = True, False
        else:
            field_start = False
        i += 1
    if in_quotes or first_end is None:
        raise ValueError("Unclosed quote or missing timestamp delimiter")
    token = line[:first_end]
    if token.startswith('"'):
        if not token.endswith('"') or '"' in token[1:-1]:
            raise ValueError("Malformed quoted timestamp")
        token = token[1:-1]
    if not _STAMP.fullmatch(token):
        raise ValueError("Source timestamp is not the declared naive format")
    return token, columns


def _segment_layout(spec: dict) -> tuple[list[tuple[str, int, int]], list[str]]:
    segments = spec["partitions"]["raw_segments_zero_based_half_open"]
    if list(segments) != ["outer_train", "validation", "test"]:
        raise ValueError("Unexpected outer-segment roster/order")
    rows = spec["source"]["advertised_rows"]
    current, result = 0, []
    for name, interval in segments.items():
        if (len(interval) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in interval)
                or interval[0] != current or not interval[0] < interval[1]):
            raise ValueError("Outer segments must partition the ordered source record")
        start, stop = interval
        result.append((name, start, stop))
        current = stop
    if current != rows:
        raise ValueError("Outer segments do not cover the advertised source row count")
    names = [spec["forecast"]["target_column"]] + list(spec["features"]["current_local_channels"])
    if len(names) != len(set(names)) or any(name not in spec["source"]["expected_csv_columns"] for name in names):
        raise ValueError("Invalid selected numeric-column roster")
    return result, names


def _stream_appliances_records(lines: Iterable[str], spec: dict,
                               permitted_roles: tuple[str, ...]) -> dict:
    """Pure streaming parser; tests supply manufactured text and index metadata.

    Only permitted-role rows are passed to csv.reader or float conversion.
    Other records contribute timestamp and record-shape integrity metadata only.
    """
    if permitted_roles not in (("outer_train", "validation"), ("test",)):
        raise ValueError("Only a development or test access roster is permitted")
    layout, numeric_names = _segment_layout(spec)
    stream = iter(lines)
    try:
        header_line = next(stream)
    except StopIteration as exc:
        raise ValueError("Source CSV has no header") from exc
    header = next(csv.reader([header_line], strict=True))
    if header != spec["source"]["expected_csv_columns"]:
        raise ValueError("Source header differs from the declared column names/order")
    selected_indices = [header.index(name) for name in numeric_names]
    selected = {}
    for name, start, stop in layout:
        if name in permitted_roles:
            selected[name] = {"name": name, "start": start, "stop": stop,
                              "numeric_columns": list(numeric_names),
                              "numeric": np.empty((stop - start, len(numeric_names)), dtype=np.float64),
                              "timestamps": []}
    role_counts = {name: 0 for name, _, _ in layout}
    numeric_counts = {name: 0 for name, _, _ in layout}
    conversion_counts = {name: 0 for name, _, _ in layout}
    timestamp = previous = first = None
    current_segment, row_count = 0, 0
    for index, line in enumerate(stream):
        if index >= spec["source"]["advertised_rows"]:
            raise ValueError("Source row count exceeds the locked metadata")
        token, column_count = _record_metadata(line)
        if column_count != len(header):
            raise ValueError("Source record column count changed")
        timestamp = datetime.strptime(token, "%Y-%m-%d %H:%M:%S")
        if (timestamp.tzinfo is not None or timestamp.second != 0 or timestamp.microsecond != 0
                or timestamp.minute % 10 != 0):
            raise ValueError("Source timestamp is not on the ten-minute naive grid")
        if previous is not None and (timestamp - previous).total_seconds() != 600:
            raise ValueError("Missing, duplicated, reordered, or irregular source timestamp")
        if first is None:
            first = timestamp
        previous = timestamp
        while index >= layout[current_segment][2]:
            current_segment += 1
        role, start, stop = layout[current_segment]
        role_counts[role] += 1
        if role in selected:
            fields = next(csv.reader([line], strict=True))
            if len(fields) != len(header) or fields[0] != token:
                raise ValueError("Numeric-row parser disagrees with timestamp/schema metadata")
            values = []
            for column_index in selected_indices:
                try:
                    value = float(fields[column_index])
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid numeric value in permitted {role} row {index}, column {header[column_index]}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite numeric value in permitted {role} row {index}, column {header[column_index]}")
                values.append(value)
            selected[role]["numeric"][index - start] = values
            selected[role]["timestamps"].append(timestamp)
            numeric_counts[role] += 1
            conversion_counts[role] += len(values)
        row_count = index + 1
    if row_count != spec["source"]["advertised_rows"]:
        raise ValueError("Source row count differs from the locked metadata")
    for role, start, stop in layout:
        if role_counts[role] != stop - start:
            raise ValueError("Source metadata role count mismatch")
        if role in selected:
            segment = selected[role]
            if len(segment["timestamps"]) != stop - start or not np.isfinite(segment["numeric"]).all():
                raise ValueError("Incomplete permitted numeric source segment")
        elif numeric_counts[role] or conversion_counts[role]:
            raise AssertionError("Forbidden-role values were numerically parsed")
    return {
        "segments": selected,
        "metadata": {
            "source_rows_metadata_checked": row_count,
            "source_row_count_verified": True,
            "source_schema_verified": True,
            "source_cadence_seconds": 600,
            "source_first_timestamp": first.isoformat(sep=" "),
            "source_last_timestamp": timestamp.isoformat(sep=" "),
            "metadata_rows_by_role": role_counts,
            "numeric_rows_by_role": numeric_counts,
            "numeric_values_by_role": conversion_counts,
            "selected_numeric_columns": numeric_names,
            "permitted_numeric_roles": list(permitted_roles),
            "unpermitted_roles_produce_no_predictive_features": True,
            "source_csv_rows_parsed": int(sum(numeric_counts.values())),
            "source_numeric_values_converted": int(sum(conversion_counts.values())),
            "test_source_numeric_values_converted": int(conversion_counts["test"]),
            "timestamp_integrity_only_for_unpermitted_roles": True,
        },
    }


def _appliances_examples(segment: dict, anchors: np.ndarray, spec: dict,
                         *, include_targets: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Construct features inside one already-permitted raw segment."""
    anchors = np.asarray(anchors)
    if anchors.ndim != 1 or anchors.dtype.kind not in "iu" or not len(anchors) or len(np.unique(anchors)) != len(anchors):
        raise ValueError("Distinct integer forecast anchors are required")
    lookback = int(spec["forecast"]["lookback_records"])
    horizons = tuple(spec["forecast"]["horizons_records"])
    start, stop = segment["start"], segment["stop"]
    if int(anchors.min()) - lookback + 1 < start or int(anchors.max()) + max(horizons) >= stop:
        raise ValueError("A history/target footprint crosses its outer partition")
    expected_names = [spec["forecast"]["target_column"]] + list(spec["features"]["current_local_channels"])
    if segment["numeric_columns"] != expected_names:
        raise ValueError("Selected numeric coordinates are out of order")
    numeric = np.asarray(segment["numeric"], dtype=np.float64)
    if numeric.shape != (stop - start, len(expected_names)) or not np.isfinite(numeric).all():
        raise ValueError("Invalid permitted raw numeric segment")
    local = anchors.astype(np.int64) - start
    energy = numeric[:, 0]
    history = energy[local[:, None] - np.arange(lookback, dtype=np.int64)]
    sensor = numeric[local, 1:]
    calendar = _metadata_module().calendar_features([segment["timestamps"][int(i)] for i in local])
    x = np.ascontiguousarray(np.concatenate((history, sensor, calendar), axis=1), dtype=np.float64)
    if x.shape != (len(anchors), spec["features"]["dimension"]) or not np.isfinite(x).all():
        raise ValueError("Incorrect final Appliances feature dimensions/values")
    y = (np.ascontiguousarray(energy[local[:, None] + np.asarray(horizons, dtype=np.int64)], dtype=np.float64)
         if include_targets else None)
    return x, y


def _read_appliances(stage: str, spec: dict) -> dict:
    lock = _read_json(STEP23 / "data/source_archive/source_lock23.json")
    if lock.get("passed") is not True or lock.get("archive_sha256") != ARCHIVE_SHA256:
        raise RuntimeError("The locked source receipt changed")
    if ARCHIVE_PATH.stat().st_size != lock["archive_bytes"] or file_sha256(ARCHIVE_PATH) != ARCHIVE_SHA256:
        raise RuntimeError("The source archive no longer matches its fixed byte identity")
    permitted_roles = ("outer_train", "validation") if stage == "development" else ("test",)
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        member = spec["source"]["expected_zip_member"]
        if archive.namelist() != [member]:
            raise ValueError("The archive member roster differs from the locked source")
        with archive.open(member, "r") as binary_stream:
            with io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="") as text_stream:
                return _stream_appliances_records(text_stream, spec, permitted_roles)


def _load_appliances(stage: str, gate: dict) -> dict:
    spec = _read_json(STEP23 / "data/appliances_protocol23.json")
    metadata_helper = _metadata_module()
    # This is the fixed index-only stream from Step23, never an outcome-based draw.
    plan = metadata_helper.make_split_plan(spec)
    if np.__version__ != spec["partitions"]["reference_bank"]["numpy_version_for_exact_index_replay"]:
        raise RuntimeError("NumPy version differs from the frozen reference-index replay")
    started = time.perf_counter_ns()
    parsed = _read_appliances(stage, spec)
    parsing_ns = time.perf_counter_ns() - started
    assembly_start = time.perf_counter_ns()
    result = {"horizons": list(spec["forecast"]["horizons_records"]),
              "primary_column": spec["forecast"]["horizons_records"].index(spec["forecast"]["primary_horizon_records"])}
    segments = parsed["segments"]
    if stage == "development":
        support, support_y = _appliances_examples(segments["outer_train"], plan["reference_priority_order"], spec, include_targets=False)
        if support_y is not None:
            raise AssertionError("Reference labels were constructed")
        train_x, train_y = _appliances_examples(segments["outer_train"], plan["head_train"], spec, include_targets=True)
        validation_x, validation_y = _appliances_examples(segments["validation"], plan["validation"], spec, include_targets=True)
        result.update({"support": support, "train_X": train_x, "train_Y": train_y,
                       "validation_X": validation_x, "validation_Y": validation_y,
                       "train_anchor_ids": plan["head_train"].copy(),
                       "validation_anchor_ids": plan["validation"].copy()})
    else:
        test_x, test_y = _appliances_examples(segments["test"], plan["test"], spec, include_targets=True)
        result.update({"test_X": test_x, "test_Y": test_y, "test_anchor_ids": plan["test"].copy()})
    assembly_ns = time.perf_counter_ns() - assembly_start
    metadata = {
        "schema": "step24-role-specific-data-access-v1", "task_id": "appliances", "stage": stage,
        "gate": gate, **parsed["metadata"],
        "source_archive_sha256": ARCHIVE_SHA256,
        "feature_names": metadata_helper.feature_names(spec),
        "reference_priority_order": _array_record(plan["reference_priority_order"]),
        "reference_labels_provided": False,
        "raw_training_footprints_may_overlap": True,
        "outer_partition_footprints_disjoint": True,
        "forecast_target_is_future_reading_not_sum": True,
        "target_numeric_convention": "Retained binary64 source values, without target clipping or standardization in the loader",
        "baseline_history_positions": {"persistence": 0,
                                      "daily_seasonal": {str(h): 144 - h for h in result["horizons"]}},
        "benchmark_trajectories_generated": 0, "test_trajectories_generated": 0,
        "ECG_or_reserve_access": False,
        "source_parsing_nanoseconds": int(parsing_ns), "feature_assembly_nanoseconds": int(assembly_ns),
        "arrays": {key: _array_record(value) for key, value in result.items() if isinstance(value, np.ndarray)},
    }
    result["metadata"] = metadata
    return result


def load_development(task_id: str, *, gate_path: Path | str | None = None) -> dict:
    """Load only the sealed support/train/validation portions of a fixed task."""
    gate = _check_gate("development", task_id, gate_path)
    return _load_appliances("development", gate) if task_id == "appliances" else _load_hmm(task_id, "development", gate)


def load_test(task_id: str, *, gate_path: Path | str | None = None) -> dict:
    """Load a fixed held-out task only after every task's fit procedure is sealed."""
    gate = _check_gate("test", task_id, gate_path)
    return _load_appliances("test", gate) if task_id == "appliances" else _load_hmm(task_id, "test", gate)
