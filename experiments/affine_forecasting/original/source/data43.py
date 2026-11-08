"""Gate-enforced exact-label access for the three fixed Step 43 task-periods.

All natural-data numeric conversion occurs only in _decode_authorized_values.
Every fitting cohort is read once and retained; the three seeds share that
cohort. Assessment history is causal, and all 18 selected seed/policy forecasts
are byte-sealed before the current target is decoded.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re

import numpy as np

import prepare43
from common43 import (
    ROOT, TASKS, STAGES, SEEDS, POLICIES, digest, event, file_ref, protocol_identity,
    read_json, require_gate, sha_file, utc_now, verify_bindings, write_json,
)

MANIFEST = "inputs/data43/manifest43.json"
DATA_LOG = "output/data_access43.jsonl"
NUMERIC = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def _relative(path, *, root=ROOT):
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError as error:
            raise ValueError("Path lies outside this study") from error
    if not path.parts or ".." in path.parts:
        raise ValueError("Unsafe study-relative path")
    return path


def _task(task):
    if task not in TASKS:
        raise ValueError("Unknown predeclared task")
    return task


def _stage(stage):
    if stage not in STAGES:
        raise ValueError("Unknown predeclared stage")
    return stage


def _stamp(text):
    value = datetime.fromisoformat(text)
    if value.tzinfo is not None or value.isoformat(timespec="seconds") != text:
        raise ValueError("Noncanonical local-naive source timestamp")
    return value


def _iso(value):
    return value.isoformat(timespec="seconds")


@lru_cache(maxsize=64)
def _cached_json(path, expected_hash):
    # Caller checks the actual bytes before consulting this immutable cache.
    value = read_json(Path(path))
    if sha_file(Path(path)) != expected_hash:
        raise ValueError("Bound JSON changed while opening it")
    return value


def _bound_json(reference, *, root=ROOT):
    path = root / _relative(reference["path"], root=root)
    if sha_file(path) != reference["sha256"]:
        raise ValueError("Bound metadata changed: " + reference["path"])
    if "bytes" in reference and path.stat().st_size != reference["bytes"]:
        raise ValueError("Bound metadata length changed")
    return _cached_json(str(path.resolve()), reference["sha256"])


def _data_gate(*, root=ROOT):
    identity = protocol_identity(root=root)
    seal = read_json(root / "protocol_seal43.json")
    for name in ("source/common43.py", "source/prepare43.py", "source/data43.py", MANIFEST,
                 "inputs/exposure43/exposure43.json", "design43.json", "design_seal43.json"):
        if seal["bound_files"].get(name) != sha_file(root / name):
            raise ValueError("Frozen data dependency differs: " + name)
    require_gate("audit/preflight43.json", root=root, protocol_seal_sha256=identity)
    manifest_hash = sha_file(root / MANIFEST)
    manifest = _cached_json(str((root / MANIFEST).resolve()), manifest_hash)
    if (manifest.get("schema") != "step43-data-manifest-v1" or manifest["task_order"] != list(TASKS)
            or manifest.get("measurement_values_decoded_or_parsed") != 0
            or manifest["design_seal_sha256"] != sha_file(root / "design_seal43.json")):
        raise ValueError("Frozen calendar manifest identity or scope differs")
    return identity, manifest_hash, manifest


def _fit_gate(task, stage, identity, *, root=ROOT):
    name = f"output/{task}/{stage}/fit_seal43.json"
    seal = read_json(root / name)
    if (seal.get("schema") != "step43-stage-fit-seal-v1" or seal.get("task") != task
            or seal.get("stage") != stage or seal.get("protocol_seal_sha256") != identity
            or seal.get("seeds") != list(SEEDS)):
        raise ValueError("All three matching seed fit banks must be sealed before validation")
    verify_bindings(seal.get("bound_files"), root=root)
    return file_ref(name, root=root)


def _assessment_gate(identity, *, root=ROOT):
    name = "output/selection_seal43.json"
    seal = read_json(root / name)
    if (seal.get("schema") != "step43-selection-seal-v1"
            or seal.get("protocol_seal_sha256") != identity):
        raise ValueError("Matching global selection seal required")
    roster_name = "output/selected_roster43.json"
    if seal.get("bound_files", {}).get(roster_name) != sha_file(root / roster_name):
        raise ValueError("Global selected roster is not bound by the selection seal")
    verify_bindings(seal["bound_files"], root=root)
    selection_hash = sha_file(root / name)
    require_gate("audit/preassessment43.json", root=root, protocol_seal_sha256=identity,
                 selection_seal_sha256=selection_hash)
    roster = read_json(root / roster_name)
    if roster.get("protocol_seal_sha256") != identity or set(roster.get("tasks", {})) != set(TASKS):
        raise ValueError("Selected task roster differs")
    expected = {f"{seed}::{policy}" for seed in SEEDS for policy in POLICIES}
    for task in TASKS:
        selections = roster["tasks"][task]["selections"]
        if set(selections) != expected or not all(
                isinstance(value.get("candidate_id"), str) and value["candidate_id"]
                for value in selections.values()):
            raise ValueError("Selected roster must bind all 18 seed/policy identities")
    return selection_hash, roster


def stage_queries(task, stage, partition, *, root=ROOT):
    """Return fixed metadata only, without crossing the numerical access gate."""
    _task(task)
    _stage(stage)
    if partition not in ("train", "validation") or (stage == "final" and partition == "validation"):
        raise ValueError("Invalid fixed cohort partition")
    manifest = read_json(root / MANIFEST)
    return [dict(q) for q in manifest["tasks"][task]["stages"][stage][partition]]


def assessment_queries(task, *, root=ROOT):
    """Return only the already frozen assessment query calendar."""
    _task(task)
    manifest = read_json(root / MANIFEST)
    return [dict(q) for q in manifest["tasks"][task]["assessment"]]


def _needed_times(queries):
    required = set()
    for query in queries:
        target = _stamp(query["target_timestamp"])
        origin, history = prepare43.required_times(target)
        if _iso(origin) != query["origin_timestamp"]:
            raise ValueError("Frozen query origin differs from its horizon")
        required.add(target)
        required.update(history)
    return sorted(required)


@lru_cache(maxsize=8)
def _payload(path, archive_hash, member, nested_gzip, member_hash, payload_hash):
    raw_member, payload = prepare43.payload_bytes(Path(path), member, nested_gzip)
    if (sha_file(Path(path)) != archive_hash or hashlib.sha256(raw_member).hexdigest() != member_hash
            or hashlib.sha256(payload).hexdigest() != payload_hash):
        raise ValueError("Original source bytes changed")
    return payload


def _source_context(task, manifest, *, root=ROOT):
    value = manifest["tasks"][task]
    source = _bound_json(value["source"], root=root)
    calendar = _bound_json(value["calendar"], root=root)
    if (source.get("task") != task or calendar.get("task") != task
            or source["source_spec"] != value["spec"] or calendar["source_spec"] != value["spec"]
            or source["archives"] != value["archives"]
            or calendar.get("target_values_decoded_or_parsed") != 0
            or calendar.get("other_measurement_values_decoded_or_parsed") != 0):
        raise ValueError("Bound source/calendar schema differs")
    payloads = {}
    for item in value["archives"]:
        path = root / _relative(item["path"], root=root)
        if sha_file(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("Original archive changed")
        payloads[item["source_id"]] = _payload(str(path.resolve()), item["sha256"], item["member"],
                                              item["nested_gzip"], item["member_sha256"], item["payload_sha256"])
    index = {row["timestamp"]: row for row in calendar["rows"]}
    if len(index) != len(calendar["rows"]):
        raise ValueError("Repeated grouped calendar timestamp")
    return value["spec"], calendar, index, payloads


def _parse_numeric(spec, token, stamp):
    text = token.decode("ascii").strip(" \t")
    if not NUMERIC.fullmatch(text):
        raise ValueError("Invalid declared numeric target at " + stamp)
    value = F(text)
    domain = spec["numeric_domain"]
    if domain == "finite_nonnegative_decimal":
        valid = value >= 0
    elif domain == "finite_nonnegative_integer":
        valid = value >= 0 and value.denominator == 1
    elif domain == "finite_decimal_at_least_minus273_15":
        valid = value >= F("-273.15")
    else:
        raise ValueError("Unknown predeclared numeric domain")
    if not valid:
        raise ValueError("Target violates its predeclared physical domain at " + stamp)
    return value


def _state_path(task, *, root=ROOT):
    return root / f"output/data43/{task}/assessment/state43.json"


def _state(task, identity, manifest_hash, selection_hash, *, root=ROOT):
    value = read_json(_state_path(task, root=root))
    if (value.get("schema") != "step43-assessment-access-state-v1" or value.get("task") != task
            or value.get("protocol_seal_sha256") != identity
            or value.get("data_manifest_sha256") != manifest_hash
            or value.get("selection_seal_sha256") != selection_hash):
        raise ValueError("Assessment access state identity differs")
    return value


def _decode_authorized_values(task, stage, phase, stamps, *, root=ROOT):
    """The only natural-data numeric conversion boundary."""
    _task(task)
    _stage(stage)
    identity, manifest_hash, manifest = _data_gate(root=root)
    stamps = list(stamps)
    fit_reference = None
    selection_hash = None
    if phase in ("train", "validation"):
        if phase == "validation":
            if stage == "final":
                raise ValueError("No final-stage validation cohort exists")
            fit_reference = _fit_gate(task, stage, identity, root=root)
        section = manifest["tasks"][task]["stages"][stage]
        queries = section[phase]
        expected = _needed_times(queries)
        if stamps != expected:
            raise ValueError("Numeric source request differs from the exact frozen cohort")
        stop_key = "train_stop_exclusive" if phase == "train" else "validation_stop_exclusive"
        maximum = datetime.fromisoformat(section[stop_key]) - timedelta(seconds=1)
    elif phase in ("assessment_history", "assessment_target"):
        if stage != "final":
            raise ValueError("Assessment source values require final models")
        selection_hash, _ = _assessment_gate(identity, root=root)
        state = _state(task, identity, manifest_hash, selection_hash, root=root)
        index = state["next_index"]
        queries = manifest["tasks"][task]["assessment"]
        if type(index) is not int or not 0 <= index < len(queries):
            raise ValueError("No current chronological assessment query")
        query = queries[index]
        if state.get("pending_query") != query:
            raise ValueError("Pending query differs from the frozen roster")
        target = _stamp(query["target_timestamp"])
        origin, history = prepare43.required_times(target)
        if phase == "assessment_history":
            if state["status"] != "HISTORY_PENDING" or state.get("history_complete"):
                raise ValueError("Assessment history has already been consumed")
            expected, maximum = history, origin
        else:
            if state["status"] != "PREDICTIONS_SEALED" or state.get("target_decode_started"):
                raise ValueError("Complete query predictions must be sealed before exactly one target decode")
            prediction_seal = _bound_json(state["prediction_seal"], root=root)
            _bound_json(prediction_seal["predictions"], root=root)
            if (prediction_seal.get("query_id") != query["id"]
                    or prediction_seal.get("query_index") != index
                    or prediction_seal.get("selection_seal_sha256") != selection_hash):
                raise ValueError("Current query prediction seal differs")
            expected, maximum = [target], target
        if stamps != expected:
            raise ValueError("Assessment source request exceeds the current authorized query")
        if phase == "assessment_target":
            state["target_decode_started"] = True
            write_json(_state_path(task, root=root), state, overwrite=True)
    else:
        raise ValueError("Unknown numeric access phase")
    if not stamps or any(value > maximum for value in stamps):
        raise ValueError("Requested source timestamp exceeds the authorized boundary")
    spec, calendar, calendar_index, payloads = _source_context(task, manifest, root=root)
    values, row_references, duplicates = {}, [], []
    for stamp in stamps:
        text = _iso(stamp)
        row = calendar_index.get(text)
        if row is None or row["declared_target_missing"]:
            raise ValueError("Authorized timestamp is absent or declared missing")
        decoded = []
        for ref in row["records"]:
            raw = payloads[ref["source_id"]]
            record = raw[ref["byte_start"]:ref["byte_stop"]]
            fields = prepare43.byte_fields(record, spec["separator"])
            if len(fields) != len(calendar["headers"]) or prepare43.parse_timestamp(spec, fields) != stamp:
                raise ValueError("Authorized source offset/schema/timestamp differs")
            decoded.append(_parse_numeric(spec, fields[spec["target_index"]], text))
            row_references.append({"timestamp": text, **ref})
        if not decoded or any(value != decoded[0] for value in decoded):
            raise ValueError("Duplicate target disagreement at " + text + "; retain failure")
        values[text] = decoded[0]
        if len(decoded) > 1:
            duplicates.append({"timestamp": text, "source_records": len(decoded), "exact_agreement": True})
    receipt = {
        "schema": "step43-authorized-numeric-access-v1", "status": "PASS", "utc": utc_now(),
        "task": task, "stage": stage, "phase": phase, "protocol_seal_sha256": identity,
        "data_manifest_sha256": manifest_hash, "fit_seal": fit_reference,
        "selection_seal_sha256": selection_hash, "requested_timestamps": [_iso(x) for x in stamps],
        "maximum_allowed_timestamp": _iso(maximum), "maximum_decoded_timestamp": _iso(max(stamps)),
        "unique_measurement_timestamps": len(stamps), "measurement_token_decodes": len(row_references),
        "source_row_references": row_references, "duplicate_agreement": duplicates,
        "source": manifest["tasks"][task]["source"], "calendar": manifest["tasks"][task]["calendar"],
        "source_archives": [{"path": x["path"], "sha256": x["sha256"], "payload_sha256": x["payload_sha256"]}
                            for x in manifest["tasks"][task]["archives"]],
        "other_measurement_values_decoded_or_parsed": 0, "outcome_based_row_filtering": False,
    }
    return values, receipt


def _normalization(task, stage, *, root=ROOT):
    path = root / f"output/data43/{task}/{stage}/normalization43.json"
    value = read_json(path)
    if (value.get("schema") != "step43-training-target-normalization-v1"
            or value.get("task") != task or value.get("stage") != stage
            or value.get("fitted_partition") != "train" or value.get("future_data_used") is not False
            or F(value["span"]) <= 0 or F(value["target_max"]) - F(value["target_min"]) != F(value["span"])):
        raise ValueError("Invalid training-only target normalization")
    return {**value, "target_min": F(value["target_min"]), "target_max": F(value["target_max"]), "span": F(value["span"])}


def _normalized(value, normalization):
    return (value - normalization["target_min"]) / normalization["span"]


def _write_X(path, X):
    if path.exists():
        raise FileExistsError("Retain the existing stored raw feature matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X, allow_pickle=False)


def _load_cached_cohort(task, stage, partition, *, root=ROOT):
    base = root / f"output/data43/{task}/{stage}"
    completion = read_json(base / f"{partition}_completed43.json")
    if completion.get("status") != "PASS":
        raise ValueError("A complete retained numeric cohort is required")
    verify_bindings(completion["bound_files"], root=root)
    name = f"output/data43/{task}/{stage}/{partition}.json"
    record = read_json(root / name)
    X = np.load(root / record["X"]["path"], allow_pickle=False)
    if X.dtype != np.float64 or X.shape != (record["n"], 24) or not np.isfinite(X).all():
        raise ValueError("Stored raw feature matrix violates its fixed schema")
    X.flags.writeable = False
    normalization = _normalization(task, stage, root=root)
    return {**record, "X": X, "y": [F(x) for x in record["y"]],
            "raw_targets": [F(x) for x in record["raw_targets"]],
            "persistence": [F(x) for x in record["persistence"]],
            "raw_persistence": [F(x) for x in record["raw_persistence"]],
            "normalization": normalization, "path": name, "sha256": sha_file(root / name)}


def _load_cohort(task, stage, partition, *, root=ROOT):
    _task(task)
    _stage(stage)
    identity, manifest_hash, manifest = _data_gate(root=root)
    if partition == "validation":
        if stage == "final":
            raise ValueError("No final validation labels are permitted")
        fit_seal = _fit_gate(task, stage, identity, root=root)
    elif partition == "train":
        fit_seal = None
    else:
        raise ValueError("Bulk assessment loading is forbidden")
    base = root / f"output/data43/{task}/{stage}"
    if (base / f"{partition}_completed43.json").exists():
        return _load_cached_cohort(task, stage, partition, root=root)
    marker = base / f"{partition}_started43.json"
    if marker.exists():
        raise ValueError("This numeric cohort access already started; retain incomplete evidence")
    queries = manifest["tasks"][task]["stages"][stage][partition]
    needed = _needed_times(queries)
    write_json(marker, {"schema": "step43-cohort-access-start-v1", "utc": utc_now(), "task": task,
                        "stage": stage, "partition": partition, "protocol_seal_sha256": identity,
                        "data_manifest_sha256": manifest_hash, "fit_seal": fit_seal})
    event("cohort_access_started", root=root, log=DATA_LOG, task=task, stage=stage, partition=partition,
          protocol_seal_sha256=identity, data_manifest_sha256=manifest_hash, queries=len(queries),
          unique_timestamps_requested=len(needed), fit_seal=fit_seal)
    try:
        values, access = _decode_authorized_values(task, stage, partition, needed, root=root)
        targets = [values[q["target_timestamp"]] for q in queries]
        access_path = base / f"{partition}_access43.json"
        write_json(access_path, access)
        normal_path = base / "normalization43.json"
        if partition == "train":
            lo, hi = min(targets), max(targets)
            if hi == lo:
                raise ValueError("Degenerate training target range; retain failure without changing the panel")
            write_json(normal_path, {
                "schema": "step43-training-target-normalization-v1", "task": task, "stage": stage,
                "fitted_partition": "train", "target_min": lo, "target_max": hi, "span": hi-lo,
                "train_target_ids_sha256": digest([q["id"] for q in queries]),
                "raw_train_targets_sha256": digest(targets), "training_access": file_ref(access_path, root=root),
                "target_values_clipped": False, "future_data_used": False,
            })
        normal = _normalization(task, stage, root=root)
        histories, raw_persistence = [], []
        for query in queries:
            origin, lags = prepare43.required_times(_stamp(query["target_timestamp"]))
            histories.append([values[_iso(stamp)] for stamp in lags])
            raw_persistence.append(values[_iso(origin)])
        X = np.asarray([[float(x) for x in row] for row in histories], dtype=np.float64)
        if X.shape != (len(queries), 24) or not np.isfinite(X).all():
            raise ValueError("Invalid binary64 raw feature conversion")
        X_path = base / f"{partition}_X.npy"
        _write_X(X_path, X)
        record = {
            "schema": "step43-causal-cohort-v1", "task": task, "stage": stage, "partition": partition,
            "protocol_seal_sha256": identity, "data_manifest_sha256": manifest_hash,
            "n": len(queries), "ids": [q["id"] for q in queries], "queries": queries,
            "X": file_ref(X_path, root=root), "raw_targets": targets,
            "y": [_normalized(x, normal) for x in targets], "raw_persistence": raw_persistence,
            "persistence": [_normalized(x, normal) for x in raw_persistence],
            "normalization": file_ref(normal_path, root=root), "access": file_ref(access_path, root=root),
            "other_measurement_predictors_used": False, "target_values_clipped": False,
        }
        record_path = base / f"{partition}.json"
        write_json(record_path, record)
        bound = [marker, access_path, normal_path, X_path, record_path]
        completion = {"schema": "step43-cohort-access-completion-v1", "status": "PASS", "utc": utc_now(),
                      "task": task, "stage": stage, "partition": partition,
                      "protocol_seal_sha256": identity, "data_manifest_sha256": manifest_hash,
                      "bound_files": {p.relative_to(root).as_posix(): sha_file(p) for p in bound}}
        write_json(base / f"{partition}_completed43.json", completion)
        event("cohort_access_completed", root=root, log=DATA_LOG, task=task, stage=stage, partition=partition,
              cohort=file_ref(record_path, root=root), access=file_ref(access_path, root=root),
              completion=file_ref(base / f"{partition}_completed43.json", root=root), n=len(queries))
        return _load_cached_cohort(task, stage, partition, root=root)
    except Exception as error:
        event("cohort_access_failed", root=root, log=DATA_LOG, task=task, stage=stage, partition=partition,
              error_type=type(error).__name__, message=str(error))
        raise


def load_training(task, stage, *, root=ROOT):
    return _load_cohort(task, stage, "train", root=root)


def load_validation(task, stage, *, root=ROOT):
    return _load_cohort(task, stage, "validation", root=root)


def _query_base(task, index, *, root=ROOT):
    return root / f"output/data43/{task}/assessment/queries/{index:04d}"


def _read_history(task, index, *, root=ROOT):
    path = _query_base(task, index, root=root) / "history43.json"
    record = read_json(path)
    if record.get("task") != task or record.get("query_index") != index:
        raise ValueError("Retained history identity differs")
    _bound_json(record["access"], root=root)
    _bound_json(record["normalization"], root=root)
    Xref = record["X"]
    if sha_file(root / Xref["path"]) != Xref["sha256"]:
        raise ValueError("Retained history feature bytes changed")
    X = np.load(root / Xref["path"], allow_pickle=False)
    if X.dtype != np.float64 or X.shape != (1, 24) or not np.isfinite(X).all():
        raise ValueError("Retained history has an invalid raw feature matrix")
    X.flags.writeable = False
    return {**record, "X": X, "raw_X": X, "raw_values": [F(x) for x in record["raw_values"]],
            "raw_persistence": F(record["raw_persistence"]), "persistence": F(record["persistence"]),
            "normalization": _normalization(task, "final", root=root),
            "history_receipt_path": path.relative_to(root).as_posix(), "history_receipt_sha256": sha_file(path)}


def load_assessment_history(task, index, *, root=ROOT):
    _task(task)
    identity, manifest_hash, manifest = _data_gate(root=root)
    selection_hash, _ = _assessment_gate(identity, root=root)
    queries = manifest["tasks"][task]["assessment"]
    if type(index) is not int or not 0 <= index < len(queries):
        raise ValueError("Invalid frozen assessment query index")
    state_path = _state_path(task, root=root)
    if not state_path.exists():
        if index != 0:
            raise ValueError("Assessment must start with the first chronological query")
        write_json(state_path, {
            "schema": "step43-assessment-access-state-v1", "task": task, "status": "READY",
            "protocol_seal_sha256": identity, "data_manifest_sha256": manifest_hash,
            "selection_seal_sha256": selection_hash, "next_index": 0, "pending_query": None,
            "previously_revealed_targets": [],
        })
    state = _state(task, identity, manifest_hash, selection_hash, root=root)
    if state["status"] == "FAILED":
        raise ValueError("Assessment stopped at a retained failure")
    if index < state["next_index"]:
        # Reporting/cost code may reuse already authorized feature bytes.
        return _read_history(task, index, root=root)
    if index != state["next_index"]:
        raise ValueError("Assessment query order cannot skip ahead")
    query = queries[index]
    if state["status"] in ("HISTORY_READY", "PREDICTIONS_SEALED") and state.get("pending_query") == query:
        return _read_history(task, index, root=root)
    if state["status"] != "READY" or state.get("pending_query") is not None:
        raise ValueError("Finish the current query before reading another history")
    state.update({"status": "HISTORY_PENDING", "pending_query": dict(query)})
    write_json(state_path, state, overwrite=True)
    origin, stamps = prepare43.required_times(_stamp(query["target_timestamp"]))
    event("assessment_history_started", root=root, log=DATA_LOG, task=task, query_index=index, query_id=query["id"],
          origin_timestamp=_iso(origin), target_timestamp=query["target_timestamp"], protocol_seal_sha256=identity,
          selection_seal_sha256=selection_hash)
    try:
        values, access = _decode_authorized_values(task, "final", "assessment_history", stamps, root=root)
        base = _query_base(task, index, root=root)
        access_path = base / "history_access43.json"
        write_json(access_path, access)
        raw_values = [values[_iso(stamp)] for stamp in stamps]
        X = np.asarray([[float(x) for x in raw_values]], dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite causal raw history conversion")
        X_path = base / "history_X.npy"
        _write_X(X_path, X)
        normal = _normalization(task, "final", root=root)
        record = {
            "schema": "step43-causal-assessment-history-v1", "task": task, "query_id": query["id"],
            "query_index": index, "target_timestamp": query["target_timestamp"], "origin_timestamp": query["origin_timestamp"],
            "protocol_seal_sha256": identity, "data_manifest_sha256": manifest_hash,
            "selection_seal_sha256": selection_hash, "history_timestamps": [_iso(x) for x in stamps],
            "raw_values": raw_values, "raw_persistence": raw_values[-1],
            "persistence": _normalized(raw_values[-1], normal),
            "X": file_ref(X_path, root=root), "access": file_ref(access_path, root=root),
            "normalization": file_ref(f"output/data43/{task}/final/normalization43.json", root=root),
            "previously_revealed_targets_in_history": sorted(set(state["previously_revealed_targets"]) & {_iso(x) for x in stamps}),
            "current_target_decoded": False, "other_measurement_predictors_used": False,
        }
        path = base / "history43.json"
        write_json(path, record)
        state.update({"status": "HISTORY_READY", "history_complete": True, "history": file_ref(path, root=root)})
        write_json(state_path, state, overwrite=True)
        event("assessment_history_completed", root=root, log=DATA_LOG, task=task, query_index=index,
              query_id=query["id"], history=file_ref(path, root=root), access=file_ref(access_path, root=root),
              current_target_decoded=False, maximum_decoded_timestamp=access["maximum_decoded_timestamp"])
        return _read_history(task, index, root=root)
    except Exception as error:
        state["status"] = "FAILED"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(state_path, state, overwrite=True)
        event("assessment_history_failed", root=root, log=DATA_LOG, task=task, query_index=index, error=state["error"])
        raise


def _validate_prediction(task, index, query, prediction_path, history, identity, roster, *, root=ROOT):
    relative = _relative(prediction_path, root=root).as_posix()
    expected_path = f"output/assessment/{task}/queries/{index:04d}/predictions43.json"
    if relative != expected_path:
        raise ValueError("Unexpected current-query prediction record path")
    reference = file_ref(relative, root=root)
    value = _bound_json(reference, root=root)
    if (value.get("schema") != "step43-query-predictions-v1" or value.get("task") != task
            or value.get("query_id") != query["id"] or value.get("query_index") != index
            or value.get("protocol_seal_sha256") != identity
            or value.get("history_receipt_sha256") != history["sha256"]):
        raise ValueError("Current prediction identity or history binding differs")
    selections = roster["tasks"][task]["selections"]
    predictions = value.get("predictions")
    if not isinstance(predictions, dict) or set(predictions) != set(selections):
        raise ValueError("All 18 seed/policy predictions must exist before target access")
    for key, chosen in selections.items():
        prediction = predictions[key]
        if prediction.get("candidate_id") != chosen["candidate_id"]:
            raise ValueError("Prediction substitutes a different selected candidate")
        text = prediction.get("reported_physical_hex")
        if not isinstance(text, str):
            raise ValueError("Missing physical binary64 prediction")
        reported = float.fromhex(text)
        if not math.isfinite(reported) or reported.hex() != text:
            raise ValueError("Nonfinite or noncanonical physical binary64 prediction")
    return reference


def reveal_assessment_target(task, index, prediction_path, *, root=ROOT):
    _task(task)
    identity, manifest_hash, manifest = _data_gate(root=root)
    selection_hash, roster = _assessment_gate(identity, root=root)
    queries = manifest["tasks"][task]["assessment"]
    if type(index) is not int or not 0 <= index < len(queries):
        raise ValueError("Invalid frozen assessment query index")
    query = queries[index]
    state = _state(task, identity, manifest_hash, selection_hash, root=root)
    if (state.get("status") != "HISTORY_READY" or state["next_index"] != index
            or state.get("pending_query") != query or not state.get("history_complete")):
        raise ValueError("Current history must precede a once-only target reveal")
    _bound_json(state["history"], root=root)
    prediction = _validate_prediction(task, index, query, prediction_path, state["history"], identity, roster, root=root)
    base = _query_base(task, index, root=root)
    seal_path = base / "predictions_seal43.json"
    seal = {"schema": "step43-query-prediction-byte-seal-v1", "utc": utc_now(), "task": task,
            "query_id": query["id"], "query_index": index, "protocol_seal_sha256": identity,
            "selection_seal_sha256": selection_hash, "predictions": prediction, "history": state["history"],
            "prediction_keys": sorted(roster["tasks"][task]["selections"]), "prediction_count": len(SEEDS)*len(POLICIES),
            "current_target_decoded": False}
    write_json(seal_path, seal)
    state.update({"status": "PREDICTIONS_SEALED", "prediction_seal": file_ref(seal_path, root=root)})
    write_json(_state_path(task, root=root), state, overwrite=True)
    event("assessment_predictions_sealed", root=root, log=DATA_LOG, task=task, query_index=index,
          query_id=query["id"], prediction_seal=file_ref(seal_path, root=root), predictions=prediction,
          prediction_count=len(SEEDS)*len(POLICIES), current_target_decoded=False)
    try:
        target = _stamp(query["target_timestamp"])
        event("assessment_target_started", root=root, log=DATA_LOG, task=task, query_index=index,
              query_id=query["id"], target_timestamp=query["target_timestamp"],
              prediction_seal=file_ref(seal_path, root=root))
        values, access = _decode_authorized_values(task, "final", "assessment_target", [target], root=root)
        access_path = base / "target_access43.json"
        write_json(access_path, access)
        raw = values[_iso(target)]
        normal = _normalization(task, "final", root=root)
        record = {
            "schema": "step43-revealed-assessment-target-v1", "task": task, "query_id": query["id"],
            "query_index": index, "target_timestamp": query["target_timestamp"], "origin_timestamp": query["origin_timestamp"],
            "protocol_seal_sha256": identity, "data_manifest_sha256": manifest_hash,
            "selection_seal_sha256": selection_hash, "raw_target": raw, "y": _normalized(raw, normal),
            "normalization": file_ref(f"output/data43/{task}/final/normalization43.json", root=root),
            "history": state["history"], "prediction_seal": file_ref(seal_path, root=root),
            "access": file_ref(access_path, root=root), "target_value_clipped": False,
        }
        path = base / "target43.json"
        write_json(path, record)
        state = _state(task, identity, manifest_hash, selection_hash, root=root)
        state["previously_revealed_targets"].append(query["target_timestamp"])
        state["next_index"] += 1
        state["pending_query"] = None
        state["status"] = "COMPLETE" if state["next_index"] == len(queries) else "READY"
        for key in ("history", "history_complete", "prediction_seal", "target_decode_started"):
            state.pop(key, None)
        write_json(_state_path(task, root=root), state, overwrite=True)
        event("assessment_target_revealed", root=root, log=DATA_LOG, task=task, query_index=index,
              query_id=query["id"], target=file_ref(path, root=root), access=file_ref(access_path, root=root),
              next_index=state["next_index"], state_status=state["status"])
        return {**record, "raw_target": raw, "y": _normalized(raw, normal),
                "path": path.relative_to(root).as_posix(), "sha256": sha_file(path)}
    except Exception as error:
        state = _state(task, identity, manifest_hash, selection_hash, root=root)
        state["status"] = "FAILED"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(_state_path(task, root=root), state, overwrite=True)
        event("assessment_target_failed", root=root, log=DATA_LOG, task=task, query_index=index, error=state["error"])
        raise
