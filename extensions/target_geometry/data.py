"""Pinned new archive collections for the target-geometry study.

The four collections and all selection rules are declared before model fitting.
Series selection examines identifiers, lengths, and training-prefix values only.
The previously used twelve collections are excluded at source-family level.
These are new collections, not a random sample of forecasting applications.

This module deliberately reuses the historical window and scaling semantics
without changing their implementation. Targets lie wholly within one split;
later forecast origins may use observations already available in that split.
There is no interpolation and no pretest refit in the new primary experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from extensions.natural_forecasting import data as inherited


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
PREPARED = ROOT / "prepared"
MANIFEST_PATH = SOURCES / "manifest.json"
ROSTER_PATH = PREPARED / "roster.json"

SOURCE_SPECS = (
    {"source_id": "australian_electricity_halfhourly", "family_id": "australian_state_electricity_demand",
     "record_id": 4659727, "frequency": "half_hourly", "context": 48, "horizon": 12,
     "seasonal_period": 48},
    {"source_id": "bitcoin_daily", "family_id": "bitinfocharts_bitcoin_indicators",
     "record_id": 5121965, "frequency": "daily", "context": 28, "horizon": 7,
     "seasonal_period": 7},
    {"source_id": "pedestrian_hourly", "family_id": "melbourne_pedestrian_counts",
     "record_id": 4656626, "frequency": "hourly", "context": 48, "horizon": 6,
     "seasonal_period": 24},
    {"source_id": "kdd_air_quality_hourly", "family_id": "kdd2018_beijing_london_air_quality",
     "record_id": 4656719, "frequency": "hourly", "context": 48, "horizon": 6,
     "seasonal_period": 24},
)
SETTINGS = {
    "selection_salt": "kahm-target-geometry-new-collections-v1",
    "max_series_per_source": 3,
    "minimum_train_windows": 96,
    "max_train_windows": 512,
    "max_eval_windows": 64,
    "split_fractions": [0.6, 0.2, 0.2],
    "pretest_refit": False,
}
HISTORICAL_MANIFESTS = (
    ROOT.parent / "natural_forecasting" / "data_sources.json",
    ROOT.parent / "forecasting_utility" / "sources.json",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(value.shape).encode())
    hasher.update(str(value.dtype).encode())
    hasher.update(value.tobytes())
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "KAHM-target-geometry-research/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def historical_families():
    records = []
    for path in HISTORICAL_MANIFESTS:
        manifest = json.loads(path.read_text())
        records.append({"path": path.relative_to(ROOT.parent.parent).as_posix(),
                        "sha256": digest(path),
                        "sources": [{"source_id": s["source_id"], "family_id": s["family_id"]}
                                    for s in manifest["sources"]]})
    old = {s["family_id"] for record in records for s in record["sources"]}
    if old.intersection(spec["family_id"] for spec in SOURCE_SPECS):
        raise ValueError("A declared new collection repeats an old source family")
    return records


def pin_sources():
    """Pin official record metadata, source bytes, licenses and content hashes."""
    SOURCES.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        for source in manifest["sources"]:
            archive = source["archive"]
            destination = SOURCES / archive["filename"]
            if not destination.exists():
                destination.write_bytes(_fetch(archive["url"]))
            if destination.stat().st_size != archive["bytes"] or digest(destination) != archive["sha256"]:
                raise ValueError(f"Changed archive bytes: {source['source_id']}")
        return manifest
    sources = []
    for spec in SOURCE_SPECS:
        rid = spec["record_id"]
        print(f"Pin official record and archive: {spec['source_id']}", flush=True)
        metadata_bytes = _fetch(f"https://zenodo.org/api/records/{rid}")
        metadata_path = SOURCES / f"{rid}.json"
        metadata_path.write_bytes(metadata_bytes)
        metadata = json.loads(metadata_bytes)
        archives = [entry for entry in metadata["files"] if entry["key"].endswith(".zip")]
        if len(archives) != 1:
            raise ValueError("Expected one TSF archive per pinned source record")
        entry = archives[0]
        archive_path = SOURCES / entry["key"]
        archive_bytes = _fetch(entry["links"]["self"])
        algorithm, checksum = entry["checksum"].split(":", 1)
        if len(archive_bytes) != entry["size"] or hashlib.new(algorithm, archive_bytes).hexdigest() != checksum:
            raise ValueError("Official archive checksum or size mismatch")
        archive_path.write_bytes(archive_bytes)
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".tsf")]
            if len(members) != 1:
                raise ValueError("Expected one TSF member per archive")
            member_hash = hashlib.sha256(archive.read(members[0])).hexdigest()
        info = metadata["metadata"]
        sources.append({
            **spec,
            "role": "new_collection_evaluation",
            "title": info["title"], "version": info.get("version"),
            "record_url": f"https://zenodo.org/records/{rid}",
            "doi": info.get("doi", f"10.5281/zenodo.{rid}"),
            "license": info.get("license"), "creators": info.get("creators"),
            "description": info.get("description"),
            "related_identifiers": info.get("related_identifiers", []),
            "metadata_file": metadata_path.name, "metadata_sha256": digest(metadata_path),
            "archive": {"filename": archive_path.name, "url": entry["links"]["self"],
                        "bytes": len(archive_bytes), "sha256": digest(archive_path),
                        "upstream_checksum": entry["checksum"], "member": members[0],
                        "member_sha256": member_hash, "encoding": "cp1252"},
        })
    result = {"schema_version": 1, "settings": SETTINGS, "sources": sources,
              "historical_exclusions": historical_families(),
              "catalogue": "https://forecastingdata.org/",
              "selection_basis": "Four distinct new collections selected by domain, provenance and feasible size before any forecasting fits. Domain overlap with earlier collections is possible; no random-domain sampling is claimed."}
    write_json(MANIFEST_PATH, result)
    return result


def iter_series(spec):
    """Read the general TSF schema, retaining the source's exact series IDs."""
    path = SOURCES / spec["archive"]["filename"]
    if digest(path) != spec["archive"]["sha256"]:
        raise ValueError("Archive changed after pinning")
    attributes, started, frequency, seen = [], False, None, set()
    with zipfile.ZipFile(path) as archive, archive.open(spec["archive"]["member"]) as binary:
        with io.TextIOWrapper(binary, encoding=spec["archive"]["encoding"]) as stream:
            for raw in stream:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("@"):
                    pieces = line.split()
                    if pieces[0].lower() == "@attribute":
                        attributes.append((pieces[1], pieces[2]))
                    elif pieces[0].lower() == "@frequency":
                        frequency = pieces[1]
                    elif pieces[0].lower() == "@data":
                        started = True
                    continue
                if not started:
                    raise ValueError("TSF observation encountered before data header")
                fields = line.split(":", len(attributes))
                if len(fields) != len(attributes) + 1:
                    raise ValueError("TSF attribute count differs from its header")
                attr = dict(zip((name for name, _ in attributes), fields[:-1]))
                sid = attr.get("series_name", fields[0])
                if sid in seen:
                    raise ValueError(f"Duplicate source series ID: {sid}")
                seen.add(sid)
                values = np.fromstring(fields[-1].replace("?", "nan"), sep=",", dtype=np.float64)
                if len(values) != fields[-1].count(",") + 1 or np.isinf(values).any():
                    raise ValueError("Malformed or infinite TSF values")
                yield inherited.SeriesRecord(spec["source_id"], sid, values,
                                             attr.get("start_timestamp", "not supplied"),
                                             frequency or spec["frequency"])


def select_records(records, spec, settings=SETTINGS):
    """Hash-ranked eligible series; values after the training split are unused."""
    retained, excluded, total, eligible = [], {}, 0, 0
    for record in records:
        total += 1
        train_end, _, _ = inherited.split_cutoffs(len(record.values))
        prefix = record.values[:train_end]
        try:
            inherited.training_mse_scale(prefix, len(prefix), spec["seasonal_period"])
            origins, _ = inherited.valid_origins(prefix, spec["context"], spec["horizon"], 0, len(prefix))
            if len(origins) < settings["minimum_train_windows"]:
                raise ValueError("fewer_than_96_finite_training_windows")
        except ValueError as error:
            excluded[record.series_id] = str(error)
            continue
        eligible += 1
        key = hashlib.sha256(f"{settings['selection_salt']}\0{spec['source_id']}\0{record.series_id}".encode()).hexdigest()
        retained.append((key, record))
        retained.sort(key=lambda item: (item[0], item[1].series_id))
        del retained[settings["max_series_per_source"]:]
    return retained, {"source_series_count": total, "eligible_count": eligible,
                      "excluded_training_only": excluded,
                      "selection_salt": settings["selection_salt"]}


def natural_arrays(record, spec, settings=SETTINGS):
    ends = inherited.split_cutoffs(len(record.values))
    scale, scale_method, pairs = inherited.training_mse_scale(record.values, ends[0], spec["seasonal_period"])
    parts = {
        "train": inherited.make_windows(record.values, spec["context"], spec["horizon"], 0, ends[0], settings["max_train_windows"]),
        "validation": inherited.make_windows(record.values, spec["context"], spec["horizon"], ends[0], ends[1], settings["max_eval_windows"]),
        "test": inherited.make_windows(record.values, spec["context"], spec["horizon"], ends[1], ends[2], settings["max_eval_windows"]),
    }
    arrays = {f"{key}_{name}": getattr(part, key) for name, part in parts.items()
              for key in ("X", "Y", "origins")}
    arrays["values"] = record.values
    unsupported = [name for name, part in parts.items() if not len(part.X)]
    meta = {"series_id": record.series_id, "frequency": record.frequency,
            "n_values": len(record.values), "start_timestamp": record.start_timestamp,
            "seasonal_period": spec["seasonal_period"],
            "train_end": ends[0], "validation_end": ends[1], "test_end": ends[2],
            "loss_scale": scale, "scale_method": scale_method, "scale_pair_count": pairs,
            "split_details": {name: part.to_dict() for name, part in parts.items()},
            "supported": not unsupported, "unsupported_splits": unsupported,
            "arrays_sha256": {key: array_hash(value) for key, value in arrays.items()}}
    return arrays, meta


def prepare():
    if ROSTER_PATH.exists():
        raise FileExistsError("Roster already exists; verify it rather than selecting new series")
    manifest = pin_sources()
    PREPARED.mkdir(parents=True, exist_ok=True)
    tasks, sources = [], []
    for spec in manifest["sources"]:
        print(f"Training-prefix eligibility and hash selection: {spec['source_id']}", flush=True)
        selected, audit = select_records(iter_series(spec), spec)
        if len(selected) != SETTINGS["max_series_per_source"]:
            raise ValueError("Fewer than three eligible series; stop before model evaluation")
        sources.append({"source_id": spec["source_id"], "family_id": spec["family_id"],
                        **audit, "selected_series": [record.series_id for _, record in selected]})
        for rank, (key, record) in enumerate(selected):
            arrays, metadata = natural_arrays(record, spec)
            stem = f"{spec['source_id']}__series{rank}"
            path = PREPARED / f"{stem}.npz"
            np.savez_compressed(path, **arrays)
            tasks.append({"base_task_id": stem, "task_id": stem,
                          "source_id": spec["source_id"], "family_id": spec["family_id"],
                          "context": spec["context"], "horizon": spec["horizon"],
                          "selection_hash": key, "array_file": path.name,
                          "array_file_sha256": digest(path), **metadata})
    roster = {"schema_version": 1, "settings": SETTINGS,
              "source_manifest_sha256": digest(MANIFEST_PATH),
              "sources": sources, "tasks": tasks,
              "purpose": "Frozen new-collection roster selected before model outcomes; no pretest refit; unsupported later splits retained."}
    write_json(ROSTER_PATH, roster)
    print(json.dumps({"tasks": len(tasks), "supported": sum(t["supported"] for t in tasks),
                      "roster_sha256": digest(ROSTER_PATH)}, indent=2), flush=True)
    return roster


def load_natural(task):
    path = PREPARED / task["array_file"]
    if digest(path) != task["array_file_sha256"]:
        raise ValueError("Prepared file bytes changed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for key, expected in task["arrays_sha256"].items():
        if array_hash(arrays[key]) != expected:
            raise ValueError(f"Prepared array content changed: {key}")
    return arrays


def verify():
    roster = json.loads(ROSTER_PATH.read_text())
    if digest(MANIFEST_PATH) != roster["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    for task in roster["tasks"]:
        arrays = load_natural(task)
        for split, start, end in (("train", 0, task["train_end"]),
                                  ("validation", task["train_end"], task["validation_end"]),
                                  ("test", task["validation_end"], task["test_end"])):
            origins = arrays[f"origins_{split}"]
            if len(origins) and (np.any(origins + 1 < start) or np.any(origins + 1 + task["horizon"] > end)):
                raise ValueError("A target crosses a split boundary")
    return {"verified_tasks": len(roster["tasks"]), "roster_sha256": digest(ROSTER_PATH)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
