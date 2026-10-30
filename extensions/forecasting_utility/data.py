"""Pinned new sources and paired synthetic arrays; no model outcomes are used."""
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
from extensions.predictive_geometry.study.datasets import make_forecast_split

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
MANIFEST_PATH = ROOT / "sources.json"
PREPARED = ROOT / "prepared"
ROSTER_PATH = PREPARED / "roster.json"


def config():
    return json.loads(CONFIG_PATH.read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "KAHM-forecasting-utility/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def pin_sources():
    """Archive official metadata and bytes before eligibility/model evaluation."""
    cache = ROOT / "cache"
    metadata_dir = ROOT / "source_metadata"
    cache.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        for source in manifest["sources"]:
            path = cache / source["archive"]["filename"]
            if not path.exists():
                path.write_bytes(_fetch(source["archive"]["url"]))
            if digest(path) != source["archive"]["sha256"]:
                raise ValueError(f"Source differs from pinned archive: {source['source_id']}")
        return manifest
    sources = []
    for spec in config()["natural"]["sources"]:
        record_id = spec["record_id"]
        print(f"Pinning official source metadata and archive: {spec['source_id']}", flush=True)
        metadata_bytes = _fetch(f"https://zenodo.org/api/records/{record_id}")
        metadata = json.loads(metadata_bytes)
        (metadata_dir / f"{record_id}.json").write_bytes(metadata_bytes)
        candidates = [item for item in metadata["files"] if item["key"].endswith(".zip")]
        if len(candidates) != 1:
            raise ValueError("Expected one explicitly identifiable TSF zip in source record")
        entry = candidates[0]
        url = entry["links"]["self"]
        path = cache / entry["key"]
        if not path.exists():
            path.write_bytes(_fetch(url))
        raw = path.read_bytes()
        if len(raw) != entry["size"]:
            raise ValueError("Source size mismatch")
        algorithm, upstream_hash = entry["checksum"].split(":", 1)
        if hashlib.new(algorithm, raw).hexdigest() != upstream_hash:
            raise ValueError("Source checksum mismatch")
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".tsf")]
            if len(members) != 1:
                raise ValueError("Expected a single TSF archive member")
            h = hashlib.sha256()
            with archive.open(members[0]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
        info = metadata["metadata"]
        sources.append({
            **spec, "role": "confirmation", "title": info["title"],
            "license": info.get("license"), "creators": info.get("creators"),
            "description": info.get("description"), "related_identifiers": info.get("related_identifiers", []),
            "record_url": f"https://zenodo.org/records/{record_id}",
            "doi": info.get("doi", f"10.5281/zenodo.{record_id}"),
            "version": info.get("version"),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "archive": {"filename": entry["key"], "url": url, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(), "upstream_checksum": entry["checksum"],
                        "member": members[0], "member_sha256": h.hexdigest(), "encoding": "cp1252"},
        })
    manifest = {"schema_version": 1, "selection_basis": "Declared source families and metadata before any new model outcomes.",
                "archive_url": "https://forecastingdata.org/", "sources": sources}
    write_json(MANIFEST_PATH, manifest)
    return manifest


def iter_series(spec):
    """Stream TSF records so the long Weather archive is not kept in memory."""
    path = ROOT / "cache" / spec["archive"]["filename"]
    if digest(path) != spec["archive"]["sha256"]:
        raise ValueError("Source checksum changed")
    attributes, started, frequency = [], False, None
    with zipfile.ZipFile(path) as archive, archive.open(spec["archive"]["member"]) as binary:
        with io.TextIOWrapper(binary, encoding=spec["archive"]["encoding"]) as stream:
            for raw in stream:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("@"):
                    pieces = line.split()
                    key = pieces[0].lower()
                    if key == "@attribute":
                        attributes.append((pieces[1], pieces[2]))
                    elif key == "@frequency":
                        frequency = pieces[1]
                    elif key == "@data":
                        started = True
                    continue
                if not started:
                    raise ValueError("Data encountered before TSF header")
                fields = line.split(":", len(attributes))
                if len(fields) != len(attributes) + 1:
                    raise ValueError("Unexpected TSF attribute count")
                values = np.fromstring(fields[-1].replace("?", "nan"), sep=",", dtype=np.float64)
                if len(values) != fields[-1].count(",") + 1:
                    raise ValueError("Malformed source values")
                attr = dict(zip((name for name, _ in attributes), fields[:-1]))
                series_id = attr.get("series_name", fields[0])
                yield inherited.SeriesRecord(spec["source_id"], series_id, values,
                                             attr.get("start_timestamp", "not supplied"), frequency or spec["frequency"])


def select_records(records, spec, settings):
    """Keep the first three hash-ranked eligible records, using only training data."""
    retained, excluded, eligible_count, total = [], {}, 0, 0
    salt = settings["selection_salt"]
    for record in records:
        total += 1
        train_end, _, _ = inherited.split_cutoffs(len(record.values))
        prefix = record.values[:train_end]
        try:
            inherited.training_mse_scale(prefix, len(prefix), spec["seasonal_period"])
            origins, _ = inherited.valid_origins(prefix, spec["context"], spec["horizon"], 0, len(prefix))
            if len(origins) < settings["minimum_train_windows"]:
                raise ValueError("insufficient_finite_training_windows")
        except ValueError as error:
            excluded[record.series_id] = str(error)
            continue
        eligible_count += 1
        key = hashlib.sha256(f"{salt}\0{spec['source_id']}\0{record.series_id}".encode()).hexdigest()
        retained.append((key, record))
        retained.sort(key=lambda item: (item[0], item[1].series_id))
        del retained[settings["max_series_per_source"]:]
    return retained, {"source_series_count": total, "eligible_count": eligible_count,
                      "excluded_training_only": excluded, "selection_salt": salt}


def natural_arrays(record, spec, settings):
    ends = inherited.split_cutoffs(len(record.values))
    scale, scale_method, pairs = inherited.training_mse_scale(record.values, ends[0], spec["seasonal_period"])
    train = inherited.make_windows(record.values, spec["context"], spec["horizon"], 0, ends[0], settings["max_train_windows"])
    validation = inherited.make_windows(record.values, spec["context"], spec["horizon"], ends[0], ends[1], settings["max_eval_windows"])
    test = inherited.make_windows(record.values, spec["context"], spec["horizon"], ends[1], ends[2], settings["max_eval_windows"])
    refit = inherited.make_windows(record.values, spec["context"], spec["horizon"], 0, ends[1], settings["max_train_windows"])
    parts = {"train": train, "validation": validation, "test": test, "refit": refit}
    arrays = {f"{key}_{name}": getattr(part, key) for name, part in parts.items() for key in ("X", "Y", "origins")}
    arrays["values"] = record.values
    meta = {"series_id": record.series_id, "frequency": record.frequency, "n_values": len(record.values),
            "seasonal_period": spec["seasonal_period"],
            "train_end": ends[0], "validation_end": ends[1], "test_end": ends[2],
            "loss_scale": scale, "scale_method": scale_method, "scale_pair_count": pairs,
            "split_details": {name: part.to_dict() for name, part in parts.items()},
            "supported": all(len(part.X) > 0 for part in parts.values()),
            "arrays_sha256": {name: array_hash(array) for name, array in arrays.items()}}
    return arrays, meta


def prepare():
    if ROSTER_PATH.exists():
        raise FileExistsError("Prepared roster already exists; verify it instead of selecting another roster")
    settings = config()["natural"]
    manifest = pin_sources()
    PREPARED.mkdir(exist_ok=True, parents=True)
    tasks, sources = [], []
    for spec in manifest["sources"]:
        print(f"Training-prefix eligibility and hash selection: {spec['source_id']}", flush=True)
        selected, details = select_records(iter_series(spec), spec, settings)
        sources.append({"source_id": spec["source_id"], "family_id": spec["family_id"], **details,
                        "selected_series": [record.series_id for _, record in selected]})
        if len(selected) != settings["max_series_per_source"]:
            raise RuntimeError("Fewer than the declared three eligible series; stop before any model outcomes")
        for rank, (selection_hash, record) in enumerate(selected):
            arrays, metadata = natural_arrays(record, spec, settings)
            stem = f"{spec['source_id']}__series{rank}"
            path = PREPARED / f"{stem}.npz"
            np.savez_compressed(path, **arrays)
            tasks.append({"base_task_id": stem, "source_id": spec["source_id"], "family_id": spec["family_id"],
                          "context": spec["context"], "horizon": spec["horizon"], "selection_hash": selection_hash,
                          "array_file": path.name, "array_file_sha256": digest(path), **metadata})
    roster = {"schema_version": 1, "source_manifest_sha256": digest(MANIFEST_PATH),
              "config_sha256": digest(CONFIG_PATH), "sources": sources, "tasks": tasks,
              "purpose": "Training-only eligibility and frozen hash roster, no model outcomes."}
    write_json(ROSTER_PATH, roster)
    print(json.dumps({"base_tasks": len(tasks), "supported": sum(t["supported"] for t in tasks),
                      "roster_sha256": digest(ROSTER_PATH)}, indent=2))
    return roster


def load_natural(task):
    path = PREPARED / task["array_file"]
    if digest(path) != task["array_file_sha256"]:
        raise ValueError("Prepared data bytes changed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for key, expected in task["arrays_sha256"].items():
        if array_hash(arrays[key]) != expected:
            raise ValueError("Prepared array content changed")
    return arrays


def synthetic_splits(law, repetition_seed, training_episodes, settings=None):
    cfg = config()["synthetic"] if settings is None else settings
    law_index = cfg["laws"].index(law)
    result, metadata = {}, {}
    sizes = {"train": max(cfg["training_episodes"]), "validation": cfg["validation_episodes"], "test": cfg["test_episodes"]}
    for split_index, (split, count) in enumerate(sizes.items()):
        seed = int(np.random.SeedSequence([repetition_seed, law_index, split_index, 731]).generate_state(1)[0])
        part = make_forecast_split(law, count, seed, history=cfg["history"], horizons=tuple(cfg["horizons"]), stride=cfg["stride"])
        keep = part["groups"] < training_episodes if split == "train" else np.ones(len(part["X"]), dtype=bool)
        for key in ("X", "Y", "groups"):
            result[f"{key}_{split}"] = part[key][keep]
        result[f"origins_{split}"] = np.tile(part["origins"], count)[keep]
        metadata[split] = {**part["metadata"], "retained_episodes": training_episodes if split == "train" else count}
    result["X_refit"], result["Y_refit"] = result["X_train"], result["Y_train"]
    result["origins_refit"], result["groups_refit"] = result["origins_train"], result["groups_train"]
    metadata["arrays_sha256"] = {name: array_hash(value) for name, value in result.items()}
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.parse_args()
    prepare()


if __name__ == "__main__":
    main()
