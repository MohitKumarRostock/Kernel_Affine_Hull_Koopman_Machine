"""Prepare the prospectively reserved natural sources without fitting a model.

The roster uses identifier hashes and initial-A eligibility only. No later
segment value, forecast, or error is used for series selection. Run this module
before freezing the protocol and before running any forecasting method.

Raw public archives stay outside the repository. The small selected arrays,
source/license provenance, exact raw indices, and audit masks are versioned.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SELECTION_SALT = "kahm-direct-reconstruction-reserved-natural-v1"
CAPS = {"A": 512, "B": 128, "C": 128, "D": 128}
MIN_A_WINDOWS = 96
SOURCES = (
    dict(source_key="dominick_weekly", record_id=4654802,
         title="Dominick Dataset", filename="dominick_dataset.zip",
         expected_bytes=12328941, expected_md5="c84cefdce4775399b5eff23085c15013",
         n_select=3, L=52, p=52, offsets=[1, 2, 4], frequency="weekly",
         original_source="https://www.chicagobooth.edu/research/kilts/datasets/dominicks",
         description="Weekly retailer SKU profit series."),
    dict(source_key="rideshare_price_hourly", record_id=5122114,
         title="Rideshare Dataset (with Missing Values)",
         filename="rideshare_dataset_with_missing_values.zip",
         expected_bytes=1031826, expected_md5="b5faa7112787bb6d42e5de56462de7c7",
         n_select=3, L=24, p=24, offsets=[1, 3, 6], frequency="hourly",
         metadata_filter={"type": "price_mean"},
         original_source="https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices",
         description="Hourly rideshare mean-price series; other attribute types excluded.",
         provenance_note="The archive description names a geography that has not been independently reconciled; no geography-dependent claim is made."),
    dict(source_key="wikipedia_daily", record_id=4656080,
         title="Kaggle Wikipedia Web Traffic Dataset (with Missing Values)",
         filename="kaggle_web_traffic_dataset_with_missing_values.zip",
         expected_bytes=145485324, expected_md5="144bd663c10656115b2bab14c017c6a4",
         n_select=3, L=28, p=7, offsets=[1, 3, 7], frequency="daily",
         original_source="https://www.kaggle.com/c/web-traffic-time-series-forecasting",
         description="Daily Wikipedia page-hit series; missing observations retained."),
    dict(source_key="sunspot_daily", record_id=4654773,
         title="Sunspot Dataset (with Missing Values)",
         filename="sunspot_dataset_with_missing_values.zip",
         expected_bytes=68865, expected_md5="9d25d55656f0fa6ad78e905d5e236727",
         n_select=1, L=28, p=1, offsets=[1, 3, 7], frequency="daily", tail=4096,
         original_source="https://www.sidc.be/silso/newdataset",
         description="One daily sunspot-number series, fixed final 4096 observations."),
)

PRIOR_INSPECTED_COLLECTIONS = [
    "Electricity weekly", "NN5 daily", "Tourism monthly", "FRED-MD",
    "Traffic weekly (road occupancy)", "Hospital monthly", "Saugeen River",
    "US births", "Weather", "Solar Weekly", "COVID-19 Deaths", "Car Parts",
    "Australian state electricity", "Bitcoin activity indicators",
    "Melbourne pedestrian counts", "KDD2018 air quality",
]


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    """Download to a partial file, then publish the completed local cache entry."""
    partial = path.with_name(path.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "KAHM-reproducible-study/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
    partial.replace(path)


def get_source(source: dict, raw_dir: Path) -> tuple[Path, dict]:
    """Resolve a pinned archive and verify the publisher's checksum and license."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_id = source["record_id"]
    metadata_path = raw_dir / f"{record_id}.json"
    if not metadata_path.exists():
        download(f"https://zenodo.org/api/records/{record_id}", metadata_path)
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = record["metadata"]
    if metadata.get("license", {}).get("id") != "cc-by-4.0":
        raise ValueError(f"Unexpected archive license: {record_id}")
    remote_file = next(f for f in record["files"] if f["key"] == source["filename"])
    if remote_file["checksum"] != "md5:" + source["expected_md5"]:
        raise ValueError(f"Publisher checksum differs from pinned protocol: {record_id}")
    path = raw_dir / source["filename"]
    download_url = f"https://zenodo.org/api/records/{record_id}/files/{source['filename']}/content"
    if not path.exists():
        download(download_url, path)
    if path.stat().st_size != source["expected_bytes"]:
        raise ValueError(f"Unexpected archive size: {path}")
    if file_digest(path, "md5") != source["expected_md5"]:
        raise ValueError(f"Archive checksum mismatch: {path}")
    provenance = {
        **source,
        "doi": metadata["doi"],
        "record_url": f"https://zenodo.org/records/{record_id}",
        "metadata_url": f"https://zenodo.org/api/records/{record_id}",
        "download_url": download_url,
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_scope": "License declared by the pinned Zenodo archive record.",
        "creators": metadata.get("creators", []),
        "publication_date": metadata.get("publication_date"),
        "version": metadata.get("version"),
        "archive_bytes": path.stat().st_size,
        "archive_md5": source["expected_md5"],
        "archive_sha256": file_digest(path),
    }
    return path, provenance


def parse_numbers(text: str) -> np.ndarray:
    tokens = text.split(",")
    values = np.fromstring(text.replace("?", "nan"), sep=",", dtype=np.float64)
    if len(values) != len(tokens):
        raise ValueError("Malformed numeric TSF row")
    return values


def chronological_bounds(n: int) -> np.ndarray:
    # Integer arithmetic avoids floating-point ambiguity in the frozen splits.
    return np.asarray([0, n * 50 // 100, n * 70 // 100, n * 85 // 100, n], dtype=np.int64)


def eligible_origins(values: np.ndarray, L: int, offsets: np.ndarray,
                     segment_start: int, segment_stop: int) -> tuple[np.ndarray, int]:
    """Finite histories and requested targets; unused intermediate targets ignored."""
    first = max(L - 1, segment_start - int(offsets.min()))
    stop = segment_stop - int(offsets.max())
    origins = np.arange(first, max(first, stop), dtype=np.int64)
    candidate_count = len(origins)
    if not candidate_count:
        return origins, candidate_count
    finite = np.isfinite(values)
    bad_prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(~finite)))
    history_ok = bad_prefix[origins + 1] == bad_prefix[origins - L + 1]
    targets_ok = np.all(finite[origins[:, None] + offsets[None, :]], axis=1)
    return origins[history_ok & targets_ok], candidate_count


def training_loss_scale(prefix: np.ndarray, p: int) -> tuple[float, str, int]:
    """Squared seasonal differences in raw A only, then first-difference fallback."""
    for lag, name in ((p, "seasonal_mse"), (1, "first_difference_mse_fallback")):
        if len(prefix) <= lag:
            continue
        valid = np.isfinite(prefix[lag:]) & np.isfinite(prefix[:-lag])
        differences = prefix[lag:][valid] - prefix[:-lag][valid]
        if len(differences):
            with np.errstate(over="ignore", invalid="ignore"):
                scale = float(np.mean(differences ** 2))
            if np.isfinite(scale) and scale > 0:
                return scale, name, len(differences)
    return float("nan"), "unavailable", 0


def identifier_hash(source_key: str, series_id: str) -> str:
    payload = "\0".join((SELECTION_SALT, source_key, series_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_series(archive: Path, source: dict) -> tuple[list[dict], dict]:
    """Stream each source; numerical eligibility looks only at each A prefix."""
    n_rows = 0
    n_metadata_eligible = 0
    n_a_eligible = 0
    duplicate_guard: set[str] = set()
    retained: list[dict] = []
    attributes: list[str] = []
    header: list[str] = []
    offsets = np.asarray(source["offsets"], dtype=np.int64)
    member_digest = hashlib.sha256()
    in_data = False
    with zipfile.ZipFile(archive) as zipped:
        members = [n for n in zipped.namelist() if n.lower().endswith(".tsf")]
        if len(members) != 1:
            raise ValueError("Expected exactly one TSF member")
        member = members[0]
        member_bytes = zipped.getinfo(member).file_size
        with zipped.open(member) as handle:
            for raw_line in handle:
                member_digest.update(raw_line)
                line = raw_line.decode("cp1252").strip()
                if not in_data:
                    header.append(line)
                    if line.lower().startswith("@attribute "):
                        attributes.append(line.split()[1])
                    elif line.lower() == "@data":
                        in_data = True
                    continue
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":", len(attributes))
                if len(parts) != len(attributes) + 1:
                    raise ValueError("Malformed TSF attributes")
                attrs = dict(zip(attributes, parts[:-1]))
                series_id = attrs["series_name"]
                if series_id in duplicate_guard:
                    raise ValueError(f"Nonunique source series identifier: {series_id}")
                duplicate_guard.add(series_id)
                n_rows += 1
                if any(attrs.get(k) != v for k, v in source.get("metadata_filter", {}).items()):
                    continue
                n_metadata_eligible += 1
                tokens = parts[-1].split(",")
                original_n = len(tokens)
                if source.get("tail") is not None:
                    tokens = tokens[-source["tail"]:]
                n = len(tokens)
                bounds = chronological_bounds(n)
                # Only A tokens are converted to numbers during roster selection.
                prefix = parse_numbers(",".join(tokens[:bounds[1]]))
                a_origins, _ = eligible_origins(prefix, source["L"], offsets, 0, int(bounds[1]))
                scale, scale_method, scale_pairs = training_loss_scale(prefix, source["p"])
                if len(a_origins) < MIN_A_WINDOWS or not np.isfinite(scale) or scale <= 0:
                    continue
                n_a_eligible += 1
                rank = identifier_hash(source["source_key"], series_id)
                if len(retained) < source["n_select"] or rank < retained[-1]["identifier_hash"]:
                    retained.append({
                        "series_id": series_id, "attributes": attrs,
                        "identifier_hash": rank, "raw_text": ",".join(tokens),
                        "original_raw_length": original_n, "raw_length": n,
                        "raw_start_index": original_n - n,
                        "loss_scale": scale, "loss_scale_method": scale_method,
                        "loss_scale_pair_count": scale_pairs,
                        "A_eligible_window_count": len(a_origins),
                    })
                    retained.sort(key=lambda item: item["identifier_hash"])
                    del retained[source["n_select"]:]
    if len(retained) != source["n_select"]:
        raise ValueError(f"Insufficient A-eligible series in {source['source_key']}: {len(retained)}")
    audit = {
        "tsf_member": member, "tsf_member_bytes": member_bytes,
        "tsf_member_sha256": member_digest.hexdigest(), "tsf_header": header,
        "source_series_count": n_rows, "metadata_eligible_series_count": n_metadata_eligible,
        "A_eligible_series_count": n_a_eligible,
        "selected_series_ids_in_hash_order": [r["series_id"] for r in retained],
    }
    return retained, audit


def cap_positions(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    # Both ends included; floor defines the same deterministic subsample on every run.
    return (np.arange(cap, dtype=np.int64) * (n - 1)) // (cap - 1)


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write stable ZIP members so independent preparations reproduce byte hashes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.kind in "iu":
                value = value.astype("<i8")
            elif value.dtype.kind == "f":
                value = value.astype("<f8")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def prepare_task(selected: dict, source: dict, output: Path, rank: int) -> dict:
    selected = dict(selected)
    values = parse_numbers(selected.pop("raw_text"))
    bounds = chronological_bounds(len(values))
    offsets = np.asarray(source["offsets"], dtype=np.int64)
    L, p, hmax = source["L"], source["p"], int(offsets.max())
    if not 1 <= p <= L:
        raise ValueError("Seasonal naive must be recoverable from the lag history")
    task_id = f"{source['source_key']}__{rank + 1:02d}"
    arrays = {
        "raw_values": values, "raw_start_index": np.asarray(selected["raw_start_index"]),
        "offsets": offsets, "segment_bounds": bounds,
        "L": np.asarray(L), "p": np.asarray(p),
        "loss_scale": np.asarray(selected["loss_scale"]),
    }
    split_counts = {}
    for j, split in enumerate("ABCD"):
        start, stop = int(bounds[j]), int(bounds[j + 1])
        available, candidate_count = eligible_origins(values, L, offsets, start, stop)
        positions = cap_positions(len(available), CAPS[split])
        origins = available[positions]
        history_indices = origins[:, None] - np.arange(L - 1, -1, -1)[None, :]
        target_indices = origins[:, None] + offsets[None, :]
        arrays[split + "_X"] = values[history_indices]
        arrays[split + "_Y"] = values[target_indices]
        arrays[split + "_origins"] = origins
        arrays[split + "_global_origins"] = origins + selected["raw_start_index"]
        arrays[split + "_groups"] = np.zeros(len(origins), dtype=np.int64)
        arrays[split + "_row_indices"] = positions
        arrays[split + "_available_origins"] = available
        # Explicit indices are redundant by design and allow independent reconstruction.
        arrays[split + "_history_indices"] = history_indices
        arrays[split + "_target_indices"] = target_indices
        split_counts[split] = {
            "structurally_available_windows": candidate_count,
            "excluded_missing_windows": candidate_count - len(available),
            "finite_available_windows": len(available), "retained_windows": len(origins),
            "cap": CAPS[split],
        }
        assert np.all(target_indices >= start) and np.all(target_indices < stop)
        assert np.all(np.isfinite(arrays[split + "_X"]))
        assert np.all(np.isfinite(arrays[split + "_Y"]))
    a_origins = arrays["A_origins"]
    n_train_nominal = 4 * len(a_origins) // 5
    validation_mask = np.arange(len(a_origins)) >= n_train_nominal
    first_validation_origin = int(a_origins[n_train_nominal])
    train_mask = (~validation_mask) & (a_origins + hmax <= first_validation_origin)
    arrays["A_inner_train_mask"] = train_mask
    arrays["A_inner_validation_mask"] = validation_mask
    arrays["A_inner_purged_mask"] = (~train_mask) & (~validation_mask)
    assert len(a_origins) >= MIN_A_WINDOWS
    assert train_mask.sum() >= 2 and validation_mask.sum() >= 1
    assert np.max(a_origins[train_mask] + hmax) <= first_validation_origin
    for previous, current in zip("ABC", "BCD"):
        previous_origins = arrays[previous + "_origins"]
        current_origins = arrays[current + "_origins"]
        if len(previous_origins) and len(current_origins):
            assert int(previous_origins[-1]) + hmax <= int(current_origins[0])
    output.mkdir(parents=True, exist_ok=True)
    filename = task_id + ".npz"
    path = output / filename
    deterministic_npz(path, arrays)
    return {
        "task_id": task_id, "source_key": source["source_key"], **selected,
        "L": L, "p": p, "Hmax": hmax, "offsets": offsets.tolist(),
        "npz_path": "data/" + filename,
        "prepared_npz_sha256": file_digest(path), "prepared_npz_bytes": path.stat().st_size,
        "split_bounds": bounds.tolist(),
        "segment_bounds": {s: [int(bounds[j]), int(bounds[j + 1])] for j, s in enumerate("ABCD")},
        "split_counts": split_counts,
        "inner_A": {
            "validation_rule": "last 20% of chronologically ordered capped A rows (integer split at floor(4*n/5))",
            "training_rule": "earlier A rows whose max-horizon label is observed by the first validation origin",
            "first_validation_origin": first_validation_origin,
            "train_rows": int(train_mask.sum()), "validation_rows": int(validation_mask.sum()),
            "purged_rows": int(arrays["A_inner_purged_mask"].sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT.parents[2] / "raw_direct_sources")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    manifest = {
        "schema_version": 1,
        "study": "direct_reconstruction_reserved_natural_v1",
        "selection_salt": SELECTION_SALT,
        "selection_rule": "For each source: filter source metadata; require >=96 finite A windows and positive A-only loss scale; choose smallest SHA256(salt + NUL + source_key + NUL + series_name).",
        "eligibility_information": "Only metadata, series length, and numerical values in the initial raw A prefix.",
        "future_screening": "None. Retain selected tasks even if later segments contain no eligible windows; report such failures.",
        "prior_inspected_collections_excluded": PRIOR_INSPECTED_COLLECTIONS,
        "pre_freeze_design_log": "PRE_FREEZE_DESIGN_LOG.md",
        "split_rule": "Raw-index boundaries [0,floor(.50*n),floor(.70*n),floor(.85*n),n].",
        "window_rule": "At origin t, X=y[t-L+1:t+1] oldest-to-newest; Y[j]=y[t+offsets[j]]. All requested targets must lie inside their segment; histories may overlap earlier segments. Exclude windows with nonfinite history or requested target; do not impute or require unused intermediate targets.",
        "maturity_rule": "Prior-segment labels and inner-A training labels must be observed by the next segment's first query origin; observations at the origin are available.",
        "row_caps": CAPS,
        "row_cap_rule": "If n>cap, select available row indices floor(j*(n-1)/(cap-1)), j=0,...,cap-1; otherwise keep every available row.",
        "loss_scale_rule": "Mean squared lag-p differences using finite pairs wholly in raw A. If unavailable, nonfinite or nonpositive, use mean squared lag-1 differences wholly in A. Require a positive finite scale. Scaled MSE = raw MSE / this scale (no square root).",
        "inner_A_rule": "The final 20% of retained chronological A origins form inner validation. Inner training uses earlier A rows satisfying origin+Hmax <= first validation origin. Purged rows remain available for final A fitting.",
        "seasonal_naive_rule": "At origin t and target offset h, predict y[t+h-ceil(h/p)*p]. With p=1 this is persistence. All p<=L.",
        "estimand": "Available finite windows within each selected series, equal-series means within each source, then equal-source means. These four chosen collections support descriptive comparisons, not population confidence intervals.",
        "task_count": sum(s["n_select"] for s in SOURCES),
        "source_count": len(SOURCES),
        "sources": [], "tasks": [],
    }
    for source in SOURCES:
        archive, provenance = get_source(source, args.raw_dir)
        selected, audit = select_series(archive, source)
        tasks = [prepare_task(item, source, args.output, rank) for rank, item in enumerate(selected)]
        provenance.update(audit)
        provenance["selected_task_ids"] = [task["task_id"] for task in tasks]
        manifest["sources"].append(provenance)
        manifest["tasks"].extend(tasks)
        print(json.dumps({"source": source["source_key"], "source_series": audit["source_series_count"],
                          "A_eligible_series": audit["A_eligible_series_count"],
                          "selected": audit["selected_series_ids_in_hash_order"]}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "sources_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(destination), "sha256": file_digest(destination),
                      "tasks": len(manifest["tasks"]), "models_fitted": 0}), flush=True)


if __name__ == "__main__":
    main()
