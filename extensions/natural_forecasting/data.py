"""Frozen public-source roster and chronological univariate forecasting tasks.

This layer performs no learned preprocessing.  All X/Y arrays retain the values
in the cited source snapshots.  A forecast origin is the last observed index;
its H targets are [origin + 1, origin + 1 + H).  Every complete target interval
must lie inside one split.  Earlier observed values may appear in later contexts.

Series eligibility and the fixed hash roster inspect training prefixes only.
Missing contexts/targets are excluded by availability, never interpolated.  A
selected series with no usable later split is retained with supported=False;
it is not replaced by a series chosen using evaluation-period information.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import urllib.request
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data_sources.json"
DEFAULT_CACHE = ROOT / "datasets" / "raw"
ROSTER_SALT = "natural-forecasting-roster-v1"


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SeriesRecord:
    source_id: str
    series_id: str
    values: np.ndarray
    start_timestamp: str
    frequency: str


@dataclass(frozen=True)
class WindowSet:
    X: np.ndarray
    Y: np.ndarray
    origins: np.ndarray
    candidate_count: int
    available_count: int

    @property
    def target_start(self) -> np.ndarray:
        return self.origins + 1

    @property
    def target_end(self) -> np.ndarray:
        return self.origins + 1 + self.Y.shape[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": len(self.origins),
            "candidate_count": self.candidate_count,
            "available_count_before_cap": self.available_count,
            "unavailable_windows": self.candidate_count - self.available_count,
            "origins": self.origins.tolist(),
            "target_start": self.target_start.tolist(),
            "target_end": self.target_end.tolist(),
        }


@dataclass(frozen=True)
class ForecastTask:
    task_id: str
    source_id: str
    family_id: str
    role: str
    series_id: str
    frequency: str
    start_timestamp: str
    context: int
    horizon: int
    seasonal_period: int
    loss_scale: float
    scale_method: str
    scale_pair_count: int
    values: np.ndarray
    train_end: int
    validation_end: int
    test_end: int
    X_train: np.ndarray
    Y_train: np.ndarray
    origin_train: np.ndarray
    X_validation: np.ndarray
    Y_validation: np.ndarray
    origin_validation: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    origin_test: np.ndarray
    split_details: Mapping[str, Mapping[str, Any]] = field(repr=False)
    provenance: Mapping[str, Any] = field(repr=False)
    selection_hash: str = ""
    supported: bool = True
    unsupported_reason: str | None = None

    @property
    def validation_start(self) -> int:
        return self.train_end

    @property
    def test_start(self) -> int:
        return self.validation_end

    @property
    def target_start_train(self) -> np.ndarray:
        return self.origin_train + 1

    @property
    def target_end_train(self) -> np.ndarray:
        return self.origin_train + 1 + self.horizon

    @property
    def target_start_validation(self) -> np.ndarray:
        return self.origin_validation + 1

    @property
    def target_end_validation(self) -> np.ndarray:
        return self.origin_validation + 1 + self.horizon

    @property
    def target_start_test(self) -> np.ndarray:
        return self.origin_test + 1

    @property
    def target_end_test(self) -> np.ndarray:
        return self.origin_test + 1 + self.horizon

    def to_dict(self) -> dict[str, Any]:
        """JSON metadata, including every selected origin and exclusive bounds."""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "role": self.role,
            "series_id": self.series_id,
            "frequency": self.frequency,
            "start_timestamp": self.start_timestamp,
            "context": self.context,
            "horizon": self.horizon,
            "seasonal_period": self.seasonal_period,
            "n_observations": len(self.values),
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "loss_scale": self.loss_scale,
            "scale_method": self.scale_method,
            "scale_pair_count": self.scale_pair_count,
            "scale_max_observation_index": self.train_end - 1,
            "selection_hash": self.selection_hash,
            "selection_salt": ROSTER_SALT,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "split_details": dict(self.split_details),
            "provenance": dict(self.provenance),
            "index_convention": "zero-based; origin last observed; target_end/split_end exclusive",
            "preprocessing": "none; fit learned preprocessing on training data in each model",
        }


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported source-manifest schema")
    identifiers = [s["source_id"] for s in manifest["sources"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate source identifiers")
    family_roles: dict[str, str] = {}
    for source in manifest["sources"]:
        role = source["role"]
        if role not in {"development", "confirmation"}:
            raise ValueError(f"Invalid source role: {role}")
        family = source["family_id"]
        if family in family_roles and family_roles[family] != role:
            raise ValueError(f"Source family crosses development/confirmation: {family}")
        family_roles[family] = role
    return manifest


def _source_spec(source_id: str, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    for source in manifest["sources"]:
        if source["source_id"] == source_id:
            return source
    raise KeyError(f"Unknown source: {source_id}")


def _source_subset(manifest: Mapping[str, Any], role: str) -> list[Mapping[str, Any]]:
    if role not in {"development", "confirmation", "all"}:
        raise ValueError("role must be development, confirmation, or all")
    return [s for s in manifest["sources"] if role == "all" or s["role"] == role]


def verify_archive(path: str | Path, spec: Mapping[str, Any]) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing public archive {path.name}; run python data.py fetch --role all")
    archive = spec["archive"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != archive["bytes"] or digest != archive["sha256"]:
        raise ValueError(f"Frozen source checksum/size mismatch: {path.name}")


def fetch_sources(
    role: str = "development", *, cache_dir: str | Path = DEFAULT_CACHE,
    manifest_path: str | Path = MANIFEST_PATH,
) -> list[Path]:
    """Fetch only manifest-pinned small public archives, atomically and serially."""
    manifest = load_manifest(manifest_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in _source_subset(manifest, role):
        archive = spec["archive"]
        destination = cache / archive["filename"]
        if destination.exists():
            verify_archive(destination, spec)
            paths.append(destination)
            continue
        request = urllib.request.Request(archive["url"], headers={"User-Agent": "natural-forecasting-reproduction/1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read(archive["bytes"] + 1)
        if len(content) != archive["bytes"] or hashlib.sha256(content).hexdigest() != archive["sha256"]:
            raise ValueError(f"Downloaded archive differs from frozen snapshot: {spec['source_id']}")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, prefix="download-", suffix=".part", delete=False) as handle:
                handle.write(content)
                temporary = Path(handle.name)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        paths.append(destination)
    return paths


def parse_tsf(text: str, source_id: str) -> list[SeriesRecord]:
    """Read the explicit string/date/univariate TSF format of the fixed sources."""
    attributes: list[tuple[str, str]] = []
    frequency: str | None = None
    in_data = False
    records: list[SeriesRecord] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            if in_data:
                raise ValueError(f"Unexpected TSF metadata after @data (line {line_number})")
            pieces = line.split()
            directive = pieces[0].lower()
            if directive == "@attribute":
                if len(pieces) != 3:
                    raise ValueError(f"Invalid TSF attribute at line {line_number}")
                attributes.append((pieces[1], pieces[2].lower()))
            elif directive == "@frequency":
                frequency = pieces[1]
            elif directive == "@data":
                in_data = True
                if attributes != [("series_name", "string"), ("start_timestamp", "date")]:
                    raise ValueError("This loader requires the manifest's series_name/start_timestamp TSF schema")
                if frequency not in {"daily", "weekly", "monthly"}:
                    raise ValueError(f"Unsupported/missing TSF frequency: {frequency}")
            continue
        if not in_data:
            raise ValueError(f"TSF observation before @data (line {line_number})")
        pieces = line.split(":")
        if len(pieces) != 3:
            raise ValueError(f"Invalid TSF row at line {line_number}")
        series_id, timestamp, numeric_values = pieces
        if series_id in seen:
            raise ValueError(f"Duplicate TSF series identifier: {series_id}")
        seen.add(series_id)
        start = datetime.strptime(timestamp, "%Y-%m-%d %H-%M-%S").isoformat()
        try:
            values = np.array([np.nan if x.strip() == "?" else float(x) for x in numeric_values.split(",")], dtype=np.float64)
        except ValueError as error:
            raise ValueError(f"Invalid TSF number at line {line_number}") from error
        if np.isinf(values).any():
            raise ValueError("Infinite source values are unsupported; use explicit TSF missing markers")
        records.append(SeriesRecord(source_id, series_id, _readonly(values), start, str(frequency)))
    if not in_data or not records:
        raise ValueError("TSF file has no series")
    return records


def load_source(
    source_id: str, *, cache_dir: str | Path = DEFAULT_CACHE,
    manifest_path: str | Path = MANIFEST_PATH,
) -> list[SeriesRecord]:
    manifest = load_manifest(manifest_path)
    spec = _source_spec(source_id, manifest)
    archive_path = Path(cache_dir) / spec["archive"]["filename"]
    verify_archive(archive_path, spec)
    with zipfile.ZipFile(archive_path) as archive:
        content = archive.read(spec["archive"]["member"])
    if hashlib.sha256(content).hexdigest() != spec["archive"]["member_sha256"]:
        raise ValueError(f"Frozen TSF member checksum mismatch: {source_id}")
    records = parse_tsf(content.decode(spec["archive"]["encoding"]), source_id)
    if len(records) != spec["snapshot"]["series_count"]:
        raise ValueError(f"Unexpected series count: {source_id}")
    if any(r.frequency != spec["frequency"] for r in records):
        raise ValueError(f"Source frequency differs from manifest: {source_id}")
    return records


def split_cutoffs(length: int) -> tuple[int, int, int]:
    """60/20/20 by observation count, with integer floor at .6 and .8."""
    if length < 1:
        raise ValueError("A series must contain observations")
    return (3 * length // 5, 4 * length // 5, length)


def training_mse_scale(values: np.ndarray, train_end: int, seasonal_period: int) -> tuple[float, str, int]:
    """Train-prefix seasonal-difference MSE; positive first-difference fallback.

    At least two finite difference pairs are required.  No validation/test
    values, interpolation, or fitted transformation enters the denominator.
    """
    prefix = np.asarray(values, dtype=np.float64)[:train_end]
    if seasonal_period < 1 or train_end < 1 or train_end > len(values):
        raise ValueError("Invalid training scale bounds or seasonal period")
    finite = prefix[np.isfinite(prefix)]
    if len(finite) < 2 or np.max(finite) == np.min(finite):
        raise ValueError("constant_or_insufficient_training_prefix")
    lags = [(seasonal_period, "seasonal_naive_mse")]
    if seasonal_period != 1:
        lags.append((1, "first_difference_mse_fallback"))
    for lag, method in lags:
        if len(prefix) <= lag:
            continue
        difference = prefix[lag:] - prefix[:-lag]
        difference = difference[np.isfinite(difference)]
        if len(difference) < 2:
            continue
        scale = float(np.mean(np.square(difference)))
        if math.isfinite(scale) and scale > 0:
            return scale, method, len(difference)
    raise ValueError("no_positive_training_difference_scale")


def valid_origins(values: np.ndarray, context: int, horizon: int, start: int, end: int) -> tuple[np.ndarray, int]:
    """Origins with complete observed context/target and targets inside [start,end)."""
    values = np.asarray(values, dtype=np.float64)
    if context < 1 or horizon < 1 or not 0 <= start <= end <= len(values):
        raise ValueError("Invalid window dimensions or split bounds")
    first = max(context - 1, start - 1)
    last_exclusive = end - horizon
    candidates = np.arange(first, max(first, last_exclusive), dtype=np.int64)
    if not len(candidates):
        return candidates, 0
    missing_prefix = np.r_[0, np.cumsum(~np.isfinite(values), dtype=np.int64)]
    begin = candidates - context + 1
    stop = candidates + 1 + horizon
    available = missing_prefix[stop] == missing_prefix[begin]
    return candidates[available], len(candidates)


def capped_origins(origins: np.ndarray, maximum: int | None) -> np.ndarray:
    """Keep evenly spread chronological indices, determined without target values."""
    origins = np.asarray(origins, dtype=np.int64)
    if maximum is not None and maximum < 1:
        raise ValueError("Window cap must be positive")
    if maximum is None or len(origins) <= maximum:
        return origins.copy()
    if maximum == 1:
        return origins[-1:].copy()
    indices = np.arange(maximum, dtype=np.int64) * (len(origins) - 1) // (maximum - 1)
    return origins[indices]


def make_windows(
    values: np.ndarray, context: int, horizon: int, start: int, end: int,
    maximum: int | None = None,
) -> WindowSet:
    all_origins, candidate_count = valid_origins(values, context, horizon, start, end)
    origins = capped_origins(all_origins, maximum)
    x_indices = origins[:, None] - context + 1 + np.arange(context)[None, :]
    y_indices = origins[:, None] + 1 + np.arange(horizon)[None, :]
    return WindowSet(
        _readonly(np.asarray(values, dtype=np.float64)[x_indices].copy()),
        _readonly(np.asarray(values, dtype=np.float64)[y_indices].copy()),
        _readonly(origins), candidate_count, len(all_origins),
    )


def selection_digest(source_id: str, series_id: str) -> str:
    return hashlib.sha256(f"{ROSTER_SALT}\0{source_id}\0{series_id}".encode("utf-8")).hexdigest()


def select_series(
    records: Iterable[SeriesRecord], regimes: Sequence[Mapping[str, int]],
    seasonal_period: int, *, maximum: int = 4, minimum_train_windows: int = 16,
) -> tuple[list[SeriesRecord], dict[str, Any]]:
    """One fixed series roster shared by every regime and computational budget."""
    if maximum < 1 or minimum_train_windows < 1 or not regimes:
        raise ValueError("Invalid roster parameters")
    eligible: list[SeriesRecord] = []
    exclusions: dict[str, str] = {}
    for record in records:
        train_end, _, _ = split_cutoffs(len(record.values))
        # Deliberately slice before either eligibility operation, so both their
        # values and any missingness queries are confined to the train prefix.
        prefix = record.values[:train_end]
        try:
            training_mse_scale(prefix, len(prefix), seasonal_period)
        except ValueError as error:
            exclusions[record.series_id] = str(error)
            continue
        counts = [len(valid_origins(prefix, int(r["context"]), int(r["horizon"]), 0, len(prefix))[0]) for r in regimes]
        if min(counts) < minimum_train_windows:
            exclusions[record.series_id] = f"fewer_than_{minimum_train_windows}_finite_train_windows_in_at_least_one_regime"
            continue
        eligible.append(record)
    eligible.sort(key=lambda r: (selection_digest(r.source_id, r.series_id), r.series_id))
    selected = eligible[:maximum]
    return selected, {
        "eligible_series_count": len(eligible),
        "selected_series": [r.series_id for r in selected],
        "selected_hashes": [selection_digest(r.source_id, r.series_id) for r in selected],
        "excluded_series": exclusions,
        "selection_information": "series length/id and training prefix only; no validation/test eligibility filter",
        "minimum_finite_training_windows_each_regime": minimum_train_windows,
    }


def make_task(
    record: SeriesRecord, spec: Mapping[str, Any], regime: Mapping[str, int], *,
    max_train_windows: int = 512, max_eval_windows: int = 64,
    manifest_sha256: str = "",
) -> ForecastTask:
    context, horizon = int(regime["context"]), int(regime["horizon"])
    train_end, validation_end, test_end = split_cutoffs(len(record.values))
    seasonal_period = int(spec["seasonal_period"])
    scale, scale_method, scale_pairs = training_mse_scale(record.values, train_end, seasonal_period)
    train = make_windows(record.values, context, horizon, 0, train_end, max_train_windows)
    validation = make_windows(record.values, context, horizon, train_end, validation_end, max_eval_windows)
    test = make_windows(record.values, context, horizon, validation_end, test_end, max_eval_windows)
    unsupported = [name for name, windows in (("train", train), ("validation", validation), ("test", test)) if not len(windows.origins)]
    provenance = {
        "manifest_sha256": manifest_sha256,
        "zenodo_record": spec["zenodo"]["record_id"],
        "doi": spec["zenodo"]["doi"],
        "archive_sha256": spec["archive"]["sha256"],
        "member_sha256": spec["archive"]["member_sha256"],
        "upstream_version": spec["zenodo"]["version"],
        "license_id": spec["license"]["id"],
    }
    return ForecastTask(
        task_id=f"{record.source_id}__{record.series_id}__L{context}_H{horizon}",
        source_id=record.source_id, family_id=str(spec["family_id"]), role=str(spec["role"]),
        series_id=record.series_id, frequency=record.frequency, start_timestamp=record.start_timestamp,
        context=context, horizon=horizon, seasonal_period=seasonal_period,
        loss_scale=scale, scale_method=scale_method, scale_pair_count=scale_pairs, values=record.values,
        train_end=train_end, validation_end=validation_end, test_end=test_end,
        X_train=train.X, Y_train=train.Y, origin_train=train.origins,
        X_validation=validation.X, Y_validation=validation.Y, origin_validation=validation.origins,
        X_test=test.X, Y_test=test.Y, origin_test=test.origins,
        split_details={"train": train.to_dict(), "validation": validation.to_dict(), "test": test.to_dict()},
        provenance=provenance, selection_hash=selection_digest(record.source_id, record.series_id),
        supported=not unsupported,
        unsupported_reason=("no_finite_windows_in_" + "_and_".join(unsupported)) if unsupported else None,
    )


def load_tasks(
    role: str = "development", *, max_train_windows: int = 512, max_eval_windows: int = 64,
    max_series: int = 4, source_ids: Sequence[str] | None = None,
    cache_dir: str | Path = DEFAULT_CACHE, manifest_path: str | Path = MANIFEST_PATH,
) -> list[ForecastTask]:
    """Prepare fixed local-series tasks; no network, model fitting, or normalization."""
    if max_train_windows < 16 or max_eval_windows < 1 or max_series < 1:
        raise ValueError("Require >=16 training windows, positive eval cap and positive series cap")
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sources = _source_subset(manifest, role)
    if source_ids is not None:
        allowed = {s["source_id"] for s in sources}
        unknown = set(source_ids) - allowed
        if unknown:
            raise ValueError(f"Sources unavailable in requested role: {sorted(unknown)}")
        sources = [s for s in sources if s["source_id"] in source_ids]
    tasks: list[ForecastTask] = []
    for spec in sources:
        records = load_source(spec["source_id"], cache_dir=cache_dir, manifest_path=manifest_path)
        selected, _ = select_series(
            records, spec["regimes"], spec["seasonal_period"], maximum=max_series,
            minimum_train_windows=int(manifest["protocol"]["minimum_train_windows"]),
        )
        for record in selected:
            for regime in spec["regimes"]:
                tasks.append(make_task(record, spec, regime, max_train_windows=max_train_windows,
                    max_eval_windows=max_eval_windows, manifest_sha256=manifest_sha256))
    return tasks


def source_roster(
    role: str = "development", *, cache_dir: str | Path = DEFAULT_CACHE,
    manifest_path: str | Path = MANIFEST_PATH, max_series: int = 4,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    result = []
    for spec in _source_subset(manifest, role):
        records = load_source(spec["source_id"], cache_dir=cache_dir, manifest_path=manifest_path)
        _, details = select_series(records, spec["regimes"], spec["seasonal_period"], maximum=max_series,
            minimum_train_windows=int(manifest["protocol"]["minimum_train_windows"]))
        result.append({"source_id": spec["source_id"], "family_id": spec["family_id"], "role": spec["role"], **details})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "describe"])
    parser.add_argument("--role", choices=["development", "confirmation", "all"], default="development")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--max-eval-windows", type=int, default=64)
    arguments = parser.parse_args(argv)
    if arguments.command == "fetch":
        paths = fetch_sources(arguments.role, cache_dir=arguments.cache_dir)
        print(json.dumps({"verified_sources": [p.name for p in paths]}, indent=2))
        return 0
    tasks = load_tasks(arguments.role, cache_dir=arguments.cache_dir,
        max_train_windows=arguments.max_train_windows, max_eval_windows=arguments.max_eval_windows)
    payload = {
        "schema_version": 1,
        "purpose": "source roster and chronological engineering audit; no model outcomes",
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "roster": source_roster(arguments.role, cache_dir=arguments.cache_dir),
        "tasks": [task.to_dict() for task in tasks],
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"tasks": len(tasks), "supported": sum(t.supported for t in tasks), "output": str(arguments.output)}))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
