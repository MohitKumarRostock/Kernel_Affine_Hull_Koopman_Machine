#!/usr/bin/env python3
"""Package immutable scaling records and a compact, auditable primary-score bundle.

Usage:
  python package_scaling_evidence.py --source-study STUDY --baseline-repo BASELINE \
      --worktree DESTINATION_REPOSITORY --deliverable OUTPUT.zip

Reads completed experiment evidence; never fits or selects a model using test
scores. Existing frozen scientific scripts are copied byte-for-byte.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np

MODES = ("standard", "minmax", "minmax_clip", "raw")
TEXT_SUFFIXES = {".json", ".csv", ".tex", ".txt", ".log"}
SCRIPTS = (
    "scaling_features.py", "run_scaling.py", "run_confirmation.py",
    "run_pendigits_scaling.py", "run_pendigits_confirmation.py",
    "analyze_comparison.py", "audit_pendigits_scaling.py",
    "audit_pendigits_confirmation.py", "repair_archive.py",
    "repair_confirmation_archives.py", "test_scaling_features.py",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(json_bytes(value))
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_bytes(rows, fieldnames=None):
    buffer = io.StringIO(newline="")
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def source_snapshot(source, baseline, replay_root):
    files = [(source / name, Path(name)) for name in SCRIPTS]
    files += [(source / name, Path(name)) for name in
              ("protocol.json", "confirmation_protocol.json", "pendigits_confirmation_protocol.json")]
    files += [(p, Path("confirmation_configs") / p.name)
              for p in sorted((source / "confirmation_configs").glob("*.json"))]
    campaign_names = list(read_json(source / "protocol.json")["configurations"])
    files += [(baseline / "extensions/certified_forecasting/configs" / f"{name}.json",
               Path("configs") / f"{name}.json") for name in campaign_names]
    files += [(p, Path("verification") / p.name)
              for p in sorted((source / "verification").glob("*.py"))]
    index = []
    for origin, relative in files:
        if not origin.is_file():
            raise FileNotFoundError(origin)
        target = replay_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        digest = sha256(origin)
        if sha256(target) != digest:
            raise RuntimeError(f"Scientific snapshot mismatch:{relative}")
        index.append({"path": relative.as_posix(), "sha256": digest,
                      "size_bytes": origin.stat().st_size})
    write_json(replay_root / "source_snapshot_manifest.json", {
        "scope": "Exact original investigation scripts, protocols, and configurations; scientific source bytes unchanged.",
        "baseline_commit": read_json(source / "protocol.json")["parent_commit"],
        "files": index,
    })
    return index


def package_records(source, phase, phase_root, repair_root, target):
    files = [(p, p.relative_to(phase_root).as_posix())
             for p in sorted(phase_root.rglob("*"))
             if p.is_file() and p.suffix in TEXT_SUFFIXES]
    if repair_root.exists():
        files += [(p, "repairs/" + p.relative_to(repair_root).as_posix())
                  for p in sorted(repair_root.rglob("*"))
                  if p.is_file() and p.suffix in TEXT_SUFFIXES]
    for p in sorted((source / "logs").glob("pendigits*.log")):
        is_confirmation = "confirmation" in p.name
        if is_confirmation == (phase == "confirmation"):
            files.append((p, "wrapper_logs/" + p.name))
    if len({relative for _, relative in files}) != len(files):
        raise RuntimeError("Duplicate record archive path")
    contents = [{"path": relative, "sha256": sha256(path),
                 "size_bytes": path.stat().st_size} for path, relative in files]
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for path, relative in files:
                    info = tarfile.TarInfo(relative)
                    info.size = path.stat().st_size
                    info.mode, info.mtime = 0o644, 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                payload = json_bytes({"phase": phase, "excluded": "All NPZ archives and corrupt binary artifacts", "files": contents})
                info = tarfile.TarInfo("archive_contents.json")
                info.size, info.mode, info.mtime = len(payload), 0o644, 0
                archive.addfile(info, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, target)
    expected = {row["path"]: row["sha256"] for row in contents}
    with tarfile.open(target, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name.endswith(".npz"):
                raise RuntimeError(f"Unexpected binary or non-file member:{member.name}")
            data = archive.extractfile(member).read()
            if member.name != "archive_contents.json" and digest_bytes(data) != expected[member.name]:
                raise RuntimeError(f"Record archive byte mismatch:{member.name}")
    return {"path": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size,
            "records": len(contents), "all_member_hashes_verified": True}


def zwrite(archive, name, data, *, stored=False):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def selected_candidate(records):
    family = [r for r in records if r["family"] == "kahm_nested"]
    return min(family, key=lambda r: (float(np.mean(r["validation_mse"])), r["n_features"], r["name"]))


def csv_selection_map(rows):
    result = {}
    for row in rows:
        if row["family"] != "kahm_nested":
            continue
        key = (row["campaign"], row["system"], row["mode"], int(row["seed"]),
               row["coordinate"], int(row["horizon"]))
        if key in result:
            raise RuntimeError(f"Duplicate canonical selection:{key}")
        result[key] = row
    return result


def package_primary(source, phases, deliverable, full_archive_index):
    synthetic_index, pen_index, arrays_expected = [], [], {}
    total_pen_rows, selected_csv_checks = 0, 0
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    temporary = deliverable.with_suffix(deliverable.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for phase, roots in phases.items():
            phase_root, analysis_root = roots["results"], roots["analysis"]
            canonical_rows = read_csv(analysis_root / "selected_candidates.csv")
            selected_map = csv_selection_map(canonical_rows)
            primary_rows = [row for row in canonical_rows if row["family"] == "kahm_nested"]
            zwrite(archive, f"canonical_selections/{phase}.csv", csv_bytes(primary_rows))
            for path in sorted(phase_root.glob("*/*/*/episode_scores.npz")):
                relative = path.relative_to(phase_root)
                mode, campaign, run_name, _ = relative.parts
                result = read_json(path.parent / "results.json")
                selected = selected_candidate(result["results"])
                manifest = read_json(path.parent / "intervention_manifest.json")
                cfg = manifest["configuration"]
                fields = ("cal_loss", "pair_score", "test_loss")
                with np.load(path, allow_pickle=False) as original:
                    arrays = {field: original[selected["name"] + "__" + field].copy() for field in fields}
                for field, array in arrays.items():
                    expected_n = cfg["n_calibration"] if field == "cal_loss" else (
                        cfg["n_calibration"] // 2 if field == "pair_score" else cfg["n_test"])
                    if (array.dtype != np.dtype("float64") or array.shape != (expected_n, len(cfg["horizons"]))
                            or not np.isfinite(array).all()):
                        raise RuntimeError(f"Invalid selected score array:{relative}:{field}")
                np.testing.assert_array_equal(arrays["test_loss"].mean(axis=0), selected["test_mse"])
                for i, horizon in enumerate(cfg["horizons"]):
                    key = (campaign, result["system"], mode, result["seed"], "scalar", horizon)
                    row = selected_map[key]
                    if row["candidate"] != selected["name"]:
                        raise RuntimeError(f"Canonical CSV selection mismatch:{key}")
                    np.testing.assert_array_equal(float(row["validation_mse"]), selected["validation_mse"][i])
                    np.testing.assert_array_equal(float(row["test_mse"]), selected["test_mse"][i])
                    selected_csv_checks += 1
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **arrays)
                packed = buffer.getvalue()
                npz_name = f"scores/{phase}/{mode}/{campaign}/{run_name}.npz"
                zwrite(archive, npz_name, packed, stored=True)
                array_metadata = {field: {"dtype": str(a.dtype), "shape": list(a.shape),
                                         "raw_float64_sha256": digest_bytes(np.ascontiguousarray(a).tobytes())}
                                  for field, a in arrays.items()}
                arrays_expected[npz_name] = array_metadata
                synthetic_index.append({
                    "phase": phase, "mode": mode, "campaign": campaign,
                    "system": result["system"], "seed": result["seed"],
                    "primary_npz": npz_name, "primary_npz_sha256": digest_bytes(packed),
                    "full_archive_relative_path": relative.as_posix(),
                    "full_archive_sha256": full_archive_index[(phase, relative.as_posix())]["sha256"],
                    "result_json_sha256": sha256(path.parent / "results.json"),
                    "frozen_candidates_sha256": sha256(path.parent / "frozen_candidates.json"),
                    "intervention_manifest_sha256": sha256(path.parent / "intervention_manifest.json"),
                    "selected_candidate": selected,
                    "validation_candidate_family": [{k: r[k] for k in ("name", "history", "n_features", "validation_mse")}
                                                    for r in result["results"] if r["family"] == "kahm_nested"],
                    "configuration": cfg, "streams": result["streams"],
                    "within_arm_confidence_scope": result["confidence_scope"],
                    "delta": result["delta"], "per_risk_tail_delta": result["tail_delta"],
                    "arrays": array_metadata,
                })
            pen_rows = []
            for mode in MODES:
                pen_dir = phase_root / mode / "pendigits_v1"
                result = read_json(pen_dir / "results.json")
                writer_rows = read_csv(pen_dir / "writer_losses.csv")
                cfg = result["configuration"]
                for seed in cfg["seeds"]:
                    for coordinate in cfg["coordinates"]:
                        candidates = [r for r in result["results"] if r["seed"] == seed and r["coordinate"] == coordinate]
                        chosen = selected_candidate(candidates)
                        if not chosen["selected_before_test"]:
                            raise RuntimeError("PenDigits selected candidate was not frozen before test")
                        selected_writer_rows = [r for r in writer_rows if int(r["seed"]) == seed and
                                                int(r["coordinate"]) == coordinate and r["candidate"] == chosen["name"]]
                        if len(selected_writer_rows) != 13 * len(cfg["horizons"]):
                            raise RuntimeError("Incomplete primary PenDigits writer rows")
                        for i, horizon in enumerate(cfg["horizons"]):
                            group = [r for r in selected_writer_rows if int(r["horizon"]) == horizon]
                            writer_mean = float(np.mean([float(r["writer_mean_squared_loss"]) for r in group]))
                            np.testing.assert_allclose(writer_mean, chosen["cohort_writer_mean_mse"][i], rtol=0, atol=2e-16)
                            key = ("pendigits_v1", "pendigits", mode, seed, str(coordinate), horizon)
                            canonical = selected_map[key]
                            if canonical["candidate"] != chosen["name"]:
                                raise RuntimeError(f"Canonical PenDigits selection mismatch:{key}")
                            np.testing.assert_array_equal(float(canonical["validation_mse"]), chosen["validation_mse"][i])
                            np.testing.assert_array_equal(float(canonical["test_mse"]), chosen["cohort_writer_mean_mse"][i])
                            selected_csv_checks += 1
                        pen_rows.extend([{"phase": phase, "mode": mode, **r} for r in selected_writer_rows])
                        pen_index.append({
                            "phase": phase, "mode": mode, "seed": seed, "coordinate": coordinate,
                            "selected_candidate": chosen,
                            "validation_candidate_family": [{k: r[k] for k in ("name", "history", "n_features", "validation_mse")}
                                                            for r in candidates if r["family"] == "kahm_nested"],
                            "writer_csv": f"pendigits/{phase}_selected_writer_losses.csv",
                            "source_writer_losses_sha256": sha256(pen_dir / "writer_losses.csv"),
                            "source_result_json_sha256": sha256(pen_dir / "results.json"),
                            "writer_rows": len(selected_writer_rows), "configuration": cfg,
                        })
            zwrite(archive, f"pendigits/{phase}_selected_writer_losses.csv", csv_bytes(pen_rows))
            total_pen_rows += len(pen_rows)
        index = {
            "scope": "PRIMARY SCORES ONLY: validation-selected kahm_nested forecasts. Other complete candidate score archives are excluded. Exact original truncated bytes are separately retained under forensics/ and are not used as scientific scores.",
            "selection_rule": "Within each arm/task/seed (and PenDigits coordinate), minimum mean validation MSE over horizons; ties use fewer features then candidate name. Test scores never choose candidates.",
            "primary_comparison": "minmax versus standard; minmax_clip and raw are fixed diagnostic arms, not test-selected replacements.",
            "regeneration": "All-candidate NPZ archives can be regenerated from the exact frozen scripts/configurations and baseline source commit. Full archive hashes, sizes, and repair records are retained in episode_archive_index.json.",
            "statistical_scope": "Synthetic arrays use independent episode averages and disjoint episode pairs. PenDigits rows are a fixed writer cohort; fresh fitting seeds do not create a fresh writer cohort. Confidence budgets are within each arm, not joint across all preprocessing arms.",
            "forensic_scope": "forensics/ contains four intentionally incomplete original inner NPZ archives and their repair records. Their invalid inner ZIP structures are preserved deliberately. The outer evidence ZIP and all score NPZs under scores/ pass integrity validation.",
            "synthetic_conditions": len(synthetic_index), "synthetic_array_count": 3 * len(synthetic_index),
            "pen_selected_coordinate_models": len(pen_index), "pen_selected_writer_rows": total_pen_rows,
            "canonical_csv_horizon_selection_checks": selected_csv_checks,
            "synthetic": synthetic_index, "pendigits": pen_index,
        }
        zwrite(archive, "selection_and_provenance_index.json", json_bytes(index))
        zwrite(archive, "episode_archive_index.json", json_bytes({"archives": list(full_archive_index.values())}))
    os.replace(temporary, deliverable)
    with zipfile.ZipFile(deliverable) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Primary evidence ZIP failed CRC validation")
        for name, metadata in arrays_expected.items():
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as saved:
                if set(saved.files) != set(metadata):
                    raise RuntimeError(f"Primary NPZ keys differ:{name}")
                for key, spec in metadata.items():
                    data = saved[key]
                    if (str(data.dtype) != spec["dtype"] or list(data.shape) != spec["shape"] or
                            digest_bytes(np.ascontiguousarray(data).tobytes()) != spec["raw_float64_sha256"]):
                        raise RuntimeError(f"Primary NPZ round-trip mismatch:{name}:{key}")
        for phase in phases:
            rows = list(csv.DictReader(io.StringIO(archive.read(f"pendigits/{phase}_selected_writer_losses.csv").decode())))
            if len(rows) != 936:
                raise RuntimeError(f"Unexpected selected PenDigits writer row count:{phase}:{len(rows)}")
    if len(synthetic_index) != 320 or total_pen_rows != 1872:
        raise RuntimeError("Primary evidence condition count differs from the frozen complete schedule")
    return {"path": str(deliverable), "sha256": sha256(deliverable),
            "size_bytes": deliverable.stat().st_size, "synthetic_conditions": len(synthetic_index),
            "primary_float64_arrays": len(arrays_expected) * 3,
            "selected_pen_writer_rows": total_pen_rows,
            "canonical_csv_horizon_selection_checks": selected_csv_checks,
            "all_zip_crc_and_npz_roundtrip_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--deliverable", type=Path, required=True)
    args = parser.parse_args()
    source, baseline, worktree = (p.resolve() for p in (args.source_study, args.baseline_repo, args.worktree))
    ext = worktree / "extensions/certified_forecasting"
    study = ext / "results/input_scaling_study"
    study.mkdir(parents=True, exist_ok=True)
    replay = ext / "experiments/input_scaling_replay"
    phases = {
        "original": {"results": source / "results", "analysis": source / "analysis", "repairs": source / "repairs"},
        "confirmation": {"results": source / "confirmation_results", "analysis": source / "confirmation_analysis", "repairs": source / "confirmation_repairs"},
    }
    print("Copying exact scientific replay sources", flush=True)
    source_files = source_snapshot(source, baseline, replay)
    verification = study / "verification"
    verification.mkdir(exist_ok=True)
    for path in sorted((source / "verification").iterdir()):
        if path.is_file() and path.suffix in {".json", ".txt", ".log"}:
            shutil.copy2(path, verification / path.name)
    record_archives, archive_entries, repair_records = [], [], []
    for phase, roots in phases.items():
        print(f"Packing scalar records and indexing full score archives: {phase}", flush=True)
        record_archives.append(package_records(source, phase, roots["results"], roots["repairs"],
                                               study / f"records_{phase}.tar.gz"))
        for path in sorted(roots["results"].glob("*/*/*/episode_scores.npz")):
            if not zipfile.is_zipfile(path):
                raise RuntimeError(f"Incomplete full score archive:{path}")
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError(f"Full score archive CRC failure:{path}")
            relative = path.relative_to(roots["results"]).as_posix()
            entry = {"phase": phase, "relative_path": relative, "sha256": sha256(path),
                     "size_bytes": path.stat().st_size, "zip_crc_valid": True}
            repair = path.parent / "artifact_repair.json"
            if repair.exists():
                repair_entry = {"phase": phase, "relative_path": repair.relative_to(roots["results"]).as_posix(),
                                "sha256": sha256(repair), "record": read_json(repair)}
                repair_records.append(repair_entry)
                entry["artifact_repair_relative_path"] = repair_entry["relative_path"]
            archive_entries.append(entry)
    if len(archive_entries) != 320:
        raise RuntimeError(f"Expected320 full synthetic archives, found{len(archive_entries)}")
    archive_index = {"scope": "Full all-candidate synthetic episode archives are indexed, not committed; exact frozen replay regenerates them.",
                     "archive_count": len(archive_entries), "total_size_bytes": sum(r["size_bytes"] for r in archive_entries),
                     "archives": archive_entries, "artifact_repairs": repair_records}
    write_json(study / "episode_archive_index.json", archive_index)
    print("Building compact primary-score evidence and validating canonical selections", flush=True)
    primary = package_primary(source, phases, args.deliverable.resolve(),
                              {(r["phase"], r["relative_path"]): r for r in archive_entries})
    # Retain exact corrupted bytes as explicitly non-scientific forensic entries.
    forensics = []
    with zipfile.ZipFile(args.deliverable, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        zwrite(archive, "artifact_repair_records.json", json_bytes(repair_records))
        for entry in repair_records:
            phase, relative = entry["phase"], Path(entry["relative_path"])
            origin = phases[phase]["results"] / relative
            corrupt = origin.parent / "episode_scores.corrupt.npz"
            payload = corrupt.read_bytes()
            record = entry["record"]
            if digest_bytes(payload) != record["corrupt_sha256"] or len(payload) != record["corrupt_bytes"]:
                raise RuntimeError(f"Preserved forensic bytes differ from repair record:{corrupt}")
            if zipfile.is_zipfile(corrupt):
                raise RuntimeError(f"Expected deliberately incomplete forensic inner archive:{corrupt}")
            archive_base = "forensics/" + phase + "/" + relative.parent.as_posix()
            forensic_name = archive_base + "/episode_scores.corrupt.npz"
            zwrite(archive, forensic_name, payload, stored=True)
            zwrite(archive, archive_base + "/artifact_repair.json", origin.read_bytes())
            forensics.append({"phase": phase, "path": forensic_name,
                              "sha256": digest_bytes(payload), "size_bytes": len(payload),
                              "deliberately_invalid_inner_npz": True,
                              "used_in_scientific_scores": False,
                              "repair_record": archive_base + "/artifact_repair.json"})
        zwrite(archive, "forensics/index.json", json_bytes({"artifacts": forensics}))
        zwrite(archive, "forensics/README.txt", (
            "FORENSIC BYTES: INTENTIONALLY INCOMPLETE INNER NPZ ARCHIVES\n\n"
            "These four files preserve the exact original truncated score archives found during independent auditing. "
            "They are retained only to substantiate the artifact-repair history and are not inputs to any reported analysis. "
            "Each adjacent artifact_repair.json records original and repaired hashes and the exact unchanged numerical replay checks.\n\n"
            "The outer KAHM_scaling_primary_evidence.zip passes ZIP CRC validation. All 320 scientific NPZ files under scores/ "
            "are complete and their 960 arrays round-trip exactly as float64. The incomplete NPZ structures under forensics/ "
            "are deliberately preserved and should not be interpreted as usable scientific score archives.\n\n"
            "Complete all-candidate score archives are regenerable from the frozen Git replay scripts/configurations; "
            "their hashes and sizes are retained in episode_archive_index.json.\n"
        ).encode())
    with zipfile.ZipFile(args.deliverable) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Final evidence archive failed CRC validation")
        for entry in forensics:
            if digest_bytes(archive.read(entry["path"])) != entry["sha256"]:
                raise RuntimeError("Forensic bytes changed in the outer evidence ZIP")
    primary.update(sha256=sha256(args.deliverable), size_bytes=args.deliverable.stat().st_size)
    primary.update(forensic_artifact_count=len(forensics),
                   preserved_corrupt_bytes=sum(row["size_bytes"] for row in forensics),
                   forensic_bytes_match_original_repair_hashes=True)
    portable = ext / "scripts/package_scaling_evidence.py"
    portable.parent.mkdir(parents=True, exist_ok=True)
    if Path(__file__).resolve() != portable.resolve():
        shutil.copy2(__file__, portable)
    report = {"record_archives": record_archives, "full_episode_archive_count": len(archive_entries),
              "full_episode_archive_total_size_bytes": archive_index["total_size_bytes"],
              "artifact_repair_count": len(repair_records), "exact_replay_source_files": len(source_files),
              "primary_evidence": primary, "packaging_script_sha256": sha256(portable)}
    write_json(verification / "evidence_packaging_validation.json", report)
    write_json(source / "verification/evidence_packaging_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
