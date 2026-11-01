#!/usr/bin/env python3
"""Archive a completed, independently verified computational reproduction.

Only the Python standard library is required. This reporting utility does not
fit, predict, analyze, repair, or alter scientific evidence. The external
manifest is the final commit marker for a verified, complete archive set.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import traceback
import zipfile
import zlib


STUDY_NAME = "crossfit_residual_reproduction_20261101"
PREFIX = "KAHM_Crossfit_Reproduction"
RECORD_DIRS = ("prepared", "results", "predictions", "execution")
COMPACT_DIRS = ("analysis", "verification", "readiness")
METADATA = (
    "config.json", "freeze.json", "FIT_LOCK.json", "PREDICTION_LOCK.json",
    "TARGET_RELEASE.json", "ORIGINAL_SOURCE.json", "PUBLISHED_SOURCE.json",
)
AUDIT_RELATIVE = "verification/reproduction_audit.json"
SOURCE_SUFFIXES = {
    ".py", ".pyi", ".pyc", ".pyo", ".sh", ".bash", ".js", ".jsx",
    ".ts", ".tsx", ".mjs", ".cjs", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".tex", ".bib", ".md", ".rst",
}
STORED_SUFFIXES = {".npz", ".gz", ".zip", ".png", ".jpg", ".jpeg", ".webp"}
BUFFER_SIZE = 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class ArchiveError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ArchiveError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def token(path):
    info = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(info.st_mode), "Expected a regular file: " + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def safe_file(root, relative):
    require(isinstance(relative, str) and "\\" not in relative,
            "Invalid evidence path: " + repr(relative))
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and parts.parts and
            all(part not in (".", "..") for part in parts.parts),
            "Unsafe evidence path: " + relative)
    path = root.joinpath(*parts.parts)
    cursor = root
    for part in parts.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "Symlink evidence is not archived: " + str(cursor))
    require(path.is_file() and path.resolve().is_relative_to(root),
            "Missing or out-of-scope evidence file: " + str(path))
    return path


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


class Evidence:
    """Cache only metadata digests; stream large numerical files during packing."""

    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.expected = {}

    def read(self, relative):
        path = safe_file(self.root, relative)
        before = token(path)
        previous = self.cache.get(path)
        require(previous is None or previous[0] == before,
                "Evidence changed before a repeated read: " + relative)
        payload = path.read_bytes()
        require(token(path) == before, "Evidence changed while reading: " + relative)
        self.cache[path] = (before, hashlib.sha256(payload).hexdigest())
        return json.loads(payload)

    def digest(self, path):
        before = token(path)
        old = self.cache.get(path)
        if old is not None:
            require(old[0] == before, "Evidence changed during preflight: " + str(path))
            return old[1]
        digest = sha_file(path)
        require(token(path) == before, "Evidence changed while hashing: " + str(path))
        self.cache[path] = (before, digest)
        return digest

    def check(self, relative, expected):
        self.register(relative, expected)
        path = safe_file(self.root, relative)
        require(self.digest(path) == expected, "Evidence hash mismatch: " + relative)

    def register(self, relative, expected):
        require(isinstance(expected, str) and HEX64.fullmatch(expected),
                "Invalid declared SHA-256 for " + relative)
        safe_file(self.root, relative)
        existing = self.expected.setdefault(relative, expected)
        require(existing == expected, "Conflicting evidence hashes: " + relative)

    def unchanged(self):
        for path, (before, _) in self.cache.items():
            require(token(path) == before, "Preflight evidence changed: " + str(path))


def task_ids(manifest, expected, label):
    tasks = manifest["tasks"]
    ids = [task["task_id"] for task in tasks]
    require(len(ids) == 192 and len(set(ids)) == 192 and set(ids) == expected,
            label + " must contain the exact 192-task cohort")
    require(manifest.get("task_count") == 192, label + " task_count differs")
    return {task["task_id"]: task for task in tasks}


def check_stage_receipt(evidence, phase, expected):
    receipt = evidence.read("execution/" + phase + "/stage_receipt.json")
    completed = receipt.get("completed_tasks", [])
    require(receipt.get("status") == "completed" and receipt.get("phase") == phase
            and receipt.get("total_expected") == 192 and len(completed) == 192
            and len(set(completed)) == 192 and set(completed) == expected,
            "Incomplete " + phase + " execution stage")
    require(receipt.get("one_numerical_thread_per_worker") is True
            and receipt.get("workers") == 3, "Unexpected execution resource policy")


def validate_completion(root):
    """Fail before creating any archive unless all declared completion gates hold."""
    evidence = Evidence(root)
    cfg = evidence.read("config.json")
    require(cfg.get("schema_version") == "kahm-crossfit-residual-v1"
            and cfg.get("master_seed") == 2026110173 and cfg.get("repetitions") == 24
            and cfg.get("task_count") == 192 and cfg.get("histories") == [12]
            and cfg.get("noise_levels") == [0.0, 0.1]
            and cfg.get("horizons") == [1, 4, 12]
            and cfg.get("laws") == ["linear_ar", "nonlinear_ar", "vanderpol", "duffing"]
            and cfg["execution"]["output_array_count_per_task"] == 42,
            "Unexpected scientific cohort")
    reproduction = cfg["computational_reproduction"]
    require(reproduction.get("same_cohort_not_independent") is True
            and reproduction.get("original_completed_run_authenticated") is False
            and reproduction.get("additional_independent_repetitions") == 0,
            "Reproduction scope is missing or changed")
    expected = {f"{law}__h12__noise{noise}__rep{rep:02d}"
                for law in cfg["laws"] for noise in (0, 1) for rep in range(24)}
    freeze = evidence.read("freeze.json")
    require(freeze.get("status") == "frozen" and freeze.get("task_count") == 192
            and freeze.get("data_generated") is False
            and freeze.get("D_generated") is False
            and freeze.get("numerical_amendment") is None,
            "Invalid reproduction source freeze")
    evidence.check("config.json", freeze["config_sha256"])
    for relative, digest in freeze["files"].items():
        evidence.check(relative, digest)
    for field, directory in (("inherited_files", "dimension_forecasting"),
                             ("previous_extension_files", "preprocessing_dimension")):
        source_root = root.parent / directory
        for relative, expected_hash in freeze[field].items():
            path = safe_file(source_root, relative)
            require(evidence.digest(path) == expected_hash,
                    "Inherited scientific source mismatch: " + directory + "/" + relative)
    evidence.check("ORIGINAL_SOURCE.json", reproduction["original_source_manifest_sha256"])
    frozen_hash = evidence.digest(root / "freeze.json")
    config_hash = evidence.digest(root / "config.json")
    publication = evidence.read("PUBLISHED_SOURCE.json")
    published_commit = publication.get("published_source_commit")
    require(isinstance(published_commit, str) and HEX40.fullmatch(published_commit),
            "PUBLISHED_SOURCE.json needs a full published_source_commit SHA")
    require(publication.get("freeze_sha256") == frozen_hash,
            "Source publication receipt is not bound to this freeze")

    abc = evidence.read("prepared/abc_manifest.json")
    terminal = evidence.read("prepared/d_manifest.json")
    abc_tasks = task_ids(abc, expected, "ABC manifest")
    terminal_tasks = task_ids(terminal, expected, "D manifest")
    for name, manifest in (("ABC", abc), ("D", terminal)):
        require(manifest["config_sha256"] == config_hash
                and manifest["freeze_sha256"] == frozen_hash,
                name + " manifest source/configuration mismatch")
        for relative, digest in manifest["base_files"].items():
            evidence.register(relative, digest)
        for task in manifest["tasks"]:
            for file_field, hash_field in (("input", "sha256"),
                                           ("query", "query_sha256"),
                                           ("target", "target_sha256")):
                if file_field in task:
                    evidence.register(task[file_field], task[hash_field])

    seals = {}
    for name, phase in (("FIT_LOCK.json", "fits"), ("PREDICTION_LOCK.json", "predictions")):
        lock = evidence.read(name)
        require(lock.get("status") == "sealed" and lock.get("phase") == phase
                and lock.get("task_count") == 192 and len(lock["task_ids"]) == 192
                and len(set(lock["task_ids"])) == 192 and set(lock["task_ids"]) == expected
                and lock["config_sha256"] == config_hash and lock["freeze_sha256"] == frozen_hash,
                "Invalid or incomplete stage seal: " + name)
        for relative, digest in lock["files"].items():
            evidence.register(relative, digest)
        seals[name] = lock
    for directory in ("results", "predictions"):
        folder = root / directory
        require(folder.is_dir(), "Missing evidence directory: " + directory)
        actual = {p.name for p in folder.iterdir() if p.is_dir()}
        require(actual == expected, "Task-directory roster differs: " + directory)

    fit_required = {"fit_summary.json", "state.pkl.gz", "pre_c_selection.json",
                    "validation_predictions.npz", "calibration_predictions.npz",
                    "basis_diagnostics.npz", "training_arrays.npz", "numerical_solver_events.json"}
    prediction_required = {"predictions.npz", "prediction_receipt.json", "query_diagnostics.json"}
    for tid in sorted(expected):
        for directory, required, seal_name in (("results", fit_required, "FIT_LOCK.json"),
                                                ("predictions", prediction_required, "PREDICTION_LOCK.json")):
            require(not (root / directory / tid / "failure.json").exists(),
                    "Failed task cannot be archived as completed: " + tid)
            require(all(f"{directory}/{tid}/{name}" in seals[seal_name]["files"]
                        for name in required), "Missing canonical task artifacts: " + tid)
        fit_relative = f"results/{tid}/fit_summary.json"
        fit = evidence.read(fit_relative)
        evidence.check(fit_relative, seals["FIT_LOCK.json"]["files"][fit_relative])
        require(fit.get("status") == "completed" and fit["task"] == abc_tasks[tid]
                and fit["freeze_sha256"] == frozen_hash
                and fit["input_sha256"] == abc_tasks[tid]["sha256"],
                "Incomplete or mismatched fitted task: " + tid)
        for name, digest in fit["files"].items():
            evidence.register(f"results/{tid}/{name}", digest)
        pred_relative = f"predictions/{tid}/prediction_receipt.json"
        pred = evidence.read(pred_relative)
        evidence.check(pred_relative, seals["PREDICTION_LOCK.json"]["files"][pred_relative])
        require(pred.get("status") == "completed" and pred["task"] == terminal_tasks[tid]
                and pred["array_count"] == 42 and pred["targets_loaded"] is False
                and pred["state_unchanged"] is True
                and pred["state_sha256"] == seals["FIT_LOCK.json"]["files"][f"results/{tid}/state.pkl.gz"]
                and pred["query_sha256"] == terminal_tasks[tid]["query_sha256"],
                "Incomplete or mismatched target-free forecast task: " + tid)
        evidence.register(f"predictions/{tid}/predictions.npz", pred["prediction_sha256"])
        for name, digest in pred["files"].items():
            evidence.register(f"predictions/{tid}/{name}", digest)
    check_stage_receipt(evidence, "fit", expected)
    check_stage_receipt(evidence, "predict", expected)

    release = evidence.read("TARGET_RELEASE.json")
    require(release.get("status") == "released" and release.get("task_count") == 192,
            "Targets are not fully released")
    for field, relative in (("fit_lock_sha256", "FIT_LOCK.json"),
                            ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                            ("d_manifest_sha256", "prepared/d_manifest.json")):
        evidence.check(relative, release[field])
    require(terminal["fit_lock_sha256"] == evidence.digest(root / "FIT_LOCK.json"),
            "D generation is not bound to the completed fit seal")
    summary = evidence.read("analysis/analysis_summary.json")
    require(summary.get("status") == "completed" and summary.get("task_count") == 192
            and summary.get("forecast_arrays_per_task") == 42
            and summary.get("independent_repetitions") == 24
            and summary.get("primary_family_size") == 4,
            "Full frozen scientific analysis is not complete")
    for name, record in summary["tables"].items():
        evidence.check("analysis/" + name, record["sha256"])
    audit = evidence.read(AUDIT_RELATIVE)
    require(audit.get("status") == "passed" and audit.get("mode") == "computational_reproduction",
            "The complete independent computational-reproduction audit must pass")
    require(audit["counts"].get("complete_tasks_verified") == 192
            and audit["counts"].get("raw_D_forecast_arrays_scored") == 8064
            and audit["counts"].get("raw_D_horizon_losses_scored") == 24192,
            "Audit coverage is not the complete 192-task, 8064-array cohort")
    require(audit["computational_reproduction_provenance"]["original_completed_run_authenticated"] is False
            and audit["computational_reproduction_provenance"]["additional_independent_repetitions"] == 0,
            "Audit lost the same-cohort/original-run limitations")
    evidence.check("audit/verify_reproduction.py", audit["audit_source_sha256"])
    evidence.check("analysis/analysis_summary.json", audit["analysis_summary_sha256"])
    for name, digest in audit["seals"].items():
        evidence.check(name, digest)
    for name, digest in audit["source_dependencies"].items():
        evidence.check(name, digest)

    chronology = [freeze["created_utc"], abc["created_utc"],
                  seals["FIT_LOCK.json"]["created_utc"], terminal["created_utc"],
                  seals["PREDICTION_LOCK.json"]["created_utc"], release["created_utc"],
                  summary["created_utc"], audit["started_utc"], audit["completed_utc"]]
    times = [datetime.fromisoformat(value) for value in chronology]
    require(all(value.tzinfo is not None for value in times)
            and all(left <= right for left, right in zip(times, times[1:])),
            "Reproduction phase chronology is inconsistent")
    return evidence, {
        "published_source_commit": published_commit,
        "original_source_commit": reproduction["original_source_commit"],
        "publication_receipt_sha256": evidence.digest(root / "PUBLISHED_SOURCE.json"),
        "freeze_sha256": frozen_hash,
        "config_sha256": config_hash,
        "analysis_summary_sha256": evidence.digest(root / "analysis/analysis_summary.json"),
        "independent_audit_sha256": evidence.digest(root / AUDIT_RELATIVE),
        "task_count": 192, "forecast_array_count": 8064,
        "same_cohort_not_independent": True,
        "original_completed_run_authenticated": False,
        "additional_independent_repetitions": 0,
        "publication_pin_basis": "The retained PUBLISHED_SOURCE.json receipt; this offline archiver performs no remote GitHub request.",
    }


@dataclass
class Entry:
    path: Path
    relative: str
    study_relative: str
    scope: str
    stamp: tuple

    @property
    def size(self):
        return self.stamp[2]


def inventory(root, repository):
    entries, exclusions = {}, []

    def add(path, scope):
        relative = path.relative_to(repository).as_posix()
        require(relative not in entries, "Duplicate inventory path: " + relative)
        require(not path.is_symlink(), "Symlink evidence is not archived: " + relative)
        stamp = token(path)
        if path.suffix.lower() in SOURCE_SUFFIXES or "__pycache__" in path.parts:
            exclusions.append({"path": relative, "size_bytes": stamp[2],
                               "reason": "Repository source or documentation; retrieve from the pinned published checkout."})
            return
        entries[relative] = Entry(path, relative, path.relative_to(root).as_posix(), scope, stamp)

    for directory, scope in [(name, "records") for name in RECORD_DIRS] + [(name, "analysis") for name in COMPACT_DIRS]:
        folder = root / directory
        require(folder.is_dir() and not folder.is_symlink(), "Missing evidence directory: " + directory)
        for current, dirs, files in os.walk(folder, followlinks=False):
            for name in dirs:
                require(not (Path(current) / name).is_symlink(),
                        "Symlink directory cannot be silently omitted: " + str(Path(current) / name))
            for name in sorted(files):
                add(Path(current) / name, scope)
    for relative in METADATA:
        add(safe_file(root, relative), "analysis")
    return sorted(entries.values(), key=lambda entry: entry.relative), sorted(exclusions, key=lambda value: value["path"])


def group_records(entries, target_bytes):
    groups, current, count = [], [], 0
    for entry in entries:
        require(entry.size <= target_bytes,
                f"Single file exceeds the {target_bytes}-byte input limit: {entry.relative} ({entry.size} bytes). "
                "No archive has been written. This tool does not split a numerical file or create an oversized part.")
        if current and count + entry.size > target_bytes:
            groups.append(current)
            current, count = [], 0
        current.append(entry)
        count += entry.size
    if current:
        groups.append(current)
    return groups


def write_archive(destination, entries, evidence):
    records = []
    with zipfile.ZipFile(destination, mode="x", allowZip64=True) as archive:
        for entry in entries:
            require(token(entry.path) == entry.stamp, "Input changed before archiving: " + entry.relative)
            compression = zipfile.ZIP_STORED if entry.path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            info = zipfile.ZipInfo(entry.relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            digest, crc, size = hashlib.sha256(), 0, 0
            with entry.path.open("rb") as source, archive.open(info, mode="w", force_zip64=True) as target:
                require((os.fstat(source.fileno()).st_dev, os.fstat(source.fileno()).st_ino) == entry.stamp[:2],
                        "Input identity changed: " + entry.relative)
                for block in iter(lambda: source.read(BUFFER_SIZE), b""):
                    target.write(block)
                    digest.update(block)
                    crc = zlib.crc32(block, crc)
                    size += len(block)
            require(token(entry.path) == entry.stamp and size == entry.size,
                    "Input changed while archiving: " + entry.relative)
            hexdigest = digest.hexdigest()
            expected = evidence.expected.get(entry.study_relative)
            if expected is not None:
                require(hexdigest == expected, "Archived evidence violates a source/data/stage/analysis seal: " + entry.relative)
            cached = evidence.cache.get(entry.path)
            if cached is not None:
                require(hexdigest == cached[1], "Preflight metadata changed: " + entry.relative)
            records.append({"path": entry.relative, "size_bytes": size, "sha256": hexdigest,
                            "crc32": f"{crc & 0xffffffff:08x}", "archive": destination.name,
                            "scope": entry.scope,
                            "compression": "stored" if compression == zipfile.ZIP_STORED else "deflated"})
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    return records


def verify_archive(path, records):
    """Read every member once; check SHA-256, length, and both CRC values."""
    expected = {record["path"]: record for record in records}
    require(len(expected) == len(records), "Duplicate expected ZIP member")
    with zipfile.ZipFile(path, mode="r") as archive:
        infos = archive.infolist()
        require(len(infos) == len(expected) and {info.filename for info in infos} == set(expected),
                "ZIP member roster differs: " + path.name)
        for info in infos:
            record = expected[info.filename]
            digest, crc, size = hashlib.sha256(), 0, 0
            with archive.open(info, mode="r") as stream:
                for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
                    digest.update(block)
                    crc = zlib.crc32(block, crc)
                    size += len(block)
            require(size == record["size_bytes"] == info.file_size
                    and digest.hexdigest() == record["sha256"]
                    and f"{crc & 0xffffffff:08x}" == record["crc32"]
                    and (crc & 0xffffffff) == info.CRC,
                    "ZIP member content verification failed: " + info.filename)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha_file(path),
            "input_bytes": sum(record["size_bytes"] for record in records),
            "member_count": len(records), "verified_member_count": len(records),
            "verification": "Every member read once with independent SHA-256, byte-count, and CRC32 checks; ZIP reader also checks CRC."}


def build(root, output, target_mib):
    root = root.resolve()
    require(root.name == STUDY_NAME and root.parent.name == "extensions", "Unexpected study root")
    repository = root.parent.parent
    require((repository / ".git").exists(), "Run against the published source checkout")
    output = output.resolve()
    require(not output.is_relative_to(repository), "Archive output must be outside the repository")
    require(0 < target_mib <= 400, "--target-mib must be greater than zero and at most 400")
    target_bytes = int(target_mib * 1024 * 1024)
    started = utc_now()
    evidence, provenance = validate_completion(root)
    entries, exclusions = inventory(root, repository)
    records = [entry for entry in entries if entry.scope == "records"]
    compact = [entry for entry in entries if entry.scope == "analysis"]
    require(records and compact, "Empty numerical or analysis inventory")
    groups = group_records(records, target_bytes)
    require(sum(entry.size for entry in compact) <= target_bytes,
            "The compact Analysis.zip input exceeds the requested limit; no archive has been written.")
    plans = [(f"{PREFIX}_Analysis.zip", compact)]
    plans.extend((f"{PREFIX}_Records_{number:02d}.zip", group) for number, group in enumerate(groups, 1))
    manifest_name = f"{PREFIX}_Archive_Manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    require(output.is_dir(), "Output destination is not a directory")
    for name in [name for name, _ in plans] + [manifest_name]:
        require(not (output / name).exists(), "Refusing to overwrite existing artifact: " + str(output / name))
    staging = Path(tempfile.mkdtemp(prefix=PREFIX + "_STAGING_", dir=output))
    promoted = []
    try:
        all_records, archives = [], []
        for name, group in plans:
            path = staging / name
            file_records = write_archive(path, group, evidence)
            archives.append(verify_archive(path, file_records))
            all_records.extend(file_records)
            print(json.dumps({"archive_verified": name, "members": len(file_records),
                              "input_bytes": archives[-1]["input_bytes"],
                              "archive_bytes": archives[-1]["size_bytes"]}), flush=True)
        actual = [record["path"] for record in all_records]
        require(len(actual) == len(set(actual)) == len(entries)
                and set(actual) == {entry.relative for entry in entries},
                "Archive set has omissions or duplicate relative files")
        current, current_exclusions = inventory(root, repository)
        require([(entry.relative, entry.scope, entry.stamp) for entry in current]
                == [(entry.relative, entry.scope, entry.stamp) for entry in entries]
                and current_exclusions == exclusions,
                "Evidence inventory changed during archive construction")
        evidence.unchanged()
        archived_relative = {entry.study_relative for entry in entries}
        numerical_expected = {relative for relative in evidence.expected
                              if relative.split("/", 1)[0] in (*RECORD_DIRS, "analysis", "verification", "readiness")
                              and Path(relative).suffix.lower() not in SOURCE_SUFFIXES}
        require(numerical_expected <= archived_relative, "A sealed numerical file is absent from the archive inventory")
        manifest = {
            "schema": "kahm-crossfit-reproduction-archive-v1", "status": "verified",
            "started_utc": started, "completed_utc": utc_now(),
            "study_directory": root.relative_to(repository).as_posix(),
            "provenance": provenance,
            "input_bytes_per_records_archive_limit": target_bytes,
            "record_inventory_directories": list(RECORD_DIRS),
            "compact_inventory_directories": list(COMPACT_DIRS),
            "compact_metadata_files": list(METADATA),
            "archive_count": len(archives), "archived_file_count": len(all_records),
            "archived_input_bytes": sum(record["size_bytes"] for record in all_records),
            "archive_bytes": sum(record["size_bytes"] for record in archives),
            "coverage": "Exact regular-file coverage of the declared numerical and compact evidence scopes, excluding explicitly listed repository source/documentation files; no duplicate relative file paths.",
            "source_code_archived": False,
            "exclusions": exclusions,
            "extraction": "Each ZIP is standalone. Extract all archives over the pinned published repository checkout; member paths begin with extensions/crossfit_residual_reproduction_20261101/.",
            "commit_marker": "This manifest is published only after every ZIP and entry passes verification. Staging directories and failure receipts are not completed deliverables.",
            "builder_sha256": sha_file(Path(__file__).resolve()),
            "python_version": sys.version,
            "archives": archives,
            "files": sorted(all_records, key=lambda record: record["path"]),
        }
        manifest_path = staging / manifest_name
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard links are atomic, fail if the destination exists, and stay on the
        # same filesystem. The manifest is promoted last as the commit marker.
        for name in [name for name, _ in plans] + [manifest_name]:
            source, destination = staging / name, output / name
            os.link(source, destination)
            promoted.append((source, destination))
        # Once the final manifest has been committed, a temporary-directory
        # cleanup problem must not turn valid delivered bytes into a failed run.
        cleanup_warning = None
        try:
            shutil.rmtree(staging)
        except OSError as error:
            cleanup_warning = "Verified outputs committed; staging cleanup failed: " + repr(error)
            print(cleanup_warning, file=sys.stderr, flush=True)
        return {"status": "verified", "output_directory": str(output),
                "manifest": str(output / manifest_name), "archives": len(archives),
                "files": len(all_records), "input_bytes": manifest["archived_input_bytes"],
                "cleanup_warning": cleanup_warning}
    except BaseException as error:
        # Undo only links this process created and still owns. Preserve every
        # staged byte and a clearly failed receipt; never publish a pass marker.
        cleanup_errors = []
        for source, destination in reversed(promoted):
            try:
                if source.exists() and destination.exists() and os.path.samefile(source, destination):
                    destination.unlink()
            except OSError as cleanup_error:
                cleanup_errors.append({"path": str(destination), "error": repr(cleanup_error)})
        failure = {"status": "failed", "started_utc": started, "failed_utc": utc_now(),
                   "error": repr(error), "traceback": traceback.format_exc(),
                   "staging_directory": str(staging), "completed_archive_set_published": False,
                   "cleanup_errors": cleanup_errors}
        failure_path = staging / "ARCHIVE_BUILD_FAILED.json"
        if staging.exists() and not failure_path.exists():
            try:
                with failure_path.open("x", encoding="utf-8") as stream:
                    json.dump(failure, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
            except OSError as receipt_error:
                print("Could not write staged failure receipt: " + repr(receipt_error), file=sys.stderr, flush=True)
        print("Archive build failed; staged evidence retained at " + str(staging), file=sys.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination outside the Git repository; existing artifact names are never overwritten.")
    parser.add_argument("--target-mib", type=float, default=400.0,
                        help="Maximum input MiB in each standalone Records ZIP, at most 400 (default: 400).")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output_dir, args.target_mib), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
