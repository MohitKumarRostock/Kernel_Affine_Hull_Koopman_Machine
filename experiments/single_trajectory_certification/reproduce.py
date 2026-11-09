#!/usr/bin/env python3
"""Verify or reproduce the single-trajectory certification experiment.

The archived implementation is executed only after its identity is checked.
All analysis products are written to a separate, explicitly supplied directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "SOURCE_MANIFEST.json"
RECEIPT_SCHEMA = "single_trajectory_reproduction_v1"
RECOMPUTED = (
    "summary.json", "saved_audit.json", "posthoc_graph_diagnostic.json",
    "bottleneck_diagnostics.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or "\\" in name
            or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts)
            or str(path) != name):
        raise ValueError(f"invalid manifest path: {name!r}")
    return Path(*path.parts)


def verify_file(base, record):
    path = base / relative_path(record["path"])
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"missing or unsafe file: {record['path']}")
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"file identity mismatch: {record['path']}")
    return path


def verify_distribution():
    manifest = json.loads((ROOT / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest["schema"] != "single_trajectory_reproduction_archive_v1":
        raise ValueError("unsupported source manifest")
    archive = verify_file(ROOT, manifest["archive"])
    records = manifest["members"]
    members = {record["path"]: record for record in records}
    if len(members) != len(records):
        raise ValueError("duplicate archive manifest entry")
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if len(infos) != len(members) or {info.filename for info in infos} != set(members):
            raise ValueError("archive members do not match the manifest")
        for info in infos:
            relative_path(info.filename)
            if (info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)
                    or info.file_size != members[info.filename]["size_bytes"]):
                raise ValueError(f"unsafe archive entry: {info.filename}")
            if hashlib.sha256(zipped.read(info)).hexdigest() != members[info.filename]["sha256"]:
                raise ValueError(f"archive member identity mismatch: {info.filename}")
        protocol = json.loads(zipped.read("protocol.json"))
        analysis = json.loads(zipped.read("analysis_manifest.json"))
        protocol_hash = members["protocol.json"]["sha256"]
        if (protocol_hash != manifest["protocol_sha256"]
                or analysis["protocol_sha256"] != protocol_hash
                or zipped.read("protocol.sha256").decode().split()[0] != protocol_hash):
            raise ValueError("protocol identities disagree")
        for name, expected in protocol["engine_sha256"].items():
            if members[name]["sha256"] != expected:
                raise ValueError(f"protocol source identity mismatch: {name}")
        if analysis["analysis_sources"] != manifest["analysis_source_sha256"]:
            raise ValueError("analysis source identities disagree")
        for group in (manifest["scientific_source_sha256"], manifest["analysis_source_sha256"]):
            for name, expected in group.items():
                if members[name]["sha256"] != expected:
                    raise ValueError(f"source identity mismatch: {name}")
    paths = set()
    for record in manifest["retained_files"] + manifest["figures"]:
        if record["path"] in paths:
            raise ValueError("duplicate retained-file entry")
        paths.add(record["path"])
        path = verify_file(ROOT, record)
        if "logical_path" in record:
            relative_path(record["logical_path"])
            digest, size = hashlib.sha256(), 0
            with gzip.open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != record["logical_sha256"] or size != record["logical_size_bytes"]:
                raise ValueError("decompressed outcome records have changed")
    run_manifest = json.loads((ROOT / "results/run_manifest.json").read_text())
    if (run_manifest["protocol_sha256"] != manifest["protocol_sha256"]
            or run_manifest["source_sha256"] != manifest["scientific_source_sha256"]):
        raise ValueError("retained run and archived source identities disagree")
    return manifest


def extract_sources(manifest, destination):
    with zipfile.ZipFile(ROOT / manifest["archive"]["path"]) as zipped:
        for record in manifest["members"]:
            target = destination / relative_path(record["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            data = zipped.read(record["path"])
            if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("archive changed after verification")
            with target.open("xb") as handle:
                handle.write(data)


def copy_evidence(manifest, output):
    for record in manifest["retained_files"]:
        source = ROOT / relative_path(record["path"])
        target = output / relative_path(record.get("logical_path", record["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if "logical_path" in record:
            with gzip.open(source, "rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            expected = record["logical_sha256"]
        else:
            shutil.copyfile(source, target)
            expected = record["sha256"]
        if sha256(target) != expected:
            raise ValueError(f"copied evidence identity mismatch: {target.name}")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_receipt(output, receipt):
    temporary = output / "reproduction.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "reproduction.json")


def run_command(label, arguments, source, log_directory, attempt, output, receipt):
    command = [sys.executable, "-B", *map(str, arguments)]
    log_path = log_directory / f"{len(attempt['commands']) + 1:02d}_{label}.log"
    record = {"name": label, "arguments": list(map(str, arguments)),
              "started_utc": utc_now(), "log": str(log_path.relative_to(output))}
    attempt["commands"].append(record)
    write_receipt(output, receipt)
    print(f"Running {label}; log: {log_path}", flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=source, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   start_new_session=(os.name == "posix"))
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
    record.update({"finished_utc": utc_now(), "returncode": returncode,
                   "log_sha256": sha256(log_path)})
    write_receipt(output, receipt)
    if returncode:
        raise RuntimeError(f"{label} failed with exit status {returncode}; see {log_path}")
    return log_path


def prepare_output(args, manifest):
    supplied = Path(args.output)
    if not supplied.is_absolute():
        raise ValueError("--output must be an absolute path")
    output = supplied.resolve()
    if output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise ValueError("--output must be separate from the supplied experiment directory")
    if args.resume:
        if args.mode != "run":
            raise ValueError("--resume is available only with --mode run")
        receipt_path = output / "reproduction.json"
        if not receipt_path.is_file():
            raise ValueError("resumption requires an output created by this wrapper")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt["schema"] != RECEIPT_SCHEMA or receipt["mode"] != "run"
                or receipt["source_archive_sha256"] != manifest["archive"]["sha256"]
                or receipt["protocol_sha256"] != manifest["protocol_sha256"]):
            raise ValueError("resumption identity mismatch")
        if any(path.is_symlink() for path in output.rglob("*")):
            raise ValueError("resumption output must not contain symbolic links")
    else:
        output.mkdir(parents=True, exist_ok=False)
        receipt = {"schema": RECEIPT_SCHEMA, "mode": args.mode,
                   "source_archive_sha256": manifest["archive"]["sha256"],
                   "source_manifest_sha256": sha256(ROOT / MANIFEST_NAME),
                   "protocol_sha256": manifest["protocol_sha256"], "attempts": []}
    return output, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), required=True,
                        help="audit retained evidence or regenerate the registered experiment")
    parser.add_argument("--output", required=True,
                        help="absolute path to a new directory outside the supplied experiment")
    parser.add_argument("--workers", type=int, default=6,
                        help="processes for experiment regeneration (default: 6)")
    parser.add_argument("--resume", action="store_true",
                        help="continue a run in an existing compatible output directory")
    parser.add_argument("--figures", action="store_true",
                        help="also regenerate scientific figures; requires matplotlib")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        manifest = verify_distribution()
        output, receipt = prepare_output(args, manifest)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Reproduction could not start: {error}\n")
    attempt = {"started_utc": utc_now(), "python_version": sys.version,
               "wrapper_sha256": sha256(Path(__file__)), "resume": args.resume,
               "figures_requested": args.figures, "workers": args.workers,
               "status": "running", "commands": []}
    receipt["attempts"].append(attempt)
    logs = output / "logs" / f"attempt_{len(receipt['attempts']):03d}"
    logs.mkdir(parents=True, exist_ok=False)
    write_receipt(output, receipt)
    try:
        if args.mode == "check":
            copy_evidence(manifest, output)
        with tempfile.TemporaryDirectory(prefix="single_trajectory_sources_") as temporary:
            source = Path(temporary)
            extract_sources(manifest, source)
            provenance = output / "provenance"
            provenance.mkdir(exist_ok=True)
            for name in ("protocol.json", "protocol.sha256", "analysis_manifest.json"):
                shutil.copyfile(source / name, provenance / name)
            invoke = lambda label, command: run_command(
                label, command, source, logs, attempt, output, receipt)
            test_log = invoke("unit_tests", ["-m", "unittest", "discover", "-q"])
            match = re.search(r"Ran (\d+) tests? in", test_log.read_text(encoding="utf-8"))
            if match is None or int(match.group(1)) != manifest["expected_test_count"]:
                raise RuntimeError("the archived test suite did not execute its expected test count")
            attempt["tests_passed"] = int(match.group(1))
            results = output / "results"
            protocol = source / "protocol.json"
            if args.mode == "run":
                command = ["run_experiment.py", "--protocol", protocol,
                           "--output", results, "--workers", str(args.workers)]
                if args.resume:
                    command.append("--resume")
                invoke("experiment", command)
            invoke("saved_certificate_audit", ["audit_saved.py", "--protocol", protocol, "--results", results])
            invoke("summary", ["summarize_experiment.py", "--protocol", protocol, "--results", results])
            invoke("posthoc_graph_diagnostic", ["diagnose_saved_graph.py", "--protocol", protocol,
                   "--results", results, "--out", results / "posthoc_graph_diagnostic.json"])
            invoke("posthoc_bottleneck_description", ["describe_bottlenecks.py",
                   "--runs", results / "runs.jsonl", "--out", results / "bottleneck_diagnostics.json"])
            if args.mode == "check":
                expected = {item["path"]: item for item in manifest["retained_files"]}
                for name in RECOMPUTED:
                    if sha256(results / name) != expected["results/" + name]["sha256"]:
                        raise RuntimeError(f"recomputed evidence differs from the retained file: {name}")
                attempt["recomputed_files_match"] = list(RECOMPUTED)
            if args.figures:
                invoke("figures", ["plot_results.py", "--summary", results / "summary.json",
                                    "--out", output / "figures"])
        verify_distribution()
        audit = json.loads((output / "results/saved_audit.json").read_text())
        attempt.update({"status": "passed", "retained_runs_checked": audit["retained_runs_checked"],
                        "full_certificates_reconstructed": audit["full_certificates_reconstructed"],
                        "supplied_evidence_unchanged": True,
                        "output_files_sha256": {
                            str(path.relative_to(output)): sha256(path)
                            for folder in ("results", "figures")
                            for path in sorted((output / folder).rglob("*")) if path.is_file()}})
    except BaseException as error:
        attempt.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        attempt["finished_utc"] = utc_now()
        write_receipt(output, receipt)
    print(f"Reproduction passed. Receipt: {output / 'reproduction.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
