"""Failure-preserving runner for the committed final four-state certificate campaign.

Only the exact final-v1 configuration is accepted. There are no parameter,
seed, environment-check, or dirty-source overrides. Outputs must be outside
this working tree and in a new directory; archive them separately afterward.
Reexecuting this final configuration reproduces replicates, not new independent evidence.

The complete schedule is saved before checks or sampling. Samples are flushed
and fsynced before evaluation. Ordinary per-replicate calculation failures are
logged and execution continues without retry. Persistence failures abort.
A zero bound or coverage violation is a valid result, not an execution failure.

A hard kill or storage failure can prevent finalization. Missing terminal
events, a summary, or a complete checksum inventory mean incomplete evidence.
SHA256SUMS covers retained files except itself; it is not scientific validation.
Importing this module does not run the campaign. Tests precede production use.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from kahkm_certificates import CERTIFICATE_ID
from kahkm_four_state_counts import COUNT_STATISTICS_ID
from kahkm_four_state_evaluation import evaluate_four_state_counts
from kahkm_four_state_reference import FOUR_STATE_REFERENCE_ID, four_state_reference
from kahkm_four_state_sampling import sample_four_state_counts

ROOT = Path(__file__).resolve().parent
RUNNER_ID = "four_state_final_runner_v1"
CONFIG_PATH = "reproduction/certificates/configs/four_state_final_v1.json"
SUPPORTED_CONFIG_SHA256 = "630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182"
MODULES = (
    "kahkm_certificates", "kahkm_certificate_statistics",
    "kahkm_four_state_reference", "kahkm_four_state_counts",
    "kahkm_four_state_sampling", "kahkm_four_state_evaluation",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_line(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def append_record(handle, value) -> None:
    handle.write(json_line(value))
    handle.flush()
    os.fsync(handle.fileno())


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        append_record(handle, value)


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stdin=subprocess.DEVNULL)


def source_snapshot() -> dict:
    """Check actual required-file bytes against HEAD, not only Git status."""
    if Path(git_bytes("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT:
        raise RuntimeError("Runner must be at the repository root.")
    commit = git_bytes("rev-parse", "HEAD").decode().strip()
    status = git_bytes("status", "--porcelain=v1", "--untracked-files=all").decode()
    if status:
        raise RuntimeError("A clean source working tree is required:\n" + status)
    required = [
        Path(__file__).name, CONFIG_PATH, "requirements-lock-arm64.txt",
        "scripts/verify_environment.py", "tests/test_kahkm_four_state_final_campaign.py",
        *[name + ".py" for name in MODULES],
        *["tests/test_" + name + ".py" for name in MODULES],
    ]
    hashes = {}
    for name in sorted(set(required)):
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Missing or symlinked source file: {name}")
        actual = file_sha256(path)
        committed = hashlib.sha256(git_bytes("show", f"{commit}:{name}")).hexdigest()
        if actual != committed:
            raise RuntimeError(f"Source bytes differ from HEAD: {name}")
        hashes[name] = actual
    return {"execution_source_commit": commit, "git_status_porcelain": status,
            "source_file_sha256": hashes}


def load_final() -> tuple[bytes, dict]:
    raw = (ROOT / CONFIG_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUPPORTED_CONFIG_SHA256:
        raise ValueError("Configuration differs from the supported final-v1 bytes.")
    config = json.loads(raw)

    if config.get("campaign_role") != "final":
        raise ValueError("Expected campaign_role='final'.")

    if config.get("campaign_id") != "four_state_original_iid_final_v1":
        raise ValueError("Unexpected final campaign identifier.")

    for key, expected in (("certificate_id", CERTIFICATE_ID),
                          ("reference_id", FOUR_STATE_REFERENCE_ID),
                          ("count_statistics_id", COUNT_STATISTICS_ID)):
        if config[key] != expected:
            raise ValueError(f"Implementation/configuration mismatch: {key}")

    expected_cases = 13
    expected_sizes = 9
    expected_replicates_per_cell = 1000
    expected_total = 117000

    if len(config["cases"]) != expected_cases:
        raise ValueError("Unexpected number of final cases.")

    if len(config["sampling"]["sample_sizes"]) != expected_sizes:
        raise ValueError("Unexpected number of final sample sizes.")

    if config["sampling"]["replicates_per_cell"] != expected_replicates_per_cell:
        raise ValueError("Unexpected final replicate count per cell.")

    planned = (
        len(config["cases"])
        * len(config["sampling"]["sample_sizes"])
        * config["sampling"]["replicates_per_cell"]
    )

    if planned != expected_total:
        raise ValueError("Unexpected total number of final replicates.")

    return raw, config


def build_schedule(config: dict) -> tuple[list[dict], dict]:
    references, jobs = {}, []
    rng, sampling = config["rng"], config["sampling"]
    for case in config["cases"]:
        references[case["case_id"]] = four_state_reference(
            p=case["p"], dynamics=case["dynamics"], kappa=case["kappa"],
        )
        for n_pairs in sampling["sample_sizes"]:
            for replicate in range(sampling["replicates_per_cell"]):
                seed = dict(campaign_key=rng["campaign_key"], case_key=case["case_key"],
                            n_pairs=n_pairs, replicate_index=replicate,
                            root_seed=rng["root_seed"])
                jobs.append({
                    "attempt_id": f"c{case['case_key']}_m{n_pairs}_r{replicate}",
                    "attempt_number": 1, "case_id": case["case_id"], "seed": seed,
                })
    if len({job["attempt_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate scheduled attempt identifiers.")
    return jobs, references


def run_preflight(output: Path) -> None:
    """Retain full check logs and verify every dependency pinned in the lock."""
    checks = [
        ("environment", [sys.executable, "scripts/verify_environment.py"]),
        ("pip_check", [sys.executable, "-m", "pip", "check"]),
        ("pip_freeze", [sys.executable, "-m", "pip", "freeze", "--all"]),
        ("unit_tests", [sys.executable, "-m", "unittest", "discover",
                        "-s", "tests", "-p", "test_kahkm_*.py", "-v"]),
    ]
    logs = output / "checks"
    logs.mkdir()
    for name, command in checks:
        write_json(logs / f"{name}.command.json", {"command": command, "cwd": str(ROOT)})
        with (logs / f"{name}.stdout.log").open("xb") as stdout, \
             (logs / f"{name}.stderr.log").open("xb") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, check=False)
        write_json(logs / f"{name}.result.json", {"returncode": completed.returncode})
        if completed.returncode != 0:
            raise RuntimeError(f"Preflight {name} failed; see retained check logs.")
    packages = []
    for line in (ROOT / "requirements-lock-arm64.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, expected = line.partition("==")
        if not separator or not expected:
            raise ValueError("Unsupported lockfile entry: " + line)
        actual = importlib.metadata.version(name)
        packages.append(dict(name=name, expected=expected, actual=actual,
                             matches=actual == expected))
    write_json(logs / "locked_packages.json", packages)
    if not all(item["matches"] for item in packages):
        raise RuntimeError("Installed packages differ from the reference lockfile.")


def execute_jobs(jobs: list[dict], references: dict, config: dict,
                 output: Path, state: dict) -> None:
    """Save draws before evaluation; do not catch evidence-write failures."""
    options = config["evaluation"]
    with ExitStack() as stack:
        files = {name: stack.enter_context((output / name).open(
            "x", encoding="utf-8", newline="\n",
        )) for name in ("attempts.jsonl", "samples.jsonl", "results.jsonl")}
        events = files["attempts.jsonl"]
        samples = files["samples.jsonl"]
        results = files["results.jsonl"]
        for job in jobs:
            identifier = job["attempt_id"]
            state["active_attempt_id"] = identifier
            append_record(events, dict(attempt_id=identifier, event="started", utc=utc_now()))
            state["started"] += 1
            stage = "sampling"
            try:
                sample = sample_four_state_counts(**job["seed"])
                sample_record = dict(attempt_id=identifier, sample=asdict(sample))
                json_line(sample_record)
            except Exception as exc:
                append_record(events, dict(attempt_id=identifier, event="failed", stage=stage,
                                          utc=utc_now(), error_type=type(exc).__name__,
                                          error=str(exc), traceback=traceback.format_exc()))
                state["failed"] += 1
                state["active_attempt_id"] = None
                continue
            # Deliberately outside the exception handler: write failure aborts.
            append_record(samples, sample_record)
            state["sampled"] += 1
            stage = "evaluation"
            try:
                result = evaluate_four_state_counts(
                    reference=references[job["case_id"]], state_counts=sample.state_counts,
                    delta=options["delta"], exclusion_tolerances=options["exclusion_tolerances"],
                    comparison_atol=options["comparison_atol"],
                )
                record = asdict(result)
                record.pop("reference")  # Stored once in references.json, keyed by case_id.
                record.update(attempt_id=identifier, case_id=job["case_id"])
                json_line(record)
            except Exception as exc:
                append_record(events, dict(attempt_id=identifier, event="failed", stage=stage,
                                          utc=utc_now(), error_type=type(exc).__name__,
                                          error=str(exc), traceback=traceback.format_exc()))
                state["failed"] += 1
                state["active_attempt_id"] = None
                continue
            append_record(results, record)
            append_record(events, dict(attempt_id=identifier, event="completed", utc=utc_now()))
            state["succeeded"] += 1
            state["active_attempt_id"] = None


def write_checksums(output: Path) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file())
    with (output / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{file_sha256(path)}  {path.relative_to(output).as_posix()}\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_campaign(output: Path) -> int:
    output = output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Choose a new output directory outside this working tree.")
    if output.exists():
        raise FileExistsError("Output already exists; it will not be reused or overwritten.")
    raw, config = load_final()
    source_before = source_snapshot()
    jobs, references = build_schedule(config)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    state = dict(status="initializing", started=0, sampled=0, succeeded=0, failed=0,
                 planned=len(jobs), active_attempt_id=None, started_utc=utc_now())
    exit_code = 2
    try:
        with (output / "campaign_spec.json").open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        write_json(output / "campaign_metadata.json", dict(
            runner_id=RUNNER_ID, campaign_id=config["campaign_id"], campaign_role=config["campaign_role"],
            run_id=output.name, output_path=str(output), repository_path=str(ROOT),
            config_path=CONFIG_PATH, config_sha256=SUPPORTED_CONFIG_SHA256,
            source=source_before, python_executable=sys.executable, python_version=sys.version,
            platform=platform.platform(), architecture=platform.machine(),
            argv=sys.argv, process_id=os.getpid(), started_utc=state["started_utc"],
            numerical_environment={name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED")},
        ))
        write_json(output / "references.json", {key: asdict(value) for key, value in references.items()})
        with (output / "schedule.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for job in jobs:
                handle.write(json_line(job))
            handle.flush()
            os.fsync(handle.fileno())
        state["status"] = "preflight"
        run_preflight(output)
        if source_snapshot() != source_before:
            raise RuntimeError("Source changed during preflight; sampling not started.")
        state["status"] = "running"
        execute_jobs(jobs, references, config, output, state)
        state["status"] = "completed_with_failures" if state["failed"] else "completed"
        exit_code = 1 if state["failed"] else 0
    except (Exception, KeyboardInterrupt) as exc:
        state["aborted_stage"] = state["status"]
        state["status"] = "aborted"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    try:
        source_after = source_snapshot()
        source_unchanged = source_after == source_before
    except Exception as exc:
        source_after = {"inspection_error": str(exc)}
        source_unchanged = False
    if not source_unchanged:
        if state["status"].startswith("completed"):
            state["status"] = "invalid_source"
        exit_code = 2
    write_json(output / "source_after.json", source_after)
    state.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic()-start,
                 exit_code=exit_code, source_unchanged=source_unchanged,
                 not_started=state["planned"]-state["started"],
                 started_without_terminal_event=state["started"]-state["succeeded"]-state["failed"],
                 independent_evidence_verification="not_yet_performed")
    write_json(output / "campaign_summary.json", state)
    write_checksums(output)
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New run directory outside the source working tree.")
    args = parser.parse_args()
    try:
        return run_campaign(args.output)
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Retain any created output directory; do not overwrite it.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
