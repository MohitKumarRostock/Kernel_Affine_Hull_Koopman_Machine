"""One command for smoke tests, full reruns, audits, and paper assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from .io import ROOT, environment, materialize, now, package_sources, read_json, verify_archive, write_json


def call(arguments, cwd=ROOT):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg",
               PYTHONDONTWRITEBYTECODE="1")
    print("+ " + " ".join(str(x) for x in arguments), flush=True)
    subprocess.run([str(x) for x in arguments], cwd=cwd, env=env, check=True)


def check_destination(value):
    path = Path(value).resolve()
    for protected in (ROOT / "study", ROOT / "archive", ROOT / "manuscript", ROOT / "reproduce", ROOT / "scripts"):
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError(f"Run/output directory overlaps protected repository content: {path}")
    return path


def start_run(args, mode):
    run = check_destination(args.run_dir or ROOT / "runs" / (mode + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    manifest_path = run / "execution_manifest.json"
    sources = package_sources()
    if run.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError("Use a new --run-dir, or --resume for an intact matching run")
        manifest = read_json(manifest_path)
        if manifest.get("mode") != mode or manifest.get("packaging_sha256") != sources:
            raise ValueError("Cannot resume a run with different orchestration/source hashes or mode")
    else:
        run.mkdir(parents=True)
        manifest = {"format": "kahm_reproduction_execution_v1", "created_utc": now(),
                    "mode": mode, "status": "running", "packaging_sha256": sources,
                    "historical_archive": verify_archive(), "environment": environment(),
                    "scope": "Computational reproduction using published seeds; not new independent evidence or a retrospective preregistration.",
                    "scientific_freezes": "Original freeze files are preserved verbatim. Current execution provenance is recorded separately.",
                    "completed_steps": []}
        write_json(manifest_path, manifest)
    return run, manifest


def step(run, manifest, name, command, cwd=ROOT):
    if name not in manifest["completed_steps"]:
        call(command, cwd)
        manifest["completed_steps"].append(name)
        manifest["updated_utc"] = now()
        write_json(run / "execution_manifest.json", manifest)


def worker_command(command, work, *extra):
    return [sys.executable, "-m", "reproduce.worker", command, "--work", work, *extra]


def assets(input_dir, output_dir):
    script = ROOT / "scripts/build_paper_assets.py"
    if not script.is_file():
        raise FileNotFoundError("Missing paper asset builder scripts/build_paper_assets.py")
    call([sys.executable, script, "--input", input_dir, "--output", output_dir])


def full(args):
    run, manifest = start_run(args, "full")
    first, capacity = run / "stage1_work", run / "capacity_work"
    for name, destination in (("materialize_stage1", first), ("materialize_capacity", capacity)):
        if name not in manifest["completed_steps"]:
            materialize(destination)
            manifest["completed_steps"].append(name)
            write_json(run / "execution_manifest.json", manifest)
    for phase in ("pilot", "confirmation", "natural"):
        step(run, manifest, phase, [sys.executable, first / "run_study.py", "--phase", phase, "--workers", args.workers], first)
    step(run, manifest, "capacity", [sys.executable, capacity / "followup_capacity.py", "--workers", args.workers], capacity)
    step(run, manifest, "numerical_sensitivity",
         worker_command("sensitivity", capacity, "--workers", str(args.workers), "--execution", str(run / "execution_manifest.json")))
    step(run, manifest, "audit_export", worker_command("audit-export", run))
    step(run, manifest, "paper_assets", [sys.executable, ROOT / "scripts/build_paper_assets.py", "--input", run / "results", "--output", run / "paper_assets"])
    verify_archive()
    if package_sources() != manifest["packaging_sha256"]:
        raise ValueError("Orchestration or asset source changed during execution; preserve this run and start a new one")
    manifest.update(status="complete", numerical_reproduction="verified_within_declared_tolerance",
                    completed_utc=now(), outputs={"records_and_audits": "results", "tables_and_figures": "paper_assets"})
    write_json(run / "execution_manifest.json", manifest)
    print(f"Full reproduction complete: {run}", flush=True)


def smoke(args):
    run, manifest = start_run(args, "smoke")
    work = run / "smoke_work"
    if "materialize" not in manifest["completed_steps"]:
        materialize(work)
        manifest["completed_steps"].append("materialize")
        write_json(run / "execution_manifest.json", manifest)
    step(run, manifest, "unit_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    step(run, manifest, "frozen_geometry_tests", [sys.executable, "-m", "unittest", "test_geometry", "test_mechanism_diagnostics", "-v"], work)
    step(run, manifest, "smoke_pipelines", worker_command("smoke", work))
    step(run, manifest, "archive_audit", worker_command("archive-audit", run / "audit_work"))
    step(run, manifest, "paper_assets", [sys.executable, ROOT / "scripts/build_paper_assets.py", "--input", ROOT / "archive", "--output", run / "paper_assets"])
    if package_sources() != manifest["packaging_sha256"]:
        raise ValueError("Orchestration or asset source changed during smoke execution")
    manifest.update(status="complete", completed_utc=now(), scope="Reduced execution checks and archived arithmetic verification; no new research results.")
    write_json(run / "execution_manifest.json", manifest)
    print(f"Smoke checks complete: {run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "full"):
        p = commands.add_parser(name)
        p.add_argument("--run-dir", type=Path)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--workers", type=int, default=4)
    p = commands.add_parser("tables", help="Regenerate paper tables and figures from archived or fresh records")
    p.add_argument("--input", type=Path, default=ROOT / "archive")
    p.add_argument("--output", type=Path, default=ROOT / "manuscript/generated")
    p = commands.add_parser("audit", help="Check immutable hashes, paired statistics, and fresh-run predictions when supplied")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "runs/archive-audit")
    p = commands.add_parser("paper", help="Build manuscript PDF after regenerating archived assets; requires latexmk")
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")
    try:
        if args.command == "full": full(args)
        elif args.command == "smoke": smoke(args)
        elif args.command == "tables": assets(args.input.resolve(), args.output.resolve())
        elif args.command == "audit":
            verify_archive()
            if args.run_dir:
                call(worker_command("audit-export", args.run_dir.resolve()))
            else:
                call(worker_command("archive-audit", check_destination(args.output)))
        else:
            assets(ROOT / "archive", ROOT / "manuscript/generated")
            call(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], ROOT / "manuscript")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Reproduction failed: {error}\n")


if __name__ == "__main__":
    main()
