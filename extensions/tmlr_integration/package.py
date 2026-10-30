#!/usr/bin/env python3
"""Build and audit the integrated anonymous TMLR review artifacts.

This is post-study packaging code. It never changes the registered scientific
sources, frozen diagnostics, or recorded outcomes. The review export omits
identifying historical overviews and carries its own explicitly post-study
inventory; it is not represented as the original complete repository checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator
import zipfile


EXTENSION = Path(__file__).resolve().parent
REPOSITORY = EXTENSION.parents[1]
RELATIVE = Path("extensions/tmlr_integration")
NATURAL = Path("extensions/natural_forecasting")
CONTROLLED = Path("extensions/predictive_geometry")
MANIFEST = "REVIEW_MANIFEST.json"
SOURCE_NAME = "TMLR_Integrated_Manuscript_Source.zip"
SUPPLEMENT_NAME = "TMLR_Integrated_Review_Supplement.zip"
PDF_NAME = "TMLR_Integrated_Manuscript.pdf"
MAX_UPLOAD_BYTES = 100_000_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_ENTRIES = 20000
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "build", "tmp", "dist", ".venv"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".aux", ".out", ".log", ".blg", ".fls", ".fdb_latexmk", ".toc"}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or ":" in path.parts[0]
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return name


def read_source(root: Path, name: str | Path) -> bytes:
    relative = safe_name(Path(name).as_posix())
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing, symbolic, or escaped source: {relative}")
    return path.read_bytes()


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load audit helper: {path.name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def controlled_helper(repository: Path):
    return load_module(repository / CONTROLLED / "scripts/package_tmlr_submission.py", "controlled_review_packager")


def natural_helper(repository: Path):
    return load_module(repository / NATURAL / "package.py", "natural_review_packager")


def source_checks(repository: Path) -> dict[str, Any]:
    """Validate immutable scientific evidence in public or review checkout."""
    controlled = controlled_helper(repository).verify_historical_inputs(repository / CONTROLLED)
    natural = natural_helper(repository)
    freeze = json.loads(read_source(repository, NATURAL / "registration/initial_freeze.json"))
    natural._verify_scientific_hashes(repository, freeze)
    sources = json.loads(read_source(repository, NATURAL / "data_sources.json"))["sources"]
    for source in sources:
        archive = source["archive"]
        raw = read_source(repository, NATURAL / "datasets/raw" / archive["filename"])
        if len(raw) != archive["bytes"] or sha256(raw) != archive["sha256"]:
            raise ValueError(f"Original dataset snapshot changed: {source['source_id']}")
        metadata = read_source(repository, NATURAL / source["zenodo"]["metadata_file"])
        if sha256(metadata) != source["zenodo"]["metadata_sha256"]:
            raise ValueError(f"Official dataset metadata changed: {source['source_id']}")
    checkpoint = json.loads(read_source(repository, NATURAL / "CHECKPOINT.json"))
    if checkpoint.get("format") == "post_study_anonymous_review_inventory_v1":
        for name, expected in checkpoint["tracked_file_sha256"].items():
            if sha256(read_source(repository, name)) != expected:
                raise ValueError(f"Review source inventory changed: {name}")
        checkpoint_result = {
            "review_inventory_files_verified": len(checkpoint["tracked_file_sha256"]),
            "original_checkpoint_verified_at_export": True,
            "scope": "Post-study source subset; not the original full repository checkpoint.",
        }
    else:
        checkpoint_result = natural.verify_checkpoint(repository)
    return {"status": "passed", "controlled": controlled, "natural": checkpoint_result,
            "frozen_scientific_dependencies_verified": len(freeze["code_sha256"]),
            "original_natural_source_archives_verified": len(sources)}


def verify_natural_run(repository: Path, run_directory: Path) -> dict[str, Any]:
    """Verify exact freezes and complete saved outcomes without model fitting."""
    helper = natural_helper(repository)
    roster_bytes = read_source(run_directory, "task_roster.json")
    roster = json.loads(roster_bytes)
    if roster["protocol_sha256"] != digest_file(repository / NATURAL / "protocol.json"):
        raise ValueError("Roster protocol digest differs")
    if roster["source_manifest_sha256"] != digest_file(repository / NATURAL / "data_sources.json"):
        raise ValueError("Roster data-source digest differs")
    freezes = {}
    for name in ("initial_freeze.json", "confirmation_freeze.json"):
        freeze = json.loads(read_source(run_directory, name))
        helper._verify_scientific_hashes(repository, freeze)
        if freeze["roster_sha256"] != sha256(roster_bytes):
            raise ValueError(f"Roster no longer matches {name}")
        freezes[name] = freeze
    final = freezes["confirmation_freeze.json"]
    if sha256(read_source(run_directory, "frozen_diagnostic.json")) != final["diagnostic_sha256"]:
        raise ValueError("Frozen diagnostic changed")
    for name, expected in final["development_result_sha256"].items():
        if sha256(read_source(run_directory, Path("development") / name)) != expected:
            raise ValueError(f"Frozen development outcome changed: {name}")
    tasks = roster["tasks"]
    if len(tasks) != 208 or len({task["task_id"] for task in tasks}) != 208:
        raise ValueError("Expected 208 distinct declared task instances")
    expected = {role: {task["task_id"] for task in tasks if task["role"] == role}
                for role in ("development", "confirmation")}
    if {role: len(ids) for role, ids in expected.items()} != {"development": 128, "confirmation": 80}:
        raise ValueError("Unexpected source-role allocation")
    caches = set()
    for task in tasks:
        if task["cache_path"] not in caches:
            if sha256(read_source(run_directory, task["cache_path"])) != task["cache_sha256"]:
                raise ValueError(f"Prepared arrays changed: {task['task_id']}")
            caches.add(task["cache_path"])
    methods = set(json.loads(read_source(repository, NATURAL / "protocol.json"))["methods"])
    predictions = 0
    for role, identifiers in expected.items():
        observed = set()
        for path in sorted((run_directory / role).glob("*.json")):
            row = json.loads(path.read_text())
            if row["task_id"] in observed or row["task_id"] not in identifiers or row["phase"] != role:
                raise ValueError("Duplicate or unexpected result record")
            observed.add(row["task_id"])
            if row.get("status") != "complete" or row.get("failures"):
                raise ValueError("Incomplete result cannot enter the review supplement")
            candidates = row["candidates"]
            if len(candidates) != len(methods) or {c["candidate_id"] for c in candidates} != methods:
                raise ValueError("Incomplete method portfolio")
            if any(c.get("status") != "ok" for c in candidates):
                raise ValueError("Failed candidate in final results")
            if sha256(read_source(run_directory / role, row["predictions_file"])) != row["predictions_sha256"]:
                raise ValueError("Forecast archive changed")
            predictions += len(candidates)
        if observed != identifiers:
            raise ValueError(f"Missing {role} task records")
    validation = json.loads(read_source(run_directory, "VALIDATION.json"))
    if (validation.get("status") != "passed" or validation.get("predictions_recomputed") != predictions
            or validation.get("frozen_protocol_intact") is not True):
        raise ValueError("Archived final validation failed")
    return {"status": "passed", "task_records": 208, "candidate_predictions": predictions,
            "prepared_array_files": len(caches), "forecasting_fits": 0,
            "original_validation_retained_unchanged": True}


def _add_tree(files: dict[str, bytes], repository: Path, relative: Path, *, suffixes=None):
    for path in sorted((repository / relative).rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(repository)
        if any(part in SKIP_PARTS for part in name.parts) or path.suffix in SKIP_SUFFIXES:
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        files[name.as_posix()] = read_source(repository, name)


def manuscript_payload(repository: Path) -> dict[str, bytes]:
    root = repository / RELATIVE / "manuscript"
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS or part.startswith(".") for part in relative.parts):
            continue
        if path.name in {"main.pdf", "BUILD_CHECK.json"} or path.suffix in SKIP_SUFFIXES:
            continue
        if path.suffix in {".tex", ".bib", ".sty", ".bst", ".bbl", ".txt", ".md", ".json", ".csv"} or ("generated" in relative.parts and path.suffix in {".pdf", ".png", ".eps", ".svg"}):
            files[relative.as_posix()] = read_source(root, relative)
    for name in ("main.tex", "references.bib", "tmlr.sty", "tmlr.bst", "fancyhdr.sty", "LICENSE_TMLR.txt"):
        if name not in files:
            raise ValueError(f"Missing integrated manuscript source: {name}")
    main = files["main.tex"].decode()
    if not re.search(r"\\usepackage\s*\{tmlr\}", main) or re.search(r"\\usepackage\[(?:accepted|preprint)\]\{tmlr\}", main):
        raise ValueError("The review manuscript must use anonymous TMLR submission mode")
    files["README_REVIEW.md"] = SOURCE_README.encode()
    return files


def supplement_payload(repository: Path, run_directory: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    original_checkpoint = json.loads(read_source(repository, NATURAL / "CHECKPOINT.json"))
    if original_checkpoint.get("format") == "post_study_anonymous_review_inventory_v1":
        raise ValueError("Export from the complete public research checkout; a review subset cannot certify the original 217-file checkpoint")
    checks = source_checks(repository)
    checks["natural_saved_run"] = verify_natural_run(repository, run_directory)
    files: dict[str, bytes] = {}
    for directory in ("study", "archive", "reproduce", "scripts", "tests"):
        _add_tree(files, repository, CONTROLLED / directory)
    _add_tree(files, repository, CONTROLLED / "manuscript/generated")
    for name in ("LICENSE", "requirements.txt", "THIRD_PARTY_NOTICES.md", "REPRODUCING.md", "pyproject.toml"):
        files[(CONTROLLED / name).as_posix()] = read_source(repository, CONTROLLED / name)
    project_path = (CONTROLLED / "pyproject.toml").as_posix()
    original_project = files[project_path]
    files[project_path] = controlled_helper(repository).review_metadata("pyproject.toml", original_project)
    # Supplied related-work documents are already anonymous. Their scientific
    # relevance and visual anonymity are assessed before the export is released.
    for path in sorted((repository / CONTROLLED / "submission/related_work").glob("*.pdf")):
        name = path.relative_to(repository).as_posix()
        files[name] = read_source(repository, name)
    for name in ("__init__.py", "run.py", "data.py", "models.py", "baselines.py", "analysis.py",
                 "report.py", "package.py", "protocol.json", "data_sources.json", "requirements.txt", "PROTOCOL.md"):
        files[(NATURAL / name).as_posix()] = read_source(repository, NATURAL / name)
    for directory in ("tests", "datasets"):
        _add_tree(files, repository, NATURAL / directory)
    _add_tree(files, repository, NATURAL / "manuscript/generated")
    for name in ("initial_freeze.json", "confirmation_freeze.json", "frozen_diagnostic.json", "task_index.json"):
        files[(NATURAL / "registration" / name).as_posix()] = read_source(repository, NATURAL / "registration" / name)
    for name, path in natural_helper(repository).run_payload(run_directory).items():
        files[(NATURAL / "runs/natural_v1" / name).as_posix()] = path.read_bytes() if isinstance(path, Path) else path
    # These exact included source bytes form a new metadata-only check used by
    # fresh review reproductions. It neither replaces nor re-dates a prior freeze.
    inventory = {name: sha256(raw) for name, raw in files.items()
                 if "/runs/" not in name and "/submission/" not in name and name != project_path}
    checkpoint_original = read_source(repository, NATURAL / "CHECKPOINT.json")
    review_checkpoint = {
        "format": "post_study_anonymous_review_inventory_v1",
        "generated_after_experiments": True,
        "original_checkpoint_manifest_sha256": sha256(checkpoint_original),
        "original_repository_checkpoint_files_verified_at_export": 217,
        "tracked_file_sha256": inventory,
        "scope": "Post-study inventory of the included unchanged source and data bytes. Identifying overviews and prior manuscript versions are omitted. The original 217-file public checkpoint and scientific freeze records remain unchanged outside this review projection. This is not a preregistration or a replacement historical freeze.",
    }
    files[(NATURAL / "CHECKPOINT.json").as_posix()] = json_bytes(review_checkpoint)
    for name, raw in manuscript_payload(repository).items():
        files[(RELATIVE / "manuscript" / name).as_posix()] = raw
    for name in ("package.py", "package_tests.py", "README.md", "build_assets.py"):
        files[(RELATIVE / name).as_posix()] = read_source(repository, RELATIVE / name)
    files["REVIEWER_README.md"] = REVIEW_README.encode()
    files[(CONTROLLED / "README.md").as_posix()] = CONTROLLED_README.encode()
    files[(NATURAL / "README.md").as_posix()] = NATURAL_README.encode()
    transformations = {
        project_path: {"original_sha256": sha256(original_project), "packaged_sha256": sha256(files[project_path]),
                       "change": "Optional first-party authors metadata omitted; scientific code unchanged."},
        (NATURAL / "CHECKPOINT.json").as_posix(): {
            "original_sha256": sha256(checkpoint_original),
            "packaged_sha256": sha256(files[(NATURAL / "CHECKPOINT.json").as_posix()]),
            "change": "Identifying full-repository metadata replaced only in this review view by explicitly post-study included-source inventory. All frozen scientific dependencies are unchanged.",
        },
    }
    for relative in (CONTROLLED / "README.md", NATURAL / "README.md"):
        name = relative.as_posix()
        transformations[name] = {
            "original_sha256": sha256(read_source(repository, relative)),
            "packaged_sha256": sha256(files[name]),
            "change": "Historical overview replaced only in the review projection by neutral entry-point instructions; scientific bytes are unchanged.",
        }
    return files, {"source_checks": checks, "review_transformations": transformations,
                   "omitted_categories": ["Git history", "first-party citation/contact metadata", "named repository links in publication records and historical overviews", "prior manuscript versions", "unrelated repository files", "build caches"],
                   "scientific_sources_freezes_and_outcomes_modified": False}


def attribution_exception(name: str) -> str | None:
    path = PurePosixPath(name)
    if "LICENSE" in path.name.upper() or path.name.upper() == "COPYING":
        return "Required existing copyright and license attribution."
    if path.suffix in {".bib", ".bst", ".sty"}:
        return "Published references or official style attribution."
    if name.endswith(("study/upstream_manifest.json", "study/CITATIONS.bib", "THIRD_PARTY_NOTICES.md")) or "/study/vendor/" in name:
        return "Unmodified upstream source, provenance, or third-party attribution."
    if "/datasets/metadata/" in name:
        return "Official third-party dataset metadata."
    return None


def audit_anonymity(files: dict[str, bytes], tokens=()) -> dict[str, Any]:
    allowed = []
    email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
    research_link = re.compile(r"https?://[^\s\"'<>}]+/(?:tree|blob)/research/", re.I)
    for name, raw in sorted(files.items()):
        safe_name(name)
        path = PurePosixPath(name)
        if any(part in {".git", "__MACOSX", "__pycache__"} for part in path.parts) or path.name in {"CITATION.cff", "main.pdf"}:
            raise ValueError(f"Excluded identity/build metadata: {name}")
        try:
            data = gzip.decompress(raw) if name.endswith(".gz") else raw
            text = data.decode()
        except (UnicodeDecodeError, OSError):
            continue
        exception = attribution_exception(name)
        # Current-study branch links are never accepted, including in citations.
        if research_link.search(text):
            raise ValueError(f"Current-study branch link in review material: {name}")
        found = email.search(text) or any(token.casefold() in text.casefold() for token in tokens)
        if found and not exception:
            raise ValueError(f"Author/contact/repository identity outside attribution: {name}")
        if path.name == "pyproject.toml" and re.search(r"(?m)^authors\s*=", text):
            raise ValueError("First-party project authors field remains")
        if exception:
            allowed.append({"path": name, "reason": exception})
    return {"status": "passed", "retained_attribution": allowed,
            "scope": "Mechanical text and path checks; required licenses, published citations, and unchanged upstream attribution remain. This is not a guarantee against identity inference from scientific content. Anonymous PDF and related-document rendering are reviewed separately."}


def content_digest(files: dict[str, bytes]) -> str:
    return sha256(json_bytes([[name, len(raw), sha256(raw)] for name, raw in sorted(files.items())]))


def with_manifest(files: dict[str, bytes], kind: str, provenance: dict[str, Any], tokens=()) -> dict[str, bytes]:
    if MANIFEST in files:
        raise ValueError("Reserved manifest name already in payload")
    review = audit_anonymity(files, tokens)
    manifest = {"format": "integrated_tmlr_review_v1", "kind": kind,
                "content_sha256": content_digest(files),
                "files": {name: {"bytes": len(raw), "sha256": sha256(raw)} for name, raw in sorted(files.items())},
                "provenance": provenance, "anonymity_audit": review,
                "digest_scope": "Every listed payload; this self-referential manifest is excluded."}
    return {**files, MANIFEST: json_bytes(manifest)}


def verify_files(files: dict[str, bytes]) -> dict[str, Any]:
    manifest = json.loads(files[MANIFEST])
    if manifest.get("format") != "integrated_tmlr_review_v1":
        raise ValueError("Unsupported review manifest")
    payload = {name: raw for name, raw in files.items() if name != MANIFEST}
    if set(payload) != set(manifest["files"]):
        raise ValueError("Missing or unlisted review files")
    for name, item in manifest["files"].items():
        safe_name(name)
        if len(payload[name]) != item["bytes"] or sha256(payload[name]) != item["sha256"]:
            raise ValueError(f"Review payload changed: {name}")
    if content_digest(payload) != manifest["content_sha256"]:
        raise ValueError("Review content-tree digest differs")
    audit_anonymity(payload)
    return manifest


def read_bundle(path: Path) -> dict[str, bytes]:
    files = {}
    if path.is_dir():
        for source in sorted(path.rglob("*")):
            if source.is_file():
                name = source.relative_to(path).as_posix()
                files[name] = read_source(path, name)
    else:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ENTRIES or sum(item.file_size for item in entries) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Unreasonably large review archive")
            for item in entries:
                name = safe_name(item.filename)
                if item.is_dir() or name in files or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f"Duplicate, directory, or symbolic entry: {name}")
                files[name] = archive.read(item)
    verify_files(files)
    return files


def verify(path: Path) -> dict[str, Any]:
    files = read_bundle(path)
    manifest = json.loads(files[MANIFEST])
    return {"status": "passed", "kind": manifest["kind"], "payload_files_verified": len(files) - 1,
            "content_sha256": manifest["content_sha256"]}


@contextmanager
def repository_view(path: Path) -> Iterator[Path]:
    if path.is_dir() and not (path / MANIFEST).exists():
        yield path.resolve()
        return
    files = read_bundle(path)
    if json.loads(files[MANIFEST])["kind"] != "review-supplement":
        raise ValueError("Scientific audit requires the review supplement, not the manuscript source ZIP")
    if path.is_dir():
        yield path.resolve()
        return
    with tempfile.TemporaryDirectory(prefix="tmlr-review-") as temporary:
        root = Path(temporary)
        for name, raw in files.items():
            destination = root / safe_name(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        yield root


def audit(repository: Path, output: Path | None = None) -> dict[str, Any]:
    """Replay statistics and saved predictions; never fit forecasting models."""
    if output is not None:
        output = output.resolve()
        if output.is_relative_to(repository.resolve()):
            raise ValueError("Replay output must be outside the supplied source tree")
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("Replay output must be empty")
    checks = source_checks(repository)
    run_directory = repository / NATURAL / "runs/natural_v1"
    checks["natural_saved_run"] = verify_natural_run(repository, run_directory)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    if any(name.startswith("extensions.natural_forecasting.run") for name in sys.modules):
        raise RuntimeError("Audit must run in a fresh Python CLI process")
    for name in list(sys.modules):
        if name == "extensions" or name.startswith("extensions."):
            del sys.modules[name]
    sys.path.insert(0, str(repository))
    runner = importlib.import_module("extensions.natural_forecasting.run")
    analysis = importlib.import_module("extensions.natural_forecasting.analysis")
    helper = natural_helper(repository)
    numeric = runner.audit_results(run_directory)
    if numeric["predictions_recomputed"] != 2496:
        raise ValueError("Expected 2,496 reconstructed task-method losses")
    roster = json.loads((run_directory / "task_roster.json").read_text())
    config = runner.analysis_configuration(roster)
    development = runner.load_results(run_directory, "development")
    confirmation = runner.load_results(run_directory, "confirmation")
    analysis.validate_tasks(development, config, role="development")
    analysis.validate_tasks(confirmation, config, role="confirmation", allow_confirmation=True)
    frozen = json.loads((run_directory / "frozen_diagnostic.json").read_text())
    reports = {
        "development": analysis.analyze_development(development, config=config),
        "confirmation": analysis.evaluate_frozen(confirmation, frozen, allow_confirmation=True),
    }
    for phase, report in reports.items():
        helper.audit_analysis_report(json.loads((run_directory / f"{phase}_analysis.json").read_text()), report, analysis, config)
    for name, features in analysis.MODEL_FEATURES.items():
        fitted = analysis.fit_diagnostic(development, features, config)
        helper.compare_content(frozen["models"][name], fitted, ignored_fields=helper.REMEASURED_FIELDS, path=f"frozen_models.{name}")
    if frozen["development_input_sha256"] != analysis._digest(sorted(development, key=lambda task: task["task_id"])):
        raise ValueError("Frozen diagnostic input digest differs")
    with tempfile.TemporaryDirectory(prefix="tmlr-controlled-audit-") as temporary:
        controlled_output = Path(temporary) / "statistics"
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
        result = subprocess.run([sys.executable, "-m", "reproduce", "audit", "--output", str(controlled_output)],
                                cwd=repository / CONTROLLED, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Controlled archive audit failed:\n" + result.stdout[-6000:] + result.stderr[-6000:])
        controlled_log = result.stdout + result.stderr
    checks.update(numeric_audit=numeric, controlled_statistical_audit="passed",
                  frozen_diagnostic_coefficients_recomputed=True, archived_analysis_reports_recomputed=True,
                  forecasting_model_or_geometry_fits=0, source_and_outcomes_modified=False,
                  timing_scope="Recorded costs are checked arithmetically; fresh small diagnostic fit/decision times are not required to equal archived timings.")
    if output is not None:
        for phase, report in reports.items():
            (output / f"{phase}_analysis_replay.json").write_bytes(json_bytes(report))
        (output / "controlled_statistical_audit.txt").write_text(controlled_log)
        (output / "REPLAY_VALIDATION.json").write_bytes(json_bytes(checks))
    return checks


def write_zip(path: Path, files: dict[str, bytes]):
    verify_files(files)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(safe_name(name), ZIP_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    if temporary.stat().st_size >= MAX_UPLOAD_BYTES:
        temporary.unlink()
        raise ValueError("Review archive exceeds the 100 MB supplementary limit")
    temporary.replace(path)


def build(repository: Path, run_directory: Path, pdf: Path, output: Path, overwrite=False) -> dict[str, Any]:
    repository, run_directory, pdf, output = (p.resolve() for p in (repository, run_directory, pdf, output))
    if output == repository or output.is_relative_to(repository / "extensions"):
        raise ValueError("Build artifacts must be outside protected scientific source directories")
    if not pdf.is_file() or pdf.is_symlink() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("Provide the compiled integrated PDF")
    source = manuscript_payload(repository)
    manuscript = repository / RELATIVE / "manuscript"
    for name in source:
        path = manuscript / name
        if path.is_file() and path.suffix in {".tex", ".bib", ".sty", ".bst", ".pdf", ".png", ".eps", ".svg"} and path.stat().st_mtime > pdf.stat().st_mtime + 1:
            raise ValueError(f"PDF predates a manuscript input: {name}")
    files, provenance = supplement_payload(repository, run_directory)
    tokens = controlled_helper(repository).identity_tokens(repository / CONTROLLED)
    source_files = with_manifest(source, "manuscript-source", {"post_study_export": True}, tokens)
    supplement_files = with_manifest(files, "review-supplement", provenance, tokens)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in (SOURCE_NAME, SUPPLEMENT_NAME, PDF_NAME, "RELEASE_MANIFEST.json")]
    if not overwrite and any(path.exists() and path != pdf for path in targets):
        raise FileExistsError("Use a new output directory or explicit --overwrite for a prior generated release")
    write_zip(output / SOURCE_NAME, source_files)
    write_zip(output / SUPPLEMENT_NAME, supplement_files)
    if sum((output / name).stat().st_size for name in (SOURCE_NAME, SUPPLEMENT_NAME)) >= MAX_UPLOAD_BYTES:
        raise ValueError("Combined source and supplementary ZIPs must remain below 100 MB")
    if pdf != output / PDF_NAME:
        shutil.copy2(pdf, output / PDF_NAME)
    result = {"format": "integrated_tmlr_release_v1", "status": "built",
              "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest_file(path)} for path in targets[:-1]},
              "source_checks": provenance["source_checks"],
              "scope": "Anonymous review files and deterministic source/reproducibility exports. Author scientific review and submission approval are not certified by this build."}
    (output / "RELEASE_MANIFEST.json").write_bytes(json_bytes(result))
    return result


SOURCE_README = """# Integrated TMLR manuscript source

This is the anonymous source of the integrated study. Build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The official TMLR submission style is included unchanged. Generated numerical
tables and figures are included; compilation requires no scientific model fits.
The separate review supplement contains executable audits, frozen sources,
saved forecasts, and source data. This file set is an unpublished author-review
draft and does not certify journal submission or author approval.
"""

CONTROLLED_README = """# Controlled predictive-geometry evidence

This review projection preserves the scientific source snapshot, archived
records, and reproduction code unchanged. The integrated paper is in
`../tmlr_integration/manuscript/`. Use the top-level REVIEWER_README.md for
review commands. The older manuscript and identifying historical overviews are
omitted; ordinary citations and required copyright notices remain.
"""

NATURAL_README = """# Natural-source forecasting evidence

The original scientific code, both freezes, task roster, source data, saved
forecasts, and recorded analyses are unchanged. The adjacent CHECKPOINT.json
is explicitly a post-study inventory of included review sources, not the
original full public repository checkpoint. Use the integrated package.py
verification/audit commands in REVIEWER_README.md; the historical standalone
package.py assumes the complete public checkpoint and is retained only as a
shared audit helper. Fresh full runs may use the unchanged run.py commands and
their own new freeze; do not overwrite the included runs/natural_v1 evidence.
"""

REVIEW_README = """# Integrated TMLR anonymous review supplement

Start here. This supplement supports one integrated manuscript, with controlled
geometry experiments and a separate natural-source forecasting study. It
contains the complete natural saved run (208 tasks and 2,496 method losses),
the controlled compact archive, exact scientific sources, pinned requirements,
all eight original natural-data snapshots, and integrated LaTeX source.

## Verify before generating any files

From this extracted root, using Python 3.12:

```bash
python extensions/tmlr_integration/package.py verify .
python extensions/tmlr_integration/package.py check-source
```

Inventory verification is read-only and uses only the standard library. It
rejects missing, changed, unlisted, duplicate, symbolic, or unsafe payloads.
It must precede compilation or adding any files to the extracted tree.

## Reconstruct the results without fitting forecasting models

Install dependencies in an environment outside the extracted artifact:

```bash
python3.12 -m venv ../tmlr-review-env
source ../tmlr-review-env/bin/activate
python -m pip install -r extensions/natural_forecasting/requirements.txt
python extensions/tmlr_integration/package.py audit .
python extensions/tmlr_integration/package.py replay . --output ../tmlr-replay
```

Audit independently checks all saved natural prediction arrays against their
targets/origins and recomputes 2,496 losses, source-balanced analyses, and frozen
diagnostic coefficients. It also replays the controlled archived paired
statistics. The small diagnostic ridge regressions are refitted; forecasting
models and geometric banks are not. Replay writes only to a separate empty
directory. Recorded pipeline costs remain archival observations, not new
measurements. Both commands accept the review ZIP directly as an alternative
to an extracted root and use a temporary verified extraction.

## Full computational reproduction and manuscript build

The controlled study's unchanged full-run command is available from its own
extension directory: `python -m reproduce full --workers 4 --run-dir PATH`.
Choose a new output path. Natural-source full runs follow PROTOCOL.md and the
unchanged run.py commands `prepare`, `initial-freeze`, `run --role development`,
`freeze-confirmation`, `run --role confirmation`, and `analyze`, each with a new
`--run-root`. These reproduce existing seeds and do not add independent evidence.
The review CHECKPOINT.json verifies the included source subset for fresh runs.

Build the anonymous paper after initial inventory verification:

```bash
cd extensions/tmlr_integration/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## What anonymity changes, and what it does not

This export omits Git history, identifying overview/contact/citation metadata,
named repository links, prior manuscript versions, and unrelated files. Optional
project-author metadata is removed from one build configuration. The original
full-repository checkpoint metadata is replaced only in this export by a clearly
labeled post-study inventory of the included unchanged source/data subset. The
review manifest records original and packaged hashes for these transformations.
The historical public checkpoint is verified before export and remains intact.

Scientific code, the original scientific freezes, sources, task identities,
predictions, and outcomes are not rewritten or re-dated. The original saved
VALIDATION.json records checks performed in the original complete repository;
it is retained as historical evidence. New review audits state their own scope.
The retained third-party licenses, published citations, and exact upstream
source-provenance records are declared attribution exceptions. These do not
assert that the upstream authors are the submitting authors. Historical generic
execution paths in frozen records are unchanged. Mechanical identity checks
cannot prevent inference from the research itself.

The study is an unpublished draft for substantive author review. The artifact
does not certify authorship, funding/conflict declarations, author agreement,
journal submission, acceptance, or independent external replication.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-source")
    for name in ("verify", "audit", "replay"):
        command = commands.add_parser(name)
        command.add_argument("path", type=Path)
        if name == "replay":
            command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("build")
    command.add_argument("--pdf", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--run-dir", type=Path, default=REPOSITORY / NATURAL / "runs/natural_v1")
    command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    if args.command == "check-source":
        result = source_checks(REPOSITORY)
    elif args.command == "verify":
        result = verify(args.path.resolve())
    elif args.command in {"audit", "replay"}:
        with repository_view(args.path.resolve()) as root:
            result = audit(root, args.output if args.command == "replay" else None)
    else:
        result = build(REPOSITORY, args.run_dir, args.pdf, args.output, args.overwrite)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
