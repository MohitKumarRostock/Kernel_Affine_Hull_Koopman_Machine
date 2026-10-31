"""Package the manuscript, supporting studies, and saved-forecast evidence.

The builder checks source dependencies, declared template hashes, and the
existing saved-forecast manifest. It copies numerical evidence unchanged and
does not compile documents, evaluate models, or rerun statistical analysis.
Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

TITLE = "Kernel Affine Hull Features for Time-Series Forecasting: Distinctness, Bounded Corrections, and Predictive Limits"
MAIN_NAME = "KAHM_Reconstruction_Prediction_Manuscript.pdf"
SUPPORT_NAME = "KAHM_Supplementary_Studies.pdf"
SOURCE_NAME = "KAHM_Reconstruction_Prediction_Manuscript_Source.zip"
REVIEW_NAME = "KAHM_TMLR_Submission_Supplement.zip"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
REVIEW_LIMIT = 100_000_000
EVIDENCE_PREFIX = "saved_forecast_supplement/"
TEMPLATE_FILES = ("tmlr.sty", "tmlr.bst", "fancyhdr.sty", "LICENSE_TMLR.txt")
SOURCE_DOCUMENTS = ("README_BUILD.md", "TEMPLATE_PROVENANCE.json", "LICENSE", "DATA_LICENSE")
DEPENDENCIES = re.compile(
    r"\\(input|include|includegraphics\*?|bibliography|bibliographystyle|usepackage|RequirePackage)"
    r"(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}"
)

REVIEW_README = f"""# Supporting studies and saved-forecast evidence

This supplement accompanies **{TITLE}**.
The main manuscript is supplied as a separate PDF. It contains the central
method, theoretical statements and complete proofs, principal experimental
protocol, essential results, and limitations.

## Contents

- `{SUPPORT_NAME}` presents the complementary controlled, natural-data,
  scaling, additive-correction, target-geometry, direct-training, and online
  adaptation studies. It includes their methods, results, and technical
  appendices, with its own notation, references, and study-level limitations.
- `saved_forecast_supplement/` contains the existing numerical evidence and
  standalone statistical verifier for the principal reconstruction-departure
  experiment. Every file in this directory is preserved byte for byte from
  the saved-forecast archive, including its original manifest.
- `SUPPLEMENT_MANIFEST.json` records the size and SHA-256 digest of every other
  file in this review supplement.

## Numerical reproduction

Use Python 3.11 or later. From the `saved_forecast_supplement` directory:

```bash
python -m pip install -r requirements.txt
python verify_saved_forecasts.py
```

The verifier checks file identities, the complete task and method roster,
numeric array identities and dimensions, training-target variance scales,
normalized forecast losses, eight analysis tables, and six primary statistical
comparisons. Its README specifies the aggregation and simultaneous inference.
The evidence contains 192 task suites, ten forecasting procedures, and 24
independent repetitions. Forecast windows within a repetition are not treated
as independent statistical observations.

This verification recomputes the principal numerical comparisons from saved
forecasts. It does not generate trajectories, refit models, repeat model
selection, remeasure computation time, recalculate feature-bound diagnostics,
or rerun the complementary studies. The complete fitting procedure is described
in the manuscript and its experimental appendix. Reproducing statistics from
saved forecasts and conducting an independent experimental replication have
different evidential scope.

The supporting studies use their own populations, procedures, and evaluation
protocols. Their results are not pooled with the principal experiment. The
unfavorable comparisons, incomplete task coverage, and other limitations are
retained in the corresponding studies.

This review supplement excludes the editable manuscript source and its
author-identifying license attribution. The separately distributed source
archive is intended for editing and compilation and is not an anonymous
review upload.
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(name: str, data: bytes) -> dict[str, Any]:
    return {"path": name, "bytes": len(data), "sha256": digest(data)}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def safe_name(name: str) -> str:
    value = PurePosixPath(name)
    require(bool(name) and "\\" not in name and not value.is_absolute(), f"Invalid archive path: {name}")
    require(all(part not in (".", "..") and not part.startswith(".") for part in value.parts),
            f"Unsafe archive path: {name}")
    require(value.as_posix() == name, f"Noncanonical archive path: {name}")
    return name


def source_file(root: Path, name: str) -> Path:
    safe_name(name)
    path = root / name
    require(path.is_file() and not path.is_symlink(), f"Missing or symbolic source file: {name}")
    require(path.resolve().is_relative_to(root), f"Source path escapes its directory: {name}")
    return path


def check_template(root: Path) -> dict[str, str]:
    provenance = json.loads(source_file(root, "TEMPLATE_PROVENANCE.json").read_text(encoding="utf-8"))
    expected = provenance.get("files", {})
    checked = {}
    for name in TEMPLATE_FILES:
        actual = digest(source_file(root, name).read_bytes())
        require(actual == expected.get(name), f"Template file differs from its declared hash: {name}")
        checked[name] = actual
    return checked


def collect_source(root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """Resolve the explicit document dependencies into an exact file whitelist."""
    checked = check_template(root)
    selected: set[str] = set(SOURCE_DOCUMENTS) | set(TEMPLATE_FILES)
    queue = ["main.tex", "supplement.tex"]
    visited: set[str] = set()

    def resolve(name: str, suffixes: tuple[str, ...]) -> str:
        require("\\" not in name and "#" not in name, f"Dynamic source dependency needs an explicit path: {name}")
        candidates = [name] if PurePosixPath(name).suffix else [name + suffix for suffix in suffixes]
        for candidate in candidates:
            safe_name(candidate)
            if (root / candidate).is_file():
                source_file(root, candidate)
                return candidate
        raise ValueError(f"Missing document dependency: {name}")

    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        selected.add(name)
        text = source_file(root, name).read_text(encoding="utf-8")
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        for command, argument in DEPENDENCIES.findall(text):
            for item in argument.split(","):
                item = item.strip()
                if command in ("usepackage", "RequirePackage"):
                    if (root / (item + ".sty")).is_file():
                        selected.add(resolve(item, (".sty",)))
                elif command in ("input", "include"):
                    queue.append(resolve(item, (".tex",)))
                elif command.startswith("includegraphics"):
                    selected.add(resolve(item, (".pdf", ".png", ".jpg", ".jpeg")))
                elif command == "bibliography":
                    selected.add(resolve(item, (".bib",)))
                else:
                    selected.add(resolve(item, (".bst",)))
    return {name: source_file(root, name).read_bytes() for name in sorted(selected)}, checked


def read_saved_forecasts(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Validate the existing archive without executing its verification code."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "Duplicate saved-forecast archive member")
        members = {}
        for info in infos:
            name = safe_name(info.filename)
            require(not info.is_dir() and name.startswith(EVIDENCE_PREFIX), f"Unexpected saved archive member: {name}")
            require(not info.flag_bits & 1, f"Encrypted saved archive member: {name}")
            mode = stat.S_IFMT(info.external_attr >> 16)
            require(mode in (0, stat.S_IFREG), f"Nonregular saved archive member: {name}")
            members[name] = archive.read(info)
    manifest_name = EVIDENCE_PREFIX + "MANIFEST.json"
    require(manifest_name in members, "Saved-forecast manifest is missing")
    manifest = json.loads(members[manifest_name])
    require(manifest.get("schema") == "kahm-saved-forecast-manifest-v1", "Unknown saved-forecast manifest schema")
    entries = manifest.get("files", [])
    require(isinstance(entries, list) and bool(entries), "Saved-forecast manifest has no files")
    listed = set()
    checked_bytes = 0
    for entry in entries:
        name = EVIDENCE_PREFIX + safe_name(entry["path"])
        require(name not in listed, f"Duplicate manifest path: {name}")
        listed.add(name)
        require(name in members, f"Manifest member is missing: {name}")
        value = members[name]
        require(len(value) == entry["bytes"] and digest(value) == entry["sha256"], f"Saved-forecast identity differs: {name}")
        checked_bytes += len(value)
    require(set(members) == listed | {manifest_name}, "Saved archive contains unmanifested members")
    return members, {
        "archive_sha256": digest(path.read_bytes()), "members_preserved": len(members),
        "manifest_files_checked": len(entries), "manifest_bytes_checked": checked_bytes,
        "member_content_preserved_byte_for_byte": True,
        "statistical_verifier_executed_by_packager": False,
    }


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            entry = zipfile.ZipInfo(safe_name(name), date_time=ZIP_TIME)
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data, compresslevel=9)
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == set(members), f"Packaged member roster differs: {path.name}")
        for name, data in members.items():
            require(archive.read(name) == data, f"Packaged member content differs: {name}")


def make_package(main_pdf: Path, supplement_pdf: Path, source_dir: Path,
                 saved_forecast_zip: Path, output_dir: Path) -> dict[str, Any]:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    require(not output_dir.exists(), "The output directory must be new")
    require(not output_dir.is_relative_to(source_dir), "Write the package outside the source directory")
    source_members, template_checked = collect_source(source_dir)
    saved_members, saved_checks = read_saved_forecasts(saved_forecast_zip)
    pdfs = {}
    for name, path in ((MAIN_NAME, main_pdf), (SUPPORT_NAME, supplement_pdf)):
        data = path.read_bytes()
        require(data.startswith(b"%PDF-"), f"Input does not have a PDF header: {name}")
        pdfs[name] = data
    source_manifest = {
        "schema": "kahm-document-source-manifest-v1", "entry_points": ["main.tex", "supplement.tex"],
        "files": [record(name, data) for name, data in source_members.items()],
        "scope": "Exact explicit document dependencies, build instructions, template provenance, and required licenses.",
        "anonymous_review_upload": False,
        "attribution": "Original author-identifying license attribution is retained.",
    }
    source_zip_members = {"manuscript/" + name: data for name, data in source_members.items()}
    source_zip_members["manuscript/SOURCE_FILE_MANIFEST.json"] = json_bytes(source_manifest)
    review_members = dict(saved_members)
    review_members[SUPPORT_NAME] = pdfs[SUPPORT_NAME]
    review_members["README.md"] = REVIEW_README.encode("utf-8")
    review_members["SUPPLEMENT_MANIFEST.json"] = json_bytes({
        "schema": "kahm-tmlr-review-supplement-v1",
        "files": [record(name, data) for name, data in sorted(review_members.items())],
        "scope": "Supporting-study PDF and unchanged saved-forecast supplement; no model-fitting implementation is included.",
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".", dir=output_dir.parent))
    try:
        for name, data in pdfs.items():
            (staging / name).write_bytes(data)
        write_zip(staging / SOURCE_NAME, source_zip_members)
        write_zip(staging / REVIEW_NAME, review_members)
        review_bytes = (staging / REVIEW_NAME).stat().st_size
        require(review_bytes < REVIEW_LIMIT, f"Review supplement exceeds 100 MB: {review_bytes} bytes")
        release = {
            "schema": "kahm-tmlr-document-release-v1", "title": TITLE,
            "artifacts": [record(path.name, path.read_bytes()) for path in sorted(staging.iterdir())],
            "submission_files": [MAIN_NAME, REVIEW_NAME], "editable_source_archive": SOURCE_NAME,
            "source_archive_is_anonymous": False,
            "packaging_checks": {
                "template_files_match_declared_provenance": template_checked,
                "explicit_source_dependencies_resolved": True, "source_member_count": len(source_zip_members),
                "required_licenses_retained": True, "archive_member_bytes_read_back_and_verified": True,
                "input_pdf_headers_checked": True, "review_archive_bytes": review_bytes,
                "review_archive_limit_bytes": REVIEW_LIMIT, "saved_forecast_integrity": saved_checks,
            },
            "scope": "Packaging integrity only. The builder does not compile or visually inspect PDFs, verify scientific claims, rerun statistics, refit models, replay selected models, or validate the anonymity of arbitrary input PDF content.",
            "reproducibility": "ZIP members use sorted names, fixed timestamps, and fixed file permissions; compressed bytes also depend on the Python/zlib implementation.",
        }
        (staging / "RELEASE_MANIFEST.json").write_bytes(json_bytes(release))
        staging.replace(output_dir)
        return release
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-pdf", type=Path, required=True)
    parser.add_argument("--supplement-pdf", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--saved-forecast-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the release artifacts")
    args = parser.parse_args()
    try:
        release = make_package(args.main_pdf, args.supplement_pdf, args.source_dir,
                               args.saved_forecast_zip, args.output_dir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Packaging failed: {error}\n")
    print(json.dumps({"artifacts": release["artifacts"],
                      "saved_forecast_integrity": release["packaging_checks"]["saved_forecast_integrity"],
                      "source_archive_is_anonymous": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
