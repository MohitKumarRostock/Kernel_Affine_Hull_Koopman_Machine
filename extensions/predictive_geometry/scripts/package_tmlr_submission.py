#!/usr/bin/env python3
"""Build deterministic TMLR review bundles without changing scientific evidence.

Run from the extension root:
    python scripts/package_tmlr_submission.py --output-dir dist/tmlr
    python scripts/package_tmlr_submission.py --verify-dir /path/to/extracted/supplement

Review metadata is a separate view. Historical source, archive, freeze records,
published citations and required license notices retain their exact bytes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "SUBMISSION_MANIFEST.json"
MAX_COMBINED_BYTES = 100_000_000
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SOURCE_NAME = "TMLR_Manuscript_Source.zip"
SUPPLEMENT_NAME = "TMLR_Reproducibility_Supplement.zip"
REQUIRED_RECORDS = (
    "archive/stage1/records.json.gz",
    "archive/stage1/margins.json",
    "archive/stage1/primary_contrasts.csv",
    "archive/stage1/mechanism_records.json.gz",
    "archive/capacity/capacity_records.json",
    "archive/capacity/capacity_margins.json",
    "archive/capacity/capacity_primary_contrasts.csv",
    "archive/numerical_sensitivity/numerical_sensitivity_records.json",
    "archive/numerical_sensitivity/numerical_sensitivity_audit.json",
)
SUPPLEMENT_FILES = (
    "README.md", "REPRODUCING.md", "TMLR_SUBMISSION.md", "PAPER_READINESS.md",
    "THIRD_PARTY_NOTICES.md", "LICENSE", "requirements.txt", "pyproject.toml",
    "SCIENTIFIC_AUDIT.txt", "VALIDATION.json",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def content_digest(files):
    """Digest sorted relative names and byte hashes, independent of ZIP metadata."""
    inventory = [[name, len(data), sha256(data)] for name, data in sorted(files.items())]
    return sha256(json_bytes(inventory))


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return name


def read_source(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source symlink or escaped root is not allowed: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Required submission input is missing: {relative}")
    return path.read_bytes()


def verify_historical_inputs(root):
    """Verify both immutable manifests without importing scientific dependencies."""
    snapshot_raw = read_source(root, "study/SNAPSHOT.json")
    archive_raw = read_source(root, "archive/MANIFEST.json")
    snapshot, archive = json.loads(snapshot_raw), json.loads(archive_raw)
    for name, entry in snapshot["files"].items():
        safe_name(entry["path"])
        data = read_source(root, entry["path"])
        if entry["path"].endswith(".gz"):
            data = gzip.decompress(data)
        if sha256(data) != entry["sha256"]:
            raise ValueError(f"Historical source changed: {name}")
    for name, entry in archive["files"].items():
        data = read_source(root, "archive/" + safe_name(name))
        decoded = gzip.decompress(data) if name.endswith(".gz") else data
        if sha256(data) != entry["sha256"] or sha256(decoded) != entry["uncompressed_sha256"]:
            raise ValueError(f"Historical archive changed: {name}")
    return {
        "snapshot_sha256": sha256(snapshot_raw),
        "archive_manifest_sha256": sha256(archive_raw),
        "source_files_verified": len(snapshot["files"]),
        "archive_files_verified": len(archive["files"]),
        "policy": "Original historical bytes are unchanged; these are internal records, not external timestamp attestations.",
    }


def selected_manuscript_files(root):
    """Only editable source, exact style notices and used/generated artwork."""
    manuscript = root / "manuscript"
    for path in sorted(manuscript.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(manuscript)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        name = relative.as_posix()
        if name == "jmlr2e.sty" or path.name == "main.pdf":
            continue
        if path.suffix in {".tex", ".bib", ".sty", ".bst"}:
            yield "manuscript/" + name
        elif name.startswith("generated/figures/") and path.suffix in {".pdf", ".png"}:
            yield "manuscript/" + name
        elif name in {"README_BUILD.md", "LICENSE_TMLR.txt", "generated/claim_sheet.json", "generated/caption_notes.json", "generated/review_diagnostics.json"}:
            yield "manuscript/" + name


def selected_supplement_files(root):
    for name in SUPPLEMENT_FILES:
        yield name
    if (root / "TMLR_VALIDATION.json").is_file():
        yield "TMLR_VALIDATION.json"
    for directory in ("study", "archive", "reproduce", "scripts", "tests", "ci"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo", ".pdf", ".zip"}:
                continue
            yield relative.as_posix()
    for path in sorted((root / "submission/related_work").glob("*.pdf")):
        if path.is_file():
            yield path.relative_to(root).as_posix()
    yield from selected_manuscript_files(root)


def review_metadata(name, data):
    """Remove optional first-party project-author metadata only in the bundle."""
    if name == "pyproject.toml":
        text = data.decode()
        text = re.sub(r"(?m)^authors\s*=\s*\[.*?^\]\s*\n", "", text, flags=re.S)
        return text.encode()
    return data


def identity_tokens(root):
    """Use public project metadata for checks, without embedding names in code."""
    citation = root / "CITATION.cff"
    if not citation.is_file():
        return []
    text = citation.read_text()
    tokens = []
    family = None
    for line in text.splitlines():
        match = re.search(r"family-names:\s*[\"']?([^\"']+)", line)
        if match:
            family = match.group(1).strip()
        match = re.search(r"given-names:\s*[\"']?([^\"']+)", line)
        if match and family:
            tokens.append(match.group(1).strip() + " " + family)
    for key in ("repository-code", "url"):
        match = re.search(rf"(?m)^{key}:\s*[\"']?([^\"'\n]+)", text)
        if match:
            tokens.append(match.group(1).strip())
    return tokens


def attribution_exception(name):
    path = PurePosixPath(name)
    if "LICENSE" in path.name.upper() or path.name.upper() == "COPYING":
        return "Existing license and copyright notice, preserved as required attribution."
    if name == "THIRD_PARTY_NOTICES.md":
        return "Published upstream software, data, and style attribution."
    if name in {"study/upstream_manifest.json", "study/CITATIONS.bib"}:
        return "Immutable source-provenance or literature record bound by historical hashes."
    if name.startswith("study/vendor/upstream/"):
        return "Unmodified vendored dependency and its published source comments."
    if path.suffix == ".bib":
        return "Ordinary literature citations and published software attribution."
    if path.suffix in {".sty", ".bst"}:
        return "Unmodified official style-file author and maintainer attribution."
    return None


def audit_review_view(files, tokens=()):
    """Reject direct identity leaks outside explicit, reported attribution files.

    This is a mechanical check, not a guarantee against identity inference from
    scientific content, legitimate citations or required copyright notices.
    """
    exceptions, historical_paths = [], []
    email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
    current_branch = re.compile(r"https?://[^\s\"'<>}]+/tree/research/(?:kahm-predictive-geometry|certified-soft-state-forecasting)", re.I)
    for name, raw in sorted(files.items()):
        safe_name(name)
        if any(part in {".git", "__MACOSX", "__pycache__"} for part in PurePosixPath(name).parts):
            raise ValueError(f"Excluded metadata in review view: {name}")
        if PurePosixPath(name).name in {"CITATION.cff", "main.pdf"}:
            raise ValueError(f"Excluded named publication or citation metadata: {name}")
        data = gzip.decompress(raw) if name.endswith(".gz") else raw
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        exception = attribution_exception(name)
        if current_branch.search(text):
            raise ValueError(f"Identifying current-study/companion branch URL: {name}")
        if re.search(r"/workspace/scratch/[a-z0-9]+/", text):
            if not (name.startswith("study/") or name.startswith("archive/")):
                raise ValueError(f"Unexpected historical absolute path in first-party overview/code: {name}")
            historical_paths.append(name)
        found = email.search(text) or any(token.casefold() in text.casefold() for token in tokens)
        if found and not exception:
            raise ValueError(f"Identifying author/contact/repository metadata outside attribution: {name}")
        if name == "pyproject.toml" and re.search(r"(?m)^authors\s*=", text):
            raise ValueError("First-party project-author metadata remains in review pyproject.toml")
        if exception:
            exceptions.append({"path": name, "reason": exception})
    return {
        "status": "mechanical_checks_passed_with_declared_attribution",
        "retained_attribution_files": exceptions,
        "retained_historical_absolute_path_files": historical_paths,
        "scope": "No first-party overview names, author/contact metadata, current-study branch links, Git history, or named manuscript PDF are distributed. Published citations and required notices are retained; they are not asserted to identify the submitting authors.",
        "limitation": "This check cannot guarantee anonymity from inference. Historical records and required licenses are not silently rewritten for review.",
    }


REVIEWER_README = """# Reproducibility supplement for TMLR review

This is a source-checkout artifact. No account, repository checkout or network
access is needed after installing the pinned Python dependencies. Extract the
ZIP into an empty directory and run the following commands from that directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/package_tmlr_submission.py --verify-dir .
python -m reproduce audit
python -m reproduce smoke
python -m reproduce full --workers 4 --run-dir runs/full-review
```

The first verification checks every bundled file against SUBMISSION_MANIFEST.json
and checks the original scientific/source archive hashes. Run it before rebuilding
generated assets. The audit recomputes the archived 29- and 12-comparison families;
smoke is an execution check. Only full regenerates all declared fits and predictions
using the recorded seeds. Those reruns are computational reproduction, not new
independent evidence. Full output needs approximately 1 GB free disk space. Use
one worker on machines with limited memory. Python 3.12 and Linux are the reference
environment; requirements.txt and study/environment.json specify versions.

The committed manuscript includes paired-outcome and saved-bank review
diagnostics. A paired-only check leaves those manuscript assets intact:

```bash
python -m reproduce review-diagnostics --output runs/review-diagnostics
```

After the completed full run above, regenerate the full diagnostic set with:

```bash
python -m reproduce review-diagnostics --run-dir runs/full-review --output manuscript/generated
```

Saved-bank rank and gate checks require --run-dir. They are not performed by
the archive-only command, and neither diagnostic route fits a new model.

```bash
python -m reproduce tables
python -m reproduce paper
```

The latter command additionally requires latexmk and a LaTeX distribution. The
separate TMLR_Manuscript_Source.zip can also be built directly with
`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. The previous named
PDF is excluded from both ZIPs; build the anonymous manuscript from its sources.

## Evidence and provenance

All study/ and archive/ bytes are unchanged. Their original manifests, internal
freeze records, margins, numerical environments, upstream licenses and scientific
source are included. All nine source records required for paper-asset generation
are present, along with the remaining archive records and the Airfoil data.
The submission manifest identifies the complete selected source tree by SHA-256,
records every source-to-package mapping and distinguishes the review metadata
view from immutable evidence. It intentionally does not link a current research
branch. Optional project-author metadata is removed only in the packaged
pyproject.toml, with both original and packaged hashes recorded.

Historical internal hash freezes are not external preregistrations or independent
timestamp attestations. Generic old scratch paths are historical provenance,
not live dependencies; the runner resolves them to verified local byte copies.
Fresh executions have separate output directories and manifests and do not
replace historical freeze files. See REPRODUCING.md for the execution contract.

## Attribution and review scope

Required copyright/license notices, published bibliography entries and unchanged
upstream source provenance are retained. They are software/literature attribution,
not an assertion about the submission's authors. SUBMISSION_MANIFEST.json lists
these narrow exceptions. We do not claim that the artifact contains zero personal
names or that identity inference is impossible. First-party overview author
metadata, current-study repository links, CITATION.cff and Git history are omitted.
No scientific source, numerical record, license or freeze has been silently
rewritten to create a false claim of exact historical provenance.
"""

SOURCE_README = """TMLR manuscript source

Build from this directory:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The official TMLR style, bibliography style, required style dependency and license
are included unchanged. The document uses anonymous submission mode. The named
historical PDF, build caches, Git metadata and experiment outputs are omitted.
SUBMISSION_MANIFEST.json records every included file's SHA-256 and content digest.
For experiment execution and all source records, use the separate reproducibility
supplement. See README_BUILD.md for the manuscript-specific build instructions.
"""


def prepare_bundle(root, kind, historical):
    source_paths = list(selected_manuscript_files(root) if kind == "manuscript" else selected_supplement_files(root))
    files, mapping, originals = {}, {}, {}
    for relative in sorted(set(source_paths)):
        original = read_source(root, relative)
        packaged = review_metadata(relative, original)
        name = relative.removeprefix("manuscript/") if kind == "manuscript" else relative
        files[name], originals[relative] = packaged, original
        mapping[name] = {
            "source_path": relative, "source_sha256": sha256(original),
            "transformation": "none" if original == packaged else "Optional first-party authors field removed from review project metadata; no scientific bytes changed.",
        }
    if kind == "manuscript":
        for required in ("main.tex", "tmlr.sty", "tmlr.bst", "fancyhdr.sty", "LICENSE_TMLR.txt"):
            if required not in files:
                raise FileNotFoundError(f"Missing official TMLR manuscript input: {required}")
        main = files["main.tex"].decode()
        if not re.search(r"\\usepackage\s*\{tmlr\}", main):
            raise ValueError("Manuscript must use the default anonymous TMLR submission mode")
        files["README_REVIEW.txt"] = SOURCE_README.encode()
    else:
        for required in REQUIRED_RECORDS:
            if required not in files:
                raise FileNotFoundError(f"Missing paper-asset source record: {required}")
        files["REVIEWER_README.md"] = REVIEWER_README.encode()
    review = audit_review_view(files, identity_tokens(root))
    inventory = {}
    for name, data in sorted(files.items()):
        inventory[name] = {"bytes": len(data), "sha256": sha256(data), **mapping.get(name, {"source_path": None, "transformation": "Generated review instructions."})}
    manifest = {
        "format": "tmlr_review_bundle_v1", "kind": kind,
        "content_sha256": content_digest(files),
        "digest_scope": "Sorted [relative_path, byte_count, sha256] records serialized as sorted, indented JSON with a final newline; excludes this self-referential manifest.",
        "selected_source_tree_sha256": content_digest(originals),
        "source_tree_scope": "All source paths selected for this bundle before optional review metadata transformation; excludes Git history and unrelated repository content.",
        "historical_integrity": historical, "review_audit": review, "files": inventory,
    }
    files[MANIFEST] = json_bytes(manifest)
    return files


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(safe_name(name), date_time=ZIP_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def verify_bundle_files(files):
    manifest = json.loads(files[MANIFEST])
    payload = {name: data for name, data in files.items() if name != MANIFEST}
    if set(payload) != set(manifest["files"]):
        raise ValueError("Submission inventory has missing or unexpected files")
    for name, entry in manifest["files"].items():
        safe_name(name)
        if len(payload[name]) != entry["bytes"] or sha256(payload[name]) != entry["sha256"]:
            raise ValueError(f"Submission file bytes changed: {name}")
    if content_digest(payload) != manifest["content_sha256"]:
        raise ValueError("Submission content-tree digest differs")
    return manifest


def verify_directory(directory):
    root = Path(directory).resolve()
    manifest = json.loads(read_source(root, MANIFEST))
    files = {name: read_source(root, safe_name(name)) for name in manifest["files"]}
    files[MANIFEST] = read_source(root, MANIFEST)
    verified = verify_bundle_files(files)
    if verified["kind"] == "supplement":
        original = verify_historical_inputs(root)
        if original != verified["historical_integrity"]:
            raise ValueError("Historical manifest identity differs from submission provenance")
    return {"status": "verified", "kind": verified["kind"], "files": len(files) - 1,
            "content_sha256": verified["content_sha256"]}


def build_packages(root, output_dir):
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    for name in ("study", "archive", "manuscript", "reproduce", "scripts", "tests"):
        protected = root / name
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output directory overlaps protected submission inputs")
    historical = verify_historical_inputs(root)
    packages = {}
    for kind, filename in (("manuscript", SOURCE_NAME), ("supplement", SUPPLEMENT_NAME)):
        files = prepare_bundle(root, kind, historical)
        manifest = verify_bundle_files(files)
        data = zip_bytes(files)
        packages[filename] = (data, manifest)
    combined = sum(len(data) for data, _ in packages.values())
    if combined >= MAX_COMBINED_BYTES:
        raise ValueError("Combined submission ZIPs must be below 100 MB")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"format": "tmlr_submission_packages_v1", "combined_bytes": combined, "packages": {}}
    for name, (data, manifest) in packages.items():
        temporary = output / (name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(output / name)
        summary["packages"][name] = {"bytes": len(data), "sha256": sha256(data),
            "content_sha256": manifest["content_sha256"], "files": len(manifest["files"]) + 1}
    (output / "TMLR_PACKAGE_MANIFEST.json").write_bytes(json_bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--verify-dir", type=Path)
    args = parser.parse_args()
    try:
        result = verify_directory(args.verify_dir) if args.verify_dir else build_packages(ROOT, args.output_dir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Submission packaging failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
