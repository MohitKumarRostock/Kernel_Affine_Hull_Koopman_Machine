#!/usr/bin/env python3
"""Verify scientific code, evidence identities and repository documentation links.

The checks use Git and the Python standard library. They do not fit models,
evaluate forecasts, or compile documents.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("docs/PUBLICATION_SCOPE.json")


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def tree(ref: str) -> dict[str, dict[str, str]]:
    records = {}
    for row in git("ls-tree", "-r", "-z", ref).split(b"\0"):
        if not row:
            continue
        header, raw_path = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        records[raw_path.decode()] = {"mode": mode, "type": kind, "git_object": oid}
    return records


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    require(bool(value) and not path.is_absolute() and ".." not in path.parts,
            f"Invalid repository path: {value}")
    return path.as_posix()


def source_commits(plan: dict) -> list[str]:
    commits = {source["commit"] for source in plan["scientific_sources"].values()}
    commits.update(plan["source_link_commits"])
    require(bool(commits) and all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in commits),
            "Source references must be full Git commit identifiers.")
    return sorted(commits)


def verify_scientific_files(reference: dict, current: dict, prefix: str,
                            documentation: set[str], metadata: set[str]) -> int:
    """Check all scientific paths and Git identities, with explicit reader-file exceptions."""
    relative_path(prefix)
    for path in documentation | metadata:
        relative_path(path)
        require(path.startswith(prefix + "/") and path in reference and path in current,
                f"Invalid reader-file exception: {path}")
        require(reference[path]["mode"] == current[path]["mode"],
                f"Reader-file mode differs: {path}")
    require(all(path.endswith(".md") for path in documentation),
            "Reader-documentation exceptions must be Markdown files.")
    require(all(path.endswith(".json") for path in metadata),
            "Documentation-manifest exceptions must be JSON files.")
    exceptions = documentation | metadata
    expected = {p: v for p, v in reference.items()
                if p.startswith(prefix + "/") and p not in exceptions}
    observed = {p: v for p, v in current.items()
                if p.startswith(prefix + "/") and p not in exceptions}
    require(bool(expected), f"No scientific files at source path: {prefix}")
    require(set(observed) == set(expected),
            f"Scientific file set differs from the source specification: {prefix}")
    for path, entry in expected.items():
        require(observed[path] == entry, f"Scientific file identity differs: {path}")
    return len(expected)


def verify_documentation_manifest(path: str, reader_files: list[str],
                                  documentation: set[str], reference_data: bytes) -> int:
    """Permit only README byte-count/hash updates in a bundle's file manifest."""
    expected = copy.deepcopy(json.loads(reference_data))
    observed = json.loads((ROOT / path).read_text())
    parent = PurePosixPath(path).parent
    require(bool(reader_files), f"No reader entries specified for {path}")
    for name in reader_files:
        relative_path(name)
        target = (parent / name).as_posix()
        require(target in documentation, f"Manifest exception is not reader documentation: {target}")
        require(name in expected["files"], f"Missing source manifest entry: {name}")
        data = (ROOT / target).read_bytes()
        expected["files"][name]["bytes"] = len(data)
        expected["files"][name]["sha256"] = hashlib.sha256(data).hexdigest()
    require(observed == expected,
            f"Manifest differs outside its reader-file identities, or has a stale identity: {path}")
    return len(reader_files)


def known_path(path: str, records: dict) -> bool:
    return not path or path in records or any(p.startswith(path.rstrip("/") + "/") for p in records)


def source_link(link: str, repository: str) -> tuple[str, str] | None:
    parsed = urlsplit(link)
    parts = unquote(parsed.path).strip("/").split("/")
    owner, name = repository.split("/")
    if parsed.netloc.lower() in {"github.com", "www.github.com"}:
        if len(parts) >= 4 and parts[:2] == [owner, name] and parts[2] in {"tree", "blob", "raw"}:
            return parts[3], "/".join(parts[4:]).rstrip("/")
    if parsed.netloc.lower() == "raw.githubusercontent.com":
        if len(parts) >= 3 and parts[:2] == [owner, name]:
            return parts[2], "/".join(parts[3:]).rstrip("/")
    return None


def verify_documentation_links(current: dict, references: dict, plan: dict) -> tuple[int, int]:
    local_count = source_count = 0
    excluded = set(plan["excluded_document_roots"])
    for name in current:
        if not name.endswith(".md"):
            continue
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        links = set(re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", text))
        links.update(re.findall(r"https?://[^\s<>`\")]+", text))
        for link in links:
            link = link.strip("<>")
            source = source_link(link, plan["repository"])
            if source is not None:
                ref, target = source
                require(ref in references, f"Undeclared source commit in {name}: {ref}")
                if target:
                    relative_path(target)
                    require(PurePosixPath(target).parts[0] not in excluded,
                            f"Document link is outside the code and evidence scope in {name}: {link}")
                require(known_path(target, references[ref]), f"Unknown source path in {name}: {link}")
                source_count += 1
            elif not urlsplit(link).scheme and not link.startswith("#"):
                target = unquote(urlsplit(link).path)
                resolved = (path.parent / target).resolve()
                require(resolved.is_relative_to(ROOT), f"Local link escapes the repository in {name}: {link}")
                relative = resolved.relative_to(ROOT).as_posix().rstrip("/")
                if relative == ".":
                    relative = ""
                require(known_path(relative, current), f"Broken or untracked local link in {name}: {link}")
                local_count += 1
    return local_count, source_count


def verify_repository(plan: dict) -> dict:
    current = tree("HEAD")
    references = {ref: tree(ref) for ref in source_commits(plan)}
    scientific_count = documentation_count = manifest_count = 0
    for prefix, source in plan["scientific_sources"].items():
        ref = source["commit"]
        require(git("rev-parse", f"{ref}:{prefix}").decode().strip() == source["tree"],
                f"Source subtree identity differs: {prefix}")
        documentation = set(source["reader_documentation"])
        manifests = source.get("documentation_manifests", {})
        scientific_count += verify_scientific_files(
            references[ref], current, prefix, documentation, set(manifests))
        documentation_count += len(documentation)
        for path, reader_files in manifests.items():
            manifest_count += verify_documentation_manifest(
                path, reader_files, documentation, git("show", f"{ref}:{path}"))
    for path, entry in current.items():
        require(entry["type"] == "blob" and entry["mode"] in {"100644", "100755"},
                f"Unexpected submodule or symbolic link: {path}")
        data = (ROOT / path).read_bytes()
        oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(oid == entry["git_object"], f"Working file differs from its Git checksum: {path}")
    require({p.split("/")[0] for p in current} == set(plan["root_entries"]),
            "Repository root entries differ from the declared file specification.")
    for root, expected in plan["study_directories"].items():
        observed = {p.split("/")[1] for p in current if p.startswith(root + "/") and len(p.split("/")) > 2}
        require(observed == set(expected), f"Unexpected study directory in {root}.")
    local_links, reference_links = verify_documentation_links(current, references, plan)
    return {
        "status": "passed",
        "checked_commit": git("rev-parse", "HEAD").decode().strip(),
        "source_commits": source_commits(plan),
        "scope": plan["scope"],
        "tracked_files_checked": len(current),
        "scientific_reference_files_checked": scientific_count,
        "reader_documentation_files": documentation_count,
        "documentation_manifest_entries_checked": manifest_count,
        "local_links_checked": local_links,
        "source_paths_checked": reference_links,
        "tracked_file_bytes": sum((ROOT / p).stat().st_size for p in current),
        "model_fitting_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path, help="New directory for verification.json")
    mode.add_argument("--list-source-commits", action="store_true", help="List Git commits required for verification")
    args = parser.parse_args()
    try:
        plan = json.loads((ROOT / PLAN).read_text())
        if args.list_source_commits:
            print("\n".join(source_commits(plan)))
            return
        out = args.output_dir.resolve()
        require(not out.exists(), "Choose a new output directory; existing reports are not overwritten.")
        report = verify_repository(plan)
        out.mkdir(parents=True)
        (out / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    except (AssertionError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Verification failed: {error}\n")


if __name__ == "__main__":
    main()
