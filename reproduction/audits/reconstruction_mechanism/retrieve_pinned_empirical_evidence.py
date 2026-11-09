#!/usr/bin/env python3
"""Retrieve explicitly selected public evidence at one immutable repository SHA.

Downloads are data only. No retrieved source is imported or executed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "7dda356138675934aab5e327ec667bde0a5eda79"
BASE = "https://raw.githubusercontent.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/" + COMMIT + "/"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "remote_evidence"
TREE = DESTINATION / "mechanism_tree.json"


def selected_paths(tree):
    paths = [row["path"] for row in tree["tree"]
             if row["type"] == "blob" and row["path"].startswith("extensions/reconstruction_mechanism/analysis/")]
    paths += ["extensions/reconstruction_mechanism/" + name for name in (
        "README.md", "PROTOCOL.md", "config.json", "freeze.json", "PACKED_EVIDENCE.json",
        "prepared_inputs/manifest.json", "FINDINGS.md", "data.py", "models.py", "run.py",
        "analyze.py", "audit.py", "evidence.py", "package_results.py", "requirements.txt",
        "publication_validation/LOCAL_EXECUTION_PROVENANCE.json",
        "publication_validation/NUMERICAL_REPLAY_PORTABILITY.json",
        "publication_validation/README.md", "publication_validation/FINAL_VALIDATION.json")]
    paths += ["extensions/tmlr_integration/" + name for name in (
        "README.md", "verify_saved_forecasts.py", "package_submission.py", "package_tmlr_review.py",
        "review/KAHM_Reconstruction_Mechanism_RELEASE_MANIFEST.json",
        "review/KAHM_TMLR_Document_RELEASE_MANIFEST.json")]
    paths += ["extensions/direct_reconstruction/" + name for name in (
        "README.md", "PROTOCOL.md", "PRE_FREEZE_DESIGN_LOG.md", "FINDINGS.md", "PUBLICATION.json",
        "results/freeze.json", "results/execution.json")]
    return sorted(set(paths))


def fetch(path, expected):
    target = DESTINATION / path
    if target.exists():
        data = target.read_bytes()
    else:
        request = urllib.request.Request(BASE + path, headers={"User-Agent": "KAHM-manuscript-evidence-audit"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if git_hash != expected["sha"]:
        raise RuntimeError("Git blob hash mismatch: " + path)
    if len(data) != expected["size"]:
        raise RuntimeError("Byte count mismatch: " + path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)
    return {"path": path, "url": BASE + path, "bytes": len(data),
            "git_blob_sha1": git_hash, "sha256": hashlib.sha256(data).hexdigest()}


def main():
    global DESTINATION, TREE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", action="store_true", help="Also fetch the complete archived evidence parts")
    parser.add_argument("--tree", type=Path, default=TREE, help="Pinned Git tree JSON for the recorded source commit")
    parser.add_argument("--destination", type=Path, default=DESTINATION, help="Destination root for the pinned repository paths")
    args = parser.parse_args()
    DESTINATION, TREE = args.destination.resolve(), args.tree.resolve()
    tree = json.loads(TREE.read_text())
    lookup = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    paths = selected_paths(tree)
    if args.parts:
        paths += sorted(path for path in lookup if path.startswith("extensions/reconstruction_mechanism/evidence_parts/"))
    records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        pending = {executor.submit(fetch, path, lookup[path]): path for path in paths}
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            print("Verified", record["path"], record["bytes"], flush=True)
    manifest = {"commit": COMMIT, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Public immutable-source retrieval; downloaded source was not executed.",
                "file_count": len(records), "total_bytes": sum(row["bytes"] for row in records),
                "files": sorted(records, key=lambda row: row["path"])}
    output = DESTINATION / ("empirical_parts_download_manifest.json" if args.parts else "empirical_download_manifest.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Manifest", output, "files", len(records), "bytes", manifest["total_bytes"], flush=True)


if __name__ == "__main__":
    main()
