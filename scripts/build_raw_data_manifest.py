#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from raw_data_capture import sha256_file, verify_raw_dataset

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--raw-dir", type=Path, default=Path("reproduction/raw_data"))
    p.add_argument("--require-nonempty", action="store_true")
    p.add_argument("--require-clean-captures", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    raw = a.raw_dir if a.raw_dir.is_absolute() else root / a.raw_dir
    raw = raw.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()

    for meta_path in sorted(raw.glob("*.json")):
        if meta_path.name == "manifest.json":
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dataset_id = str(meta["dataset_id"])
        if dataset_id in seen:
            raise RuntimeError(f"Duplicate dataset_id: {dataset_id}")
        seen.add(dataset_id)
        npz_path = raw / f"{dataset_id}.npz"
        if not npz_path.is_file():
            raise RuntimeError(f"Missing archive: {npz_path}")
        verify_raw_dataset(npz_path, meta_path)
        git = meta.get("git", {})
        records.append({
            "dataset_id": dataset_id,
            "role": meta.get("role"),
            "source_script": meta.get("source_script"),
            "generator_seed": meta.get("generator_seed"),
            "content_sha256": meta["content_sha256"],
            "archive_sha256": meta["archive_sha256"],
            "metadata_sha256": sha256_file(meta_path),
            "git_commit": git.get("commit"),
            "tree_dirty_before_capture": bool(git.get("tree_dirty")),
            "archive_path": str(npz_path),
            "metadata_path": str(meta_path),
        })

    if a.require_nonempty and not records:
        raise RuntimeError("No captured raw datasets found.")

    dirty = [r["dataset_id"] for r in records if r["tree_dirty_before_capture"]]
    if a.require_clean_captures and dirty:
        raise RuntimeError("Dirty-tree captures: " + ", ".join(dirty))

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True
    ).stdout.strip()
    manifest = {
        "manifest_version": 1,
        "generated_from_git_commit": head,
        "dataset_count": len(records),
        "all_datasets_verified": True,
        "all_captures_started_clean": not dirty,
        "dirty_capture_dataset_ids": dirty,
        "capture_git_commits": sorted({
            str(r["git_commit"]) for r in records if r["git_commit"]
        }),
        "datasets": records,
    }
    out = raw / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Verified datasets: {len(records)}")
    print("All captures started clean:", manifest["all_captures_started_clean"])

if __name__ == "__main__":
    main()
