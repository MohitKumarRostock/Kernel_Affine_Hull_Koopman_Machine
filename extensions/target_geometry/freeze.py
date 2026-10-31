"""Create/verify the one-time scientific execution snapshot, before full runs."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
from .data import digest, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def scientific_inputs():
    files = set()
    for pattern in ("*.py", "config.json", "PROTOCOL.md", "THEORY.md", "requirements.txt",
                    "sources/*.json", "sources/README.md", "prepared/*"):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    # Asset rendering is downstream of audited results; narrative/rendering may change.
    files.discard(ROOT / "build_assets.py")
    for name in ("extensions/natural_forecasting/__init__.py", "extensions/natural_forecasting/data.py",
                 "extensions/natural_forecasting/models.py", "extensions/forecasting_utility/__init__.py",
                 "extensions/forecasting_utility/models.py", "extensions/predictive_geometry/study/__init__.py",
                 "extensions/predictive_geometry/study/datasets.py", "extensions/predictive_geometry/study/geometry.py",
                 "extensions/predictive_geometry/study/heads.py"):
        files.add(REPO / name)
    files.update((REPO / "extensions/predictive_geometry/study/vendor/upstream").rglob("*.py"))
    return sorted(p for p in files if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    args = parser.parse_args()
    destination = ROOT / "freeze.json"
    if args.command == "verify":
        frozen = json.loads(destination.read_text())
        for name, expected in frozen["files"].items():
            if digest(REPO / name) != expected:
                raise ValueError(f"Changed scientific input: {name}")
        print(json.dumps({"valid": True, "files": len(frozen["files"]), "sha256": digest(destination)}))
        return
    if destination.exists() or (ROOT / "results").exists():
        raise FileExistsError("Freeze cannot overwrite a snapshot or follow a full study run")
    manifest = {"schema_version": 1, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "files": {p.relative_to(REPO).as_posix(): digest(p) for p in scientific_inputs()},
                "scope": "Prospective scientific execution snapshot, not external preregistration. Public source checkpoint must precede full evaluation. Narrative and asset rendering are outside the freeze."}
    write_json(destination, manifest)
    print(json.dumps({"files": len(manifest["files"]), "sha256": digest(destination)}))


if __name__ == "__main__":
    main()
