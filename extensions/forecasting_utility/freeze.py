"""Create the one-time execution manifest before looking at study outcomes.

Publishing the resulting Git commit is a separate, recorded step. This command
refuses to replace an existing freeze or run after an output directory exists.
Documentation and manuscript rendering may evolve after this scientific freeze.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import subprocess

from .data import digest, write_json

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def inputs():
    files = set()
    for pattern in ("*.py", "config.json", "PROTOCOL.md", "requirements.txt",
                    "sources.json", "source_metadata/*.json", "prepared/*"):
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    files.update(REPOSITORY / name for name in (
        "extensions/natural_forecasting/__init__.py",
        "extensions/natural_forecasting/data.py",
        "extensions/natural_forecasting/models.py",
        "extensions/natural_forecasting/baselines.py",
        "extensions/predictive_geometry/study/datasets.py",
        "extensions/predictive_geometry/study/geometry.py",
        "extensions/predictive_geometry/study/heads.py",
    ))
    files.update((REPOSITORY / "extensions/predictive_geometry/study/vendor/upstream").rglob("*.py"))
    return sorted(path for path in files if path.is_file())


def main():
    target = ROOT / "freeze.json"
    if target.exists():
        raise FileExistsError("The original freeze is immutable; document any amendment separately.")
    if (ROOT / "results").exists():
        raise RuntimeError("An output directory already exists; do not claim an outcome-blind freeze.")
    roster = json.loads((ROOT / "prepared/roster.json").read_text())
    current_configuration = digest(ROOT / "config.json")
    if roster["config_sha256"] != current_configuration:
        raise RuntimeError("The prepared roster has not been reconciled to the final configuration.")
    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "protocol_sha256": digest(ROOT / "PROTOCOL.md"),
        "roster_sha256": digest(ROOT / "prepared/roster.json"),
        "files": {str(path.relative_to(REPOSITORY)): digest(path) for path in inputs()},
        "scope": "Scientific execution and independent analysis, including inherited numerical implementations, configuration, source provenance and prepared inputs. Manuscript text, rendering and documentation are outside this freeze.",
        "status": "Execution freeze; publication must precede the first full study run. Not an external registration.",
    }
    write_json(target, manifest)
    print(json.dumps({"path": str(target), "sha256": digest(target), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
