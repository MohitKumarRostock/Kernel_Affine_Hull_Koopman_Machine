"""Create a private, self-contained development-evidence archive.

This packaging utility does not fit, analyze, or modify experimental evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not args.report.is_file():
        raise FileNotFoundError(args.report)
    audit=json.loads((HERE/"analysis"/"audit.json").read_text())
    decision=json.loads((HERE/"analysis"/"decision.json").read_text())
    paths={str(p.relative_to(ROOT)):p for p in HERE.rglob("*") if p.is_file()
           and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".zip")}
    for path in (HERE/"archives").glob("*.zip"):
        paths[str(path.relative_to(ROOT))]=path
    for filename in ("LICENSE", "DATA_LICENSE"):
        paths[filename]=ROOT/filename
    source=ROOT/"extensions"/"direct_reconstruction"/"data"
    for filename in ("dominick_weekly__01.npz","dominick_weekly__02.npz","dominick_weekly__03.npz",
                     "sunspot_daily__01.npz","sources_manifest.json","README.md"):
        path=source/filename
        paths[str(path.relative_to(ROOT))]=path
    paths[args.report.name]=args.report
    release={"created_at":datetime.now(timezone.utc).isoformat(),
             "parent_commit":"88e9c568ee983856804a2b2da41f6290f259b8cd",
             "original_prospective_local_commit":"3d374de47d694f6ae3267e4b6ac82787755eeca5",
             "original_prospective_freeze_sha256":"8b5b77100b285fff3a6cb81af3b1e3b13081771255fe898e2d6d39da6ab0473f",
             "amended_freeze_sha256":hashlib.sha256((HERE/"DEVELOPMENT_FREEZE.json").read_bytes()).hexdigest(),
             "amendment_status":"A1 numerical repair after original outcomes were observed; no new confirmatory evidence.",
             "complete_attempts":2,
             "decision":decision["decision"],"completed_runs":decision["completed_runs"],
             "scored_forecasts":decision["scored_forecasts"],"audit_status":audit.get("status"),
             "publication_status":"Private study deliverable; no public push performed.",
             "scope":"Focused development pilot, not independent confirmation or a revised full manuscript."}
    try:
        release["local_release_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        release["local_release_commit"]=None
    readme="""# KAHM focused development study

Start with `KAHM_Adaptive_Development_Report.pdf` and
`extensions/adaptive_development/FINDINGS.md` for the completed decision.

This archive includes the full frozen protocol, forecasting implementation,
prepared inputs, optional original-input regeneration files, all 144 runs,
issued forecasts, timing/capacity traces, analysis tables, decision, and
independent computational audit. The complete first attempt is preserved in
`extensions/adaptive_development/archives/initial_attempt.zip`; the current
evidence is the full repeat after documented numerical amendment A1. It also
includes a proposed LaTeX subsection
for a clearly labelled exploratory appendix; it does not replace the full
manuscript or claim that new-source confirmation has been completed.

The optional LaTeX subsection is `extensions/adaptive_development/manuscript_addendum.tex`.
Its companion `manuscript_addendum_standalone.tex` provides a minimal wrapper.
From that extension directory, run `pdflatex manuscript_addendum_standalone.tex`
twice to compile it independently. The supplied pair was compiled and visually
checked; it uses standard LaTeX commands and the geometry package.

## Reproduce or inspect

Use the extracted directory containing this README as your working directory.
Install `extensions/adaptive_development/requirements.txt` in a suitable Python
environment. Then follow the commands in that extension's README. Choose new
output directories for refitting so the supplied evidence is preserved.

The saved analysis and audit can be recomputed without fitting models. The
optional report renderer additionally uses matplotlib 3.10.8 and reportlab
4.4.9 and, when available, the system DejaVu Sans fonts:

```
python -m extensions.adaptive_development.analyze --output /tmp/kahm_analysis
python -m extensions.adaptive_development.audit --analysis /tmp/kahm_analysis --output /tmp/kahm_analysis/audit.json
python -m extensions.adaptive_development.reporting --output /tmp/KAHM_Adaptive_Development_Report.pdf
```

`PACKAGE_INVENTORY.json` records each included file's byte size and SHA-256.
The inventory excludes itself. `RELEASE.json` records the frozen local source
checkpoints and the final release identity. The freezes were local;
they were not public preregistration. The second freeze followed inspection of
the first attempt, as AMENDMENT_A1.md explains. Reporting and packaging utilities are
post-freeze presentation work and do not alter the fixed experiment or gates.

## Attribution

Code follows the repository MIT LICENSE (Mohit Kumar, 2026). Research data and
non-code outputs follow DATA_LICENSE, subject to its third-party exclusions.
Natural inputs retain the Monash/Zenodo CC BY 4.0 attribution and transformation
notices in the two included provenance documents and source manifest. No
third-party dependency code is bundled. AI tools assisted the implementation,
checks and reporting; the audit is computational verification, not independent
external replication.
"""
    extra={"README.md":readme.encode(),"RELEASE.json":(json.dumps(release,indent=2,sort_keys=True)+"\n").encode()}
    inventory={}
    prefix="KAHM_Adaptive_Development_Study/"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(paths.items()):
            data=path.read_bytes()
            inventory[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
            z.writestr(prefix+name,data)
        for name,data in extra.items():
            inventory[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
            z.writestr(prefix+name,data)
        z.writestr(prefix+"PACKAGE_INVENTORY.json",json.dumps(inventory,indent=2,sort_keys=True)+"\n")
    with zipfile.ZipFile(args.output) as z:
        if z.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        for name,expected in inventory.items():
            data=z.read(prefix+name)
            assert len(data)==expected["bytes"]
            assert hashlib.sha256(data).hexdigest()==expected["sha256"]
    print(json.dumps({"archive":str(args.output),"bytes":args.output.stat().st_size,
                      "files":len(inventory)+1,"integrity":"passed","release":release},indent=2))


if __name__=="__main__":
    main()
