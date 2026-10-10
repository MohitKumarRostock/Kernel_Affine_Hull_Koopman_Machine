#!/usr/bin/env python3
"""Repackage unchanged repository evidence for the historical reanalysis."""
import argparse
import hashlib
from pathlib import Path
import zipfile
CAMPAIGN="four_state_original_iid_final_v1_run001"
EXPECTED="4776942edbd659f7a1c02ec554507f7c0259ca410449e2e5fd66f5353bf638d5"
def run(root,output):
    d=root/"reproduction"/"prospective_campaigns"/CAMPAIGN
    for name in ("ARCHIVE.json","SHA256SUMS","evidence.tar.gz","summary_v1/SHA256SUMS"):
        if not (d/name).is_file(): raise FileNotFoundError(d/name)
    h=hashlib.sha256()
    with (d/"evidence.tar.gz").open("rb") as stream:
        for b in iter(lambda:stream.read(1<<20),b""):h.update(b)
    if h.hexdigest()!=EXPECTED:raise RuntimeError("historical evidence hash mismatch")
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(d.rglob("*")):
            if file.is_file():z.write(file,arcname=f"{CAMPAIGN}/{file.relative_to(d).as_posix()}")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--repo-root",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    v=a.parse_args()
    run(v.repo_root.resolve(),v.out.resolve())
