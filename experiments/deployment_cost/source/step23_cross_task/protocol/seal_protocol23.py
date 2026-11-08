"""One-time seal of the completed prospective protocol, without benchmark use."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads((ROOT/p).read_text())


def main():
    out=ROOT/"protocol/protocol_seal23.json"
    if out.exists():
        raise FileExistsError("The prospective protocol seal is write-once")
    module_spec=importlib.util.spec_from_file_location("build_protocol23",ROOT/"protocol/build_protocol23.py")
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert read("PROTOCOL.json")==module.specification()
    assert read("protocol/candidate_templates23.json")==module.candidate_roster()
    protocol=read("PROTOCOL.json")
    assert protocol["budget"]["primary_total_six_tasks"]==3924
    assert protocol["known_prior_outcomes"]["ecg_data_access_allowed"] is False
    receipts=["theory/controlled_preflight23_v2.json",
              "tests/appliances_metadata_preflight23_v2.json",
              "tests/policy_preflight23_v2.json"]
    for rel in receipts:
        receipt=read(rel)
        assert receipt["passed"],rel
        for name,digest in receipt["source_sha256"].items():
            p=Path(name)
            if p.parts[0]==ROOT.name:p=Path(*p.parts[1:])
            assert sha(ROOT/p)==digest,(rel,name)
    controlled=read(receipts[0]);metadata=read(receipts[1]);policy=read(receipts[2])
    assert controlled["benchmark_trajectories_generated"]==controlled["model_fits"]==0
    assert metadata["source_csv_rows_parsed"]==metadata["benchmark_fits"]==0
    assert policy["model_fits"]==policy["real_dataset_rows_parsed"]==0
    for row in read("protocol/inherited_source_manifest23.json")["files"]:
        p=ROOT/row["package_path"]
        assert p.stat().st_size==row["bytes"] and sha(p)==row["sha256"]
    source=read("data/source_archive/source_lock23.json")
    archive=ROOT.parent/source["local_archive"]
    assert sha(archive)==source["archive_sha256"]
    review=(ROOT/"audit/design_review23.txt").read_text()
    assert "STATUS: PASSED" in review and sha(ROOT/"PROTOCOL.json") in review
    for p in ROOT.rglob("*.py"):
        if "__pycache__" not in p.parts:
            compile(p.read_text(),str(p),"exec")
    paths=[ROOT/"PROTOCOL.json",ROOT/"README23.txt",ROOT/"manuscript_protocol23.tex"]
    for folder in ["controlled","data","theory","tests","protocol","inherited_source","audit","reproducibility"]:
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix!=".pyc")
    files=[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)}
           for p in sorted(set(paths)) if p!=out]
    result={"schema":"step23-prospective-protocol-seal-v1","passed":True,
        "sealed_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":sha(ROOT/"PROTOCOL.json"),
        "candidate_count_per_task":654,"primary_fits_planned":3924,
        "benchmark_trajectories_generated":0,"source_csv_rows_parsed":0,
        "model_fits":0,"ECG_or_reserve_access":False,
        "public_preregistration":False,
        "scope":"Local prospective scientific specification and manufactured checks only. Future implementation, fitting and outcome seals remain distinct.",
        "active_receipts":receipts,"files":files}
    with out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="files"},indent=2))
    print("Sealed files:",len(files))


if __name__=="__main__":main()
