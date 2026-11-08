"""Exercise audit adapters on a temporary synthetic training/validation tree.

Only deterministic constructed arrays are passed to producer training APIs.
The Seoul archive and all actual scientific cohorts remain unopened.
"""
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import audit38 as audit
import data38 as data
import run38 as run

ROOT=Path(__file__).resolve().parents[1]

def main():
    started=time.perf_counter();protocol=json.loads((ROOT/"protocol38.json").read_text())
    protocol["data"]["expected_n"]["train"]=64
    with tempfile.TemporaryDirectory(prefix="step38_artificial_adapter_") as directory:
        temporary=Path(directory);(temporary/"source").mkdir()
        for path in (ROOT/"source").glob("*.py"):shutil.copyfile(path,temporary/"source"/path.name)
        (temporary/"protocol_seal38.json").write_text('{"artificial_only":true}\n')
        run.ROOT=temporary;run.OUT=temporary/"output";data.ROOT=temporary;audit.ROOT=temporary
        cohorts={}
        for role,n in (("train",64),("validation",31)):
            offset=0 if role=="train" else 100
            X=np.asarray([[((i+offset+3)*(j+5)%97)+(i%7)*.125+(j%3)*.25 for j in range(24)] for i in range(n)],dtype=np.float64)
            ids=["artificial_"+role+":"+str(i) for i in range(n)]
            y=[F((i*13)%64,63) for i in range(n)];persistence=[F((i*7)%64,63) for i in range(n)]
            path=temporary/"output/cohorts"/(role+"_X.npy");path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,X,allow_pickle=False)
            record={"ids":ids,"y":list(map(str,y)),"n":n,"X_path":str(path.relative_to(temporary)),"X_sha256":data.sha_file(path),
                "persistence":list(map(str,persistence)),"artificial_only":True}
            data.write_json(temporary/"output/cohorts"/(role+".json"),record)
            cohorts[role]={**record,"record":record,"X":X,"y":y,"persistence":persistence,
                "summary":{"status":"PASS","artificial_only":True,"n":n}}
        normalization={"target_min":0,"target_max":63,"span":63,"artificial_only":True}
        data.write_json(temporary/"output/normalization38.json",normalization)
        cohorts["train"]["normalization"]=normalization;cohorts["validation"]["normalization"]=normalization
        data.load_training=lambda:cohorts["train"]
        data.load_validation=lambda:cohorts["validation"]
        run.train(protocol)
        training=audit.audit_training(protocol,cohorts["train"])
        run.validate(protocol)
        validation=audit.audit_validation(protocol,cohorts["validation"],audit.load_context())
        assert training["heads"]==90 and training["training_class_fits"]==36
        assert validation["original_label_losses_verified"]==93 and validation["selection_count"]==7
        source_names=["source/"+name for name in ("audit38.py","check_audit_integration38.py","run38.py","learners38.py","data38.py",
            "independent_math35.py","audit_math37.py","representations35.py","finite_feature35.py","full_graph37.py")]
        result={"schema":"step38-artificial-audit-adapter-integration-v1","status":"PASS","artificial_only":True,
            "scientific_count_values_decoded":0,"scientific_archive_opened":False,"synthetic_training_rows":64,"synthetic_validation_rows":31,
            "training_models":training,"validation_and_selection":validation,
            "source_sha256":{name:audit.sha(ROOT/name) for name in source_names},"wall_seconds":time.perf_counter()-started}
        data.write_json(ROOT/"tests/artificial_audit_integration38.json",result)
        print(json.dumps({"status":"PASS","synthetic_training_heads":90,"synthetic_validation_losses":93,
            "scientific_count_values_decoded":0,"seconds":result["wall_seconds"]}),flush=True)

if __name__=="__main__":main()
