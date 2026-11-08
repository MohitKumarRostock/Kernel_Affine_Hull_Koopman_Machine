"""Run the frozen Step39 phases once, stopping at any failed independent gate."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
from pathlib import Path
import subprocess
import sys
import time
import common39 as c

ROOT=c.ROOT

def main():
    protocol=c.read(ROOT/"protocol39.json")
    gate=c.read(ROOT/"audit/preexecution_gate39.json")
    if gate["status"]!="PASS" or gate["protocol_seal_sha256"]!=c.sha(ROOT/"protocol_seal39.json"):
        raise RuntimeError("Matching preexecution gate required")
    commands=[]
    for task in protocol["tasks"]:
        for stage in protocol["selection"]["folds"]:
            for source,phase in (("run39.py","fit"),("audit39.py","fit"),("run39.py","validate"),("audit39.py","validate")):
                commands.append([source,phase,"--task",task,"--stage",stage])
        commands.extend([["run39.py","select","--task",task],["audit39.py","selection","--task",task],
                         ["run39.py","fit","--task",task,"--stage","final"],
                         ["audit39.py","fit","--task",task,"--stage","final"],
                         ["audit39.py","pretest","--task",task],["run39.py","test","--task",task],
                         ["audit39.py","postrun","--task",task]])
    commands.extend([["run39.py","aggregate"],["audit39.py","aggregate"]])
    c.save(ROOT/"output/primary_execution39.json",{"schema":"step39-one-pass-execution-v1","started_utc":c.utc(),
        "protocol_seal_sha256":c.sha(ROOT/"protocol_seal39.json"),"commands":commands,"source_sha256":c.sha(Path(__file__))})
    logs=ROOT/"output/execution_logs";logs.mkdir(parents=True,exist_ok=True)
    for index,command in enumerate(commands):
        print(c.canonical_bytes({"event":"COMMAND_START","index":index,"total":len(commands),"command":command}).decode(),end="",flush=True)
        began=time.perf_counter();path=logs/(str(index).zfill(2)+"_"+command[0][:-3]+"_"+command[1]+".txt")
        with path.open("x") as log:
            proc=subprocess.Popen([sys.executable,str(ROOT/"source"/command[0]),*command[1:]],cwd=ROOT,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end="",flush=True)
            code=proc.wait()
        record={"event":"COMMAND_END","index":index,"exit_code":code,"seconds":time.perf_counter()-began,"log_sha256":c.sha(path)}
        print(c.canonical_bytes(record).decode(),end="",flush=True)
        if code:
            c.save(ROOT/"output/execution_stopped39.json",record)
            raise SystemExit(code)
    c.save(ROOT/"output/execution_completed39.json",{"schema":"step39-execution-completion-v1","status":"PASS",
        "completed_utc":c.utc(),"commands_completed":len(commands),"result_sha256":c.sha(ROOT/"output/result39.json"),
        "audit_sha256":c.sha(ROOT/"audit/postrun_audit39.json")})

if __name__=="__main__":main()
