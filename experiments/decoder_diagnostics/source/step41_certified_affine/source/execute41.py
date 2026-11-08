"""Sequential execution, independent gates, and preserved failure logs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import subprocess,sys,time
from pathlib import Path
import common41 as c
def commands(p):
    rows=[('audit41.py',['preflight'])]
    for task in p['tasks']:
        for stage in ('f1','f2','f3'):
            for script,phase in [('run41.py','fit'),('audit41.py','fit'),('run41.py','validate'),('audit41.py','validate')]:rows.append((script,[phase,'--task',task,'--stage',stage]))
        rows += [('run41.py',['select','--task',task]),('audit41.py',['selection','--task',task]),('run41.py',['fit','--task',task,'--stage','final']),('audit41.py',['fit','--task',task,'--stage','final']),('audit41.py',['pretest','--task',task]),('run41.py',['test','--task',task]),('audit41.py',['postrun','--task',task]),('run41.py',['benchmark','--task',task]),('audit41.py',['benchmark','--task',task])]
    rows += [('run41.py',['aggregate']),('audit41.py',['aggregate'])]
    return rows
def main():
    p=c.read(c.ROOT/'protocol41.json');plan=commands(p);out=c.ROOT/'output';out.mkdir(exist_ok=True)
    c.save(out/'execution_started41.json',{'schema':'step41-controller-start-v1','utc':c.utc(),'protocol_seal_sha256':c.sha(c.ROOT/'protocol_seal41.json'),'commands':[[s,*args] for s,args in plan]})
    rows=[];start=time.perf_counter()
    for index,(script,args) in enumerate(plan):
        tag=f'{index:02d}_'+script[:-3]+'_'+'_'.join(a for a in args if not a.startswith('--'));stdout=out/'logs'/(tag+'.stdout.txt');stderr=out/'logs'/(tag+'.stderr.txt');stdout.parent.mkdir(exist_ok=True)
        t=time.perf_counter()
        with stdout.open('xb') as fo,stderr.open('xb') as fe:
            process=subprocess.run([sys.executable,str(c.ROOT/'source'/script),*args],stdout=fo,stderr=fe,cwd=c.ROOT,timeout=5400)
        row={'index':index,'command':[script,*args],'returncode':process.returncode,'seconds':time.perf_counter()-t,'stdout_path':c.relative(stdout),'stdout_sha256':c.sha(stdout),'stderr_path':c.relative(stderr),'stderr_sha256':c.sha(stderr)};rows.append(row)
        print(c.canonical_bytes(row).decode(),end='',flush=True)
        if process.returncode:
            c.save(out/'execution_failed41.json',{'status':'STOP_RETAIN_FAILURE','completed_commands':rows,'utc':c.utc()});print(stdout.read_text()[-5000:]);print(stderr.read_text()[-5000:]);raise SystemExit(process.returncode)
    c.save(out/'execution_completed41.json',{'schema':'step41-controller-completion-v1','status':'PASS','command_count':len(rows),'commands':rows,'wall_seconds':time.perf_counter()-start,'utc':c.utc(),'scientific_reruns':0})
if __name__=='__main__':main()
