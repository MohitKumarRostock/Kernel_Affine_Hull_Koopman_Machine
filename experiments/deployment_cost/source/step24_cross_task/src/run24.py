"""Command line lifecycle for the sealed Step24 experimental panel."""
from __future__ import annotations
import argparse
import importlib.metadata
import platform
import subprocess
import sys
import time
from common24 import *


def seal_implementation():
    import data24 as D
    out=RUN/'implementation_seal.json'
    if out.exists():check_implementation_seal();return
    active=['data_preflight24.json','models_preflight24.json','geometry_preflight24_v2.json',
            'oracle_preflight24.json','integration_preflight24.json','evaluation_preflight24.json','timing_preflight24.json']
    for name in active:
        receipt=read_json(ROOT/'tests'/name)
        assert receipt['passed'],name
        for rel,digest in receipt['source_sha256'].items():
            p=(BASE/rel if rel.startswith(('step23_cross_task/','step24_cross_task/')) else ROOT/rel).resolve()
            assert sha(p)==digest,(name,rel)
    assert len(templates())==654
    for p in (ROOT/'src').glob('*.py'):compile(p.read_text(),str(p),'exec')
    sources=source_map()
    assert all(sources.get(k)==v for k,v in D.source_bindings().items())
    assert sha(D.ARCHIVE_PATH)==D.ARCHIVE_SHA256
    data={
        'schema':'step24-implementation-seal-v1','passed':True,'sealed_utc':now(),
        'protocol_sha256':sha(PARENT/'PROTOCOL.json'),'source_sha256':sources,
        'archive_sha256':D.ARCHIVE_SHA256,'scientific_menu_changes':False,
        'manufactured_receipts':{name:sha(ROOT/'tests'/name) for name in active},
        'environment':{'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),
                       'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','threadpoolctl']}},
        'benchmark_trajectories_generated':0,'benchmark_model_fits':0,'source_csv_rows_parsed':0,
        'ECG_or_reserve_access':False}
    write_json(out,data);log('all','implementation_sealed',sources=len(sources),manufactured_receipts=len(active))


def fit_panel():
    check_implementation_seal()
    logdir=RUN/'logs';logdir.mkdir(parents=True,exist_ok=True)
    queue=TASKS.copy();active={};completed=[];start=time.perf_counter()
    while queue or active:
        while queue and len(active)<3:
            task=queue.pop(0);f=(logdir/f'fit_{task}.log').open('a')
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'fit','--task',task],
                                  cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            active[task]=(proc,f);log(task,'worker_started',pid=proc.pid)
        for task,(proc,f) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            f.close();del active[task]
            if code:
                for other,(p,stream) in active.items():p.terminate()
                for other,(p,stream) in active.items():p.wait();stream.close()
                raise RuntimeError(f'{task} failed; remaining workers stopped. See run/logs/fit_{task}.log')
            completed.append(task);log(task,'worker_completed',elapsed_seconds=round(time.perf_counter()-start,2))
        if active:time.sleep(1)
    from fit24 import freeze_all
    freeze_all();log('all','development_panel_complete',tasks=completed,wall_seconds=time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['seal','fit','fit-panel','freeze','evaluate','time','finalize'])
    parser.add_argument('--task',choices=TASKS)
    args=parser.parse_args()
    try:
        if args.stage=='seal':seal_implementation()
        elif args.stage=='fit-panel':fit_panel()
        elif args.stage=='freeze':
            from fit24 import freeze_all
            freeze_all()
        else:
            if args.task is None:raise ValueError('Task required for this stage')
            if args.stage=='fit':
                from fit24 import fit_task
                fit_task(args.task)
            else:
                from evaluate24 import evaluate_task,time_task,finalize_task
                {'evaluate':evaluate_task,'time':time_task,'finalize':finalize_task}[args.stage](args.task)
    except Exception as exc:
        fail_record(args.task,args.stage,exc)
        raise


if __name__=='__main__':main()
