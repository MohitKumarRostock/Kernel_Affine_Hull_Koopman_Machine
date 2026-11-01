#!/usr/bin/env python3
"""Run one approved stage in fresh single-thread task processes; never seal it."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from study_contract import verify_freeze,verify_abc_manifest,verify_fit_lock,task_roster,digest
from dimension_worker import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('fit','predict'),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--freeze-file',type=Path,required=True)
    parser.add_argument('--fit-lock',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();root=args.root.resolve()
    if args.workers<1 or args.workers>3:raise ValueError('Declared worker range is1..3')
    verify_freeze(root,args.freeze_file)
    config=json.loads((root/'config.json').read_text())
    abc=verify_abc_manifest(root,args.freeze_file,hash_inputs=True)
    if args.phase=='fit':
        if (root/'prepared/d_manifest.json').exists():raise ValueError('Cannot begin fitting after D materialization')
        tasks=abc['tasks']
    else:
        if args.fit_lock is None:raise ValueError('Prediction requires complete fit lock')
        verify_fit_lock(root,args.fit_lock,args.freeze_file,full_hash=True)
        manifest=json.loads((root/'prepared/d_manifest.json').read_text())
        if manifest['fit_lock_sha256']!=digest(args.fit_lock):raise ValueError('D input fit-lock identity differs')
        tasks=manifest['tasks']
        expected={t['task_id'] for t in task_roster(config)}
        if {t['task_id'] for t in tasks}!=expected or len(tasks)!=len(expected):
            raise ValueError('Incomplete/duplicated D query roster')
        for task in tasks:
            if digest(root/task['query'])!=task['query_sha256']:raise ValueError('D query changed')
    cpus=sorted(os.sched_getaffinity(0))[:args.workers]
    if len(cpus)!=args.workers:raise ValueError('Insufficient distinct CPUs for declared worker count')
    logdir=root/'execution'/args.phase
    logdir.mkdir(parents=True,exist_ok=False)
    pending=queue.Queue()
    for task in tasks:pending.put(task)
    stop=threading.Event();lock=threading.Lock();completed=[];failures=[]
    start=time.perf_counter()
    def lane(cpu):
        while not stop.is_set():
            try:task=pending.get_nowait()
            except queue.Empty:return
            task_id=task['task_id']
            command=[sys.executable,str(root/'code/dimension_worker.py'),args.phase,
                     '--root',str(root),'--task-id',task_id,'--cpu',str(cpu),
                     '--freeze-file',str(args.freeze_file.resolve())]
            if args.fit_lock is not None:command+=['--fit-lock',str(args.fit_lock.resolve())]
            with (logdir/(task_id+'.log')).open('x') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
            with lock:
                if result.returncode:
                    failures.append({'task_id':task_id,'returncode':result.returncode});stop.set()
                else:completed.append(task_id)
                print(json.dumps({'phase':args.phase,'completed':len(completed),
                                  'total':len(tasks),'failed':len(failures)}),flush=True)
            pending.task_done()
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures=[pool.submit(lane,cpu) for cpu in cpus]
        for future in futures:future.result()
    receipt={'phase':args.phase,'status':'completed' if not failures and len(completed)==len(tasks) else 'failed',
        'completed_utc':datetime.now(timezone.utc).isoformat(),'task_count':len(tasks),
        'completed_task_ids':completed,'failures':failures,'wall_seconds':time.perf_counter()-start,
        'worker_cpus':cpus,'numerical_threads_per_worker':1,'fresh_process_per_task':True,
        'freeze_sha256':digest(args.freeze_file),
        'fit_lock_sha256':None if args.fit_lock is None else digest(args.fit_lock)}
    write_json(root/'execution'/(args.phase+'_stage.json'),receipt)
    if receipt['status']!='completed':raise RuntimeError('Stage stopped after an unexpected task failure')


if __name__=='__main__':main()
