#!/usr/bin/env python3
"""Verify child-local lock checks and target-free replay of a preserved fixture."""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import study_contract as contract

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root',type=Path,default=ROOT/'verification/staged_chain_fixture')
    parser.add_argument('--output',type=Path,default=ROOT/'verification/child_lock_optimization')
    parser.add_argument('--receipt',type=Path,default=ROOT/'verification/child_lock_optimization.json')
    args=parser.parse_args();fixture=args.fixture_root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    freeze=fixture/'freeze.json';fitlock=fixture/'FIT_LOCK.json'
    contract.verify_freeze(fixture,freeze)
    contract.verify_fit_lock(fixture,fitlock,freeze,full_hash=True)
    manifest=json.loads((fixture/'prepared/d_manifest.json').read_text());task=manifest['tasks'][0]
    seen=[];original_digest=contract.digest
    def track(path):
        seen.append(Path(path).resolve());return original_digest(path)
    with patch.object(contract,'digest',side_effect=track):
        contract.verify_fit_lock(fixture,fitlock,freeze,full_hash=False,task_id=task['task_id'])
    summaries=[p for p in seen if p.name=='fit_summary.json']
    assert summaries==[fixture/'results'/task['task_id']/'fit_summary.json']
    bad=json.loads(fitlock.read_text())
    another=manifest['tasks'][1]['task_id']
    del bad['files'][f'results/{another}/states/native20.pkl.gz']
    badpath=out/'missing_other_task_state_lock.json';badpath.write_text(json.dumps(bad))
    try:contract.verify_fit_lock(fixture,badpath,freeze,full_hash=False,task_id=task['task_id'])
    except ValueError as error:assert 'Canonical fitted artifacts missing' in str(error)
    else:raise AssertionError('Child failed to check complete canonical artifact membership')
    replay=out/'prediction'
    command=[sys.executable,'-c',
        'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
        'from dimension_worker import predict_task,sha256_file; '
        'predict_task(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]),fit_lock_sha256=sha256_file(Path(sys.argv[5])))',
        str(ROOT/'code'),str(fixture/'results'/task['task_id']),str(fixture/task['query']),str(replay),str(fitlock)]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    with np.load(replay/'predictions.npz') as new,np.load(fixture/'predictions'/task['task_id']/'predictions.npz') as old:
        assert set(new.files)==set(old.files)
        for key in new.files:np.testing.assert_array_equal(new[key],old[key])
        compared=len(new.files)
    record={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Target-free replay of one preserved non-study fixture; no new fits or outcomes read',
        'full_parent_lock_verified':True,'child_parsed_only_own_summary':True,
        'child_checks_other_tasks_canonical_membership':True,'exact_prediction_arrays':compared,
        'fresh_process_replay':True,'fixture_root':str(fixture),
        'code_sha256':{name:original_digest(ROOT/'code'/name) for name in
                       ('study_contract.py','dimension_worker.py','focused_dimension_model.py')}}
    if args.receipt.exists():raise FileExistsError(args.receipt)
    args.receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'passed','exact_prediction_arrays':compared}))


if __name__=='__main__':main()
