#!/usr/bin/env python3
"""Exercise all phase seals on a tiny separate non-study fixture, without scoring.

The isolated fixture uses one law, two noise levels, tiny episode counts
and a distinct master seed. Its preflight marker is explicitly synthetic test
input to the seal API, not an independent review of the production study.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir',type=Path,default=ROOT/'verification/staged_chain_fixture')
    parser.add_argument('--receipt',type=Path,default=ROOT/'verification/staged_chain_tests.json')
    args=parser.parse_args();fixture=args.fixture_dir.resolve()
    if fixture.exists():raise FileExistsError(fixture)
    fixture.mkdir(parents=True)
    for directory in ('code','runtime','vendor_original'):
        shutil.copytree(ROOT/directory,fixture/directory,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    config=json.loads((ROOT/'config.json').read_text())
    config.update(study_id='Separate staged-chain fixture: no production inference',master_seed=91847300,
        repetitions=1,laws=['linear_ar'],nonlinear_laws=[],task_count=2,
        split_episodes={'A':8,'B':4,'C':4,'D':4},
        split_caps={'A':64,'B':32,'C':32,'D':None})
    config['seed_derivation']['distinct_seed_count']=14
    config['analysis']['primary_repetitions']=1
    (fixture/'readiness').mkdir()
    shutil.copy2(ROOT/'readiness/prior_seed_exclusions.json',fixture/'readiness/prior_seed_exclusions.json')
    (fixture/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (fixture/'PROTOCOL.md').write_text('# Separate construction fixture\n\nNo production data and no target scoring.\n')
    (fixture/'requirements.txt').write_text('numpy\nscipy\nscikit-learn\nthreadpoolctl\n')
    (fixture/'audit').mkdir()
    (fixture/'audit/prefreeze_review.json').write_text(json.dumps({
        'status':'passed','synthetic_seal_api_input':True,
        'scope':'Dummy review marker solely for testing seal preconditions in an isolated non-study fixture; not production review'})+'\n')
    logs=[];checks=0
    def run(script,arguments,expect_success=True):
        nonlocal checks
        command=[sys.executable,str(fixture/'code'/script),*arguments]
        started=time.perf_counter()
        process=subprocess.run(command,capture_output=True,text=True)
        logs.append({'script':script,'arguments':arguments,'returncode':process.returncode,
                     'seconds':time.perf_counter()-started,'stdout':process.stdout,'stderr':process.stderr})
        if expect_success and process.returncode:
            raise RuntimeError('Fixture command failed:\n'+process.stdout+process.stderr)
        if not expect_success and not process.returncode:
            raise AssertionError('A forbidden stage transition unexpectedly passed')
        checks+=1
    base=['--root',str(fixture)];freeze=str(fixture/'freeze.json');fitlock=str(fixture/'FIT_LOCK.json')
    run('study_seals.py',['freeze',*base])
    run('prepare_dimension_data.py',['d',*base,'--freeze-file',freeze,'--fit-lock',fitlock],False)
    assert not (fixture/'prepared/targets').exists();checks+=1
    run('prepare_dimension_data.py',['abc',*base,'--freeze-file',freeze])
    assert not (fixture/'prepared/d_manifest.json').exists();checks+=1
    run('study_seals.py',['fits',*base],False)
    run('run_dimension_stage.py',['--phase','fit',*base,'--freeze-file',freeze,'--workers','2'])
    run('study_seals.py',['fits',*base])
    run('prepare_dimension_data.py',['d',*base,'--freeze-file',freeze,'--fit-lock',fitlock])
    run('study_seals.py',['release',*base],False)
    run('run_dimension_stage.py',['--phase','predict',*base,'--freeze-file',freeze,'--fit-lock',fitlock,'--workers','2'])
    run('study_seals.py',['predictions',*base])
    run('study_seals.py',['release',*base])
    abc=json.loads((fixture/'prepared/abc_manifest.json').read_text())
    d=json.loads((fixture/'prepared/d_manifest.json').read_text())
    assert len(abc['tasks'])==len(d['tasks'])==2;checks+=1
    assert abc['D_episodes_generated'] is False;checks+=1
    assert d['generation_after_complete_fit_lock'] is True;checks+=1
    fit_record=json.loads((fixture/'FIT_LOCK.json').read_text())
    pred_record=json.loads((fixture/'PREDICTION_LOCK.json').read_text())
    release=json.loads((fixture/'TARGET_RELEASE.json').read_text())
    assert fit_record['created_utc']<=d['created_utc']<=pred_record['created_utc']<=release['created_utc'];checks+=1
    for task in abc['tasks']:
        summary=json.loads((fixture/'results'/task['task_id']/'fit_summary.json').read_text())
        assert summary['historical_native2_identity']['status']=='passed';checks+=1
        assert summary['reference_identity_across_candidates']['status']=='passed';checks+=1
        if task['history']==12:
            assert summary['native12_native20_identity']['status']=='passed';checks+=1
            assert all(s['selected_cap']!=20 for s in summary['selections'].values());checks+=1
    result={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Tiny separate seed/count fixture; no terminal targets loaded/scored by test or worker',
            'assertions':checks,'fixture_root':str(fixture),'tasks':2,'candidate_fits':14,
            'stage_commands':logs,'forbidden_transitions_rejected':[
                'D generation without fitlock','fitseal before allfits','targetrelease before predictionlock']}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    if args.receipt.exists():raise FileExistsError(args.receipt)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','assertions':checks,'fixture':str(fixture)}))


if __name__=='__main__':main()
