#!/usr/bin/env python3
"""Verify the stored numerical results for the forecasting experiments."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time

ROOT=Path(__file__).resolve().parent
GROUPS=('controlled_calibration','ecg_cohort','class_diagnosis','deployment_cost','public_screening','decoder_diagnostics','affine_forecasting')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def verify_manifest():
    path=ROOT/'MANIFEST.json';manifest=json.loads(path.read_text())
    for row in manifest['files']:
        p=(ROOT/row['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            raise RuntimeError('File integrity check failed: '+row['path'])
    return {'sha256':digest(path),'files':len(manifest['files'])}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only',choices=GROUPS,nargs='+')
    ap.add_argument('--output',type=Path,default=ROOT/'reproduction')
    args=ap.parse_args();output=args.output.resolve()
    if output==ROOT or any(output.is_relative_to(ROOT/p) for p in ['docs','experiments','scripts','validation']):
        ap.error('Choose a separate reproduction output directory.')
    output.mkdir(parents=True,exist_ok=True)
    integrity=verify_manifest();groups=args.only or GROUPS
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:env[name]='1'
    report={'status':'RUNNING','source_manifest':integrity,'python':platform.python_version(),'platform':platform.platform(),
      'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','threadpoolctl','mpmath']},'groups':{}}
    summary=output/'summary.json'
    for group in groups:
        target=output/group if group=='class_diagnosis' else output/(group+'.json')
        result=target/'summary.json' if group=='class_diagnosis' else target
        log=output/(group+'.log')
        command=[sys.executable,str(ROOT/'experiments'/group/'replay.py'),'--output',str(target)]
        if group=='deployment_cost':command.append('--predict')
        started=time.perf_counter()
        with log.open('w') as stream:
            done=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record=json.loads(result.read_text()) if result.is_file() else {}
        passed=done.returncode==0 and record.get('status')=='PASS'
        report['groups'][group]={'status':'PASS' if passed else 'FAIL','return_code':done.returncode,'wall_seconds':time.perf_counter()-started,
            'command':['python',str((ROOT/'experiments'/group/'replay.py').relative_to(ROOT)),'--output',str(target.relative_to(output))]+(['--predict'] if group=='deployment_cost' else []),
            'result_file':str(result.relative_to(output)),'result_sha256':digest(result) if result.is_file() else None,'log_file':log.name}
        summary.write_text(json.dumps(report,indent=2)+'\n')
        print(group+': '+report['groups'][group]['status'],flush=True)
        if not passed:
            report['status']='FAIL';summary.write_text(json.dumps(report,indent=2)+'\n')
            raise SystemExit('Reproduction stopped; inspect '+str(log))
    if verify_manifest()!=integrity:
        raise RuntimeError('The source manifest changed during reproduction.')
    report['source_integrity_rechecked']=True
    report['status']='PASS';report['scope']='Numerical replay of fixed records; no full training rerun or new scientific timing campaign.'
    summary.write_text(json.dumps(report,indent=2)+'\n')
    print('All requested numerical checks passed. Summary: '+str(summary))

if __name__=='__main__':main()
