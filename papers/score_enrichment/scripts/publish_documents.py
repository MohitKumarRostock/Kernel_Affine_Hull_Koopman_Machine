#!/usr/bin/env python3
"""Verify approved inputs and publish a branch-scoped document build.

Only main.pdf, supplement.pdf, and DOCUMENT_BUILD.json may be committed.
The build does not execute, modify, or verify numerical forecasting experiments.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

REPO = 'MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine'
BRANCH = 'paper/geometry-induced-soft-state-enrichment-2026-11-02'
PREFIX = 'papers/score_enrichment/'
ROOT = Path(__file__).resolve().parents[1]

def verify_inputs() -> None:
    manifest = json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    if len(manifest['files']) != 41:
        raise RuntimeError('Unexpected approved-input count.')
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Invalid manifest path.')
        data = (ROOT/relative).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Approved input mismatch: {name}')
    print('Verified all 41 approved document-build inputs.')

def describe_outputs() -> dict:
    outputs = {}
    for name, pages in (('main.pdf',12),('supplement.pdf',14)):
        path = ROOT/name
        if not path.is_file():
            raise RuntimeError(f'Missing compiled document: {name}')
        metadata = subprocess.check_output(['pdfinfo',str(path)],text=True)
        match = re.search(r'^Pages:\s+(\d+)', metadata, flags=re.MULTILINE)
        if not match or int(match.group(1)) != pages:
            raise RuntimeError(f'Unexpected page count for {name}')
        if 'Predictive Enrichment of Geometry-Induced Soft-State Abstractions' not in metadata:
            raise RuntimeError(f'Unexpected PDF title for {name}')
        data = path.read_bytes()
        outputs[name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'pages':pages}
    return outputs

def publish() -> None:
    if os.environ.get('GITHUB_REPOSITORY') != REPO or os.environ.get('GITHUB_REF_NAME') != BRANCH:
        raise RuntimeError('Publishing is restricted to the named repository and publication branch.')
    source = os.environ.get('GITHUB_SHA','')
    token = os.environ.get('GITHUB_TOKEN','')
    if not re.fullmatch(r'[0-9a-f]{40}',source) or not token:
        raise RuntimeError('An exact source commit and GitHub workflow token are required.')
    base_url = 'https://api.github.com/repos/'+REPO
    def request(method: str, endpoint: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(base_url+endpoint, data=data, method=method,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                     'Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28'})
        with urllib.request.urlopen(req,timeout=60) as response:
            return json.load(response)
    ref = '/git/ref/heads/'+BRANCH
    if request('GET',ref)['object']['sha'] != source:
        raise RuntimeError('The publication branch advanced; no stale build will be committed.')
    outputs = describe_outputs()
    receipt = {'schema':'score-enrichment-document-build-v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_commit':source,
        'workflow_run_id':os.environ.get('GITHUB_RUN_ID'),
        'scope':'Document compilation only; no model fit, saved-loss analysis, or numerical audit.',
        'outputs':outputs,
        'compiler':subprocess.check_output(['pdflatex','--version'],text=True).splitlines()[0],
        'input_manifest_sha256':hashlib.sha256((ROOT/'PUBLICATION_MANIFEST.json').read_bytes()).hexdigest()}
    data_files = {name:(ROOT/name).read_bytes() for name in outputs}
    data_files['DOCUMENT_BUILD.json'] = (json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()
    elements = []
    for name,data in data_files.items():
        blob = request('POST','/git/blobs',{'content':base64.b64encode(data).decode('ascii'),'encoding':'base64'})
        elements.append({'path':PREFIX+name,'mode':'100644','type':'blob','sha':blob['sha']})
    parent = request('GET','/git/commits/'+source)
    tree = request('POST','/git/trees',{'base_tree':parent['tree']['sha'],'tree':elements})
    commit = request('POST','/git/commits',{'message':'Build score-enrichment article and supplement from '+source,
                                        'tree':tree['sha'],'parents':[source]})
    if request('GET',ref)['object']['sha'] != source:
        raise RuntimeError('Concurrent branch change; the generated commit is left unreferenced.')
    request('PATCH','/git/refs/heads/'+BRANCH,{'sha':commit['sha'],'force':False})
    print(json.dumps({'published_commit':commit['sha'],'source_commit':source,'files':outputs},indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-inputs',action='store_true')
    group.add_argument('--publish',action='store_true')
    args = parser.parse_args()
    if args.verify_inputs:
        verify_inputs()
    else:
        publish()

if __name__ == '__main__':
    main()
