#!/usr/bin/env python3
"""Reanalyze frozen 117,000 four-state counts; never resample trajectories.

Reproducible pipeline: validate ZIP inventory, archive member digests,
original count/result parity, original 95% certificate and all 13x9x1000
expected trial slots; compute four new/old certificates, summarize.

The original ZIP is read only. For every generated report record, all count
and design fields originate exclusively in the frozen samples/spec JSON.
"""
import argparse
import csv
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import io
import json
import math
import platform
from pathlib import Path
import sys
import tarfile
import tempfile
import zipfile

import numpy as np
from certificate_reanalysis import four_state_row

PFX='four_state_original_iid_final_v1_run001/'
ORIGINAL_SHA='4776942edbd659f7a1c02ec554507f7c0259ca410449e2e5fd66f5353bf638d5'
CONFIG_SHA='630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182'
ORIGINAL_INVENTORY_SHA='851025653f42a15bac4f2f17c2c1f792562aaccf461d94b8232ca5ac06023024'
METHODS=('label','centered','prototype','ERM_generic')


def sha(x):
    h=hashlib.sha256()
    if isinstance(x,(bytes,bytearray)):
        h.update(x)
    else:
        with open(x,'rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):
                h.update(chunk)
    return h.hexdigest()


def assert_sums(lines,files,description):
    for line in lines.splitlines():
        if not line.strip():continue
        hexd,name=line.split(maxsplit=1)
        name=name.lstrip('*')
        if sha(files(name)) != hexd:
            raise RuntimeError(f'{description} checksum mismatch for {name}')


def read_source(zip_path):
    """Check nested SHA manifests, extracting once to avoid repeated gzip seeks."""
    with zipfile.ZipFile(zip_path) as z:
        def zread(n):return z.read(PFX+n)
        assert_sums(zread('SHA256SUMS').decode(),zread,'outer ZIP')
        def sread(n):return z.read(PFX+'summary_v1/'+n)
        assert_sums(sread('SHA256SUMS').decode(),sread,'summary ZIP')
        meta=json.loads(zread('ARCHIVE.json'))
        raw=zread('evidence.tar.gz')
    if sha(raw)!=meta['archive_sha256'] or sha(raw)!=ORIGINAL_SHA:
        raise RuntimeError('evidence.tar.gz does not match frozen SHA')
    tmp=tempfile.TemporaryDirectory(prefix='kahkm_original_')
    root=Path(tmp.name)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        members=meta['members']
        if len(members)!=32 or len(t.getmembers())!=32:
            raise RuntimeError('unexpected archive membership')
        got={m.name for m in t.getmembers()}
        if got!={m['path'] for m in members}:
            raise RuntimeError('unexpected archive member path')
        # Manifest-based allowlist, regular files only, under fixed directory.
        for member in t.getmembers():
            if not member.isfile() or not (member.name.startswith(PFX) or member.name.startswith(PFX.rstrip('/')+'_numerical_verification001/')) or '..' in Path(member.name).parts:
                raise RuntimeError('invalid archive member path/type')
        t.extractall(root,filter='data')
    for item in meta['members']:
        f=root/item['path']
        if sha(f)!=item['sha256'] or f.stat().st_size!=item['bytes']:
            raise RuntimeError('member hash/size mismatch: '+item['path'])
    def fetch(n):return (root/PFX/n).read_bytes()
    if sha(fetch('campaign_spec.json'))!=CONFIG_SHA:
        raise RuntimeError('frozen configuration mismatch')
    if sha(fetch('SHA256SUMS'))!=ORIGINAL_INVENTORY_SHA:
        raise RuntimeError('frozen inventory mismatch')
    assert_sums(fetch('SHA256SUMS').decode(),lambda n:fetch(n),'inner TAR')
    return tmp,root/PFX,meta,json.loads(fetch('campaign_spec.json')),json.loads(fetch('campaign_metadata.json'))


def calc(src_zip,out_dir):
    tempdir,original,meta,spec,source_meta=read_source(src_zip)
    cases={c['case_key']:c for c in spec['cases']}
    by_id={c['case_id']:c for c in spec['cases']}
    sample_sizes=spec['sampling']['sample_sizes']
    repeats=spec['sampling']['replicates_per_cell']
    alpha=spec['evaluation']['delta']
    if alpha!=.05 or len(cases)!=13 or len(sample_sizes)!=9 or repeats!=1000:
        raise RuntimeError('unexpected experimental design')
    joint_alpha=alpha/len(METHODS)
    out_dir.mkdir(parents=True,exist_ok=True)
    # Archived JSONL arrays are sequential; verify ID equality in every line.
    f_samples=(original/'samples.jsonl').open('rb')
    f_results=(original/'results.jsonl').open('rb')
    paths={'pointwise':out_dir/'replicates_pointwise.csv','joint':out_dir/'replicates_joint.csv'}
    columns=['case_id','case_key','dynamics','design_role','M','replicate',
             'n_a','n_b','n_d','n_e','p','delta_per_method',
             's_hat','f_hat','t_hat','q_hat_fixed_prototype',
             'exact_population_min_RMSE','exact_empirical_min_RMSE',
             'L_label','L_centered','L_prototype','L_ERM_generic','L_class_exact']
    filers={q:open(path,'w',newline='') for q,path in paths.items()}
    writers={q:csv.DictWriter(f,columns) for q,f in filers.items()}
    for w in writers.values():w.writeheader()
    sums={q:defaultdict(lambda: defaultdict(list)) for q in paths}
    seen=set();n_samples=n_results=0
    max_err={k:0. for k in ('s_hat','f_hat','L_label')}
    counts=Counter()
    try:
        for b_s in f_samples:
            n_samples+=1
            b_r=f_results.readline()
            if not b_r:raise AssertionError('results shorter than samples')
            n_results+=1
            js=json.loads(b_s);jr=json.loads(b_r)
            if js['attempt_id']!=jr['attempt_id']:
                raise RuntimeError('sample/result ID mismatch')
            sample=js['sample'];case=by_id[jr['case_id']]
            if sample['case_key']!=case['case_key'] or sample['n_pairs'] not in sample_sizes:
                raise RuntimeError('case/sample-size metadata mismatch')
            M=int(sample['n_pairs']);repeat=int(sample['replicate_index'])
            if not 0<=repeat<repeats or sum(sample['state_counts'])!=M:
                raise RuntimeError('invalid replicate metadata/count vector')
            key=(case['case_id'],M,repeat)
            if key in seen:raise RuntimeError('duplicate trial '+str(key))
            seen.add(key); counts[(case['case_id'],M)]+=1
            if sample['state_counts']!=jr['count_statistics']['state_counts']:
                raise RuntimeError('count input conflict with original result')
            if sample['state_order']!=['a','b','d','e']:
                raise RuntimeError('unexpected state ordering')
            n=sample['state_counts']
            inp={'dataset_id':js['attempt_id'],'dynamics':case['dynamics'],
                 'p':str(case['p_exact']['numerator'])+'/'+str(case['p_exact']['denominator']),
                 'n_a':n[0], 'n_b':n[1], 'n_d':n[2], 'n_e':n[3]}
            run_point=four_state_row(inp,alpha)
            old=jr['certificate']
            for kk,new,oldval in [('s_hat',run_point['s_hat'],old['s_hat']),
                                  ('f_hat',run_point['f_hat'],old['f_hat']),
                                  ('L_label',run_point['L_label'],old['L_kappa_delta'])]:
                err=abs(new-oldval)
                max_err[kk]=max(max_err[kk],err)
                if err>2e-11:
                    raise RuntimeError(f'original certificate mismatch: {kk}, {key}, diff={err}')
            # Secondary reanalysis at two levels: pointwise and joint within methods.
            for scope,delta in [('pointwise',alpha),('joint',joint_alpha)]:
                row=run_point if scope=='pointwise' else four_state_row(inp,delta)
                record={'case_id':case['case_id'],'case_key':case['case_key'],
                        'dynamics':case['dynamics'],'design_role':case['design_role'],
                        'M':M,'replicate':repeat,'n_a':n[0], 'n_b':n[1], 'n_d':n[2],
                        'n_e':n[3], 'p':case['p'],'delta_per_method':delta}
                record.update({k:row[k] for k in columns if k in row})
                writers[scope].writerow(record)
                cell=(case['case_id'],M)
                for m in METHODS:
                    value=row['L_'+m]
                    sums[scope][cell][m].append(value)
                sums[scope][cell]['special_exact'].append(row['L_class_exact'])
                sums[scope][cell]['true'].append(row['exact_population_min_RMSE'])
                sums[scope][cell]['empirical_opt'].append(row['exact_empirical_min_RMSE'])
            if n_samples%40000==0:
                print(f'Processed {n_samples:,} original count/result pairs...',flush=True)
        if f_results.readline(): raise RuntimeError('results longer than samples')
    finally:
        for f in filers.values():f.close()
        f_samples.close()
        f_results.close()
    expected=len(cases)*len(sample_sizes)*repeats
    if n_samples!=expected or n_results!=expected or len(seen)!=expected:
        raise RuntimeError('archive does not have all expected evaluations')
    if len(counts)!=len(cases)*len(sample_sizes) or set(counts.values())!={repeats}:
        raise RuntimeError('cell replicate totals do not match frozen design')

    summaries={}
    for scope,percell in sums.items():
        out=out_dir/f'cell_summary_{scope}.csv'; summaries[scope]=out
        fields=['case_id','M','dynamics','p','true_min_RMSE','replicates','delta_per_method']
        for m in METHODS:
            fields.extend([f'{m}_mean',f'{m}_median',f'{m}_positive_n',f'{m}_excludes_0p20_n',f'{m}_coverage_fail_n'])
        fields.extend(['prototype_gt_erm_n','erm_gt_prototype_n','prototype_equals_erm_n',
                       'best_structural_gt_erm_n','erm_gt_best_structural_n',
                       'special_exact_mean'])
        with out.open('w',newline='') as f:
            wr=csv.DictWriter(f,fields);wr.writeheader()
            for case in spec['cases']:
                for M in sample_sizes:
                    d=percell[(case['case_id'],M)]
                    vals=d['true']; true=vals[0]
                    if not all(abs(x-true)<1e-12 for x in vals):raise RuntimeError('nonconstant true risk')
                    row={'case_id':case['case_id'],'M':M,'dynamics':case['dynamics'],
                         'p':case['p'],'true_min_RMSE':true,'replicates':len(vals),
                         'delta_per_method':alpha if scope=='pointwise' else joint_alpha}
                    for m in METHODS:
                        a=np.array(d[m])
                        row.update({f'{m}_mean':float(a.mean()),f'{m}_median':float(np.median(a)),
                             f'{m}_positive_n':int((a>0).sum()),
                             f'{m}_excludes_0p20_n':int((a>.20).sum()),
                             f'{m}_coverage_fail_n':int((a>true+1e-12).sum())})
                    p=np.array(d['prototype']);er=np.array(d['ERM_generic'])
                    best=np.maximum.reduce([np.array(d[m]) for m in ('label','centered','prototype')])
                    row.update(prototype_gt_erm_n=int((p>er+1e-12).sum()),
                         erm_gt_prototype_n=int((er>p+1e-12).sum()),
                         prototype_equals_erm_n=int((np.abs(er-p)<=1e-12).sum()),
                         best_structural_gt_erm_n=int((best>er+1e-12).sum()),
                         erm_gt_best_structural_n=int((er>best+1e-12).sum()),
                         special_exact_mean=float(np.mean(d['special_exact'])))
                    wr.writerow(row)
    with (out_dir/'cell_summary_joint.csv').open() as f:
        rows=list(csv.DictReader(f))
    with (out_dir/'cell_summary_pointwise.csv').open() as f:
        rows_pt=list(csv.DictReader(f))
    a={k:sum(int(row[k]) for row in rows) for k in [f'{m}_coverage_fail_n' for m in METHODS]}
    b={k:sum(int(row[k]) for row in rows_pt) for k in [f'{m}_coverage_fail_n' for m in METHODS]}
    stats={
        'original_archive_integrity_verified':True,
        'archive_original_run':spec['campaign_id'],
        'original_execution_commit_as_recorded':source_meta['source']['execution_source_commit'],
        'original_archive_sha256':ORIGINAL_SHA,
        'uploaded_zip_sha256':sha(src_zip),
        'config_sha256':CONFIG_SHA,
        'original_inventory_sha256':ORIGINAL_INVENTORY_SHA,
        'source_archive_records':n_samples,
        'original_certificate_recalculation_max_abs_error':max_err,
        'expected_and_observed_cells':len(counts),
        'design_replicates_per_cell':repeats,
        'confidence_scope':'pointwise per predetermined case,M,replicate; joint 95% over four methods via Bonferroni within one dataset; NOT across cases, M, or all replicates',
        'confidence_delta_total':alpha,
        'joint_correction':f'Bonferroni of {len(METHODS)} methods, delta={joint_alpha} each',
        'joint_coverage_failure_counts':a,
        'pointwise_coverage_failure_counts':b,
        'prototypes':'h1=(p,1-p), h2=(1-p,p) fixed from configuration; oracle design-information, not learned from evaluation',
        'class_exact':'Separate special case using population fact tau=0; excluded from four-method confidence allocation',
        'original_runs_unchanged':True,
        'new_simulations':0,
        'historical_commit_independently_authenticated_via_git':False,
        'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'program_sha256':sha(Path(__file__)),
        'formula_script_sha256':sha(Path(__file__).parent/'certificate_reanalysis.py'),
        'source_files':{
          'original_zip':str(src_zip.resolve()),
          'archive_member_hashes_validated':meta['archived_file_count'],
          'sample_member_sha256':next(m['sha256'] for m in meta['members'] if m['path']==PFX+'samples.jsonl'),
          'result_member_sha256':next(m['sha256'] for m in meta['members'] if m['path']==PFX+'results.jsonl'),
        },
        'outputs':{str(p.relative_to(out_dir)):sha(p) for p in [*paths.values(),*summaries.values()]}
    }
    (out_dir/'manifest.json').write_text(json.dumps(stats,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:stats[k] for k in ('source_archive_records','original_certificate_recalculation_max_abs_error','joint_coverage_failure_counts','pointwise_coverage_failure_counts')},indent=2))
    print('Saved summaries, replicate-level results, and manifest under',out_dir)
    tempdir.cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True,help='historical uploaded ZIP (read-only)')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    calc(args.archive,args.out)

if __name__=='__main__':main()
