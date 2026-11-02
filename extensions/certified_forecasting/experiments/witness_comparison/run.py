#!/usr/bin/env python3
"""Execute the frozen finite-law comparison; refuse existing output paths."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import scipy
from core import (ETA,SIZES,REPETITIONS,SEED,TOLERANCES,PRIMARY,METHODS,laws,population,
                  pair_summary,eb_lower,centered_from_counts,solve_lipschitz,
                  erm_kl_lower,stratified_from_summaries)

HERE=Path(__file__).resolve().parent
EXTENSION=HERE.parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_counts(sample,m):
    stop=2*(len(sample)//2)
    return np.bincount(m*sample[:stop:2]+sample[1:stop:2],minlength=m*m).reshape(m,m).tolist()


def stratum_statistics(law,sample,retain_matrices=False):
    _,_,_,labels=law.arrays()
    score=law.score_matrix()
    summaries=[];counts=[];matrices=[]
    for label in np.unique(labels):
        local=sample[labels[sample]==label]
        summaries.append(pair_summary(local,score))
        counts.append(len(local))
        if retain_matrices:
            matrices.append(pair_counts(local,len(law.u)))
    return summaries,counts,matrices


def trial(law,law_index,n,repetition):
    seed=[SEED,law_index,n,repetition]
    sample=np.random.default_rng(np.random.SeedSequence(seed)).choice(len(law.u),size=n,p=law.probabilities)
    # Timings start after the representation-valued sample is available.
    score=law.score_matrix()
    endpoints={};timings={};details={}
    start=time.perf_counter_ns()
    pair=pair_summary(sample,score)
    endpoints['pair_eb']=eb_lower(pair['mean'],pair['variance'],pair['n_pairs'],2.,ETA)
    timings['pair_eb']=(time.perf_counter_ns()-start)/1e9
    start=time.perf_counter_ns()
    support_pair=pair_summary(sample,score)
    endpoints['pair_support_eb']=eb_lower(support_pair['mean'],support_pair['variance'],support_pair['n_pairs'],float(score.max()),ETA)
    timings['pair_support_eb']=(time.perf_counter_ns()-start)/1e9
    start=time.perf_counter_ns()
    counts=np.bincount(sample,minlength=len(law.u))
    endpoints['centered_range'],details['centered_range']=centered_from_counts(law,counts)
    timings['centered_range']=(time.perf_counter_ns()-start)/1e9
    start=time.perf_counter_ns()
    counts=np.bincount(sample,minlength=len(law.u))
    qp=solve_lipschitz(law,counts/n)
    optimization_seconds=(time.perf_counter_ns()-start)/1e9
    start=time.perf_counter_ns()
    endpoints['erm_kl']=erm_kl_lower(qp['lower'],n)
    timings['erm_kl']=optimization_seconds+(time.perf_counter_ns()-start)/1e9
    start=time.perf_counter_ns()
    endpoints['erm_hoeffding']=max(0.,qp['lower']-4*math.sqrt(math.log(1/ETA)/(2*n)))
    timings['erm_hoeffding']=optimization_seconds+(time.perf_counter_ns()-start)/1e9
    for name,support,known in (('stratified_eb',False,False),
                               ('stratified_support_eb',True,False),
                               ('stratified_knownmass_support_eb',True,True)):
        start=time.perf_counter_ns()
        summaries,stratum_counts,_=stratum_statistics(law,sample)
        endpoints[name],details[name]=stratified_from_summaries(law,summaries,stratum_counts,n,
                                                              support=support,known_masses=known)
        timings[name]=(time.perf_counter_ns()-start)/1e9
    summaries,stratum_counts,stratum_matrices=stratum_statistics(law,sample,True)
    u,y,probabilities,labels=law.arrays()
    frequencies=counts/n
    sd_y=math.sqrt(float(frequencies@(y-frequencies@y)**2)*n/(n-1))
    sd_u=min(float(np.ptp(u))/2,math.sqrt(float(frequencies@(u-frequencies@u)**2)*n/(n-1)))
    strat_raw=float(np.asarray(stratum_counts)/n@np.asarray([s['mean'] for s in summaries]))
    known_raw=float(np.bincount(labels,weights=probabilities)@np.asarray([s['mean'] for s in summaries]))
    raw_statistics={'pair_eb':pair['mean'],'pair_support_eb':pair['mean'],
                    'centered_range':max(0.,sd_y-law.kappa*sd_u)**2,
                    'erm_kl':qp['lower'],'erm_hoeffding':qp['lower'],
                    'stratified_eb':strat_raw,'stratified_support_eb':strat_raw,
                    'stratified_knownmass_support_eb':known_raw}
    return {'law':law.name,'law_index':law_index,'n':n,'repetition':repetition,'seed':seed,
            'counts':counts.tolist(),'pair_counts':pair_counts(sample,len(law.u)),
            'stratum_counts':stratum_counts,'stratum_pair_counts':stratum_matrices,
            'pair_summary':pair,'stratum_summaries':summaries,'empirical_qp':qp,
            'lower':endpoints,'raw_statistics':raw_statistics,'seconds':timings,'shared_erm_optimization_seconds':optimization_seconds,
            'details':details}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    source_files=sorted(p for p in HERE.iterdir() if p.suffix in ('.py','.md'))
    source_hashes={str(p.relative_to(EXTENSION)):sha256(p) for p in source_files}
    try:
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXTENSION,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        git_head=None
    config={'seed_entropy':SEED,'calibration_sizes':list(SIZES),'repetitions':REPETITIONS,
            'eta_per_method':ETA,'joint_confidence_across_methods':False,
            'tolerances_rmse':list(TOLERANCES),'primary_methods':list(PRIMARY),
            'all_methods':list(METHODS),'laws':[asdict(law) for law in laws()]}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'configuration':config,
              'source_sha256':source_hashes,'base_git_commit':git_head,
              'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
                             'platform':platform.platform(),'processor':platform.processor(),
                             'machine':platform.machine(),'cpu_count':os.cpu_count(),
                             'thread_environment':{key:os.environ.get(key) for key in
                              ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},
              'confidence_scope':'95% separately for each one-sided method in each fixed law/size/trial; no joint 95% assertion across methods or trials.',
              'timing_scope':'Certificate computation after sample generation/representation lookup; ERM methods both include the same measured QP cost.'}
    # Manifest and protocol snapshot are committed to disk before any draw.
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'protocol.md').write_bytes((HERE/'protocol.md').read_bytes())
    populations=[population(law) for law in laws()]
    (output/'population.json').write_text(json.dumps(populations,indent=2)+'\n')
    count=0; max_gap=0.; fallbacks=0
    start=time.perf_counter()
    try:
        # mtime=0 avoids a timestamp in the compressed evidence bytes.
        with (output/'trials.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
                for li,law in enumerate(laws()):
                    for n in SIZES:
                        for repetition in range(REPETITIONS):
                            row=trial(law,li,n,repetition)
                            max_gap=max(max_gap,row['empirical_qp']['gap'])
                            fallbacks+=int(row['empirical_qp']['fallback'])
                            stream.write((json.dumps(row,separators=(',',':'))+'\n').encode())
                            count+=1
                        print(f'{law.name}: n={n}, completed {count} trials',flush=True)
    except Exception:
        (output/'failure.json').write_text(json.dumps({'completed_trials':count,'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
    current_hashes={str(p.relative_to(EXTENSION)):sha256(p) for p in source_files}
    if source_hashes!=current_hashes:
        raise RuntimeError('Source changed during benchmark; retained output is not a frozen run')
    summary={'status':'completed','n_trials':count,'n_bound_evaluations':count*len(METHODS),
             'elapsed_seconds':time.perf_counter()-start,'maximum_empirical_qp_gap':max_gap,
             'dual_fallback_count':fallbacks,'source_unchanged_during_run':True,
             'historical_known_zero_input_variance_expected_endpoint_lower':
              1-1/4096-4*math.sqrt(math.log(2/(.05/3))/(2*4096))}
    (output/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    hashes={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
