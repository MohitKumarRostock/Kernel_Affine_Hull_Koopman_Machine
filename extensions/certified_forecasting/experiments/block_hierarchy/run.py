"""Run the prospectively frozen centered-block certificate comparison."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/witness_comparison'))
from certified_forecasting.block_witness import block_score_bound
from core import laws, solve_lipschitz, erm_kl_lower

SIZES = (256, 1024, 4096)
BLOCKS = (2, 4, 8, 16, 32)
REPETITIONS = 100
SEED = 202610291901
ALPHA = .05/(len(SIZES)*(len(BLOCKS)+1))
SOURCES = ('experiments/block_hierarchy/protocol.md',
           'experiments/block_hierarchy/run.py',
           'experiments/witness_comparison/core.py',
           'src/certified_forecasting/block_witness.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {p:digest(ROOT/p) for p in SOURCES}
    freeze = {'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'seed':SEED,'sample_sizes':SIZES,'block_sizes':BLOCKS,
              'repetitions':REPETITIONS,'alpha_per_endpoint':ALPHA,
              'joint_delta_per_law_repetition':.05,'source_sha256':source_hashes,
              'laws':[law.__dict__ for law in laws()]}
    with tarfile.open(output/'source_snapshot.tar.gz','w:gz') as archive:
        for path in SOURCES:
            archive.add(ROOT/path, arcname=path)
    freeze['snapshot_sha256']=digest(output/'source_snapshot.tar.gz')
    (output/'protocol_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    (output/'protocol.md').write_text((ROOT/SOURCES[0]).read_text())
    print('Protocol frozen before sampling: '+digest(output/'protocol_freeze.json'), flush=True)
    started = time.perf_counter()
    rows, population = [], []
    cache_count = 0
    maximum_gap = 0.
    failures = 0
    sequences = np.random.SeedSequence(SEED).spawn(len(laws())*REPETITIONS)
    with gzip.open(output/'samples.jsonl.gz','wt') as samples_file, gzip.open(output/'qp_certificates.jsonl.gz','wt') as qp_file:
        for law_index, law in enumerate(laws()):
            u,y,p,_ = law.arrays()
            oracle = solve_lipschitz(law,p)
            population.append({'law':law.name, 'pair_witness':float(p@law.score_matrix()@p),
                               'optimum_lower':oracle['lower'],'optimum_upper':oracle['upper'],
                               'qp':oracle})
            cache = {}
            score_matrix = law.score_matrix()
            def lookup(counts):
                nonlocal maximum_gap,cache_count
                key = tuple(int(v) for v in counts)
                if key not in cache:
                    result=solve_lipschitz(law,np.asarray(counts,float)/sum(counts))
                    maximum_gap=max(maximum_gap,result['gap'])
                    cache[key]=result['lower']
                    qp_file.write(json.dumps({'law':law.name,'counts':key,**result},separators=(',',':'))+'\n')
                    cache_count+=1
                return cache[key]
            for repetition in range(REPETITIONS):
                rng=np.random.default_rng(sequences[law_index*REPETITIONS+repetition])
                sample=rng.choice(len(p),size=max(SIZES),p=p)
                samples_file.write(json.dumps({'law':law.name,'repetition':repetition,'sample':sample.tolist()},separators=(',',':'))+'\n')
                simultaneous_failure=False
                for n in SIZES:
                    prefix=sample[:n]
                    for b in BLOCKS:
                        blocks=prefix[:b*(n//b)].reshape(-1,b)
                        if b==2:
                            scores=score_matrix[blocks[:,0],blocks[:,1]]
                        else:
                            counts=np.stack([(blocks==a).sum(axis=1) for a in range(len(p))],axis=1)
                            scores=np.asarray([lookup(c)*b/(b-1) for c in counts])
                        bound=block_score_bound(b,2.)
                        if np.any(scores<0) or np.any(scores>bound+1e-9):
                            raise ArithmeticError('Score outside declared range')
                        lower=erm_kl_lower(float(scores.mean()),len(scores),ALPHA,bound)
                        violated=lower>oracle['upper']+1e-9
                        simultaneous_failure |= violated
                        rows.append({'law':law.name,'repetition':repetition,'n':n,
                                     'method':f'block_{b}','block_size':b,'independent_scores':len(scores),
                                     'mean':float(scores.mean()),'variance':float(scores.var(ddof=1)),
                                     'score_min':float(scores.min()),'score_max':float(scores.max()),
                                     'bound':bound,'alpha':ALPHA,'lower':lower,'coverage_failure':int(violated)})
                    counts=np.bincount(prefix,minlength=len(p))
                    dual=lookup(counts)
                    lower=erm_kl_lower(dual,n,ALPHA,4.)
                    violated=lower>oracle['upper']+1e-9
                    simultaneous_failure |= violated
                    rows.append({'law':law.name,'repetition':repetition,'n':n,'method':'full_erm',
                                 'block_size':n,'independent_scores':n,'mean':dual,'variance':'',
                                 'score_min':'','score_max':'','bound':4.,'alpha':ALPHA,
                                 'lower':lower,'coverage_failure':int(violated)})
                failures+=int(simultaneous_failure)
            print(f'{law.name}: 100 complete streams, {len(cache)} distinct programs',flush=True)
    with (output/'endpoints.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'population.json').write_text(json.dumps(population,indent=2)+'\n')
    analytic=[]
    for b in range(2,513):
        for epsilon in (.01,.05,.10):
            value=(2*b*epsilon-1+(1-2*epsilon)**b)/(4*(b-1))
            analytic.append({'law':'rare_conflict','parameter':epsilon,'block_size':b,
                             'witness':value,'risk':epsilon/2,'ratio':value/(epsilon/2)})
        for m in (4,16,64):
            value=(b-m*(1-(1-1/m)**b))/(b-1)
            analytic.append({'law':'separated_noise','parameter':m,'block_size':b,
                             'witness':value,'risk':1.,'ratio':value})
    with (output/'analytic_hierarchy.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(analytic[0]));writer.writeheader();writer.writerows(analytic)
    if source_hashes != {p:digest(ROOT/p) for p in SOURCES}:
        raise RuntimeError('Execution source changed during run')
    summary={'law_repetition_streams':len(laws())*REPETITIONS,'endpoint_count':len(rows),
             'distinct_optimization_programs':cache_count,'maximum_empirical_qp_gap':maximum_gap,
             'simultaneous_coverage_failures':failures,'elapsed_seconds':time.perf_counter()-started,
             'source_unchanged':True,
             'files_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    (output/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
