from __future__ import annotations
import csv, json, math, subprocess, sys, time
from pathlib import Path

ROOT=Path('/mnt/data/kahkm_direct_sweep_fast')
SCRIPT=Path('/mnt/data/kahkm_archive/experiment_16_ai_classic_control_closed_loop_pylance_clean.py')
ROOT.mkdir(parents=True, exist_ok=True)
TASKS=['CartPole-v1','MountainCar-v0']
CLUSTERS=[10,20,50,100]
OMEGAS=[1,2,4,8]
BASE=[
    sys.executable, str(SCRIPT),
    '--train-episodes','8','--test-episodes','4','--max-steps','0',
    '--horizons','1','5','10','20','50',
    '--train-seed','0','--test-seed','1000','--random-state','0',
    '--nb','100','--subspace-dim','20','--beta','0.1','--nlms-epochs','20',
    '--tau','1e-6','--batch-size','512','--n-jobs','1','--kmeans-kind','full','--max-train-per-cluster','0'
]
index=[]; one=[]; multi=[]

def readcsv(p):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def writecsv(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

for task in TASKS:
  for c in CLUSTERS:
    for omega in OMEGAS:
      name=f'{task}_C{c}_omega{str(omega).replace(".","p")}'
      out=ROOT/'runs'/name
      out.mkdir(parents=True, exist_ok=True)
      cmd=BASE + ['--output-dir', str(out), '--tasks', task, '--n-clusters', str(c), '--omega', str(omega)]
      log=out/'run.log'
      print('RUN', name, flush=True)
      t0=time.time(); status='ok'
      try:
        with log.open('w',encoding='utf-8') as lf:
          subprocess.run(cmd, cwd='/mnt/data/kahkm_archive', stdout=lf, stderr=subprocess.STDOUT, check=True, timeout=240)
      except subprocess.TimeoutExpired:
        status='timeout'
      except subprocess.CalledProcessError as e:
        status=f'failed:{e.returncode}'
      sec=time.time()-t0
      index.append({'task':task,'n_clusters':c,'omega':omega,'status':status,'seconds':sec,'output_dir':str(out)})
      writecsv(ROOT/'combo_index.csv', index)
      if status!='ok':
        continue
      one_rows=readcsv(out/'experiment_16_one_step_results.csv')
      for r in one_rows:
        if r['method']=='kahkm_nlms':
          one.append({'task':task,'n_clusters':c,'omega':omega,'test_error':r['test_error'],'test_r2':r['test_r2'],'train_error':r['train_error'],'train_r2':r['train_r2'],'simplex_violation':r['simplex_violation']})
      multi_rows=readcsv(out/'experiment_16_multistep_summary.csv')
      for r in multi_rows:
        if r['method']=='kahkm_nlms':
          multi.append({'task':task,'n_clusters':c,'omega':omega,'horizon':r['horizon'],'relative_error':r['relative_error_mean'],'association_r2':r['association_r2_mean'],'simplex_violation':r['simplex_violation_mean']})
      writecsv(ROOT/'kahkm_one_step.csv', one)
      writecsv(ROOT/'kahkm_multistep.csv', multi)

# best by h50 and balanced mean horizons
best=[]
for task in TASKS:
  rows=[r for r in multi if r['task']==task]
  configs=sorted(set((int(r['n_clusters']), float(r['omega'])) for r in rows))
  for objective in ['h50','mean_horizons']:
    scored=[]
    for c,o in configs:
      rs=[r for r in rows if int(r['n_clusters'])==c and float(r['omega'])==o]
      if objective=='h50':
        rr=[r for r in rs if int(float(r['horizon']))==50]
        if not rr: continue
        score=float(rr[0]['relative_error']); r2=float(rr[0]['association_r2'])
      else:
        vals=[float(r['relative_error']) for r in rs if int(float(r['horizon'])) in {1,5,10,20,50}]
        score=sum(vals)/len(vals); r2=math.nan
      scored.append((score,c,o,r2))
    if scored:
      score,c,o,r2=min(scored)
      best.append({'task':task,'objective':objective,'n_clusters':c,'omega':o,'score':score,'h50_r2':r2})
writecsv(ROOT/'best_configs.csv', best)
print('DONE', flush=True)
