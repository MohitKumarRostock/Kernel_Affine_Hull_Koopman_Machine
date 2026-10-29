#!/usr/bin/env python3
"""Generate compact manuscript assets from retained confirmatory results."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True,type=Path);a=p.parse_args();folder=a.results
with (folder/'summary.csv').open(newline='') as f:summary=list(csv.DictReader(f))
with (folder/'endpoints.csv').open(newline='') as f:endpoints=list(csv.DictReader(f))
lines=[r'\begin{table}[t]',r'\centering',r'\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Regime & Episodes & Upper accepts & Unconditional & Stratified \\',r'\midrule']
for row in summary:
 regime=row['regime'];n=row['n_calibration'];den=row['n_repetitions'];u=row['upper_acceptances']
 key='minimum_cost_certificates' if regime=='repairable' else 'joint_infeasibility_certificates'
 un=row['unconditional_'+key];st=row['stratified_'+key]
 lines.append(f'{regime.capitalize()} & {int(n):,} & {u}/{den} & {un}/{den} & {st}/{den} '+r'\\')
 if regime=='repairable' and n=='16384':lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Decisions for fitted KAHM representations over 20 fresh trials per regime. The two retained classes have costs 2 and 6, common budget $\kappa=2$, and fixed MSE tolerance $0.10$. Upper accepts counts trials in which a fitted model passes. The last two columns count certification of minimum cost 6 in the repairable regime, or exclusion of both classes in the irreducible regime. Stratification takes the maximum of separately budgeted lower bounds from two frozen learned partitions. All bounds, trials, regimes, and sample-size prefixes share a total failure allocation of $0.05$.}',r'\label{tab:learned-kahm-decision}',r'\end{table}']
(folder/'learned_kahm_decision.tex').write_text('\n'.join(lines)+'\n')
rows=[]
for regime in ['repairable','irreducible']:
 for cost in [2,6]:
  selected=[r for r in endpoints if r['regime']==regime and int(r['cost'])==cost and int(r['n_calibration'])==16384]
  out={'regime':regime,'cost':cost,'n_trials':len(selected)}
  for name in ['mse','primary_lower','upper','unconditional_lower','coarse_stratified_lower','fine_stratified_lower']:
   vals=np.array([float(r[name]) for r in selected]);out.update({name+'_mean':float(vals.mean()),name+'_sd':float(vals.std(ddof=1)),name+'_minimum':float(vals.min()),name+'_maximum':float(vals.max())})
  rows.append(out)
with (folder/'fullsize_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(folder/'analysis_summary.json').write_text(json.dumps({'scope':'Descriptive aggregation of all predeclared confirmatory trials, no selection or new confidence claims','fullsize':rows},indent=2)+'\n')
manifest={'source_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['summary.csv','endpoints.csv','results.json']},'asset_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['learned_kahm_decision.tex','fullsize_summary.csv','analysis_summary.json']}}
(folder/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
