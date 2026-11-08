"""Recompute prediction losses and preserve primary failures in decoder studies."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gzip,json,time
ROOT=Path(__file__).resolve().parent
def q(v):return F.from_float(v) if isinstance(v,float) else F(v)
def check(ok,msg):
 if not ok:raise AssertionError(msg)
def mse(y,p):
 check(len(y)>0 and len(y)==len(p),'Target/prediction roster')
 return sum(((q(a)-q(b))**2 for a,b in zip(y,p)),F())/len(y)
def interval_loss(y,lo,hi):
 y,lo,hi=map(q,(y,lo,hi));check(lo<=hi,'Prediction interval order')
 return max(lo-y,y-hi,F())**2,max((lo-y)**2,(hi-y)**2)
def run():
 summaries=[];risk_count=0;ablation_rows=0
 for path in sorted((ROOT/'data').glob('*.json.gz')):
  with gzip.open(path,'rt') as f:d=json.load(f)
  n=d['study'];task=d['task']
  if n==42:
   risks={};exact={}
   for row in d['rows']:
    physical=mse(d['targets'],[float.fromhex(x) for x in row['prediction_hex']]);ideal=mse(d['targets'],row['exact_prediction'])
    check(physical==q(row['physical_risk']['mse']),f'{task}/{row["id"]}: physical loss')
    check(ideal==q(row['exact_risk']['mse']),f'{task}/{row["id"]}: exact reference loss')
    risks[row['id']]=physical;exact[row['id']]=ideal
   effects={}
   for label,letter in [('project','P'),('recenter','R'),('clip','C')]:
    counts={'pairs':0,'exact_equal':0,'exact_improved':0,'exact_worsened':0}
    for cid,value in risks.items():
     if letter+'0' not in cid.split('::')[-1]:continue
     partner=cid.replace(letter+'0',letter+'1');other=risks[partner]
     counts['pairs']+=1;counts['exact_equal']+=other==value;counts['exact_improved']+=other<value;counts['exact_worsened']+=other>value
    for k,v in counts.items():check(v==d['effects'][label][k],f'{task}/{label}/{k}: ablation count')
    effects[label]=counts
   check(len(risks)==720,f'{task}: ablation configuration roster')
   ablation_rows+=len(risks);risk_count+=2*len(risks)
   summaries.append({'study':'affine_ablation','task':task,'configurations':len(risks),'effects':effects,'independent_confirmation':False})
   continue
  risks={cid:[F(),F()] for cid in d['expected_risks']}
  for obs in d['observations']:
   for cid,pred in obs['predictions'].items():
    if n==41:
     v=q(float.fromhex(pred['hex']));a=b=(v-q(obs['target']))**2
    else:a,b=interval_loss(obs['target'],pred['lower'],pred['upper'])
    risks[cid][0]+=a;risks[cid][1]+=b
  count=len(d['observations'])
  check(count>0 and len({o['id'] for o in d['observations']})==count,f'{task}: unique query roster')
  for cid,bounds in risks.items():
   bounds[:]=[v/count for v in bounds];expected=d['expected_risks'][cid]
   check(expected['n']==count,f'{task}/{cid}: observation count')
   low=q(expected.get('lower',expected.get('mse')));high=q(expected.get('upper',expected.get('mse')))
   check(bounds==[low,high],f'{n}/{task}/{cid}: risk interval')
  names=d['names'];gate=d['gate'];risk_count+=len(risks)
  def less(a,b,factor=F(1)):
   return (names[a]==names[b] and factor==1) or risks[names[a]][1]<=factor*risks[names[b]][0]
  if n==38:
   gain=less('qp_all','legacybest',F(19,20));ordinary=less('qp_all','ordinarybest');passed=gain and ordinary
   check(gain==gate['at_least_five_percent_mse_improvement'] and ordinary==gate['no_worse_than_validation_selected_ordinary_baseline'] and passed==gate['numerical_practical_transfer_criterion'],f'{task}: primary criterion')
  elif n in [39,40]:
   reference='default_qp_rolling' if n==39 else 'qp_rolling'
   gain=less('joint_rolling',reference,F(19,20));ordinary=less('joint_rolling','ordinary_rolling')
   affine=True if n==39 else less('joint_rolling','affine_rolling')
   if n==40:check(affine==gate['no_worse_than_selected_affine'],f'{task}: affine comparison')
   passed=gain and ordinary and affine
   check(gain==gate['at_least_five_percent_mse_improvement'] and ordinary==gate['no_worse_than_selected_ordinary'] and passed==gate['numerical_criterion_pass'],f'{task}: primary criterion')
  else:
   check(risks[names['ordinary_rolling']][0]>0,f'{task}: positive reference risk')
   gain=less('affine_rolling','ordinary_rolling',F(19,20));within=less('affine_rolling','ordinary_rolling',F(21,20));passed=within
   check(gain==gate['at_least_five_percent_lower_mse'] and within==gate['within_five_percent_mse'],f'{task}: affine criterion')
  summaries.append({'study':{38:'training_transfer',39:'temporal_selection',40:'mixed_head_selection',41:'affine_comparison'}[n],'task':task,'observations':count,'candidate_risks_checked':len(risks),'primary_criterion_pass':passed})
 return {'status':'PASS','candidate_risks_recomputed':risk_count,'ablation_configurations':ablation_rows,'results':summaries,
 'scope':'Losses and primary decisions recomputed from retained prediction intervals/binary64 outputs and original targets. This replay does not rerun training, feature construction, or optimization; exposed-period ablations remain diagnostic.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path);args=ap.parse_args();start=time.perf_counter();r=run();r['wall_seconds']=time.perf_counter()-start
 if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r,indent=2))
