#!/usr/bin/env python3
"""Check the new ablation's saved records and reproduce descriptive summaries."""
from collections import defaultdict
from pathlib import Path
from itertools import product
import csv, json, math, statistics
from scipy.stats import t
P=Path(__file__).resolve().parent

def read(name):
    with (P/name).open(newline='') as f: return list(csv.DictReader(f))

def grouped(records,keys):
    d=defaultdict(list)
    for r in records: d[tuple(r[k] for k in keys)].append(r)
    return d

def summary(records,keys,name):
    out=[]
    for key,rs in sorted(grouped(records,keys).items()):
        assert len({r['rep'] for r in rs})==len(rs)
        l=[float(r['loss']) for r in rs]
        assert all(math.isfinite(x) and x>=0 for x in l)
        out.append(dict(zip(keys,key),n=len(rs),mean_loss=statistics.mean(l),sd_loss=statistics.stdev(l) if len(l)>1 else None,mean_df=statistics.mean(float(r['df']) for r in rs),mean_rank=statistics.mean(float(r['rank']) for r in rs)))
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out

def main():
    fr,cr=read('forecast_losses.csv'),read('controlled_losses.csv'); fd,cd=read('forecast_diagnostics.csv'),read('controlled_diagnostics.csv')
    fm=grouped(fr,['law','cap','method','encoding','selection']); cm=grouped(cr,['method','omega','task','encoding','selection'])
    laws=['linear_ar','nonlinear_ar','van_der_pol','duffing']; methods=['K','R','A','center']; encs=['mem','local','sep','rawmem','gatedraw']
    expected=set((l,'0','raw','raw','validation') for l in laws)|set((l,c,m,e,'validation') for l,c,m,e in product(laws,['4','12'],methods,encs))
    assert {k for k in fm if k[-1]=='validation'}==expected and all(len(fm[k])==32 for k in expected)
    expc=set(('raw','0',task,'raw','validation') for task in ('aligned','minor','partial'))|set((m,w,task,e,'validation') for m,w,task,e in product(methods,['1','4','16','64'],['aligned','minor','partial'],encs))
    assert {k for k in cm if k[-1]=='validation'}==expc and all(len(cm[k])==32 for k in expc)
    # Matched df omissions must correspond to the declared numerical rank criterion.
    for d,df,eligible in [(fm,8,lambda k:k[-2]!='mem'),(cm,2.85,lambda k:k[-2] in ('local','sep','rawmem'))]:
        for k,rs in d.items():
            if k[-1]!='validation': continue
            match=d.get(k[:-1]+('matched_df',),[])
            expected_reps={r['rep'] for r in rs if eligible(k) and int(r['rank'])>df}
            assert {r['rep'] for r in match}==expected_reps
            assert all(abs(float(r['df'])-df)<1e-8 for r in match)
    assert len(fd)==256 and len(cd)==512
    assert len({(r['law'],r['rep'],r['cap']) for r in fd})==256
    assert len({(r['rep'],r['method'],r['omega']) for r in cd})==512
    comparisons=read('primary_comparisons.csv'); assert len(comparisons)==12
    maxerr=0.; directions=defaultdict(int)
    for r in comparisons:
        la,me=r['law'],r['method']; a={s['rep']:float(s['loss']) for s in fm[la,'4',me,'local','validation']}; b={s['rep']:float(s['loss']) for s in fm[la,'4',me,'sep','validation']}
        d=[a[i]-b[i] for i in a]; mean=statistics.mean(d); half=t.ppf(1-.05/24,31)*statistics.stdev(d)/math.sqrt(32)
        for k,v in [('local',statistics.mean(a.values())),('sep',statistics.mean(b.values())),('difference',mean),('lower',mean-half),('upper',mean+half)]: maxerr=max(maxerr,abs(float(r[k])-v))
        assert int(r['wins'])==sum(x<0 for x in d)
        directions['lower' if mean+half<0 else 'higher' if mean-half>0 else 'unresolved']+=1
    assert maxerr<1e-12
    fs=summary(fr,['law','cap','method','encoding','selection'],'forecast_summary.csv'); cs=summary(cr,['method','omega','task','encoding','selection'],'controlled_summary.csv')
    dsum=[]
    for (m,w),rs in sorted(grouped(cd,['method','omega']).items()):
        dsum.append(dict(method=m,omega=w,n=len(rs),mean_cluster_accuracy=statistics.mean(float(r['cluster_accuracy']) for r in rs),mean_positive_margin_fraction=statistics.mean(float(r['positive_margin_fraction']) for r in rs if r['positive_margin_fraction']),mean_median_margin=statistics.mean(float(r['median_margin']) for r in rs if r['median_margin']),mean_median_weighted_sigma=statistics.mean(float(r['median_weighted_sigma']) for r in rs)))
    with (P/'controlled_diagnostic_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dsum[0]));w.writeheader();w.writerows(dsum)
    result=dict(status='PASS',forecast_records=len(fr),controlled_records=len(cr),forecast_validation_records=sum(r['selection']=='validation' for r in fr),controlled_validation_records=sum(r['selection']=='validation' for r in cr),forecast_matched_records=sum(r['selection']=='matched_df' for r in fr),controlled_matched_records=sum(r['selection']=='matched_df' for r in cr),primary_directions=dict(directions),max_primary_recalculation_error=maxerr,forecast_stacked_full_rank=sum(int(r['stacked_rank'])==12 for r in fd),forecast_rank_collections=len(fd),forecast_fallback_count=sum(int(r['fallback']) for r in fd),controlled_perfect_clusters=sum(float(r['cluster_accuracy'])==1 for r in cd if r['method']=='K' and r['omega']=='1'),controlled_fallback_count=sum(int(r['fallback']) for r in cd if r['method']=='K' and r['omega']=='1'))
    (P/'verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
