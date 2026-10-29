"""Reconstruct scaling selections and mean contrasts from retained raw JSON records."""
import argparse, csv, hashlib, json, math, statistics, tarfile
from collections import defaultdict
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study', type=Path, default=Path(__file__).resolve().parents[1] / 'results/input_scaling_study')
parser.add_argument('--output', type=Path, required=True, help='New JSON file outside the archived study directory')
args = parser.parse_args()
ROOT = args.study.resolve()
OUT = args.output.resolve()
if OUT.exists():
    parser.error('The output already exists; choose a new verification file.')
if OUT == ROOT or ROOT in OUT.parents:
    parser.error('Verification output must be outside the archived study directory.')
report={'phases':{}}
for phase in ('original','confirmation'):
    raw={}; count=0
    with tarfile.open(ROOT/f'records_{phase}.tar.gz') as archive:
        for name in archive.getnames():
            if not name.endswith('/results.json'): continue
            parts=Path(name).parts
            if len(parts) not in (3,4): continue
            mode,campaign=parts[:2]
            d=json.load(archive.extractfile(name))
            if 'results' not in d:continue
            for c in d['results']:
                count+=1
                pen=campaign=='pendigits_v1'
                seed=c['seed'] if pen else d['seed']
                coord=str(c['coordinate']) if pen else 'scalar'
                system='pendigits' if pen else d['system']
                key=(campaign,system,mode,seed,coord,c['family'])
                raw.setdefault(key,[]).append(c)
    chosen={}
    for key,cs in raw.items():
        if key[0]=='main_v1': f=lambda c:sum(c['validation_mse'])/len(c['validation_mse'])
        else:f=lambda c:(sum(c['validation_mse'])/len(c['validation_mse']),c['n_features'],c['name'])
        chosen[key]=min(cs,key=f)
    maxerr=0.; checked=0
    with (ROOT/f'analysis_{phase}/selected_candidates.csv').open() as f:
        for row in csv.DictReader(f):
            key=(row['campaign'],row['system'],row['mode'],int(row['seed']),row['coordinate'],row['family'])
            c=chosen[key]
            assert row['candidate']==c['name'],(phase,key,row['candidate'],c['name'])
            hs=[1,2,4] if key[0]=='pendigits_v1' else [1,4,12]
            idx=hs.index(int(row['horizon']))
            values=c['cohort_writer_mean_mse'] if key[0]=='pendigits_v1' else c['test_mse']
            for left,right in [(float(row['validation_mse']),c['validation_mse'][idx]),(float(row['test_mse']),values[idx]),(float(row['test_rmse']),math.sqrt(values[idx]))]:
                err=abs(left-right);maxerr=max(maxerr,err);assert err<=1e-12
            checked+=1
    assert checked==3*len(chosen)
    pooled=defaultdict(list)
    for key,c in chosen.items():
        campaign,system,mode,seed,coord,family=key
        values=c['cohort_writer_mean_mse'] if campaign=='pendigits_v1' else c['test_mse']
        for h,value in zip([1,2,4] if campaign=='pendigits_v1' else [1,4,12],values):
            pooled[(campaign,system,mode,seed,family,str(h))].append(value)
        pooled[(campaign,system,mode,seed,family,'all')].append(math.fsum(values)/len(values))
    pooled={key:math.fsum(values)/len(values) for key,values in pooled.items()}
    ncontrasts=0; maxcontrast=0.; primary=[]
    with (ROOT/f'analysis_{phase}/taskwise_comparison.csv').open() as f:
        for row in csv.DictReader(f):
            seeds=json.loads(row['seeds']);camp,system,mode,family,h=[row[x] for x in ('campaign','system','mode','family','horizon')]
            a=[pooled[(camp,system,'standard',seed,family,h)] for seed in seeds]
            b=[pooled[(camp,system,mode,seed,family,h)] for seed in seeds]
            ma,mb=statistics.mean(a),statistics.mean(b);diff=[y-x for x,y in zip(a,b)]
            checks={'standard_mean_mse':ma,'alternative_mean_mse':mb,'mse_difference_alternative_minus_standard':statistics.mean(diff),'relative_mse_change_percent':100*(mb/ma-1),'paired_mse_difference_sd_over_seeds':statistics.stdev(diff)}
            for k,v in checks.items():
                err=abs(float(row[k])-v);maxcontrast=max(maxcontrast,err);assert err<=1e-10,(phase,k,row[k],v)
            assert int(row['n_seeds_improved'])==sum(v < -1e-12 for v in diff)
            assert int(row['n_seeds_degraded'])==sum(v > 1e-12 for v in diff)
            ncontrasts+=1
            if mode=='minmax' and family=='kahm_nested' and h=='all':primary.append({'campaign':camp,'task':system,'percent_change':checks['relative_mse_change_percent'],'improving_seeds':int(row['n_seeds_improved'])})
    report['phases'][phase]={'status':'passed','raw_fitted_candidates':count,'validation_selected_models':len(chosen),'selected_horizon_rows_checked':checked,'taskwise_contrast_rows_checked':ncontrasts,'max_selected_absolute_error':maxerr,'max_contrast_absolute_error':maxcontrast,'primary_effects':primary}
report['scope']='Independent raw-JSON selection and mean-effect reconstruction; no refitting and no full-score confidence-interval audit.'
report['input_sha256'] = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for phase in ('original', 'confirmation')
    for path in (ROOT / f'records_{phase}.tar.gz',
                 ROOT / f'analysis_{phase}/selected_candidates.csv',
                 ROOT / f'analysis_{phase}/taskwise_comparison.csv')
}
report['verification_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='primary_effects'} for k,v in report['phases'].items()},indent=2))
