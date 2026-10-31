"""Scratch-only numerical cross-check; imports no study model or analyzer."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path('/workspace/scratch/fa9a3361d0d9/work/kahm_repository/extensions/direct_reconstruction/results')
OUT = Path('/workspace/scratch/fa9a3361d0d9/work/independent_check.json')
execution = json.loads((ROOT/'execution.json').read_text())
assert execution['status'] in ('complete', 'complete_with_failures') and execution['completed_utc']
protocol = json.loads((ROOT/'protocol.json').read_text())
methods, tasks = protocol['methods'], protocol['tasks']
assert len(methods) == 17 and len(tasks) == 46
task_losses, natural_scales, unavailable, inputs = {}, {}, [], []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def agree(left, right, location):
    if left is None or right is None:
        assert left is right, (location, left, right)
    else:
        assert np.isclose(left, right, atol=1e-12, rtol=1e-9), (location, left, right)


for task in tasks:
    identifier = task['task_id']
    prepared_path = ROOT/task['prepared_path']
    assert sha(prepared_path) == task['prepared_sha256'], identifier
    with np.load(prepared_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    directory = ROOT/'tasks'/identifier
    prediction_path = directory/'predictions.npz'
    if not prediction_path.is_file():
        assert task['suite'] == 'natural'
        assert task['source_key'] == 'rideshare_price_hourly'
        assert len(arrays['B_X']) == 0 and (directory/'failure.json').is_file()
        unavailable.append(identifier)
        continue
    assert (directory/'complete.json').is_file()
    complete = json.loads((directory/'complete.json').read_text())
    assert sha(prediction_path) == complete['predictions_sha256'], identifier
    inputs.extend([{'path':str(prepared_path), 'sha256':sha(prepared_path)},
                   {'path':str(prediction_path), 'sha256':sha(prediction_path)}])
    truth = arrays['D_Y']
    assert truth.ndim == 2 and truth.shape[1] == 3 and np.isfinite(truth).all()
    if task['suite'] == 'synthetic':
        assert truth.shape == (1664, 3)
        groups, code, count = np.unique(arrays['D_groups'], return_inverse=True, return_counts=True)
        assert len(groups) == 128 and np.all(count == 13)
        scale = 1.0
    else:
        values = arrays['raw_values']
        prefix = values[:len(values)//2]
        period = int(arrays['p'])
        scale, used_lag, pair_count = None, None, 0
        for lag in (period, 1):
            pairs = [(float(prefix[i]), float(prefix[i-lag])) for i in range(lag,len(prefix))
                     if np.isfinite(prefix[i]) and np.isfinite(prefix[i-lag])]
            if pairs:
                candidate_scale = math.fsum((a-b)**2 for a,b in pairs)/len(pairs)
                if np.isfinite(candidate_scale) and candidate_scale > 0:
                    scale, used_lag, pair_count = candidate_scale, lag, len(pairs)
                    break
        assert scale is not None
        agree(scale, float(arrays['loss_scale']), identifier+' NPZ A scale')
        agree(scale, task['loss_scale'], identifier+' protocol A scale')
        natural_scales[identifier] = {'recomputed_A_scale':scale, 'lag':used_lag,
                                     'pairs':pair_count, 'raw_A_stop':len(values)//2}
    with np.load(prediction_path, allow_pickle=False) as saved:
        assert np.array_equal(saved['D_targets'], truth)
        assert {key[3:] for key in saved.files if key.startswith('D__')} == set(methods)
        by_method = {}
        for method in methods:
            estimate = saved['D__'+method]
            assert estimate.shape == truth.shape and np.isfinite(estimate).all()
            squared = (estimate-truth)**2
            if task['suite'] == 'synthetic':
                per_origin = np.sum(squared, axis=1)/squared.shape[1]
                episode_losses = np.bincount(code, weights=per_origin)/count
                mse = math.fsum(float(value) for value in episode_losses)/len(episode_losses)
            else:
                mse = math.fsum(float(value) for value in squared.ravel())/squared.size
            by_method[method] = {'mse':mse, 'scaled_mse':mse/scale,
                                 'horizon_mse':np.mean(squared, axis=0).tolist()}
        task_losses[identifier] = by_method

assert len(task_losses) == 43 and len(unavailable) == 3
synthetic, primary = {}, []
critical = float(t.isf(0.05/(2*18), 11))
for law in protocol['laws']:
    roster = sorted([task for task in tasks if task['suite']=='synthetic' and task['law']==law], key=lambda row:row['seed'])
    assert len(roster) == 12 and [task['seed'] for task in roster] == list(range(84200,84212))
    synthetic[law] = {method:math.fsum(task_losses[task['task_id']][method]['mse'] for task in roster)/12
                      for method in methods}
    for control in protocol['primary_controls']:
        candidate = protocol['primary_candidate']
        difference = [task_losses[task['task_id']][control]['mse']-
                      task_losses[task['task_id']][candidate]['mse'] for task in roster]
        mean = math.fsum(difference)/12
        variance = math.fsum((value-mean)**2 for value in difference)/11
        margin = critical*math.sqrt(variance/12)
        low, high = mean-margin, mean+margin
        conclusion = 'positive' if low>0 else 'negative' if high<0 else 'unresolved'
        primary.append({'law':law,'control':control,'candidate':candidate,
                        'mean_difference':mean,'interval_low':low,'interval_high':high,
                        'relative_improvement_percent':100*mean/synthetic[law][control],
                        'conclusion':conclusion,'positive_repetitions':sum(value>0 for value in difference),
                        'differences':difference,'critical_t':critical,'n_repetitions':12,
                        'degrees_of_freedom':11,'family_size':18,'family_alpha':0.05})

natural_sources = {}
for source in sorted({task['source_key'] for task in tasks if task['suite']=='natural'}):
    roster = [task for task in tasks if task['suite']=='natural' and task['source_key']==source]
    available = [task for task in roster if task['task_id'] in task_losses]
    natural_sources[source] = {
        'planned_series':len(roster),'available_series':len(available),
        'means':{method:(math.fsum(task_losses[task['task_id']][method]['scaled_mse'] for task in available)/len(available)
                         if available else None) for method in methods}}
available_sources = [source for source,data in natural_sources.items() if data['available_series']>0]
assert len(available_sources) == 3
natural_available = {method:math.fsum(natural_sources[source]['means'][method] for source in available_sources)/3
                     for method in methods}
natural_contrasts = []
candidate = protocol['primary_candidate']
for method in methods:
    if method==candidate:
        continue
    differences = {source:natural_sources[source]['means'][method]-natural_sources[source]['means'][candidate]
                   for source in available_sources}
    mean_difference = natural_available[method]-natural_available[candidate]
    natural_contrasts.append({'control':method,'candidate':candidate,
                              'available_mean_difference':mean_difference,
                              'relative_improvement_percent':100*mean_difference/natural_available[method],
                              'source_differences':differences,
                              'source_wins':sum(value>0 for value in differences.values())})

report = {'scope':'Independent direct saved-array recomputation; no study model/analyzer imports or refitting',
          'planned_tasks':46,'available_tasks':43,'unavailable_tasks':unavailable,
          'planned_method_outputs':782,'available_method_outputs':731,'unavailable_method_outputs':51,
          'synthetic_method_outputs':36*17,'natural_method_outputs':7*17,
          'synthetic_means':synthetic,'primary_contrasts':primary,
          'primary_outcome_counts':dict(Counter(row['conclusion'] for row in primary)),
          'natural_sources':natural_sources,'natural_available_means':natural_available,
          'natural_available_source_count':3,'natural_available_series_count':7,
          'natural_four_collection_aggregate':None,'natural_source_contrasts':natural_contrasts,
          'natural_recomputed_A_scales':natural_scales,'task_losses':task_losses,'input_hashes':inputs}

if (ROOT/'summary.json').is_file() and (ROOT/'audit_report.json').is_file():
    summary = json.loads((ROOT/'summary.json').read_text())
    audit = json.loads((ROOT/'audit_report.json').read_text())
    assert audit['passed'] and audit['selected_models_replayed'] and summary['audit_passed']
    assert summary['expected_tasks']==46 and summary['expected_method_outputs']==782 and summary['available_method_outputs']==731
    assert not summary['complete_roster'] and len(summary['documented_unavailable_tasks'])==3
    discrepancies = []
    official_primary = {(row['law'],row['control']):row for row in summary['primary_contrasts']}
    for row in primary:
        reference = official_primary[(row['law'],row['control'])]
        assert reference['conclusion']==row['conclusion']
        for key in ('mean_difference','interval_low','interval_high','critical_t','relative_improvement_percent'):
            agree(row[key],reference[key],str((row['law'],row['control'],key)))
            discrepancies.append(abs(row[key]-reference[key]))
    official_aggregates = {(row['suite'],row['group'],row['method']):row for row in summary['method_aggregates']}
    for law in synthetic:
        for method in methods:
            agree(synthetic[law][method],official_aggregates[('synthetic',law,method)]['mean_loss'],law+'/'+method)
    for method in methods:
        agree(natural_available[method],official_aggregates[('natural','available_collection',method)]['mean_loss'],method+' available natural')
        full = official_aggregates[('natural','equal_collection',method)]
        assert full['mean_loss'] is None and not full['complete']
    for row in summary['natural_collection_metrics']:
        agree(natural_sources[row['source_key']]['means'][row['method']],row['mean_scaled_mse'],row['source_key']+'/'+row['method'])
    with (ROOT/'task_metrics.csv').open(newline='') as stream:
        task_rows = list(csv.DictReader(stream))
    assert len(task_rows)==731
    max_mse_difference = 0.0
    for row in task_rows:
        calculated = task_losses[row['task_id']][row['method']]
        for key in ('mse','scaled_mse'):
            agree(calculated[key],float(row[key]),row['task_id']+'/'+row['method']+'/'+key)
            max_mse_difference = max(max_mse_difference,abs(calculated[key]-float(row[key])))
    report['comparison'] = {'passed':True,'audit_passed':True,'all_731_task_losses_match':True,
                            'all_18_intervals_match':True,'full_four_collection_undefined_verified':True,
                            'max_primary_numeric_absolute_difference':max(discrepancies),
                            'max_task_metric_absolute_difference':max_mse_difference,
                            'summary_sha256':sha(ROOT/'summary.json'),'audit_sha256':sha(ROOT/'audit_report.json')}
else:
    report['comparison'] = {'passed':None,'status':'Waiting for completed independent audit summary'}

OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
selected = ('fixed_kahm','gate_kahm','reconstruction_kahm','raw_rbf','raw_hgb',
            'reconstruction_affine','reconstruction_locality','reconstruction_rbf_local','nonkahm_policy')
print(json.dumps({'output':str(OUT),'comparison':report['comparison'],
                  'synthetic_means':{law:{m:synthetic[law][m] for m in selected} for law in synthetic},
                  'primary_counts':report['primary_outcome_counts'],'primary_contrasts':primary,
                  'natural_means':{m:natural_available[m] for m in selected},
                  'natural_source_means':{source:{m:data['means'][m] for m in selected} for source,data in natural_sources.items()},
                  'natural_source_contrasts':[row for row in natural_contrasts if row['control'] in selected]},indent=2))
