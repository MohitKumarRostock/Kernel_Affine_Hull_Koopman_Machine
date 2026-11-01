"""Descriptive natural analysis and a separate saved-array numerical audit.

No independent-source sampling claim is made for anonymous M4 series.
"""
from __future__ import annotations
import csv
import gzip
import io
import json
from pathlib import Path
import pickle
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import natural_pipeline as p


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def task_rows(root):
    rows, horizon_rows, selections, resources = [], [], [], []
    for task in p.roster(root):
        tid = task['task_id']
        a = load_arrays(root / 'prepared/abc' / (tid + '.npz'))
        y = load_arrays(root / 'prepared/target' / (tid + '.npz'))['D_Y']
        arrays = load_arrays(root / 'predictions' / tid / 'predictions.npz')
        fit = json.loads((root / 'results' / tid / 'fit_summary.json').read_text())
        if set(arrays) != p.expected_keys():
            raise AssertionError('Natural analysis has incomplete forecast arrays')
        for key, forecast in sorted(arrays.items()):
            sq = (y - forecast) ** 2
            absolute = np.abs(y - forecast)
            mse = np.mean(sq, axis=0)
            mae = np.mean(absolute, axis=0)
            row = {'task_id': tid, 'series_id': task['series_id'], 'frequency': task['frequency'],
                'method_key': key, 'n_origins': len(y), 'n_horizons': y.shape[1],
                'SMSE': float(np.mean(mse / a['loss_scale'])),
                'seasonal_MASE': float(np.mean(mae) / a['mase_scale']),
                'frequency_MASE': float(np.mean(mae) / a['frequency_mase_scale']),
                'MSE': float(np.mean(mse)), 'MAE': float(np.mean(mae)),
                'max_abs_error': float(np.max(absolute))}
            rows.append(row)
            for h, offset in enumerate((1, 4, 12)):
                horizon_rows.append({**{k: row[k] for k in ('task_id', 'series_id', 'frequency', 'method_key')},
                    'horizon': offset, 'MSE': float(mse[h]), 'MAE': float(mae[h]),
                    'SMSE': float(mse[h] / a['loss_scale'][h])})
        for policy in ('P0', 'P1', 'P0plus'):
            selections.append({'task_id': tid, 'frequency': task['frequency'], 'policy': policy,
                'selected_identity': fit['policies'][policy]['selected_identity']})
        resources.append({'task_id': tid, 'frequency': task['frequency'],
            'fit_wall_seconds': fit['resources']['total_wall_seconds'],
            'fit_cpu_seconds': fit['resources']['total_cpu_seconds'],
            'peak_rss_KiB': fit['resources']['peak_rss_KiB'],
            'state_gzip_bytes': fit['resources']['state_gzip_bytes'],
            'extra_fit_wall_seconds': fit['extra']['diagnostics']['fit_wall_seconds'],
            'KAHM_head_fit_seconds': fit['residual']['diagnostics']['head_fit_seconds_by_kind']['kahm'],
            'RBF_head_fit_seconds': fit['residual']['diagnostics']['head_fit_seconds_by_kind']['rbf']})
    return rows, horizon_rows, selections, resources


def aggregate_rows(rows):
    keys = sorted({r['method_key'] for r in rows})
    output = []
    for freq in ('Weekly', 'Hourly', 'Overall'):
        for key in keys:
            cells = [r for r in rows if r['method_key'] == key and (freq == 'Overall' or r['frequency'] == freq)]
            if len(cells) != (48 if freq == 'Overall' else 24):
                raise AssertionError('Natural aggregation has missing or duplicated series')
            # Both frequency panels have exactly 24 series; overall arithmetic
            # mean equals the prespecified equal-frequency/equal-series mean.
            output.append({'panel': freq, 'method_key': key, 'series_count': len(cells),
                **{metric: float(np.mean([r[metric] for r in cells]))
                   for metric in ('SMSE', 'seasonal_MASE', 'frequency_MASE')}})
    return output


def contrasts(aggregates, rows):
    lookup = {(r['panel'], r['method_key']): r for r in aggregates}
    result = []
    for panel in ('Weekly', 'Hourly', 'Overall'):
        for ref in ('P0', 'P0plus'):
            left, right = lookup[panel, 'policy__calibrated__P1'], lookup[panel, 'policy__calibrated__' + ref]
            for metric in ('SMSE', 'seasonal_MASE', 'frequency_MASE'):
                a, b = left[metric], right[metric]
                result.append({'panel': panel, 'comparison': 'P1_vs_' + ref, 'metric': metric,
                    'P1_loss': a, 'reference_loss': b, 'difference': a-b,
                    'gain_percent': 100*(1-a/b) if b else None,
                    'inference': 'descriptive; dependent anonymous benchmark series'})
    return result


def analyze(root):
    root = Path(root).resolve()
    p.verify_freeze(root)
    p.verify_seal(root, 'fit')
    p.verify_seal(root, 'prediction')
    p.manifest(root, 'target')
    rows, horizons, selections, resources = task_rows(root)
    aggregates = aggregate_rows(rows)
    compare = contrasts(aggregates, rows)
    counts = {}
    for panel in ('Weekly', 'Hourly', 'Overall'):
        counts[panel] = {}
        for policy in ('P0', 'P1', 'P0plus'):
            identities = [r['selected_identity'] for r in selections if r['policy'] == policy
                and (panel == 'Overall' or r['frequency'] == panel)]
            counts[panel][policy] = {k: identities.count(k) for k in sorted(set(identities))}
    per_task = {(r['task_id'], r['method_key']): r for r in rows}
    chosen = {s['task_id'] for s in selections if s['policy'] == 'P1' and s['selected_identity'] == 'residual__insample_kahm'}
    conditional = []
    for label, tids in [('KAHM_selected_by_C', chosen), ('KAHM_not_selected_by_C', {t['task_id'] for t in p.roster(root)} - chosen)]:
        for ref in ('P0', 'P0plus'):
            a = [per_task[tid, 'policy__calibrated__P1']['SMSE'] for tid in sorted(tids)]
            b = [per_task[tid, 'policy__calibrated__' + ref]['SMSE'] for tid in sorted(tids)]
            conditional.append({'subset': label, 'reference': ref, 'series_count': len(tids),
                'P1_SMSE': float(np.mean(a)) if a else None, 'reference_SMSE': float(np.mean(b)) if b else None,
                'gain_percent': 100*(1-float(np.mean(a))/float(np.mean(b))) if b and np.mean(b) else None,
                'wins': sum(x < y for x, y in zip(a, b)), 'ties': sum(x == y for x, y in zip(a, b)),
                'losses': sum(x > y for x, y in zip(a, b)),
                'interpretation': 'C-defined selected subset, descriptive and not a new primary claim'})
    out = root / 'analysis'
    for name, values in [('task_methods.csv', rows), ('horizon_methods.csv', horizons),
        ('aggregates.csv', aggregates), ('policy_contrasts.csv', compare),
        ('policy_selections.csv', selections), ('resources.csv', resources), ('selected_subset.csv', conditional)]:
        p.write_bytes(out / name, csv_bytes(values))
    floors = {'seasonal_MASE': 0, 'frequency_MASE': 0}
    for task in p.roster(root):
        a = load_arrays(root / 'prepared/abc' / (task['task_id'] + '.npz'))
        floors['seasonal_MASE'] += int(a['mase_scale_floor_active'])
        floors['frequency_MASE'] += int(a['frequency_mase_scale_floor_active'])
    summary = {'status': 'completed', 'utc': p.now(), 'series_count': 48, 'frequencies': ['Weekly', 'Hourly'],
        'origins_per_series': 32, 'horizons': [1, 4, 12], 'forecast_arrays': len(rows),
        'official_M4_Test_files_downloaded': False, 'evidence_status': 'external benchmark-development evidence',
        'statistical_inference': 'descriptive only; neither independent source sampling nor official M4 ranking',
        'selection_counts': counts, 'policy_contrasts': compare, 'conditional_subsets': conditional,
        'MASE_denominator_floors': floors,
        'resources': {'fit_worker_wall_seconds_sum': sum(r['fit_wall_seconds'] for r in resources),
            'fit_worker_cpu_seconds_sum': sum(r['fit_cpu_seconds'] for r in resources),
            'peak_worker_rss_MiB': max(r['peak_rss_KiB'] for r in resources)/1024,
            'complete_state_gzip_bytes': sum(r['state_gzip_bytes'] for r in resources),
            'scope': 'Entire finite research suite including controls and rejected candidates; not policy deployment latency'},
        'freeze_sha256': p.digest(root / 'freeze.json'), 'fit_lock_sha256': p.digest(root / 'FIT_LOCK.json'),
        'prediction_lock_sha256': p.digest(root / 'PREDICTION_LOCK.json'),
        'target_manifest_sha256': p.digest(root / 'prepared/target_manifest.json'),
        'files': {f.name: p.digest(f) for f in out.iterdir() if f.is_file()}}
    p.write_json(out / 'analysis_summary.json', summary)
    return {'status': 'completed', 'series': 48, 'forecast_arrays': len(rows),
        'overall_SMSE_contrasts': [r for r in compare if r['panel'] == 'Overall' and r['metric'] == 'SMSE'],
        'P1_selections': counts['Overall']['P1']}


def verify(root):
    """Independently reconstruct chronology, saved selections and all losses."""
    root = Path(root).resolve()
    p.verify_freeze(root)
    p.verify_seal(root, 'fit')
    p.verify_seal(root, 'prediction')
    for stage in ('abc', 'query', 'target'):
        p.manifest(root, stage)
    c = p.configuration(root)
    counts = {'series': 0, 'D_arrays': 0, 'B_alpha_paths': 0, 'C_eta_choices': 0,
              'policy_choices': 0, 'selected_ridge_normal_equations': 0}
    independent = []
    p.fd.load_runtime()
    for task, strings in p.source_rows(root, p.roster(root)):
        tid = task['task_id']
        values = np.asarray(strings, dtype=float)
        # Independently derive origins backwards from the last matured target.
        d, H, gap = 12, 12, 12
        ds = np.arange(len(values)-H-32, len(values)-H)
        cs = np.arange(ds[0]-gap-64, ds[0]-gap)
        bs = np.arange(cs[0]-gap-64, cs[0]-gap)
        last_a = bs[0]-gap-1
        aa = np.arange(max(d-1, last_a-511), last_a+1)
        ix = {'A': aa, 'B': bs, 'C': cs, 'D': ds}
        abc = load_arrays(root / 'prepared/abc' / (tid + '.npz'))
        query = load_arrays(root / 'prepared/query' / (tid + '.npz'))
        target = load_arrays(root / 'prepared/target' / (tid + '.npz'))
        for split in ('A', 'B', 'C', 'D'):
            saved = abc if split != 'D' else query
            np.testing.assert_array_equal(saved[split + '_origins'], ix[split])
            np.testing.assert_array_equal(saved[split + '_X'], np.asarray([values[t-11:t+1] for t in ix[split]]))
            sy = abc[split + '_Y'] if split != 'D' else target['D_Y']
            np.testing.assert_array_equal(sy, values[ix[split][:, None] + np.array([1, 4, 12])])
        for left, right in (('A', 'B'), ('B', 'C'), ('C', 'D')):
            assert ix[left][-1] + H < ix[right][0]
        scale = np.maximum(np.var(abc['A_Y'], axis=0), 1e-12)
        np.testing.assert_array_equal(scale, abc['loss_scale'])
        start = int(aa[0])-11
        training = values[start:int(aa[-1])+13]
        lag = int(task['seasonal_period'])
        seasonal_scale = max(float(np.mean(np.abs(training[lag:]-training[:-lag]))), 1e-12)
        lag2 = 1 if task['frequency'] == 'Weekly' else 24
        conventional_scale = max(float(np.mean(np.abs(training[lag2:]-training[:-lag2]))), 1e-12)
        np.testing.assert_allclose(seasonal_scale, abc['mase_scale'], rtol=1e-15)
        np.testing.assert_allclose(conventional_scale, abc['frequency_mase_scale'], rtol=1e-15)
        for split in ('B', 'C', 'D'):
            sv = abc if split != 'D' else query
            origins = ix[split]
            h = np.array([1, 4, 12])
            direct_seasonal = values[origins[:, None]+h-lag]
            direct_drift = values[origins, None]+h*((values[origins]-values[start])/(origins-start))[:, None]
            np.testing.assert_array_equal(sv[split+'_seasonal_naive'], direct_seasonal)
            np.testing.assert_array_equal(sv[split+'_drift'], direct_drift)
        fit = json.loads((root/'results'/tid/'fit_summary.json').read_text())
        validation = load_arrays(root/'results'/tid/'validation_predictions.npz')
        training_arrays = load_arrays(root/'results'/tid/'training_arrays.npz')
        np.testing.assert_array_equal(abc['A_Y']-training_arrays['A_full_common'], training_arrays['A_insample_residual'])
        bias = np.mean(abc['B_Y']-training_arrays['B_common'], axis=0)
        np.testing.assert_array_equal(training_arrays['B_common']+bias, training_arrays['B_shared_bias'])
        np.testing.assert_array_equal(bias, fit['residual']['selections']['shared_bias']['value'])
        for family, kind in (('residual','kahm'), ('residual','rbf'), ('extra',None)):
            selection = fit[family]['selections'][kind] if kind else fit[family]['selections']
            scores = []
            for record in selection['candidates']:
                key = record['state_key'] if kind else record['candidate_id']
                paths = validation[family+'__'+key+'__alpha_paths']
                losses = np.mean((abc['B_Y'][None]-paths)**2, axis=1)
                ai = np.argmin(losses, axis=0)
                np.testing.assert_array_equal(ai, record['selected_alpha_indices'])
                np.testing.assert_allclose(losses, record['B_mse_by_alpha_horizon'], rtol=1e-14, atol=1e-14)
                chosen = np.column_stack([paths[ai[j],:,j] for j in range(3)])
                np.testing.assert_array_equal(chosen, validation[family+'__'+key])
                scores.append(float(np.mean(np.mean((abc['B_Y']-chosen)**2,axis=0)/scale)))
                counts['B_alpha_paths'] += len(paths)
            assert int(np.argmin(scores)) == selection['selected_index']
        cp = load_arrays(root/'results'/tid/'calibration_predictions.npz')
        dp = load_arrays(root/'predictions'/tid/'predictions.npz')
        for family, method in [('residual',m) for m in p.SLOPES]+[('extra','extra_rbf')]:
            common_c = cp['residual__calibrated__shared_bias']
            raw_c = cp[family+'__eta1__'+method]
            losses = np.stack([np.mean((abc['C_Y']-(common_c+g*(raw_c-common_c)))**2,axis=0)
                for g in c['model']['gamma_grid']])
            eta = np.asarray(c['model']['gamma_grid'])[np.argmin(losses,axis=0)]
            observed_eta = (fit['residual']['calibration']['choices'][method]['selected_eta'] if family=='residual'
                else fit['extra']['calibration']['selected_eta'])
            np.testing.assert_array_equal(eta, observed_eta)
            np.testing.assert_allclose(cp[family+'__calibrated__'+method], common_c+eta*(raw_c-common_c), atol=1e-10,rtol=1e-13)
            common_d, raw_d = dp['residual__calibrated__shared_bias'], dp[family+'__eta1__'+method]
            np.testing.assert_allclose(dp[family+'__calibrated__'+method], common_d+eta*(raw_d-common_d),atol=1e-10,rtol=1e-13)
            counts['C_eta_choices'] += 3
        for policy, order in c['policy_orders'].items():
            scores = []
            for identity in order:
                key = p.shared.policy_key(identity, 'calibrated')
                scores.append(float(np.mean(np.mean((abc['C_Y']-cp[key])**2,axis=0)/scale)))
            choice = order[int(np.argmin(scores))]
            assert choice == fit['policies'][policy]['selected_identity']
            for mode in ('calibrated','eta1'):
                key = p.shared.policy_key(choice,mode)
                np.testing.assert_array_equal(cp['policy__'+mode+'__'+policy],cp[key])
                np.testing.assert_array_equal(dp['policy__'+mode+'__'+policy],dp[key])
            counts['policy_choices'] += 1
        state = pickle.loads(gzip.decompress((root/'results'/tid/'state.pkl.gz').read_bytes()))
        for method in p.SLOPES:
            ss = state['residual'].departures[method]
            raw,_ = p.inherited._checked_raw(ss.bank,abc['A_X'],'independent_natural_selected_head')
            fa = p.secondary._bounded(p.fd.load_runtime(),raw[ss.kind],ss.feature_scaler)
            z = fa-fa.mean(axis=0)
            yy = training_arrays['A_insample_residual']
            yy = yy-yy.mean(axis=0)
            for j,alpha in enumerate(ss.head['alpha_by_horizon']):
                oracle = np.linalg.solve(z.T@z+len(z)*alpha*np.eye(z.shape[1]),z.T@yy[:,j])
                np.testing.assert_allclose(oracle,ss.head['coef'][:,j],atol=1e-7,rtol=2e-6)
                counts['selected_ridge_normal_equations'] += 1
        from extra_rbf import _scaled
        ss=state['extra']
        scaled=_scaled(abc['A_X'],ss.selections['selected_scaler'],ss.history_audit,ss.raw_scaler)
        runtime=p.fd.load_runtime()
        fa=runtime.radial_bound(runtime._raw_rbf_features(ss.feature_state,scaled))
        z=fa-fa.mean(axis=0)
        yy=training_arrays['A_insample_residual']
        yy=yy-yy.mean(axis=0)
        for j,alpha in enumerate(ss.head['alpha_by_horizon']):
            oracle=np.linalg.solve(z.T@z+len(z)*alpha*np.eye(z.shape[1]),z.T@yy[:,j])
            np.testing.assert_allclose(oracle,ss.head['coef'][:,j],atol=1e-7,rtol=2e-6)
            counts['selected_ridge_normal_equations'] += 1
        for key,forecast in dp.items():
            error=target['D_Y']-forecast
            mse=np.mean(error**2,axis=0)
            mae=np.mean(np.abs(error),axis=0)
            independent.append({'task_id':tid,'frequency':task['frequency'],'method_key':key,
                'SMSE':float(np.mean(mse/scale)), 'seasonal_MASE':float(np.mean(mae)/seasonal_scale),
                'frequency_MASE':float(np.mean(mae)/conventional_scale)})
            counts['D_arrays'] += 1
        counts['series'] += 1
    with (root/'analysis/task_methods.csv').open() as stream:
        published=list(csv.DictReader(stream))
    actual={(r['task_id'],r['method_key']):r for r in published}
    assert len(actual)==len(independent)==48*len(p.expected_keys())
    for r in independent:
        published_row=actual[r['task_id'],r['method_key']]
        for metric in ('SMSE','seasonal_MASE','frequency_MASE'):
            np.testing.assert_allclose(r[metric],float(published_row[metric]),rtol=1e-13,atol=1e-13)
    with (root/'analysis/aggregates.csv').open() as stream:
        aggregates=list(csv.DictReader(stream))
    for row in aggregates:
        for metric in ('SMSE','seasonal_MASE','frequency_MASE'):
            frequencies=[]
            for freq in ('Weekly','Hourly'):
                cells=[r[metric] for r in independent if r['method_key']==row['method_key'] and r['frequency']==freq]
                assert len(cells)==24
                frequencies.append(float(np.mean(cells)))
            expected=np.mean(frequencies) if row['panel']=='Overall' else frequencies[0 if row['panel']=='Weekly' else 1]
            np.testing.assert_allclose(expected,float(row[metric]),rtol=1e-13,atol=1e-13)
    result={'status':'passed','utc':p.now(),'checks':counts,'source_and_stage_seals':True,
        'source_rows_and_all_split_arrays_reconstructed':True,'causal_direct_controls':True,
        'independent_B_alpha_and_family_selection':True,'independent_C_eta_and_policy_selection':True,
        'all_D_losses_and_frequency_aggregates_recomputed':True,
        'selected_ridge_coefficients_independently_solved':True,
        'scope':'Reuses inherited feature maps; does not refit every unselected model or independently rebuild every terminal feature map',
        'analysis_summary_sha256':p.digest(root/'analysis/analysis_summary.json')}
    p.write_json(root/'audit/verification.json',result)
    return result
