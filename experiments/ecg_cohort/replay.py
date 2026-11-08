#!/usr/bin/env python3
"""Recompute exact finite-cohort risk witnesses and exclusion decisions."""
from __future__ import annotations

import os
for option in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[option] = '1'

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source/step19_full_cohort/study'))
import exact19 as exact
import audit19 as audit


def read(path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data)


def check_manifest():
    manifest = read(HERE/'manifest.json')
    for name,expected in manifest['sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
            raise AssertionError('Artifact hash differs: '+name)
    return len(manifest['sha256'])


def run():
    started = time.perf_counter()
    checked = check_manifest()
    metadata = read(HERE/'provenance/input_manifest.json')
    expected_summary = read(HERE/'records/summary.json')
    with np.load(HERE/'records/inputs.npz',allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    population = len(data['Y'])
    exact.need(population==8192, 'Cohort size differs')
    exact.need(len(np.unique(data['patient_id']))==population, 'Patient identities repeat')
    reference = exact.exact_mse(data['Y'],data['reference_pred'])
    threshold_exact = Fraction(11,10)*reference
    threshold_upper = Fraction.from_float(metadata['threshold_upper'])
    exact.need(Fraction.from_float(metadata['threshold_lower']) <= threshold_exact <= threshold_upper,
               'Reference threshold enclosure is invalid')
    exact.need(reference == exact.from_record(expected_summary['exact_reference_mse']), 'Reference MSE differs')
    rows = []
    programs = edges = 0
    for prefix in (2,6,14):
        key = str(prefix)
        blocks = audit.slices_for(prefix)
        kappa = float(metadata['selected_kappa'][key])
        lo,hi,_ = exact.quantize_boxes(data['F_lo_'+key],data['F_hi_'+key])
        for family in ('consecutive','greedy_nearest'):
            base = audit.partition_independent(data['F_'+key],family)
            previous = previous_values = previous_components = None
            previous_risk = Fraction(0)
            for b in (32,64,128):
                name = f'p{prefix}_{family}_b{b}'
                expected = read(HERE/'records'/(name+'.json.gz'))
                with np.load(HERE/'records'/(name+'.npz'),allow_pickle=False) as z:
                    indices,new_duals = z['indices'],z['new_signed_dual']
                exact.need(np.array_equal(indices,base.reshape(-1,b)), 'Partition or nesting differs')
                count = len(indices)
                exact.need(count*b==population and len(expected['blocks'])==count,'Incomplete witness roster')
                chosen = np.empty_like(new_duals)
                values = []
                components_kept = []
                for start in range(0,count,16):
                    stop = min(start+16,count)
                    distances,_ = exact.edge_upper_numerators(lo,hi,indices[start:stop],blocks)
                    for local,block_id in enumerate(range(start,stop)):
                        raw,components = exact.exact_dual(new_duals[block_id],data['Y'][indices[block_id]],distances[local],kappa)
                        if b==32:
                            alpha,value,source = exact.exact_best(new_duals[block_id],raw)
                        else:
                            left,right = 2*block_id,2*block_id+1
                            alpha,value,source = exact.exact_best(new_duals[block_id],raw,
                                left_alpha=previous[left],right_alpha=previous[right],
                                left_value=previous_values[left],right_value=previous_values[right])
                        saved = expected['blocks'][block_id]
                        exact.need(raw == exact.from_record(saved['new_raw_sse']), 'Exact new dual differs')
                        exact.need(value == exact.from_record(saved['exact_selected_sse']), 'Exact selected dual differs')
                        exact.need(source == saved['source'], 'Exact witness choice differs')
                        exact.need(exact.array_hash(alpha)==saved['chosen_signed_dual_sha256'],'Selected witness bits differ')
                        if source=='new':
                            terms = tuple(components[k] for k in ('linear','quadratic','distance_penalty'))
                        elif source=='zero':
                            terms = (Fraction(0),)*3
                        else:
                            terms = tuple(a+c for a,c in zip(previous_components[left],previous_components[right],strict=True))
                        exact.need(value==terms[0]-terms[1]-terms[2], 'Selected objective decomposition differs')
                        chosen[block_id] = alpha
                        values.append(value)
                        components_kept.append(terms)
                total = sum(values,Fraction(0))
                lower = total/population
                exact.need(lower == exact.from_record(expected['exact_risk_lower']), 'Risk normalization differs')
                exact.need(lower>=previous_risk,'Nested lower bound decreased')
                expected_row = next(r for r in expected_summary['configurations'] if r['prefix']==prefix and r['family']==family and r['block_size']==b)
                exact.need(lower==exact.from_record(expected_row['exact_lower']),'Configuration lower differs')
                sums = [sum((terms[j] for terms in components_kept),Fraction(0)) for j in range(3)]
                necessary = audit.necessary_regularity(sums[0]-sums[1],sums[2],population,threshold_upper)
                exact.need(necessary==expected_row['necessary_regularity_unchanged_threshold'], 'Necessary sensitivity calculation differs')
                excludes = lower>threshold_upper
                exact.need(excludes == expected_row['strictly_excludes_unchanged_threshold_upper'],'Exclusion decision differs')
                rows.append({'prefix':prefix,'family':family,'block_size':b,'exact_lower':exact.rational_record(lower),
                    'strictly_excludes_unchanged_threshold_upper':excludes,
                    'necessary_regularity_unchanged_threshold':necessary})
                programs += count
                edges += count*b*(b-1)//2
                previous,previous_values,previous_components,previous_risk = chosen,values,components_kept,lower
    exact.need(programs==2688 and edges==5431296 and len(rows)==18,'Incomplete computation census')
    best = {}
    for prefix in (2,6,14):
        candidates = [r for r in rows if r['prefix']==prefix]
        winner = max(candidates,key=lambda r:exact.from_record(r['exact_lower']))
        regularity = audit.best_regularity(candidates,'necessary_regularity_unchanged_threshold')
        saved = expected_summary['best_by_prefix'][str(prefix)]
        exact.need(winner['exact_lower']==saved['exact_lower'],'Best geometry bound differs')
        exact.need(regularity==saved['necessary_regularity_unchanged_threshold'],'Best necessary budget differs')
        best[str(prefix)] = dict(winner,necessary_regularity_unchanged_threshold=regularity)
    return {'status':'PASS','experiment':'ecg_cohort','checked_artifacts':checked,
        'patients':population,'configurations':len(rows),'exact_duals_recomputed':programs,
        'edge_constraints_with_repetition':edges,'best_by_prefix':best,'configurations_recomputed':rows,
        'reference_mse':exact.rational_record(reference),'reference_requirement':exact.rational_record(threshold_exact),
        'threshold_upper':exact.rational_record(threshold_upper),
        'scope':{'arithmetic':'Exact integer and rational objectives conditional on inherited canonical-feature enclosures.',
                 'training_rerun':False,'native_feature_construction_replayed':False,
                 'new_patient_or_reserve_access':False,'population_risk_guarantee':False},
        'elapsed_seconds':time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','experiment','patients','configurations','exact_duals_recomputed','elapsed_seconds')}))


if __name__=='__main__':
    main()
