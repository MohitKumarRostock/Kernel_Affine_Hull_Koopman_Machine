#!/usr/bin/env python3
"""Recompute the controlled class-risk comparisons from stored witnesses."""
from __future__ import annotations

import os
for option in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[option] = '1'

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source/step17_certificate_utility/study'))
import verify17 as audit


def read(path):
    payload = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(payload)


def check_manifest():
    manifest = read(HERE/'manifest.json')
    for name, expected in manifest['sha256'].items():
        path = HERE/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError('Artifact hash differs: '+name)
    return len(manifest['sha256'])


def decode(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'__array__'}:
            return arrays[value['__array__']]
        return {k: decode(v, arrays) for k,v in value.items()}
    if isinstance(value, list):
        return [decode(v, arrays) for v in value]
    return value


def run():
    started = time.perf_counter()
    checked_files = check_manifest()
    frozen = read(HERE/'provenance/freeze.json')
    expected_summary = read(HERE/'records/summary.json')
    if frozen['streams'] != audit.independent_streams():
        raise AssertionError('Random stream specification differs from the frozen design')
    total = Counter()
    maxima = {}
    paired = {(r,n):Counter() for r in audit.REGIMES for n in audit.SAMPLE_SIZES}
    cases = []
    for case in frozen['cases']:
        index = case['case']
        meta = read(HERE/'records'/f'case_{index:02d}.json.gz')
        with np.load(HERE/'records'/f'case_{index:02d}.npz', allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        observed = {k.removeprefix('observations__'):v for k,v in arrays.items() if k.startswith('observations__')}
        audit.check_data(observed, 4096, frozen['streams'][f'r{index}_cal'], case['regime'])
        features = arrays['features']
        audit.simplex(features)
        witness = decode(meta['witness_records'], arrays)
        reconstructed = {}
        for pi,prefix in enumerate((2,6)):
            model = meta['models'][str(prefix)]
            coef = np.asarray(model['coef'], dtype=float)
            norm_sum = sum(float(np.linalg.norm(coef[lo:hi])) for lo,hi in audit.slices(prefix))
            audit.require(norm_sum <= 2.+2e-12, 'Stored affine head exceeds its declared guarded norm budget')
            predictions = audit.prediction(features, model, prefix)
            audit.near(predictions, arrays['predictions'][:,pi], 2e-12, 2e-12)
            result, counts, errors = audit.audit_certificate(features[:,:prefix], observed['Y'], predictions, witness[str(prefix)])
            reconstructed[prefix] = result
            total.update(counts)
            for key,value in errors.items():
                maxima[key] = max(maxima.get(key,0.), float(value))
        decisions = {}
        for n in audit.SAMPLE_SIZES:
            actual = audit.decisions(reconstructed, meta['costs'], case['regime'], n)
            audit.check_decisions(meta['calibration']['decisions'][str(n)], actual)
            paired[(case['regime'],n)][actual['comparison']['paired_resolution']] += 1
            decisions[str(n)] = actual
        cases.append({'case':index, 'regime':case['regime'], 'decisions':decisions})
    audit.require(total['block_programs'] == 190464, 'Incomplete block-program replay')
    audit.require(total['one_sided_events'] == 1008, 'Confidence-event count differs')
    rows = []
    for (regime,n),counts in paired.items():
        values = {key:counts[key] for key in ('both','hierarchy_only','recycled_only','neither')}
        expected = next(row for row in expected_summary['paired_resolution'] if row['regime']==regime and row['n']==n)
        audit.require(values == expected['paired_counts'], 'Paired decision summary differs')
        rows.append({'regime':regime,'n':n,'paired_counts':values})
    return {
        'status':'PASS', 'experiment':'controlled_calibration',
        'checked_artifacts':checked_files, 'learning_cases':len(cases),
        'selected_classes':2*len(cases), 'replayed_block_programs':total['block_programs'],
        'replayed_one_sided_endpoints':total['one_sided_events'],
        'paired_resolution':rows, 'maxima':maxima, 'cases':cases,
        'scope':{
            'synthetic_calibration_observations_regenerated':True,
            'stored_fitted_predictions_recomputed':True,
            'dual_objectives_and_primal_diagnostics_recomputed':True,
            'confidence_endpoints_and_decisions_recomputed':True,
            'training_rerun':False, 'native_feature_construction_replayed':False,
            'arithmetic':'Guarded floating replay; no end-to-end interval proof of population decisions.',
        },
        'elapsed_seconds':time.perf_counter()-started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','experiment','learning_cases','replayed_block_programs','replayed_one_sided_endpoints','elapsed_seconds')}))


if __name__=='__main__':
    main()
