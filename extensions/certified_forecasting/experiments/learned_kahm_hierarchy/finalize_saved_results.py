#!/usr/bin/env python3
"""Finalize saved v1 records after the disclosed ancillary benchmark-log change.

This script does not draw data, fit models, solve QPs, or recompute/select
endpoints. It validates the exact one-file provenance exception, preserves
both benchmark logs, and reconstructs the terminal inventory from records.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import tarfile


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT.parents[1]
ALLOWED = 'extensions/certified_forecasting/experiments/learned_kahm_hierarchy/development_benchmark.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    output = args.results
    if (output/'run_summary.json').exists() or (output/'provenance_incident.json').exists():
        raise FileExistsError('Finalization is write-once')
    frozen = json.loads((output/'protocol_freeze.json').read_text())
    config = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'source_manifest.json').read_text())['files']
    for key, name in [('protocol_sha256', 'protocol.json'), ('source_snapshot_sha256', 'source_snapshot.tar.gz'),
                      ('source_manifest_sha256', 'source_manifest.json')]:
        assert digest(output/name) == frozen[key]
    assert digest(ROOT/frozen['models_path']) == frozen['models_sha256']
    changed = [m for m in manifest if digest(REPOSITORY/m['path']) != m['sha256']]
    assert len(changed) == 1 and changed[0]['path'] == ALLOWED, changed
    with tarfile.open(output/'source_snapshot.tar.gz', 'r:gz') as archive:
        for member in manifest:
            assert hashlib.sha256(archive.extractfile(member['path']).read()).hexdigest() == member['sha256']
        frozen_bytes = archive.extractfile(ALLOWED).read()
    late_bytes = (REPOSITORY/ALLOWED).read_bytes()
    old, late = json.loads(frozen_bytes), json.loads(late_bytes)
    assert late['rows'][:len(old['rows'])] == old['rows']
    assert len(late['rows']) == 28
    assert not Path('/proc/60446').exists(), 'Development process still alive'
    incident_dir = output/'ancillary_log_incident'
    incident_dir.mkdir()
    (incident_dir/'frozen_development_benchmark.json').write_bytes(frozen_bytes)
    (incident_dir/'completed_development_benchmark.json').write_bytes(late_bytes)
    endpoint_rows = read_csv(output/'endpoints.csv')
    decision_rows = read_csv(output/'decisions.csv')
    assert len(endpoint_rows) == config['endpoint_counts']['total'] == 1760
    assert len(decision_rows) == 880
    statistics = {'qp_count': 0, 'unconverged_qp_count': 0, 'arithmetic_fallback_batches': 0,
                  'maximum_normalized_gap': 0., 'maximum_feasibility_residual': 0.}
    with gzip.open(output/'qp_certificates.jsonl.gz', 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            statistics['qp_count'] += len(row['lower'])
            statistics['unconverged_qp_count'] += sum(not x for x in row['converged'])
            statistics['arithmetic_fallback_batches'] += int(row['arithmetic_fallback'] is not None)
            statistics['maximum_normalized_gap'] = max(statistics['maximum_normalized_gap'], max(row['gap']))
            statistics['maximum_feasibility_residual'] = max(statistics['maximum_feasibility_residual'], max(row['feasibility_residual']))
    assert statistics['qp_count'] == 143440
    samples = 0
    with gzip.open(output/'samples.jsonl.gz', 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            assert len(row['y']) == 4096
            samples += 1
    assert samples == 40
    finalized = datetime.now(timezone.utc).isoformat()
    numeric_completed = datetime.fromtimestamp((output/'endpoints.csv').stat().st_mtime, timezone.utc).isoformat()
    incident = {'recorded_at_utc': finalized, 'numerical_records_completed_at_utc': numeric_completed,
        'numerical_completion_time_source': 'endpoints.csv filesystem write time, recorded after all40 banks and before final hash check',
        'event': 'The runner finished and saved every numerical record, then failed its terminal all-source hash check because an ancillary development benchmark log acquired eight additional development-only rows.',
        'changed_path': ALLOWED, 'frozen_sha256': hashlib.sha256(frozen_bytes).hexdigest(),
        'completed_sha256': hashlib.sha256(late_bytes).hexdigest(), 'frozen_rows': len(old['rows']),
        'completed_rows': len(late['rows']), 'completed_log_updated_utc': late['updated_utc'],
        'benchmark_pid': 60446, 'benchmark_exited_before_finalization': True,
        'stop_attempts': 'SIGTERM and SIGKILL were attempted against the observed host PID from sibling managed execution contexts; the process remained active and eventually exited normally. No success is claimed for those attempts.',
        'correction_to_development_disclosure': 'The frozen protocol described the dense benchmark as stopped at n256 and the second development seed as unreached. The preserved frozen log already contains20 rows across both seeds; the process completed28 rows later. That process-status description was mistaken. The machine-readable sampling law, seeds, design, allocations, model selection, and solver were unchanged.',
        'numerical_effect': 'None from the log contents: development_benchmark.json is enumerated by source_paths for snapshot/hash validation but is not read by calibration computations. All executable source files, frozen configurations and archived models exactly match the prospective snapshot.',
        'runtime_effect': 'The lingering development process may have caused resource contention. Timings are execution diagnostics, not isolated runtime comparisons or evidence of superiority.',
        'confirmation_rerun': False, 'confirmation_data_regenerated': False,
        'model_or_endpoint_reselection': False, 'frozen_snapshot_modified': False,
        'executable_sources_unchanged': True,
        'pre_finalization_result_hashes': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        'finalizer_sha256': digest(__file__)}
    write_json(output/'provenance_incident.json', incident)
    start = json.loads((output/'calibration_started.json').read_text())['started_at_utc']
    elapsed = (datetime.fromisoformat(numeric_completed)-datetime.fromisoformat(start)).total_seconds()
    eta = config['confidence_delta']/config['endpoint_counts']['total']
    summary = {'completed_at_utc': finalized, 'numerical_records_completed_at_utc': numeric_completed,
        'scope': config['scope'], 'n_banks': samples, 'n_selected_models': 80, 'n_endpoints': len(endpoint_rows),
        'n_decisions': len(decision_rows), 'eta_per_endpoint': eta,
        'total_allocated_delta': eta*len(endpoint_rows), 'elapsed_seconds': elapsed,
        'elapsed_seconds_method': 'Reconstructed from recorded calibration-start time and endpoints.csv write time; possible contention from development benchmark',
        'source_unchanged': False, 'executable_sources_unchanged': True,
        'ancillary_source_changes': [ALLOWED], 'provenance_incident': 'provenance_incident.json',
        'finalization_scope': 'Inventory reconstructed from all already-saved records; no data or endpoint rerun',
        **statistics, 'files_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'run_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
