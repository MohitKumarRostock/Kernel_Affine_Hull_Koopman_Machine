#!/usr/bin/env python3
"""Retain complete matched-study records and selected full-precision evidence.

No models are fitted or selected here. Selection comes exclusively from the
frozen analyzer. Full synthetic score files are indexed; the portable evidence
ZIP includes the primary selected scores and all PenDigits writer losses.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(data):
    return (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(data))


def deterministic_tar(files, target):
    """Write and re-read every member against its original SHA-256."""
    files = sorted(files, key=lambda item: item[1])
    if len({name for _, name in files}) != len(files):
        raise ValueError('Duplicate archive paths')
    index = [{'path': name, 'sha256': digest(path), 'size_bytes': path.stat().st_size}
             for path, name in files]
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode='w') as archive:
                for path, name in files:
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                    with path.open('rb') as stream:
                        archive.addfile(info, stream)
                data = json_bytes(index)
                info = tarfile.TarInfo('archive_contents.json')
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
        raw.flush()
        os.fsync(raw.fileno())
    expected = {entry['path']: entry for entry in index}
    seen = set()
    with tarfile.open(temporary, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError('Only regular files are allowed in evidence archives')
            data = archive.extractfile(member).read()
            if member.name == 'archive_contents.json':
                if json.loads(data) != index:
                    raise ValueError('Archive index changed')
                continue
            entry = expected[member.name]
            if hashlib.sha256(data).hexdigest() != entry['sha256'] or len(data) != entry['size_bytes']:
                raise ValueError(f'Archive member differs: {member.name}')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Incomplete record archive')
    temporary.replace(target)
    return {'path': target.name, 'sha256': digest(target), 'size_bytes': target.stat().st_size,
            'n_members': len(index), 'all_member_hashes_verified': True}


def source_snapshot(study, output):
    manifest = load(study / 'manifest.json')
    files = {}
    for relative, expected in manifest['source_sha256'].items():
        path = PARENT / relative[len('parent/'):] if relative.startswith('parent/') else ROOT / relative
        if digest(path) != expected:
            raise ValueError(f'Fitting/analysis source changed after freeze: {relative}')
        target = relative[len('parent/'):] if relative.startswith('parent/') else 'extensions/certified_forecasting/' + relative
        files[target] = path
    prefix = 'extensions/certified_forecasting/'
    files[prefix + 'docs/MATCHED_GEOMETRY_PROTOCOL.md'] = study / 'protocol.md'
    for item in manifest['planned']:
        config = ROOT / 'configs' / Path(item['source_config']).name
        if digest(config) != item['source_config_sha256']:
            raise ValueError('Reference configuration changed after freeze')
        files[prefix + 'configs/' + config.name] = config
    for path in sorted((study / 'frozen_configs').rglob('*.json')):
        files[prefix + 'configs/frozen/' + path.relative_to(study / 'frozen_configs').as_posix()] = path
    files[prefix + 'requirements-reference.txt'] = ROOT / 'requirements-reference.txt'
    instructions = output / 'source_snapshot_README.txt'
    instructions.write_text(
        'Extract source_snapshot.tar.gz into an empty directory. It preserves the parent\n'
        'KAHM Python files and extensions/certified_forecasting/ ancestry required by imports.\n'
        'The three matched_geometry_*.json reference configs are included.\n\n'
        'cd extensions/certified_forecasting\n'
        'python -m pip install -r requirements-reference.txt\n'
        'python experiments/prepare_pendigits.py --help\n'
        'Prepare the pinned original PenDigits arrays using that script, then run:\n'
        'python experiments/run_matched_geometry.py --output results/reproduced_matched_geometry '
        '--workers 6 --prepared data_cache/pendigits/pendigits_original.npz\n'
        'python experiments/analyze_matched_geometry.py --study results/reproduced_matched_geometry '
        '--output results/reproduced_matched_geometry_analysis\n\n'
        'This snapshot recreates the experiment sources and protocol, not Git history.\n'
        'The frozen source/config hashes identify exact scientific inputs. Dependency pins\n'
        'are supplied as reproduction support. See the full repository for the manuscript,\n'
        'historical studies, tests, publication generators, and complete documentation.\n'
    )
    files['README_SOURCE_SNAPSHOT.txt'] = instructions
    return deterministic_tar([(p, name) for name, p in files.items()], output / 'source_snapshot.tar.gz')


def verify_analysis_binding(study, analysis):
    """Refuse stale, incomplete, or mismatched analysis/evidence inputs."""
    manifest = load(study / 'manifest.json')
    summary, audit = load(analysis / 'summary.json'), load(analysis / 'audit.json')
    protocol, completion = load(study / 'protocol.json'), load(study / 'completion.json')
    if summary['original_manifest_sha256'] != digest(study / 'manifest.json'):
        raise ValueError('Analysis refers to a different frozen study')
    if summary['analysis_source_sha256'] != manifest['source_sha256']['experiments/analyze_matched_geometry.py']:
        raise ValueError('Analysis source is not the frozen source')
    if audit['status'] != 'passed' or audit['allow_smoke'] or summary['allow_smoke']:
        raise ValueError('A complete passing reference audit is required')
    checks = list(load(study / 'freeze_sha256.json').items()) + list(audit['input_sha256'].items())
    for name, expected in checks:
        path = (study / name).resolve()
        if not path.is_relative_to(study) or digest(path) != expected:
            raise ValueError(f'Frozen or analyzed input changed: {name}')
    if digest(ROOT / 'docs/MATCHED_GEOMETRY_PROTOCOL.md') != manifest['protocol_sha256']:
        raise ValueError('The authoritative protocol changed after freeze')
    for relative, expected in manifest['source_sha256'].items():
        source = PARENT / relative[len('parent/'):] if relative.startswith('parent/') else ROOT / relative
        if digest(source) != expected:
            raise ValueError(f'Frozen scientific source changed: {relative}')
    if (completion['scheduled_jobs'] != 124 or completion['completed_jobs'] != 124
            or completion['failed_jobs'] != 0 or len(completion['results']) != 124
            or any(r['status'] != 'completed' for r in completion['results'])):
        raise ValueError('All 124 scheduled jobs must be completed without failures')
    scheduled = {job['id'] for job in load(study / 'schedule.json')}
    if len(scheduled) != 124 or {result['id'] for result in completion['results']} != scheduled:
        raise ValueError('Completion accounting differs from the frozen job identities')
    for result in completion['results']:
        destination = Path(result['destination'])
        if not destination.exists():
            parts = destination.parts
            destination = study.joinpath(*parts[parts.index('results'):])
        for name, expected in result['artifact_sha256'].items():
            if digest(destination / name) != expected:
                raise ValueError(f'Completed result artifact changed: {result["id"]}/{name}')
    if (summary['n_candidates'] != 13440 or audit['n_candidates'] != 13440
            or protocol['scheduled_candidate_fits'] != 13440 or not protocol['reference_design']):
        raise ValueError('Only the complete reference schedule can be packaged')

    selected = load(analysis / 'selected_score_index.json')
    expected = {(task, seed, geometry + '_' + head)
                for task in ('harmonic', 'quasiperiodic', 'nonlinear_ar', 'logistic', 'vanderpol', 'duffing')
                for seed in range(5701, 5706) for geometry in ('kahm', 'rbf', 'affine')
                for head in ('nested', 'frobenius', 'koopman')}
    observed = [(r['task'], r['seed'], r['family']) for r in selected]
    if len(observed) != 270 or set(observed) != expected or any('primary' not in r['selection_scopes'] for r in selected):
        raise ValueError('Selected evidence must cover all 270 primary synthetic models')
    with (analysis / 'selected_candidates.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['analysis'] == 'primary' and r['task'] != 'pendigits']
    csv_keys = {(r['task'], int(r['seed']), r['family'], r['mode'], r['candidate']) for r in rows}
    index_keys = {(r['task'], r['seed'], r['family'], r['mode'], r['candidate']) for r in selected}
    if len(rows) != 810 or csv_keys != index_keys:
        raise ValueError('Selected evidence index differs from the primary selection table')
    return summary, audit, protocol


def zip_write(archive, name, data, stored=False):
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    archive.writestr(info, data)


def evidence_zip(study, analysis, retained, deliverable):
    selections = load(analysis / 'selected_score_index.json')
    grouped = defaultdict(list)
    for selection in selections:
        grouped[selection['source_archive_relative_path']].append(selection)
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    temporary = deliverable.with_suffix(deliverable.suffix + '.tmp')
    primary_index, member_hashes = [], {}
    n_arrays, n_values = 0, 0
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        def add(name, data, stored=False):
            if name in member_hashes:
                raise ValueError(f'Duplicate ZIP member: {name}')
            member_hashes[name] = hashlib.sha256(data).hexdigest()
            zip_write(archive, name, data, stored)

        for path in sorted(retained.rglob('*')):
            if path.is_file():
                add('retained/' + path.relative_to(retained).as_posix(), path.read_bytes(),
                    stored=path.suffix == '.gz')
        for relative, entries in sorted(grouped.items()):
            source = study / relative
            result = load(study / entries[0]['source_result_relative_path'])
            descriptions = {row['name']: row for row in result['results']}
            arrays = {}
            with np.load(source, allow_pickle=False) as saved:
                for entry in entries:
                    if set(entry['array_keys']) != {entry['candidate'] + '__' + suffix
                                                  for suffix in ('cal_loss', 'pair_score', 'test_loss')}:
                        raise ValueError('Unexpected selected-array contract')
                    for key in entry['array_keys']:
                        values = saved[key].copy()
                        if values.dtype != np.dtype('float64') or not np.isfinite(values).all():
                            raise ValueError(f'Invalid evidence array: {relative}:{key}')
                        if key in arrays:
                            np.testing.assert_array_equal(arrays[key], values)
                        else:
                            arrays[key] = values
                            n_arrays += 1
                            n_values += values.size
                    np.testing.assert_array_equal(
                        arrays[entry['candidate'] + '__test_loss'].mean(axis=0),
                        descriptions[entry['candidate']]['test_mse'])
                    primary_index.append(entry | {'source_archive_sha256': digest(source)})
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            payload = buffer.getvalue()
            with np.load(io.BytesIO(payload), allow_pickle=False) as checked:
                if set(checked.files) != set(arrays):
                    raise ValueError('Selected array package is incomplete')
                for key, value in arrays.items():
                    np.testing.assert_array_equal(checked[key], value)
            add('selected_scores/' + relative, payload, stored=True)
        add('selected_score_index.json', json_bytes(primary_index))
        add('README.txt', (
            'Matched KAHM geometry study: complete retained records and primary evidence.\n\n'
            'retained/records.tar.gz contains every scalar candidate record, every PenDigits\n'
            'writer loss, frozen configurations, manifests, logs, and completion accounting.\n'
            'retained/source_snapshot.tar.gz contains exact frozen fitting and analysis source.\n'
            'retained/analysis/ contains all comparisons, including regressions and unresolved bounds.\n'
            'selected_scores/ retains float64 calibration losses, pair witnesses, and test losses\n'
            'for the validation-selected primary and head-selected synthetic models.\n'
            'The complete archive index identifies all full synthetic score files. Those larger\n'
            'files can be regenerated with experiments/run_matched_geometry.py.\n'
            'All model choices use validation data. PenDigits repeats use the same writer cohort.\n'
            'See the frozen protocol for the exact comparison and confidence scopes.\n'
        ).encode())
        zip_write(archive, 'zip_member_sha256.json', json_bytes(member_hashes))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Evidence ZIP CRC failure')
        if set(archive.namelist()) != set(member_hashes) | {'zip_member_sha256.json'}:
            raise ValueError('Evidence ZIP member set differs')
        for name, expected in member_hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Evidence ZIP member hash differs: {name}')
    temporary.replace(deliverable)
    return {'file': deliverable.name, 'sha256': digest(deliverable),
            'size_bytes': deliverable.stat().st_size, 'n_selected_models': len(primary_index),
            'n_selected_arrays': n_arrays, 'n_float64_values': n_values,
            'all_member_hashes_verified': True, 'selected_arrays_exact_to_sources': True}


def package(study, analysis, output, deliverable):
    if output.exists() or deliverable.exists():
        raise ValueError('Retained output and deliverable must be new paths')
    if (output.is_relative_to(study) or output.is_relative_to(analysis)
            or deliverable.is_relative_to(output) or deliverable.is_relative_to(study)
            or deliverable.is_relative_to(analysis)):
        raise ValueError('Output locations must not recursively include study, analysis, or deliverable')
    summary, audit, protocol = verify_analysis_binding(study, analysis)
    output.mkdir(parents=True)
    shutil.copytree(analysis, output / 'analysis')
    for name in ('manifest.json', 'protocol.md', 'protocol.json', 'schedule.json',
                 'freeze_sha256.json', 'completion.json'):
        shutil.copy2(study / name, output / name)
    shutil.copytree(study / 'frozen_configs', output / 'frozen_configs')
    records = [(p, p.relative_to(study).as_posix()) for p in sorted(study.rglob('*'))
               if p.is_file() and p.suffix in ('.json', '.csv', '.log', '.md', '.txt')
               and analysis not in p.parents]
    archived = deterministic_tar(records, output / 'records.tar.gz')
    source = source_snapshot(study, output)
    index = []
    for path in sorted((study / 'results').rglob('episode_scores.npz')):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Invalid full score archive: {path}')
            count = len(archive.namelist())
        index.append({'path': path.relative_to(study).as_posix(), 'sha256': digest(path),
                      'size_bytes': path.stat().st_size, 'n_arrays': count})
    if len(index) != 120:
        raise ValueError('Reference study requires all 120 synthetic score archives')
    write_json(output / 'episode_archive_index.json', index)
    write_json(output / 'packaging.json', {'records': archived, 'source_snapshot': source,
               'n_candidates': summary['n_candidates'], 'n_full_score_archives': len(index),
               'analysis_audit_sha256': digest(analysis / 'audit.json'),
               'full_scores_excluded_from_git': True})
    evidence = evidence_zip(study, analysis, output, deliverable)
    if evidence['n_selected_models'] != 270 or evidence['n_selected_arrays'] != 810:
        raise ValueError('Primary selected-score coverage is incomplete')
    write_json(output / 'evidence_package.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deliverable', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.study.resolve(), args.analysis.resolve(),
                             args.output.resolve(), args.deliverable.resolve()), indent=2))


if __name__ == '__main__':
    main()
