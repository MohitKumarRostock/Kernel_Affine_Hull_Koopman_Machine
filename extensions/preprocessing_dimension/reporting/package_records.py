#!/usr/bin/env python3
"""Preserve every numerical record in verified, bounded-size ZIP archives."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path('extensions/preprocessing_dimension')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def archive_one(number, files, output):
    name = f'KAHM_Preprocessing_Dimension_Records_{number + 1:02d}.zip'
    path = output / name
    if path.exists(): raise FileExistsError(path)
    members = []
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for original in sorted(files):
            relative = original.relative_to(ROOT)
            member = str(PREFIX / relative)
            original_hash = sha256(original)
            archive.write(original, member)
            info = archive.getinfo(member)
            members.append({'path': member, 'bytes': original.stat().st_size,
                            'sha256': original_hash, 'crc32': f'{info.CRC:08x}'})
    # Read every archived member back and compare its full original hash.
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != len(members): raise IOError('Archive member count changed')
        for member in members:
            h = hashlib.sha256(); count = 0
            with archive.open(member['path']) as stream:
                for b in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(b); count += len(b)
            if count != member['bytes'] or h.hexdigest() != member['sha256']:
                raise IOError('Archived numerical record changed: ' + member['path'])
    return {'file_name': name, 'bytes': path.stat().st_size, 'sha256': sha256(path),
            'repetitions': [2 * number, 2 * number + 1], 'members': members,
            'all_members_roundtrip_sha256_verified': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT.parents[2] / 'output/data')
    a = p.parse_args(); output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if not (ROOT / 'TARGET_RELEASE.json').is_file() or not (ROOT / 'analysis/analysis_summary.json').is_file():
        raise ValueError('Complete released analysis required before packaging')
    groups = [[] for _ in range(12)]
    for directory in ('prepared', 'results', 'predictions', 'execution'):
        for file in (ROOT / directory).rglob('*'):
            if not file.is_file(): continue
            if file.suffix == '.tmp': raise ValueError('Unfinished temporary artifact')
            match = re.search(r'__rep(\d+)', str(file.relative_to(ROOT)))
            index = int(match.group(1)) // 2 if match else 0
            groups[index].append(file)
    for name in ('freeze.json', 'FIT_LOCK.json', 'PREDICTION_LOCK.json', 'TARGET_RELEASE.json'):
        groups[0].append(ROOT / name)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(archive_one, i, files, output) for i, files in enumerate(groups)]
        records = []
        for future in futures:
            item = future.result(); records.append(item)
            print(json.dumps({'archive': item['file_name'], 'bytes': item['bytes'], 'members': len(item['members'])}), flush=True)
    if any(item['bytes'] >= 500_000_000 for item in records):
        raise ValueError('An archive exceeds the 500 MB delivery limit; preserve it and split before upload')
    manifest = {'schema': 'kahm-preprocessing-numerical-archives-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': '5cfd7a1745a52459b7b5045fe84803b9bbb2b56a',
        'restore_prefix': str(PREFIX), 'archives': records,
        'source': 'https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5cfd7a1745a52459b7b5045fe84803b9bbb2b56a',
        'task_count': 192, 'independent_repetitions': 24,
        'scope': 'Every prepared array, fitted state, forecast array, numerical diagnostic and execution receipt; code is available at the pinned Git commit.'}
    target = output / 'KAHM_Preprocessing_Dimension_Archive_Manifest.json'
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    readme = ('KAHM PREPROCESSING AND DIMENSION: REPRODUCIBILITY\n\n'
        'Numerical source: ' + manifest['source'] + '\n\n'
        'Obtain that source commit, preserving extensions/dimension_forecasting and\n'
        'extensions/preprocessing_dimension. The former provides the verified runtime.\n'
        'Extract all twelve Records ZIP files at the repository root. Their members\n'
        'restore under extensions/preprocessing_dimension. Archives are disjoint.\n'
        'The first archive additionally includes global manifests and stage seals.\n\n'
        'To replay the analysis after restoring the records, run from the study extension:\n'
        'python code/pipeline.py verify\n'
        'python analysis_code/analyze.py --root . --output verification/reanalysis\n\n'
        'Choose a new output directory; existing results are never overwritten.\n'
        'The protocol and README explain fresh computation from the frozen seeds.\n'
        'Recomputing those same seeds is a reproduction, not an independent study.\n\n'
        'Every archived member was read back and its complete SHA-256 matched to\n'
        'the original. The JSON manifest supplies archive/member checksums and sizes.\n')
    readme_path = output / 'README_Reproducibility.txt'; readme_path.write_text(readme)
    analysis_zip = output / 'KAHM_Preprocessing_Dimension_Analysis.zip'
    with zipfile.ZipFile(analysis_zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted((ROOT / 'analysis').rglob('*')):
            if file.is_file(): archive.write(file, str(Path('analysis') / file.relative_to(ROOT / 'analysis')))
        for file in (target, readme_path): archive.write(file, file.name)
        for file in (ROOT / 'audit').glob('*.json'): archive.write(file, 'audit/' + file.name)
        for file in (ROOT / 'reporting/figures').glob('*.json'): archive.write(file, 'figures/' + file.name)
    with zipfile.ZipFile(analysis_zip) as archive:
        if archive.testzip() is not None: raise IOError('Analysis archive CRC check failed')
    print(json.dumps({'status': 'complete', 'record_archives': len(records),
        'record_bytes': sum(r['bytes'] for r in records), 'analysis_archive': str(analysis_zip),
        'analysis_bytes': analysis_zip.stat().st_size, 'manifest': str(target)}), flush=True)


if __name__ == '__main__': main()
