#!/usr/bin/env python3
"""Publish lossless compact records after the complete analysis and audit.

This reporting-only operation preserves every original byte. It writes three
gzip copies, a summary of the completed full audit, the numerical archive
index/summary, a restoration manifest and a verification receipt. Existing
outputs are never replaced. No scientific implementation is imported or run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import zlib

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from package_numerical_records import (BLOCK, SOURCE_COMMIT, SOURCE_TREE, digest,
    json_bytes, read_json, require, safe_relative, validate_completed_study,
    verify_source_maps)
from restore_compact_records import FILES, restore

MANIFEST = 'reporting/compact_release_manifest.json'
VALIDATION = 'reporting/compact_release_validation.json'
AUDIT_SUMMARY = 'verification/independent_audit_summary.json'
INDEX_GZIP = 'reporting/numerical_archive_index.json.gz'
ARCHIVE_SUMMARY = 'reporting/numerical_archive_summary.json'


def identity(path):
    path = Path(path)
    before = path.stat()
    result = {'bytes': before.st_size, 'sha256': digest(path)}
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns),
            'Record changed while hashing: ' + str(path))
    return result


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == payload, 'Reporting output did not preserve its bytes')


def compress_exact(source, destination, expected):
    require(identity(source) == expected, 'Compression source differs from its verified identity')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        with gzip.GzipFile(filename='', mode='wb', fileobj=outgoing, mtime=0,
                           compresslevel=9) as zipped:
            shutil.copyfileobj(incoming, zipped, length=BLOCK)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    with destination.open('rb') as stream:
        header = stream.read(10)
    require(header[:3] == b'\x1f\x8b\x08' and header[4:8] == bytes(4)
            and not header[3] & 8, 'Gzip timestamp/name header is not deterministic')
    hashed, count = hashlib.sha256(), 0
    with gzip.open(destination, 'rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            hashed.update(block)
            count += len(block)
    require({'bytes': count, 'sha256': hashed.hexdigest()} == expected,
            'Gzip decompression does not reproduce the original record')
    require(identity(source) == expected, 'Compression changed its source record')
    return {**identity(destination), 'mtime': 0, 'original_filename_header': ''}


def verify_archive_index(index_path, documents, hashes):
    index, index_hash = read_json(index_path)
    require(index['schema'] == 'soft-state-joint-numerical-archive-index-v1'
            and index['status'] == 'complete' and index['reporting_only'] is True,
            'A completed numerical-archive index is required')
    require(index['tasks'] == 256 and index['independent_repetitions'] == 32
            and index['primary_family_size'] == 6 and index['D_forecast_arrays'] == 4096
            and index['C_forecast_arrays'] == 4096 and index['B_penalty_path_arrays'] == 76800,
            'Numerical archive coverage differs from the declared complete study')
    source = index['identity']
    require(source['source_commit'] == SOURCE_COMMIT and source['source_tree_sha'] == SOURCE_TREE,
            'Archive source is not the actual published scientific commit/tree')
    for field, name in (
            ('config_sha256', 'config.json'), ('freeze_sha256', 'freeze.json'),
            ('publication_sha256', 'PUBLICATION.json'),
            ('analysis_summary_sha256', 'analysis/analysis_summary.json'),
            ('independent_audit_sha256', 'verification/independent_audit.json')):
        require(source[field] == hashes[name], 'Archive snapshot no longer matches ' + name)
    require(source['repository'] == documents['PUBLICATION.json']['repository']
            and source['branch'] == documents['PUBLICATION.json']['branch'],
            'Archive repository or branch metadata differs')
    require(index['checks']['every_archive_member_reopened_and_sha256_verified'] is True
            and index['coverage']['all_requested_trees_complete'] is True,
            'Archive index does not record complete byte verification')
    for name, record in index['archives'].items():
        require(Path(name).name == name and name.endswith('.zip'), 'Unsafe archive filename')
        path = index_path.parent / name
        require(path.is_file() and not path.is_symlink(), 'A verified archive is absent: ' + name)
        require(identity(path) == {key: record[key] for key in ('bytes', 'sha256')},
                'A numerical archive changed after member verification: ' + name)
        require(record['bytes'] < index['maximum_archive_bytes_exclusive'] <= 500_000_000,
                'An archive violates its strict size limit')
    require(digest(index_path) == index_hash, 'Numerical archive index changed during inspection')
    return index, {'bytes': index_path.stat().st_size, 'sha256': index_hash}


def build(root, index_path):
    root, index_path = root.resolve(), index_path.resolve()
    # This gate opens no numerical payload until completed analysis and audit
    # identities have been established. Originals remain in place throughout.
    documents, hashes, _ = validate_completed_study(root)
    source_maps = verify_source_maps(root, documents['freeze.json'])
    index, index_identity = verify_archive_index(index_path, documents, hashes)
    full_audit = documents['verification/independent_audit.json']
    analysis = documents['analysis/analysis_summary.json']
    original_tables = {}
    for name, record in analysis['tables'].items():
        path = safe_relative(root, 'analysis/' + name)
        original_tables['analysis/' + name] = identity(path)
        require(original_tables['analysis/' + name]['sha256'] == record['sha256'],
                'An original analysis table differs from the completed analysis')
    originals = {name: identity(safe_relative(root, name)) for name in FILES}
    require(originals['verification/independent_audit.json']['sha256'] == hashes['verification/independent_audit.json'],
            'Full independent audit identity differs')
    targets = [name + '.gz' for name in FILES] + [AUDIT_SUMMARY, INDEX_GZIP, ARCHIVE_SUMMARY, MANIFEST, VALIDATION]
    for name in targets:
        require(not safe_relative(root, name).exists(), 'Refusing to replace compact-release output: ' + name)
    (root / 'reporting').mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.compact-release-', dir=root / 'reporting'))
    installed = []
    try:
        records = []
        for name in FILES:
            compressed = compress_exact(root / name, staging / (name + '.gz'), originals[name])
            records.append({'path': name, **originals[name],
                'identity_source': ('completed independent audit; exact full receipt' if name.startswith('verification/')
                                    else 'analysis/analysis_summary.json/tables'),
                'gzip': {'path': name + '.gz', **compressed}})
        gzip_index = compress_exact(index_path, staging / INDEX_GZIP, index_identity)
        audit_record = next(record for record in records if record['path'].startswith('verification/'))
        audit_summary = {key: value for key, value in full_audit.items() if key not in ('schema', 'tasks_detail')}
        audit_summary.update(schema='independent-soft-state-joint-audit-summary-v1',
            source_audit_schema=full_audit['schema'], summary_created_utc=datetime.now(timezone.utc).isoformat(),
            summary_only=True, omitted_fields=['tasks_detail'], full_audit=audit_record)
        write_new(staging / AUDIT_SUMMARY, json_bytes(audit_summary))
        archive_summary = {
            'schema': 'soft-state-joint-numerical-archive-public-summary-v1', 'status': 'complete',
            'identity': index['identity'], 'tasks': 256, 'independent_repetitions': 32,
            'primary_family_size': 6, 'D_forecast_arrays': 4096, 'C_forecast_arrays': 4096,
            'B_penalty_path_arrays': 76800, 'archive_layout': index['archive_layout'],
            'archives': {name: {key: record[key] for key in ('bytes', 'sha256', 'member_count')}
                         for name, record in index['archives'].items()},
            'full_index': {'name': index_path.name, **index_identity,
                          'compressed_file': Path(INDEX_GZIP).name,
                          'compressed_bytes': gzip_index['bytes'], 'compressed_sha256': gzip_index['sha256']},
            'availability': 'Numerical archives are separate verified author deliverables. Distribution links are recorded separately after persistent saving; no public download URL is asserted here.',
            'checks': index['checks'], 'limitations': index['limits'],
            'sealed_identical_temporary_copies': index['coverage']['sealed_identical_temporary_copies']}
        write_new(staging / ARCHIVE_SUMMARY, json_bytes(archive_summary))
        documentary = {name: {'bytes': (root / name).stat().st_size, 'sha256': sha}
                       for name, sha in hashes.items()}
        documentary.update(original_tables)
        reporting_files = {}
        for name in ('reporting/README.md', 'reporting/package_numerical_records.py',
                     'reporting/build_compact_release.py', 'reporting/restore_compact_records.py'):
            path = safe_relative(root, name)
            if path.is_file():
                reporting_files[name] = identity(path)
        for name in (AUDIT_SUMMARY, ARCHIVE_SUMMARY):
            reporting_files[name] = identity(staging / name)
        manifest = {
            'schema': 'soft-state-joint-compact-release-v1', 'status': 'complete', 'reporting_only': True,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Exact-byte reporting transformation after complete frozen analysis and independent audit; no scientific recomputation.',
            'source_identity': {**index['identity'], 'full_independent_audit_sha256': hashes['verification/independent_audit.json']},
            'compression': {'format': 'gzip', 'compression_level': 9, 'mtime': 0,
                            'original_filename_header': '', 'python': platform.python_version(), 'zlib': zlib.ZLIB_VERSION},
            'records': records, 'original_records_preserved': True,
            'documentary_inputs': documentary, 'reporting_files': reporting_files,
            'plain_analysis_files': [name for name in sorted(original_tables) if name not in FILES] + ['analysis/analysis_summary.json'],
            'restoration_utility': 'reporting/restore_compact_records.py',
            'limits': ['The full originals remain in the separate verified numerical archives and completed local evidence.',
                       'The summary omits only tasks_detail; it is not another audit execution.',
                       'Reporting tools and this manifest are outside the prospective scientific freeze.',
                       'No result-release commit or public numerical-download URL is asserted.']}
        write_new(staging / MANIFEST, json_bytes(manifest))
        for name in FILES:
            require(identity(root / name) == originals[name], 'Original record changed before compact publication')
        for name, expected in original_tables.items():
            require(identity(root / name) == expected, 'Original analysis table changed before compact publication')
        verify_source_maps(root, documents['freeze.json'])

        # All sources, originals and collisions have been checked before any
        # installation. Link only new paths and remove our own links on failure.
        for name in [target for target in targets if target != VALIDATION]:
            destination = safe_relative(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(staging / name, destination)
            installed.append((destination, (staging / name).stat().st_ino))
        restoration = restore(root, check_only=True)
        validation = {
            'schema': 'soft-state-joint-compact-release-validation-v1', 'status': 'passed', 'reporting_only': True,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'production_recomputed': False,
            'source_commit': SOURCE_COMMIT, 'source_tree_sha': SOURCE_TREE,
            'analysis_summary_sha256': hashes['analysis/analysis_summary.json'],
            'full_independent_audit_sha256': hashes['verification/independent_audit.json'],
            'compact_manifest': identity(root / MANIFEST),
            'all_original_analysis_table_hashes_unchanged': True,
            'analysis_table_count': len(original_tables), 'all_three_original_records_unchanged': True,
            'compression_checks': [{'path': record['path'], 'byte_exact_decompression': True,
                                    'mtime_zero_empty_filename': True} for record in records],
            'index_byte_exact_decompression': True,
            'scientific_source_maps_verified_unchanged': {key: value['file_count'] for key, value in source_maps.items()},
            'audit_summary_preserves_original_top_level_fields_except_schema_and_tasks_detail': True,
            'restoration_check_only': restoration,
            'reporting_files': reporting_files,
            'notes': ['Verification uses the actual completed records; no artificial or scientific data were generated.',
                      'The archive packager previously reopened and hashed every ZIP member; this stage verifies archive identities again.',
                      'No record was rounded, filtered, reformatted, replaced, or removed.']}
        write_new(staging / VALIDATION, json_bytes(validation))
        os.link(staging / VALIDATION, root / VALIDATION)
        installed.append((root / VALIDATION, (staging / VALIDATION).stat().st_ino))
        result = {'status': 'complete', 'reporting_only': True,
                  'files': {name: identity(root / name) for name in targets},
                  'original_records_preserved': True,
                  'restoration_check_only_status': restoration['status']}
    except Exception:
        for path, inode in reversed(installed):
            if path.exists() and path.stat().st_ino == inode:
                path.unlink()
        raise
    finally:
        shutil.rmtree(staging)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=ROOT)
    parser.add_argument('--archive-index', type=Path, required=True,
                        help='Completed numerical_archive_index.json beside all verified numerical ZIPs')
    args = parser.parse_args()
    print(json.dumps(build(args.study_root, args.archive_index), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
