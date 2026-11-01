#!/usr/bin/env python3
"""Package both audited policy studies without refitting or changing evidence.

All archive members are complete files. Record parts are independent ZIPs,
not byte-split volumes. Frozen scientific code is supplied by the immutable
public Git commit; reporting helpers are supplied by the later results release.
This package preserves the complete generated numerical evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import zlib

HERE = Path(__file__).resolve()
REPOSITORY_ROOT = HERE.parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT.parent / 'deliverables'
SOURCE_COMMIT = 'b1915813e98a59572a4bbb800adc2bd1f3fccc1b'
EARLIER_SYNTHETIC_SOURCE_COMMIT = 'e41b6e4fb902e49c8438f1adf2a3e98cb429cddd'
SOURCE_BRANCH = 'research/kahm-policy-augmentation-2026-11-01'
REPOSITORY_URL = 'https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine'
M4_SOURCE_COMMIT = '6c1067e5a57161249b17289a565178dc7a3fb3ca'
PREFIX = 'KAHM_Policy_Augmentation'
MAX_PAYLOAD = 400 * 1024 * 1024
BLOCK = 1024 * 1024
RECORD_DIRS = ('prepared', 'results', 'predictions', 'execution', 'input_sources')
CODE_DIRS = ('code', 'analysis_code')
COMPRESSED_SUFFIXES = {'.npz', '.gz', '.zip', '.bz2', '.xz', '.zst', '.7z'}
SOURCE_SUFFIXES = {'.py', '.pyc', '.pyo', '.sh', '.ipynb', '.js', '.ts', '.c', '.cpp', '.h'}
STUDIES = (
    ('synthetic', 'extensions/policy_augmentation_20261101', 'verification/policy_audit.json', 192),
    ('natural', 'extensions/policy_augmentation_natural_20261101', 'audit/verification.json', 48),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b''):
            h.update(chunk)
    return h.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def safe_member(name):
    p = PurePosixPath(name)
    require(bool(name) and bool(p.parts) and not p.is_absolute() and '..' not in p.parts and '.' not in p.parts,
            'Unsafe archive member path: ' + name)
    require(str(p) == name and '\\' not in name and '\x00' not in name,
            'Noncanonical archive member path: ' + name)
    return name


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_json(path):
    return json.loads(Path(path).read_text())


def audit_gate(repository):
    """Require passed audits that bind the exact current analysis summaries."""
    status = []
    for study, relative, audit_relative, expected in STUDIES:
        root = repository / relative
        audit_path, summary_path = root / audit_relative, root / 'analysis/analysis_summary.json'
        require(audit_path.is_file(), 'Required completed audit is missing: ' + str(audit_path))
        audit = load_json(audit_path)
        require(audit.get('status') == 'passed', 'Required scientific audit has not passed: ' + study)
        require(audit.get('analysis_summary_sha256') == sha256(summary_path),
                'Audit does not bind the current analysis summary: ' + study)
        summary = load_json(summary_path)
        tables = ({name: record['sha256'] for name, record in summary['tables'].items()}
                  if study == 'synthetic' else summary['files'])
        for name, expected_hash in tables.items():
            safe_member(name)
            require(sha256(root / 'analysis' / name) == expected_hash,
                    'An audited analysis table changed: ' + study + '/' + name)
        for key, name in (('freeze_sha256', 'freeze.json'),
                          ('prediction_lock_sha256', 'PREDICTION_LOCK.json')):
            require(summary.get(key) == sha256(root / name),
                    'Analysis provenance changed: ' + study + '/' + name)
        if study == 'synthetic':
            require(audit.get('tasks') == expected and audit.get('forecast_arrays') == 192 * 50,
                    'Synthetic audit did not cover the complete declared roster')
            require(audit.get('independent_repetitions') == 24,
                    'Synthetic repetition count differs from the fixed study')
            require(audit.get('freeze_sha256') == sha256(root / 'freeze.json'),
                    'Synthetic audit source-freeze identity differs')
            release = load_json(root / 'TARGET_RELEASE.json')
            require(release.get('status') == 'released', 'Synthetic terminal targets were not released')
            require(summary.get('target_release_sha256') == sha256(root / 'TARGET_RELEASE.json'),
                    'Synthetic analysis does not bind the current target release')
            for key, name in (('fit_lock_sha256', 'FIT_LOCK.json'),
                              ('prediction_lock_sha256', 'PREDICTION_LOCK.json'),
                              ('d_manifest_sha256', 'prepared/d_manifest.json')):
                require(release[key] == sha256(root / name), 'Synthetic release identity differs: ' + name)
        else:
            require(audit.get('checks', {}).get('series') == expected
                    and audit.get('checks', {}).get('D_arrays') == 48 * 42,
                    'Natural audit did not cover the complete declared roster')
            require((root / 'prepared/target_manifest.json').is_file(),
                    'Natural terminal targets were not released')
            for key, name in (('fit_lock_sha256', 'FIT_LOCK.json'),
                              ('target_manifest_sha256', 'prepared/target_manifest.json')):
                require(summary.get(key) == sha256(root / name),
                        'Natural analysis provenance changed: ' + name)
        for name in ('FIT_LOCK.json', 'PREDICTION_LOCK.json'):
            seal = load_json(root / name)
            require(seal.get('status') == 'sealed' and seal.get('task_count') == expected,
                    'Complete scientific stage seal is missing: ' + study + '/' + name)
            require(seal.get('freeze_sha256') == sha256(root / 'freeze.json'),
                    'Stage seal has a different source freeze: ' + study + '/' + name)
        status.append({'study': study, 'root': relative, 'expected_units': expected,
            'audit_path': relative + '/' + audit_relative, 'audit_status': audit['status'],
            'audit_sha256': sha256(audit_path), 'analysis_summary_path': relative + '/analysis/analysis_summary.json',
            'analysis_summary_sha256': sha256(summary_path), 'freeze_sha256': sha256(root / 'freeze.json'),
            'audited_table_hash_checks': len(tables),
            'fit_lock_sha256': sha256(root / 'FIT_LOCK.json'),
            'prediction_lock_sha256': sha256(root / 'PREDICTION_LOCK.json')})
    return status


def classify(path, relative, sealed=()):
    """Production numerical directories are exhaustive; code/fixtures are excluded."""
    if relative.as_posix() in sealed:
        return 'records', 'explicit scientific seal requires preservation'
    parts, name = relative.parts, path.name
    lower = name.lower()
    if any(p in ('__pycache__', '.git', '.pytest_cache', 'tmp', 'temp', 'fixtures', 'fixture') for p in parts):
        return None, 'cache, temporary, or fixture directory'
    if lower.endswith(('.tmp', '.temp', '.partial', '.pyc', '.pyo')) or name.startswith('.'):
        return None, 'temporary, cache, or hidden file'
    if parts[0] in RECORD_DIRS:
        return 'records', 'complete production numerical directory'
    if parts[0] == 'reporting' and path.suffix.lower() in SOURCE_SUFFIXES:
        return None, ('reporting/reproduction helper supplied by the results release on '
                      + SOURCE_BRANCH + '; not asserted to belong to the frozen scientific-source commit')
    if parts[0] in CODE_DIRS or path.suffix.lower() in SOURCE_SUFFIXES:
        return None, 'source code supplied by immutable public Git commit'
    if 'fixture' in lower or 'manufactured' in lower or 'preflight' in lower:
        return None, 'manufactured fixture/preflight record; supplied by source checkout when required'
    if parts[0] == 'readiness' and name not in ('prior_seed_exclusions.json', 'independent_preproduction_review.json'):
        return None, 'preproduction readiness/fixture record'
    # Include every non-code metadata, analysis, audit, figure, text report,
    # and provenance file left in the two task-specific study roots.
    return 'analysis', 'study metadata, audit, analysis, or reporting artifact'


def describe_file(path, member, study, category, origin='repository'):
    require(path.is_file() and not path.is_symlink(), 'Expected a regular non-symlink file: ' + str(path))
    stat = path.stat()
    return {'source': path, 'source_origin': origin, 'path': safe_member(member),
            'original_relative_path': safe_member(member), 'study': study, 'category': category,
            'size_bytes': stat.st_size, 'sha256': sha256(path), 'source_mtime_ns': stat.st_mtime_ns}


def inventory(repository):
    included, excluded, directory_inventory = [], [], []
    for study, relative, _, _ in STUDIES:
        root = repository / relative
        require(root.is_dir(), 'Study root is missing: ' + str(root))
        sealed = set().union(*(load_json(root / name)['files']
                               for name in ('FIT_LOCK.json', 'PREDICTION_LOCK.json')))
        files = sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
        for file in files:
            rel = file.relative_to(root)
            category, reason = classify(file, rel, sealed)
            member = file.relative_to(repository).as_posix()
            if category is None:
                excluded.append({'path': member, 'study': study, 'reason': reason,
                                 'size_bytes': file.stat().st_size if file.is_file() else None})
                continue
            record = describe_file(file, member, study, category)
            if rel.as_posix() in sealed and classify(file, rel)[0] is None:
                record['retained_by_explicit_scientific_seal'] = True
            included.append(record)
        for directory in RECORD_DIRS:
            source_dir = root / directory
            if not source_dir.exists():
                if directory == 'input_sources' and study == 'synthetic':
                    continue
                raise ValueError('Required numerical record directory is missing: ' + str(source_dir))
            source_files = sorted(p for p in source_dir.rglob('*') if p.is_file())
            valid = [p for p in source_files if classify(p, p.relative_to(root), sealed)[0] is not None]
            paths = {p.relative_to(repository).as_posix() for p in valid}
            packaged = [r for r in included if r['path'] in paths]
            require(len(packaged) == len(paths), 'Record-directory inventory is incomplete')
            directory_inventory.append({'study': study, 'directory': relative + '/' + directory,
                'source_file_count': len(source_files), 'included_file_count': len(packaged),
                'excluded_cache_or_temporary_count': len(source_files) - len(valid),
                'included_bytes': sum(r['size_bytes'] for r in packaged),
                'relative_paths_sha256': hashlib.sha256(('\n'.join(sorted(paths)) + '\n').encode()).hexdigest()})
    require(len({r['path'] for r in included}) == len(included), 'Duplicate original-relative archive path')
    return included, excluded, directory_inventory


def verify_sealed_members(repository, members):
    index = {r['path']: r for r in members}
    checks = 0
    for _, relative, _, _ in STUDIES:
        for name in ('FIT_LOCK.json', 'PREDICTION_LOCK.json'):
            seal = load_json(repository / relative / name)
            for path, expected in seal['files'].items():
                member = relative + '/' + path
                require(member in index, 'A sealed scientific record is absent from the archive: ' + member)
                require(index[member]['sha256'] == expected, 'A sealed scientific record changed: ' + member)
                checks += 1
    return checks


def verify_prepared_members(repository, members):
    """Check saved input/target container identities without parsing arrays."""
    index = {r['path']: r for r in members}
    checks = 0
    for study, relative, _, expected_count in STUDIES:
        root = repository / relative
        names = (('abc_manifest.json', 'd_manifest.json') if study == 'synthetic'
                 else ('abc_manifest.json', 'query_manifest.json', 'target_manifest.json'))
        for name in names:
            manifest = load_json(root / 'prepared' / name)
            require(manifest['freeze_sha256'] == sha256(root / 'freeze.json'),
                    'Prepared manifest source freeze changed: ' + relative + '/' + name)
            require(len(manifest['tasks']) == expected_count,
                    'Prepared manifest task roster is incomplete: ' + relative + '/' + name)
            mapping = dict(manifest.get('base_files', {}))
            for task in manifest['tasks']:
                pairs = ((('input', 'sha256'),) if name == 'abc_manifest.json'
                         else (('query', 'query_sha256'), ('target', 'target_sha256')))
                if study == 'natural':
                    pairs = (('path', 'sha256'),)
                for path_key, hash_key in pairs:
                    path, expected_hash = task[path_key], task[hash_key]
                    require(path not in mapping or mapping[path] == expected_hash,
                            'Conflicting prepared container identities: ' + path)
                    mapping[path] = expected_hash
            for path, expected_hash in mapping.items():
                member = relative + '/' + safe_member(path)
                require(member in index and index[member]['sha256'] == expected_hash,
                        'A prepared input or target is missing or changed: ' + member)
                checks += 1
        if study == 'natural':
            for path, expected_hash in load_json(root / 'freeze.json')['input_files'].items():
                member = relative + '/' + safe_member(path)
                require(member in index and index[member]['sha256'] == expected_hash,
                        'A frozen natural source CSV is missing or changed: ' + member)
                checks += 1
    return checks


def split_records(records, maximum):
    groups, current, size = [], [], 0
    for record in sorted(records, key=lambda r: r['path']):
        require(record['size_bytes'] <= maximum, 'One record exceeds the allowed archive payload: ' + record['path'])
        if current and size + record['size_bytes'] > maximum:
            groups.append(current)
            current, size = [], 0
        current.append(record)
        size += record['size_bytes']
    if current:
        groups.append(current)
    return groups


def read_member_hashes(stream):
    h, crc, size = hashlib.sha256(), 0, 0
    for chunk in iter(lambda: stream.read(BLOCK), b''):
        h.update(chunk)
        crc = zlib.crc32(chunk, crc)
        size += len(chunk)
    return size, h.hexdigest(), crc & 0xffffffff


def make_archive(destination, records):
    require(not destination.exists(), 'Refusing to overwrite an existing deliverable: ' + str(destination))
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4, allowZip64=True, strict_timestamps=False) as archive:
        for record in records:
            source = record['source']
            st = source.stat()
            require(st.st_size == record['size_bytes'] and st.st_mtime_ns == record['source_mtime_ns'],
                    'Source changed during packaging: ' + str(source))
            kind = zipfile.ZIP_STORED if source.suffix.lower() in COMPRESSED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(source, arcname=record['path'], compress_type=kind,
                          compresslevel=None if kind == zipfile.ZIP_STORED else 4)
            record['archive'] = destination.name
            record['compression'] = 'stored' if kind == zipfile.ZIP_STORED else 'deflated-level-4'
    expected = {r['path']: r for r in records}
    compression_counts = Counter()
    with zipfile.ZipFile(destination) as archive:
        infos = archive.infolist()
        require(len(infos) == len(expected) and {i.filename for i in infos} == set(expected),
                'ZIP member count or exact member roster differs')
        for info in infos:
            record = expected[info.filename]
            with archive.open(info) as stream:
                size, actual, crc = read_member_hashes(stream)
            require(size == info.file_size == record['size_bytes'], 'ZIP member size differs: ' + info.filename)
            require(actual == record['sha256'], 'ZIP bytes differ from source SHA-256: ' + info.filename)
            require(crc == info.CRC, 'ZIP member CRC differs: ' + info.filename)
            # A final source identity check also detects concurrent changes that
            # might otherwise leave a coherent ZIP with stale provenance.
            require(sha256(record['source']) == actual, 'Source bytes changed after ZIP creation: ' + info.filename)
            record['crc32'] = f'{crc:08x}'
            record['compressed_size_bytes'] = info.compress_size
            compression_counts[record['compression']] += 1
    return {'file_name': destination.name, 'size_bytes': destination.stat().st_size,
        'sha256': sha256(destination), 'member_count': len(records),
        'uncompressed_member_payload_bytes': sum(r['size_bytes'] for r in records),
        'compression_member_counts': dict(compression_counts),
        'complete_zip_crc_verified': True, 'every_member_sha256_and_size_verified_against_source': True,
        'archive_is_independently_readable': True}


def readme(source_commit, record_count):
    return f'''KAHM POLICY AUGMENTATION: VERIFIED RESEARCH ARCHIVE

Contents
--------
{PREFIX}_Analysis.zip contains the complete analysis tables, study protocols,
configuration, roster/provenance metadata, passed production audits, source
freezes, fitted/prediction seals, and reporting metrics available at packaging.
{PREFIX}_Records_01.zip through Records_{record_count:02d}.zip contain ALL generated
prepared arrays and targets, fitted states, B/C forecasts, basis records,
terminal forecasts, logs, and the natural study's official Train/metadata CSVs.
Each part is an ordinary independent ZIP containing complete files. No
byte-concatenation or multipart ZIP software is needed. Restore all parts to
reproduce the complete experiment; a single part is not the whole study.

The final PDF report is distributed separately. Source code and the complete
Git repository are not duplicated in these archives. Unsealed cache,
temporary, and manufactured fixture files are excluded and listed in the
manifest. Every file explicitly named by a scientific seal is retained,
including noncanonical filenames, so those exact seals remain verifiable.

Scientific source
-----------------
Repository: {REPOSITORY_URL}
Branch: {SOURCE_BRANCH}
Combined scientific source commit: {source_commit}
Earlier synthetic-only source publication: {EARLIER_SYNTHETIC_SOURCE_COMMIT}

The immutable combined source commit contains BOTH study runners and frozen
configurations. Its natural-source publication occurred after that evaluation
completed. The local pre-evaluation source freeze documents its actual
chronology; this is NOT a public preregistration. No prior-source date is
retrospectively substituted for the actual natural publication.

Restore into a fresh checkout
----------------------------
git clone {REPOSITORY_URL}.git kahm-policy-reproduction
cd kahm-policy-reproduction
git checkout {source_commit}

Using Python, extract the Analysis ZIP and every Records ZIP into this
repository root. Members preserve their original extensions/... paths.
Review {PREFIX}_Archive_Manifest.json before extraction. The source checkout
may already contain identical compact metadata; extract only to a fresh
reproduction checkout and preserve original ZIPs unchanged.

Verify archive files and every member without importing model code
-----------------------------------------------------------------
Run the following from the directory holding the downloaded archives and
the manifest. This uses only the Python standard library, verifies archive
SHA-256, exact membership, every member's SHA-256/size, and each ZIP CRC.

python3 - <<'PY'
import hashlib, json, pathlib, zipfile, zlib
base = pathlib.Path('.')
m = json.loads((base / '{PREFIX}_Archive_Manifest.json').read_text())
def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()
verified = 0
for a in m['archives']:
    assert pathlib.PurePosixPath(a['file_name']).name == a['file_name']
    path = base / a['file_name']
    assert path.stat().st_size == a['size_bytes']
    assert file_hash(path) == a['sha256']
    expected = {{r['path']: r for r in m['members'] if r['archive'] == a['file_name']}}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        assert len(infos) == len(expected) == a['member_count']
        assert {{i.filename for i in infos}} == set(expected)
        for info in infos:
            r = expected[info.filename]
            h, crc, size = hashlib.sha256(), 0, 0
            with z.open(info) as f:
                for b in iter(lambda: f.read(1048576), b''):
                    h.update(b); crc = zlib.crc32(b, crc); size += len(b)
            assert size == info.file_size == r['size_bytes']
            assert h.hexdigest() == r['sha256']
            assert (crc & 0xffffffff) == info.CRC == int(r['crc32'], 16)
            verified += 1
assert verified == m['member_count']
print('Verified archives:', len(m['archives']), 'members:', verified)
PY

Extract the verified members into the fresh checkout
--------------------------------------------------
Run this from the new repository root, replacing ../downloaded-archives with
the directory containing the archive files. Existing identical metadata from
the source checkout is retained. Any conflicting existing file stops restore.

python3 - ../downloaded-archives <<'PY'
import hashlib, json, pathlib, shutil, sys, zipfile
base, root = pathlib.Path(sys.argv[1]), pathlib.Path('.').resolve()
m = json.loads((base / '{PREFIX}_Archive_Manifest.json').read_text())
for a in m['archives']:
    assert pathlib.PurePosixPath(a['file_name']).name == a['file_name']
    expected = {{r['path']: r for r in m['members'] if r['archive'] == a['file_name']}}
    with zipfile.ZipFile(base / a['file_name']) as z:
        for info in z.infolist():
            name = pathlib.PurePosixPath(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            target = (root / name).resolve()
            assert target.is_relative_to(root)
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest() == expected[info.filename]['sha256']
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1048576)
print('Restored complete archive into', root)
PY

Recompute the synthetic analysis and audit from saved arrays
-----------------------------------------------------------
Install the declared requirements in an isolated Python environment. From
the restored repository root:

python extensions/policy_augmentation_20261101/analysis_code/analyze_policy.py --output extensions/policy_augmentation_20261101/analysis_recomputed
python extensions/policy_augmentation_20261101/audit/verify_policy.py --analysis extensions/policy_augmentation_20261101/analysis_recomputed --output extensions/policy_augmentation_20261101/verification/policy_audit_recomputed.json

Recompute the natural analysis and audit from saved arrays
---------------------------------------------------------
The frozen natural command intentionally refuses to overwrite outputs.
Preserve the extracted original outputs under different names, then run:

mv extensions/policy_augmentation_natural_20261101/analysis extensions/policy_augmentation_natural_20261101/analysis_saved
mv extensions/policy_augmentation_natural_20261101/audit/verification.json extensions/policy_augmentation_natural_20261101/audit/verification_saved.json
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py analyze
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py verify

This reanalysis does not fit models or create additional independent
observations. A new fit reproduction needs a separate clean output directory;
do not overwrite completed or failed execution records. Per-study README.md
and PROTOCOL.md specify fitting and prediction stages.

Data and inference scope
------------------------
Synthetic: 192 task suites, 24 new independent repetition aggregates, four
generating laws, two noise levels, history 12, horizons 1/4/12. Four fixed
simultaneous primary criteria are evaluated at the repetition level.
Natural: 24 Weekly and 24 Hourly M4 series identities, excluding all 72 previously
studied exact M4 identities. Its last 32 matured origins per series are reserved
within official Train files. The official M4 Test files are NOT downloaded.
The natural panel is descriptive external benchmark-development evidence;
anonymous source series can be dependent. It is not an official M4 ranking.
M4 input commit: {M4_SOURCE_COMMIT}
M4 source: https://github.com/Mcompetitions/M4-methods
M4 remains the original data source. Consult its original distribution for
applicable terms; this package does not assign a replacement data license.

Natural P0 deliberately omits the simulation's independent-episode OOF RBF
option and adds causal seasonal-naive and drift controls. Its results are not
pooled with the simulation. Both studies retain the established independently
validation-selected native caps/scalers and fixed calibration roles.

Manifest and integrity scope
----------------------------
{PREFIX}_Archive_Manifest.json records every member's original relative path,
size, SHA-256, CRC32, archive assignment, and compression. It records archive
hashes/sizes, all complete production-directory inventories, passed audit
identities, and source references. The manifest is external to the ZIP files
to avoid a circular self-hash. It is not itself a member it claims to hash.
Archive parts have at most 400 MiB of original member payload; NPZ and GZ
members remain ZIP_STORED, while text uses DEFLATE level 4. Packaging verifies
all member bytes against their source; it does not recompute model fits.
'''


def verify_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    m = load_json(manifest_path)
    members = m['members']
    require(len(members) == m['member_count'], 'Manifest member count differs')
    require(len({r['path'] for r in members}) == len(members), 'Duplicate manifest member path')
    verified = 0
    for archive in m['archives']:
        safe_member(archive['file_name'])
        require(PurePosixPath(archive['file_name']).name == archive['file_name'],
                'Archive file name must be a basename')
        path = manifest_path.parent / archive['file_name']
        require(path.stat().st_size == archive['size_bytes'] and sha256(path) == archive['sha256'],
                'Archive hash/size mismatch: ' + str(path))
        expected = {r['path']: r for r in members if r['archive'] == archive['file_name']}
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            require(len(infos) == len(expected) == archive['member_count']
                    and {i.filename for i in infos} == set(expected), 'Archive membership mismatch')
            for info in infos:
                safe_member(info.filename)
                with z.open(info) as stream:
                    size, sha, crc = read_member_hashes(stream)
                row = expected[info.filename]
                require(size == info.file_size == row['size_bytes'] and sha == row['sha256'],
                        'Member hash/size mismatch: ' + info.filename)
                require(crc == info.CRC == int(row['crc32'], 16), 'Member CRC mismatch: ' + info.filename)
                verified += 1
    require(verified == len(members), 'Some manifest members were not verified')
    return {'status': 'verified', 'archives': len(m['archives']), 'members': verified,
            'model_fits_performed': 0}


def package(repository, output, source_commit, maximum):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    require(re.fullmatch('[0-9a-f]{40}', source_commit) is not None, 'Use an exact forty-character source commit')
    require(0 < maximum <= MAX_PAYLOAD, 'Record payload maximum may not exceed 400 MiB')
    audits = audit_gate(repository)
    output.mkdir(parents=True, exist_ok=True)
    reserved = [output / (PREFIX + suffix) for suffix in
        ('_Analysis.zip', '_Archive_Manifest.json', '_README.txt', '_Source_References.json')]
    existing = [str(p) for p in reserved if p.exists()]
    existing += [str(p) for p in output.glob(PREFIX + '_Records_*.zip')]
    require(not existing, 'Refusing to overwrite existing deliverables: ' + ', '.join(existing))
    print(json.dumps({'stage': 'inventory', 'both_scientific_audits_passed': True}), flush=True)
    members, excluded, directory_inventory = inventory(repository)
    sealed_checks = verify_sealed_members(repository, members)
    prepared_checks = verify_prepared_members(repository, members)
    record_groups = split_records([r for r in members if r['category'] == 'records'], maximum)
    require(record_groups, 'No numerical records were found')

    references_path = output / (PREFIX + '_Source_References.json')
    references = {'repository': REPOSITORY_URL, 'branch': SOURCE_BRANCH,
        'source_commit': source_commit, 'source_url': REPOSITORY_URL + '/tree/' + source_commit,
        'earlier_synthetic_only_source_commit': EARLIER_SYNTHETIC_SOURCE_COMMIT,
        'natural_publication_timing': 'Combined public scientific source published after completed natural evaluation',
        'public_preregistration': False, 'chronology_evidence': 'Actual local source freezes and stage seals',
        'M4_source_commit': M4_SOURCE_COMMIT, 'M4_url': 'https://github.com/Mcompetitions/M4-methods',
        'studies': audits, 'packaging_script_sha256': sha256(HERE), 'model_fits_performed': 0}
    exclusive_json(references_path, references)
    readme_path = output / (PREFIX + '_README.txt')
    with readme_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(readme(source_commit, len(record_groups)))
    for path in (readme_path, references_path):
        members.append(describe_file(path, path.name, 'combined', 'analysis', 'package_metadata'))
    optional_metrics = sorted(set(output.glob(PREFIX + '*Metrics*.json')))
    for path in optional_metrics:
        members.append(describe_file(path, path.name, 'combined', 'analysis', 'report_metrics'))
    require(len({r['path'] for r in members}) == len(members), 'Combined archive paths are duplicated')

    archive_records = []
    analysis_members = sorted((r for r in members if r['category'] == 'analysis'), key=lambda r: r['path'])
    analysis_path = output / (PREFIX + '_Analysis.zip')
    archive_records.append(make_archive(analysis_path, analysis_members))
    print(json.dumps({'stage': 'archive_verified', **archive_records[-1]}), flush=True)
    for number, group in enumerate(record_groups, 1):
        path = output / f'{PREFIX}_Records_{number:02d}.zip'
        result = make_archive(path, group)
        require(result['uncompressed_member_payload_bytes'] <= maximum, 'Record archive exceeds payload maximum')
        archive_records.append(result)
        print(json.dumps({'stage': 'archive_verified', **result}), flush=True)

    # Re-scan the production directories after ZIP verification. New, removed,
    # or changed files invalidate the package instead of being silently lost.
    for entry in directory_inventory:
        directory = repository / entry['directory']
        study_root = next(repository / relative for study, relative, _, _ in STUDIES if study == entry['study'])
        sealed = set().union(*(load_json(study_root / name)['files']
                               for name in ('FIT_LOCK.json', 'PREDICTION_LOCK.json')))
        files = [p for p in directory.rglob('*') if p.is_file()]
        current = {p.relative_to(repository).as_posix() for p in files
                   if classify(p, p.relative_to(study_root), sealed)[0] is not None}
        expected = {r['path'] for r in members if r['path'].startswith(entry['directory'] + '/')}
        require(current == expected and len(current) == entry['included_file_count'],
                'Production record directory changed during packaging: ' + entry['directory'])
    require(sum(a['member_count'] for a in archive_records) == len(members),
            'Archive totals do not cover the complete member roster')
    public_members = []
    for row in sorted(members, key=lambda r: r['path']):
        public_members.append({k: v for k, v in row.items() if k not in ('source', 'source_mtime_ns')})
    manifest = {'schema': 'kahm-policy-augmentation-complete-archives-v1', 'created_utc': timestamp(),
        'status': 'complete_and_verified', 'repository': REPOSITORY_URL, 'source_branch': SOURCE_BRANCH,
        'source_commit': source_commit, 'earlier_synthetic_only_source_commit': EARLIER_SYNTHETIC_SOURCE_COMMIT,
        'source_references_file': references_path.name, 'studies': audits,
        'public_preregistration': False,
        'natural_source_publication_after_evaluation': True,
        'member_count': len(public_members), 'archive_count': len(archive_records),
        'record_archive_count': len(record_groups), 'record_archive_payload_limit_bytes': maximum,
        'total_original_member_bytes': sum(r['size_bytes'] for r in public_members),
        'archives': archive_records, 'members': public_members,
        'complete_record_directory_inventory': directory_inventory,
        'sealed_scientific_member_hash_checks': sealed_checks,
        'prepared_input_target_and_source_file_hash_checks': prepared_checks,
        'excluded_files': excluded,
        'excluded_scope': 'Git-backed source code, unsealed caches/temporary files, manufactured fixtures; every explicitly sealed record is retained; older studies are outside the two selected roots',
        'verification': {'every_archive_member_count_and_path_checked': True,
            'every_member_source_sha256_size_and_zip_crc_checked': True,
            'every_sealed_fit_and_prediction_file_included': True,
            'production_directory_rosters_unchanged_after_packaging': True,
            'archive_sha256_recorded': True, 'model_fits_performed': 0},
        'manifest_scope': 'External manifest; not included as a self-hashed ZIP member',
        'packaging_script_sha256': sha256(HERE)}
    manifest_path = output / (PREFIX + '_Archive_Manifest.json')
    exclusive_json(manifest_path, manifest)
    return {'status': 'complete_and_verified', 'output': str(output),
        'archives': len(archive_records), 'record_archives': len(record_groups), 'members': len(public_members),
        'manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
        'total_archive_bytes': sum(a['size_bytes'] for a in archive_records), 'model_fits_performed': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=REPOSITORY_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--source-commit', default=SOURCE_COMMIT)
    parser.add_argument('--max-record-payload-mib', type=int, default=400)
    parser.add_argument('--verify-manifest', type=Path,
                        help='Verify an existing complete archive manifest; do not package or import model code')
    args = parser.parse_args()
    if args.verify_manifest:
        result = verify_manifest(args.verify_manifest)
    else:
        result = package(args.repository_root, args.output, args.source_commit,
                         args.max_record_payload_mib * 1024 * 1024)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
