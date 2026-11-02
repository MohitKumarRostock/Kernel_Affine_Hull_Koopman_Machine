#!/usr/bin/env python3
"""Build the completed revision from an actual, freshly extracted input ZIP.

Reporting only. The completed REVISION_STATUS and actual result commit must
already be installed. This helper never fits models, changes prose/status,
downloads dependencies, performs visual review, or writes document_verification.
It installs verified canonical PDF/BBL pairs only after both documents pass.

Run from the repository root after the completed-results publication::

    python papers/soft_state_forecasting/build_evidence/\
joint_selection_completed_20261102/verify_clean_build.py

The default work directory must not exist. Failed work is retained for diagnosis;
use a new --work-root for a subsequent attempt. Source-input hashes use exactly
package_revision.build_input_hashes, as required by the final package gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import fitz
from package_revision import (
    DEFAULT_STUDY, RECEIPT_RELATIVE, ROOT, ZIP_ROOT, build_input_hashes,
    check_preservation, eligible, read, require, sha, verify_scientific_status,
)


DEFAULT_WORK = Path('/workspace/scratch/125265929068/joint_completed_build')
RECEIPT_NAME = 'isolated_build_verification.json'
ENTRY_POINTS = ('main', 'supplement')
ROOT_BUILD_OUTPUTS = {stem + suffix for stem in ENTRY_POINTS
                      for suffix in ('.pdf', '.bbl', '.dvi', '.ps', '.xdv')}
STALE_PATTERNS = (
    ('unresolved_result_commit', r'RESULTS[\s\-\u2010-\u2015]+COMMIT[\s\-\u2010-\u2015]+PENDING'),
    ('pending_revision_status', r'joint_selection_review_checkpoint_production_pending'),
    ('pending_joint_study',
     r'\bjoint(?:[\s\-\u2010-\u2015]+selection|\s+(?:input[\s\-]+)?scaling'
     r'(?:\s*[-\u2010-\u2015]+\s*|\s+and\s+)dimension)'
     r'\s+(?:(?:study|experiment|revision|results?|production)\s+)?'
     r'(?:(?:is|are|remains?)\s+)?'
     r'(?:pending|blocked|unpublished|preproduction|not\s+yet\s+'
     r'(?:run|complete|completed|published|audited))\b'),
)
FINAL_LOG_PATTERNS = (
    ('overfull_box', r'Overfull\s+\\[hv]box'),
    ('undefined_reference_or_citation',
     r'(?:LaTeX|Package\s+natbib)\s+Warning:\s*'
     r'(?:Citation|Reference)[^\n]*(?:\n[^\n]*)?\bundefined\b'),
    ('undefined_summary', r'There were undefined (?:references|citations)'),
    ('rerun_crossreferences', r'Label\(s\) may have changed|Rerun to get (?:cross-references|outlines) right'),
    ('missing_bibliography', r'No file [^\n]*\.bbl|Please\s+\(?re\)?run\s+(?:Biber|BibTeX)'),
    ('multiply_defined_labels', r'multiply[- ]defined|multiply defined'),
    ('tex_error', r'^!|Undefined control sequence|Emergency stop|Fatal error occurred'),
    ('missing_glyph', r'Missing character:'),
)


def identity(path):
    path = Path(path)
    before = path.stat()
    result = {'bytes': before.st_size, 'sha256': sha(path)}
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns),
            'File changed during inspection: ' + str(path))
    return result


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def scan(text, patterns):
    findings = []
    for label, pattern in patterns:
        for match in re.finditer(pattern, text, re.I | re.M):
            start = text.rfind('\n', 0, match.start()) + 1
            end = text.find('\n', match.end())
            if end == -1:
                end = len(text)
            findings.append({'check': label, 'line': text.count('\n', 0, match.start()) + 1,
                             'excerpt': text[start:end][:500]})
    return findings


def stale_markers(text):
    normalized = unicodedata.normalize('NFKC', text).replace('\u00ad', '')
    return scan(normalized, STALE_PATTERNS)


def check_live_source(root, inputs):
    findings = []
    for name in inputs:
        if Path(name).suffix in ('.tex', '.bib'):
            for finding in stale_markers((root / name).read_text(encoding='utf-8')):
                findings.append({'path': name, **finding})
    require(not findings, 'Stale current joint-study marker(s): ' + json.dumps(findings))
    return findings


def source_snapshot(root, work, inputs):
    """Write, reopen, and extract every eligible input byte in a fresh tree."""
    files = []
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if name not in ROOT_BUILD_OUTPUTS and eligible(path, root):
            files.append((name, path))
    inventory = {name: identity(path) for name, path in files}
    require(all(name in inventory and inventory[name]['sha256'] == expected
                for name, expected in inputs.items()), 'A compiler input is missing from the source ZIP')
    archive = work / 'source_inputs.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for name, path in files:
            payload = path.read_bytes()
            require({'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()} == inventory[name],
                    'Source changed before archiving: ' + name)
            member = zipfile.ZipInfo(ZIP_ROOT + '/' + name, (1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            zipped.writestr(member, payload, compresslevel=9)
    with archive.open('rb') as stream:
        os.fsync(stream.fileno())
    extracted = work / 'extracted' / ZIP_ROOT
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zipped:
        require(len(zipped.infolist()) == len(inventory), 'Unexpected source ZIP member count')
        for member in zipped.infolist():
            prefix = ZIP_ROOT + '/'
            require(member.filename.startswith(prefix), 'Source ZIP member has the wrong root')
            relative = member.filename[len(prefix):]
            require(relative in inventory and not Path(relative).is_absolute() and
                    '..' not in Path(relative).parts, 'Unexpected or unsafe source ZIP member')
            payload = zipped.read(member)
            require({'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()} == inventory[relative],
                    'Source ZIP member failed byte verification: ' + relative)
            destination = extracted / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(payload)
            require(identity(destination) == inventory[relative], 'Extracted source differs: ' + relative)
    require(build_input_hashes(extracted) == inputs, 'Extracted compiler input map differs')
    require(all(not (extracted / name).exists() for name in ROOT_BUILD_OUTPUTS),
            'Compiled root artifacts were included in the clean input ZIP')
    write_new_json(work / 'source_input_inventory.json', {
        'schema': 'joint-revision-source-input-inventory-v1',
        'source_input_zip': identity(archive), 'zip_root': ZIP_ROOT,
        'files': inventory, 'excluded_root_build_outputs': sorted(ROOT_BUILD_OUTPUTS),
        'source_build_inputs': inputs,
    })
    sync_directory(work)
    return archive, extracted, inventory


def tool_version(program, option):
    result = subprocess.run([program, option], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True)
    return {'executable': shutil.which(program), 'version_output': result.stdout.strip()}


def compile_document(stem, extracted, work, stamp):
    log_dir = work / 'build_logs'
    log_dir.mkdir(exist_ok=True)
    transcript = log_dir / (stem + '_clean_build_' + stamp + '.txt')
    command = ['latexmk', '-norc', '-pdf', '-bibtex', '-interaction=nonstopmode',
               '-halt-on-error', '-file-line-error',
               '-pdflatex=pdflatex -no-shell-escape %O %S', stem + '.tex']
    environment = os.environ.copy()
    # No custom latexmk rc, shell escape, download, asset generator, or numerical
    # program is invoked. Installed TeX/BibTeX packages supply the dependencies.
    with transcript.open('xb') as stream:
        result = subprocess.run(command, cwd=extracted, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.flush()
        os.fsync(stream.fileno())
    require(result.returncode == 0, stem + ' latexmk failed; see ' + str(transcript))
    paths = {suffix: extracted / (stem + suffix) for suffix in ('.pdf', '.bbl', '.log', '.blg')}
    require(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()),
            'The clean build did not produce every PDF/BBL/TeX/BibTeX record for ' + stem)
    # Earlier latexmk passes may have temporary undefined citations. Only the
    # final TeX log governs the converged unresolved/overfull check.
    warnings = scan(paths['.log'].read_text(encoding='utf-8', errors='replace'), FINAL_LOG_PATTERNS)
    require(not warnings, stem + ' final TeX log has unresolved/overfull errors: ' + json.dumps(warnings))
    biblog = paths['.blg'].read_text(encoding='utf-8', errors='replace')
    bibliography_errors = scan(biblog, (
        ('bibtex_missing_entry', r'Warning--I didn.t find a database entry'),
        ('bibtex_error', r'I couldn.t open|I was expecting|Illegal,|Repeated entry|error messages?'),
    ))
    require(not bibliography_errors, stem + ' BibTeX errors: ' + json.dumps(bibliography_errors))
    with fitz.open(paths['.pdf']) as document:
        require(not document.is_encrypted and len(document) > 0, 'Unreadable or empty ' + stem + ' PDF')
        pages = len(document)
        text = '\n'.join(page.get_text() for page in document)
    stale = stale_markers(text)
    require(not stale, stem + ' PDF retains a stale joint-study marker: ' + json.dumps(stale))
    require('??' not in text, stem + ' PDF contains an unresolved ?? reference')
    final_log = log_dir / (stem + '_final_latex_' + stamp + '.txt')
    bibtex_log = log_dir / (stem + '_bibtex_' + stamp + '.txt')
    shutil.copyfile(paths['.log'], final_log)
    shutil.copyfile(paths['.blg'], bibtex_log)
    metadata = {**identity(paths['.pdf']), 'pages': pages,
                'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'no_unresolved_or_overfull_warnings': True,
                'unresolved_or_overfull': [], 'stale_status_markers': [],
                'unresolved_pdf_reference_markers': []}
    return {'pdf': paths['.pdf'], 'bbl': paths['.bbl'], 'pdf_metadata': metadata,
            'command': command, 'returncode': result.returncode,
            'logs': [transcript, final_log, bibtex_log]}


def stage_copy(source, destination):
    """Stage in the target directory for an atomic same-filesystem replace."""
    fd, name = tempfile.mkstemp(prefix='.joint-selection-clean-build-', dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as outgoing, source.open('rb') as incoming:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.chmod(temporary, 0o644)
        require(identity(temporary) == identity(source), 'Staged install differs: ' + str(source))
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def install(root, work, products, receipt):
    receipts = root / RECEIPT_RELATIVE
    receipts.mkdir(parents=True, exist_ok=True)
    pending_receipt = work / RECEIPT_NAME
    write_new_json(pending_receipt, receipt)
    pairs = []
    for stem, product in products.items():
        pairs.extend([(product['pdf'], root / (stem + '.pdf')),
                      (product['bbl'], root / (stem + '.bbl'))])
        pairs.extend((log, receipts / log.name) for log in product['logs'])
    pairs.append((pending_receipt, receipts / RECEIPT_NAME))
    previous = work / 'previous_canonical_outputs'
    previous.mkdir()
    staged, installed, backups = [], [], {}
    try:
        for source, destination in pairs:
            require(not destination.is_symlink(), 'Refusing a symlink install destination')
            if destination.exists():
                backup = previous / destination.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(destination, backup)
                require(identity(backup) == identity(destination), 'Previous output backup differs')
                backups[destination] = backup
            staged.append((stage_copy(source, destination), destination, identity(source)))
        for temporary, destination, expected in staged:
            os.replace(temporary, destination)
            installed.append(destination)
            require(identity(destination) == expected, 'Installed output differs: ' + str(destination))
        sync_directory(receipts)
        sync_directory(root)
    except BaseException:
        for destination in reversed(installed):
            if destination in backups:
                replacement = stage_copy(backups[destination], destination)
                os.replace(replacement, destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _, _ in staged:
            temporary.unlink(missing_ok=True)


def verify_clean_build(root, study, work):
    root, study, work = Path(root).resolve(), Path(study).resolve(), Path(work).resolve()
    require(not work.is_relative_to(root), 'Clean-build work must be outside the paper source tree')
    require(not work.exists(), 'A fresh, nonexistent work directory is required: ' + str(work))
    status = read(root / 'REVISION_STATUS.json')
    _, summary, audit, evidence_hashes = verify_scientific_status(root, study, status)
    preservation = check_preservation(root)
    inputs = build_input_hashes(root)
    source_markers = check_live_source(root, inputs)
    for program in ('latexmk', 'pdflatex', 'bibtex'):
        require(shutil.which(program) is not None, 'Required local tool is unavailable: ' + program)
    started = datetime.now(timezone.utc)
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    work.mkdir(parents=True)
    try:
        archive, extracted, inventory = source_snapshot(root, work, inputs)
        require(check_preservation(extracted) == preservation, 'Extracted historical payload differs')
        tools = {name: tool_version(name, option) for name, option in
                 (('latexmk', '-v'), ('pdflatex', '--version'), ('bibtex', '--version'))}
        products = {stem: compile_document(stem, extracted, work, stamp) for stem in ENTRY_POINTS}
        require(build_input_hashes(root) == inputs == build_input_hashes(extracted),
                'Compiler inputs changed during the clean build')
        for name, expected in inventory.items():
            require(identity(root / name) == expected, 'Paper input changed during the clean build: ' + name)
        require(read(root / 'REVISION_STATUS.json') == status, 'Revision status changed during the build')
        require(check_preservation(root) == preservation, 'Historical payload changed during the build')
        receipt = {
            'schema': 'joint-selection-completed-isolated-build-verification-v1',
            'status': 'passed', 'reporting_only': True,
            'started_utc': started.isoformat(), 'created_utc': datetime.now(timezone.utc).isoformat(),
            'method': 'Actual eligible source-input ZIP reopened and verified member-by-member, extracted fresh, and built sequentially with latexmk/BibTeX before canonical PDF/BBL installation.',
            'scientific_source_commit': status['scientific_source_commit'],
            'completed_results_commit': status['completed_results_commit'],
            'revision_status_sha256': sha(root / 'REVISION_STATUS.json'),
            'joint_asset_manifest_sha256': status['joint_asset_manifest_sha256'],
            'scientific_evidence_hashes': evidence_hashes,
            'tasks': summary['tasks'], 'audit_checks': audit['checks'],
            'helper_sha256': sha(SCRIPT),
            'packager_sha256': sha(SCRIPT.parent / 'package_revision.py'),
            'source_input_zip_sha256': sha(archive),
            'source_input_zip_bytes': archive.stat().st_size,
            'source_input_zip_path': str(archive),
            'source_input_file_count': len(inventory),
            'source_input_inventory_sha256': sha(work / 'source_input_inventory.json'),
            'source_build_inputs': inputs,
            'source_build_inputs_helper': 'package_revision.build_input_hashes',
            'source_build_inputs_unchanged': True,
            'root_PDF_and_BBL_inputs_excluded': True,
            'source_stale_status_markers': source_markers,
            'pdfs': {stem: product['pdf_metadata'] for stem, product in products.items()},
            'bibliographies': {stem: identity(product['bbl']) for stem, product in products.items()},
            'commands': {stem: {'argv': product['command'], 'returncode': product['returncode']}
                         for stem, product in products.items()},
            'logs': {log.name: identity(log) for product in products.values() for log in product['logs']},
            'tools': tools, 'preservation': preservation,
            'isolated_directory': str(extracted),
            'network_required': False, 'model_fitting_required': False,
            'shell_escape_enabled': False, 'latexmk_rc_files_loaded': False,
            'visual_review_performed': False, 'document_verification_written': False,
        }
        install(root, work, products, receipt)
        require(build_input_hashes(root) == inputs, 'Source input map differs after output installation')
        return {'status': 'passed', 'receipt': str(root / RECEIPT_RELATIVE / RECEIPT_NAME),
                'receipt_identity': identity(root / RECEIPT_RELATIVE / RECEIPT_NAME),
                'source_input_zip': {'path': str(archive), **identity(archive)},
                'pdfs': receipt['pdfs'], 'visual_review_performed': False}
    except Exception as error:
        failure = work / 'build_failure.json'
        if not failure.exists():
            write_new_json(failure, {'status': 'failed', 'reporting_only': True,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'error': str(error), 'visual_review_performed': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root', type=Path, default=ROOT)
    parser.add_argument('--study-root', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--work-root', type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    result = verify_clean_build(args.paper_root, args.study_root, args.work_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
