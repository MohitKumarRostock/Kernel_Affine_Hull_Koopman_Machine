#!/usr/bin/env python3
"""Package the explicitly unexecuted preproduction manuscript revision."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / 'soft_state_forecasting_source.zip'
SKIP_SUFFIX = {'.aux', '.blg', '.fdb_latexmk', '.fls', '.out', '.pyc'}


def eligible(path):
    rel = path.relative_to(ROOT)
    return (path.is_file() and '__pycache__' not in rel.parts
            and path.suffix not in SKIP_SUFFIX
            and str(rel) not in ('main.log', 'supplement.log'))


def main():
    record = ROOT / 'SOURCE_FILE_MANIFEST.json'
    prior = ROOT / 'revision_review/prior_metadata/ablation_input_SOURCE_FILE_MANIFEST.json'
    if not prior.exists():
        shutil.copy2(record, prior)
    for name in ('main_review.json', 'supplement_review.json'):
        receipt = json.loads((ROOT / 'build_evidence/ablation_preproduction_20261102' / name).read_text())
        if name == 'main_review.json':
            good = receipt['result'].startswith('No unresolved material content or layout issue')
            actual = hashlib.sha256((ROOT / 'main.pdf').read_bytes()).hexdigest()
            good = good and receipt['pdf_sha256'] == actual
        else:
            good = receipt['status'] == 'passed' and receipt['issues'] == []
            actual = hashlib.sha256((ROOT / 'supplement.pdf').read_bytes()).hexdigest()
            good = good and receipt['pdf']['sha256'] == actual
        if not good:
            raise ValueError('Review receipt does not approve the current PDF: ' + name)
    files = sorted(p for p in ROOT.rglob('*') if eligible(p) and p != record)
    manifest = {
        'title': 'Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines',
        'revision_date': '2026-11-02',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Preproduction revision with frozen, unexecuted four-arm study and preserved completed evidence',
        'anonymous_review_upload': False,
        'baseline_archive_sha256': 'd49f0580a832f73258592016bfd6a87b04e5609f6021604371d34b1d9ea906e0',
        'entry_points': ['main.tex', 'supplement.tex'],
        'pages': {'main': 33, 'supplement': 90},
        'historical_source_commit': 'b1915813e98a59572a4bbb800adc2bd1f3fccc1b',
        'historical_results_and_repository_parent': 'b59410407a7cd34155aa1955fe0c801f62381364',
        'repository_publication_of_this_revision': False,
        'new_production_model_fits': False,
        'new_evaluation_observations': False,
        'artificial_preflight_only': True,
        'generated_files_preserved': 106,
        'evidence_files_preserved': 48,
        'inventory_excludes': ['SOURCE_FILE_MANIFEST.json', 'temporary TeX and Python outputs'],
        'files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},
    }
    record.write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(record)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, 'KAHM_Reconstruction_Prediction/' + str(path.relative_to(ROOT)))
    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity failure')
        for rel, meta in manifest['files'].items():
            content = archive.read('KAHM_Reconstruction_Prediction/' + rel)
            if hashlib.sha256(content).hexdigest() != meta['sha256']:
                raise ValueError('Archive inventory differs: ' + rel)
    print(json.dumps({'path': str(OUT), 'members': len(files), 'bytes': OUT.stat().st_size,
                      'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
