#!/usr/bin/env python3
"""One-time cleanup for the revised risk-certificates manuscript."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
GROUP_TREES = {
    'controlled_calibration': 'b773c5e9a186ab44f624ff5ef8c55e90d04f7b85',
    'ecg_cohort': '76fa9afd87646424f58d763419f8af3166c3574c',
    'class_diagnosis': '796a78e045d5bfbad0f60ae20659b6690327fa3b',
    'deployment_cost': '47adb2e6ffa5ad82c0c56e058b94e2569fe37609',
    'public_screening': 'b3225c24057177c629fbda1096f6336900ef3f2f',
    'decoder_diagnostics': 'd740dd67f705fc95d0f15429ad9b3973dee253e4',
}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Expected exactly one occurrence: ' + old[:100])
    return text.replace(old, new, 1)

def remove_row(text, marker):
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if marker in line]
    if len(hits) != 1:
        raise RuntimeError('Expected one table row: ' + marker)
    del lines[hits[0]]
    return ''.join(lines)

manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
for row in manifest['files']:
    p = (ROOT / row['path']).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise RuntimeError('Missing or invalid manifest path: ' + row['path'])
    if p.stat().st_size != row['bytes'] or sha256(p) != row['sha256']:
        raise RuntimeError('Original input checksum mismatch: ' + row['path'])
for group, expected in GROUP_TREES.items():
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD:experiments/' + group], text=True).strip()
    if actual != expected:
        raise RuntimeError('Unexpected retained experiment revision: ' + group)
actual_groups = {p.name for p in (ROOT / 'experiments').iterdir() if p.is_dir()}
if actual_groups != set(GROUP_TREES) | {'affine_forecasting'}:
    raise RuntimeError('Unexpected experiment directory set: ' + str(actual_groups))

edits = {}
text = (ROOT / 'README.md').read_text()
text = replace_once(text, '# A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations', '# A Hierarchy of Risk Certificates for Representation-Based Forecasting')
text = remove_row(text, '| Are specified stored forecasts evaluated accurately?')
text = replace_once(text, 'Population-risk bounds, capacity calculations on a fixed dataset and numerical output checks answer different questions and require different assumptions.', 'Population-risk confidence bounds and empirical-risk bounds on fixed datasets answer different questions and require different assumptions.')
text = replace_once(text, 'runs the seven experiment groups', 'runs the six experiment groups')
text = replace_once(text, ' The affine forecast checker bounds numerical error relative to a fixed rational predictor evaluated on trusted stored features.', '')
old_source = next(line for line in text.splitlines() if line.startswith('The experiment source is available at '))
text = replace_once(text, old_source, 'This branch supplies the six experiment groups used by the revised manuscript. Cite the exact commit used for reproduction. Individual experiment directories retain the historical provenance of their source and numerical records. File identities are recorded in `MANIFEST.json`. The [manuscript alignment record](docs/MANUSCRIPT_ALIGNMENT.md) identifies the retained experiments and the scope of this cleanup.')
edits['README.md'] = text

text = (ROOT / 'REPRODUCIBILITY.md').read_text()
text = remove_row(text, '| Affine forecasting |')
text = replace_once(text, 'The results include zero joint deployment-utility passes, unsuccessful screening, mixed or failed predictor-transfer criteria, and monthly forecasting failures. The aggregate affine-forecasting criterion compares mean squared errors on a fixed evaluation dataset. It is not a statistical noninferiority test.', 'The results include zero joint deployment-utility passes, unsuccessful screening, and mixed or failed predictor-transfer criteria. These outcomes remain part of the retained evidence.')
edits['REPRODUCIBILITY.md'] = text

text = (ROOT / 'DATA.md').read_text()
if text.count('Predictor diagnostics and affine forecasting') != 3:
    raise RuntimeError('Unexpected dataset-use table')
text = text.replace('Predictor diagnostics and affine forecasting', 'Predictor diagnostics')
text = replace_once(text, ' For the Household power study using 2008 data, byte identity of the pinned mirror with the official archive was not established.', '')
edits['DATA.md'] = text

text = (ROOT / 'docs/EXPERIMENT_GUIDE.md').read_text()
text = replace_once(text, 'The experiments separate this question from two others: whether a selection rule improves deployment costs, and whether a stored prediction is evaluated accurately.', 'The experiments also evaluate whether selecting predictors using risk bounds improves forecasting accuracy or reduces computational cost.')
text = remove_row(text, '| Are the selected stored affine forecasts evaluated within their numerical error budgets?')
start = text.index('**Numerical-output checks.**')
end = text.index('**Accuracy and computational cost.**', start)
text = text[:start] + text[end:]
edits['docs/EXPERIMENT_GUIDE.md'] = text

text = (ROOT / 'reproduce.py').read_text()
text = replace_once(text, ",'affine_forecasting')", ')')
edits['reproduce.py'] = text

edits['docs/MANUSCRIPT_ALIGNMENT.md'] = '''# Alignment with the revised JMLR manuscript

This source tree supports **A Hierarchy of Risk Certificates for Representation-Based Forecasting**, by Mohit Kumar and Somayeh Kargaran, after removal of the single-stationary-trajectory contribution and the separate affine numerical-output appendix.

## Retained experiments

| Directory under `experiments/` | Manuscript evidence |
| --- | --- |
| `controlled_calibration` | Simultaneous population-risk confidence bounds using independent episodes |
| `ecg_cohort` | Empirical-risk lower bounds for the fixed PTB-XL dataset |
| `class_diagnosis` | Bike Sharing empirical-risk bounds for 54 predictor classes and 108 fitted predictors |
| `deployment_cost` | Forecast accuracy, storage, and inference time for the retained deployment comparisons |
| `public_screening` | Beijing and Bike Sharing predictor-class screening and fit-avoidance results |
| `decoder_diagnostics` | Retained predictor comparisons and Household 2010/Jena 2017 affine constraint ablations |

The source, inputs, and recorded outcomes inside these six experiment directories are unchanged. In particular, the affine constraint ablations remain: they are separate from the deleted three-period numerical-output experiment. Household, Jena, and Metro data used by retained predictor comparisons are also preserved.

## Removed material and source history

The current paper tree excludes `experiments/affine_forecasting` and `experiments/single_trajectory_certification`, their active reproduction commands, and their validation outputs. Historical commits are preserved for provenance; this cleanup does not rewrite Git history or remove material belonging to the separate feature-augmentation paper.

The consolidated source before cleanup was commit `8878ae4ebe922c0184aed1e0fbb2afbb52c304f3`. The former trajectory branch was at `02589f72a5694dfa89f1d37f89de105780b0872a`. All six retained experiment directories had identical Git tree hashes in those two revisions.

## Verification and citation

`MANIFEST.json` lists the SHA-256 hashes of the retained source, documentation, and numerical inputs. `validation/summary.json` records the numerical replay actually performed for the manifest identified in that report. A replay is a check of stored records, not a new model-training run or a new timing study.

Use the exact cleaned commit in the manuscript's repository links. The previously cited commit `4c09d24a27837d7f165b253f3ab4d3cf28b71c0e` is a historical snapshot and still includes the deleted affine experiment.

This alignment follows the uploaded manuscript source and the subsequent section and appendix deletions agreed in the manuscript revision. The final 49-page source was not available for a new line-by-line manuscript-to-code audit.
'''

for path, text in edits.items():
    if path != 'docs/MANUSCRIPT_ALIGNMENT.md' and any(s in text for s in ('experiments/affine_forecasting', 'experiments/single_trajectory_certification')):
        raise RuntimeError('Retired experiment reference remains in ' + path)

removed = [row for row in manifest['files'] if row['path'].startswith('experiments/affine_forecasting/')]
shutil.rmtree(ROOT / 'experiments/affine_forecasting')
if (ROOT / 'validation').exists():
    shutil.rmtree(ROOT / 'validation')
for path, text in edits.items():
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)

# The one-time maintenance driver and workflow are not part of the paper source.
(ROOT / '.github/workflows/manuscript-cleanup.yml').unlink()
Path(__file__).unlink()

paths = {row['path'] for row in manifest['files'] if not row['path'].startswith('experiments/affine_forecasting/')}
paths.add('docs/MANUSCRIPT_ALIGNMENT.md')
manifest['files'] = [{'path': name, 'bytes': (ROOT / name).stat().st_size, 'sha256': sha256(ROOT / name)} for name in sorted(paths)]
(ROOT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')

for group in GROUP_TREES:
    subprocess.run(['git', 'diff', '--exit-code', '--', 'experiments/' + group], check=True)
for name in ['reproduce.py', 'scripts/explain_bike_certificate.py'] + ['experiments/' + group + '/replay.py' for group in GROUP_TREES]:
    compile((ROOT / name).read_text(), name, 'exec')
for name in ['README.md', 'REPRODUCIBILITY.md', 'DATA.md', 'docs/EXPERIMENT_GUIDE.md']:
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', (ROOT / name).read_text()):
        if '://' in target or target.startswith('#'):
            continue
        target = unquote(target.split('#', 1)[0])
        if target and not ((ROOT / name).parent / target).exists():
            raise RuntimeError('Broken local documentation link: ' + name + ' -> ' + target)

retired_refs = []
for group in GROUP_TREES:
    for p in (ROOT / 'experiments' / group).rglob('*.py'):
        text = p.read_text(errors='replace')
        if 'experiments/affine_forecasting' in text or 'experiments/single_trajectory_certification' in text:
            retired_refs.append(str(p.relative_to(ROOT)))
if retired_refs:
    raise RuntimeError('Retained Python source refers to a deleted experiment: ' + str(retired_refs))

audit = {
    'status': 'PASS',
    'scope': 'Structural cleanup and retained-file integrity; numerical replay is reported separately in summary.json.',
    'original_consolidated_commit': '8878ae4ebe922c0184aed1e0fbb2afbb52c304f3',
    'original_trajectory_commit': '02589f72a5694dfa89f1d37f89de105780b0872a',
    'preserved_experiment_trees': GROUP_TREES,
    'removed_from_consolidated_tree': {'directory': 'experiments/affine_forecasting', 'manifest_files': len(removed), 'bytes': sum(row['bytes'] for row in removed)},
    'single_trajectory_directory_absent': not (ROOT / 'experiments/single_trajectory_certification').exists(),
    'manifest_files': len(manifest['files']),
    'manifest_sha256': sha256(ROOT / 'MANIFEST.json'),
    'local_documentation_links_checked': True,
    'entry_points_compile': True,
    'retained_experiment_sources_unchanged': True,
    'deleted_directory_import_path_scan_passed': True,
}
(Path(os.environ['RUNNER_TEMP']) / 'manuscript_cleanup_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
