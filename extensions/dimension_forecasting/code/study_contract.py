"""Portable source-freeze, task-roster and phase-boundary checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def expected_prediction_keys(config):
    keys=set()
    for mode in ('calibrated','gamma1'):
        for candidate in config['candidates']:
            keys.update(f"{candidate['candidate_id']}__{mode}__{method}"
                        for method in ['ridge',*config['geometry_methods']])
        keys.update(f'shared__{mode}__{method}' for method in config['shared_methods'])
        for context in config['contexts']:
            keys.update(f'context_{context}__{mode}__{method}' for method in config['methods'])
        keys.update(f'matched__{mode}__{method}' for method in config['geometry_methods'])
    return keys


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def seed_from(parts):
    return int(np.random.SeedSequence([int(x) for x in parts]).generate_state(1)[0])


def seed_records(config):
    master=int(config['master_seed']);purposes=config['seed_purposes'];records=[]
    for repetition in range(int(config['repetitions'])):
        parts=[master,repetition,purposes['identifier']]
        records.append({'role':'identifier','repetition':repetition,'components':parts,'seed':seed_from(parts)})
        for law_index,law in enumerate(config['laws']):
            parts=[master,repetition,law_index,purposes['fit']]
            records.append({'role':'fit','repetition':repetition,'law':law,'components':parts,'seed':seed_from(parts)})
            for split,split_index in config['split_indices'].items():
                for purpose in ('generation','unit_noise','rows'):
                    parts=[master,repetition,law_index,split_index,purposes[purpose]]
                    records.append({'role':purpose,'repetition':repetition,'law':law,'split':split,
                                    'components':parts,'seed':seed_from(parts)})
    values=[r['seed'] for r in records]
    if len(values)!=len(set(values)):
        raise ValueError('An intended-distinct uint32 seed collision occurred before generation')
    expected= int(config['repetitions'])*(1+len(config['laws'])*(1+len(config['split_indices'])*3))
    if len(values)!=expected: raise AssertionError('Seed roster count mismatch')
    exclusion=config.get('seed_derivation',{}).get('excluded_seed_manifest')
    if exclusion:
        root=Path(__file__).resolve().parents[1]
        prior=json.loads(_safe_path(root,exclusion).read_text())
        collision=set(values)&{int(r['seed']) for r in prior['records']}
        if collision:raise ValueError('Fresh streams collide with a prior study seed: '+repr(sorted(collision)))
    return records


def task_roster(config):
    records=seed_records(config)
    identifiers={r['repetition']:r['seed'] for r in records if r['role']=='identifier'}
    fits={(r['repetition'],r['law']):r['seed'] for r in records if r['role']=='fit'}
    tasks=[]
    for repetition in range(int(config['repetitions'])):
        for law_index,law in enumerate(config['laws']):
            for history in config['histories']:
                for noise_index,noise_level in enumerate(config['noise_levels']):
                    tasks.append({'task_id':f'{law}__h{history}__noise{noise_index}__rep{repetition:02d}',
                        'law':law,'law_index':law_index,'history':int(history),
                        'seed':identifiers[repetition],'repetition':repetition,
                        'noise_index':noise_index,'noise_level':float(noise_level),
                        'fit_seed':fits[(repetition,law)]})
    if len(tasks)!=config['task_count']:raise ValueError('Config task_count differs from declared factorial')
    if len({t['task_id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate task IDs')
    return tasks


def _safe_path(root,relative):
    relative=Path(relative);path=(root/relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
        raise ValueError('Lock path escapes study root')
    return path


def verify_file_map(root,files):
    root=Path(root).resolve()
    if not isinstance(files,dict) or not files:raise ValueError('Nonempty files mapping required')
    for relative,expected in files.items():
        path=_safe_path(root,relative)
        if not path.is_file() or digest(path)!=expected:raise ValueError('Locked file changed: '+relative)


def verify_freeze(root,freeze_file):
    root=Path(root).resolve();config=json.loads((root/'config.json').read_text())
    freeze=json.loads(Path(freeze_file).read_text())
    if freeze.get('status')!='frozen':raise ValueError('Source/configuration freeze is not final')
    if freeze.get('config_sha256')!=digest(root/'config.json'):raise ValueError('Frozen configuration changed')
    if freeze.get('task_count')!=len(task_roster(config)):raise ValueError('Frozen task count differs')
    if (root/'source_distribution.json').exists():
        from source_distribution import verify_source_distribution
        verification=verify_source_distribution(root,freeze)
        freeze=dict(freeze)
        freeze['distribution_verification']=verification
    else:
        verify_file_map(root,freeze['files'])
    return freeze


def verify_abc_manifest(root,freeze_file,*,hash_inputs=True):
    root=Path(root).resolve();config=json.loads((root/'config.json').read_text())
    manifest=json.loads((root/'prepared/abc_manifest.json').read_text())
    if manifest['config_sha256']!=digest(root/'config.json'):raise ValueError('ABC configuration identity differs')
    if manifest['freeze_sha256']!=digest(freeze_file):raise ValueError('ABC freeze identity differs')
    expected={t['task_id']:t for t in task_roster(config)}
    actual={t['task_id']:t for t in manifest['tasks']}
    if set(actual)!=set(expected) or len(actual)!=len(manifest['tasks']):raise ValueError('Incomplete or duplicated ABC roster')
    for key,task in actual.items():
        if any(task[field]!=value for field,value in expected[key].items()):raise ValueError('ABC task metadata differs: '+key)
        if hash_inputs and digest(_safe_path(root,task['input']))!=task['sha256']:
            raise ValueError('ABC task input changed: '+key)
    return manifest


def verify_fit_lock(root,fit_lock,freeze_file,*,full_hash=True,task_id=None):
    root=Path(root).resolve();config=json.loads((root/'config.json').read_text())
    lock=json.loads(Path(fit_lock).read_text())
    if lock.get('status')!='sealed' or lock.get('phase')!='fits':raise ValueError('Complete fit lock required')
    if lock.get('config_sha256')!=digest(root/'config.json') or lock.get('freeze_sha256')!=digest(freeze_file):
        raise ValueError('Fit lock configuration/freeze identity differs')
    expected={t['task_id'] for t in task_roster(config)}
    if lock.get('task_count')!=len(expected) or set(lock.get('task_ids',[]))!=expected or len(lock.get('task_ids',[]))!=len(expected):
        raise ValueError('Fit lock does not contain every expected task')
    files=lock['files']
    # Every child checks the complete canonical roster and its lock membership.
    # The stage runner/root seal verify all file bytes globally; a child parses
    # only its own summary to avoid repeated all-task JSON work.
    required=['fit_summary.json','validation_predictions.npz','basis_diagnostics.npz',
              'pre_c_selection.json','calibration_predictions.npz']
    required += [f"states/{candidate['candidate_id']}.pkl.gz" for candidate in config['candidates']]
    required += ['states/shared.pkl.gz']
    for identity in expected:
        if any(f'results/{identity}/{name}' not in files for name in required):
            raise ValueError('Canonical fitted artifacts missing from global lock: '+identity)
    if task_id is not None and task_id not in expected:raise ValueError('Query task is outside fitted roster')
    inspect=expected if task_id is None else (task_id,)
    for identity in inspect:
        relative=f'results/{identity}/fit_summary.json'
        path=_safe_path(root,relative)
        if digest(path)!=files[relative]:raise ValueError('Locked fit summary changed: '+identity)
        summary=json.loads(path.read_text())
        if summary['status']!='completed' or summary['task']['task_id']!=identity:
            raise ValueError('Incomplete or mismatched fitted task')
        if len(summary['candidates'])!=len(config['candidates']):raise ValueError('Incomplete candidate roster')
        for local,sha in summary['files'].items():
            if files.get(f'results/{identity}/{local}')!=sha:
                raise ValueError('Fitted artifact missing from global lock: '+local)
    if full_hash:verify_file_map(root,files)
    return lock
