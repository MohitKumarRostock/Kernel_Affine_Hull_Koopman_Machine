"""Write-once records, exact scoring, and fixed selector definitions for Step 42."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('household_power2010', 'jena_temperature2017')
STAGES = ('f1', 'f2', 'f3', 'final')
LAMBDA_GRID = (('1e-6', '1/1000000'), ('1e-4', '1/10000'),
               ('1e-2', '1/100'), ('1', '1'), ('100', '100'))


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_bytes(value))
    return {'path': relative(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def maps():
    return sorted(['raw_' + s for s in ('minmax01', 'standard')]
                  + ['nystrom_' + s + '_w' + str(w)
                     for s in ('minmax01', 'standard') for w in (8, 24)]
                  + ['kahm_' + s + '_cap' + str(k) + '_p' + str(p)
                     for s in ('minmax01', 'standard')
                     for k in (4, 12, 20) for p in (1, 2)])


def recipe(project, recenter, clip):
    return f'P{int(project)}R{int(recenter)}C{int(clip)}'


def configurations():
    result = []
    for name in maps():
        for token, ridge_lambda in LAMBDA_GRID:
            for project in (False, True):
                for recenter in (False, True):
                    for clip in (False, True):
                        base = name + '::lambda_' + token
                        result.append({'candidate_id': base + '::' + recipe(project, recenter, clip),
                                       'base_key': base, 'map': name,
                                       'family': name.split('_')[0],
                                       'scaling': 'minmax01' if '_minmax01' in name else 'standard',
                                       'lambda_token': token, 'ridge_lambda': ridge_lambda,
                                       'project': project, 'recenter': recenter, 'clip': clip})
    assert len(result) == 720
    return result


def exact_number(hex_value):
    value = float.fromhex(hex_value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite operational prediction')
    return F.from_float(value)


def score(predictions, targets, *, hex_predictions=False):
    if not targets or len(predictions) != len(targets):
        raise ValueError('Prediction/target roster mismatch')
    ps = [exact_number(x) if hex_predictions else F(x) for x in predictions]
    ys = list(map(F, targets))
    errors = [p-y for p, y in zip(ps, ys)]
    sse = sum((e*e for e in errors), F(0))
    mse = sse / len(ys)
    return {'n': len(ys), 'sse': str(sse), 'mse': str(mse),
            'rmse': math.sqrt(float(mse)),
            'mae': str(sum(map(abs, errors), F(0)) / len(ys))}


def selector_specs():
    specs = []
    for project in (False, True):
        for recenter in (False, True):
            for clip in (False, True):
                tag = recipe(project, recenter, clip)
                fixed = {'project': project, 'recenter': recenter, 'clip': clip}
                specs.append({'label': 'recipe_' + tag, 'fixed': fixed,
                              'pool_size': 90, 'tie_rule': 'lexical_candidate_id'})
                for family, count in [('raw', 10), ('nystrom', 20), ('kahm', 60)]:
                    specs.append({'label': family + '_recipe_' + tag,
                                  'fixed': {**fixed, 'family': family},
                                  'pool_size': count, 'tie_rule': 'lexical_candidate_id'})
                for name in maps():
                    specs.append({'label': 'map_' + name + '_recipe_' + tag,
                                  'fixed': {**fixed, 'map': name},
                                  'pool_size': 5, 'tie_rule': 'lexical_candidate_id'})
    specs.append({'label': 'original41_replay',
                  'fixed': {'project': True, 'recenter': True, 'clip': True,
                            'lambda_token': '1e-6'},
                  'pool_size': 18,
                  'tie_rule': 'legacy_f3_deployment_bytes_then_lexical_candidate_id'})
    assert len(specs) == 177
    return specs


def select(rows, spec):
    pool = [r for r in rows if all(r[k] == v for k, v in spec['fixed'].items())]
    if len(pool) != spec['pool_size']:
        raise ValueError('Selector pool differs from fixed specification')
    if spec['label'] == 'original41_replay':
        key = lambda r: (F(r['pooled_physical_mse']), r['legacy_f3_deployment_bytes'],
                         r['candidate_id'])
    else:
        key = lambda r: (F(r['pooled_physical_mse']), r['candidate_id'])
    winner = min(pool, key=key)
    return {'candidate_id': winner['candidate_id'], 'pool_size': len(pool),
            'pooled_physical_mse': winner['pooled_physical_mse'],
            'rule': spec['tie_rule'], 'fixed': spec['fixed']}


def assert_frozen():
    seal = read(ROOT / 'protocol_seal42.json')
    for path, expected in seal['bound_files'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Frozen dependency changed: ' + path)
    return seal
