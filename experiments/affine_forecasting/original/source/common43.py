"""Shared serialization, immutable records and event ancestry; no model or data code."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('household_power2008', 'jena_temperature2018', 'metro_traffic2018')
STAGES = ('f1', 'f2', 'f3', 'final')
SEEDS = (3901, 4301, 4302)
POLICIES = ('raw', 'nystrom', 'kahm', 'all_map', 'ordinary', 'ridge')


def canonical(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, 'tolist'):
        return canonical(value.tolist())
    if hasattr(value, 'item'):
        return canonical(value.item())
    return value


def canonical_bytes(value):
    return (json.dumps(canonical(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError('Retain the existing record: ' + str(path))
    if overwrite:
        temporary = path.with_name(path.name + '.pending')
        temporary.write_bytes(canonical_bytes(value))
        temporary.replace(path)
    else:
        with path.open('xb') as stream:
            stream.write(canonical_bytes(value))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_ref(path, *, root=ROOT):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha_file(path), 'bytes': path.stat().st_size}


def seal_files(path, files, *, schema, root=ROOT, **extra):
    files = sorted(set(Path(x) if Path(x).is_absolute() else root / x for x in files))
    record = {'schema': schema, 'utc': utc_now(),
              'bound_files': {x.relative_to(root).as_posix(): sha_file(x) for x in files}, **extra}
    write_json(path if Path(path).is_absolute() else root / path, record)
    return record


def verify_bindings(bindings, *, root=ROOT):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Missing file bindings')
    for name, expected in bindings.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe bound path')
        if sha_file(root / relative) != expected:
            raise ValueError('Bound file changed: ' + name)


def protocol_identity(*, root=ROOT, deep=False):
    seal_path = root / 'protocol_seal43.json'
    seal = read_json(seal_path)
    if seal.get('schema') != 'step43-protocol-source-seal-v1':
        raise ValueError('Invalid protocol/source seal')
    if sha_file(root / 'protocol43.json') != seal['bound_files']['protocol43.json']:
        raise ValueError('Frozen protocol changed')
    if deep:
        verify_bindings(seal['bound_files'], root=root)
    return sha_file(seal_path)


def require_gate(relative, *, root=ROOT, **bindings):
    gate = read_json(root / relative)
    if gate.get('status') != 'PASS' or any(gate.get(k) != v for k, v in bindings.items()):
        raise ValueError('Matching independent PASS required: ' + relative)
    return gate


def event(kind, *, root=ROOT, log='output/events43.jsonl', **fields):
    path = root / log
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    previous = hashlib.sha256(lines[-1]).hexdigest() if lines else None
    record = {'index': len(lines), 'kind': kind, 'utc': utc_now(),
              'monotonic_ns': time.monotonic_ns(), 'previous_event_sha256': previous, **fields}
    with path.open('ab') as stream:
        stream.write(canonical_bytes(record))
        stream.flush()
    return record
