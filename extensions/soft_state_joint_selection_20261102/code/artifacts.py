"""Checked, append-only artifacts and portable array/tree state serialization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)},
                        sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + value.tobytes()).hexdigest()


def safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def json_bytes(value):
    return (json.dumps(safe(value), indent=2, sort_keys=True,
                       allow_nan=False) + '\n').encode()


def write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if digest(temporary) != hashlib.sha256(payload).hexdigest():
        raise IOError('Atomic output digest mismatch')
    # Hard linking fails if another process has installed the destination.
    os.link(temporary, path)
    temporary.unlink()


def write_json(path, value):
    write_bytes(path, json_bytes(value))


def npz_bytes(values):
    for key, value in values.items():
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise TypeError('Object arrays are not portable: ' + key)
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    payload = stream.getvalue()
    with np.load(io.BytesIO(payload), allow_pickle=False) as check:
        if set(check.files) != set(values):
            raise IOError('Array roster changed during serialization')
        for key, array in values.items():
            if array_hash(array) != array_hash(check[key]):
                raise IOError('Array round trip changed: ' + key)
    return payload


def write_npz(path, values):
    write_bytes(path, npz_bytes(values))


def pack_tree(value):
    """Return a JSON tree and named numeric arrays; no arbitrary object code."""
    arrays = {}

    def visit(item, parts):
        if isinstance(item, np.ndarray):
            key = '__'.join(parts) or 'root'
            if key in arrays or item.dtype.hasobject:
                raise ValueError('Invalid or colliding portable array key: ' + key)
            arrays[key] = np.array(item, copy=True, order='C')
            return {'__ndarray__': key, 'dtype': str(item.dtype),
                    'shape': list(item.shape), 'sha256': array_hash(item)}
        if isinstance(item, np.generic):
            return visit(item.item(), parts)
        if isinstance(item, dict):
            if any(not isinstance(k, str) or '__' in k for k in item):
                raise ValueError('Portable state dictionary keys must be strings without double underscores')
            return {k: visit(v, [*parts, k]) for k, v in item.items()}
        if isinstance(item, tuple):
            return {'__tuple__': [visit(v, [*parts, str(i)]) for i, v in enumerate(item)]}
        if isinstance(item, list):
            return [visit(v, [*parts, str(i)]) for i, v in enumerate(item)]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError('Unsupported portable state value: ' + type(item).__name__)

    tree = visit(value, [])
    json_bytes(tree)  # Reject nonfinite scalar metadata before writing anything.
    return tree, arrays


def unpack_tree(tree, arrays):
    used = set()

    def visit(item):
        if isinstance(item, list):
            return [visit(v) for v in item]
        if not isinstance(item, dict):
            return item
        if '__ndarray__' in item:
            if set(item) != {'__ndarray__', 'dtype', 'shape', 'sha256'}:
                raise ValueError('Invalid portable array descriptor')
            key = item['__ndarray__']
            array = np.asarray(arrays[key])
            if (str(array.dtype) != item['dtype'] or list(array.shape) != item['shape']
                    or array_hash(array) != item['sha256']):
                raise ValueError('Portable array identity differs: ' + key)
            used.add(key)
            return array.copy()
        if '__tuple__' in item:
            if set(item) != {'__tuple__'}:
                raise ValueError('Invalid portable tuple')
            return tuple(visit(v) for v in item['__tuple__'])
        return {k: visit(v) for k, v in item.items()}

    result = visit(tree)
    if used != set(arrays):
        raise ValueError('Unused or missing portable state arrays')
    return result


def state_fingerprint(value):
    tree, _ = pack_tree(value)
    return hashlib.sha256(json_bytes(tree)).hexdigest()


def write_portable_state(directory, value):
    directory = Path(directory)
    tree, arrays = pack_tree(value)
    payload = npz_bytes(arrays)
    document = {'schema': 'kahm-soft-state-portable-v1',
                'array_file': 'state.npz',
                'array_sha256': hashlib.sha256(payload).hexdigest(),
                'tree': tree}
    write_bytes(directory / 'state.npz', payload)
    write_json(directory / 'state.json', document)
    restored = load_portable_state(directory)
    if state_fingerprint(restored) != state_fingerprint(value):
        raise IOError('Portable state round trip changed')
    return {'array_count': len(arrays), 'array_bytes': sum(v.nbytes for v in arrays.values()),
            'state_npz_bytes': len(payload), 'state_json_bytes': len(json_bytes(document)),
            'state_fingerprint': state_fingerprint(value)}


def load_portable_state(directory):
    directory = Path(directory)
    document = json.loads((directory / 'state.json').read_text())
    if document['schema'] != 'kahm-soft-state-portable-v1' or document['array_file'] != 'state.npz':
        raise ValueError('Unexpected portable state schema or array path')
    path = directory / document['array_file']
    if digest(path) != document['array_sha256']:
        raise ValueError('Portable array archive changed')
    with np.load(path, allow_pickle=False) as values:
        return unpack_tree(document['tree'], {k: values[k] for k in values.files})
