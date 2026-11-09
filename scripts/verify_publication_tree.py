#!/usr/bin/env python3
"""Verify scientific file identities, documentation links and LaTeX inputs.

Scientific files are checked against a specified source commit. Reader guides
are checked for valid links. These checks do not fit models or evaluate forecasts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=ROOT)


def tree(ref: str) -> dict[str, dict[str, str]]:
    records = {}
    for row in git('ls-tree', '-r', '-z', ref).split(b'\0'):
        if not row:
            continue
        header, raw_path = row.split(b'\t', 1)
        mode, kind, oid = header.decode().split()
        records[raw_path.decode()] = {'mode': mode, 'type': kind, 'git_object': oid}
    return records


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_scientific_files(reference: dict, current: dict,
                            prefixes: list[str], documentation: set[str]) -> int:
    """Check complete scientific-file sets and Git identities within each prefix."""
    for path in documentation:
        require(path.endswith('.md') and path in reference and path in current,
                f'Invalid reader-documentation path: {path}')
        require(any(path.startswith(prefix + '/') for prefix in prefixes),
                f'Documentation path is outside the declared directories: {path}')
    checked = 0
    for prefix in prefixes:
        expected = {p: v for p, v in reference.items()
                    if p.startswith(prefix + '/') and p not in documentation}
        observed = {p: v for p, v in current.items()
                    if p.startswith(prefix + '/') and p not in documentation}
        require(bool(expected), f'No scientific files at source path: {prefix}')
        require(set(observed) == set(expected),
                f'Scientific file set differs from the source specification: {prefix}')
        for path, entry in expected.items():
            require(observed[path] == entry, f'Scientific file identity differs: {path}')
        checked += len(expected)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(not out.exists(), 'Choose a new output directory; existing reports are not overwritten.')
    plan = json.loads((ROOT/'docs/PUBLICATION_SCOPE.json').read_text())
    reference_commit = plan['reference_commit']
    reference, current = tree(reference_commit), tree('HEAD')
    for prefix, expected in plan['scientific_subtrees'].items():
        require(git('rev-parse', f'{reference_commit}:{prefix}').decode().strip() == expected,
                f'Source subtree identity differs: {prefix}')
    scientific_count = verify_scientific_files(
        reference, current, list(plan['scientific_subtrees']),
        set(plan['reader_documentation']))
    for path, entry in current.items():
        require(entry['type'] == 'blob' and entry['mode'] in ('100644', '100755'),
                f'Unexpected submodule or symbolic link: {path}')
        data = (ROOT/path).read_bytes()
        oid = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        require(oid == entry['git_object'], f'Working file differs from its Git checksum: {path}')
    require({p.split('/')[0] for p in current} == set(plan['root_entries']),
            'Repository root entries differ from the declared file specification.')
    require({p.split('/')[1] for p in current if p.startswith('papers/')} == {'score_enrichment'},
            'Unexpected manuscript directory.')
    require({p.split('/')[1] for p in current if p.startswith('extensions/')} == {'score_enrichment_20261102'},
            'Unexpected study directory.')
    local_links = reference_links = tex_dependencies = 0
    reference_prefix = plan['reference_url'].rstrip('/')+'/'
    for name in current:
        path = ROOT/name
        if path.suffix == '.md':
            text = path.read_text(encoding='utf-8')
            for link in re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)', text):
                link = link.strip('<>')
                if link.startswith(reference_prefix):
                    target = unquote(link[len(reference_prefix):]).split('#')[0].rstrip('/')
                    require(target in reference or any(p.startswith(target+'/') for p in reference),
                            f'Unknown source path: {target}')
                    reference_links += 1
                elif not urlsplit(link).scheme and not link.startswith('#'):
                    target = unquote(urlsplit(link).path)
                    resolved = (path.parent/target).resolve()
                    require(resolved.is_relative_to(ROOT) and resolved.exists(),
                            f'Broken local link in {name}: {link}')
                    local_links += 1
        if path.suffix == '.tex':
            text = path.read_text(encoding='utf-8')
            for kind, target in re.findall(r'\\(input|include|bibliography)\{([^{}]+)\}', text):
                for item in target.split(','):
                    suffix = '.bib' if kind == 'bibliography' else '.tex'
                    dep = ROOT/'papers/score_enrichment'/item
                    if not dep.suffix:
                        dep = dep.with_suffix(suffix)
                    require(dep.is_file(), f'Missing document dependency in {name}: {item}')
                    tex_dependencies += 1
    report = {
        'status': 'passed', 'reference_commit': reference_commit,
        'checked_commit': git('rev-parse', 'HEAD').decode().strip(),
        'scope': 'Scientific file identities, tracked-file checksums, documentation links and LaTeX dependencies.',
        'tracked_files_checked': len(current),
        'scientific_reference_files_checked': scientific_count,
        'reader_documentation_files': len(plan['reader_documentation']),
        'local_links_checked': local_links, 'source_paths_checked': reference_links,
        'tex_dependencies_checked': tex_dependencies,
        'tracked_file_bytes': sum((ROOT/p).stat().st_size for p in current),
        'model_fitting_performed': False
    }
    out.mkdir(parents=True)
    (out/'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
