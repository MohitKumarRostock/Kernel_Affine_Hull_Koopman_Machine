#!/usr/bin/env python3
"""Verify a paper-focused checkout against its archived predecessor.

This checks file preservation, local links and document inputs. It does not
refit models, authenticate the original experiment, or rewrite Git history.
"""
from __future__ import annotations
import argparse
from collections import Counter
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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(not out.exists(), 'Choose a new output directory; audit records are not overwritten.')
    plan = json.loads((ROOT/'docs/PUBLICATION_SCOPE.json').read_text())
    baseline = plan['archive_commit']
    before, after = tree(baseline), tree('HEAD')
    protected_count = 0
    for prefix, expected in plan['unchanged_subtrees'].items():
        require(git('rev-parse', f'{baseline}:{prefix}').decode().strip() == expected,
                f'Archive subtree differs: {prefix}')
        require(git('rev-parse', f'HEAD:{prefix}').decode().strip() == expected,
                f'Protected subtree changed: {prefix}')
        protected_count += sum(p.startswith(prefix+'/') for p in after)
    for path, entry in after.items():
        require(entry['type'] == 'blob' and entry['mode'] in ('100644', '100755'),
                f'Unexpected submodule or symbolic link: {path}')
        data = (ROOT/path).read_bytes()
        oid = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        require(oid == entry['git_object'], f'Working file differs from commit: {path}')
    require({p.split('/')[0] for p in after} == set(plan['root_entries']),
            'The active root differs from its declared publication scope.')
    require({p.split('/')[1] for p in after if p.startswith('papers/')} == {'score_enrichment'},
            'Unexpected manuscript directory.')
    require({p.split('/')[1] for p in after if p.startswith('extensions/')} == {'score_enrichment_20261102'},
            'Unexpected study directory.')
    local_links = archive_links = tex_dependencies = 0
    archive_prefix = plan['archive_url'].rstrip('/')+'/'
    for name in after:
        path = ROOT/name
        if path.suffix == '.md':
            text = path.read_text(encoding='utf-8')
            for link in re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)', text):
                link = link.strip('<>')
                if link.startswith(archive_prefix):
                    target = unquote(link[len(archive_prefix):]).split('#')[0].rstrip('/')
                    require(target in before or any(p.startswith(target+'/') for p in before),
                            f'Unknown archived path: {target}')
                    archive_links += 1
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
    removed = sorted(set(before)-set(after))
    report = {
        'status': 'passed', 'archive_commit': baseline,
        'checked_commit': git('rev-parse', 'HEAD').decode().strip(),
        'scope': 'Working-tree preservation, local links, pinned historical paths and TeX dependencies; no numerical model replay.',
        'files_before': len(before), 'files_after': len(after),
        'files_removed': len(removed),
        'files_added': len(set(after)-set(before)),
        'files_modified': sum(before[p] != after[p] for p in set(before)&set(after)),
        'protected_files_unchanged': protected_count,
        'local_links_checked': local_links, 'archived_paths_checked': archive_links,
        'tex_dependencies_checked': tex_dependencies,
        'current_tracked_bytes': sum((ROOT/p).stat().st_size for p in after),
        'removed_by_root': dict(sorted(Counter(p.split('/')[0] for p in removed).items())),
        'history_rewritten': False,
        'size_note': 'Current-file bytes exclude Git history; this is not a claim about full clone size.'
    }
    out.mkdir(parents=True)
    (out/'cleanup_audit.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    with (out/'removed_files.tsv').open('w', encoding='utf-8') as handle:
        handle.write('path\tmode\tgit_object\tarchive_commit\n')
        for p in removed:
            handle.write(f"{p}\t{before[p]['mode']}\t{before[p]['git_object']}\t{baseline}\n")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
