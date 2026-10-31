"""Wrap long table headings for the manuscript, preserving numerical entries.

This post-execution display utility does not change the frozen statistical
analysis. Original generated tables remain byte-identical to its outputs.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'extensions/direct_reconstruction/results/generated'
DESTINATION = ROOT / 'extensions/tmlr_integration/manuscript/generated/direct_reconstruction'
REPLACEMENTS = {
    'method_table.tex': {
        'Natural available-case scaled MSE': r'\shortstack{Natural subset\\scaled MSE}',
    },
    'cost_table.tex': {
        'Synthetic time (s)': r'\shortstack{Synthetic\\time (s)}',
        'Available natural time (s)': r'\shortstack{Natural subset\\time (s)}',
        'Synthetic state (KiB)': r'\shortstack{Synthetic\\state (KiB)}',
        'Available natural state (KiB)': r'\shortstack{Natural subset\\state (KiB)}',
    },
    'primary_table.tex': {},
    'natural_table.tex': {},
}


def main():
    records = []
    for name, replacements in REPLACEMENTS.items():
        original = (SOURCE / name).read_bytes()
        if (DESTINATION / name).read_bytes() != original:
            raise ValueError(f'Original manuscript table differs from audited output: {name}')
        text = original.decode('utf-8')
        for before, after in replacements.items():
            if text.count(before) != 1:
                raise ValueError(f'Unexpected heading occurrence: {before}')
            text = text.replace(before, after)
        path = DESTINATION / ('display_' + name)
        path.write_text(text, encoding='utf-8')
        records.append({'source': name, 'output': path.name,
                        'source_sha256': hashlib.sha256(original).hexdigest(),
                        'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'heading_replacements': replacements})
    manifest = {'scope': 'Heading line breaks only; numerical entries unchanged',
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'tables': records}
    (DESTINATION / 'DISPLAY_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
