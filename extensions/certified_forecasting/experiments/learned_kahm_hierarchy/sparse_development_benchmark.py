#!/usr/bin/env python3
"""Reproduce the development-only sparse solver benchmark (no confirmation)."""
from pathlib import Path
import argparse
import json
import sys
import time
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/learned_kahm_decision'))
from certified_forecasting.sparse_block_witness import lipschitz_block_brackets
from run_study import read_gzip_json, restore_bank, draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = read_gzip_json(ROOT/'results/learned_kahm_decision_v1/frozen_models.json.gz')
    rows = []
    with threadpool_limits(limits=1):
        for regime_index, record in enumerate([records[0], records[20]]):
            x, y = draw(202610291801+regime_index, 1024, record['regime'])
            bank = restore_bank(record)
            features = bank.transform(x)
            for prefix, width in [(1, 2), (2, 6)]:
                for size in [2, 4, 8, 16, 256]:
                    started = time.perf_counter()
                    result = lipschitz_block_brackets(features[:, :width].reshape(-1, size, width),
                        y.reshape(-1, size), 2., bank.block_slices_[:prefix], tolerance=1e-8)
                    row = {'regime': record['regime'], 'prefix': prefix, 'block': size,
                        'seconds': time.perf_counter()-started, 'n_blocks': len(result['lower']),
                        'mean_lower': float(result['lower'].mean()), 'max_gap': float(result['gap'].max()),
                        'mean_gap': float(result['gap'].mean()), 'max_feasibility': float(result['feasibility_residual'].max()),
                        'max_iterations': int(result['iterations'].max()), 'n_unconverged': int((~result['converged']).sum())}
                    rows.append(row)
                    args.output.write_text(json.dumps({'scope': 'Development-only streams; not confirmation',
                        'seeds': [202610291801, 202610291802], 'rows': rows}, indent=2)+'\n')
                    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
