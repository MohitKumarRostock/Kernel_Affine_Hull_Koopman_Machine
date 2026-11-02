#!/usr/bin/env python3
"""Development-only solver benchmark; never used for confirmation endpoints."""
from pathlib import Path
import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/learned_kahm_decision'))
from certified_forecasting.block_witness import lipschitz_block_bracket
from run_study import read_gzip_json, restore_bank, draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = read_gzip_json(ROOT / 'results/learned_kahm_decision_v1/frozen_models.json.gz')
    rows = []
    with threadpool_limits(limits=1):
        for regime_index, record in enumerate([records[0], records[20]]):
            seed = 202610291701 + regime_index
            x, y = draw(seed, 256, record['regime'])
            bank = restore_bank(record)
            features = bank.transform(x)
            for prefix, width in [(1, 2), (2, 6)]:
                for size in [4, 8, 16, 32, 64, 128, 256]:
                    start = time.perf_counter()
                    bracket = lipschitz_block_bracket(features[:size, :width], y[:size], 2., bank.block_slices_[:prefix])
                    row = dict(regime=record['regime'], bank_repetition=record['repetition'],
                               seed=seed, prefix=prefix, size=size, seconds=time.perf_counter()-start,
                               lower=bracket['lower'], upper=bracket['upper'], gap=bracket['gap'],
                               fallback=bracket['fallback'], iterations=bracket['iterations'])
                    rows.append(row)
                    args.output.write_text(json.dumps({'scope':'Development-only solver benchmark; outcomes do not enter any confirmatory endpoint',
                        'updated_utc':datetime.now(timezone.utc).isoformat(), 'rows':rows}, indent=2)+'\n')
                    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
