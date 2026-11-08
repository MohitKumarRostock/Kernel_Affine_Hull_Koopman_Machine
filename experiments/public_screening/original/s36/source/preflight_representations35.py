"""Artificial-only API preflight; never opens a public or patient dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

from representations35 import (
    DEFAULT_CONFIG, deserialize_map, fit_representations, numeric_bytes,
    payload_bytes, serialize_map, temporal_indices, transform_one,
)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "tests/artificial_representation_preflight35.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(350035)
    artificial = rng.normal(size=(96, 24))
    artificial[:, -1] = 7.0
    queries = rng.normal(size=(9, 24))
    queries[0, 0] = 20.0
    queries[:, -1] = -9.0
    before = hashlib.sha256(artificial.tobytes()).hexdigest()
    started = time.perf_counter()
    maps, metadata = fit_representations(artificial, DEFAULT_CONFIG, seed=350035)
    after = hashlib.sha256(artificial.tobytes()).hexdigest()
    assert before == after, "Factory modified caller training inputs"
    assert len(maps) == 18
    records = []
    for name, representation in maps.items():
        predictions = transform_one(representation, queries)
        assert predictions.shape == (9, representation.n_features_out_)
        assert np.isfinite(predictions).all()
        payload = serialize_map(representation)
        restored = deserialize_map(payload)
        replay = transform_one(restored, queries)
        assert np.array_equal(predictions, replay), name + " payload replay changed outputs"
        assert len(payload) == payload_bytes(representation)
        assert numeric_bytes(representation) > 8
        assert payload_bytes(representation) > numeric_bytes(representation)
        scalar = float(representation.metric_scale)
        assert np.isfinite(scalar) and scalar > 0
        records.append({
            "name": name,
            "query_shape": list(predictions.shape),
            "finite": True,
            "pickle_replay_bitwise_equal": True,
            "numeric_bytes": numeric_bytes(representation),
            "payload_bytes": len(payload),
            "metric_scale": scalar,
        })
    raw_minmax = maps["raw_minmax01"]
    raw_unscaled = raw_minmax.core.transform(queries)
    assert raw_unscaled[0, 0] > 1.0, "Held-out minmax values were clipped"
    assert np.all(raw_unscaled[:, -1] == 0.0), "Training-constant axis was not frozen to zero"
    assert temporal_indices(96, 8).tolist() == [6, 18, 30, 42, 54, 66, 78, 90]
    assert len(np.unique(temporal_indices(96, 24))) == 24
    assert metadata["native_cost_accounting"]["regrouped_banks_fitted"] == 6
    assert metadata["native_cost_accounting"]["native_kmeans_block_fit_count"] == 4
    receipt = {
        "schema": "step35-artificial-representation-preflight-v1",
        "status": "PASS",
        "data_kind": "manufactured_gaussian_96_by_24_with_one_constant_axis",
        "artificial_seed": 350035,
        "public_data_values_read": 0,
        "patient_data_values_read": 0,
        "scientific_fits": 0,
        "artificial_representation_factory_calls": 1,
        "caller_array_unchanged": True,
        "heldout_minmax_clipping": False,
        "training_constant_axis_frozen_zero": True,
        "elapsed_seconds": time.perf_counter() - started,
        "native_cost_accounting": metadata["native_cost_accounting"],
        "maps": records,
        "factory_metadata": metadata,
    }
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": receipt["status"], "path": str(output),
        "artificial_maps": len(records),
        "elapsed_seconds": receipt["elapsed_seconds"],
        "native_cost_accounting": receipt["native_cost_accounting"],
    }))


if __name__ == "__main__":
    main()
