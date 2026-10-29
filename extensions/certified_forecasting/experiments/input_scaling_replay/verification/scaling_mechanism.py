"""Tiny deterministic mechanism check, not a forecasting performance experiment."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2] / "kahm-study"
sys.path.insert(0, str(REPO))
from autoencoder import autoencoder
from autoencoder_filtering_extended import autoencoder_filtering_extended
from kxa_matrix import kxa_matrix

rng = np.random.default_rng(2810)
y = rng.uniform(-1, 1, size=(3, 24))
queries = rng.uniform(-1, 1, size=(3, 20))
models, rows = [], []
for scale in [0.5, 1.0, 2.0]:
    ae = autoencoder(scale * y, 2)
    gram = kxa_matrix(ae["y_data_n_subspace"], ae["y_data_n_subspace"],
                      ae["sqrt_weights_matrix"])
    reconstruction, folding = autoencoder_filtering_extended(
        scale * queries, ae, "folding")
    lam = float(np.trace(np.linalg.inv(ae["B"]) - gram) / 24)
    models.append((gram, folding, reconstruction / scale))
    rows.append({
        "input_scale": scale,
        "inferred_retained_regularization": lam,
        "mean_folding_score": float(folding.mean()),
        "mean_original_units_reconstruction_norm": float(
            np.linalg.norm(reconstruction / scale - queries, axis=0).mean()),
    })
for row, (gram, folding, reconstruction) in zip(rows, models):
    row.update({
        "max_gram_difference_vs_scale_one": float(np.max(np.abs(gram - models[1][0]))),
        "max_folding_difference_vs_scale_one": float(np.max(np.abs(folding - models[1][1]))),
        "max_original_units_reconstruction_difference_vs_scale_one": float(
            np.max(np.abs(reconstruction - models[1][2]))),
    })
record = {
    "seed": 2810, "n_training_points": 24, "n_queries": 20,
    "dimension": 3, "subspace_dimension": 2, "rows": rows,
    "interpretation": "Demonstrates scale sensitivity of inherited KAHM reconstruction/folding even when Gaussian Grams are equal. It is not evidence of a forecasting accuracy improvement.",
    "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                       for name in ["autoencoder.py", "autoencoder_filtering_extended.py",
                                    "kernel_regularized_least_squares.py", "dim_reduce.py",
                                    "kxx_matrix.py", "kxa_matrix.py", "distance_matrix.py"]},
}
path = Path(__file__).with_suffix(".json")
path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
print(json.dumps(record, indent=2, allow_nan=False))
