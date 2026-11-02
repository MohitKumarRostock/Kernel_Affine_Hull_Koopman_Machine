#!/usr/bin/env python3
"""Reproduce KAHM scale sensitivity on a fixed deterministic reference cloud.

The experiment compares identical references and queries under positive scalar
rescaling. It records the Gaussian Gram matrix, fitted regularization, folding
scores, and reconstructions expressed in the original input units. Equal Gram
matrices do not imply equal fitted reconstruction scores. This is a mechanism
diagnostic, not an experiment on forecasting accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY))

from autoencoder import autoencoder
from autoencoder_filtering_extended import autoencoder_filtering_extended
from kxa_matrix import kxa_matrix


def evaluate():
    rng = np.random.default_rng(2810)
    references = rng.uniform(-1, 1, size=(3, 24))
    queries = rng.uniform(-1, 1, size=(3, 20))
    values = []
    for scale in (0.5, 1.0, 2.0):
        model = autoencoder(scale * references, 2)
        gram = kxa_matrix(model["y_data_n_subspace"],
                          model["y_data_n_subspace"],
                          model["sqrt_weights_matrix"])
        reconstruction, folding = autoencoder_filtering_extended(
            scale * queries, model, "folding")
        regularization = float(np.trace(np.linalg.inv(model["B"]) - gram) / 24)
        values.append((scale, regularization, gram, folding, reconstruction / scale))

    reference = values[1]
    rows = []
    for scale, regularization, gram, folding, reconstruction in values:
        rows.append({
            "input_scale": scale,
            "inferred_retained_regularization": regularization,
            "mean_folding_score": float(folding.mean()),
            "mean_original_units_reconstruction_norm": float(
                np.linalg.norm(reconstruction - queries, axis=0).mean()),
            "max_gram_difference_vs_scale_one": float(np.max(np.abs(gram - reference[2]))),
            "max_folding_difference_vs_scale_one": float(np.max(np.abs(folding - reference[3]))),
            "max_original_units_reconstruction_difference_vs_scale_one": float(
                np.max(np.abs(reconstruction - reference[4]))),
        })

    sources = ["autoencoder.py", "autoencoder_filtering_extended.py",
               "kernel_regularized_least_squares.py", "dim_reduce.py",
               "kxx_matrix.py", "kxa_matrix.py", "distance_matrix.py"]
    return {
        "seed": 2810,
        "n_training_points": 24,
        "n_queries": 20,
        "dimension": 3,
        "subspace_dimension": 2,
        "rows": rows,
        "checks": {
            "gram_agreement_at_1e_minus_10": all(
                row["max_gram_difference_vs_scale_one"] < 1e-10 for row in rows),
            "nontrivial_folding_change": all(
                rows[i]["max_folding_difference_vs_scale_one"] > 1e-4 for i in (0, 2)),
            "nontrivial_reconstruction_change": all(
                rows[i]["max_original_units_reconstruction_difference_vs_scale_one"] > 1e-4
                for i in (0, 2)),
        },
        "interpretation": (
            "For this fixed cloud, uniform rescaling preserves the Gaussian Gram matrix "
            "to the stated tolerance while changing local regularization, reconstructions, "
            "and folding scores. This demonstrates a mechanism of scale sensitivity, "
            "not a general affine-invariance theorem or a forecasting accuracy benefit."
        ),
        "source_sha256": {name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest()
                          for name in sources},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON output; existing evidence is never overwritten.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    record = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "checks": record["checks"]}, indent=2))
    if not all(record["checks"].values()):
        raise SystemExit("The fixed scale-sensitivity diagnostic failed; inspect its evidence.")


if __name__ == "__main__":
    main()
