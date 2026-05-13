#!/usr/bin/env python3
"""Soft KAHM regression demo on a Gaussian-process-style curve fitting problem.

This example uses the canonical 1D function often used in Gaussian Process
regression tutorials:

    f(x) = x * sin(x)

Changes from the earlier demo:
    1. The KAHM hard-prediction baseline is removed.
    2. Both input x and output y are scaled to [-1, 1] before KAHM training,
       soft-parameter tuning, prediction, and plotting.
    3. No separate held-out split is used. Soft tuning runs on the training
       samples, with top-k fixed internally to the effective cluster count.

Expected files in the same directory:
    - kahm_regression_clean_updated.py
    - parallel_autoencoders.py
    - combine_multiple_autoencoders_extended.py

Install Python dependencies:
    pip install numpy scikit-learn joblib matplotlib tqdm

Run:
    python demo_kahm_gp_style_curve_fit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast, overload

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from kahm_regression_clean_updated import (
    kahm_regress,
    train_kahm_regressor,
    tune_cluster_centers_nlms,
    tune_soft_params,
)

ArrayF = NDArray[np.float32]


def gp_demo_function(x: np.ndarray) -> np.ndarray:
    """Canonical Gaussian-process regression demo target: f(x)=x*sin(x)."""
    return x * np.sin(x)


@overload
def as_kahm_columns(x: np.ndarray, y: None = None) -> ArrayF:
    ...


@overload
def as_kahm_columns(x: np.ndarray, y: np.ndarray) -> tuple[ArrayF, ArrayF]:
    ...


def as_kahm_columns(x: np.ndarray, y: np.ndarray | None = None) -> ArrayF | tuple[ArrayF, ArrayF]:
    """Convert 1D vectors to the KAHM script's column-major convention.

    KAHM regressor expects:
        X: (D_in, N)
        Y: (D_out, N)

    For scalar curve fitting, both dimensions are 1.
    """
    X = np.asarray(x, dtype=np.float32).reshape(1, -1)

    if y is None:
        return X

    Y = np.asarray(y, dtype=np.float32).reshape(1, -1)
    return X, Y


def to_column_vector(values: np.ndarray) -> NDArray[np.float64]:
    """Return values as a 2D column vector for sklearn scalers/models."""
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def from_column_vector(values: np.ndarray) -> ArrayF:
    """Return a 2D sklearn output column as a flat float32 vector."""
    return np.asarray(values, dtype=np.float32).reshape(-1)


def make_dataset(
    *,
    n_train: int,
    n_grid: int,
    noise_std: float,
    random_state: int,
) -> dict[str, ArrayF]:
    """Create noisy training samples plus a dense noiseless grid."""
    rng = np.random.default_rng(random_state)

    # GP tutorials often use [0, 10] for f(x)=x*sin(x).
    x_grid = np.linspace(0.0, 10.0, n_grid, dtype=np.float32)
    y_grid_true = gp_demo_function(x_grid).astype(np.float32)

    # KAHM trains one classifier per output cluster, so it benefits from more
    # samples than the ultra-sparse toy GP examples.
    x_train = np.sort(rng.uniform(0.0, 10.0, size=n_train).astype(np.float32))
    y_train_true = gp_demo_function(x_train).astype(np.float32)
    y_train_noisy = y_train_true + rng.normal(0.0, noise_std, size=x_train.shape).astype(np.float32)

    return {
        "x_train_raw": x_train,
        "y_train_raw": y_train_noisy,
        "x_grid_raw": x_grid,
        "y_grid_true_raw": y_grid_true,
    }


def add_scaled_data(data: dict[str, ArrayF]) -> tuple[dict[str, ArrayF], MinMaxScaler, MinMaxScaler]:
    """Fit training-only scalers and append [-1, 1] scaled arrays.

    The input scaler is fitted to the observed training x values.
    The output scaler is fitted to the observed noisy training y values.
    The noiseless grid target is transformed with the same output scaler so that
    all KAHM metrics and plots are in the scaled output coordinate system.
    """
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    x_scaler.fit(to_column_vector(data["x_train_raw"]))
    y_scaler.fit(to_column_vector(data["y_train_raw"]))

    scaled = dict(data)
    scaled["x_train"] = from_column_vector(x_scaler.transform(to_column_vector(data["x_train_raw"])))
    scaled["x_grid"] = from_column_vector(x_scaler.transform(to_column_vector(data["x_grid_raw"])))

    scaled["y_train"] = from_column_vector(y_scaler.transform(to_column_vector(data["y_train_raw"])))
    scaled["y_grid_true"] = from_column_vector(y_scaler.transform(to_column_vector(data["y_grid_true_raw"])))

    return scaled, x_scaler, y_scaler


def train_reference_gpr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    *,
    random_state: int,
) -> tuple[ArrayF, ArrayF]:
    """Train a conventional GaussianProcessRegressor for visual reference.

    The reference model is trained in the same scaled coordinate system used by
    KAHM, namely x in [-1, 1] and y in [-1, 1].
    """
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e3),
    ) + WhiteKernel(
        noise_level=0.1,
        noise_level_bounds=(1e-6, 1e1),
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=random_state,
        n_restarts_optimizer=5,
    )

    gpr.fit(to_column_vector(x_train), np.asarray(y_train, dtype=np.float64))

    prediction = gpr.predict(to_column_vector(x_grid), return_std=True)
    mean_raw, std_raw = cast(tuple[np.ndarray, np.ndarray], prediction)

    return np.asarray(mean_raw, dtype=np.float32), np.asarray(std_raw, dtype=np.float32)


def train_kahm_curve_model(args: argparse.Namespace, data: dict[str, ArrayF]) -> dict[str, Any]:
    """Train, tune, and optionally NLMS-refine the soft KAHM curve fitter."""
    X_train, Y_train = as_kahm_columns(data["x_train"], data["y_train"])

    # For scalar output, target clustering is effectively clustering scaled y-values.
    model = train_kahm_regressor(
        X_train,
        Y_train,
        n_clusters=args.n_clusters,
        subspace_dim=20,
        Nb=args.nb,
        random_state=args.random_state,
        verbose=args.verbose,
        kmeans_kind="full",
        model_dtype="float32",
        save_ae_to_disk=args.save_ae_to_disk,
        ae_dir=args.ae_dir,
        overwrite_ae_dir=args.overwrite_ae_dir,
        singleton_strategy="augment",
        singleton_aux_mix=0.05,
        cluster_strategy="y_then_x",
        max_x_splits_per_y_cluster=10
    )

    soft_result = tune_soft_params(
        model,
        X_train,
        Y_train,
        alphas=args.alphas,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    print(
        "Best KAHM soft parameters on scaled training data after one-epoch NLMS: "
        f"alpha={soft_result.best_alpha:g}, "
        f"topk={soft_result.best_topk}, "
        f"post-NLMS train MSE={soft_result.best_mse:.6g}"
    )

    if args.nlms_epochs > 0:        
        nlms_result = tune_cluster_centers_nlms(
            model,
            X_train,
            Y_train,
            mu=args.nlms_mu,
            epsilon=1.0,
            epochs=args.nlms_epochs,
            batch_size=args.batch_size,
            shuffle=True,
            random_state=args.random_state,
            alpha=soft_result.best_alpha,
            topk=soft_result.best_topk,
            anchor_lambda=args.anchor_lambda,
            n_jobs=args.n_jobs,
            preload_classifier=args.preload_classifier,
            verbose=args.verbose,
        )
        print(
            "Additional KAHM NLMS prototype refinement on scaled training data: "
            f"epochs={nlms_result.epochs}, "
            f"final train MSE={nlms_result.final_mse:.6g}"
        )

    return model


def plot_results(
    *,
    data: dict[str, ArrayF],
    kahm_soft: np.ndarray,
    gpr_mean: np.ndarray | None,
    gpr_std: np.ndarray | None,
    output_path: Path,
) -> None:
    """Plot the true curve, noisy samples, and fitted soft regressors."""
    x_grid = data["x_grid"]
    y_true = data["y_grid_true"]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(x_grid, y_true, linewidth=2, label="True scaled f(x) = x sin(x)")
    ax.scatter(
        data["x_train"],
        data["y_train"],
        s=24,
        alpha=0.75,
        label="Scaled noisy KAHM train samples",
    )
    ax.plot(
        x_grid,
        kahm_soft.ravel(),
        linewidth=2,
        label="KAHM tuned soft prediction",
    )

    if gpr_mean is not None and gpr_std is not None:
        ax.plot(
            x_grid,
            gpr_mean,
            linewidth=1.8,
            label="Reference sklearn GPR mean",
        )
        ax.fill_between(
            x_grid,
            gpr_mean - 1.96 * gpr_std,
            gpr_mean + 1.96 * gpr_std,
            alpha=0.15,
            label="Reference GPR 95% interval",
        )

    ax.set_title("Soft KAHM regression on scaled GP-style curve fitting data")
    ax.set_xlabel("scaled x")
    ax.set_ylabel("scaled y")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    print(f"Saved plot to: {output_path}")
    plt.show(block=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate soft KAHM regression on the canonical GP curve-fitting "
            "function f(x)=x*sin(x), with x and y scaled to [-1, 1]."
        )
    )

    parser.add_argument("--n-train", type=int, default=500, help="Number of noisy training samples.")
    parser.add_argument("--n-grid", type=int, default=2000, help="Number of dense grid points for evaluation/plotting.")
    parser.add_argument("--noise-std", type=float, default=0.35, help="Standard deviation of Gaussian observation noise.")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of output clusters/prototypes for KAHM.")
    parser.add_argument("--nb", type=int, default=100, help="Number of KAHM autoencoder bases per cluster.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for prediction/NLMS operations.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs passed to KAHM distance evaluation.")
    parser.add_argument("--random-state", type=int, default=0, help="Reproducibility seed.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0], help="Soft KAHM alpha values to sweep for tuning.")
    parser.add_argument("--nlms-epochs", type=int, default=1000, help="Optional extra NLMS epochs after tune_soft_params; set >0 to run additional refinement.")
    parser.add_argument("--nlms-mu", type=float, default=0.1, help="NLMS step size for cluster-center refinement.")
    parser.add_argument("--anchor-lambda", type=float, default=0.0, help="Small anchor regularizer for NLMS centers.")
    parser.add_argument("--preload-classifier", action="store_true", help="Preload disk-backed AE shards before NLMS.")
    parser.add_argument("--save-ae-to-disk", action="store_true", help="Store AE shards on disk instead of in memory.")
    parser.add_argument("--ae-dir", type=str, default=None, help="Directory for AE shards when --save-ae-to-disk is used.")
    parser.add_argument("--overwrite-ae-dir", action="store_true", help="Allow overwriting an existing AE shard directory.")
    parser.add_argument("--plot-path", type=Path, default=Path("kahm_gp_style_curve_fit_scaled_soft_only.png"))
    parser.add_argument("--no-gpr-reference", action="store_true", help="Skip the optional sklearn GPR reference curve.")
    parser.add_argument("--quiet", action="store_true", help="Reduce KAHM progress output.")

    args = parser.parse_args()
    args.verbose = not args.quiet

    return args


def main() -> None:
    args = parse_args()

    raw_data = make_dataset(
        n_train=args.n_train,
        n_grid=args.n_grid,
        noise_std=args.noise_std,
        random_state=args.random_state,
    )
    data, _, _ = add_scaled_data(raw_data)

    model = train_kahm_curve_model(args, data)

    X_grid = as_kahm_columns(data["x_grid"])
    Y_grid_true = data["y_grid_true"].reshape(1, -1).astype(np.float32)

    kahm_soft = kahm_regress(
        model,
        X_grid,
        mode="soft",
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        show_progress=args.verbose,
    )

    print(f"alpha={model['soft_alpha']}, topk={model['soft_topk']}")

    print(
        "KAHM tuned soft grid MSE versus scaled noiseless true curve: "
        f"{mean_squared_error(Y_grid_true.ravel(), kahm_soft.ravel()):.6g}"
    )

    gpr_mean = None
    gpr_std = None

    if not args.no_gpr_reference:
        gpr_mean, gpr_std = train_reference_gpr(
            data["x_train"],
            data["y_train"],
            data["x_grid"],
            random_state=args.random_state,
        )
        print(f"Reference sklearn GPR scaled grid MSE: {mean_squared_error(data['y_grid_true'], gpr_mean):.6g}")

    plot_results(
        data=data,
        kahm_soft=kahm_soft,
        gpr_mean=gpr_mean,
        gpr_std=gpr_std,
        output_path=args.plot_path,
    )


if __name__ == "__main__":
    main()