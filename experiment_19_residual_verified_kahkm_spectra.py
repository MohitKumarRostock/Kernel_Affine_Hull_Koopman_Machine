#!/usr/bin/env python3
"""Experiment 19: rank-aware, residual-verified KAHKM spectra.

Purpose
-------
This script implements the manuscript's R3-style RKHS spectral verification
diagnostic for the finite-rank KAHM Koopman kernel

    k_KAHM(x, x') = Psi(x)^T Psi(x').

For snapshot pairs (x_i, y_i), write

    Phi = Psi(X)      shape (C, N)
    Chi = Psi(Y)      shape (C, N).

Candidate spectral modes are constructed in the empirically supported
coefficient subspace.  From the training association matrix Phi_train, compute

    Phi_train = U Sigma V^T

and retain the left singular vectors whose singular values satisfy

    sigma_j > rank_rtol * sigma_max.

The resulting basis Q_Phi has shape (C, r_Phi).  For each fitted feature
operator M (NLMS: M = B^T; ridge-LS: M fitted from Chi_train ~= M Phi_train),
form the reduced operator

    M_red = Q_Phi^T M Q_Phi,

solve

    M_red w_tilde = lambda w_tilde,

and lift the candidate coefficient vector as

    w = Q_Phi w_tilde.

R3 verification
---------------
On a held-out spectral subsample, represent the lifted candidate by kernel
sections using ridge-regularized coefficients

    v_hat = argmin_v ||Phi v - w||_2^2
                    + representation_ridge ||v||_2^2.

The representation defect is

    eta_rep = ||Phi v_hat - w||_2^2 / ||w||_2^2.

The successor-normalized squared RKHS residual is

    rho(lambda, v_hat)
        = ||Chi v_hat - lambda Phi v_hat||_2^2
          / ||Chi v_hat||_2^2.

With

    G = Phi^T Phi,
    A = Chi^T Phi,
    R = Chi^T Chi,

the same residual is verified independently by

    rho = v^* (R - lambda A - conj(lambda) A^* + |lambda|^2 G) v
          / (v^* R v).

The main Experiment 19 CSVs report these rank-aware R3 quantities.  For
provenance, the script also writes separate legacy CSVs reproducing the
previous ambient-C eigenvector / source-normalized unsquared residual
diagnostic.

Recommended full run
--------------------
python experiment_19_residual_verified_kahkm_spectra.py \
  --systems duffing vanderpol \
  --train-seeds 0 1 2 --test-seeds 100 101 102 \
  --n-steps-train 1200 --n-steps-test 800 \
  --spectral-subsample 800

Fast smoke test
---------------
python experiment_19_residual_verified_kahkm_spectra.py \
  --systems duffing --train-seeds 0 --test-seeds 100 \
  --n-steps-train 300 --n-steps-test 200 \
  --spectral-subsample 120 --duffing-c 5 --nb 30 --nlms-epochs 3 \
  --output-dir kahkm_exp19_r3_smoke --zip-name kahkm_exp19_r3_smoke_results

Outputs
-------
The output directory contains:

- experiment_19_system_summary.csv
- experiment_19_mode_residuals.csv
- experiment_19_legacy_ambient_system_summary.csv
- experiment_19_legacy_ambient_mode_residuals.csv
- experiment_19_config.json
- experiment_19_<system>_feature_rank.npz
- experiment_19_<system>_subsample_indices.npz
- spectrum scatter plots, if matplotlib is installed and --no-plots is not set
- a zipped copy of the output directory

The default Van der Pol representation is the centered noise-aware selection
C=25, omega=4.  Duffing remains C=10, omega=2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd().resolve()
for _path in (SCRIPT_DIR, CWD):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    KAHKMFitResult,
    fit_kahkm,
    kahm_associations,
    make_duffing_snapshots,
    simplex_violation,
)

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]
SystemName: TypeAlias = Literal["duffing", "vanderpol", "acrobot"]
OperatorName: TypeAlias = Literal["nlms", "ridge_ls"]


@dataclass(frozen=True)
class SystemConfig:
    name: SystemName
    dt: float
    n_clusters: int
    omega: float
    tau: float
    subspace_dim: int


@dataclass(frozen=True)
class ExperimentArgs:
    systems: tuple[SystemName, ...]
    operators: tuple[OperatorName, ...]
    train_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    n_steps_train: int
    n_steps_test: int
    output_dir: str
    zip_name: str
    duffing_c: int
    duffing_omega: float
    vanderpol_c: int
    vanderpol_omega: float
    acrobot_c: int
    acrobot_omega: float
    tau: float
    subspace_dim: int
    acrobot_subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    kmeans_batch_size: int
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int | None
    ridge: float
    representation_ridge: float
    rank_rtol: float
    spectral_subsample: int
    residual_thresholds: tuple[float, ...]
    top_modes: int
    random_state: int
    vanderpol_mu: float
    acrobot_train_episodes: int
    acrobot_test_episodes: int
    acrobot_max_steps: int
    acrobot_exploration_eps: float
    save_ae_to_disk: bool
    no_plots: bool
    skip_existing: bool


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def _parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one float value is required.")
    return parsed


def _parse_system_tuple(values: Sequence[str]) -> tuple[SystemName, ...]:
    valid = {"duffing", "vanderpol", "acrobot"}
    parsed: list[SystemName] = []
    for value in values:
        if value not in valid:
            raise argparse.ArgumentTypeError("systems must contain only duffing, vanderpol, and/or acrobot.")
        parsed.append(cast(SystemName, value))
    if not parsed:
        raise argparse.ArgumentTypeError("At least one system is required.")
    return tuple(parsed)


def _parse_operator_tuple(values: Sequence[str]) -> tuple[OperatorName, ...]:
    valid = {"nlms", "ridge_ls"}
    parsed: list[OperatorName] = []
    for value in values:
        if value not in valid:
            raise argparse.ArgumentTypeError("operators must contain only nlms and/or ridge_ls.")
        parsed.append(cast(OperatorName, value))
    if not parsed:
        raise argparse.ArgumentTypeError("At least one operator is required.")
    return tuple(parsed)


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Residual-verified KAHKM spectral diagnostics.")
    parser.add_argument("--systems", nargs="+", default=["duffing", "vanderpol"])
    parser.add_argument("--operators", nargs="+", default=["nlms", "ridge_ls"])
    parser.add_argument("--train-seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--test-seeds", nargs="+", default=["100", "101", "102"])
    parser.add_argument("--n-steps-train", type=int, default=1200)
    parser.add_argument("--n-steps-test", type=int, default=800)

    parser.add_argument("--output-dir", default="kahkm_experiment_19_r3_residual_verified_spectra")
    parser.add_argument("--zip-name", default="kahkm_experiment_19_r3_residual_verified_spectra_results")

    # Tuned manuscript settings.
    parser.add_argument("--duffing-c", type=int, default=10)
    parser.add_argument("--duffing-omega", type=float, default=2.0)
    parser.add_argument("--vanderpol-c", type=int, default=25)
    parser.add_argument("--vanderpol-omega", type=float, default=4.0)
    parser.add_argument("--acrobot-c", type=int, default=150)
    parser.add_argument("--acrobot-omega", type=float, default=0.5)

    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--acrobot-subspace-dim", type=int, default=20)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-8,
        help="Ridge parameter for the comparison ridge-LS feature operator.",
    )
    parser.add_argument(
        "--representation-ridge",
        type=float,
        default=1e-8,
        help="Ridge parameter used to represent lifted spectral candidates by held-out kernel sections.",
    )
    parser.add_argument(
        "--rank-rtol",
        type=float,
        default=1e-10,
        help="Relative SVD threshold for the training-supported feature basis: sigma_j > rank_rtol * sigma_max.",
    )

    parser.add_argument(
        "--spectral-subsample",
        type=int,
        default=800,
        help="Number of test snapshot pairs used for RKHS residual verification. <=0 means use all.",
    )
    parser.add_argument(
        "--residual-thresholds",
        nargs="+",
        default=["1e-3", "1e-2", "5e-2", "1e-1"],
        help="Residual thresholds used for mode-count summaries.",
    )
    parser.add_argument("--top-modes", type=int, default=10, help="Number of lowest-residual modes to print.")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--vanderpol-mu", type=float, default=1.0)

    # Optional Acrobot data collection. Acrobot is not run unless --systems includes acrobot.
    parser.add_argument("--acrobot-train-episodes", type=int, default=12)
    parser.add_argument("--acrobot-test-episodes", type=int, default=6)
    parser.add_argument("--acrobot-max-steps", type=int, default=500)
    parser.add_argument("--acrobot-exploration-eps", type=float, default=0.10)

    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    ns = parser.parse_args()
    max_train_per_cluster = int(ns.max_train_per_cluster)
    if float(ns.representation_ridge) < 0.0:
        parser.error("--representation-ridge must be nonnegative.")
    if not (0.0 < float(ns.rank_rtol) < 1.0):
        parser.error("--rank-rtol must lie strictly between 0 and 1.")
    return ExperimentArgs(
        systems=_parse_system_tuple([str(v) for v in ns.systems]),
        operators=_parse_operator_tuple([str(v) for v in ns.operators]),
        train_seeds=_parse_int_tuple([str(v) for v in ns.train_seeds]),
        test_seeds=_parse_int_tuple([str(v) for v in ns.test_seeds]),
        n_steps_train=int(ns.n_steps_train),
        n_steps_test=int(ns.n_steps_test),
        output_dir=str(ns.output_dir),
        zip_name=str(ns.zip_name),
        duffing_c=int(ns.duffing_c),
        duffing_omega=float(ns.duffing_omega),
        vanderpol_c=int(ns.vanderpol_c),
        vanderpol_omega=float(ns.vanderpol_omega),
        acrobot_c=int(ns.acrobot_c),
        acrobot_omega=float(ns.acrobot_omega),
        tau=float(ns.tau),
        subspace_dim=int(ns.subspace_dim),
        acrobot_subspace_dim=int(ns.acrobot_subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=None if max_train_per_cluster <= 0 else max_train_per_cluster,
        ridge=float(ns.ridge),
        representation_ridge=float(ns.representation_ridge),
        rank_rtol=float(ns.rank_rtol),
        spectral_subsample=int(ns.spectral_subsample),
        residual_thresholds=_parse_float_tuple([str(v) for v in ns.residual_thresholds]),
        top_modes=int(ns.top_modes),
        random_state=int(ns.random_state),
        vanderpol_mu=float(ns.vanderpol_mu),
        acrobot_train_episodes=int(ns.acrobot_train_episodes),
        acrobot_test_episodes=int(ns.acrobot_test_episodes),
        acrobot_max_steps=int(ns.acrobot_max_steps),
        acrobot_exploration_eps=float(ns.acrobot_exploration_eps),
        save_ae_to_disk=bool(ns.save_ae_to_disk),
        no_plots=bool(ns.no_plots),
        skip_existing=bool(ns.skip_existing),
    )


def make_vanderpol_snapshots(
    n_steps: int = 1600,
    dt: float = 0.02,
    seed: int = 0,
    mu: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Generate Van der Pol snapshot pairs using the same convention as Experiment 15."""
    rng = np.random.default_rng(int(seed))

    def f(z: FloatArray) -> FloatArray:
        q = float(z[0])
        p = float(z[1])
        return np.array([p, float(mu) * (1.0 - q * q) * p - q], dtype=np.float64)

    def rk4_step(z: FloatArray) -> FloatArray:
        k1 = f(z)
        k2 = f(z + 0.5 * float(dt) * k1)
        k3 = f(z + 0.5 * float(dt) * k2)
        k4 = f(z + float(dt) * k3)
        return np.asarray(z + (float(dt) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)

    z = np.array([2.0, 0.0], dtype=np.float64) + 0.05 * rng.normal(size=2)
    states = np.empty((2, int(n_steps) + 1), dtype=np.float64)
    states[:, 0] = z
    for t in range(int(n_steps)):
        z = rk4_step(z) + 1e-5 * rng.normal(size=2)
        states[:, t + 1] = z
    return states[:, :-1], states[:, 1:]


def make_snapshots(
    system: SystemName,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    vanderpol_mu: float,
) -> tuple[FloatArray, FloatArray]:
    if system == "duffing":
        X0, X1 = make_duffing_snapshots(n_steps=int(n_steps), dt=float(dt), seed=int(seed))
        return np.asarray(X0, dtype=np.float64), np.asarray(X1, dtype=np.float64)
    if system == "vanderpol":
        return make_vanderpol_snapshots(n_steps=int(n_steps), dt=float(dt), seed=int(seed), mu=float(vanderpol_mu))
    raise ValueError(f"make_snapshots does not support {system}; Acrobot is collected separately.")


def make_concatenated_training_data(
    system: SystemName,
    *,
    seeds: tuple[int, ...],
    n_steps: int,
    dt: float,
    vanderpol_mu: float,
) -> tuple[FloatArray, FloatArray]:
    x0_parts: list[FloatArray] = []
    x1_parts: list[FloatArray] = []
    for seed in seeds:
        X0, X1 = make_snapshots(
            system,
            n_steps=int(n_steps),
            dt=float(dt),
            seed=int(seed),
            vanderpol_mu=float(vanderpol_mu),
        )
        x0_parts.append(X0)
        x1_parts.append(X1)
    return (
        np.concatenate(x0_parts, axis=1).astype(np.float64, copy=False),
        np.concatenate(x1_parts, axis=1).astype(np.float64, copy=False),
    )


def collect_acrobot_train_test(args: ExperimentArgs) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    try:
        from experiment_18_acrobot_sequential_decision_pylance_clean import collect_acrobot_data  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Acrobot support requires experiment_18_acrobot_sequential_decision_pylance_clean.py "
            "and gymnasium[classic-control]. Either install the dependency or omit acrobot from --systems."
        ) from exc

    # For Acrobot we mirror the manuscript's fixed-policy closed-loop data generation.
    train = collect_acrobot_data(
        episodes=int(args.acrobot_train_episodes),
        max_steps=int(args.acrobot_max_steps),
        seed=int(args.train_seeds[0]),
        exploration_eps=float(args.acrobot_exploration_eps),
    )
    test = collect_acrobot_data(
        episodes=int(args.acrobot_test_episodes),
        max_steps=int(args.acrobot_max_steps),
        seed=int(args.test_seeds[0]),
        exploration_eps=float(args.acrobot_exploration_eps),
    )
    return (
        np.asarray(train.X0, dtype=np.float64),
        np.asarray(train.X1, dtype=np.float64),
        np.asarray(test.X0, dtype=np.float64),
        np.asarray(test.X1, dtype=np.float64),
    )


def _relative_error(pred: FloatArray, target: FloatArray, eps: float = 1e-12) -> float:
    numerator = float(np.sum((target - pred) ** 2))
    denominator = float(np.sum(target * target))
    return numerator / max(denominator, eps)


def _r2(pred: FloatArray, target: FloatArray) -> float:
    residual_ss = float(np.sum((pred - target) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def _ridge_feature_operator(Phi: FloatArray, Chi: FloatArray, ridge: float) -> FloatArray:
    """Fit M such that Chi ~= M Phi using C-dimensional ridge least squares."""
    c = int(Phi.shape[0])
    gram = Phi @ Phi.T
    return np.asarray((Chi @ Phi.T) @ np.linalg.inv(gram + float(ridge) * np.eye(c)), dtype=np.float64)


def _select_subsample(
    Phi: FloatArray,
    Chi: FloatArray,
    *,
    spectral_subsample: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    n = int(Phi.shape[1])
    if int(spectral_subsample) <= 0 or int(spectral_subsample) >= n:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = np.sort(rng.choice(n, size=int(spectral_subsample), replace=False)).astype(np.int64)
    return Phi[:, idx].astype(np.float64, copy=False), Chi[:, idx].astype(np.float64, copy=False), idx


def _preimage_coefficients(Phi: FloatArray, w: ComplexArray, ridge: float) -> ComplexArray:
    """Return v with Phi v ~= w using ridge-regularized minimum-norm preimage."""
    c = int(Phi.shape[0])
    lhs = Phi @ Phi.T + float(ridge) * np.eye(c, dtype=np.float64)
    alpha = np.linalg.solve(lhs.astype(np.complex128), w.reshape(c).astype(np.complex128))
    return (Phi.T.astype(np.complex128) @ alpha).astype(np.complex128, copy=False)


def _training_feature_basis(
    Phi_train: FloatArray,
    *,
    rank_rtol: float,
) -> tuple[FloatArray, FloatArray, float]:
    """Return Q_Phi, singular values, and the explicit numerical-rank threshold.

    The basis is estimated from training associations only.  Phi_train is left
    uncentered so the constant/simplex direction remains available to the
    spectral diagnostic.
    """
    Phi = np.asarray(Phi_train, dtype=np.float64)
    if Phi.ndim != 2:
        raise ValueError(f"Phi_train must be two-dimensional, got shape {Phi.shape}.")
    if Phi.shape[0] == 0 or Phi.shape[1] == 0:
        raise ValueError("Phi_train must be nonempty.")

    U, singular_values, _Vh = np.linalg.svd(Phi, full_matrices=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.size == 0 or not math.isfinite(float(singular_values[0])):
        raise ValueError("Could not determine a finite singular spectrum for Phi_train.")

    sigma_max = float(singular_values[0])
    if sigma_max <= 0.0:
        raise ValueError("Phi_train has zero numerical norm; no spectral coefficient space exists.")

    threshold = float(rank_rtol) * sigma_max
    rank = int(np.sum(singular_values > threshold))
    if rank <= 0:
        raise ValueError(
            f"Numerical feature rank is zero at rank_rtol={rank_rtol:g}; "
            f"sigma_max={sigma_max:.6g}."
        )

    Q_phi = np.asarray(U[:, :rank], dtype=np.float64)
    return Q_phi, singular_values, threshold


def _r3_mode_residual_rows(
    *,
    system: SystemName,
    operator_name: OperatorName,
    M: FloatArray,
    Q_phi: FloatArray,
    Phi_sub: FloatArray,
    Chi_sub: FloatArray,
    representation_ridge: float,
    rank_rtol: float,
    rank_threshold: float,
    singular_values: FloatArray,
    residual_thresholds: tuple[float, ...],
) -> tuple[list[CsvRow], CsvRow]:
    """Compute rank-aware manuscript-R3 spectral residual rows.

    Candidates are eigenpairs of Q_phi^T M Q_phi.  Each lifted candidate
    w = Q_phi w_tilde is represented by held-out kernel sections Phi_sub v,
    then verified with the successor-normalized squared residual

        rho = ||Chi_sub v - lambda Phi_sub v||^2 / ||Chi_sub v||^2.
    """
    M_full = np.asarray(M, dtype=np.float64)
    Q = np.asarray(Q_phi, dtype=np.float64)
    Phi = np.asarray(Phi_sub, dtype=np.float64)
    Chi = np.asarray(Chi_sub, dtype=np.float64)

    c = int(M_full.shape[0])
    rank = int(Q.shape[1])
    n_sub = int(Phi.shape[1])

    if M_full.shape != (c, c):
        raise ValueError(f"M must be square, got {M_full.shape}.")
    if Q.shape[0] != c:
        raise ValueError(f"Q_phi shape {Q.shape} is incompatible with M shape {M_full.shape}.")
    if Phi.shape[0] != c or Chi.shape != Phi.shape:
        raise ValueError(
            f"Phi_sub/Chi_sub must both have shape (C, N); got {Phi.shape} and {Chi.shape} for C={c}."
        )

    M_reduced = np.asarray(Q.T @ M_full @ Q, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eig(M_reduced.astype(np.complex128))

    # Held-out finite-rank KAHM Gram matrices used to verify the direct R3 formula.
    G = Phi.T @ Phi
    A = Chi.T @ Phi
    R = Chi.T @ Chi

    rows: list[CsvRow] = []
    rho_values: list[float] = []
    matrix_rho_values: list[float] = []
    eta_values: list[float] = []
    legacy_reduced_values: list[float] = []

    Phi_complex = Phi.astype(np.complex128)
    Chi_complex = Chi.astype(np.complex128)
    G_complex = G.astype(np.complex128)
    A_complex = A.astype(np.complex128)
    R_complex = R.astype(np.complex128)

    sigma_max = float(singular_values[0]) if singular_values.size else float("nan")
    sigma_min_retained = (
        float(singular_values[rank - 1])
        if singular_values.size >= rank and rank > 0
        else float("nan")
    )

    for mode_idx in range(rank):
        lam = complex(eigvals[mode_idx])
        w_tilde = eigvecs[:, mode_idx].astype(np.complex128, copy=False)
        reduced_norm = float(np.linalg.norm(w_tilde))
        if reduced_norm <= 1e-14 or not math.isfinite(reduced_norm):
            continue
        w_tilde = w_tilde / reduced_norm

        w = (Q.astype(np.complex128) @ w_tilde).astype(np.complex128, copy=False)
        w_norm_sq = float(np.real(np.vdot(w, w)))
        if w_norm_sq <= 1e-28 or not math.isfinite(w_norm_sq):
            continue

        v = _preimage_coefficients(Phi, w, ridge=float(representation_ridge))
        phi_v = (Phi_complex @ v).astype(np.complex128, copy=False)
        chi_v = (Chi_complex @ v).astype(np.complex128, copy=False)

        rep_residual = phi_v - w
        eta_rep = float(np.real(np.vdot(rep_residual, rep_residual)) / w_norm_sq)

        spectral_residual = chi_v - lam * phi_v
        numerator_sq = float(np.real(np.vdot(spectral_residual, spectral_residual)))
        successor_norm_sq = float(np.real(np.vdot(chi_v, chi_v)))
        source_norm_sq = float(np.real(np.vdot(phi_v, phi_v)))

        if successor_norm_sq <= 1e-28 or not math.isfinite(successor_norm_sq):
            rho_feature = float("nan")
        else:
            rho_feature = float(max(numerator_sq, 0.0) / successor_norm_sq)

        # Independent matrix-form verification:
        # ||(Chi-lambda Phi)v||^2 =
        # v^*(R-lambda A-conj(lambda)A^*+|lambda|^2G)v.
        v_col = v.reshape(-1, 1)
        residual_matrix = (
            R_complex
            - lam * A_complex
            - np.conj(lam) * A_complex.conj().T
            + (abs(lam) ** 2) * G_complex
        )
        matrix_numerator = complex((v_col.conj().T @ residual_matrix @ v_col)[0, 0])
        matrix_successor_denominator = complex((v_col.conj().T @ R_complex @ v_col)[0, 0])
        if abs(matrix_successor_denominator) <= 1e-28:
            rho_matrix = float("nan")
        else:
            rho_matrix = float(
                max(float(np.real(matrix_numerator / matrix_successor_denominator)), 0.0)
            )

        # Same reduced candidate under the previous source-normalized,
        # unsquared residual.  This is diagnostic only; the exact old ambient-C
        # procedure is written separately by _legacy_ambient_mode_residual_rows.
        if source_norm_sq <= 1e-28 or not math.isfinite(source_norm_sq):
            legacy_source_residual_reduced_candidate = float("nan")
        else:
            legacy_source_residual_reduced_candidate = float(
                math.sqrt(max(numerator_sq, 0.0) / source_norm_sq)
            )

        rows.append(
            {
                "system": system,
                "operator": operator_name,
                "mode_index": int(mode_idx),
                "lambda_real": float(np.real(lam)),
                "lambda_imag": float(np.imag(lam)),
                "lambda_abs": float(abs(lam)),
                "r3_residual_rho": rho_feature,
                "r3_residual_matrix_formula": rho_matrix,
                "r3_feature_matrix_formula_abs_diff": (
                    abs(rho_feature - rho_matrix)
                    if math.isfinite(rho_feature) and math.isfinite(rho_matrix)
                    else float("nan")
                ),
                "representation_defect_eta": eta_rep,
                "legacy_source_normalized_residual_reduced_candidate": (
                    legacy_source_residual_reduced_candidate
                ),
                "successor_denominator_norm_sq": successor_norm_sq,
                "source_denominator_norm_sq": source_norm_sq,
                "representation_ridge": float(representation_ridge),
                "rank_rtol": float(rank_rtol),
                "rank_threshold": float(rank_threshold),
                "sigma_max": sigma_max,
                "sigma_min_retained": sigma_min_retained,
                "effective_feature_rank": rank,
                "ambient_num_regimes": c,
                "subsample_size": n_sub,
            }
        )

        if math.isfinite(rho_feature):
            rho_values.append(rho_feature)
        if math.isfinite(rho_matrix):
            matrix_rho_values.append(rho_matrix)
        if math.isfinite(eta_rep):
            eta_values.append(eta_rep)
        if math.isfinite(legacy_source_residual_reduced_candidate):
            legacy_reduced_values.append(legacy_source_residual_reduced_candidate)

    rows.sort(
        key=lambda row: (
            not math.isfinite(float(row["r3_residual_rho"])),
            float(row["r3_residual_rho"]),
        )
    )
    for rank_index, row in enumerate(rows, start=1):
        row["rank_by_r3_residual"] = int(rank_index)

    rho_arr = np.asarray(rho_values, dtype=np.float64)
    matrix_rho_arr = np.asarray(matrix_rho_values, dtype=np.float64)
    eta_arr = np.asarray(eta_values, dtype=np.float64)
    legacy_arr = np.asarray(legacy_reduced_values, dtype=np.float64)
    eig_abs = np.abs(eigvals)

    summary: CsvRow = {
        "system": system,
        "operator": operator_name,
        "ambient_num_regimes": c,
        "effective_feature_rank": rank,
        "rank_rtol": float(rank_rtol),
        "rank_threshold": float(rank_threshold),
        "sigma_max": sigma_max,
        "sigma_min_retained": sigma_min_retained,
        "subsample_size": n_sub,
        "num_valid_r3_modes": int(rho_arr.size),
        "spectral_radius": float(np.max(eig_abs)) if eig_abs.size else float("nan"),
        "reduced_spectral_radius": float(np.max(eig_abs)) if eig_abs.size else float("nan"),
        "max_abs_imag_lambda": (
            float(np.max(np.abs(np.imag(eigvals)))) if eigvals.size else float("nan")
        ),
        "min_r3_residual_rho": float(np.min(rho_arr)) if rho_arr.size else float("nan"),
        "median_r3_residual_rho": (
            float(np.median(rho_arr)) if rho_arr.size else float("nan")
        ),
        "mean_r3_residual_rho": float(np.mean(rho_arr)) if rho_arr.size else float("nan"),
        "q25_r3_residual_rho": (
            float(np.quantile(rho_arr, 0.25)) if rho_arr.size else float("nan")
        ),
        "q75_r3_residual_rho": (
            float(np.quantile(rho_arr, 0.75)) if rho_arr.size else float("nan")
        ),
        "max_r3_residual_rho": float(np.max(rho_arr)) if rho_arr.size else float("nan"),
        "median_representation_defect_eta": (
            float(np.median(eta_arr)) if eta_arr.size else float("nan")
        ),
        "max_representation_defect_eta": (
            float(np.max(eta_arr)) if eta_arr.size else float("nan")
        ),
        "median_r3_matrix_formula_rho": (
            float(np.median(matrix_rho_arr)) if matrix_rho_arr.size else float("nan")
        ),
        "max_r3_feature_matrix_formula_abs_diff": (
            max(
                (
                    float(row["r3_feature_matrix_formula_abs_diff"])
                    for row in rows
                    if math.isfinite(float(row["r3_feature_matrix_formula_abs_diff"]))
                ),
                default=float("nan"),
            )
        ),
        "median_legacy_source_residual_reduced_candidate": (
            float(np.median(legacy_arr)) if legacy_arr.size else float("nan")
        ),
    }
    for threshold in residual_thresholds:
        key = f"modes_with_r3_rho_le_{threshold:g}"
        summary[key] = int(np.sum(rho_arr <= float(threshold))) if rho_arr.size else 0
    return rows, summary


def _legacy_ambient_mode_residual_rows(
    *,
    system: SystemName,
    operator_name: OperatorName,
    M: FloatArray,
    Phi_sub: FloatArray,
    Chi_sub: FloatArray,
    ridge: float,
    residual_thresholds: tuple[float, ...],
) -> tuple[list[CsvRow], CsvRow]:
    """Reproduce the pre-R3 ambient-C/source-normalized diagnostic for provenance."""
    eigvals, eigvecs = np.linalg.eig(np.asarray(M, dtype=np.float64).astype(np.complex128))
    c = int(M.shape[0])
    n_sub = int(Phi_sub.shape[1])

    G = Phi_sub.T @ Phi_sub
    A = Chi_sub.T @ Phi_sub
    R = Chi_sub.T @ Chi_sub

    rows: list[CsvRow] = []
    residuals: list[float] = []
    matrix_residuals: list[float] = []
    representation_errors: list[float] = []

    for mode_idx in range(c):
        lam = complex(eigvals[mode_idx])
        w = eigvecs[:, mode_idx].astype(np.complex128, copy=False)
        w_norm = float(np.linalg.norm(w))
        if w_norm <= 1e-14:
            continue
        w = w / w_norm

        v = _preimage_coefficients(Phi_sub, w, ridge=float(ridge))
        phi_v = (Phi_sub.astype(np.complex128) @ v).astype(np.complex128, copy=False)
        chi_v = (Chi_sub.astype(np.complex128) @ v).astype(np.complex128, copy=False)

        denom = float(np.linalg.norm(phi_v))
        if denom <= 1e-14 or not math.isfinite(denom):
            feature_residual = float("nan")
        else:
            feature_residual = float(np.linalg.norm(chi_v - lam * phi_v) / denom)

        rep_den = max(float(np.linalg.norm(w)), 1e-14)
        representation_error = float(np.linalg.norm(phi_v - w) / rep_den)

        v_col = v.reshape(-1, 1)
        mat = (
            R.astype(np.complex128)
            - lam * A.astype(np.complex128)
            - np.conj(lam) * A.T.astype(np.complex128)
            + (abs(lam) ** 2) * G.astype(np.complex128)
        )
        numerator = complex((v_col.conj().T @ mat @ v_col)[0, 0])
        denominator = complex(
            (v_col.conj().T @ G.astype(np.complex128) @ v_col)[0, 0]
        )
        if abs(denominator) <= 1e-14:
            matrix_residual = float("nan")
        else:
            quotient = max(float(np.real(numerator / denominator)), 0.0)
            matrix_residual = float(math.sqrt(quotient))

        rows.append(
            {
                "system": system,
                "operator": operator_name,
                "mode_index": int(mode_idx),
                "lambda_real": float(np.real(lam)),
                "lambda_imag": float(np.imag(lam)),
                "lambda_abs": float(abs(lam)),
                "legacy_rkhs_residual_feature": feature_residual,
                "legacy_rkhs_residual_matrix_formula": matrix_residual,
                "legacy_feature_matrix_formula_abs_diff": (
                    abs(feature_residual - matrix_residual)
                    if math.isfinite(feature_residual) and math.isfinite(matrix_residual)
                    else float("nan")
                ),
                "legacy_representation_error_unsquared": representation_error,
                "legacy_denominator_norm": denom,
                "subsample_size": n_sub,
                "num_regimes": c,
            }
        )
        if math.isfinite(feature_residual):
            residuals.append(feature_residual)
        if math.isfinite(matrix_residual):
            matrix_residuals.append(matrix_residual)
        if math.isfinite(representation_error):
            representation_errors.append(representation_error)

    rows.sort(
        key=lambda row: (
            not math.isfinite(float(row["legacy_rkhs_residual_feature"])),
            float(row["legacy_rkhs_residual_feature"]),
        )
    )
    for rank_index, row in enumerate(rows, start=1):
        row["legacy_rank_by_residual"] = int(rank_index)

    res_arr = np.asarray(residuals, dtype=np.float64)
    rep_arr = np.asarray(representation_errors, dtype=np.float64)
    eig_abs = np.abs(eigvals)
    summary: CsvRow = {
        "system": system,
        "operator": operator_name,
        "num_regimes": c,
        "subsample_size": n_sub,
        "num_valid_modes": int(res_arr.size),
        "legacy_ambient_spectral_radius": (
            float(np.max(eig_abs)) if eig_abs.size else float("nan")
        ),
        "legacy_max_abs_imag_lambda": (
            float(np.max(np.abs(np.imag(eigvals)))) if eigvals.size else float("nan")
        ),
        "legacy_min_rkhs_residual": (
            float(np.min(res_arr)) if res_arr.size else float("nan")
        ),
        "legacy_median_rkhs_residual": (
            float(np.median(res_arr)) if res_arr.size else float("nan")
        ),
        "legacy_mean_rkhs_residual": (
            float(np.mean(res_arr)) if res_arr.size else float("nan")
        ),
        "legacy_q25_rkhs_residual": (
            float(np.quantile(res_arr, 0.25)) if res_arr.size else float("nan")
        ),
        "legacy_q75_rkhs_residual": (
            float(np.quantile(res_arr, 0.75)) if res_arr.size else float("nan")
        ),
        "legacy_max_rkhs_residual": (
            float(np.max(res_arr)) if res_arr.size else float("nan")
        ),
        "legacy_median_representation_error_unsquared": (
            float(np.median(rep_arr)) if rep_arr.size else float("nan")
        ),
        "legacy_max_representation_error_unsquared": (
            float(np.max(rep_arr)) if rep_arr.size else float("nan")
        ),
        "legacy_median_matrix_formula_residual": (
            float(np.median(np.asarray(matrix_residuals, dtype=np.float64)))
            if matrix_residuals
            else float("nan")
        ),
    }
    for threshold in residual_thresholds:
        key = f"legacy_modes_with_residual_le_{threshold:g}"
        summary[key] = int(np.sum(res_arr <= float(threshold))) if res_arr.size else 0
    return rows, summary


def _write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[CsvRow]:
    """Read an existing experiment CSV for safe --skip-existing resumption."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]  # type: ignore[return-value]


def _plot_spectrum(
    output_dir: Path,
    rows: Sequence[CsvRow],
    *,
    system: str,
    operator_name: str,
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
        import matplotlib.patches as mpatches  # type: ignore[import]
    except Exception:
        return

    x = np.asarray([float(row["lambda_real"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["lambda_imag"]) for row in rows], dtype=np.float64)
    rho = np.asarray([float(row["r3_residual_rho"]) for row in rows], dtype=np.float64)
    rho_plot = np.log10(np.maximum(rho, 1e-16))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(x, y, c=rho_plot, s=36)
    unit = mpatches.Circle((0.0, 0.0), 1.0, fill=False, linestyle="--", linewidth=1.0)
    ax.add_patch(unit)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title(f"R3-verified reduced KAHKM spectrum: {system}, {operator_name}")
    ax.set_aspect("equal", adjustable="datalim")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10 R3 residual rho")
    fig.tight_layout()
    fig.savefig(output_dir / f"experiment_19_spectrum_{system}_{operator_name}.png", dpi=200)
    plt.close(fig)


def _system_config(system: SystemName, args: ExperimentArgs) -> SystemConfig:
    if system == "duffing":
        return SystemConfig(
            name=system,
            dt=0.03,
            n_clusters=int(args.duffing_c),
            omega=float(args.duffing_omega),
            tau=float(args.tau),
            subspace_dim=int(args.subspace_dim),
        )
    if system == "vanderpol":
        return SystemConfig(
            name=system,
            dt=0.02,
            n_clusters=int(args.vanderpol_c),
            omega=float(args.vanderpol_omega),
            tau=float(args.tau),
            subspace_dim=int(args.subspace_dim),
        )
    if system == "acrobot":
        return SystemConfig(
            name=system,
            dt=float("nan"),
            n_clusters=int(args.acrobot_c),
            omega=float(args.acrobot_omega),
            tau=float(args.tau),
            subspace_dim=int(args.acrobot_subspace_dim),
        )
    raise ValueError(f"Unknown system: {system}")


def _load_system_data(system: SystemName, cfg: SystemConfig, args: ExperimentArgs) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if system in {"duffing", "vanderpol"}:
        X0_train, X1_train = make_concatenated_training_data(
            system,
            seeds=args.train_seeds,
            n_steps=int(args.n_steps_train),
            dt=float(cfg.dt),
            vanderpol_mu=float(args.vanderpol_mu),
        )
        X0_test, X1_test = make_concatenated_training_data(
            system,
            seeds=args.test_seeds,
            n_steps=int(args.n_steps_test),
            dt=float(cfg.dt),
            vanderpol_mu=float(args.vanderpol_mu),
        )
        return X0_train, X1_train, X0_test, X1_test
    if system == "acrobot":
        return collect_acrobot_train_test(args)
    raise ValueError(f"Unknown system: {system}")


def _fit_for_system(system: SystemName, cfg: SystemConfig, args: ExperimentArgs, output_dir: Path) -> KAHKMFitResult:
    X0_train, X1_train, _X0_test, _X1_test = _load_system_data(system, cfg, args)
    print(f"Training data for {system}: X0={X0_train.shape}, X1={X1_train.shape}")
    effective_dim = min(int(cfg.subspace_dim), int(X0_train.shape[0]))
    print(
        f"Fitting KAHKM for {system}: C={cfg.n_clusters}, omega={cfg.omega}, "
        f"requested_subspace_dim={cfg.subspace_dim}, effective_subspace_dim={effective_dim}"
    )
    ae_dir = str(output_dir / f"ae_cache_{system}") if args.save_ae_to_disk else None
    return fit_kahkm(
        X0_train,
        X1_train,
        n_clusters=int(cfg.n_clusters),
        subspace_dim=effective_dim,
        Nb=int(args.nb),
        omega=float(cfg.omega),
        tau=float(cfg.tau),
        beta=float(args.beta),
        nlms_epochs=int(args.nlms_epochs),
        random_state=int(args.random_state),
        kmeans_kind=cast(Any, args.kmeans_kind),
        kmeans_batch_size=int(args.kmeans_batch_size),
        singleton_strategy="augment",
        max_train_per_cluster=args.max_train_per_cluster,
        save_ae_to_disk=bool(args.save_ae_to_disk),
        ae_dir=ae_dir,
        overwrite_ae_dir=False,
        n_jobs=int(args.n_jobs),
        batch_size=int(args.batch_size),
        project_stochastic=True,
        verbose=True,
    )


def run() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload: dict[str, Any] = dict(asdict(args))
    config_payload["diagnostic_definition"] = {
        "feature_basis": "uncentered SVD of Phi_train; retain sigma_j > rank_rtol * sigma_max",
        "reduced_operator": "M_red = Q_Phi.T @ M @ Q_Phi",
        "representation_defect_eta": "||Phi_spec v - w||^2 / ||w||^2",
        "r3_residual_rho": "||Chi_spec v - lambda Phi_spec v||^2 / ||Chi_spec v||^2",
        "matrix_formula_denominator": "v^* R v, R = Chi_spec^T Chi_spec",
        "legacy_outputs": (
            "Separate CSVs reproduce the previous ambient-C candidate construction "
            "and source-normalized unsquared residual."
        ),
    }
    with (output_dir / "experiment_19_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    mode_path = output_dir / "experiment_19_mode_residuals.csv"
    summary_path = output_dir / "experiment_19_system_summary.csv"
    legacy_mode_path = output_dir / "experiment_19_legacy_ambient_mode_residuals.csv"
    legacy_summary_path = output_dir / "experiment_19_legacy_ambient_system_summary.csv"

    if args.skip_existing:
        all_mode_rows = _read_csv_rows(mode_path)
        all_summary_rows = _read_csv_rows(summary_path)
        all_legacy_mode_rows = _read_csv_rows(legacy_mode_path)
        all_legacy_summary_rows = _read_csv_rows(legacy_summary_path)
    else:
        all_mode_rows: list[CsvRow] = []
        all_summary_rows: list[CsvRow] = []
        all_legacy_mode_rows: list[CsvRow] = []
        all_legacy_summary_rows: list[CsvRow] = []

    rng = np.random.default_rng(int(args.random_state))

    for system in args.systems:
        cfg = _system_config(system, args)
        marker = output_dir / f"_{system}_done.marker"
        if args.skip_existing and marker.exists():
            print(f"Skipping {system}: marker exists at {marker}")
            continue

        print(f"\n=== Experiment 19 R3 residual-verified spectra: {system} ===")
        start = time.time()
        X0_train, X1_train, X0_test, X1_test = _load_system_data(system, cfg, args)
        print(
            f"Train snapshots: {X0_train.shape[1]}; test snapshots: {X0_test.shape[1]}; "
            f"state_dim={X0_train.shape[0]}"
        )

        effective_dim = min(int(cfg.subspace_dim), int(X0_train.shape[0]))
        ae_dir = str(output_dir / f"ae_cache_{system}") if args.save_ae_to_disk else None
        fit = fit_kahkm(
            X0_train,
            X1_train,
            n_clusters=int(cfg.n_clusters),
            subspace_dim=effective_dim,
            Nb=int(args.nb),
            omega=float(cfg.omega),
            tau=float(cfg.tau),
            beta=float(args.beta),
            nlms_epochs=int(args.nlms_epochs),
            random_state=int(args.random_state),
            kmeans_kind=cast(Any, args.kmeans_kind),
            kmeans_batch_size=int(args.kmeans_batch_size),
            singleton_strategy="augment",
            max_train_per_cluster=args.max_train_per_cluster,
            save_ae_to_disk=bool(args.save_ae_to_disk),
            ae_dir=ae_dir,
            overwrite_ae_dir=False,
            n_jobs=int(args.n_jobs),
            batch_size=int(args.batch_size),
            project_stochastic=True,
            verbose=True,
        )

        print("Computing train/test KAHM associations for residual diagnostics...")
        Phi_train = kahm_associations(
            fit.abstraction_model,
            X0_train,
            omega=fit.omega,
            tau=fit.tau,
            n_jobs=int(args.n_jobs),
            batch_size=int(args.batch_size),
            show_progress=True,
        )
        Chi_train = kahm_associations(
            fit.abstraction_model,
            X1_train,
            omega=fit.omega,
            tau=fit.tau,
            n_jobs=int(args.n_jobs),
            batch_size=int(args.batch_size),
            show_progress=True,
        )
        Phi_test = kahm_associations(
            fit.abstraction_model,
            X0_test,
            omega=fit.omega,
            tau=fit.tau,
            n_jobs=int(args.n_jobs),
            batch_size=int(args.batch_size),
            show_progress=True,
        )
        Chi_test = kahm_associations(
            fit.abstraction_model,
            X1_test,
            omega=fit.omega,
            tau=fit.tau,
            n_jobs=int(args.n_jobs),
            batch_size=int(args.batch_size),
            show_progress=True,
        )

        Q_phi, singular_values, rank_threshold = _training_feature_basis(
            Phi_train,
            rank_rtol=float(args.rank_rtol),
        )
        feature_rank = int(Q_phi.shape[1])
        print(
            f"Training-supported coefficient rank for {system}: "
            f"r_Phi={feature_rank}/{cfg.n_clusters}, "
            f"threshold={rank_threshold:.6g}, "
            f"sigma_max={float(singular_values[0]):.6g}, "
            f"sigma_min_retained={float(singular_values[feature_rank - 1]):.6g}"
        )
        np.savez_compressed(
            output_dir / f"experiment_19_{system}_feature_rank.npz",
            Q_phi=Q_phi,
            singular_values=singular_values,
            rank_threshold=np.asarray([rank_threshold], dtype=np.float64),
            rank_rtol=np.asarray([float(args.rank_rtol)], dtype=np.float64),
        )

        Phi_sub, Chi_sub, subsample_idx = _select_subsample(
            Phi_test,
            Chi_test,
            spectral_subsample=int(args.spectral_subsample),
            rng=rng,
        )
        np.savez_compressed(
            output_dir / f"experiment_19_{system}_subsample_indices.npz",
            subsample_idx=subsample_idx,
        )

        pred_train_nlms = fit.B.T @ Phi_train
        pred_test_nlms = fit.B.T @ Phi_test
        M_ridge = _ridge_feature_operator(Phi_train, Chi_train, ridge=float(args.ridge))
        pred_train_ridge = M_ridge @ Phi_train
        pred_test_ridge = M_ridge @ Phi_test

        closure_by_operator: dict[OperatorName, dict[str, float]] = {
            "nlms": {
                "train_closure_error": _relative_error(pred_train_nlms, Chi_train),
                "test_closure_error": _relative_error(pred_test_nlms, Chi_test),
                "test_association_r2": _r2(pred_test_nlms, Chi_test),
                "test_simplex_violation": simplex_violation(pred_test_nlms),
            },
            "ridge_ls": {
                "train_closure_error": _relative_error(pred_train_ridge, Chi_train),
                "test_closure_error": _relative_error(pred_test_ridge, Chi_test),
                "test_association_r2": _r2(pred_test_ridge, Chi_test),
                "test_simplex_violation": simplex_violation(pred_test_ridge),
            },
        }
        operator_matrices: dict[OperatorName, FloatArray] = {
            # fit.B.T maps current associations to next associations.
            "nlms": np.asarray(fit.B.T, dtype=np.float64),
            "ridge_ls": np.asarray(M_ridge, dtype=np.float64),
        }

        for operator_name in args.operators:
            print(
                f"Computing rank-aware R3 modes for {system}, operator={operator_name}..."
            )
            mode_rows, summary = _r3_mode_residual_rows(
                system=system,
                operator_name=operator_name,
                M=operator_matrices[operator_name],
                Q_phi=Q_phi,
                Phi_sub=Phi_sub,
                Chi_sub=Chi_sub,
                representation_ridge=float(args.representation_ridge),
                rank_rtol=float(args.rank_rtol),
                rank_threshold=float(rank_threshold),
                singular_values=singular_values,
                residual_thresholds=args.residual_thresholds,
            )
            summary.update(
                {
                    "n_clusters": int(cfg.n_clusters),
                    "omega": float(cfg.omega),
                    "tau": float(cfg.tau),
                    "state_dim": int(X0_train.shape[0]),
                    "requested_subspace_dim": int(cfg.subspace_dim),
                    "effective_subspace_dim": effective_dim,
                    "train_snapshots": int(X0_train.shape[1]),
                    "test_snapshots": int(X0_test.shape[1]),
                    "fit_train_closure_error": float(fit.train_closure_error),
                    "fit_train_association_r2": float(fit.association_r2),
                    "wall_seconds_so_far_system": float(time.time() - start),
                }
            )
            summary.update(closure_by_operator[operator_name])
            all_summary_rows.append(summary)
            all_mode_rows.extend(mode_rows)

            legacy_rows, legacy_summary = _legacy_ambient_mode_residual_rows(
                system=system,
                operator_name=operator_name,
                M=operator_matrices[operator_name],
                Phi_sub=Phi_sub,
                Chi_sub=Chi_sub,
                ridge=float(args.ridge),
                residual_thresholds=args.residual_thresholds,
            )
            legacy_summary.update(
                {
                    "n_clusters": int(cfg.n_clusters),
                    "omega": float(cfg.omega),
                    "tau": float(cfg.tau),
                    "wall_seconds_so_far_system": float(time.time() - start),
                }
            )
            legacy_summary.update(closure_by_operator[operator_name])
            all_legacy_summary_rows.append(legacy_summary)
            all_legacy_mode_rows.extend(legacy_rows)

            if not args.no_plots:
                _plot_spectrum(
                    output_dir,
                    mode_rows,
                    system=system,
                    operator_name=operator_name,
                )

            print(
                f"{system}/{operator_name}: "
                f"r_Phi={feature_rank}, "
                f"min R3 rho={float(summary['min_r3_residual_rho']):.6g}, "
                f"median R3 rho={float(summary['median_r3_residual_rho']):.6g}, "
                f"median eta_rep={float(summary['median_representation_defect_eta']):.6g}, "
                f"test R2={float(summary['test_association_r2']):.6g}"
            )
            top = mode_rows[: max(int(args.top_modes), 0)]
            for row in top:
                print(
                    f"  rank {int(row['rank_by_r3_residual']):>2}: "
                    f"lambda={float(row['lambda_real']):+.5f}"
                    f"{float(row['lambda_imag']):+.5f}i, "
                    f"|lambda|={float(row['lambda_abs']):.5f}, "
                    f"rho={float(row['r3_residual_rho']):.6g}, "
                    f"eta_rep={float(row['representation_defect_eta']):.3g}"
                )

        marker.write_text("done\n", encoding="utf-8")

        # Write incrementally so partial results survive long runs.
        _write_csv(mode_path, all_mode_rows)
        _write_csv(summary_path, all_summary_rows)
        _write_csv(legacy_mode_path, all_legacy_mode_rows)
        _write_csv(legacy_summary_path, all_legacy_summary_rows)

    _write_csv(mode_path, all_mode_rows)
    _write_csv(summary_path, all_summary_rows)
    _write_csv(legacy_mode_path, all_legacy_mode_rows)
    _write_csv(legacy_summary_path, all_legacy_summary_rows)

    zip_base = str(Path(args.zip_name))
    if zip_base.endswith(".zip"):
        zip_base = zip_base[:-4]
    archive_path = shutil.make_archive(zip_base, "zip", output_dir)
    print(f"\nWrote R3 results to: {output_dir.resolve()}")
    print(f"Wrote archive to: {archive_path}")


if __name__ == "__main__":
    run()
