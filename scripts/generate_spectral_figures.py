#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SOURCE = Path("kahkm_experiment_19_residual_verified_spectra/experiment_19_mode_residuals.csv")
FIGURE_DIR = Path("figures")

OUTPUTS = {
    ("duffing", "nlms"): "fig_spectral_duffing_nlms.pdf",
    ("duffing", "ridge_ls"): "fig_spectral_duffing_ridge_ls.pdf",
    ("vanderpol", "nlms"): "fig_spectral_vanderpol_nlms.pdf",
    ("vanderpol", "ridge_ls"): "fig_spectral_vanderpol_ridge_ls.pdf",
}

def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def plot(rows: list[dict[str, str]], system: str, operator: str, output: Path) -> None:
    selected = [
        row for row in rows
        if row["system"] == system and row["operator"] == operator
    ]

    if not selected:
        raise RuntimeError(f"No rows found for {system}/{operator}")

    x = np.asarray([float(row["lambda_real"]) for row in selected], dtype=float)
    y = np.asarray([float(row["lambda_imag"]) for row in selected], dtype=float)
    residual = np.asarray([
        float(row["rkhs_residual_feature"]) for row in selected
    ], dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(residual)
    x = x[finite]
    y = y[finite]
    residual = residual[finite]

    if x.size == 0:
        raise RuntimeError(f"No finite spectral rows for {system}/{operator}")

    residual_plot = np.log10(np.maximum(residual, 1e-14))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    scatter = ax.scatter(x, y, c=residual_plot, s=36)

    unit = mpatches.Circle(
        (0.0, 0.0),
        1.0,
        fill=False,
        linestyle="--",
        linewidth=1.0,
    )
    ax.add_patch(unit)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title(f"Residual-verified KAHKM spectrum: {system}, {operator}")
    ax.set_aspect("equal", adjustable="datalim")

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("log10 RKHS residual")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    rows = read_rows()

    for key, filename in OUTPUTS.items():
        system, operator = key
        output = FIGURE_DIR / filename
        plot(rows, system, operator, output)
        print(f"Wrote {output}")

if __name__ == "__main__":
    main()
