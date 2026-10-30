#!/usr/bin/env python3
"""Retrospective descriptive checks; never fit a model or change primary results.

From the extension root:
  python scripts/review_diagnostics.py --output manuscript/generated
  python scripts/review_diagnostics.py --run-dir runs/NAME --output manuscript/generated

The first command checks archived paired differences. The second additionally
reads locally generated banks from a completed full reproduction, replays the
original test inputs, checks their hashes, and describes local-basis conditioning
and test-gate ranges. It creates only files named review_* in the chosen output.
Deleting one repetition is an influence diagnostic, not a new confidence family.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.stats import skew, t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduce.io import digest, read_json, verify_archive, write_json
from scripts.build_paper_assets import Archive, SYSTEMS, SYSTEM_LABELS, verify_contrasts


def paired_interval(differences, family_size, alpha=0.05):
    """Student interval for a mean at the unchanged family-wise tail allocation."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("At least three finite paired differences are required")
    if isinstance(family_size, bool) or int(family_size) != family_size or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    critical = float(student_t.ppf(1.0 - alpha / (2 * family_size), n - 1))
    return {"n": n, "degrees_of_freedom": n - 1, "family_size": int(family_size),
            "mean": mean, "sd": sd, "se": se, "critical_t": critical,
            "lower": mean - critical * se, "upper": mean + critical * se}


def paired_diagnostics(differences, family_size, margin, alpha=0.05):
    """Describe the supplied paired observations and every one-case deletion."""
    values = np.asarray(differences, dtype=float)
    original = paired_interval(values, family_size, alpha)
    if len(values) < 4 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Need at least four differences and a finite nonnegative margin")
    deletion = []
    for index in range(len(values)):
        interval = paired_interval(np.delete(values, index), family_size, alpha)
        deletion.append({"omitted_replicate": index, **interval})
    lower = [row["lower"] for row in deletion]
    upper = [row["upper"] for row in deletion]
    return {
        "original": original, "practical_margin": float(margin),
        "bias_corrected_sample_skewness": float(skew(values, bias=False)) if original["sd"] > 0 else None,
        "minimum": float(values.min()), "median": float(np.median(values)),
        "maximum": float(values.max()), "positive_repetitions": int(np.count_nonzero(values > 0)),
        "repetitions_above_margin": int(np.count_nonzero(values > margin)),
        "leave_one_out": deletion,
        "leave_one_out_min_lower": min(lower), "leave_one_out_max_lower": max(lower),
        "leave_one_out_min_upper": min(upper), "leave_one_out_max_upper": max(upper),
        "every_leave_one_out_lower_above_margin": bool(min(lower) > margin),
        "paired_differences": values.tolist(),
    }


def stacked_basis_diagnostics(bases):
    """Numerical SVD of vertically stacked local transposed bases.

    Full numerical column rank is a sufficient coordinate-recovery diagnostic.
    A deficient stack does not imply that the complete representation loses
    information: its gates can encode directions missing from the stack.
    """
    matrices = [np.asarray(basis, dtype=float) for basis in bases]
    if not matrices or any(b.ndim != 2 or not np.isfinite(b).all() for b in matrices):
        raise ValueError("Expected a nonempty list of finite two-dimensional bases")
    dimension = matrices[0].shape[0]
    if dimension == 0 or any(b.shape[0] != dimension for b in matrices):
        raise ValueError("Bases must share a positive ambient dimension")
    stack = np.concatenate([basis.T for basis in matrices], axis=0)
    singular = np.linalg.svd(stack, compute_uv=False)
    singular = np.pad(singular, (0, max(0, dimension - len(singular))))
    largest = float(singular[0])
    tolerance = largest * max(stack.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > tolerance))
    smallest = float(singular[-1])
    full = rank == dimension
    return {"ambient_dimension": dimension, "stacked_rows": stack.shape[0],
            "numerical_rank": rank, "singular_values": singular.tolist(),
            "sigma_max": largest, "sigma_min": smallest, "rank_tolerance": tolerance,
            "rank_tolerance_rule": "sigma_max * max(rows, columns) * eps(float64)",
            "full_numerical_column_rank": full,
            "condition_number_2": largest / smallest if full else None,
            "linear_inverse_norm_2": 1.0 / smallest if full else None}


def recover_from_gated_coordinates(phi, gated, bases, centers):
    """Recover scaled queries from positive gates and a full-rank basis stack.

    This describes an algebraic inverse. It is not a trained downstream head,
    a stability certificate, or a converse when the basis stack is deficient.
    """
    phi, gated = np.asarray(phi, dtype=float), np.asarray(gated, dtype=float)
    bases = [np.asarray(basis, dtype=float) for basis in bases]
    centers = [np.asarray(center, dtype=float) for center in centers]
    summary = stacked_basis_diagnostics(bases)
    if not summary["full_numerical_column_rank"]:
        raise ValueError("The stacked coordinates do not have full numerical column rank")
    if (phi.ndim != 2 or phi.shape[1] != len(bases) or np.any(phi <= 0)
            or not np.isfinite(phi).all() or gated.ndim != 2
            or gated.shape != (len(phi), sum(b.shape[1] for b in bases))
            or not np.isfinite(gated).all() or len(centers) != len(bases)
            or any(c.shape != (b.shape[0],) or not np.isfinite(c).all()
                   for c, b in zip(centers, bases))):
        raise ValueError("Incompatible finite coordinates, centers, or nonpositive gates")
    projections, start = [], 0
    for index, (basis, center) in enumerate(zip(bases, centers)):
        stop = start + basis.shape[1]
        projections.append(gated[:, start:stop] / phi[:, index:index + 1] + center @ basis)
        start = stop
    stack = np.concatenate([basis.T for basis in bases], axis=0)
    right = np.concatenate(projections, axis=1)
    return np.linalg.lstsq(stack, right.T, rcond=None)[0].T


def contrast_checks(input_directory):
    archive = Archive(input_directory)
    stage1 = archive.read("stage1/records.json")
    capacity = archive.read("capacity/capacity_records.json")
    first = verify_contrasts(stage1, archive.read("stage1/primary_contrasts.csv"),
                             archive.read("stage1/margins.json"))
    second = verify_contrasts(capacity, archive.read("capacity/capacity_primary_contrasts.csv"),
                              archive.read("capacity/capacity_margins.json"), capacity=True)
    results = []
    for family, rows in (("stage1", first), ("capacity", second)):
        for key, row in rows.items():
            diagnostic = paired_diagnostics(row["paired_differences"], len(rows), row["practical_margin"])
            results.append({"family": family, "contrast_id": key,
                            "setting_id": row["setting_id"], "control_feature": row.get("control_feature"),
                            "landmarks": int(row["landmarks"]) if family == "capacity" else None,
                            **diagnostic})
    return results, archive.sources


def _load_study_modules(run_directory):
    """Use the immutable local scientific code only after comparing run copies."""
    study = ROOT / "study"
    names = ("geometry.py", "datasets.py", "run_study.py", "followup_capacity.py", "config.json", "followup_config.json")
    for work in ("stage1_work", "capacity_work"):
        for name in names:
            if digest(study / name) != digest(run_directory / work / name):
                raise ValueError(f"Stored run source differs from the scientific snapshot: {work}/{name}")
    if str(study) not in sys.path:
        sys.path.insert(0, str(study))
    import geometry
    import datasets
    import run_study
    import followup_capacity
    for module in (geometry, datasets, run_study, followup_capacity):
        if Path(module.__file__).resolve().parent != study.resolve():
            raise ValueError(f"Ambiguous scientific module import: {module.__name__}")
    return geometry, datasets, run_study, followup_capacity


def bank_checks(run_directory):
    manifest = read_json(run_directory / "execution_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("mode") != "full":
        raise ValueError("Bank diagnostics require a completed full computational reproduction")
    geometry, datasets, run_study, followup_capacity = _load_study_modules(run_directory)
    base = read_json(run_directory / "stage1_work/config.json")
    followup = read_json(run_directory / "capacity_work/followup_config.json")
    primary_settings = ("branch_c0_n1024", "branch_c1_n1024") + SYSTEMS
    expected = []
    for phase, settings, directory in (
        ("stage1", primary_settings, "stage1_work/results/confirmation"),
        ("capacity", SYSTEMS, "capacity_work/results_capacity"),
    ):
        for setting in settings:
            for rep in range(24):
                expected.append((phase, setting, rep, run_directory / directory / f"r{rep:02d}_{setting}"))
    results = []
    for index, (phase, setting, rep, directory) in enumerate(expected, 1):
        result = read_json(directory / "result.json")
        if result.get("status") != "complete" or result.get("replicate") != rep:
            raise ValueError(f"Missing or mismatched completed bank result: {directory}")
        bank_path = directory / "bank.pkl"
        bank = geometry.GeometryBank.load(bank_path)
        models = [model for block in bank.blocks_ for model in block["models"]]
        bases, centers = [m.basis for m in models], [m.center for m in models]
        # Compare numerical bank contents with the human-readable result metadata.
        stored = result["geometry"]
        if bank.metadata()["blocks"] != stored["blocks"]:
            raise ValueError(f"Stored basis/reference metadata do not match the bank: {directory}")
        if phase == "stage1":
            test = run_study.make_splits(base, "confirmation", rep, result["task"])[0][2]
        else:
            seed = followup_capacity.child_seed(followup, rep, result["system"], 2)
            if seed != result["seeds"]["test"]:
                raise ValueError(f"Capacity test-seed mismatch: {directory}")
            fc = base["forecast"]
            test = datasets.make_forecast_split(result["system"], fc["test_episodes"], seed,
                                                history=fc["history"], horizons=tuple(fc["horizons"]), stride=fc["stride"])
        test_hashes = {key: run_study.array_hash(test[array])
                       for key, array in (("X_sha256", "X"), ("Y_sha256", "Y"), ("group_sha256", "groups"))}
        if any(value != result["split_metadata"][2][key] for key, value in test_hashes.items()):
            raise ValueError(f"Replayed test inputs/targets/groups differ from stored hashes: {directory}")
        summary = stacked_basis_diagnostics(bases)
        features = bank.transform(test["X"])
        gates = {}
        for name in ("kahm", "affine"):
            phi = features[f"{name}:phi"]
            gate_summary = {"minimum": float(phi.min()), "maximum": float(phi.max()),
                            "zero_count": int(np.count_nonzero(phi == 0)), "values_checked": int(phi.size),
                            "minimum_by_local_model": phi.min(axis=0).tolist(),
                            "maximum_by_local_model": phi.max(axis=0).tolist()}
            if summary["full_numerical_column_rank"] and np.all(phi > 0):
                recovered = recover_from_gated_coordinates(phi, features[f"{name}:local"][:, len(models):], bases, centers)
                error = recovered - bank._scale_inputs(test["X"])
                gate_summary["max_scaled_input_recovery_l2_error"] = float(np.linalg.norm(error, axis=1).max())
            gates[name] = gate_summary
        results.append({"phase": phase, "setting_id": setting, "replicate": rep,
                        "bank_path": str(bank_path.relative_to(run_directory)), "bank_sha256": digest(bank_path),
                        "result_sha256": digest(directory / "result.json"),
                        "test_hashes_verified": test_hashes, "test_rows": len(test["X"]),
                        "local_models": len(models), **summary, "test_gate_ranges": gates})
        if index % 24 == 0:
            print(f"Described {index}/{len(expected)} stored banks; no fitting", flush=True)
    return results


def aggregate_banks(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["setting_id"])].append(row)
    output = []
    for (phase, setting), group in groups.items():
        if sorted(r["replicate"] for r in group) != list(range(24)):
            raise ValueError(f"Incomplete bank diagnostic group: {phase}/{setting}")
        full = [r for r in group if r["full_numerical_column_rank"]]
        gate_summary = {}
        for name in ("kahm", "affine"):
            gate_summary[name] = {
                "minimum": min(r["test_gate_ranges"][name]["minimum"] for r in group),
                "maximum": max(r["test_gate_ranges"][name]["maximum"] for r in group),
                "zero_count": sum(r["test_gate_ranges"][name]["zero_count"] for r in group),
                "values_checked": sum(r["test_gate_ranges"][name]["values_checked"] for r in group),
            }
        output.append({"phase": phase, "setting_id": setting, "banks": len(group),
                       "ambient_dimension": group[0]["ambient_dimension"],
                       "rank_min": min(r["numerical_rank"] for r in group),
                       "rank_max": max(r["numerical_rank"] for r in group),
                       "full_column_rank_count": len(full),
                       "sigma_min_range": [min(r["sigma_min"] for r in group), max(r["sigma_min"] for r in group)],
                       "maximum_condition_number_full_rank_only": max((r["condition_number_2"] for r in full), default=None),
                       "test_gate_ranges": gate_summary})
    return output


def _scientific(value):
    if value is None:
        return "---"
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def _vdp_rows(rows):
    return [row for row in rows if row["family"] == "capacity" and row["setting_id"] == "forecast_van_der_pol"]


def write_assets(output, payload):
    for directory in (output, output / "tables", output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    write_json(output / "review_diagnostics.json", payload)
    rows = _vdp_rows(payload["paired_contrasts"])
    lines = [r"% Retrospective diagnostics; generated by scripts/review_diagnostics.py.",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Control & $M$ & Skewness & Original lower & Min. deletion lower & $D_r>\delta$ \\", r"\midrule"]
    for row in rows:
        label = "Raw" if row["control_feature"] == "raw" else "Affine local"
        lines.append(f"{label} & {row['landmarks']} & {row['bias_corrected_sample_skewness']:.2f} & "
                     f"{1e3*row['original']['lower']:.4f} & {1e3*row['leave_one_out_min_lower']:.4f} & "
                     f"{row['repetitions_above_margin']}/{row['original']['n']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "tables/review_paired_diagnostics.tex").write_text("\n".join(lines) + "\n")
    snippet = [r"% Retrospective supporting checks; no new primary confidence claim.",
               r"\subsection{Retrospective influence and coordinate-recovery diagnostics}\label{app:review-diagnostics}",
               "These descriptive checks were added after the original study. They reuse the declared",
               "paired repetitions and do not change the primary estimates, margins, or comparison",
               "families. For each of the 41 primary contrasts, we compute bias-corrected sample",
               "skewness and delete each repetition once. Each deletion recomputes the Student",
               "interval with 23 remaining repetitions, 22 degrees of freedom, and the original",
               "family size (29 or 12). The range of deletion endpoints is an influence diagnostic,",
               "not a confidence interval or an additional simultaneous-coverage claim.",
               r"Table~\ref{tab:review-paired} shows the four Van der Pol comparisons; the complete",
               "descriptive results accompany the executable artifact."]
    if all(row["every_leave_one_out_lower_above_margin"] for row in rows):
        snippet += ["All four Van der Pol comparisons retain lower endpoints above the fixed practical",
                    "margin after any one-repetition deletion. This does not establish a distribution-free",
                    "mean guarantee or remove the normal-approximation qualification of the primary analysis."]
    snippet += [r"\begin{table}[htbp]",
                r"\caption{Retrospective Van der Pol influence diagnostics. Lower endpoints are multiplied",
                r"by $10^3$; the fixed margin on this scale is $0.148907$. Skewness is the bias-corrected",
                r"Fisher--Pearson sample coefficient. The minimum deletion lower endpoint is the smallest",
                r"over all 24 one-repetition deletions. The final column counts individual paired",
                r"differences exceeding the margin; it is not a mean-effect confidence statement.}",
                r"\label{tab:review-paired}",
                r"\centering\small\setlength{\tabcolsep}{4pt}",
                r"\input{generated/tables/review_paired_diagnostics}", r"\end{table}",
                r"\begin{figure}[htbp]", r"\centering",
                r"\includegraphics[width=.96\textwidth]{generated/figures/review_paired_differences.pdf}",
                r"\caption{The 24 original paired differences for each Van der Pol capacity comparison,",
                r"in units of $10^{-3}$ MSE. The dashed line is the frozen practical margin.",
                r"No resampling, refitting, or new primary analysis is performed.}",
                r"\label{fig:review-paired}", r"\end{figure}"]
    if payload["bank_groups"]:
        lines = [r"% Full column rank is sufficient; numerical ranks are diagnostics, not exact proofs.",
                 r"\begin{tabular}{llrrrr}", r"\toprule",
                 r"Phase / setting & Rank / $d$ & Min. $\sigma_d$ & Max. $\kappa_2$ & Min. $\phi^{\K}$ & Min. $\phi^{\A}$ \\", r"\midrule"]
        for group in payload["bank_groups"]:
            setting = SYSTEM_LABELS.get(group["setting_id"], "Straight branch" if "c0" in group["setting_id"] else "Curved branch")
            phase = "First" if group["phase"] == "stage1" else "Capacity"
            rank = str(group["rank_min"]) if group["rank_min"] == group["rank_max"] else f"{group['rank_min']}--{group['rank_max']}"
            condition = group["maximum_condition_number_full_rank_only"] if group["full_column_rank_count"] == group["banks"] else None
            values = [f"{phase}: {setting}", f"{rank}/{group['ambient_dimension']}",
                      _scientific(group["sigma_min_range"][0]), _scientific(condition),
                      _scientific(group["test_gate_ranges"]["kahm"]["minimum"]),
                      _scientific(group["test_gate_ranges"]["affine"]["minimum"])]
            lines.append(" & ".join(values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (output / "tables/review_bank_diagnostics.tex").write_text("\n".join(lines) + "\n")
        snippet += ["", "We also inspect the 120 first-stage primary banks and 72 capacity banks from a",
                    "completed computational reproduction using the published seeds. No bank or head is",
                    "refitted for these diagnostics. Replayed test inputs, targets, and groups are checked",
                    "against their recorded hashes before the original feature map is evaluated.",
                    r"Let $A$ vertically stack the $P_c^\top$ matrices. Proposition~\ref{prop:local-injectivity}",
                    r"shows that positive gates and full column rank of $A$ suffice to recover the scaled",
                    r"input from the gated local coordinates and gates:",
                    r"division by $\phi_c$ recovers $P_c^\top(z-\mu_c)$, after which one solves for $z$.",
                    "A deficient stack does not establish information loss in the full code, because the",
                    "gates can carry additional information. Numerical ranks use the threshold",
                    r"$\sigma_1\max\{\text{rows}(A),d\}\epsilon_{64}$; they are finite-precision diagnostics,",
                    "not exact rank proofs. Small singular values and small gates can make coordinate",
                    "recovery sensitive to perturbations. Test-sample gate minima are not population",
                    "lower bounds, and these computations do not guarantee a well-conditioned inverse",
                    "or easy decoding by any particular fitted head.",
                    r"\begin{table}[htbp]",
                    r"\caption{Retrospective diagnostics over 24 stored banks per row. Rank ranges concern",
                    r"the stacked local bases in input dimension $d$. The smallest singular value and",
                    "gate entries are minima over the inspected banks (and, for gates, their original",
                    r"test observations); $\kappa_2$ is the largest stack condition number. A dash means",
                    "that not every stack in that row has full numerical column rank. KAHM and affine",
                    "representations share the basis stack. Computational reproduction supplies no new",
                    "independent statistical replications.}",
                    r"\label{tab:review-banks}",
                    r"\centering\small\setlength{\tabcolsep}{3pt}",
                    r"\input{generated/tables/review_bank_diagnostics}", r"\end{table}"]
    (output / "review_diagnostics.tex").write_text("\n".join(snippet) + "\n")
    plot_differences(output, rows)


def plot_differences(output, rows):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.7), sharey=True, layout="constrained")
    for axis, row in zip(axes.flat, rows):
        values = 1e3 * np.asarray(row["paired_differences"])
        axis.scatter(values, np.arange(1, len(values) + 1), s=18, color="#156c91")
        axis.axvline(1e3 * row["practical_margin"], color="#8b4b00", ls="--", lw=1.3)
        low, high = min(values.min(), 1e3 * row["practical_margin"]), values.max()
        pad = max((high - low) * 0.07, 0.005)
        axis.set_xlim(low - pad, high + pad)
        comparator = "Raw" if row["control_feature"] == "raw" else "Affine local"
        axis.set_title(f"{comparator} minus KAHM local, M={row['landmarks']}", fontsize=10)
        axis.set_xlabel(r"Paired MSE difference $\times 10^3$", fontsize=9)
        axis.set_yticks([1, 8, 16, 24])
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Original repetition")
    axes[1, 0].set_ylabel("Original repetition")
    fig.savefig(output / "figures/review_paired_differences.pdf", metadata={"Title": "Retrospective paired-difference diagnostics", "Author": ""})
    fig.savefig(output / "figures/review_paired_differences.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Completed full run; includes stored-bank diagnostics")
    parser.add_argument("--input", type=Path, help="Paired-analysis inputs; defaults to archive or RUN/results")
    parser.add_argument("--output", type=Path, required=True, help="Only review_* files are created here")
    args = parser.parse_args()
    output = args.output.resolve()
    for protected in (ROOT / "study", ROOT / "archive"):
        if output == protected or protected in output.parents:
            parser.error("Diagnostic output may not alter the scientific snapshot or historical archive")
    archive_validation = verify_archive()
    run = args.run_dir.resolve() if args.run_dir else None
    inputs = args.input.resolve() if args.input else (run / "results" if run else ROOT / "archive")
    rows, sources = contrast_checks(inputs)
    banks = bank_checks(run) if run else []
    payload = {
        "format": "kahm_retrospective_review_diagnostics_v1",
        "scope": "Post-hoc description of existing evidence. No refitting, new primary family, new confidence guarantee, or additional independent replications.",
        "deletion_interpretation": "Each one-repetition deletion retains its original family size and uses n-1 observations with n-2 degrees of freedom. The endpoint envelope is not a confidence interval.",
        "rank_interpretation": "Full numerical column rank with positive gates is a sufficient coordinate-recovery diagnostic. Deficient rank does not establish information loss of the complete code. Neither finite-sample gate minima nor a computed numerical rank certify population stability.",
        "script_sha256": digest(Path(__file__)), "historical_integrity": archive_validation,
        "source_mode": "completed computational reproduction" if run else "historical archive",
        "source_files": sources,
        "run_manifest_sha256": digest(run / "execution_manifest.json") if run else None,
        "paired_contrasts": rows, "banks": banks, "bank_groups": aggregate_banks(banks),
        "bank_scope": "First-stage confirmation: 3 forecasting laws and c=0/c=1 branches at n=1024; capacity: 3 forecasting laws; 24 banks each. Excludes pilots, n=256 secondary branches, and Airfoil.",
    }
    with tempfile.TemporaryDirectory(prefix="kahm-review-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        write_assets(output, payload)
    print(json.dumps({"status": "ok", "paired_contrasts": len(rows), "stored_banks": len(banks),
                      "new_fits": 0, "output": str(output)}))


if __name__ == "__main__":
    main()
