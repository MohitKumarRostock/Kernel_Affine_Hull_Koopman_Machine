#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reproduction" / "generated" / "selection_table"

DUFFING_RAW = (
    ROOT
    / "kahkm_exp_duffing_tuning"
    / "experiment_duffing_tuning_raw.csv"
)

VDP_EXPERIMENT_OUTPUT = ROOT / "kahkm_exp09_retvar_fullgrid_selection"
VDP_RAW = VDP_EXPERIMENT_OUTPUT / "experiment_09_multistep_raw_results.csv"

EXPECTED_VDP_SELECTED = (25, 6.0)
EXPECTED_VDP_FINALISTS = {
    (25, 6.0),
    (30, 6.0),
    (20, 4.0),
}


def run_duffing_generator(output: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "step_02_generate_table3_duffing.py"),
        "--no-run",
        "--mean-std-mode",
        "horizon-mean",
        "--table-output-dir",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def run_vanderpol_generator(output: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "step_03_generate_table4_vanderpol.py"),
        "--no-run",
        "--experiment-output-dir",
        str(VDP_EXPERIMENT_OUTPUT),
        "--table-output-dir",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def observed_seeds(path: Path) -> list[int]:
    rows = read_csv(path)
    values: set[int] = set()

    for row in rows:
        value = row.get("seed")
        if value not in (None, ""):
            values.add(int(float(value)))

    return sorted(values)


def finite_float(row: dict[str, str], key: str, *, context: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"Missing {key!r} in {context}.")
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Non-numeric {key!r}={value!r} in {context}."
        ) from exc
    return result


def finite_int(row: dict[str, str], key: str, *, context: str) -> int:
    value = finite_float(row, key, context=context)
    result = int(round(value))
    if abs(value - result) > 1e-12:
        raise RuntimeError(
            f"Expected integer-valued {key!r} in {context}, got {value!r}."
        )
    return result


def latex_value(value: str) -> str:
    match = re.fullmatch(
        r"\(([^ ]+) \+/- ([^)]+)\) x 10\^(-?\d+)",
        value.strip(),
    )
    if match is None:
        return value

    mean, std, exponent = match.groups()
    return rf"$({mean}\pm{std})\times10^{{{exponent}}}$"


def validate_vanderpol_rows(
    rows: list[dict[str, str]],
) -> tuple[dict[str, str], float, float, float]:
    if len(rows) != len(EXPECTED_VDP_FINALISTS):
        raise RuntimeError(
            "Expected exactly three Van der Pol one-SE finalists, "
            f"found {len(rows)}."
        )

    observed_finalists = {
        (
            finite_int(row, "C", context="Van der Pol Table 4 row"),
            finite_float(row, "omega", context="Van der Pol Table 4 row"),
        )
        for row in rows
    }
    if observed_finalists != EXPECTED_VDP_FINALISTS:
        raise RuntimeError(
            "Unexpected Van der Pol one-SE finalist set: "
            f"{sorted(observed_finalists)}"
        )

    selected = [
        row
        for row in rows
        if finite_int(
            row,
            "selected_by_centered_retvar_rule",
            context="Van der Pol Table 4 row",
        )
        == 1
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one selected Van der Pol row, found {len(selected)}."
        )

    selected_row = selected[0]
    selected_key = (
        finite_int(
            selected_row,
            "C",
            context="selected Van der Pol Table 4 row",
        ),
        finite_float(
            selected_row,
            "omega",
            context="selected Van der Pol Table 4 row",
        ),
    )
    if selected_key != EXPECTED_VDP_SELECTED:
        raise RuntimeError(
            f"Expected selected Van der Pol configuration "
            f"{EXPECTED_VDP_SELECTED}, found {selected_key}."
        )

    best_mean = finite_float(
        selected_row,
        "centered_best_mean_error",
        context="selected Van der Pol Table 4 row",
    )
    best_se = finite_float(
        selected_row,
        "centered_best_standard_error",
        context="selected Van der Pol Table 4 row",
    )
    one_se_limit = finite_float(
        selected_row,
        "centered_one_se_limit",
        context="selected Van der Pol Table 4 row",
    )

    if abs((best_mean + best_se) - one_se_limit) > 1e-12:
        raise RuntimeError(
            "Van der Pol one-SE limit does not equal best mean error + best SE."
        )

    for row in rows:
        inside = finite_int(
            row,
            "inside_centered_one_se_set",
            context="Van der Pol Table 4 row",
        )
        if inside != 1:
            raise RuntimeError(
                "Table 4 manuscript rows must all be centered one-SE finalists."
            )

    return selected_row, best_mean, best_se, one_se_limit


def write_combined_csv(
    path: Path,
    duffing: list[dict[str, str]],
    vdp: list[dict[str, str]],
) -> None:
    fieldnames = [
        "system",
        "selection_rule",
        "rank",
        "C",
        "omega",
        "n_runs",
        # Duffing-specific clean-data relative-error diagnostics.
        "mean_Eh_mean",
        "mean_Eh_std",
        "E50_mean",
        "E50_std",
        "E100_mean",
        "E100_std",
        "E200_mean",
        "E200_std",
        "mean_Eh_formatted",
        "E50_formatted",
        "E100_formatted",
        "E200_formatted",
        # Van der Pol centered/retained-variation diagnostics.
        "mean_multihorizon_R2",
        "mean_centered_error",
        "centered_error_sd",
        "centered_error_se",
        "rho_var_mean",
        "rho_var_std",
        "effective_rank_mean",
        "effective_rank_std",
        "rho_rank_mean",
        "rho_rank_std",
        "inside_centered_one_se_set",
        "selected_by_centered_retvar_rule",
        "retained_variation_rank_within_one_se",
        "centered_best_mean_error",
        "centered_best_standard_error",
        "centered_one_se_limit",
        "mean_multihorizon_R2_table_cell",
        "centered_error_se_table_cell",
        "rho_var_table_cell",
        "effective_rank_table_cell",
        "rho_rank_table_cell",
        "selected_table_cell",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in duffing:
            writer.writerow(
                {
                    "system": "Duffing",
                    "selection_rule": (
                        "rank by mean relative association error over "
                        "h={1,10,50,100,200}"
                    ),
                    "rank": row["rank_ref"],
                    "C": row["C"],
                    "omega": row["omega"],
                    "n_runs": row["n_runs"],
                    "mean_Eh_mean": row["mean_Eh_mean"],
                    "mean_Eh_std": row["mean_Eh_std"],
                    "E50_mean": row["E50_mean"],
                    "E50_std": row["E50_std"],
                    "E100_mean": row["E100_mean"],
                    "E100_std": row["E100_std"],
                    "E200_mean": row["E200_mean"],
                    "E200_std": row["E200_std"],
                    "mean_Eh_formatted": row["mean_Eh_formatted"],
                    "E50_formatted": row["E50_formatted"],
                    "E100_formatted": row["E100_formatted"],
                    "E200_formatted": row["E200_formatted"],
                }
            )

        for row in vdp:
            writer.writerow(
                {
                    "system": "Van der Pol",
                    "selection_rule": (
                        "rank by mean_h(1-R_h^2); one-SE set; choose largest "
                        "rho_var with rho_rank secondary"
                    ),
                    "rank": row["centered_rank"],
                    "C": row["C"],
                    "omega": row["omega"],
                    "n_runs": row["n_runs"],
                    "mean_multihorizon_R2": row["mean_multihorizon_R2"],
                    "mean_centered_error": row["mean_centered_error"],
                    "centered_error_sd": row["centered_error_sd"],
                    "centered_error_se": row["centered_error_se"],
                    "rho_var_mean": row["rho_var_mean"],
                    "rho_var_std": row["rho_var_std"],
                    "effective_rank_mean": row["effective_rank_mean"],
                    "effective_rank_std": row["effective_rank_std"],
                    "rho_rank_mean": row["rho_rank_mean"],
                    "rho_rank_std": row["rho_rank_std"],
                    "inside_centered_one_se_set": row[
                        "inside_centered_one_se_set"
                    ],
                    "selected_by_centered_retvar_rule": row[
                        "selected_by_centered_retvar_rule"
                    ],
                    "retained_variation_rank_within_one_se": row[
                        "retained_variation_rank_within_one_se"
                    ],
                    "centered_best_mean_error": row[
                        "centered_best_mean_error"
                    ],
                    "centered_best_standard_error": row[
                        "centered_best_standard_error"
                    ],
                    "centered_one_se_limit": row["centered_one_se_limit"],
                    "mean_multihorizon_R2_table_cell": row[
                        "mean_multihorizon_R2_table_cell"
                    ],
                    "centered_error_se_table_cell": row[
                        "centered_error_se_table_cell"
                    ],
                    "rho_var_table_cell": row["rho_var_table_cell"],
                    "effective_rank_table_cell": row[
                        "effective_rank_table_cell"
                    ],
                    "rho_rank_table_cell": row["rho_rank_table_cell"],
                    "selected_table_cell": row["selected_table_cell"],
                }
            )


def write_tex(
    path: Path,
    duffing: list[dict[str, str]],
    vdp: list[dict[str, str]],
) -> None:
    lines = [
        r"\begin{tabular}{rrllll}",
        r"\toprule",
        r"\multicolumn{6}{l}{\textbf{Duffing: relative-association-error ranking}}\\",
        r"$C$ & $\omega$ & $\overline E$ & $E_{50}$ & $E_{100}$ & $E_{200}$ \\",
        r"\midrule",
    ]

    for row in duffing:
        lines.append(
            f"{row['C']} & {row['omega']} & "
            f"{latex_value(row['mean_Eh_formatted'])} & "
            f"{latex_value(row['E50_formatted'])} & "
            f"{latex_value(row['E100_formatted'])} & "
            f"{latex_value(row['E200_formatted'])} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{0.75em}",
            "",
            r"\begin{tabular}{rrrllllll}",
            r"\toprule",
            r"\multicolumn{9}{l}{\textbf{Van der Pol: centered one-SE/retained-variation selection}}\\",
            (
                r"Rank & $C$ & $\omega$ & mean $R^2$ & centered error $\pm$ SE & "
                r"$\rho_{\mathrm{var}}$ & $r_{\mathrm{eff}}$ & "
                r"$\rho_{\mathrm{rank}}$ & selected \\"
            ),
            r"\midrule",
        ]
    )

    vdp_sorted = sorted(
        vdp,
        key=lambda row: int(float(row["centered_rank"])),
    )
    for row in vdp_sorted:
        lines.append(
            f"{row['centered_rank']} & {row['C']} & {row['omega']} & "
            f"{row['mean_multihorizon_R2_table_cell']} & "
            f"{row['centered_error_se_table_cell']} & "
            f"{row['rho_var_table_cell']} & "
            f"{row['effective_rank_table_cell']} & "
            f"{row['rho_rank_table_cell']} & "
            f"{row['selected_table_cell']} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    duffing_seeds = observed_seeds(DUFFING_RAW)
    vdp_seeds = observed_seeds(VDP_RAW)

    if duffing_seeds != [0, 1, 2]:
        raise RuntimeError(f"Unexpected Duffing seeds: {duffing_seeds}")

    if vdp_seeds != [0, 1, 2]:
        raise RuntimeError(f"Unexpected Van der Pol seeds: {vdp_seeds}")

    with tempfile.TemporaryDirectory(prefix="kahkm_selection_") as temp_name:
        temp = Path(temp_name)
        duffing_dir = temp / "duffing"
        vdp_dir = temp / "vanderpol"

        run_duffing_generator(duffing_dir)
        run_vanderpol_generator(vdp_dir)

        duffing = read_csv(
            duffing_dir / "table3_duffing_values.csv"
        )
        vdp = read_csv(
            vdp_dir / "table4_vanderpol_values.csv"
        )

    selected_vdp, best_mean, best_se, one_se_limit = validate_vanderpol_rows(
        vdp
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    combined_path = OUTPUT / "selection_table_values.csv"
    write_combined_csv(combined_path, duffing, vdp)

    tex_path = OUTPUT / "selection_table_values.tex"
    write_tex(tex_path, duffing, vdp)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    metadata: dict[str, Any] = {
        "result_id": "R04",
        "description": (
            "Expanded clean-data selection diagnostics with centered "
            "retained-variation Van der Pol selection"
        ),
        "seeds": [0, 1, 2],
        "horizons": [1, 10, 50, 100, 200],
        "panels": {
            "duffing": {
                "selection_rule": (
                    "rank by mean relative association error over "
                    "h={1,10,50,100,200}"
                ),
                "aggregation_mode": "horizon-mean",
                "uncertainty_definition": (
                    "At each horizon, archived standard deviation across seeds "
                    "0,1,2; for overline_E, displayed uncertainty is the "
                    "arithmetic mean of the five horizon-specific standard "
                    "deviations."
                ),
                "raw": str(DUFFING_RAW.relative_to(ROOT)),
                "generator": "step_02_generate_table3_duffing.py",
                "rows": len(duffing),
            },
            "vanderpol": {
                "selection_rule": (
                    "rank configurations by mean centered error "
                    "mean_h(1-R_h^2); form one-SE set using the SE of the "
                    "minimum-error configuration; within the set choose largest "
                    "normalized retained variation rho_var with normalized "
                    "effective rank rho_rank as secondary tie-break"
                ),
                "uncertainty_definition": (
                    "Centered error is mean across seeds 0,1,2 with standard "
                    "error SD/sqrt(3); retained-variation and effective-rank "
                    "diagnostics report archived across-seed mean and SD."
                ),
                "raw": str(VDP_RAW.relative_to(ROOT)),
                "experiment_output_dir": str(
                    VDP_EXPERIMENT_OUTPUT.relative_to(ROOT)
                ),
                "generator": "step_03_generate_table4_vanderpol.py",
                "one_se_finalists": len(vdp),
                "selected_C": finite_int(
                    selected_vdp,
                    "C",
                    context="selected Van der Pol row",
                ),
                "selected_omega": finite_float(
                    selected_vdp,
                    "omega",
                    context="selected Van der Pol row",
                ),
                "centered_best_mean_error": best_mean,
                "centered_best_standard_error": best_se,
                "centered_one_se_limit": one_se_limit,
            },
        },
        "git_commit": commit,
        "outputs": [
            str(combined_path.relative_to(ROOT)),
            str(tex_path.relative_to(ROOT)),
        ],
    }

    metadata_path = OUTPUT / "selection_table_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {combined_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {metadata_path}")
    print(f"Verified Duffing seeds: {duffing_seeds}")
    print(f"Verified Van der Pol seeds: {vdp_seeds}")
    print(
        "Verified Van der Pol centered one-SE finalists: "
        + ", ".join(
            f"(C={int(float(row['C']))}, omega={float(row['omega']):g})"
            for row in sorted(
                vdp,
                key=lambda row: int(float(row["centered_rank"])),
            )
        )
    )
    print(
        "Verified Van der Pol selected configuration: "
        f"C={int(float(selected_vdp['C']))}, "
        f"omega={float(selected_vdp['omega']):g}"
    )


if __name__ == "__main__":
    main()
