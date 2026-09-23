#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reproduction" / "generated" / "selection_table"

DUFFING_RAW = ROOT / "kahkm_exp_duffing_tuning" / "experiment_duffing_tuning_raw.csv"
VDP_RAW = ROOT / "kahkm_exp09_vanderpol_tuning" / "experiment_09_multistep_raw_results.csv"

def run_generator(script: str, output: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / script),
        "--no-run",
        "--mean-std-mode",
        "horizon-mean",
        "--table-output-dir",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def observed_seeds(path: Path) -> list[int]:
    rows = read_csv(path)
    values = set()

    for row in rows:
        value = row.get("seed")
        if value not in (None, ""):
            values.add(int(float(value)))

    return sorted(values)

def latex_value(value: str) -> str:
    match = re.fullmatch(
        r"\(([^ ]+) \+/- ([^)]+)\) x 10\^(-?\d+)",
        value.strip(),
    )

    if match is None:
        return value

    mean, std, exponent = match.groups()
    return rf"$({mean}\pm{std})\times10^{{{exponent}}}$"

def write_tex(path: Path, duffing: list[dict[str, str]], vdp: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{rrllll}",
        r"\toprule",
        r"$C$ & $\omega$ & $\overline E$ & $E_{50}$ & $E_{100}$ & $E_{200}$ \\",
        r"\midrule",
        r"\multicolumn{6}{l}{Duffing}\\",
    ]

    for row in duffing:
        lines.append(
            f"{row['C']} & {row['omega']} & "
            f"{latex_value(row['mean_Eh_formatted'])} & "
            f"{latex_value(row['E50_formatted'])} & "
            f"{latex_value(row['E100_formatted'])} & "
            f"{latex_value(row['E200_formatted'])} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{Van der Pol}\\",
    ])

    for row in vdp:
        lines.append(
            f"{row['C']} & {row['omega']} & "
            f"{latex_value(row['mean_Eh_formatted'])} & "
            f"{latex_value(row['E50_formatted'])} & "
            f"{latex_value(row['E100_formatted'])} & "
            f"{latex_value(row['E200_formatted'])} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    duffing_seeds = observed_seeds(DUFFING_RAW)
    vdp_seeds = observed_seeds(VDP_RAW)

    if duffing_seeds != [0, 1, 2]:
        raise RuntimeError(f"Unexpected Duffing seeds: {duffing_seeds}")

    if vdp_seeds != [0, 1, 2]:
        raise RuntimeError(f"Unexpected Van der Pol seeds: {vdp_seeds}")

    with tempfile.TemporaryDirectory(prefix="kahkm_selection_") as temp:
        temp = Path(temp)
        duffing_dir = temp / "duffing"
        vdp_dir = temp / "vanderpol"

        run_generator(
            "step_02_generate_table3_duffing.py",
            duffing_dir,
        )

        run_generator(
            "step_03_generate_table4_vanderpol.py",
            vdp_dir,
        )

        duffing = read_csv(
            duffing_dir / "table3_duffing_values.csv"
        )

        vdp = read_csv(
            vdp_dir / "table4_vanderpol_values.csv"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    combined_path = OUTPUT / "selection_table_values.csv"

    fieldnames = [
        "system",
        "C",
        "omega",
        "n_runs",
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
    ]

    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for system, rows in (
            ("Duffing", duffing),
            ("Van der Pol", vdp),
        ):
            for row in rows:
                writer.writerow({
                    "system": system,
                    **{key: row[key] for key in fieldnames if key != "system"},
                })

    tex_path = OUTPUT / "selection_table_values.tex"
    write_tex(tex_path, duffing, vdp)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    metadata = {
        "result_id": "R04",
        "description": "Expanded clean-data selection diagnostics",
        "aggregation_mode": "horizon-mean",
        "seeds": [0, 1, 2],
        "horizons": [1, 10, 50, 100, 200],
        "mean_definition": (
            "For each configuration, overline_E is the arithmetic mean "
            "of the five horizon-specific seed means."
        ),
        "uncertainty_definition": (
            "At each horizon, uncertainty is the archived standard deviation "
            "across seeds 0,1,2. For overline_E, the displayed uncertainty is "
            "the arithmetic mean of those five horizon-specific standard deviations."
        ),
        "duffing_raw": str(DUFFING_RAW.relative_to(ROOT)),
        "vanderpol_raw": str(VDP_RAW.relative_to(ROOT)),
        "duffing_generator": "step_02_generate_table3_duffing.py",
        "vanderpol_generator": "step_03_generate_table4_vanderpol.py",
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

if __name__ == "__main__":
    main()
