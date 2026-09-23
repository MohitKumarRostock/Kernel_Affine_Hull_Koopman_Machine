#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reproduction" / "generated" / "behavioral_summary"

TABLE12 = ROOT / "kahkm_table12_interpretability" / "table12_interpretability_values.csv"
CP_STATS = ROOT / "kahkm_exp17_cartpole_tuned_seed0" / "experiment_17_regime_statistics.csv"
MC_STATS = ROOT / "kahkm_exp17_mountaincar_tuned_seed0" / "experiment_17_regime_statistics.csv"

ACROBOT_VALUES = (
    ROOT
    / "kahkm_table14_acrobot_interpretability_selected"
    / "table14_acrobot_interpretability_values.csv"
)

ACROBOT_METADATA = (
    ROOT
    / "kahkm_table14_acrobot_interpretability_selected"
    / "table14_acrobot_interpretability_metadata.json"
)

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def finite(value: str | None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        return None

def weighted_regime_summary(
    task: str,
    path: Path,
    C: int,
    omega: float,
) -> dict[str, object]:

    rows = read_csv(path)
    active = [
        row
        for row in rows
        if int(float(row["count"])) > 0
    ]

    total = sum(int(float(row["count"])) for row in active)

    action_columns = sorted({
        key
        for row in active
        for key in row
        if key.startswith("action_")
        and key.endswith("_rate")
    })

    def purity(row: dict[str, str]) -> float:
        values = [
            finite(row.get(column))
            for column in action_columns
        ]
        values = [
            value
            for value in values
            if value is not None
        ]
        return max(values)

    action_purity = (
        sum(
            int(float(row["count"])) * purity(row)
            for row in active
        )
        / total
    )

    high_purity_mass = (
        sum(
            int(float(row["count"]))
            for row in active
            if purity(row) >= 0.9
        )
        / total
    )

    soft_num = 0.0
    soft_den = 0
    hard_num = 0.0
    hard_den = 0

    for row in active:
        count = int(float(row["count"]))

        soft = finite(row.get("learned_persistence"))
        if soft is not None:
            soft_num += count * soft
            soft_den += count

        hard = finite(row.get("empirical_persistence"))
        if hard is not None:
            hard_num += count * hard
            hard_den += count

    return {
        "task": task,
        "C": C,
        "omega": omega,
        "n_fits": 1,
        "active_mean": float(len(active)),
        "active_std": "",
        "active_denominator": len(rows),
        "action_purity_mean": action_purity,
        "action_purity_std": "",
        "high_purity_mass_mean": high_purity_mass,
        "high_purity_mass_std": "",
        "soft_persistence_mean": soft_num / soft_den,
        "soft_persistence_std": "",
        "hard_persistence_mean": hard_num / hard_den,
        "hard_persistence_std": "",
        "aggregation": "single diagnostic fit, seed 0",
    }

def acrobot_summary() -> dict[str, object]:

    rows = {
        row["statistic"]: row
        for row in read_csv(ACROBOT_VALUES)
    }

    metadata = json.loads(
        ACROBOT_METADATA.read_text(encoding="utf-8")
    )

    active = rows["Active regimes"]
    purity = rows["Count-weighted action purity"]

    high_key = next(
        key
        for key in rows
        if key.startswith("Mass in regimes with action purity")
    )

    high = rows[high_key]

    return {
        "task": "Acrobot-v1",
        "C": int(metadata["selected_n_clusters"]),
        "omega": float(metadata["selected_omega"]),
        "n_fits": int(active["n"]),
        "active_mean": float(active["mean"]),
        "active_std": float(active["std"]),
        "active_denominator": int(metadata["selected_n_clusters"]),
        "action_purity_mean": float(purity["mean"]),
        "action_purity_std": float(purity["std"]),
        "high_purity_mass_mean": float(high["mean"]),
        "high_purity_mass_std": float(high["std"]),
        "soft_persistence_mean": "",
        "soft_persistence_std": "",
        "hard_persistence_mean": "",
        "hard_persistence_std": "",
        "aggregation": "population mean and SD over seeds 0,1,2",
    }

def fmt_single(value: float) -> str:
    return f"{value:.3f}"

def fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"

def write_latex(
    path: Path,
    rows: list[dict[str, object]],
) -> None:

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Task & Active & Action purity & Mass purity $\geq 0.9$ & Soft pers. & Hard pers. \\",
        r"\midrule",
    ]

    for row in rows:

        if int(row["n_fits"]) == 1:
            active = (
                f"{int(float(row['active_mean']))}/"
                f"{int(row['active_denominator'])}"
            )

            purity = fmt_single(
                float(row["action_purity_mean"])
            )

            high = fmt_single(
                float(row["high_purity_mass_mean"])
            )

            soft = fmt_single(
                float(row["soft_persistence_mean"])
            )

            hard = fmt_single(
                float(row["hard_persistence_mean"])
            )

        else:
            active = (
                rf"${float(row['active_mean']):.1f}"
                rf"\pm{float(row['active_std']):.1f}"
                rf"/{int(row['active_denominator'])}$"
            )

            purity = (
                rf"${float(row['action_purity_mean']):.4f}"
                rf"\pm{float(row['action_purity_std']):.4f}$"
            )

            high = (
                rf"${float(row['high_purity_mass_mean']):.4f}"
                rf"\pm{float(row['high_purity_mass_std']):.4f}$"
            )

            soft = r"--"
            hard = r"--"

        lines.append(
            f"{row['task']} & {active} & {purity} & "
            f"{high} & {soft} & {hard} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

def main() -> None:

    task_rows = {
        row["task"]: row
        for row in read_csv(TABLE12)
    }

    cp = weighted_regime_summary(
        "CartPole-v1",
        CP_STATS,
        int(task_rows["CartPole-v1"]["C"]),
        float(task_rows["CartPole-v1"]["omega"]),
    )

    mc = weighted_regime_summary(
        "MountainCar-v0",
        MC_STATS,
        int(task_rows["MountainCar-v0"]["C"]),
        float(task_rows["MountainCar-v0"]["omega"]),
    )

    acrobot = acrobot_summary()

    rows = [cp, mc, acrobot]

    OUTPUT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT / "behavioral_summary_values.csv"

    fieldnames = list(rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    tex_path = OUTPUT / "behavioral_summary_values.tex"
    write_latex(tex_path, rows)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    metadata = {
        "result_id": "R17",
        "description": "Dominant-regime behavioral summaries",
        "cartpole_source": str(
            CP_STATS.relative_to(ROOT)
        ),
        "mountaincar_source": str(
            MC_STATS.relative_to(ROOT)
        ),
        "acrobot_source": str(
            ACROBOT_VALUES.relative_to(ROOT)
        ),
        "definitions": {
            "action_purity": (
                "count-weighted maximum empirical action "
                "frequency within each active regime"
            ),
            "high_purity_mass": (
                "fraction of samples assigned to active "
                "regimes with action purity >= 0.9"
            ),
            "soft_persistence": (
                "count-weighted learned self-transition "
                "probability"
            ),
            "hard_persistence": (
                "count-weighted empirical persistence"
            ),
        },
        "aggregation": {
            "CartPole-v1": "single diagnostic fit, seed 0",
            "MountainCar-v0": "single diagnostic fit, seed 0",
            "Acrobot-v1": (
                "population mean and standard deviation "
                "over validated seeds 0,1,2"
            ),
        },
        "git_commit": commit,
        "outputs": [
            str(csv_path.relative_to(ROOT)),
            str(tex_path.relative_to(ROOT)),
        ],
    }

    metadata_path = (
        OUTPUT / "behavioral_summary_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {metadata_path}")

if __name__ == "__main__":
    main()
