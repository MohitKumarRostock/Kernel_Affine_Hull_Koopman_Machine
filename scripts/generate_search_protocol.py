#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reproduction" / "generated" / "search_protocol"

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def cli_defaults(path: Path, wanted: set[str]) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    result: dict[str, Any] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not node.args:
            continue

        try:
            flag = ast.literal_eval(node.args[0])
        except Exception:
            continue

        if flag not in wanted:
            continue

        default: Any = None

        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    default = ast.literal_eval(kw.value)
                except Exception:
                    default = ast.unparse(kw.value)

        result[str(flag)] = default

    return result

def find_dict_with_keys(
    obj: Any,
    required: set[str],
) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if required.issubset(obj.keys()):
            return obj

        for value in obj.values():
            found = find_dict_with_keys(value, required)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = find_dict_with_keys(value, required)
            if found is not None:
                return found

    return None

def fmt_numbers(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)

def main() -> None:
    duffing_reference_script = (
        ROOT / "experiment_02_duffing_sensitivity.py"
    )
    duffing_expanded_meta = load_json(
        ROOT
        / "kahkm_table3_duffing"
        / "table3_duffing_metadata.json"
    )
    vdp_clean_meta = load_json(
        ROOT
        / "kahkm_table4_vanderpol"
        / "table4_vanderpol_metadata.json"
    )

    noise_meta = load_json(
        ROOT
        / "kahkm_exp12_vanderpol_noise_aware_tuning"
        / "experiment_12_noise_tuning_metadata.json"
    )

    acrobot_script = (
        ROOT
        / "experiment_18_acrobot_sequential_decision_pylance_clean.py"
    )
    acrobot_runner = (
        ROOT
        / "run_exp18_acrobot_tuning_pylance_clean.py"
    )

    acrobot_validation_meta = load_json(
        ROOT
        / "kahkm_table13_acrobot_benchmark_tuned"
        / "table13_acrobot_benchmark_metadata.json"
    )

    reference_defaults = cli_defaults(
        duffing_reference_script,
        {"--clusters", "--omegas", "--seeds"},
    )

    assert reference_defaults["--clusters"] == [15, 25, 35]
    assert reference_defaults["--omegas"] == [4.0, 8.0, 12.0]
    assert reference_defaults["--seeds"] == [0, 1, 2]

    reference_source = duffing_reference_script.read_text(
        encoding="utf-8"
    )
    assert "Best mean test closure setting" in reference_source
    assert "test_closure_error_mean" in reference_source

    noise_config = find_dict_with_keys(
        noise_meta,
        {
            "clusters",
            "omegas",
            "noise_levels",
            "random_states",
            "horizons",
        },
    )

    if noise_config is None:
        raise RuntimeError(
            "Could not locate noise-aware configuration in metadata."
        )

    acrobot_defaults = cli_defaults(
        acrobot_runner,
        {
            "--seeds",
            "--clusters",
            "--omegas",
            "--horizons",
            "--test-seed-base",
        },
    )

    assert acrobot_defaults["--clusters"] == [
        "20", "40", "50", "75", "100", "150"
    ]
    assert acrobot_defaults["--omegas"] == [
        "0.5", "1", "2", "4", "8"
    ]
    assert acrobot_defaults["--horizons"] == [
        "1", "5", "10", "20", "50"
    ]
    assert acrobot_defaults["--seeds"] == ["0"]
    assert acrobot_defaults["--test-seed-base"] == 100

    acrobot_source = acrobot_script.read_text(encoding="utf-8")

    assert "best_horizon = max(args.horizons)" in acrobot_source
    assert 'score = float(selected[0]["relative_error"])' in acrobot_source

    validation_seeds = (
        acrobot_validation_meta["cli_args"]["seeds"]
    )

    assert validation_seeds == [0, 1, 2]

    rows = [
        {
            "study": "Duffing reference",
            "C": [15, 25, 35],
            "omega": [4, 8, 12],
            "grid_count": 9,
            "search_seeds": [0, 1, 2],
            "noise_levels": [],
            "horizons": [1],
            "objective": "E1",
            "objective_detail": (
                "minimize mean test closure error"
            ),
            "scheduled_search_evaluations": 27,
            "completed_search_evaluations": "",
            "validation": "",
            "source": "experiment_02_duffing_sensitivity.py",
        },
        {
            "study": "Duffing expanded",
            "C": duffing_expanded_meta["clusters"],
            "omega": duffing_expanded_meta["omegas"],
            "grid_count": (
                len(duffing_expanded_meta["clusters"])
                * len(duffing_expanded_meta["omegas"])
            ),
            "search_seeds": duffing_expanded_meta["random_states"],
            "noise_levels": [],
            "horizons": duffing_expanded_meta["horizons"],
            "objective": "mean_E_H",
            "objective_detail": (
                duffing_expanded_meta["ranking_objective"]
            ),
            "scheduled_search_evaluations": (
                len(duffing_expanded_meta["clusters"])
                * len(duffing_expanded_meta["omegas"])
                * len(duffing_expanded_meta["random_states"])
            ),
            "completed_search_evaluations": "",
            "validation": "",
            "source": (
                "run_exp_duffing_tuning.py; "
                "kahkm_table3_duffing/table3_duffing_metadata.json"
            ),
        },
        {
            "study": "Van der Pol clean",
            "C": vdp_clean_meta["clusters"],
            "omega": vdp_clean_meta["omegas"],
            "grid_count": (
                len(vdp_clean_meta["clusters"])
                * len(vdp_clean_meta["omegas"])
            ),
            "search_seeds": vdp_clean_meta["random_states"],
            "noise_levels": [],
            "horizons": vdp_clean_meta["horizons"],
            "objective": "mean_E_H",
            "objective_detail": (
                vdp_clean_meta["ranking_objective"]
            ),
            "scheduled_search_evaluations": (
                len(vdp_clean_meta["clusters"])
                * len(vdp_clean_meta["omegas"])
                * len(vdp_clean_meta["random_states"])
            ),
            "completed_search_evaluations": "",
            "validation": "",
            "source": (
                "run_exp09_vanderpol_tuning.py; "
                "kahkm_table4_vanderpol/table4_vanderpol_metadata.json"
            ),
        },
        {
            "study": "Van der Pol noise-aware",
            "C": noise_config["clusters"],
            "omega": noise_config["omegas"],
            "grid_count": (
                len(noise_config["clusters"])
                * len(noise_config["omegas"])
            ),
            "search_seeds": noise_config["random_states"],
            "noise_levels": noise_config["noise_levels"],
            "horizons": noise_config["horizons"],
            "objective": (
                "0.25*E50 + 0.25*E100 + 0.50*E200"
            ),
            "objective_detail": (
                "mean weighted long-horizon score across "
                "search seeds and training-noise levels"
            ),
            "scheduled_search_evaluations": int(
                noise_meta["total_jobs"]
            ),
            "completed_search_evaluations": int(
                noise_meta["completed_jobs"]
            ),
            "validation": (
                "clean held-out evaluation; noise applied "
                "only to training snapshots"
            ),
            "source": (
                "run_exp12_vanderpol_noise_aware_tuning.py; "
                "kahkm_exp12_vanderpol_noise_aware_tuning/"
                "experiment_12_noise_tuning_metadata.json"
            ),
        },
        {
            "study": "Acrobot",
            "C": [
                int(value)
                for value in acrobot_defaults["--clusters"]
            ],
            "omega": [
                float(value)
                for value in acrobot_defaults["--omegas"]
            ],
            "grid_count": (
                len(acrobot_defaults["--clusters"])
                * len(acrobot_defaults["--omegas"])
            ),
            "search_seeds": [0],
            "noise_levels": [],
            "horizons": [
                int(value)
                for value in acrobot_defaults["--horizons"]
            ],
            "objective": "E50",
            "objective_detail": (
                "minimize KAHKM relative error at "
                "max(search horizons)=50"
            ),
            "scheduled_search_evaluations": (
                len(acrobot_defaults["--clusters"])
                * len(acrobot_defaults["--omegas"])
            ),
            "completed_search_evaluations": "",
            "validation": (
                "selected C=150, omega=0.5 configuration "
                "validated separately on seeds 0,1,2"
            ),
            "source": (
                "experiment_18_acrobot_sequential_decision_pylance_clean.py; "
                "run_exp18_acrobot_tuning_pylance_clean.py"
            ),
        },
    ]

    assert rows[0]["grid_count"] == 9
    assert rows[1]["grid_count"] == 64
    assert rows[2]["grid_count"] == 49
    assert rows[3]["grid_count"] == 30
    assert rows[4]["grid_count"] == 30

    assert rows[1]["scheduled_search_evaluations"] == 192
    assert rows[2]["scheduled_search_evaluations"] == 147
    assert rows[3]["scheduled_search_evaluations"] == 450
    assert rows[4]["scheduled_search_evaluations"] == 30

    assert rows[3]["completed_search_evaluations"] == 450

    OUTPUT.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT / "search_protocol.json"
    json_path.write_text(
        json.dumps(
            {
                "result_id": "R23",
                "count_convention": (
                    "grid_count is |C|*|omega| and excludes "
                    "seed and noise-level repetitions"
                ),
                "protocols": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = OUTPUT / "search_protocol.csv"

    fields = [
        "study",
        "C",
        "omega",
        "grid_count",
        "search_seeds",
        "noise_levels",
        "horizons",
        "objective",
        "objective_detail",
        "scheduled_search_evaluations",
        "completed_search_evaluations",
        "validation",
        "source",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            item = dict(row)
            for key in (
                "C",
                "omega",
                "search_seeds",
                "noise_levels",
                "horizons",
            ):
                item[key] = fmt_numbers(item[key])

            writer.writerow(item)

    tex_path = OUTPUT / "search_protocol.tex"

    tex_lines = [
        r"\begin{tabular}{lp{0.25\linewidth}p{0.21\linewidth}rp{0.19\linewidth}}",
        r"\toprule",
        r"Study & $C$ & $\omega$ & Count & Objective \\",
        r"\midrule",
        r"Duffing reference & $15,25,35$ & $4,8,12$ & 9 & $E_1$ \\",
        r"Duffing expanded & $8,10,12,15,20,25,30,40$ & $2,4,6,8,10,12,16,20$ & 64 & $\overline E_{\mathcal H}$ \\",
        r"Van der Pol clean & $10,15,20,25,30,40,50$ & $0.5,1,2,4,6,8,12$ & 49 & $\overline E_{\mathcal H}$ \\",
        r"Van der Pol noise-aware & $10,15,20,25,30,40$ & $0.25,0.5,1,2,4$ & 30 & $\tfrac14E_{50}+\tfrac14E_{100}+\tfrac12E_{200}$ \\",
        r"Acrobot & $20,40,50,75,100,150$ & $0.5,1,2,4,8$ & 30 & $E_{50}$ \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    tex_path.write_text(
        "\n".join(tex_lines) + "\n",
        encoding="utf-8",
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    audit_path = OUTPUT / "search_protocol_audit.json"

    audit_path.write_text(
        json.dumps(
            {
                "git_commit": commit,
                "verified": {
                    "duffing_reference_grid": True,
                    "duffing_reference_objective": (
                        "mean test closure error / manuscript E1"
                    ),
                    "duffing_expanded_grid": True,
                    "vanderpol_clean_grid": True,
                    "vanderpol_noise_grid": True,
                    "noise_aware_jobs": "450/450",
                    "acrobot_grid": True,
                    "acrobot_selection_horizon": 50,
                    "acrobot_search_seed": 0,
                    "acrobot_validation_seeds": [0, 1, 2],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {audit_path}")

if __name__ == "__main__":
    main()
