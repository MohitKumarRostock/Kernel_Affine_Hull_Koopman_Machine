#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reproduction" / "generated" / "search_protocol"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def fmt_numbers(values: Sequence[Any]) -> str:
    return ",".join(str(value) for value in values)


def finite_float(row: dict[str, str], key: str, *, context: str) -> float:
    text = row.get(key, "").strip()
    if not text:
        raise ValueError(
            f"Missing required field {key!r} for {context}. "
            f"Available fields: {sorted(row)}"
        )
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} for {context}: {text!r}")
    return value


def finite_int(row: dict[str, str], key: str, *, context: str) -> int:
    return int(round(finite_float(row, key, context=context)))


def parse_selection_weights(text: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        horizon_text, weight_text = item.split(":", maxsplit=1)
        result[int(horizon_text)] = float(weight_text)
    return result


def assert_close(
    observed: float,
    expected: float,
    label: str,
    *,
    atol: float = 1e-12,
) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(
            f"{label}: observed {observed:.17g}, expected {expected:.17g}"
        )


def reconstruct_noise_centered_ranking(
    raw_path: Path,
    noise_config: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, str], int]:
    """Reconstruct the Experiment 12 centered ranking from the original raw grid.

    The original full-grid Experiment 12 run predates the centered-summary CSVs.
    This function reconstructs exactly the later selection rule from the raw
    association R^2 values without rerunning any KAHKM fits.
    """
    rows = load_csv(raw_path)
    weights = {50: 0.25, 100: 0.25, 200: 0.50}

    expected_configs = {
        (int(c), float(omega))
        for c in noise_config["clusters"]
        for omega in noise_config["omegas"]
    }
    expected_seeds = sorted(int(seed) for seed in noise_config["random_states"])
    expected_noise_levels = sorted(
        float(value) for value in noise_config["noise_levels"]
    )
    expected_horizons = sorted(int(value) for value in noise_config["horizons"])

    observed_configs = {
        (
            finite_int(row, "n_clusters", context="Experiment 12 raw row"),
            finite_float(row, "omega", context="Experiment 12 raw row"),
        )
        for row in rows
    }
    observed_seeds = sorted(
        {
            finite_int(row, "seed", context="Experiment 12 raw row")
            for row in rows
        }
    )
    observed_noise_levels = sorted(
        {
            finite_float(row, "noise_level", context="Experiment 12 raw row")
            for row in rows
        }
    )
    observed_horizons = sorted(
        {
            finite_int(row, "horizon", context="Experiment 12 raw row")
            for row in rows
        }
    )

    if observed_configs != expected_configs:
        raise RuntimeError(
            "Experiment 12 raw configuration grid does not match metadata."
        )
    if observed_seeds != expected_seeds:
        raise RuntimeError(
            f"Experiment 12 raw seeds {observed_seeds} do not match metadata "
            f"{expected_seeds}."
        )
    if observed_noise_levels != expected_noise_levels:
        raise RuntimeError(
            "Experiment 12 raw noise levels do not match metadata."
        )
    if observed_horizons != expected_horizons:
        raise RuntimeError(
            "Experiment 12 raw horizons do not match metadata."
        )

    expected_rows = (
        len(expected_configs)
        * len(expected_seeds)
        * len(expected_noise_levels)
        * len(expected_horizons)
    )
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} Experiment 12 raw horizon rows, "
            f"found {len(rows)}."
        )

    grouped: dict[
        tuple[int, float, int, float],
        dict[int, float],
    ] = defaultdict(dict)

    for row in rows:
        horizon = finite_int(
            row,
            "horizon",
            context="Experiment 12 raw row",
        )
        if horizon not in weights:
            continue

        key = (
            finite_int(
                row,
                "n_clusters",
                context="Experiment 12 raw row",
            ),
            finite_float(
                row,
                "omega",
                context="Experiment 12 raw row",
            ),
            finite_int(
                row,
                "seed",
                context="Experiment 12 raw row",
            ),
            finite_float(
                row,
                "noise_level",
                context="Experiment 12 raw row",
            ),
        )
        if horizon in grouped[key]:
            raise RuntimeError(
                f"Duplicate Experiment 12 raw row for {key}, h={horizon}."
            )
        grouped[key][horizon] = finite_float(
            row,
            "association_r2",
            context="Experiment 12 raw row",
        )

    seed_noise_scores: dict[
        tuple[int, float, int],
        list[float],
    ] = defaultdict(list)

    for c, omega in sorted(expected_configs):
        for seed in expected_seeds:
            for noise_level in expected_noise_levels:
                key = (c, omega, seed, noise_level)
                horizon_r2 = grouped.get(key)
                if horizon_r2 is None or set(horizon_r2) != set(weights):
                    raise RuntimeError(
                        "Missing centered-selection horizons for "
                        f"C={c}, omega={omega:g}, seed={seed}, "
                        f"noise={noise_level:g}."
                    )
                score = sum(
                    weights[h] * (1.0 - horizon_r2[h])
                    for h in weights
                )
                seed_noise_scores[(c, omega, seed)].append(score)

    ranking_numeric: list[dict[str, float | int]] = []
    for c, omega in sorted(expected_configs):
        seed_scores: list[float] = []
        for seed in expected_seeds:
            values = seed_noise_scores[(c, omega, seed)]
            if len(values) != len(expected_noise_levels):
                raise RuntimeError(
                    f"Expected {len(expected_noise_levels)} noise scores for "
                    f"C={c}, omega={omega:g}, seed={seed}; found {len(values)}."
                )
            seed_scores.append(mean(values))

        score_mean = mean(seed_scores)
        score_sd = stdev(seed_scores)
        score_se = score_sd / math.sqrt(len(seed_scores))
        ranking_numeric.append(
            {
                "centered_rank": 0,
                "n_clusters": c,
                "omega": omega,
                "n_seeds": len(seed_scores),
                "n_noise_levels_per_seed": len(expected_noise_levels),
                "centered_robust_score_mean": score_mean,
                "centered_robust_score_sd": score_sd,
                "centered_robust_score_se": score_se,
                "mean_robust_multihorizon_r2": 1.0 - score_mean,
                "inside_centered_one_se_set": 0,
                "selected_by_centered_retvar_rule": 0,
            }
        )

    ranking_numeric.sort(
        key=lambda row: float(row["centered_robust_score_mean"])
    )
    best_mean = float(
        ranking_numeric[0]["centered_robust_score_mean"]
    )
    best_se = float(
        ranking_numeric[0]["centered_robust_score_se"]
    )
    one_se_limit = best_mean + best_se

    for rank, row in enumerate(ranking_numeric, start=1):
        row["centered_rank"] = rank
        row["centered_best_mean_score"] = best_mean
        row["centered_best_standard_error"] = best_se
        row["centered_one_se_limit"] = one_se_limit
        row["inside_centered_one_se_set"] = int(
            float(row["centered_robust_score_mean"])
            <= one_se_limit + 1e-15
        )

    finalists = [
        row
        for row in ranking_numeric
        if int(row["inside_centered_one_se_set"]) == 1
    ]

    # In the validated Experiment 12 grid the one-SE set contains one
    # configuration, so the retained-variation tie-break is not invoked.
    # If a future rerun yields multiple finalists, fail loudly rather than
    # pretending the raw legacy CSV contains retained-variation diagnostics.
    if len(finalists) != 1:
        raise RuntimeError(
            "Reconstructed Experiment 12 centered one-SE set contains "
            f"{len(finalists)} finalists. Retained-variation diagnostics are "
            "required to resolve multiple finalists."
        )

    selected_numeric = finalists[0]
    selected_numeric["selected_by_centered_retvar_rule"] = 1

    weight_text = "50:0.25,100:0.25,200:0.5"

    def to_string_row(row: dict[str, float | int]) -> dict[str, str]:
        return {
            key: (
                str(value)
                if isinstance(value, int)
                else f"{float(value):.17g}"
            )
            for key, value in row.items()
        } | {
            "selection_weights": weight_text,
        }

    ranking_rows = [to_string_row(row) for row in ranking_numeric]
    selected_row = to_string_row(selected_numeric)

    return ranking_rows, selected_row, len(finalists)


def main() -> None:
    duffing_reference_script = (
        ROOT / "experiment_02_duffing_sensitivity.py"
    )
    duffing_expanded_meta = load_json(
        ROOT
        / "kahkm_table3_duffing"
        / "table3_duffing_metadata.json"
    )

    # Centered/retained-variation clean Van der Pol selection.
    vdp_clean_meta_path = (
        ROOT
        / "kahkm_table4_centered_vanderpol"
        / "table4_vanderpol_metadata.json"
    )
    vdp_clean_meta = load_json(vdp_clean_meta_path)

    # Original Experiment 12 metadata remains the authoritative source for
    # the full 30-point search grid, five training-noise levels, and 450 fits.
    noise_output_dir = ROOT / "kahkm_exp12_vanderpol_noise_aware_tuning"
    noise_meta_path = (
        noise_output_dir / "experiment_12_noise_tuning_metadata.json"
    )
    noise_meta = load_json(noise_meta_path)

    # The original full-grid run predates the centered summary CSVs. Reconstruct
    # the centered ranking exactly from its raw 2250-row horizon-level output.
    noise_raw_path = (
        noise_output_dir / "experiment_12_noise_tuning_raw.csv"
    )

    acrobot_script = (
        ROOT / "experiment_18_acrobot_sequential_decision_pylance_clean.py"
    )
    acrobot_runner = (
        ROOT / "run_exp18_acrobot_tuning_pylance_clean.py"
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

    (
        noise_centered_ranking,
        noise_selected,
        noise_one_se_finalist_count,
    ) = reconstruct_noise_centered_ranking(
        noise_raw_path,
        noise_config,
    )
    noise_centered_selected_rows = [noise_selected]

    # ----- Validate centered clean Van der Pol selection provenance -----
    clean_grid = vdp_clean_meta.get("grid")
    clean_rule = vdp_clean_meta.get("selection_rule")
    clean_selected = vdp_clean_meta.get("selected")
    clean_finalists = vdp_clean_meta.get("one_se_finalists")

    if not isinstance(clean_grid, dict):
        raise RuntimeError("Missing Table 4 centered grid metadata.")
    if not isinstance(clean_rule, dict):
        raise RuntimeError("Missing Table 4 centered selection-rule metadata.")
    if not isinstance(clean_selected, dict):
        raise RuntimeError("Missing Table 4 centered selected-config metadata.")
    if not isinstance(clean_finalists, list):
        raise RuntimeError("Missing Table 4 one-SE finalist metadata.")

    clean_clusters = [int(value) for value in clean_grid["clusters"]]
    clean_omegas = [float(value) for value in clean_grid["omegas"]]
    clean_seeds = [int(value) for value in clean_grid["random_states"]]
    clean_horizons = [
        int(value) for value in clean_rule["selection_horizons"]
    ]

    assert clean_clusters == [10, 15, 20, 25, 30, 40, 50]
    assert clean_omegas == [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    assert clean_seeds == [0, 1, 2]
    assert clean_horizons == [1, 10, 50, 100, 200]
    assert int(vdp_clean_meta["full_centered_ranking_rows"]) == 49
    assert len(clean_finalists) == 3

    clean_selected_c = int(clean_selected["n_clusters"])
    clean_selected_omega = float(clean_selected["omega"])
    assert clean_selected_c == 25
    assert_close(
        clean_selected_omega,
        6.0,
        "clean Van der Pol selected omega",
    )
    assert int(clean_selected["inside_one_se"]) == 1
    assert int(clean_selected["selected"]) == 1
    assert (
        clean_rule["primary_objective"]
        == "mean centered error mean_h(1 - R_h^2)"
    )
    assert (
        clean_rule["within_one_se_primary_tiebreak"]
        == "largest normalized retained variation rho_var"
    )
    assert (
        clean_rule["within_one_se_secondary_tiebreak"]
        == "largest normalized effective rank rho_rank"
    )

    # ----- Validate centered noise-aware Van der Pol selection provenance -----
    expected_noise_grid_count = (
        len(noise_config["clusters"]) * len(noise_config["omegas"])
    )
    assert expected_noise_grid_count == 30
    assert len(noise_centered_ranking) == expected_noise_grid_count
    assert noise_one_se_finalist_count == 1

    centered_ranks = sorted(
        finite_int(
            row,
            "centered_rank",
            context="noise-aware centered ranking",
        )
        for row in noise_centered_ranking
    )
    assert centered_ranks == list(
        range(1, expected_noise_grid_count + 1)
    )

    if len(noise_centered_selected_rows) != 1:
        raise RuntimeError(
            "Expected exactly one centered noise-aware selected-config row, "
            f"found {len(noise_centered_selected_rows)}."
        )
    noise_selected = noise_centered_selected_rows[0]
    noise_selected_c = finite_int(
        noise_selected,
        "n_clusters",
        context="noise-aware selected config",
    )
    noise_selected_omega = finite_float(
        noise_selected,
        "omega",
        context="noise-aware selected config",
    )
    assert noise_selected_c == 25
    assert_close(
        noise_selected_omega,
        4.0,
        "noise-aware selected omega",
    )
    assert (
        finite_int(
            noise_selected,
            "inside_centered_one_se_set",
            context="noise-aware selected config",
        )
        == 1
    )
    assert (
        finite_int(
            noise_selected,
            "selected_by_centered_retvar_rule",
            context="noise-aware selected config",
        )
        == 1
    )

    noise_weights_text = noise_selected["selection_weights"].strip()
    noise_weights = parse_selection_weights(noise_weights_text)
    expected_noise_weights = {50: 0.25, 100: 0.25, 200: 0.50}
    assert set(noise_weights) == set(expected_noise_weights)
    for horizon, expected_weight in expected_noise_weights.items():
        assert_close(
            noise_weights[horizon],
            expected_weight,
            f"noise-aware weight at h={horizon}",
        )

    # Confirm the selected row is also the unique selected row in the
    # full centered ranking and that it belongs to the one-SE set.
    noise_selected_from_ranking = [
        row
        for row in noise_centered_ranking
        if finite_int(
            row,
            "selected_by_centered_retvar_rule",
            context="noise-aware centered ranking",
        )
        == 1
    ]
    assert len(noise_selected_from_ranking) == 1
    selected_rank_row = noise_selected_from_ranking[0]
    assert (
        finite_int(
            selected_rank_row,
            "n_clusters",
            context="noise-aware centered ranking selected row",
        )
        == 25
    )
    assert_close(
        finite_float(
            selected_rank_row,
            "omega",
            context="noise-aware centered ranking selected row",
        ),
        4.0,
        "noise-aware centered-ranking selected omega",
    )
    assert (
        finite_int(
            selected_rank_row,
            "inside_centered_one_se_set",
            context="noise-aware centered ranking selected row",
        )
        == 1
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

    validation_seeds = acrobot_validation_meta["cli_args"]["seeds"]
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
            "objective_detail": "minimize mean test closure error",
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
            "objective_detail": duffing_expanded_meta[
                "ranking_objective"
            ],
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
            "C": clean_clusters,
            "omega": clean_omegas,
            "grid_count": len(clean_clusters) * len(clean_omegas),
            "search_seeds": clean_seeds,
            "noise_levels": [],
            "horizons": clean_horizons,
            "objective": "mean_centered_error_one_se_retvar",
            "objective_detail": (
                "rank by mean_h(1-R_h^2) over h={1,10,50,100,200}; "
                "form one-SE set using the SE of the minimum-error "
                "configuration; within the set choose largest rho_var "
                "with rho_rank as secondary tie-break"
            ),
            "scheduled_search_evaluations": (
                len(clean_clusters)
                * len(clean_omegas)
                * len(clean_seeds)
            ),
            "completed_search_evaluations": "",
            "validation": (
                "centered one-SE finalists verified; selected "
                "C=25, omega=6 by retained-variation rule"
            ),
            "source": (
                "run_exp09_vanderpol_tuning.py; "
                "kahkm_table4_centered_vanderpol/"
                "table4_vanderpol_metadata.json"
            ),
        },
        {
            "study": "Van der Pol noise-aware",
            "C": noise_config["clusters"],
            "omega": noise_config["omegas"],
            "grid_count": expected_noise_grid_count,
            "search_seeds": noise_config["random_states"],
            "noise_levels": noise_config["noise_levels"],
            "horizons": noise_config["horizons"],
            "objective": "weighted_centered_error_one_se_retvar",
            "objective_detail": (
                "per seed/noise: 0.25*(1-R2_50) + "
                "0.25*(1-R2_100) + 0.50*(1-R2_200); "
                "average across noise levels per seed, then across seeds; "
                "form centered one-SE set and use retained variation "
                "as tie-break when multiple finalists remain"
            ),
            "scheduled_search_evaluations": int(
                noise_meta["total_jobs"]
            ),
            "completed_search_evaluations": int(
                noise_meta["completed_jobs"]
            ),
            "validation": (
                "clean held-out evaluation; noise applied only to "
                "training snapshots; full centered 30-config ranking "
                "reconstructed from raw data; unique one-SE finalist and "
                "selected configuration C=25, omega=4"
            ),
            "source": (
                "run_exp12_vanderpol_noise_aware_tuning.py; "
                "kahkm_exp12_vanderpol_noise_aware_tuning/"
                "experiment_12_noise_tuning_metadata.json; "
                "experiment_12_noise_tuning_raw.csv; centered ranking "
                "reconstructed deterministically from raw association R2"
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
        (
            r"Van der Pol clean & $10,15,20,25,30,40,50$ & "
            r"$0.5,1,2,4,6,8,12$ & 49 & "
            r"$\overline{(1-R^2)}_{\mathcal H}$; "
            r"1-SE/$\rho_{\rm var}$ \\"
        ),
        (
            r"Van der Pol noise-aware & $10,15,20,25,30,40$ & "
            r"$0.25,0.5,1,2,4$ & 30 & weighted $(1-R^2)$; "
            r"1-SE/$\rho_{\rm var}$ \\"
        ),
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
                    "vanderpol_clean_centered_selection": {
                        "verified": True,
                        "selected_C": clean_selected_c,
                        "selected_omega": clean_selected_omega,
                        "one_se_finalists": len(clean_finalists),
                        "selection_horizons": clean_horizons,
                        "primary_objective": clean_rule[
                            "primary_objective"
                        ],
                        "retained_variation_tiebreak": clean_rule[
                            "within_one_se_primary_tiebreak"
                        ],
                    },
                    "vanderpol_noise_grid": True,
                    "vanderpol_noise_centered_selection": {
                        "verified": True,
                        "full_centered_ranking_rows": len(
                            noise_centered_ranking
                        ),
                        "selected_C": noise_selected_c,
                        "selected_omega": noise_selected_omega,
                        "selection_weights": noise_weights,
                        "one_se_finalists": noise_one_se_finalist_count,
                        "ranking_source": "reconstructed from original raw association R2",
                    },
                    "noise_aware_jobs": "450/450",
                    "acrobot_grid": True,
                    "acrobot_selection_horizon": 50,
                    "acrobot_search_seed": 0,
                    "acrobot_validation_seeds": [0, 1, 2],
                },
                "provenance": {
                    "vanderpol_clean_centered_metadata": str(
                        vdp_clean_meta_path
                    ),
                    "vanderpol_noise_full_grid_metadata": str(
                        noise_meta_path
                    ),
                    "vanderpol_noise_raw_full_grid": str(
                        noise_raw_path
                    ),
                    "vanderpol_noise_centered_ranking_reconstruction": (
                        "scripts/generate_search_protocol.py reconstructs "
                        "weighted centered scores from the original raw grid"
                    ),
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
    print(
        "Verified centered Van der Pol clean selection: "
        f"C={clean_selected_c}, omega={clean_selected_omega:g}."
    )
    print(
        "Verified centered Van der Pol noise-aware selection: "
        f"C={noise_selected_c}, omega={noise_selected_omega:g}; "
        f"{len(noise_centered_ranking)} grid configurations reconstructed "
        "from raw data."
    )


if __name__ == "__main__":
    main()
