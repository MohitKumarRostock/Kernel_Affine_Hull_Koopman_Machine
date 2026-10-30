"""Build the integrated paper's displays from the two unchanged study archives.

This is post-study reporting. It does not fit, tune, or select a forecasting model.
Run from any directory after restoring the natural_v1 reference run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    generated = HERE / "manuscript/generated"
    generated.mkdir(parents=True, exist_ok=True)
    sources = {}
    for stage, extension in (("controlled", "predictive_geometry"), ("natural", "natural_forecasting")):
        source = ROOT / "extensions" / extension / "manuscript/generated"
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            destination = generated / stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            sources[str(path.relative_to(ROOT))] = digest(path)

    reports = ROOT / "extensions/natural_forecasting/runs/natural_v1/reports"
    def rows(name):
        path = reports / name
        sources[str(path.relative_to(ROOT))] = digest(path)
        with path.open(newline="") as stream:
            return {row["id"]: row for row in csv.DictReader(stream) if row["phase"] == "confirmation"}

    methods, policies = rows("method_summary.csv"), rows("policy_summary.csv")
    if len(methods) != 12 or any(int(v["tasks"]) != 80 or int(v["source_families"]) != 4 for v in methods.values()):
        raise ValueError("Expected all twelve candidates on eighty tasks from four confirmation families")
    loss = lambda m: float(methods[m]["scaled_test_mse"])
    cost = lambda m: float(methods[m]["total_policy_wall_seconds"])
    probe = policies["metadata_probe"]
    geometry = policies["metadata_probe_geometry"]
    if probe["scaled_test_mse"] != geometry["scaled_test_mse"]:
        raise ValueError("Archived primary diagnostic pair no longer has identical reported loss")
    macros = {
        "NaturalRawLoss": f"{loss('raw_rbf'):.4f}",
        "NaturalKAHMLoss": f"{loss('raw_kahm_rbf'):.4f}",
        "NaturalRidgeLoss": f"{loss('raw_ridge'):.4f}",
        "NaturalRawCost": f"{cost('raw_rbf'):.4f}",
        "NaturalKAHMCost": f"{cost('raw_kahm_rbf'):.4f}",
        "NaturalRidgeCost": f"{cost('raw_ridge'):.4f}",
        "NaturalErrorIncrease": f"{100 * (loss('raw_kahm_rbf') / loss('raw_rbf') - 1):.2f}",
        "NaturalCostRatio": f"{cost('raw_kahm_rbf') / cost('raw_rbf'):.2f}",
        "NaturalProbeLoss": f"{float(probe['scaled_test_mse']):.6f}",
        "NaturalProbeCost": f"{float(probe['total_policy_wall_seconds']):.4f}",
        "NaturalProbeCostRatio": f"{float(probe['total_policy_wall_seconds']) / cost('raw_rbf'):.2f}",
        "NaturalProbeGainPercent": f"{100 * (1 - float(probe['scaled_test_mse']) / loss('raw_rbf')):.3f}",
    }
    (generated / "natural_numbers.tex").write_text(
        "% Generated from the unchanged confirmation method/policy reports.\n" +
        "".join("\\newcommand{\\" + key + "}{" + value + "}\n" for key, value in macros.items()))

    order = ["raw_ridge", "raw_rbf", "affine_rbf", "locality_rbf", "kahm_rbf",
             "raw_affine_rbf", "raw_locality_rbf", "raw_kahm_rbf", "seasonal_naive", "holt", "hist_gradient", "mlp"]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\caption{All twelve standalone methods on the reserved natural sources. Scaled MSE and warm-library wall time use equal source-family weights and the frozen within-source hierarchy. Times include required preprocessing, search, refitting, prediction, serialization, and measured bookkeeping; imports and launch are separate. These are descriptive results from four source families.}",
             r"\label{tab:natural-main}", r"\begin{tabular}{lrr}", r"\toprule", r"Method & Scaled MSE & Wall seconds per task \\", r"\midrule"]
    for key in order:
        label = methods[key]["label"]
        lines.append(f"{label} & {loss(key):.4f} & {cost(key):.4f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (generated / "natural_main_table.tex").write_text("\n".join(lines) + "\n")
    result = {"format": "tmlr_integrated_assets_v1", "scope": "Post-study reporting; no scientific source or outcomes changed",
              "input_sha256": sources, "macros": macros,
              "generated_sha256": {str(p.relative_to(generated)): digest(p) for p in sorted(generated.rglob('*')) if p.is_file()}}
    (HERE / "manuscript/ASSET_PROVENANCE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"status": "passed", "source_files": len(sources), "generated_files": len(result["generated_sha256"]), "forecasting_fits": 0}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
