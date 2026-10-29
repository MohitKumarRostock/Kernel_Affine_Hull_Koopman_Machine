"""Independent protocol and intervention-isolation audit of a fixed first run.

Does not rank model performance or choose preprocessing using outcomes.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parent / "kahm-study"
EXT = REPO / "extensions/certified_forecasting"
sys.path.insert(0, str(EXT / "src"))
from certified_forecasting.data import make_episodes, histories_targets


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def without_timings(value):
    if isinstance(value, dict):
        return {k: without_timings(v) for k, v in value.items()
                if k not in {"train_seconds", "elapsed_seconds"}}
    if isinstance(value, list):
        return [without_timings(v) for v in value]
    return value


manifest = json.loads((STUDY / "results/study_manifest.json").read_text())
checks = {}
for key, filename in [("protocol_sha256", "protocol.json"),
                      ("harness_sha256", "run_scaling.py"),
                      ("scaler_sha256", "scaling_features.py")]:
    checks["unchanged_" + key] = manifest[key] == digest(STUDY / filename)
checks["parent_sources_match_frozen_hashes"] = all(
    digest(REPO / name) == sha for name, sha in manifest["source_sha256"].items())
checks["repository_working_tree_clean"] = not subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=REPO, text=True).strip()

cfg = json.loads((EXT / "configs/main_v1.json").read_text())
record_name, seed = "harmonic_1701", 1701
train = make_episodes("harmonic", cfg["n_train"], cfg["episode_length"], seed * 100 + 11)
independent = {}
for history in cfg["histories"]:
    X, _ = histories_targets(train, history, cfg["horizons"], cfg["origin_stride"],
                              max(cfg["histories"]) - 1)
    X = X.reshape(-1, X.shape[-1])
    idx = np.sort(np.random.default_rng(seed + history).choice(
        len(X), cfg["max_geometry_inputs"], replace=False))
    independent[history] = {
        "n_rows": len(X), "min": X.min(axis=0).tolist(), "max": X.max(axis=0).tolist(),
        "mean": X.mean(axis=0).tolist(), "std": X.std(axis=0).tolist(),
        "index_hash": hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest(),
    }

standard_path = STUDY / "results/standard/main_v1" / record_name
standard = json.loads((standard_path / "results.json").read_text())
original = json.loads((EXT / "results/main_v1" / record_name / "results.json").read_text())
checks["standard_replay_equals_published_non_timing_record"] = (
    without_timings(standard) == without_timings(original))
baseline = {r["name"]: without_timings(r) for r in standard["results"]
            if not r["family"].startswith("kahm")}

banks_summary = []
for mode in ("minmax", "minmax_clip", "raw"):
    directory = STUDY / "results" / mode / "main_v1" / record_name
    run_manifest = json.loads((directory / "intervention_manifest.json").read_text())
    checks[f"{mode}_manifest_matches_frozen_config"] = (
        run_manifest["configuration"] == cfg and
        run_manifest["config_sha256"] == digest(EXT / "configs/main_v1.json"))
    checks[f"{mode}_manifest_matches_frozen_sources"] = all(
        run_manifest[k] == manifest[k]
        for k in ["protocol_sha256", "harness_sha256", "scaler_sha256", "source_sha256"])
    result = json.loads((directory / "results.json").read_text())
    checks[f"{mode}_same_candidates_and_streams"] = (
        result["streams"] == standard["streams"] and
        [r["name"] for r in result["results"]] == [r["name"] for r in standard["results"]])
    controls = {r["name"]: without_timings(r) for r in result["results"]
                if not r["family"].startswith("kahm")}
    checks[f"{mode}_non_kahm_records_exactly_unchanged"] = controls == baseline
    with np.load(standard_path / "episode_scores.npz") as old, \
            np.load(directory / "episode_scores.npz") as new:
        control_keys = [name for name in old.files if not name.startswith("kahm_")]
        checks[f"{mode}_non_kahm_episode_scores_exactly_unchanged"] = all(
            np.array_equal(old[name], new[name]) for name in control_keys)
    banks = json.loads((directory / "scaling_diagnostics.json").read_text())
    for bank in banks:
        h = bank["n_input_features"]
        reference = independent[h]
        stem = f"{mode}_{bank['kind']}_m{h}"
        checks[stem + "_training_statistics_and_indices"] = (
            bank["training_rows_for_scaler"] == reference["n_rows"] and
            bank["training_rows_for_geometry"] == cfg["max_geometry_inputs"] and
            bank["training_min"] == reference["min"] and bank["training_max"] == reference["max"] and
            bank["training_mean"] == reference["mean"] and bank["training_std"] == reference["std"] and
            bank["fit_indices_sha256"] == reference["index_hash"])
        if bank["kind"] != "kahm":
            checks[stem + "_inherits_standard_preprocessing"] = bank["effective_mode"] == "standard"
            continue
        fit = bank["diagnostics"]["fit_input"]
        checks[stem + "_training_range"] = (
            fit["preclip_min"] >= -1.0 and fit["preclip_max"] <= 1.0 and
            fit["outside_training_extrema_coordinates"] == 0)
        if mode.startswith("minmax"):
            checks[stem + "_training_extrema_exact_minus_plus_one"] = (
                fit["preclip_min"] == -1.0 and fit["preclip_max"] == 1.0)
        banks_summary.append({
            "mode": mode, "history": h, "scaler_rows": bank["training_rows_for_scaler"],
            "geometry_rows": bank["training_rows_for_geometry"],
            "fit_indices_sha256": bank["fit_indices_sha256"],
            "fit_range": [fit["preclip_min"], fit["preclip_max"]],
            "evaluation_diagnostics": bank["diagnostics"]["unlabeled"],
        })

output = {
    "scope": "Fixed harmonic_1701 run; no performance-ranking or outcome-based model selection",
    "checks": checks, "all_passed": all(checks.values()),
    "checks_count": len(checks), "banks": banks_summary,
    "interpretation": "Pure minmax guarantees the training range only; clipping is separately evaluated. Diagnostic counts include repeated transforms. Unchanged original risk deltas are within-arm, not joint over four arms.",
}
destination = Path(__file__).with_name("preprocessing_audit.json")
destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
print(json.dumps({"checks_count": len(checks), "all_passed": output["all_passed"],
                  "failed": [name for name, passed in checks.items() if not passed]}, indent=2))
if not output["all_passed"]:
    raise SystemExit(1)
