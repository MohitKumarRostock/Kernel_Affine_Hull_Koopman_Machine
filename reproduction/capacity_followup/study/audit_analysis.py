"""Independent, prespecified analysis for the KAHM geometry study.

This module does not import the data generators, geometry, head, or runner.
It consumes their stored numeric records and recomputes paired statistics.
It intentionally makes no distribution-free confidence claim.

Examples:
  python audit_analysis.py freeze --records pilot_records.json --output margins.json
  python audit_analysis.py analyze --records records.json --margins margins.json \
      --output-dir independent_analysis --check-unit-losses
  python audit_analysis.py selftest
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import t as student_t


STUDY_ROOT = Path(__file__).resolve().parent
PRIMARY_SETTINGS = (
    "branch_c0_n1024_l0", "branch_c0_n1024_l0p5", "branch_c0_n1024_l1",
    "branch_c1_n1024_l0", "branch_c1_n1024_l0p5", "branch_c1_n1024_l1",
    "forecast_nonlinear_ar", "forecast_van_der_pol", "forecast_duffing",
)
CELL_CONTRASTS = (
    ("G", "affine:phi", "kahm:phi", "KAHM reconstruction advantage"),
    ("N", "kahm:phi", "kahm:phi_logz", "Log-normalizer augmentation benefit"),
    ("L", "kahm:phi", "kahm:local", "Local-coordinate augmentation benefit"),
)
REQUIRED_FEATURES = ("raw", "affine:phi", "kahm:phi", "kahm:phi_logz", "kahm:local")
N_CONFIRMATION = 24
N_PILOT = 2
N_PRIMARY = 29
FAMILY_ALPHA = 0.05
PRACTICAL_FRACTION = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(record: dict[str, Any]) -> str:
    clean = {key: value for key, value in record.items() if not key.startswith("_audit_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_records(paths) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read a record array, {records:[...]}, JSONL, CSV, or bank-result directory."""
    expanded = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.json")))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("No result inputs found")
    if len(expanded) != len(set(expanded)):
        raise ValueError("The same result file was supplied more than once")
    rows, sources = [], []
    for path in expanded:
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as stream:
                part = list(csv.DictReader(stream))
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            part = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                part = payload
            elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
                part = payload["records"]
            elif isinstance(payload, dict) and all(key in payload for key in
                                                   ("phase", "replicate", "setting_id", "feature", "head")):
                part = [payload]
            else:
                # Frozen geometry metadata is not an outcome record.
                continue
        for row in part:
            if not isinstance(row, dict):
                raise ValueError(f"Non-object result row in {path}")
            for key in ("phase", "replicate", "setting_id", "feature", "head"):
                if key not in row:
                    raise ValueError(f"Missing {key!r} in {path}")
            normalized = dict(row)
            normalized["replicate"] = str(row["replicate"])
            normalized["_audit_source_file"] = str(path)
            rows.append(normalized)
        sources.append({"path": str(path), "sha256": sha256_file(path), "records": len(part)})
    if not rows:
        raise ValueError("No numeric result records were found")
    return rows, sources


def key_of(row):
    return (str(row["phase"]), str(row["replicate"]), str(row["setting_id"]),
            str(row["feature"]), str(row["head"]))


def make_index(rows):
    index = {}
    for row in rows:
        key = key_of(row)
        if key in index:
            raise ValueError(f"Duplicate result key {key}; do not mix aggregated and original records")
        index[key] = row
    return index


def mse_of(row):
    try:
        value = float(row["test_mse"])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Missing or invalid test MSE at {key_of(row)}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Nonfinite or negative test MSE at {key_of(row)}")
    return value


def sorted_replicates(values):
    return sorted(values, key=lambda value: (0, int(value)) if str(value).lstrip("-").isdigit()
                  else (1, str(value)))


def source_versions():
    names = ("ANALYSIS_SPEC.txt", "THEORETICAL_NOTE.txt", "audit_analysis.py")
    return {name: sha256_file(STUDY_ROOT / name) for name in names}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze_margins(records, sources, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite frozen margins: {output}")
    if any(row["phase"] == "confirmation" for row in records):
        raise ValueError("Cannot freeze margins from inputs containing confirmation records")
    index = make_index(records)
    cells = {}
    common_replicates = None
    for setting in PRIMARY_SETTINGS:
        raw_rows = [row for key, row in index.items()
                    if key[0] == "pilot" and key[2:] == (setting, "raw", "ridge")]
        replicates = sorted_replicates({row["replicate"] for row in raw_rows})
        if len(raw_rows) != N_PILOT or len(replicates) != N_PILOT:
            raise ValueError(f"{setting}: need exactly {N_PILOT} pilot raw-ridge outcomes")
        if common_replicates is None:
            common_replicates = replicates
        elif replicates != common_replicates:
            raise ValueError("Primary settings do not share the same two pilot replicate IDs")
        raw_rows.sort(key=lambda row: replicates.index(row["replicate"]))
        mses = [mse_of(row) for row in raw_rows]
        cells[setting] = {
            "margin": PRACTICAL_FRACTION * float(np.mean(mses)),
            "pilot_raw_ridge_mean_mse": float(np.mean(mses)),
            "pilot_raw_ridge_mses": mses,
            "pilot_replicates": replicates,
            "pilot_record_sha256": [record_hash(row) for row in raw_rows],
        }
    interactions = {}
    for curvature in (0, 1):
        endpoints = [f"branch_c{curvature}_n1024_l0", f"branch_c{curvature}_n1024_l1"]
        interactions[f"I_c{curvature}"] = {
            "margin": float(np.mean([cells[name]["margin"] for name in endpoints])),
            "endpoints": endpoints,
            "formula": "geometry advantage at lambda=0 minus advantage at lambda=1",
        }
    payload = {
        "format": "kahm_prespecified_margins_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prospective_local_plan_not_external_registration": True,
        "pilot_replicates": common_replicates,
        "confirmation_replicates_planned": N_CONFIRMATION,
        "primary_family_size": N_PRIMARY,
        "family_alpha": FAMILY_ALPHA,
        "practical_fraction": PRACTICAL_FRACTION,
        "margin_scope": "Fixed study-design margin conditional on two independent pilots; not an application requirement.",
        "cells": cells,
        "interactions": interactions,
        "source_versions": source_versions(),
        "input_sources": sources,
        "confirmation_records_in_inputs": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return {"status": "margins_frozen", "output": str(output), "sha256": sha256_file(output),
            "n_cells": len(cells), "n_interactions": len(interactions)}


def paired_stats(values, margin):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != N_CONFIRMATION or not np.isfinite(values).all():
        raise ValueError(f"A primary contrast needs {N_CONFIRMATION} finite paired differences")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    critical = float(student_t.ppf(1.0 - FAMILY_ALPHA / (2.0 * N_PRIMARY), len(values) - 1))
    lower, upper = mean - critical * se, mean + critical * se
    if lower > margin:
        classification = "practical_benefit"
    elif lower > 0:
        classification = "positive_practical_size_unresolved"
    elif upper < 0:
        classification = "harm"
    else:
        classification = "direction_unresolved"
    centered = values - mean
    empirical_m2 = float(np.mean(centered**2))
    skewness = (float(np.mean(centered**3)) / empirical_m2**1.5) if empirical_m2 > 0 else 0.0
    max_standardized = float(np.max(np.abs(centered)) / sd) if sd > 0 else 0.0
    return {
        "n_replicates": len(values), "mean_improvement": mean,
        "sd_improvement": sd, "se_improvement": se,
        "simultaneous_ci_lower": lower, "simultaneous_ci_upper": upper,
        "practical_margin": float(margin), "classification": classification,
        "within_prespecified_practical_band": bool(lower >= -margin and upper <= margin),
        "critical_t": critical, "degrees_of_freedom": len(values) - 1,
        "family_size": N_PRIMARY, "family_alpha": FAMILY_ALPHA,
        "minimum_replicate_improvement": float(values.min()),
        "maximum_replicate_improvement": float(values.max()),
        "positive_replicates": int(np.sum(values > 0)),
        "zero_variance": bool(sd == 0),
        "empirical_skewness_diagnostic": skewness,
        "max_absolute_standardized_deviation": max_standardized,
    }


def load_and_check_margins(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("format") != "kahm_prespecified_margins_v1":
        raise ValueError("Unrecognized frozen-margin format")
    if data.get("primary_family_size") != N_PRIMARY or data.get("confirmation_replicates_planned") != N_CONFIRMATION:
        raise ValueError("Frozen analysis family or replicate count differs from this analyzer")
    if set(data.get("cells", {})) != set(PRIMARY_SETTINGS):
        raise ValueError("Frozen margins do not cover exactly the nine primary settings")
    for setting in PRIMARY_SETTINGS:
        cell = data["cells"][setting]
        mses = np.asarray(cell["pilot_raw_ridge_mses"], dtype=float)
        if len(mses) != N_PILOT or not np.isfinite(mses).all() or np.any(mses < 0):
            raise ValueError(f"Invalid pilot MSEs in frozen margin for {setting}")
        expected = PRACTICAL_FRACTION * float(mses.mean())
        if not math.isclose(float(cell["margin"]), expected, rel_tol=1e-14, abs_tol=1e-15):
            raise ValueError(f"Frozen margin arithmetic mismatch for {setting}")
    for curvature in (0, 1):
        key = f"I_c{curvature}"
        endpoints = [f"branch_c{curvature}_n1024_l0", f"branch_c{curvature}_n1024_l1"]
        expected = float(np.mean([data["cells"][name]["margin"] for name in endpoints]))
        if data["interactions"][key]["endpoints"] != endpoints or not math.isclose(
            float(data["interactions"][key]["margin"]), expected, rel_tol=1e-14, abs_tol=1e-15
        ):
            raise ValueError(f"Frozen interaction margin mismatch for {key}")
    mismatches = []
    for name, frozen_hash in data.get("source_versions", {}).items():
        actual_hash = sha256_file(STUDY_ROOT / name)
        if actual_hash != frozen_hash:
            mismatches.append({"file": name, "frozen": frozen_hash, "current": actual_hash})
    # A later analyzer correction must remain visible; it does not silently
    # change the frozen scientific plan or margin inputs.
    return data, mismatches


def unit_loss_audit(records, study_root):
    groups = defaultdict(list)
    missing_metadata = []
    for row in records:
        if row["phase"] not in {"confirmation", "natural"}:
            continue
        file_value, key = row.get("unit_loss_file"), row.get("unit_loss_key")
        if not file_value or not key:
            missing_metadata.append(list(key_of(row)))
            continue
        raw_path = Path(file_value)
        candidates = [raw_path] if raw_path.is_absolute() else [
            Path(study_root) / raw_path,
            Path(row["_audit_source_file"]).parent / raw_path,
        ]
        path = next((value.resolve() for value in candidates if value.is_file()), None)
        if path is None:
            raise ValueError(f"Stored unit-loss file not found for {key_of(row)}: {file_value}")
        groups[path].append(row)
    checked, discrepancies, file_hashes = 0, [], []
    for path, rows in sorted(groups.items(), key=lambda pair: str(pair[0])):
        with np.load(path, allow_pickle=False) as arrays:
            for row in rows:
                key = row["unit_loss_key"]
                if key not in arrays:
                    raise ValueError(f"Missing unit-loss array {key!r} in {path}")
                losses = np.asarray(arrays[key], dtype=float)
                if losses.size == 0 or not np.isfinite(losses).all() or np.any(losses < 0):
                    raise ValueError(f"Invalid unit-loss array {key!r} in {path}")
                reproduced = float(losses.mean())
                recorded = mse_of(row)
                error = abs(reproduced - recorded)
                if not math.isclose(reproduced, recorded, rel_tol=1e-10, abs_tol=1e-12):
                    discrepancies.append({"record_key": list(key_of(row)), "recorded": recorded,
                                          "reproduced": reproduced, "absolute_error": error})
                checked += 1
        file_hashes.append({"path": str(path), "sha256": sha256_file(path), "checked_records": len(rows)})
    return {"checked_records": checked, "missing_unit_loss_metadata": missing_metadata,
            "discrepancies": discrepancies, "files": file_hashes,
            "scope": "Recomputed mean of stored independent-unit losses; this does not by itself verify predictions or fit independence."}


def summarize_methods(records):
    groups = defaultdict(list)
    for row in records:
        if row["phase"] in {"confirmation", "natural"}:
            groups[(row["phase"], row["setting_id"], row["feature"], row["head"])].append(row)
    summaries = []
    for (phase, setting, feature, head), rows in sorted(groups.items()):
        values, failures = [], 0
        conditional = []
        for row in rows:
            try:
                values.append(mse_of(row))
            except ValueError:
                failures += 1
            cm = row.get("conditional_mean_mse")
            if cm is not None and cm != "":
                value = float(cm)
                if math.isfinite(value):
                    conditional.append(value)
        summaries.append({
            "phase": phase, "setting_id": setting, "feature": feature, "head": head,
            "n_records": len(rows), "n_valid": len(values), "n_failures": failures,
            "mean_test_mse": float(np.mean(values)) if values else None,
            "sd_test_mse": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "minimum_test_mse": float(np.min(values)) if values else None,
            "maximum_test_mse": float(np.max(values)) if values else None,
            "mean_conditional_mean_mse": float(np.mean(conditional)) if conditional else None,
            "summary_scope": ("descriptive_fitting_variation_on_fixed_cohort" if "airfoil" in setting.lower()
                              else "independent_synthetic_replicate_descriptive_summary"),
        })
    return summaries


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty summary without a schema")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(records, sources, margins_path, output_dir, check_units=False, study_root=STUDY_ROOT):
    index = make_index(records)
    margins, source_mismatches = load_and_check_margins(margins_path)
    replicate_sets = {}
    for setting in PRIMARY_SETTINGS:
        for feature in REQUIRED_FEATURES:
            matches = {key[1] for key in index
                       if key[0] == "confirmation" and key[2:] == (setting, feature, "ridge")}
            replicate_sets[(setting, feature)] = matches
            if len(matches) != N_CONFIRMATION:
                raise ValueError(f"{setting}/{feature}: expected exactly {N_CONFIRMATION} confirmation replicates, got {len(matches)}")
    reference_ids = replicate_sets[(PRIMARY_SETTINGS[0], "raw")]
    for key, values in replicate_sets.items():
        if values != reference_ids:
            raise ValueError(f"Unmatched confirmation replicate identifiers at {key}")
    replicate_ids = sorted_replicates(reference_ids)
    curves, contrast_rows, paired_rows = {}, [], []
    for setting in PRIMARY_SETTINGS:
        margin = float(margins["cells"][setting]["margin"])
        for code, control_feature, candidate_feature, title in CELL_CONTRASTS:
            controls, candidates = [], []
            for replicate in replicate_ids:
                control = mse_of(index[("confirmation", replicate, setting, control_feature, "ridge")])
                candidate = mse_of(index[("confirmation", replicate, setting, candidate_feature, "ridge")])
                controls.append(control)
                candidates.append(candidate)
            values = np.asarray(controls) - np.asarray(candidates)
            contrast_id = f"{setting}__{code}"
            curves[(setting, code)] = values
            row = {"contrast_id": contrast_id, "contrast_type": "cellwise", "setting_id": setting,
                   "contrast_code": code, "description": title, "control_feature": control_feature,
                   "candidate_feature": candidate_feature, "head": "ridge",
                   "control_mean_test_mse": float(np.mean(controls)),
                   "candidate_mean_test_mse": float(np.mean(candidates)),
                   **paired_stats(values, margin)}
            contrast_rows.append(row)
            for replicate, control, candidate, value in zip(replicate_ids, controls, candidates, values):
                paired_rows.append({"contrast_id": contrast_id, "replicate": replicate,
                                    "control_value": float(control), "candidate_value": float(candidate),
                                    "improvement": float(value)})
    for curvature in (0, 1):
        first, second = f"branch_c{curvature}_n1024_l0", f"branch_c{curvature}_n1024_l1"
        control, candidate = curves[(first, "G")], curves[(second, "G")]
        values = control - candidate
        code = f"I_c{curvature}"
        row = {"contrast_id": code, "contrast_type": "target_alignment_interaction",
               "setting_id": f"branch_c{curvature}_n1024_l0_minus_l1", "contrast_code": code,
               "description": "KAHM geometry advantage for regime target minus within-branch target",
               "control_feature": "geometry_advantage_at_lambda0", "candidate_feature": "geometry_advantage_at_lambda1",
               "head": "ridge", "control_mean_test_mse": float(control.mean()),
               "candidate_mean_test_mse": float(candidate.mean()),
               **paired_stats(values, float(margins["interactions"][code]["margin"]))}
        contrast_rows.append(row)
        for replicate, before, after, value in zip(replicate_ids, control, candidate, values):
            paired_rows.append({"contrast_id": code, "replicate": replicate, "control_value": float(before),
                                "candidate_value": float(after), "improvement": float(value)})
    if len(contrast_rows) != N_PRIMARY:
        raise AssertionError("Primary family size changed")
    pilot_provenance = []
    for setting, cell in margins["cells"].items():
        for replicate, frozen_hash in zip(cell["pilot_replicates"], cell["pilot_record_sha256"]):
            key = ("pilot", str(replicate), setting, "raw", "ridge")
            if key in index:
                actual_hash = record_hash(index[key])
                if actual_hash != frozen_hash:
                    raise ValueError(f"Pilot record changed after margin freeze: {key}")
                pilot_provenance.append({"key": list(key), "verified": True})
    unit_audit = unit_loss_audit(records, study_root) if check_units else {"performed": False}
    if unit_audit.get("discrepancies"):
        raise ValueError(f"Stored unit losses disagree with {len(unit_audit['discrepancies'])} reported MSEs")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classification_counts = defaultdict(int)
    for row in contrast_rows:
        classification_counts[row["classification"]] += 1
    summary = {
        "format": "kahm_independent_analysis_v1", "status": "complete_primary_family",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_family_size": N_PRIMARY, "confirmation_replicates": replicate_ids,
        "n_confirmation_replicates": len(replicate_ids), "family_alpha": FAMILY_ALPHA,
        "interval": "Two-sided Bonferroni paired-replicate Student t; exact under paired-normal model, otherwise approximate.",
        "not_distribution_free": True,
        "classification_counts": dict(classification_counts),
        "source_versions": source_versions(), "source_changes_since_margin_freeze": source_mismatches,
        "frozen_margin_file": str(Path(margins_path).resolve()), "frozen_margin_sha256": sha256_file(Path(margins_path)),
        "input_sources": sources, "verified_pilot_records": pilot_provenance,
        "unit_loss_audit": unit_audit, "contrasts": contrast_rows,
        "limitations": [
            "Twenty-four complete synthetic replicates; t coverage relies on a paired-normal model or adequate approximation.",
            "The synthetic generator, sampling law, implementation, and candidate grid define the population of this study.",
            "Airfoil rows, if present, describe fitting variation on a fixed grouped cohort and are outside the primary family.",
            "This audit recomputes summaries; separate source review assesses fit leakage and same-component matching.",
        ],
    }
    write_json(output_dir / "independent_analysis.json", summary)
    write_csv(output_dir / "primary_contrasts.csv", contrast_rows)
    write_csv(output_dir / "primary_paired_differences.csv", paired_rows)
    write_csv(output_dir / "method_summary.csv", summarize_methods(records))
    return {"status": summary["status"], "output_dir": str(output_dir), "primary_contrasts": len(contrast_rows),
            "classification_counts": dict(classification_counts), "source_changes": source_mismatches,
            "unit_loss_records_checked": unit_audit.get("checked_records", 0)}


def selftest():
    # Independent closed-form moments for the integers 1,...,24.
    result = paired_stats(np.arange(1.0, 25.0), 0.05)
    assert result["mean_improvement"] == 12.5
    assert math.isclose(result["sd_improvement"]**2, 50.0, rel_tol=1e-14)
    assert math.isclose(result["se_improvement"]**2, 50.0 / 24.0, rel_tol=1e-14)
    assert len(PRIMARY_SETTINGS) * len(CELL_CONTRASTS) + 2 == N_PRIMARY
    assert result["simultaneous_ci_lower"] < 12.5 < result["simultaneous_ci_upper"]
    assert paired_stats(np.ones(24), 0.05)["zero_variance"]
    return {"status": "arithmetic_selftest_passed", "primary_family_size": N_PRIMARY,
            "critical_t_df23": result["critical_t"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Freeze practical margins using exactly two pilots")
    freeze.add_argument("--records", nargs="+", required=True)
    freeze.add_argument("--output", required=True)
    audit = commands.add_parser("analyze", help="Independently recompute the 29 primary contrasts")
    audit.add_argument("--records", nargs="+", required=True)
    audit.add_argument("--margins", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--check-unit-losses", action="store_true")
    audit.add_argument("--study-root", default=str(STUDY_ROOT))
    commands.add_parser("selftest", help="Check arithmetic against closed-form sample moments")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        result = selftest()
    else:
        records, sources = read_records(args.records)
        if args.command == "freeze":
            result = freeze_margins(records, sources, args.output)
        else:
            result = analyze(records, sources, args.margins, args.output_dir,
                             args.check_unit_losses, args.study_root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, KeyError) as error:
        print(f"AUDIT ERROR: {error}", file=sys.stderr)
        sys.exit(2)
