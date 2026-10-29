#!/usr/bin/env python3
"""Writer-disjoint PenDigits forecasting with writer-level risk summaries.

Run from the extension root, after prepare_pendigits.py, with PYTHONPATH=src.
All configured coordinate/seed families are fitted, selected using development writers,
and recorded before the held-out writers are scored. We report empirical
writer-balanced cohort errors. The additional Hoeffding bounds require both
iid writer sampling and a model family independent of the evaluation cohort.
The dataset does not establish the sampling assumption. Follow-up designs that
reuse an examined cohort have descriptive endpoints unless their adaptivity is
separately accounted for. Writers, rather than strokes or windows, are the
units used in the concentration calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import scipy
import sklearn

from certified_forecasting.data import histories_targets
from certified_forecasting.certificates import episode_mse
from prepare_pendigits import ARCHIVE_SHA256, load_prepared_strokes
from run_forecasting import build_candidates, candidate_description, flat, write_json


FAMILY_LABELS = {
    "kahm_nested": "KAHM blocks",
    "rbf_nested": "RBF blocks",
    "affine_nested": "Affine-hull blocks",
    "kahm_frobenius": "KAHM, Frobenius budget",
    "kahm_duplicate": "KAHM duplicate blocks",
    "kahm_hard": "KAHM hard assignment",
    "kahm_koopman": "KAHM Koopman rollout",
    "ridge_ar": "Ridge autoregression",
    "polynomial": "Quadratic regression",
    "polynomial_koopman": "Quadratic Koopman rollout",
    "persistence": "Persistence",
    "constant": "Constant",
}


def writer_balanced_losses(stroke_losses, writer_ids):
    """Average origins within a stroke, strokes within a writer, then writers."""
    writers = np.unique(writer_ids)
    if len(stroke_losses) != len(writer_ids):
        raise ValueError("Writer vector and stroke loss rows do not align")
    return writers, np.stack(
        [stroke_losses[writer_ids == writer].mean(axis=0) for writer in writers]
    )


def select_records(records, writers, min_length):
    return [
        r for r in records
        if r["writer"] in writers and len(r["points"]) >= min_length
    ]


def episodes_from_records(records, length, coordinate):
    episodes = np.stack([r["points"][:length, coordinate] for r in records]).astype(float)
    if not np.isfinite(episodes).all() or np.any((episodes < 0) | (episodes > 500)):
        raise ValueError("Raw data violate the predeclared acquisition range")
    return (episodes / 250.0 - 1.0)[:, :, None]


def subsample_training(records, writer_ids, per_writer, seed):
    rng = np.random.default_rng(seed * 1000 + 37)
    selected = []
    for writer in writer_ids:
        available = [r for r in records if r["writer"] == writer]
        if len(available) < per_writer:
            raise ValueError(f"Writer {writer} lacks {per_writer} eligible strokes")
        chosen = rng.choice(len(available), size=per_writer, replace=False)
        selected.extend(available[i] for i in sorted(chosen))
    return selected


def record_id(record):
    return {
        "split": record["split"], "writer": record["writer"],
        "writer_sample": record["writer_sample"],
        "digit_source_id": record["digit_source_id"],
        "stroke_in_digit": record["stroke_in_digit"],
    }


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = [
        "experiments/run_pendigits.py", "experiments/prepare_pendigits.py",
        "experiments/run_forecasting.py", "src/certified_forecasting/data.py",
        "src/certified_forecasting/models.py", "src/certified_forecasting/features.py",
        "src/certified_forecasting/certificates.py",
    ]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def run(config_path, prepared_path, output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite a nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes)
    if cfg["bound_method"] != "hoeffding" or cfg["scalar_squared_loss_bound"] != 4.0:
        raise ValueError("This campaign predeclares Hoeffding with scalar loss range [0,4]")
    dataset_manifest = json.loads(prepared_path.with_name("manifest.json").read_text())
    if dataset_manifest["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Prepared data have the wrong source archive hash")
    if hashlib.sha256(prepared_path.read_bytes()).hexdigest() != dataset_manifest["array_file_sha256"]:
        raise ValueError("Prepared array hash differs from its manifest")
    write_json(output / "config.json", cfg)
    manifest = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_hashes": source_hashes(),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "source_archive_sha256": ARCHIVE_SHA256,
        "prepared_array_sha256": dataset_manifest["array_file_sha256"],
        "dataset_doi": "10.24432/C5MG6K",
        "selection": "One candidate per family minimizes mean validation-writer MSE over all horizons; ties use fewer features then lexical candidate name",
        "risk_definition": "Uniform writer, then uniform eligible first stroke, then uniform forecast origin; scalar-coordinate squared loss",
        "statistical_scope": (
            "Descriptive errors concern this fixed held-out cohort. The reported upper bounds "
            "require iid new-writer sampling, independence from development writers, and the "
            "same eligible-stroke protocol. The dataset documentation does not establish these assumptions."
        ),
        "confidence_scope": "Joint upper bounds across both coordinates, all seeds, all frozen candidates, and all horizons, under the stated writer sampling assumption",
        "no_population_claim_from_windows": True,
        "all_candidates_frozen_before_test_scoring": True,
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()

    # Phase 1 uses only the original 30-writer development file.
    development = load_prepared_strokes(prepared_path, "tra", first_only=True)
    eligible_training = select_records(
        development, set(cfg["train_writer_ids"]), cfg["episode_length"]
    )
    validation_records = select_records(
        development, set(cfg["validation_writer_ids"]), cfg["episode_length"]
    )
    validation_writers = np.asarray([r["writer"] for r in validation_records])
    first_origin = max(cfg["histories"]) - 1
    families, frozen = [], []
    for seed in cfg["seeds"]:
        chosen_training = subsample_training(
            eligible_training, cfg["train_writer_ids"],
            cfg["training_strokes_per_writer"], seed,
        )
        for coordinate in cfg["coordinates"]:
            run_start = time.perf_counter()
            train = episodes_from_records(chosen_training, cfg["episode_length"], coordinate)
            validation = episodes_from_records(validation_records, cfg["episode_length"], coordinate)
            candidates, _ = build_candidates(train, validation, cfg, seed)
            val_arrays = {
                history: histories_targets(
                    validation, history, cfg["horizons"], cfg["origin_stride"], first_origin
                )
                for history in cfg["histories"]
            }
            # Rebalance validation at writer level, rather than number of digits.
            for candidate in candidates:
                Xv, Yv = val_arrays[candidate.history]
                prediction = candidate.predict(candidate.transform(flat(Xv)), cfg["horizons"])
                stroke_losses = episode_mse(Yv, prediction.reshape(Yv.shape))
                _, losses = writer_balanced_losses(stroke_losses, validation_writers)
                candidate.validation_mse = losses.mean(axis=0)
            selected = {}
            for family in sorted({c.family for c in candidates}):
                c = min(
                    [c for c in candidates if c.family == family],
                    key=lambda c: (float(np.mean(c.validation_mse)), c.n_features, c.name),
                )
                selected[family] = c.name
            families.append((seed, coordinate, candidates, selected))
            frozen.append({
                "seed": seed, "coordinate": coordinate, "n_candidates": len(candidates),
                "n_training_strokes": len(chosen_training),
                "n_validation_strokes": len(validation_records),
                "training_record_ids": [record_id(r) for r in chosen_training],
                "selected_by_family": selected,
                "candidates": [candidate_description(c) for c in candidates],
                "fit_seconds": time.perf_counter() - run_start,
            })
            print(f"FROZEN seed={seed} coordinate={coordinate} K={len(candidates)} seconds={time.perf_counter()-run_start:.2f}", flush=True)
    write_json(output / "frozen_candidates.json", frozen)
    n_tests = sum(len(cs) * len(cfg["horizons"]) for _, _, cs, _ in families)
    tail_delta = cfg["delta"] / n_tests

    # Phase 2 scores the original, writer-disjoint 14-writer test cohort.
    all_test_records = load_prepared_strokes(prepared_path, "tes", first_only=True)
    if set(record["writer"] for record in all_test_records) != set(range(1, 15)):
        raise ValueError("Original test file does not contain its documented 14 writer-like groups")
    test_records = select_records(
        all_test_records,
        set(range(1, 15)), cfg["episode_length"],
    )
    test_writers = np.asarray([r["writer"] for r in test_records])
    unique_writers = np.unique(test_writers)
    if len(unique_writers) < 2:
        raise ValueError("The length restriction leaves fewer than two eligible writers")
    excluded_writers = sorted(set(range(1, 15)) - set(unique_writers.tolist()))
    bound_width = cfg["scalar_squared_loss_bound"] * np.sqrt(
        np.log(1 / tail_delta) / (2 * len(unique_writers))
    )
    results, rows, selected_rows = [], [], []
    for seed, coordinate, candidates, selected in families:
        test = episodes_from_records(test_records, cfg["episode_length"], coordinate)
        arrays = {
            history: histories_targets(
                test, history, cfg["horizons"], cfg["origin_stride"], first_origin
            )
            for history in cfg["histories"]
        }
        for candidate in candidates:
            X, Y = arrays[candidate.history]
            prediction = candidate.predict(candidate.transform(flat(X)), cfg["horizons"])
            if np.any((prediction < -1 - 1e-10) | (prediction > 1 + 1e-10)):
                raise ValueError(f"Unbounded prediction from {candidate.name}")
            stroke_losses = episode_mse(Y, prediction.reshape(Y.shape))
            writer_ids, writer_losses = writer_balanced_losses(stroke_losses, test_writers)
            mean = writer_losses.mean(axis=0)
            upper = np.minimum(cfg["scalar_squared_loss_bound"], mean + bound_width)
            chosen = selected[candidate.family] == candidate.name
            result = {
                "seed": seed, "coordinate": coordinate, **candidate_description(candidate),
                "selected_before_test": chosen,
                "cohort_writer_mean_mse": mean.tolist(),
                "cohort_writer_mean_rmse": np.sqrt(mean).tolist(),
                "conditional_iid_writer_mse_upper": upper.tolist(),
                "conditional_iid_writer_rmse_upper": np.sqrt(upper).tolist(),
                "tail_delta": tail_delta, "n_writers": len(writer_ids),
            }
            results.append(result)
            for idx, writer in enumerate(writer_ids):
                for hidx, horizon in enumerate(cfg["horizons"]):
                    rows.append({
                        "seed": seed, "coordinate": coordinate, "candidate": candidate.name,
                        "family": candidate.family, "selected_before_test": chosen,
                        "writer": int(writer), "horizon": horizon,
                        "n_eligible_strokes": int(np.sum(test_writers == writer)),
                        "writer_mean_squared_loss": float(writer_losses[idx, hidx]),
                    })
            if chosen:
                for hidx, horizon in enumerate(cfg["horizons"]):
                    selected_rows.append({
                        "seed": seed, "coordinate": coordinate, "family": candidate.family,
                        "candidate": candidate.name, "horizon": horizon,
                        "history": candidate.history, "n_features": candidate.n_features,
                        "validation_mse": float(candidate.validation_mse[hidx]),
                        "cohort_mse": float(mean[hidx]), "cohort_rmse": float(np.sqrt(mean[hidx])),
                        "conditional_iid_writer_mse_upper": float(upper[hidx]),
                        "conditional_iid_writer_rmse_upper": float(np.sqrt(upper[hidx])),
                    })
        print(f"SCORED seed={seed} coordinate={coordinate} candidates={len(candidates)}", flush=True)
    for filename, records in [("writer_losses.csv", rows), ("selected_results.csv", selected_rows)]:
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    record = {
        "configuration": cfg, "n_upper_bound_statements": n_tests,
        "per_statement_delta": tail_delta, "hoeffding_mse_addition": float(bound_width),
        "n_held_out_writers": len(unique_writers), "n_held_out_eligible_digits": len(test_records),
        "n_original_held_out_writers": 14,
        "held_out_writer_ids_with_no_eligible_digits": excluded_writers,
        "eligibility_caveat": "The writer risk is conditional on at least one first stroke of the predeclared minimum length; no loss is invented for a writer with zero eligible strokes.",
        "held_out_record_ids": [record_id(r) for r in test_records],
        "n_validation_eligible_digits": len(validation_records),
        "origins": list(range(first_origin, cfg["episode_length"]-max(cfg["horizons"]),cfg["origin_stride"])),
        "results": results, "elapsed_seconds": time.perf_counter()-started,
    }
    write_json(output / "results.json", record)
    summary = summarise_results(record)
    write_json(output / "summary.json", summary)
    make_latex(summary, output)
    return summary


def summarise_results(record):
    cfg = record["configuration"]
    selected = [row for row in record["results"] if row["selected_before_test"]]
    family_rows = []
    for family in FAMILY_LABELS:
        subset = [row for row in selected if row["family"] == family]
        seed_rmse = []
        seed_bounds = []
        for seed in cfg["seeds"]:
            runs = [row for row in subset if row["seed"] == seed]
            seed_rmse.append(np.sqrt(np.mean([r["cohort_writer_mean_mse"] for r in runs], axis=0)))
            seed_bounds.append(np.sqrt(np.mean([r["conditional_iid_writer_mse_upper"] for r in runs], axis=0)))
        seed_rmse, seed_bounds = np.asarray(seed_rmse), np.asarray(seed_bounds)
        family_rows.append({
            "family": family, "label": FAMILY_LABELS[family],
            "mean_coordinate_pooled_rmse_over_seeds": seed_rmse.mean(axis=0).tolist(),
            "sd_coordinate_pooled_rmse_over_seeds": seed_rmse.std(axis=0, ddof=1).tolist(),
            "mean_selected_features": float(np.mean([row["n_features"] for row in subset])),
            "mean_conditional_iid_writer_rmse_upper": seed_bounds.mean(axis=0).tolist(),
            "selected_names": [row["name"] for row in subset],
        })
    return {
        "families": family_rows,
        "horizons": cfg["horizons"], "seeds": cfg["seeds"],
        "n_held_out_writers": record["n_held_out_writers"],
        "held_out_writer_ids_with_no_eligible_digits": record["held_out_writer_ids_with_no_eligible_digits"],
        "n_held_out_eligible_digits": record["n_held_out_eligible_digits"],
        "n_validation_eligible_digits": record["n_validation_eligible_digits"],
        "n_candidates_per_coordinate_seed": len(record["results"])//(len(cfg["seeds"])*len(cfg["coordinates"])),
        "n_upper_bound_statements": record["n_upper_bound_statements"],
        "per_statement_delta": record["per_statement_delta"],
        "hoeffding_mse_addition": record["hoeffding_mse_addition"],
        "delta": cfg["delta"], "elapsed_seconds": record["elapsed_seconds"],
        "interpretation": "Descriptive writer-disjoint cohort prediction; risk UCBs require an unverified iid-writer assumption. Seed SD measures training variation, not independent test-cohort uncertainty.",
    }


def make_latex(summary, output):
    table = [
        "% Automatically generated by experiments/run_pendigits.py; do not hand-edit numbers.",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Family & Features & $h=1$ & $h=2$ & $h=4$ \\",
        r"\midrule",
    ]
    for row in summary["families"]:
        values = [
            f"${mean:.4f}\\pm{sd:.4f}$" for mean, sd in zip(
                row["mean_coordinate_pooled_rmse_over_seeds"],
                row["sd_coordinate_pooled_rmse_over_seeds"],
            )
        ]
        table.append(f"{row['label']} & {row['mean_selected_features']:.1f} & " + " & ".join(values) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output / "pendigits_table.tex").write_text("\n".join(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pendigits_v1.json"))
    parser.add_argument("--prepared", type=Path, default=Path("data_cache/pendigits/pendigits_original.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/pendigits_v1"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite a nonempty run directory: {args.output}")
    try:
        summary = run(args.config, args.prepared, args.output)
    except Exception:
        if args.output.exists():
            (args.output / "FAILED.txt").write_text(traceback.format_exc())
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
