#!/usr/bin/env python3
"""Independently rebuild the central study's statistics from saved selected losses.

This is new analysis code, not the original verify_saved_analysis.py. It does
not reconstruct forecasts, fit models, or validate the native feature engine.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "KAHM-SCORE-20261102-v1"
ORIGINAL_SOURCE_MANIFEST = "af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28"
LAWS = ("linear_ar", "nonlinear_ar", "van_der_pol", "duffing")
VIEWS = ("independent", "shared")
SPLITS = ("C", "D")
DECODERS = ("linear", "nonlinear")
PROCEDURES = tuple(f"{rule}__{encoding}__{decoder}"
                   for rule in ("K", "A", "R")
                   for encoding in ("mem", "norm", "local", "score")
                   for decoder in DECODERS) + tuple(f"raw__raw__{d}" for d in DECODERS)
N = 64
CAPS = (1, 2, 4, 8, 12)
ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BANDWIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0)
HORIZONS = (1, 4, 12)
FAMILY_SIZE = 40
FAMILY_ALPHA = 0.05
# Tolerances concern reconstructed summary arithmetic, not forecast accuracy.
RTOL = 2e-10
ATOL = 2e-12
META_FIELDS = ("head_id", "cap", "alpha", "bandwidth_multiplier", "anchor_cap",
               "feature_width", "centered_rank")
LOSS_FIELDS = ("smse",) + tuple(f"{kind}_h{h}" for h in HORIZONS for kind in ("mse", "smse"))
SELECTED_FIELDS = ("task", "law", "repetition", "split", "view", "procedure", "head_id",
                   "smse", "cap", "alpha", "bandwidth_multiplier", "anchor_cap",
                   "feature_width", "centered_rank") + tuple(
    f"{kind}_h{h}" for h in HORIZONS for kind in ("mse", "smse"))
SUMMARY_FIELDS = ("split", "law", "view", "procedure", "n", "mean_smse", "sd_smse")
PAIRED_FIELDS = ("contrast_id", "repetition", "candidate_loss", "reference_loss", "difference")
PRIMARY_FIELDS = ("id", "law", "decoder", "candidate", "reference", "view", "question", "n",
                  "mean_difference", "paired_sd", "standard_error", "critical_t", "lower", "upper",
                  "halfwidth", "classification", "candidate_mean_smse", "reference_mean_smse",
                  "relative_gain_percent", "candidate_lower_repetitions", "candidate_higher_repetitions",
                  "paired_median", "paired_min", "paired_max", "leave_one_out_mean_min",
                  "leave_one_out_mean_max")
CONTRASTS = (
    ("local_vs_mem", "K__local", "K__mem", "Local-position enrichment"),
    ("norm_vs_mem", "K__norm", "K__mem", "Normalizer enrichment"),
    ("K_vs_A_local", "K__local", "A__local", "Matched affine control"),
    ("K_vs_R_local", "K__local", "R__local", "Matched positive-kernel control"),
    ("K_local_vs_raw", "K__local", "raw__raw", "Raw-input control"),
)
EXPECTED_EVIDENCE = {
    "COMPUTATIONAL_AUDIT.json": (46119, "23de74efa9ed966cd360495dfdb223953489a0ce63a2b0918f50fac64117602e"),
    "FREEZE.json": (547, "f2d7d31103acef84c5df3fa7db2d24670ea68d1b2121d56e7e1c8340eb960aa1"),
    "RESULTS_RELEASE.json": (936, "d370d07ab2c6359cfba7630cea47e717c70c553da1fb6c69e6246afc434da5dc"),
    "all_procedure_summary.csv": (35209, "bf000deeb52ccac69199e67302553fbcf1b6491cf209e4b67703b6492c2ae426"),
    "analysis_receipt.json": (982, "867aabef1ceee5b5628b1df07d9d353914402032a87f06311666990332648ecb"),
    "horizon_summary.csv": (53121, "e79084866210457c9f533782ab17f009e51ce911873baddd96e8b8a7289f77b0"),
    "numerical_diagnostics.json": (378169, "ec5fd0c1098da2855268b16e6f8a434bb708a87b11e84b34eb46b1c6866f5003"),
    "primary_contrasts.csv": (19164, "2e4c632c030fa69dc0b0f61c09f670633d5570b703c675d42d369e6ea4127818"),
    "primary_paired_repetitions.csv": (256811, "6dd86a3f5031358d85332a1367e8bbb6809b5903cf292eb5cd0315d35f07c9b3"),
    "protocol.json": (4680, "9808accf661f35db239c486dd72a8a279e19830c1a3d87c5f92add4e040e67c1"),
    "resource_summary.csv": (454, "c5d78feea79b823735fb9df200ca3af58ca4c06c820e899a96e6ab501fc5513d"),
    "results_summary.json": (60882, "f3155aad072ac448f7c6de18be1ed856688a3235d8377628d6c6a9dd2cc06fb6"),
    "same_bank_normalizer_score.csv": (2647, "5025441c0eceba0a51f82c204e9ec567880aaae3d5cfc8eb780839d4387d4e25"),
    "saved_analysis_verification.json": (3525, "901e07dc158795472e00f970d563b17784920074d485a90258eee73383dbeca3"),
    "secondary_selection_comparisons.csv": (10293, "d81fe59c248062e74c0e252eeac6716545896b1d42fba36750334b07a1ce0a8b"),
    "selected_losses.csv": (6003675, "f2cf659e19a58aadc9ff74aeefaff15568f19063367e1fb0f82263d90b3ea81b"),
    "selection_counts.csv": (118166, "41e5e0de18d8752bd11d4272e827e5876cfde5ff96dd7658e4c9c563eac445e4"),
}
COMPRESSED_LOSSES = (1644630, "9ac64cd11e165c7cffcabfec892b3f5f9ccb2756aeae9134a08d0c550687930a")
SUPPORT_FILES = {"README.md", "requirements.txt", "check_saved_losses.py", "tests/test_check_saved_losses.py"}


class CheckError(ValueError):
    """A source-identity, roster, or reconstructed-statistic check failed."""


def require(condition, message):
    if not condition:
        raise CheckError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def physical_path(name):
    return "evidence/" + name + (".gz" if name == "selected_losses.csv" else "")


def load_verified_bundle(root=ROOT):
    """Read one hash-checked in-memory snapshot; never extract the compressed CSV."""
    root = Path(root)
    manifest_bytes = (root / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest["schema"] == "central-saved-loss-bundle-v1", "Unexpected bundle manifest")
    require(manifest["protocol_id"] == PROTOCOL_ID, "Wrong manifest study identity")
    required_paths = {physical_path(n) for n in EXPECTED_EVIDENCE} | SUPPORT_FILES
    require(set(manifest["files"]) == required_paths, "Manifest file roster differs")
    original = manifest["original_evidence"]
    require(set(original) == set(EXPECTED_EVIDENCE), "Original evidence roster differs")
    payloads = {}
    for name in sorted(required_paths):
        data = (root / name).read_bytes()
        entry = manifest["files"][name]
        require(len(data) == entry["bytes"] and sha256(data) == entry["sha256"],
                f"File identity mismatch: {name}")
        payloads[name] = data
    evidence = {}
    for name, (size, digest) in EXPECTED_EVIDENCE.items():
        entry = original[name]
        require(entry["physical_path"] == physical_path(name)
                and entry["bytes"] == size and entry["sha256"] == digest,
                f"Original logical identity mismatch: {name}")
        data = payloads[physical_path(name)]
        if name == "selected_losses.csv":
            require((len(data), sha256(data)) == COMPRESSED_LOSSES,
                    "Compressed selected-loss identity mismatch")
            require(entry["compression"] == "gzip", "Wrong selected-loss compression")
            data = gzip.decompress(data)
        else:
            require(entry["compression"] is None, f"Unexpected compression: {name}")
        require(len(data) == size and sha256(data) == digest, f"Evidence identity mismatch: {name}")
        evidence[name] = data
    return evidence, sha256(manifest_bytes)


def csv_rows(data, fields, name):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    require(tuple(reader.fieldnames or ()) == tuple(fields), f"Unexpected columns: {name}")
    rows = list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows),
            f"Malformed CSV row: {name}")
    return rows


def finite(value, label):
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise CheckError(f"Invalid number: {label}") from error
    require(math.isfinite(number), f"Nonfinite number: {label}")
    return number


def integer(value, label):
    try:
        number = int(value)
    except (ValueError, TypeError) as error:
        raise CheckError(f"Invalid integer: {label}") from error
    require(str(number) == str(value), f"Noncanonical integer: {label}")
    return number


def close(actual, expected, label, differences=None):
    expected = finite(expected, label)
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
            f"Arithmetic mismatch: {label}: {actual!r} versus {expected!r}")
    if differences is not None:
        differences.append(abs(actual - expected))


def validate_protocol(protocol):
    require(protocol["protocol_id"] == PROTOCOL_ID, "Wrong study protocol")
    require(protocol["laws"] == list(LAWS) and protocol["repetitions"] == list(range(N)),
            "Unexpected law/repetition design")
    require(protocol["horizons"] == list(HORIZONS), "Unexpected forecast horizons")
    require(protocol["primary_selection"] == "independent"
            and protocol["secondary_selection"] == "shared cap selected by A__local__linear on B"
            and protocol["post_selection_refit"] is False, "Unexpected selection protocol")
    model = protocol["model"]
    require(model["caps"] == list(CAPS) and model["alphas"] == list(ALPHAS)
            and model["bandwidth_multipliers"] == list(BANDWIDTHS), "Unexpected candidate grid")
    inference = protocol["inference"]
    require(inference["family_size"] == FAMILY_SIZE and inference["family_alpha"] == FAMILY_ALPHA
            and inference["paired_units_per_contrast"] == N and inference["degrees_of_freedom"] == N - 1
            and inference["pool_laws_for_primary"] is False
            and inference["practical_margin"] is None
            and inference["noninferiority_or_equivalence"] is False, "Unexpected inference family")


def validate_selected(rows):
    """Check the complete selected-record roster, not the unavailable B score grid."""
    expected = {(law, rep, split, view, proc) for law in LAWS for rep in range(N)
                for split in SPLITS for view in VIEWS for proc in PROCEDURES}
    require(len(rows) == len(expected) == 26624, "Selected-loss row count differs")
    indexed = {}
    for row in rows:
        rep = integer(row["repetition"], "repetition")
        key = (row["law"], rep, row["split"], row["view"], row["procedure"])
        require(key in expected, f"Unexpected selected-loss row: {key}")
        require(key not in indexed, f"Duplicate selected-loss row: {key}")
        require(row["task"] == f"{row['law']}__r{rep:03d}", f"Task identity mismatch: {key}")
        require(integer(row["head_id"], "head_id") >= 0, f"Negative fitted-model identity: {key}")
        require(finite(row["alpha"], "alpha") in ALPHAS, f"Penalty outside protocol grid: {key}")
        require(integer(row["anchor_cap"], "anchor_cap") in CAPS, f"Invalid shared cap: {key}")
        if row["procedure"].startswith("raw__"):
            require(row["cap"] == "", f"Raw histories must have no dimension cap: {key}")
        else:
            require(integer(row["cap"], "cap") in CAPS, f"Cap outside protocol grid: {key}")
        if row["procedure"].endswith("__linear"):
            require(row["bandwidth_multiplier"] == "", f"Linear predictor has a bandwidth: {key}")
        else:
            require(finite(row["bandwidth_multiplier"], "bandwidth") in BANDWIDTHS,
                    f"Bandwidth outside protocol grid: {key}")
        width, rank = integer(row["feature_width"], "feature_width"), integer(row["centered_rank"], "rank")
        require(width > 0 and 0 <= rank <= min(width, 511), f"Invalid stored width/rank: {key}")
        values = {field: finite(row[field], f"{key}/{field}") for field in LOSS_FIELDS}
        require(all(v >= 0 for v in values.values()), f"Negative squared loss: {key}")
        close(math.fsum(values[f"smse_h{h}"] for h in HORIZONS) / 3, values["smse"],
              f"horizon-average SMSE/{key}")
        indexed[key] = row
    require(set(indexed) == expected, "Incomplete selected-loss roster")
    for key, row in indexed.items():
        law, rep, split, view, proc = key
        other = indexed[(law, rep, "D" if split == "C" else "C", view, proc)]
        require(all(row[f] == other[f] for f in META_FIELDS), f"C/D selection identity differs: {key}")
        anchor = indexed[(law, rep, split, "independent", "A__local__linear")]["cap"]
        require(row["anchor_cap"] == anchor, f"Recorded anchor cap differs: {key}")
        if view == "shared":
            require(proc.startswith("raw__") or row["cap"] == anchor, f"Shared cap differs: {key}")
            independent = indexed[(law, rep, split, "independent", proc)]
            if proc.startswith("raw__") or independent["cap"] == anchor:
                require(all(row[f] == independent[f] for f in META_FIELDS + LOSS_FIELDS),
                        f"Same-cap/raw selection differs between views: {key}")
    return indexed


def mean_sd(values):
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return mean, sd


def rebuild_summaries(indexed):
    rebuilt = []
    for split in SPLITS:
        for law in LAWS:
            for view in VIEWS:
                for proc in PROCEDURES:
                    values = [float(indexed[(law, rep, split, view, proc)]["smse"]) for rep in range(N)]
                    mean, sd = mean_sd(values)
                    rebuilt.append(dict(split=split, law=law, view=view, procedure=proc,
                                        n=N, mean_smse=mean, sd_smse=sd))
    return rebuilt


def classify(lower, upper):
    return "candidate_lower_mean_loss" if upper < 0 else (
        "candidate_higher_mean_loss" if lower > 0 else "unresolved")


def rebuild_primary(indexed):
    from scipy.stats import t
    critical = float(t.ppf(1 - FAMILY_ALPHA / (2 * FAMILY_SIZE), N - 1))
    pairs, primary = [], []
    for law in LAWS:
        for decoder in DECODERS:
            for suffix, candidate_base, reference_base, question in CONTRASTS:
                candidate, reference = f"{candidate_base}__{decoder}", f"{reference_base}__{decoder}"
                identity = f"{law}__{decoder}__{suffix}"
                a = [float(indexed[(law, rep, "D", "independent", candidate)]["smse"]) for rep in range(N)]
                b = [float(indexed[(law, rep, "D", "independent", reference)]["smse"]) for rep in range(N)]
                differences = [x - y for x, y in zip(a, b)]
                for rep, (x, y, d) in enumerate(zip(a, b, differences)):
                    pairs.append(dict(contrast_id=identity, repetition=rep,
                                      candidate_loss=x, reference_loss=y, difference=d))
                mean, sd = mean_sd(differences)
                # This pinned release has no zero-variance primary contrast.
                # Its protocol permits no Student inference if such a case occurs.
                require(sd > 0, f"Zero variance: no Student interval is permitted for {identity}")
                se = sd / math.sqrt(N)
                half = critical * se
                lower, upper = mean - half, mean + half
                mean_a, mean_b = math.fsum(a) / N, math.fsum(b) / N
                require(mean_b > 0, f"Relative gain has a nonpositive denominator: {identity}")
                total = math.fsum(differences)
                loo = [(total - d) / (N - 1) for d in differences]
                primary.append(dict(
                    id=identity, law=law, decoder=decoder, candidate=candidate, reference=reference,
                    view="independent", question=question, n=N, mean_difference=mean, paired_sd=sd,
                    standard_error=se, critical_t=critical, lower=lower, upper=upper, halfwidth=half,
                    classification=classify(lower, upper), candidate_mean_smse=mean_a,
                    reference_mean_smse=mean_b, relative_gain_percent=100 * (1 - mean_a / mean_b),
                    candidate_lower_repetitions=sum(d < 0 for d in differences),
                    candidate_higher_repetitions=sum(d > 0 for d in differences),
                    paired_median=statistics.median(differences), paired_min=min(differences),
                    paired_max=max(differences), leave_one_out_mean_min=min(loo),
                    leave_one_out_mean_max=max(loo)))
    return pairs, primary, critical


def compare_table(rebuilt, published, key_fields, numeric_fields, integer_fields, label):
    """Compare the full keyed roster and every published field, regardless of row order."""
    require(len(rebuilt) == len(published), f"Wrong row count: {label}")
    by_key = {}
    for row in published:
        key = tuple(str(row[field]) for field in key_fields)
        require(key not in by_key, f"Duplicate published row: {label}/{key}")
        by_key[key] = row
    differences = []
    for actual in rebuilt:
        key = tuple(str(actual[field]) for field in key_fields)
        require(key in by_key, f"Missing published row: {label}/{key}")
        expected = by_key.pop(key)
        require(set(actual) == set(expected), f"Field roster differs: {label}/{key}")
        for field, value in actual.items():
            field_label = f"{label}/{key}/{field}"
            if field in numeric_fields:
                close(value, expected[field], field_label, differences)
            elif field in integer_fields:
                require(value == integer(expected[field], field_label), f"Count mismatch: {field_label}")
            else:
                require(str(value) == expected[field], f"Label/decision mismatch: {field_label}")
    require(not by_key, f"Unexpected published rows: {label}")
    return max(differences, default=0.0)


def run_checks(root=ROOT):
    evidence, manifest_hash = load_verified_bundle(root)
    protocol = json.loads(evidence["protocol.json"])
    validate_protocol(protocol)
    rows = csv_rows(evidence["selected_losses.csv"], SELECTED_FIELDS, "selected losses")
    indexed = validate_selected(rows)
    summaries = rebuild_summaries(indexed)
    pairs, primary, critical = rebuild_primary(indexed)
    errors = {}
    errors["procedure_summaries"] = compare_table(
        summaries, csv_rows(evidence["all_procedure_summary.csv"], SUMMARY_FIELDS, "summaries"),
        ("split", "law", "view", "procedure"), {"mean_smse", "sd_smse"}, {"n"}, "summaries")
    errors["paired_differences"] = compare_table(
        pairs, csv_rows(evidence["primary_paired_repetitions.csv"], PAIRED_FIELDS, "paired rows"),
        ("contrast_id", "repetition"), {"candidate_loss", "reference_loss", "difference"},
        {"repetition"}, "paired rows")
    text_fields = {"id", "law", "decoder", "candidate", "reference", "view", "question", "classification"}
    int_fields = {"n", "candidate_lower_repetitions", "candidate_higher_repetitions"}
    errors["primary_contrasts"] = compare_table(
        primary, csv_rows(evidence["primary_contrasts.csv"], PRIMARY_FIELDS, "primary contrasts"),
        ("id",), set(PRIMARY_FIELDS) - text_fields - int_fields, int_fields, "primary contrasts")
    historical = json.loads(evidence["saved_analysis_verification.json"])
    receipt = json.loads(evidence["analysis_receipt.json"])
    release = json.loads(evidence["RESULTS_RELEASE.json"])
    for record in (historical, receipt):
        close(critical, record["critical_t"], "recorded Student critical value")
    for original_path, digest in historical["input_hashes"].items():
        name = Path(original_path).name
        require(name in EXPECTED_EVIDENCE and EXPECTED_EVIDENCE[name][1] == digest,
                f"Saved-analysis input binding differs: {original_path}")
    for original_path, digest in release["files"].items():
        name = Path(original_path).name
        if name in EXPECTED_EVIDENCE:
            require(EXPECTED_EVIDENCE[name][1] == digest, f"Release binding differs: {original_path}")
    require(historical["source_manifest_sha256"] == ORIGINAL_SOURCE_MANIFEST,
            "Historical experimental source identity differs")
    decisions = {row["id"]: row["classification"] for row in primary}
    require(decisions == historical["decisions"], "Historical decision map differs")
    import scipy
    report = {
        "status": "PASS", "checker": "independent-central-saved-loss-checker-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": PROTOCOL_ID,
        "scope": "Independent arithmetic reconstruction from saved selected losses; no forecast replay or model fitting.",
        "environment": {"python": platform.python_version(), "scipy": scipy.__version__},
        "bundle_manifest_sha256": manifest_hash,
        "original_evidence_records_hash_checked": len(evidence),
        "logical_selected_losses_sha256": EXPECTED_EVIDENCE["selected_losses.csv"][1],
        "compressed_selected_losses_sha256": COMPRESSED_LOSSES[1],
        "selected_loss_rows": len(rows), "tasks": len(LAWS) * N, "procedures": len(PROCEDURES),
        "views": list(VIEWS), "splits": list(SPLITS), "procedure_summary_rows_rebuilt": len(summaries),
        "paired_difference_rows_rebuilt": len(pairs), "primary_comparisons_rebuilt": len(primary),
        "critical_t": critical, "decision_counts": dict(sorted(Counter(decisions.values()).items())),
        "decisions": decisions, "maximum_absolute_arithmetic_difference": errors,
        "arithmetic_tolerances": {"relative": RTOL, "absolute": ATOL},
        "selection_checks": ["Complete unique law/repetition/view/split/procedure roster",
            "C/D fitted-model and setting identity", "Shared cap from independent affine-local linear choice",
            "Raw-control and same-cap identity across selection views", "Declared cap/penalty/bandwidth grids",
            "Each stored SMSE equals its three-horizon average"],
        "not_recomputed": ["Validation candidate scores or optimality of the recorded selections",
            "Predictions, target variances, features, clustering, ODE integration, or model fits",
            "The original computational audit and native numerical diagnostics",
            "Auxiliary horizon/selection/resource/secondary tables beyond their exact file identities",
            "Original execution chronology or external scientific replication"],
        "sampling_scope": "64 paired repetition differences per contrast; no pooling of laws, windows, horizons or selection views as independent units.",
        "confidence_scope": "Forty Bonferroni-adjusted paired-Student intervals; approximate coverage, not distribution-free or an equivalence claim.",
        "original_source_manifest_sha256": ORIGINAL_SOURCE_MANIFEST,
        "original_source_manifest_included": False,
    }
    return report, {"rebuilt_all_procedure_summary.csv": (SUMMARY_FIELDS, summaries),
                    "rebuilt_primary_paired_repetitions.csv": (PAIRED_FIELDS, pairs),
                    "rebuilt_primary_contrasts.csv": (PRIMARY_FIELDS, primary)}


def write_outputs(output, report, tables):
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output),
            "Output directory must be separate from this evidence bundle")
    require(not output.exists(), "Choose a fresh output directory; existing results are not overwritten")
    output.mkdir(parents=True)
    for name, (fields, rows) in tables.items():
        with (output / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Fresh directory for the three rebuilt CSVs and verification.json; otherwise print report only")
    args = parser.parse_args()
    try:
        report, tables = run_checks()
        if args.output:
            write_outputs(args.output, report, tables)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (CheckError, OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Saved-loss check failed: {error}\n")


if __name__ == "__main__":
    main()
