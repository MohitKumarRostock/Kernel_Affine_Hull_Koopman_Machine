#!/usr/bin/env python3
"""Explain a saved comparison from the forecasting feature-augmentation study.

This read-only command displays summary-statistic evidence from the specified
40-comparison family. It does not fit models, reconstruct repetition losses,
or recompute confidence intervals. Use verify_primary_summary.py
checker for the full summary-arithmetic checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = Path("extensions/score_enrichment_20261102")
DEFAULT_CSV = REPOSITORY / STUDY / "primary_contrasts.csv"
PUBLISHED_SHA256 = "2e4c632c030fa69dc0b0f61c09f670633d5570b703c675d42d369e6ea4127818"
CHECKER_COMMAND = f"python3 {STUDY.as_posix()}/verify_primary_summary.py"
EXAMPLES = (
    "van_der_pol__nonlinear__local_vs_mem",
    "duffing__nonlinear__K_local_vs_raw",
    "nonlinear_ar__nonlinear__local_vs_mem",
)
LAWS = {
    "linear_ar": "Linear autoregression",
    "nonlinear_ar": "Nonlinear autoregression",
    "van_der_pol": "Van der Pol oscillator",
    "duffing": "Duffing oscillator",
}
PREDICTORS = {
    "linear": "Linear ridge regression",
    "nonlinear": (
        "Ridge regression on the representation concatenated with Gaussian-kernel Nyström features "
        "from 64 training landmarks"
    ),
}
RECONSTRUCTIONS = {
    "K": "KAHM",
    "A": "Affine projection",
    "R": "Kernel-weighted averaging",
}
ENCODINGS = {
    "mem": "memberships",
    "norm": "memberships with log normalization constants",
    "local": "memberships with membership-weighted local coordinates",
    "score": "full reconstruction scores",
}
OUTCOMES = {
    "candidate_lower_mean_loss": "LOWER CANDIDATE MEAN LOSS",
    "candidate_higher_mean_loss": "HIGHER CANDIDATE MEAN LOSS",
    "unresolved": "UNRESOLVED",
}
REQUIRED_FIELDS = {
    "id", "law", "decoder", "candidate", "reference", "view", "question",
    "n", "candidate_mean_smse", "reference_mean_smse", "mean_difference",
    "lower", "upper", "relative_gain_percent", "classification",
}


def representation(identifier: str, predictor: str) -> str:
    parts = identifier.split("__")
    if len(parts) != 3 or parts[2] != predictor:
        raise ValueError(f"Unexpected procedure identifier: {identifier}")
    method, encoding, _ = parts
    if (method, encoding) == ("raw", "raw"):
        return "Raw histories"
    if method not in RECONSTRUCTIONS or encoding not in ENCODINGS:
        raise ValueError(f"Unexpected representation identifier: {identifier}")
    return f"{RECONSTRUCTIONS[method]} {ENCODINGS[encoding]}"


def read_published_records(path: Path) -> list[dict[str, str]]:
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != PUBLISHED_SHA256:
        raise ValueError(
            "CSV identity does not match the specified primary summary. "
            "Use the supplied primary_contrasts.csv; this command's protocol "
            "interpretation is specific to that 40-comparison family."
        )
    reader = csv.DictReader(io.StringIO(contents.decode("utf-8")))
    if not REQUIRED_FIELDS.issubset(reader.fieldnames or []):
        raise ValueError("The published summary is missing required fields.")
    rows = list(reader)
    if len(rows) != 40 or len({row["id"] for row in rows}) != 40:
        raise ValueError("Expected the 40 distinct published comparisons.")
    for row in rows:
        if row["law"] not in LAWS or row["decoder"] not in PREDICTORS:
            raise ValueError(f"Unexpected data-generating process or predictor in {row['id']}")
        if row["view"] != "independent" or int(row["n"]) != 64:
            raise ValueError(f"Unexpected selection rule or sample size in {row['id']}")
        for field in ("candidate_mean_smse", "reference_mean_smse",
                      "mean_difference", "lower", "upper", "relative_gain_percent"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Nonfinite {field} in {row['id']}")
        lo, hi = float(row["lower"]), float(row["upper"])
        outcome = ("candidate_lower_mean_loss" if hi < 0 else
                   "candidate_higher_mean_loss" if lo > 0 else "unresolved")
        if lo > hi or row["classification"] != outcome:
            raise ValueError(f"Inconsistent interval classification in {row['id']}")
        representation(row["candidate"], row["decoder"])
        representation(row["reference"], row["decoder"])
    return rows


def evidence_context(path: Path) -> dict:
    return {
        "source_csv": str(path.resolve()),
        "published_sha256": PUBLISHED_SHA256,
        "file_identity_verified": True,
        "evidence_level": "Saved primary summary statistics.",
        "protocol_id": "KAHM-SCORE-20261102-v1",
        "family_size": 40,
        "paired_repetitions_per_comparison": 64,
        "smse": (
            "Mean squared forecast error at each horizon, divided by that horizon's "
            "retained-training target variance (floor 1e-12), then averaged over "
            "horizons 1, 4, and 12. Lower is better."
        ),
        "difference": "Candidate SMSE minus reference SMSE; negative favors the candidate.",
        "relative_gain": (
            "100 * (reference mean SMSE - candidate mean SMSE) / reference mean SMSE. "
            "This is a ratio of means, not the mean of repetition-wise percentages."
        ),
        "selection": (
            "The stored view 'independent' means separately selected settings using "
            "the same B validation data within a repetition. It does not mean "
            "independent validation samples. Local dimension caps and predictor "
            "settings may differ; raw histories have no cap. These comparisons "
            "assess complete selected procedures."
        ),
        "uncertainty": (
            "The published intervals use a two-sided paired-Student calculation "
            "with Bonferroni adjustment over 40 comparisons. The target is 95% "
            "simultaneous coverage when the marginal Student approximation is "
            "adequate. Paired units are repetitions, not overlapping windows. "
            "No practical-improvement, noninferiority, or equivalence margin is assigned."
        ),
        "scope": (
            "The command reads the published summaries and checks file identity "
            "and internal labels. It does not fit models, reconstruct saved losses, "
            "or recompute interval arithmetic."
        ),
        "full_summary_check": CHECKER_COMMAND + " (requires SciPy)",
    }


def explain(row: dict[str, str]) -> dict:
    classification = row["classification"]
    if classification == "candidate_lower_mean_loss":
        interpretation = (
            "The adjusted interval is wholly below zero. This comparison favors "
            "lower mean forecast loss for the candidate under the stated data-generating process and "
            "selected procedures."
        )
    elif classification == "candidate_higher_mean_loss":
        interpretation = (
            "The adjusted interval is wholly above zero. This comparison favors "
            "the reference: the candidate has higher mean forecast loss under "
            "the stated data-generating process and selected procedures."
        )
    else:
        difference = float(row["mean_difference"])
        point_order = ("The point means favor the candidate" if difference < 0 else
                       "The point means favor the reference" if difference > 0 else
                       "The point means are equal")
        interpretation = (
            f"{point_order}, but the adjusted interval includes zero, so the "
            "mean-loss direction remains unresolved. An unresolved result does "
            "not establish equivalence or absence of an effect."
        )
    return {
        "id": row["id"],
        "question": row["question"],
        "law": LAWS[row["law"]],
        "predictor": PREDICTORS[row["decoder"]],
        "candidate": {
            "id": row["candidate"],
            "representation": representation(row["candidate"], row["decoder"]),
            "mean_smse": float(row["candidate_mean_smse"]),
        },
        "reference": {
            "id": row["reference"],
            "representation": representation(row["reference"], row["decoder"]),
            "mean_smse": float(row["reference_mean_smse"]),
        },
        "mean_difference": float(row["mean_difference"]),
        "adjusted_interval": [float(row["lower"]), float(row["upper"])],
        "relative_gain_percent": float(row["relative_gain_percent"]),
        "classification": classification,
        "interpretation": interpretation,
        "published_record": row,
    }


def print_explanation(item: dict) -> None:
    print(f"\n{item['id']}")
    print(f"Question: {item['question']}")
    print(f"Process: {item['law']}")
    print(f"Predictor: {item['predictor']}")
    print(f"Candidate: {item['candidate']['representation']} ({item['candidate']['id']})")
    print(f"Reference: {item['reference']['representation']} ({item['reference']['id']})")
    print(f"Candidate mean SMSE: {item['candidate']['mean_smse']:.10g}")
    print(f"Reference mean SMSE: {item['reference']['mean_smse']:.10g}")
    print(f"Mean difference (candidate - reference): {item['mean_difference']:+.10g}")
    lo, hi = item["adjusted_interval"]
    print(f"Adjusted interval for the mean difference: [{lo:+.10g}, {hi:+.10g}]")
    print(f"Relative gain: {item['relative_gain_percent']:+.3f}% (positive favors candidate)")
    print(f"Outcome: {OUTCOMES[item['classification']]}")
    print(item["interpretation"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=("Examples: --examples; --list; "
                "van_der_pol__nonlinear__local_vs_mem; "
                "nonlinear_ar__nonlinear__local_vs_mem --json"),
    )
    parser.add_argument("comparison_id", nargs="?", help="Published comparison ID; use --list to find IDs")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true", help="List all 40 comparison IDs and outcomes")
    modes.add_argument("--examples", action="store_true", help="Explain one lower-loss, one higher-loss, and one unresolved result")
    parser.add_argument("--json", action="store_true", help="Emit JSON; selected comparisons include exact CSV field strings")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to the specified primary-summary CSV; its SHA256 must match")
    args = parser.parse_args(argv)
    if args.comparison_id and (args.list or args.examples):
        parser.error("Choose a comparison ID, --list, or --examples.")
    if not (args.comparison_id or args.list or args.examples):
        if args.json:
            parser.error("--json requires a comparison ID, --list, or --examples.")
        parser.print_help()
        return 0
    try:
        rows = read_published_records(args.csv)
        context = evidence_context(args.csv)
        by_id = {row["id"]: row for row in rows}
        if args.comparison_id and args.comparison_id not in by_id:
            parser.error("Unknown comparison ID. Use --list to see the published IDs.")
        if args.list:
            items = [{key: row[key] for key in
                      ("id", "law", "decoder", "question", "classification")}
                     for row in rows]
        else:
            identifiers = EXAMPLES if args.examples else (args.comparison_id,)
            items = [explain(by_id[identifier]) for identifier in identifiers]
        if args.json:
            print(json.dumps({"mode": "list" if args.list else "explain",
                              "evidence": context, "comparisons": items},
                             indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        print("Primary comparison summaries: feature augmentation for forecasting")
        print("KAHM = Kernel Affine Hull Machine.")
        print(f"Source: {context['source_csv']} (published file identity verified)")
        print(context["evidence_level"])
        print(f"Family: {context['family_size']} comparisons; "
              f"{context['paired_repetitions_per_comparison']} paired repetitions per comparison.")
        if args.list:
            for item in items:
                print(f"\n{item['id']}")
                print(f"  {item['question']}: {OUTCOMES[item['classification']]}")
        else:
            print(f"SMSE: {context['smse']}")
            print(f"Difference: {context['difference']}")
            for item in items:
                print_explanation(item)
            print(f"\nRelative gain definition: {context['relative_gain']}")
        print(f"\nSelection: {context['selection']}")
        print(f"Uncertainty: {context['uncertainty']}")
        print(f"Evidence scope: {context['scope']}")
        print(f"Full summary-arithmetic check: {context['full_summary_check']}")
        return 0
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
