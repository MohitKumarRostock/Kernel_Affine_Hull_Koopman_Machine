#!/usr/bin/env python3
"""Read historical summaries only; never open the new study's data or results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from scipy.optimize import brentq
from scipy.stats import nct, t


def assessment(repository):
    n, family = 32, 4
    critical = float(t.ppf(1 - .05 / family, n - 1))
    half = critical / n ** .5
    effect80 = float(brentq(lambda d: nct.sf(critical, n - 1, n ** .5 * d) - .8, 0, 2))
    effect90 = float(brentq(lambda d: nct.sf(critical, n - 1, n ** .5 * d) - .9, 0, 2))
    sources, records = {}, []
    for study, key in (("preprocessing_dimension", "primary_contrasts"),
                       ("policy_augmentation_20261101", "primary_results")):
        relative = f"extensions/{study}/analysis/analysis_summary.json"
        path = repository / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in json.loads(path.read_text())[key]:
            sd = row.get("margin_gap_sample_sd", row.get("margin_difference_sd"))
            reference = row["comparator_mean_smse"]
            records.append({"study": study, "historical_contrast": row["contrast_id"],
                "historical_margin_sd": sd, "historical_reference_mean_smse": reference,
                "illustrative_n32_allowance_smse": half * sd,
                "illustrative_n32_allowance_percent_reference": 100 * half * sd / reference,
                "illustrative_80pct_margin_excess_percent_reference": 100 * effect80 * sd / reference})
    return {"status": "historical_only_fixed_budget", "new_production_data_used": False,
        "n": n, "family_size": family, "family_alpha": .05, "critical_t": critical,
        "allowance_per_paired_sd": half,
        "allowance_fraction_of_n24": float(half / (t.ppf(.9875, 23) / 24 ** .5)),
        "normal_model_individual_power_standardized_margin_excess_80": effect80,
        "normal_model_individual_power_standardized_margin_excess_90": effect90,
        "sources": sources, "historical_illustrations": records,
        "limitations": ["No historical gate-only variance estimate.",
            "Historical procedures have raw-baseline and selection differences.",
            "These illustrations do not validate power for the new representation-only estimands.",
            "Fixed budget: no production-dependent extension or optional stopping."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = assessment(args.repository.resolve())
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")
