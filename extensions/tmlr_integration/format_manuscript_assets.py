"""Apply scientific captions and notation to derived manuscript displays.

The transformation changes only explicitly listed prose and symbol labels.
Experimental data, numeric table entries, and frozen source files are unchanged.
Run after regenerating displays to reproduce their manuscript presentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent / "manuscript"
REPLACEMENTS = {
    "natural/diagnostic_details.tex": [
        ("Do not add the four rows: their developmental labels and some features can be shared.",
         "The rows share development labels and some features, so their costs are not additive."),
    ],
    "scaling/natural_main_table.tex": [
        ("retains the raw skip", "retains the standardized raw-history component"),
        ("All new comparisons", "These comparisons"),
    ],
    "scaling/natural_cost_table.tex": [
        ("the inherited partial import timer is not a complete cold-start measure.",
         "the recorded import interval covers only part of initialization."),
        ("These single-execution observations are implementation-specific and should not be divided by timings from the earlier execution.",
         "These single-execution observations are implementation-specific. Cost ratios use reference methods measured in this same execution."),
    ],
    "utility/policy_choices.tex": [
        ("Costs include examining rejected corrections; a raw decision is not a free decision.",
         "Selection cost includes the correction searches even when the policy chooses the raw baseline."),
    ],
    "utility/failures.tex": [
        ("No source substitution, changed cluster count, or silent fallback is applied.",
         "The specified method reports a fitting failure when the requested groups cannot be formed."),
    ],
    "utility/synthetic_intervals.tex": [("Every primary new synthetic contrast.", "All primary synthetic contrasts.")],
    "utility/synthetic_controls.tex": [("the same new synthetic factorial", "the same synthetic factorial")],
    "utility/archive_sources.tex": [
        ("by new archive source", "by archive source"),
        ("both frozen policies", "both specified policies"),
    ],
    "target_geometry/diagnostics.tex": [
        (r"$J$ is the regularized", r"$\mathcal J$ is the regularized"),
        (r"Median $J_0-J_*$", r"Median $\mathcal J_0-\mathcal J_{\rm fit}$"),
        ("A finite improved iterate", r"$\mathcal J_0$ and $\mathcal J_{\rm fit}$ are the identity-metric and selected fitted-metric objectives. A finite improved iterate"),
        ("their OOF residual MSE", "their out-of-fold (OOF) residual MSE"),
    ],
    "target_geometry/archive_main.tex": [
        ("The new-collection archive experiment:", "Natural-data forecasting results:"),
        ("Change is relative to raw ridge;", "IS denotes in-sample residuals and OOF denotes out-of-fold residuals. Change is relative to raw ridge;"),
    ],
    "target_geometry/synthetic_intervals.tex": [
        ("No claim joins this family to earlier experiments.", "Coverage applies to this family separately from the other experiments."),
    ],
    "adaptive_development/resource_table.tex": [
        ("P99 is first computed", "P99 denotes the 99th percentile and is first computed"),
        ("worker RSS includes", "worker resident set size (RSS) includes"),
    ],
}


def description(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def format_assets(paper: Path = PAPER) -> dict:
    paper = paper.resolve()
    generated = paper / "generated"
    records = {}
    changed = []
    for name, replacements in REPLACEMENTS.items():
        path = generated / name
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_text()
        current = original
        for before, after in replacements:
            if after in current:
                continue
            if before in current:
                current = current.replace(before, after)
            else:
                raise ValueError(f"Expected caption text is absent: {name}: {before}")
        if current != original:
            path.write_text(current)
            changed.append(name)
        records[name] = description(path)
    transform = {
        "schema_version": 1,
        "scope": "Scientific caption and notation transformation of derived displays; numeric results unchanged.",
        "script": description(Path(__file__)),
        "outputs": records,
    }
    write_json(paper / "PRESENTATION_PROVENANCE.json", transform)
    for rel in ["generated/utility/ASSET_PROVENANCE.json", "generated/adaptive_development/ASSET_PROVENANCE.json"]:
        path = paper / rel
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text())
        for name in metadata.get("generated_sha256", {}):
            output = path.parent / name
            if output.is_file():
                metadata["generated_sha256"][name] = description(output)["sha256"]
        outputs = metadata.get("outputs", {})
        if isinstance(outputs, dict):
            for name in outputs:
                output = path.parent / name
                if output.is_file():
                    outputs[name] = description(output)
        metadata["presentation_transform"] = {
            "script": transform["script"],
            "inventory": "PRESENTATION_PROVENANCE.json",
        }
        write_json(path, metadata)
    path = paper / "ASSET_PROVENANCE.json"
    if path.is_file():
        metadata = json.loads(path.read_text())
        metadata["generated_sha256"] = {
            p.relative_to(generated).as_posix(): description(p)["sha256"]
            for p in sorted(generated.rglob("*")) if p.is_file()
        }
        metadata["presentation_transform"] = {
            "script": transform["script"],
            "inventory": "PRESENTATION_PROVENANCE.json",
        }
        write_json(path, metadata)
    return {"status": "passed", "formatted_displays": len(records),
            "changed_displays": changed, "numeric_results_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=PAPER)
    args = parser.parse_args()
    print(json.dumps(format_assets(args.paper), indent=2, sort_keys=True))
