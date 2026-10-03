"""Recompute descriptive pilot tables from independently verified saved counts.

No random sampling, fitting, or changes to original evidence. Each cell is a
fixed case and sample size. Monte Carlo probability intervals use the number
of replicates, NOT the number of evaluation pairs or pooled case counts.
Clopper-Pearson intervals are pointwise, two-sided; no simultaneous coverage
across cells or tolerances is claimed. Certificate quantiles describe replicate
variation, not confidence intervals for a mean. All zero bounds and violations
are retained. Power is labeled only when the exact error exceeds the tolerance.

CLI use requires a clean committed source tree and the locked NumPy/SciPy
versions. A fresh numerical audit precedes aggregation. Output must be a new
directory outside the source/evidence trees. A partial output is not a complete
summary; retain it rather than overwriting it. Importing performs no file I/O.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
from scipy.stats import binomtest

SUMMARY_ID = "four_state_pilot_summary_v1"
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def binomial_rate(flags, confidence):
    """Pointwise interval for one Bernoulli outcome across independent replicates."""
    flags = list(flags)
    require(flags and all(type(x) is bool for x in flags), "Expected nonempty Boolean outcomes")
    require(type(confidence) in (int, float) and 0 < confidence < 1, "Invalid confidence level")
    successes, total = sum(flags), len(flags)
    interval = binomtest(successes, total).proportion_ci(
        confidence_level=confidence, method="exact",
    )
    return dict(count=successes, trials=total, rate=successes/total,
                ci_low=float(interval.low), ci_high=float(interval.high))


def distribution(values):
    """Descriptive quantiles, with signed inputs retained unchanged."""
    values = list(values)
    require(values and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            "Expected nonempty finite numeric values")
    quantiles = np.quantile(values, [.05, .5, .95], method="linear")
    return dict(mean=math.fsum(values)/len(values), minimum=min(values),
                q05=float(quantiles[0]), median=float(quantiles[1]),
                q95=float(quantiles[2]), maximum=max(values))


def summarize_cells(config, references, results):
    """Aggregate audited results; independently reject omitted/duplicated cells."""
    options = config["evaluation"]
    require(config["campaign_role"] == "pilot", "Expected pilot evidence")
    require(options["binomial_interval_method"] == "clopper_pearson"
            and options["quantile_method"] == "linear"
            and options["certificate_quantiles"] == [.05, .5, .95],
            "Unsupported summary settings")
    cases = {case["case_id"]: case for case in config["cases"]}
    require(len(cases) == len(config["cases"]) and set(references) == set(cases),
            "Reference/case mismatch")
    sizes = config["sampling"]["sample_sizes"]
    repetitions = config["sampling"]["replicates_per_cell"]
    require(type(repetitions) is int and repetitions > 0, "Invalid replicate count")
    require(len(set(sizes)) == len(sizes), "Duplicate sample sizes")
    expected_cells = {(key, n) for key in cases for n in sizes}
    groups, seen = defaultdict(list), set()
    for row in results:
        key = (row["case_id"], row["certificate"]["n_pairs"])
        require(key in expected_cells, "Unexpected result cell")
        require(row["attempt_id"] not in seen, "Duplicate result identifier")
        seen.add(row["attempt_id"])
        groups[key].append(row)
    require(set(groups) == expected_cells, "Missing cells")
    cells, exclusions = [], []
    confidence = options["binomial_interval_confidence"]
    for case in sorted(cases.values(), key=lambda c: c["case_key"]):
        case_id = case["case_id"]
        ref = references[case_id]
        optimum = ref["exact_optimal_rmse"]
        for n in sorted(sizes):
            rows = sorted(groups[(case_id, n)], key=lambda r: r["attempt_id"])
            expected_ids = {f"c{case['case_key']}_m{n}_r{rep}" for rep in range(repetitions)}
            require(len(rows) == repetitions and {r["attempt_id"] for r in rows} == expected_ids,
                    "Missing or mismatched replicate identifiers")
            base = dict(case_id=case_id, case_key=case["case_key"], dynamics=case["dynamics"],
                        p=case["p"], endpoint_case=case["endpoint_case"], kappa=case["kappa"],
                        n_pairs=n, replicates=repetitions, delta=options["delta"],
                        monte_carlo_ci_confidence=confidence, exact_optimal_rmse=optimum,
                        population_certificate_limit=ref["population_certificate_limit"])
            cell = dict(base, population_bound_slack=optimum-ref["population_certificate_limit"])
            metrics = {
                "certificate": [r["certificate"]["L_kappa_delta"] for r in rows],
                "gap_to_optimum": [r["gap_to_optimum"] for r in rows],
                "sampling_gap": [r["sampling_gap"] for r in rows],
                "assignment_upper": [r["certificate"]["F_delta"] for r in rows],
                "variance_lower": [r["certificate"]["V_delta"] for r in rows],
                "empirical_variance": [r["certificate"]["s_hat"] for r in rows],
            }
            for metric, values in metrics.items():
                cell.update({metric+"_"+name: value for name, value in distribution(values).items()})
            cell["median_certificate_over_optimum"] = (
                cell["certificate_median"]/optimum if optimum > 0 else None
            )
            for flag in ("zero_certificate", "strict_coverage", "tolerance_aware_coverage"):
                cell.update({flag+"_"+name: value for name, value in binomial_rate(
                    [r[flag] for r in rows], confidence,
                ).items()})
            cells.append(cell)
            for position, tolerance in enumerate(options["exclusion_tolerances"]):
                decisions = []
                for row in rows:
                    require(len(row["exclusions"]) == len(options["exclusion_tolerances"]),
                            "Wrong number of exclusion decisions")
                    decision = row["exclusions"][position]
                    require(decision["tolerance"] == tolerance, "Exclusion tolerance mismatch")
                    truth = optimum > tolerance
                    require(decision["exact_error_exceeds_tolerance"] is truth,
                            "Inconsistent exclusion ground truth")
                    require(type(decision["excluded"]) is bool, "Non-Boolean exclusion flag")
                    require(decision["erroneous_exclusion"] is (decision["excluded"] and not truth),
                            "Inconsistent erroneous-exclusion flag")
                    decisions.append(decision)
                exclusion = dict(base, tolerance=tolerance,
                                 exact_error_exceeds_tolerance=truth,
                                 interpretation="exclusion_power" if truth else "false_exclusion_rate",
                                 erroneous_exclusion_count=sum(d["erroneous_exclusion"] for d in decisions))
                exclusion.update(binomial_rate([d["excluded"] for d in decisions], confidence))
                exclusions.append(exclusion)
    return cells, exclusions


def source_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)
    require(not git("status", "--porcelain=v1", "--untracked-files=all").strip(),
            "Commit the summary script and tests; use a clean source tree")
    commit = git("rev-parse", "HEAD").decode().strip()
    names = ("scripts/summarize_four_state_pilot.py", "scripts/verify_four_state_pilot_numerics.py",
             "tests/test_kahkm_four_state_summary.py", "requirements-lock-arm64.txt")
    hashes = {}
    for name in names:
        path = ROOT/name
        require(path.is_file() and not path.is_symlink(), "Missing or symlinked source: "+name)
        hashes[name] = file_hash(path)
        require(hashes[name] == hashlib.sha256(git("show", f"{commit}:{name}")).hexdigest(),
                "Source bytes differ from HEAD: "+name)
    locked = dict(line.strip().split("==", 1) for line in (ROOT/names[-1]).read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#"))
    versions = dict(numpy=np.__version__, scipy=scipy.__version__)
    require(all(locked.get(name) == value for name, value in versions.items()),
            "NumPy/SciPy differ from the reference lockfile; use .venv-arm64")
    return dict(summary_source_commit=commit, source_sha256=hashes,
                python_version=sys.version, package_versions=versions)


def load_verifier():
    path = ROOT/"scripts"/"verify_four_state_pilot_numerics.py"
    spec = importlib.util.spec_from_file_location("_pilot_summary_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    require(bool(rows), "Cannot write empty table")
    columns = list(rows[0])
    require(all(set(row) == set(columns) for row in rows), "Inconsistent table columns")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_evidence(evidence, output):
    evidence, output = Path(evidence).resolve(), Path(output).resolve()
    require(evidence.is_dir(), "Evidence directory not found")
    require(not output.exists(), "Output already exists; do not overwrite it")
    require(ROOT != output and ROOT not in output.parents and output not in ROOT.parents,
            "Output must be outside the source tree")
    require(evidence != output and evidence not in output.parents and output not in evidence.parents,
            "Output must be separate from the evidence tree")
    source = source_provenance()
    verifier = load_verifier()
    audit = verifier.verify_evidence(evidence)
    config = verifier.document(evidence, "campaign_spec.json")
    references = verifier.document(evidence, "references.json")
    cells, exclusions = summarize_cells(config, references, verifier.records(evidence, "results.jsonl"))
    require(sum(row["replicates"] for row in cells) == audit["verified_replicates"],
            "Summary replicate accounting differs from audit")
    require(verifier.check_inventory(evidence)[0] == audit["evidence_manifest_sha256"],
            "Evidence changed during aggregation")
    require(source_provenance() == source, "Source changed during aggregation")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/"cell_summary.csv", cells)
    write_csv(output/"exclusion_summary.csv", exclusions)
    write_json(output/"numerical_audit.json", audit)
    readme = """# Four-state pilot summaries

Pilot only, not final manuscript evidence. Every scheduled replicate is retained.
cell_summary.csv has one row per fixed case and sample size. Probability columns
use independent replicate datasets as trials, not individual evaluation pairs.
exclusion_summary.csv has one row per cell and tolerance. The interpretation
column distinguishes power from false-exclusion rate using the analytical optimum.

Probability intervals are two-sided pointwise 95% Clopper-Pearson intervals.
They are not simultaneous across cells, metrics, or tolerances. Zero observed
failures do not establish a zero failure probability. Certificate q05, median,
and q95 are descriptive replicate quantiles using linear interpolation, not
confidence intervals. Rates are fractions in [0,1]. Signed gaps are not clipped.
A blank certificate/optimum ratio means undefined because the optimum is zero.

Source/configuration/evidence identities are in summary_metadata.json. A fresh
independent numerical audit is retained in numerical_audit.json. No samples were
drawn and no original files or campaign summaries were modified. The execution
commit and the later summary source commit describe different operations.
A completed package requires summary_metadata.json and a valid SHA256SUMS.
"""
    with (output/"README.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)
    require(source_provenance() == source, "Source changed while saving summary")
    require(verifier.check_inventory(evidence)[0] == audit["evidence_manifest_sha256"],
            "Evidence changed while saving summary")
    report = dict(summary_id=SUMMARY_ID, status="completed", campaign_role="pilot",
                  campaign_id=config["campaign_id"], cell_rows=len(cells),
                  exclusion_rows=len(exclusions), summarized_replicates=audit["verified_replicates"],
                  execution_source_commit=audit["execution_source_commit"], source=source,
                  config_sha256=audit["config_sha256"],
                  evidence_manifest_sha256=audit["evidence_manifest_sha256"],
                  evidence_directory=str(evidence), output_directory=str(output),
                  numerical_audit_status=audit["status"],
                  random_samples_generated=0, evidence_modified=False,
                  command=[sys.executable, *sys.argv])
    write_json(output/"summary_metadata.json", report)
    files = sorted(path for path in output.iterdir() if path.is_file())
    with (output/"SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{file_hash(path)}  {path.name}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = summarize_evidence(args.evidence, args.output)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Retain any created output directory; do not overwrite it.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
