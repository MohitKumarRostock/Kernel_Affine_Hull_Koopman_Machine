"""Generate final controlled-certificate summary tables from verified evidence.

This script is specific to `four_state_original_iid_final_v1`. It draws no
samples and fits no predictors. A fresh independent numerical audit of the
retained evidence is required before aggregation.

Threshold interpretation uses the exact rational p metadata frozen in the
final configuration, not binary64 comparisons:

- `above`: exact quantity is strictly greater than the tolerance;
- `below`: exact quantity is strictly less than the tolerance;
- `boundary`: exact equality.

For every tolerance the output distinguishes:
1. relation of the exact optimal RMSE to the tolerance;
2. relation of the population certificate limit to the tolerance;
3. observed finite-sample exclusion rate.

Thus a zero finite-sample exclusion rate can be separated into:
- structural impossibility of exclusion by the population bound;
- a population-bound boundary case;
- finite-sample lack of power despite population-level detectability.

Monte Carlo probability intervals use independent replicate datasets as
Bernoulli trials, never the M evaluation pairs inside a replicate.
Clopper-Pearson intervals are pointwise and two-sided. Descriptive certificate
quantiles are not confidence intervals.

All zero certificates, coverage failures, false exclusions, signed gaps, and
boundary cases are retained. No original evidence file or campaign summary is
modified. Output must be a new directory outside the repository and evidence
tree. Importing this module performs no file I/O or random-number generation.
"""

import argparse
import csv
from collections import defaultdict
from fractions import Fraction
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


SUMMARY_ID = "four_state_final_summary_v1"

EXPECTED_CAMPAIGN_ID = "four_state_original_iid_final_v1"
EXPECTED_CAMPAIGN_ROLE = "final"
EXPECTED_CONFIG_SHA256 = (
    "630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182"
)
EXPECTED_REPLICATES = 117000
EXPECTED_CASES = 13
EXPECTED_SAMPLE_SIZES = 9
EXPECTED_REPLICATES_PER_CELL = 1000
EXPECTED_CELL_ROWS = 117
EXPECTED_EXCLUSION_ROWS = 702

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(
            handle,
            "sha256",
        ).hexdigest()


def binomial_rate(flags, confidence):
    """Pointwise exact binomial interval across replicate datasets."""

    flags = list(flags)

    require(
        bool(flags)
        and all(type(value) is bool for value in flags),
        "Expected a nonempty sequence of Boolean outcomes",
    )

    require(
        type(confidence) in (int, float)
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
        and 0.0 < confidence < 1.0,
        "Invalid confidence level",
    )

    successes = sum(flags)
    trials = len(flags)

    interval = binomtest(
        successes,
        trials,
    ).proportion_ci(
        confidence_level=confidence,
        method="exact",
    )

    return dict(
        count=successes,
        trials=trials,
        rate=successes / trials,
        ci_low=float(interval.low),
        ci_high=float(interval.high),
    )


def distribution(values):
    """Descriptive finite-sample distribution, retaining signed values."""

    values = list(values)

    require(
        bool(values)
        and all(
            type(value) in (int, float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ),
        "Expected nonempty finite numeric values",
    )

    quantiles = np.quantile(
        values,
        [0.05, 0.5, 0.95],
        method="linear",
    )

    return dict(
        mean=math.fsum(values) / len(values),
        minimum=min(values),
        q05=float(quantiles[0]),
        median=float(quantiles[1]),
        q95=float(quantiles[2]),
        maximum=max(values),
    )


def exact_design_p(case):
    metadata = case["p_exact"]

    numerator = metadata["numerator"]
    denominator = metadata["denominator"]

    require(
        type(numerator) is int
        and type(denominator) is int
        and denominator > 0,
        f"Invalid exact p metadata: {case['case_id']}",
    )

    exact = Fraction(
        numerator,
        denominator,
    )

    require(
        case["p"] == float(exact),
        f"Floating/exact p mismatch: {case['case_id']}",
    )

    return exact


def exact_population_quantities(case):
    """Exact theoretical quantities used only for scientific interpretation."""

    p = exact_design_p(case)

    if case["dynamics"] == "identity":
        return dict(
            exact_optimal_rmse=Fraction(0),
            population_certificate_limit=Fraction(0),
        )

    return dict(
        exact_optimal_rmse=(
            p - Fraction(1, 2)
        ),
        population_certificate_limit=max(
            Fraction(0),
            3 * p - Fraction(5, 2),
        ),
    )


def tolerance_fraction(value):
    """Use the configured decimal spelling as the scientific threshold."""

    require(
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(value),
        "Tolerance must be finite numeric",
    )

    return Fraction(str(value))


def exact_relation(quantity, tolerance):
    difference = quantity - tolerance

    if difference > 0:
        return "above"

    if difference < 0:
        return "below"

    return "boundary"


def summarize_cells(config, references, results):
    """Aggregate all audited final results without dropping any replicate."""

    require(
        config["campaign_id"] == EXPECTED_CAMPAIGN_ID,
        "Unexpected campaign identifier",
    )

    require(
        config["campaign_role"] == EXPECTED_CAMPAIGN_ROLE,
        "Expected final campaign evidence",
    )

    cases = {
        case["case_id"]: case
        for case in config["cases"]
    }

    require(
        len(cases)
        == len(config["cases"])
        == EXPECTED_CASES,
        "Unexpected final case count",
    )

    require(
        set(references) == set(cases),
        "Reference/case mismatch",
    )

    sample_sizes = config["sampling"]["sample_sizes"]
    replicates_per_cell = config["sampling"]["replicates_per_cell"]

    require(
        len(sample_sizes) == EXPECTED_SAMPLE_SIZES
        and len(set(sample_sizes)) == EXPECTED_SAMPLE_SIZES,
        "Unexpected or duplicated sample sizes",
    )

    require(
        replicates_per_cell
        == EXPECTED_REPLICATES_PER_CELL,
        "Unexpected replicate count per cell",
    )

    options = config["evaluation"]

    require(
        options["binomial_interval_method"]
        == "clopper_pearson",
        "Unsupported binomial interval method",
    )

    require(
        options["binomial_interval_confidence"]
        == 0.95,
        "Unexpected binomial confidence level",
    )

    require(
        options["certificate_quantiles"]
        == [0.05, 0.5, 0.95],
        "Unexpected certificate quantiles",
    )

    require(
        options["quantile_method"] == "linear",
        "Unsupported quantile method",
    )

    tolerances = options["exclusion_tolerances"]

    require(
        len(tolerances) == 6
        and len(set(tolerances)) == 6,
        "Unexpected exclusion tolerances",
    )

    expected_cells = {
        (case_id, n_pairs)
        for case_id in cases
        for n_pairs in sample_sizes
    }

    groups = defaultdict(list)
    seen_attempts = set()

    for row in results:
        identifier = row["attempt_id"]

        require(
            identifier not in seen_attempts,
            f"Duplicate result identifier: {identifier}",
        )

        seen_attempts.add(identifier)

        key = (
            row["case_id"],
            row["certificate"]["n_pairs"],
        )

        require(
            key in expected_cells,
            f"Unexpected result cell: {key}",
        )

        groups[key].append(row)

    require(
        set(groups) == expected_cells,
        "Missing final result cells",
    )

    require(
        len(seen_attempts) == EXPECTED_REPLICATES,
        "Unexpected total final replicate count",
    )

    confidence = options[
        "binomial_interval_confidence"
    ]

    cell_rows = []
    exclusion_rows = []

    ordered_cases = sorted(
        cases.values(),
        key=lambda case: case["case_key"],
    )

    for case in ordered_cases:
        case_id = case["case_id"]
        reference = references[case_id]

        exact = exact_population_quantities(
            case
        )

        exact_optimum = exact[
            "exact_optimal_rmse"
        ]

        exact_population_limit = exact[
            "population_certificate_limit"
        ]

        # Saved binary64 values are separately audited before summarization.
        saved_optimum = reference[
            "exact_optimal_rmse"
        ]

        saved_population_limit = reference[
            "population_certificate_limit"
        ]

        require(
            math.isclose(
                saved_optimum,
                float(exact_optimum),
                rel_tol=5e-12,
                abs_tol=5e-14,
            ),
            f"Saved/exact optimum mismatch: {case_id}",
        )

        require(
            math.isclose(
                saved_population_limit,
                float(exact_population_limit),
                rel_tol=5e-12,
                abs_tol=5e-14,
            ),
            f"Saved/exact population-limit mismatch: {case_id}",
        )

        for n_pairs in sorted(sample_sizes):
            rows = sorted(
                groups[(case_id, n_pairs)],
                key=lambda row: row["attempt_id"],
            )

            expected_ids = {
                (
                    f"c{case['case_key']}_"
                    f"m{n_pairs}_"
                    f"r{replicate}"
                )
                for replicate
                in range(replicates_per_cell)
            }

            require(
                len(rows) == replicates_per_cell
                and {
                    row["attempt_id"]
                    for row in rows
                } == expected_ids,
                f"Missing/mismatched replicates: {case_id}, M={n_pairs}",
            )

            base = dict(
                case_id=case_id,
                case_key=case["case_key"],
                dynamics=case["dynamics"],
                design_role=case["design_role"],
                p=case["p"],
                p_numerator=case["p_exact"]["numerator"],
                p_denominator=case["p_exact"]["denominator"],
                endpoint_case=case["endpoint_case"],
                kappa=case["kappa"],
                n_pairs=n_pairs,
                replicates=replicates_per_cell,
                delta=options["delta"],
                monte_carlo_ci_confidence=confidence,
                exact_optimal_rmse=float(
                    exact_optimum
                ),
                population_certificate_limit=float(
                    exact_population_limit
                ),
                population_bound_slack=float(
                    exact_optimum
                    - exact_population_limit
                ),
            )

            metrics = {
                "certificate": [
                    row["certificate"]["L_kappa_delta"]
                    for row in rows
                ],
                "gap_to_optimum": [
                    row["gap_to_optimum"]
                    for row in rows
                ],
                "sampling_gap": [
                    row["sampling_gap"]
                    for row in rows
                ],
                "assignment_upper": [
                    row["certificate"]["F_delta"]
                    for row in rows
                ],
                "variance_lower": [
                    row["certificate"]["V_delta"]
                    for row in rows
                ],
                "empirical_variance": [
                    row["certificate"]["s_hat"]
                    for row in rows
                ],
            }

            cell = dict(base)

            for metric, values in metrics.items():
                statistics = distribution(values)

                cell.update({
                    f"{metric}_{name}": value
                    for name, value
                    in statistics.items()
                })

            cell[
                "median_certificate_over_optimum"
            ] = (
                cell["certificate_median"]
                / float(exact_optimum)
                if exact_optimum > 0
                else None
            )

            cell[
                "median_certificate_over_population_limit"
            ] = (
                cell["certificate_median"]
                / float(exact_population_limit)
                if exact_population_limit > 0
                else None
            )

            for flag in (
                "zero_certificate",
                "strict_coverage",
                "tolerance_aware_coverage",
            ):
                rate = binomial_rate(
                    [
                        row[flag]
                        for row in rows
                    ],
                    confidence,
                )

                cell.update({
                    f"{flag}_{name}": value
                    for name, value
                    in rate.items()
                })

            cell_rows.append(cell)

            for position, tolerance in enumerate(
                tolerances
            ):
                threshold = tolerance_fraction(
                    tolerance
                )

                optimum_relation = exact_relation(
                    exact_optimum,
                    threshold,
                )

                population_relation = exact_relation(
                    exact_population_limit,
                    threshold,
                )

                decisions = []

                for row in rows:
                    require(
                        len(row["exclusions"])
                        == len(tolerances),
                        "Wrong number of exclusion decisions",
                    )

                    decision = row[
                        "exclusions"
                    ][position]

                    require(
                        decision["tolerance"]
                        == tolerance,
                        "Exclusion tolerance mismatch",
                    )

                    require(
                        type(decision["excluded"]) is bool,
                        "Non-Boolean exclusion decision",
                    )

                    decisions.append(
                        decision
                    )

                exclusion_flags = [
                    decision["excluded"]
                    for decision in decisions
                ]

                rate = binomial_rate(
                    exclusion_flags,
                    confidence,
                )

                if optimum_relation == "above":
                    interpretation = "exclusion_power"
                elif optimum_relation == "below":
                    interpretation = "false_exclusion_rate"
                else:
                    interpretation = "boundary_exclusion_rate"

                if population_relation == "above":
                    population_detectability = (
                        "population_bound_can_exclude"
                    )
                elif population_relation == "below":
                    population_detectability = (
                        "population_bound_cannot_exclude"
                    )
                else:
                    population_detectability = (
                        "population_bound_boundary"
                    )

                exclusion = dict(
                    base,
                    tolerance=tolerance,
                    exact_optimum_minus_tolerance=float(
                        exact_optimum
                        - threshold
                    ),
                    population_limit_minus_tolerance=float(
                        exact_population_limit
                        - threshold
                    ),
                    exact_optimum_relation=optimum_relation,
                    population_limit_relation=population_relation,
                    interpretation=interpretation,
                    population_detectability=population_detectability,
                    raw_saved_exact_error_exceeds_tolerance_count=sum(
                        decision[
                            "exact_error_exceeds_tolerance"
                        ]
                        for decision in decisions
                    ),
                    raw_saved_erroneous_exclusion_count=sum(
                        decision[
                            "erroneous_exclusion"
                        ]
                        for decision in decisions
                    ),
                    scientific_false_exclusion_count=(
                        rate["count"]
                        if optimum_relation == "below"
                        else None
                    ),
                )

                exclusion.update(rate)

                exclusion_rows.append(
                    exclusion
                )

    require(
        len(cell_rows) == EXPECTED_CELL_ROWS,
        "Unexpected final cell-summary row count",
    )

    require(
        len(exclusion_rows)
        == EXPECTED_EXCLUSION_ROWS,
        "Unexpected final exclusion-summary row count",
    )

    require(
        sum(
            row["replicates"]
            for row in cell_rows
        )
        == EXPECTED_REPLICATES,
        "Final summary replicate accounting mismatch",
    )

    return cell_rows, exclusion_rows


def load_verifier():
    path = (
        ROOT
        / "scripts"
        / "verify_four_state_final_numerics.py"
    )

    spec = importlib.util.spec_from_file_location(
        "_final_summary_numerical_audit",
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def source_provenance():
    """Require a clean committed summary implementation and locked packages."""

    def git(*args):
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.PIPE,
        )

    require(
        not git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).strip(),
        "Commit the final summary script and tests; "
        "use a clean source tree before generating summaries",
    )

    commit = git(
        "rev-parse",
        "HEAD",
    ).decode().strip()

    names = (
        "scripts/summarize_four_state_final.py",
        "scripts/verify_four_state_final_numerics.py",
        "tests/test_kahkm_four_state_final_summary.py",
        "requirements-lock-arm64.txt",
    )

    hashes = {}

    for name in names:
        path = ROOT / name

        require(
            path.is_file()
            and not path.is_symlink(),
            f"Missing or symlinked source: {name}",
        )

        hashes[name] = file_sha256(
            path
        )

        committed = git(
            "show",
            f"{commit}:{name}",
        )

        require(
            hashes[name]
            == hashlib.sha256(
                committed
            ).hexdigest(),
            f"Source bytes differ from HEAD: {name}",
        )

    locked = {}

    for line in (
        ROOT
        / "requirements-lock-arm64.txt"
    ).read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if (
            not line
            or line.startswith("#")
        ):
            continue

        name, separator, version = line.partition(
            "=="
        )

        require(
            separator
            and version,
            f"Unsupported lockfile entry: {line}",
        )

        locked[name] = version

    versions = dict(
        numpy=np.__version__,
        scipy=scipy.__version__,
    )

    require(
        all(
            locked.get(name) == version
            for name, version
            in versions.items()
        ),
        "NumPy/SciPy differ from reference lockfile; "
        "use .venv-arm64",
    )

    return dict(
        summary_source_commit=commit,
        source_sha256=hashes,
        python_version=sys.version,
        package_versions=versions,
    )


def write_json(path, value):
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(path, rows):
    require(
        bool(rows),
        "Cannot write an empty summary table",
    )

    columns = list(rows[0])

    require(
        all(
            set(row) == set(columns)
            for row in rows
        ),
        "Inconsistent summary-table columns",
    )

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def summarize_evidence(evidence, output):
    evidence = Path(evidence).resolve()
    output = Path(output).resolve()

    require(
        evidence.is_dir()
        and not evidence.is_symlink(),
        "Evidence directory not found or unsafe",
    )

    require(
        not output.exists()
        and not output.is_symlink(),
        "Output already exists; do not overwrite it",
    )

    require(
        output != ROOT
        and ROOT not in output.parents,
        "Output must be outside the source tree",
    )

    require(
        output != evidence
        and evidence not in output.parents
        and output not in evidence.parents,
        "Output must be separate from evidence tree",
    )

    source_before = source_provenance()

    verifier = load_verifier()

    audit = verifier.verify_evidence(
        evidence
    )

    require(
        audit["status"] == "passed",
        "Fresh numerical audit did not pass",
    )

    require(
        audit["verified_replicates"]
        == EXPECTED_REPLICATES,
        "Numerical audit replicate count mismatch",
    )

    require(
        audit["verified_cases"]
        == EXPECTED_CASES,
        "Numerical audit case count mismatch",
    )

    require(
        audit["verified_sample_sizes"]
        == EXPECTED_SAMPLE_SIZES,
        "Numerical audit sample-size count mismatch",
    )

    require(
        audit["config_sha256"]
        == EXPECTED_CONFIG_SHA256,
        "Numerical audit configuration mismatch",
    )

    config = verifier.document(
        evidence,
        "campaign_spec.json",
    )

    references = verifier.document(
        evidence,
        "references.json",
    )

    cell_rows, exclusion_rows = summarize_cells(
        config,
        references,
        verifier.records(
            evidence,
            "results.jsonl",
        ),
    )

    require(
        verifier.check_inventory(
            evidence
        )[0]
        == audit["evidence_manifest_sha256"],
        "Evidence changed during aggregation",
    )

    require(
        source_provenance()
        == source_before,
        "Source changed during aggregation",
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output / "cell_summary.csv",
        cell_rows,
    )

    write_csv(
        output / "exclusion_summary.csv",
        exclusion_rows,
    )

    write_json(
        output / "numerical_audit.json",
        audit,
    )

    readme = """# Final four-state certificate summaries

These tables summarize the final controlled certificate-validation campaign.

`cell_summary.csv` contains one row per final design case and evaluation sample
size. It reports the finite-sample certificate distribution, population-bound
slack, sampling gap, zero-certificate frequency, and empirical coverage.

`exclusion_summary.csv` contains one row per case, sample size, and tolerance.
Scientific threshold relations use the exact rational p metadata frozen in the
final configuration. They do not use binary64 near-equality heuristics.

`exact_optimum_relation` describes whether the true optimal RMSE is above,
below, or exactly on the tolerance.

`population_limit_relation` separately describes whether the limiting
population certificate is above, below, or exactly on the tolerance. This
distinguishes structural non-detectability from finite-sample underpower.

`interpretation` is based on the exact optimal error:
- exclusion_power
- false_exclusion_rate
- boundary_exclusion_rate

`population_detectability` is based on the exact population certificate limit:
- population_bound_can_exclude
- population_bound_cannot_exclude
- population_bound_boundary

Probability intervals are two-sided pointwise 95% Clopper-Pearson intervals
over independent replicate datasets. They are not simultaneous across cells
or tolerances. Certificate q05, median, and q95 are descriptive replicate
quantiles, not confidence intervals.

Zero observed failures do not imply a zero failure probability. All zero
certificates, signed gaps, coverage outcomes, exclusions, and exact threshold
boundaries are retained.

`numerical_audit.json` is a fresh independent recomputation of the complete
retained final evidence. No random samples are regenerated. Original evidence
is not modified.
"""

    with (
        output / "README.md"
    ).open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(readme)

    require(
        source_provenance()
        == source_before,
        "Source changed while saving summaries",
    )

    require(
        verifier.check_inventory(
            evidence
        )[0]
        == audit["evidence_manifest_sha256"],
        "Evidence changed while saving summaries",
    )

    metadata = dict(
        summary_id=SUMMARY_ID,
        status="completed",
        campaign_id=config["campaign_id"],
        campaign_role=config["campaign_role"],
        cell_rows=len(cell_rows),
        exclusion_rows=len(exclusion_rows),
        summarized_replicates=EXPECTED_REPLICATES,
        summarized_cases=EXPECTED_CASES,
        summarized_sample_sizes=EXPECTED_SAMPLE_SIZES,
        execution_source_commit=audit[
            "execution_source_commit"
        ],
        source=source_before,
        config_sha256=audit["config_sha256"],
        evidence_manifest_sha256=audit[
            "evidence_manifest_sha256"
        ],
        evidence_directory=str(evidence),
        output_directory=str(output),
        numerical_audit_status=audit["status"],
        exact_rational_threshold_interpretation=True,
        random_samples_generated=0,
        evidence_modified=False,
        command=[
            sys.executable,
            *sys.argv,
        ],
    )

    write_json(
        output / "summary_metadata.json",
        metadata,
    )

    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file()
    )

    with (
        output / "SHA256SUMS"
    ).open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        for path in files:
            handle.write(
                f"{file_sha256(path)}  "
                f"{path.name}\n"
            )

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    try:
        metadata = summarize_evidence(
            args.evidence,
            args.output,
        )

        print(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )

        return 0

    except Exception as exc:
        print(
            f"STOP: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        print(
            "Retain any created output directory; "
            "do not overwrite it.",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(main())
