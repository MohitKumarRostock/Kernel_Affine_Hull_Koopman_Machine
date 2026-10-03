"""Read-only independent numerical audit of final four-state evidence.

This verifier imports none of the experiment's certificate, statistics,
sampling, reference, or runner modules. Empirical quantities are reconstructed
from retained state counts with exact Fraction arithmetic applied to the exact
binary64 p used by the experiment. Certificate corrections are recomputed with
80-digit Decimal arithmetic.

The final design also stores exact rational p metadata. Those rational values
are used independently to audit the analytical population formulas and the
scientific relation of the exact optimum/population limit to reporting
tolerances. This is kept separate from the raw binary64 comparisons retained
in the original evidence.

The verifier:
- checks the complete evidence inventory and SHA-256 hashes;
- reconstructs all 117,000 scheduled jobs from the frozen configuration;
- checks every retained sample, result, and terminal event;
- recomputes empirical f_hat and s_hat from counts;
- recomputes every finite-sample certificate independently;
- checks analytical references and matrix feasibility;
- checks saved coverage/exclusion flags exactly as originally computed;
- reports exact-rational threshold boundary counts separately.

A correctly recorded zero certificate, coverage failure, or false exclusion is
an experimental outcome, not a verifier failure.

No samples are regenerated and no evidence is modified. This does not prove
the theorem or empirically establish independence of random streams. CLI use
requires this verifier and its tests to be committed in a clean source tree.
"""

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


VERIFIER_ID = "four_state_final_independent_numerics_v1"

CONFIG_SHA256 = (
    "630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182"
)

EXPECTED_CAMPAIGN_ID = "four_state_original_iid_final_v1"
EXPECTED_CAMPAIGN_ROLE = "final"
EXPECTED_REPLICATES = 117000
EXPECTED_CASES = 13
EXPECTED_SAMPLE_SIZES = 9
EXPECTED_REPLICATES_PER_CELL = 1000

PRECISION = 80
RTOL = 5e-12
ATOL = 5e-14


EXPECTED_CASE_DESIGN = {
    "identity_p_31_over_32": (
        "identity", 31, 32, "exact_closure_control"
    ),
    "conflicting_successors_p_4_over_5": (
        "conflicting_successors", 4, 5, "structurally_inconclusive"
    ),
    "conflicting_successors_p_5_over_6": (
        "conflicting_successors", 5, 6, "positive_certificate_boundary"
    ),
    "conflicting_successors_p_17_over_20": (
        "conflicting_successors", 17, 20, "population_limit_0.05"
    ),
    "conflicting_successors_p_13_over_15": (
        "conflicting_successors", 13, 15, "population_limit_0.10"
    ),
    "conflicting_successors_p_9_over_10": (
        "conflicting_successors", 9, 10, "population_limit_0.20"
    ),
    "conflicting_successors_p_14_over_15": (
        "conflicting_successors", 14, 15, "population_limit_0.30"
    ),
    "conflicting_successors_p_19_over_20": (
        "conflicting_successors", 19, 20,
        "interior_between_0.30_and_0.40"
    ),
    "conflicting_successors_p_29_over_30": (
        "conflicting_successors", 29, 30, "population_limit_0.40"
    ),
    "conflicting_successors_p_31_over_32": (
        "conflicting_successors", 31, 32, "manuscript_reference"
    ),
    "conflicting_successors_p_59_over_60": (
        "conflicting_successors", 59, 60, "population_limit_0.45"
    ),
    "conflicting_successors_p_99_over_100": (
        "conflicting_successors", 99, 100, "above_0.45_threshold"
    ),
    "conflicting_successors_p_1": (
        "conflicting_successors", 1, 1, "hard_assignment_endpoint"
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(text):
    """Strict JSON: reject duplicate keys and nonfinite constants."""

    def pairs(items):
        result = {}

        for key, value in items:
            require(
                key not in result,
                f"Duplicate JSON key: {key}",
            )
            result[key] = value

        return result

    def nonfinite(value):
        raise ValueError(
            f"Nonfinite JSON literal: {value}"
        )

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=nonfinite,
    )


def document(folder, name):
    return read_json(
        (folder / name).read_text(encoding="utf-8")
    )


def records(folder, name):
    with (folder / name).open(
        encoding="utf-8"
    ) as handle:
        for number, line in enumerate(handle, 1):
            require(
                bool(line.strip()),
                f"Blank record in {name}:{number}",
            )
            yield read_json(line)


def index_records(folder, name):
    result = {}

    for row in records(folder, name):
        key = row["attempt_id"]

        require(
            key not in result,
            f"Duplicate attempt in {name}: {key}",
        )

        result[key] = row

    return result


def check_inventory(folder):
    """Verify complete retained-file inventory and every SHA-256 digest."""

    folder = Path(folder)
    manifest = folder / "SHA256SUMS"

    require(
        manifest.is_file() and not manifest.is_symlink(),
        "Missing or unsafe SHA256SUMS",
    )

    expected = {}

    for line in manifest.read_text(
        encoding="utf-8"
    ).splitlines():

        fields = line.split("  ", 1)

        require(
            len(fields) == 2,
            "Malformed checksum entry",
        )

        checksum, name = fields
        relative = PurePosixPath(name)

        require(
            re.fullmatch(
                r"[0-9a-f]{64}",
                checksum,
            ) is not None,
            "Malformed checksum",
        )

        require(
            name
            and not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == name
            and name != "SHA256SUMS",
            "Unsafe inventory path",
        )

        require(
            name not in expected,
            "Duplicate inventory entry",
        )

        expected[name] = checksum

    actual = set()

    for path in folder.rglob("*"):
        require(
            not path.is_symlink(),
            f"Evidence symlink: {path}",
        )

        if path.is_file() and path != manifest:
            actual.add(
                path.relative_to(folder).as_posix()
            )

    require(
        actual == set(expected),
        "Missing or unlisted evidence files",
    )

    for name, checksum in expected.items():
        require(
            file_sha256(folder / name) == checksum,
            f"Checksum mismatch: {name}",
        )

    return file_sha256(manifest), len(expected)


class Comparison:
    """Track numerical agreement without clipping discrepancies."""

    def __init__(self):
        self.max_errors = {}
        self.numbers_checked = 0
        self.context = ""

    def close(self, actual, expected, group):
        if isinstance(expected, (tuple, list)):
            require(
                isinstance(actual, (tuple, list))
                and len(actual) == len(expected),
                f"Shape mismatch {self.context}: {group}",
            )

            for a, e in zip(actual, expected):
                self.close(a, e, group)

            return

        require(
            not isinstance(actual, bool)
            and isinstance(
                actual,
                (int, float, Fraction, Decimal),
            ),
            f"Invalid numeric value "
            f"{self.context}: {group}",
        )

        actual_float = float(actual)
        expected_float = float(expected)

        require(
            math.isfinite(actual_float)
            and math.isfinite(expected_float),
            f"Nonfinite numeric value "
            f"{self.context}: {group}",
        )

        error = abs(
            actual_float - expected_float
        )

        self.max_errors[group] = max(
            self.max_errors.get(group, 0.0),
            error,
        )

        self.numbers_checked += 1

        require(
            math.isclose(
                actual_float,
                expected_float,
                rel_tol=RTOL,
                abs_tol=ATOL,
            ),
            f"Numerical mismatch "
            f"{self.context}/{group}: "
            f"saved={actual!r}, "
            f"recomputed={expected!r}",
        )


def binary_fraction(value):
    """Exact rational value of the binary64 input used numerically."""
    return Fraction.from_float(float(value))


def exact_design_p(case):
    metadata = case["p_exact"]

    numerator = metadata["numerator"]
    denominator = metadata["denominator"]

    require(
        type(numerator) is int
        and type(denominator) is int
        and denominator > 0,
        "Invalid exact p metadata",
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


def decimal_value(value):
    if isinstance(value, Fraction):
        return (
            Decimal(value.numerator)
            / Decimal(value.denominator)
        )

    return Decimal.from_float(float(value))


def weighted_statistics_binary64(
    p_float,
    dynamics,
    counts,
):
    """Exact weighted sums for the binary64 coordinates used by the run."""

    p = binary_fraction(p_float)
    q = 1 - p

    current = (
        (p, q),
        (p, q),
        (q, p),
        (q, p),
    )

    if dynamics == "identity":
        future = current
    else:
        future = (
            current[0],
            current[2],
            current[2],
            current[2],
        )

    labels = (0, 0, 1, 1)
    total = sum(counts)

    require(
        total > 0,
        "Empty count vector",
    )

    means = []
    sses = []
    class_sizes = []

    for class_index in range(2):
        indices = [
            i
            for i in range(4)
            if labels[i] == class_index
        ]

        size = sum(
            counts[i]
            for i in indices
        )

        mean = tuple(
            (
                sum(
                    counts[i] * future[i][j]
                    for i in indices
                )
                / size
            )
            if size
            else Fraction(1, 2)
            for j in range(2)
        )

        sse = sum(
            (
                counts[i]
                * (
                    future[i][j]
                    - mean[j]
                ) ** 2
                for i in indices
                for j in range(2)
            ),
            Fraction(0),
        )

        class_sizes.append(size)
        means.append(mean)
        sses.append(sse)

    f_hat = (
        sum(
            counts[i]
            * (
                1
                - current[i][labels[i]]
            ) ** 2
            for i in range(4)
        )
        / total
    )

    s_hat = sum(sses) / total

    return dict(
        f_hat=f_hat,
        s_hat=s_hat,
        class_sizes=class_sizes,
        means=means,
        sses=sses,
        current=current,
        future=future,
    )


def independent_certificate(
    f_hat,
    s_hat,
    n_pairs,
    kappa,
    delta,
):
    """Independent Eq. (8) calculation with high-precision Decimal arithmetic."""

    with localcontext() as context:
        context.prec = PRECISION

        zero = Decimal(0)
        one = Decimal(1)
        two = Decimal(2)

        f = decimal_value(f_hat)
        s = decimal_value(s_hat)
        k = decimal_value(kappa)
        d = decimal_value(delta)

        r = (
            (two / d).ln()
            / Decimal(n_pairs)
        )

        # Positive root of:
        #     F - f = sqrt(2 r F)
        # before clipping at one.
        F = min(
            one,
            (
                (r / two).sqrt()
                + (f + r / two).sqrt()
            ) ** 2,
        )

        V = max(
            zero,
            s - (two * r).sqrt(),
        )

        L = max(
            zero,
            V.sqrt()
            - k * (two * F).sqrt(),
        )

        return dict(
            r_delta=float(r),
            F_delta=float(F),
            V_delta=float(V),
            L_kappa_delta=float(L),
        )


def exact_population_quantities(case):
    """Exact rational theoretical quantities from final-design metadata."""

    p = exact_design_p(case)

    if case["dynamics"] == "identity":
        return dict(
            exact_optimal_rmse=Fraction(0),
            exact_optimal_mse=Fraction(0),
            assignment_loss=(1 - p) ** 2,
            population_certificate_limit=Fraction(0),
            current_coordinate_variance=(
                (2 * p - 1) ** 2
                / 2
            ),
        )

    optimum = p - Fraction(1, 2)
    mse = optimum ** 2
    assignment = (1 - p) ** 2
    limit = max(
        Fraction(0),
        3 * p - Fraction(5, 2),
    )
    variance = (
        (2 * p - 1) ** 2
        / 2
    )

    return dict(
        exact_optimal_rmse=optimum,
        exact_optimal_mse=mse,
        assignment_loss=assignment,
        population_certificate_limit=limit,
        current_coordinate_variance=variance,
    )


def tolerance_fraction(value):
    """Interpret configured decimal thresholds by their decimal spelling."""
    return Fraction(str(value))


def exact_relation(value, threshold):
    difference = value - threshold

    if difference > 0:
        return "above"
    if difference < 0:
        return "below"
    return "boundary"


def verify_case_design(config):
    """Audit all exact final-design cases before reading replicate results."""

    cases = config["cases"]

    require(
        len(cases) == EXPECTED_CASES,
        "Unexpected final case count",
    )

    require(
        {case["case_id"] for case in cases}
        == set(EXPECTED_CASE_DESIGN),
        "Final case identifiers differ from frozen design",
    )

    require(
        len({case["case_key"] for case in cases})
        == EXPECTED_CASES,
        "Duplicate final case keys",
    )

    for case in cases:
        expected = EXPECTED_CASE_DESIGN[
            case["case_id"]
        ]

        dynamics, numerator, denominator, role = expected

        require(
            case["dynamics"] == dynamics,
            f"Dynamics mismatch: {case['case_id']}",
        )

        require(
            case["p_exact"]
            == {
                "numerator": numerator,
                "denominator": denominator,
            },
            f"Exact p mismatch: {case['case_id']}",
        )

        require(
            case["design_role"] == role,
            f"Design-role mismatch: {case['case_id']}",
        )

        exact_design_p(case)

    return {
        case["case_id"]: case
        for case in cases
    }


def verify_reference(saved, case, compare):
    """Check saved analytical reference independently."""

    binary = weighted_statistics_binary64(
        case["p"],
        case["dynamics"],
        (1, 1, 1, 1),
    )

    exact = exact_population_quantities(case)

    require(
        saved["reference_id"]
        == "four_state_uniform_reference_v1",
        "Unexpected reference identifier",
    )

    require(
        saved["dynamics"] == case["dynamics"],
        "Reference dynamics mismatch",
    )

    require(
        saved["p"] == case["p"],
        "Reference p mismatch",
    )

    require(
        saved["kappa"] == case["kappa"],
        "Reference kappa mismatch",
    )

    require(
        saved["state_names"]
        == ["a", "b", "d", "e"],
        "State-name mismatch",
    )

    require(
        saved["states"]
        == [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        "State-coordinate mismatch",
    )

    require(
        saved["probabilities"]
        == [0.25] * 4,
        "Reference probability mismatch",
    )

    require(
        saved["labels"]
        == [0, 0, 1, 1],
        "Reference label mismatch",
    )

    expected_successor_indices = (
        [0, 1, 2, 3]
        if case["dynamics"] == "identity"
        else [0, 2, 2, 2]
    )

    require(
        saved["successor_indices"]
        == expected_successor_indices,
        "Reference successor-map mismatch",
    )

    compare.close(
        saved["phi"],
        binary["current"],
        "reference_coordinates",
    )

    compare.close(
        saved["successors"],
        binary["future"],
        "reference_coordinates",
    )

    # Verify saved B actually attains the appropriate conditional means.
    B = saved["optimal_B"]

    require(
        isinstance(B, list)
        and len(B) == 2
        and all(
            isinstance(row, list)
            and len(row) == 2
            for row in B
        ),
        "Prediction matrix must be 2x2",
    )

    B_fraction = [
        [
            binary_fraction(value)
            for value in row
        ]
        for row in B
    ]

    for i, row in enumerate(binary["current"]):
        prediction = [
            sum(
                row[h]
                * B_fraction[h][j]
                for h in range(2)
            )
            for j in range(2)
        ]

        compare.close(
            prediction,
            binary["means"][i // 2],
            "conditional_mean_attainment",
        )

    # Independently verify spectral norm <= configured kappa.
    with localcontext() as context:
        context.prec = PRECISION

        gram = [
            [
                decimal_value(
                    sum(
                        B_fraction[h][i]
                        * B_fraction[h][j]
                        for h in range(2)
                    )
                )
                for j in range(2)
            ]
            for i in range(2)
        ]

        discriminant = (
            (
                gram[0][0]
                - gram[1][1]
            ) ** 2
            + Decimal(4)
            * gram[0][1] ** 2
        )

        largest_eigenvalue = (
            gram[0][0]
            + gram[1][1]
            + discriminant.sqrt()
        ) / Decimal(2)

        spectral = (
            largest_eigenvalue.sqrt()
        )

    require(
        float(spectral)
        <= case["kappa"] + ATOL,
        "Saved optimizer exceeds spectral-norm budget",
    )

    # Exact rational theory versus saved binary64 reference values.
    for name in (
        "assignment_loss",
        "exact_optimal_mse",
        "exact_optimal_rmse",
        "population_certificate_limit",
        "current_coordinate_variance",
    ):
        compare.close(
            saved[name],
            exact[name],
            "reference_exact_population",
        )

    compare.close(
        saved["sigma_res_sq"],
        exact["exact_optimal_mse"],
        "reference_exact_population",
    )


def reconstruct_schedule(config):
    planned = {}

    for case in config["cases"]:
        for n_pairs in config["sampling"]["sample_sizes"]:
            for replicate in range(
                config["sampling"]["replicates_per_cell"]
            ):
                identifier = (
                    f"c{case['case_key']}_"
                    f"m{n_pairs}_"
                    f"r{replicate}"
                )

                require(
                    identifier not in planned,
                    "Duplicate planned identifier",
                )

                planned[identifier] = dict(
                    attempt_id=identifier,
                    attempt_number=1,
                    case_id=case["case_id"],
                    seed=dict(
                        campaign_key=config["rng"]["campaign_key"],
                        case_key=case["case_key"],
                        n_pairs=n_pairs,
                        replicate_index=replicate,
                        root_seed=config["rng"]["root_seed"],
                    ),
                )

    require(
        len(planned) == EXPECTED_REPLICATES,
        "Unexpected final schedule size",
    )

    return planned


def verify_evidence(folder):
    folder = Path(folder).resolve()

    manifest_before, file_count = check_inventory(
        folder
    )

    require(
        file_sha256(
            folder / "campaign_spec.json"
        ) == CONFIG_SHA256,
        "Wrong final configuration fingerprint",
    )

    config = document(
        folder,
        "campaign_spec.json",
    )

    metadata = document(
        folder,
        "campaign_metadata.json",
    )

    campaign_summary = document(
        folder,
        "campaign_summary.json",
    )

    references = document(
        folder,
        "references.json",
    )

    require(
        config["campaign_id"]
        == metadata["campaign_id"]
        == EXPECTED_CAMPAIGN_ID,
        "Campaign identifier mismatch",
    )

    require(
        config["campaign_role"]
        == metadata["campaign_role"]
        == EXPECTED_CAMPAIGN_ROLE,
        "Campaign role mismatch",
    )

    require(
        metadata["config_sha256"]
        == CONFIG_SHA256,
        "Configuration metadata mismatch",
    )

    require(
        document(
            folder,
            "source_after.json",
        ) == metadata["source"],
        "Before/after source records differ",
    )

    require(
        metadata["source"]["git_status_porcelain"]
        == "",
        "Recorded execution source was not clean",
    )

    require(
        len(config["sampling"]["sample_sizes"])
        == EXPECTED_SAMPLE_SIZES,
        "Unexpected final sample-size count",
    )

    require(
        config["sampling"]["replicates_per_cell"]
        == EXPECTED_REPLICATES_PER_CELL,
        "Unexpected final replicates-per-cell",
    )

    cases = verify_case_design(config)

    require(
        set(references) == set(cases),
        "Missing or extra analytical references",
    )

    compare = Comparison()

    for case_id, case in cases.items():
        compare.context = case_id

        verify_reference(
            references[case_id],
            case,
            compare,
        )

    planned = reconstruct_schedule(config)

    require(
        index_records(
            folder,
            "schedule.jsonl",
        ) == planned,
        "Saved schedule differs from frozen configuration",
    )

    samples = index_records(
        folder,
        "samples.jsonl",
    )

    require(
        set(samples) == set(planned),
        "Missing or extra retained samples",
    )

    # Require exactly started -> completed for every final replicate.
    event_counts = Counter()

    for event in records(
        folder,
        "attempts.jsonl",
    ):
        identifier = event["attempt_id"]

        require(
            identifier in planned,
            "Unscheduled attempt event",
        )

        expected_event = (
            "started"
            if event_counts[identifier] == 0
            else "completed"
        )

        require(
            event_counts[identifier] < 2
            and event["event"] == expected_event,
            f"Unexpected terminal-event sequence: {identifier}",
        )

        event_counts[identifier] += 1

    require(
        set(event_counts) == set(planned)
        and all(
            count == 2
            for count in event_counts.values()
        ),
        "Incomplete attempt accounting",
    )

    seen = set()

    outcome_counts = Counter(
        zero_certificates=0,
        strict_coverage_failures=0,
        tolerance_aware_coverage_failures=0,
        raw_erroneous_exclusions=0,
    )

    exact_optimum_relations = Counter()
    exact_population_limit_relations = Counter()

    for result in records(
        folder,
        "results.jsonl",
    ):
        identifier = result["attempt_id"]
        compare.context = identifier

        require(
            identifier in planned
            and identifier not in seen,
            f"Unscheduled or duplicate result: {identifier}",
        )

        job = planned[identifier]
        case = cases[job["case_id"]]
        saved_reference = references[
            job["case_id"]
        ]
        sample = samples[identifier]["sample"]

        for name, expected in job["seed"].items():
            require(
                type(sample[name]) is int
                and sample[name] == expected,
                f"Seed mismatch "
                f"{identifier}/{name}",
            )

        require(
            sample["seed_material"] == [
                job["seed"][name]
                for name
                in config["rng"]["seed_material_order"]
            ],
            f"Seed-material mismatch: {identifier}",
        )

        for name, expected in (
            (
                "sampler_id",
                "four_state_multinomial_pcg64_v1",
            ),
            (
                "bit_generator",
                "PCG64",
            ),
            (
                "seed_sequence",
                "SeedSequence",
            ),
            (
                "seed_sequence_pool_size",
                4,
            ),
            (
                "draws_per_replicate",
                1,
            ),
            (
                "state_order",
                ["a", "b", "d", "e"],
            ),
            (
                "probabilities",
                [0.25] * 4,
            ),
        ):
            require(
                sample[name] == expected,
                f"Sampling metadata mismatch "
                f"{identifier}/{name}",
            )

        counts = sample["state_counts"]
        n_pairs = job["seed"]["n_pairs"]

        require(
            isinstance(counts, list)
            and len(counts) == 4
            and all(
                type(value) is int
                and value >= 0
                for value in counts
            )
            and sum(counts) == n_pairs,
            f"Invalid counts: {identifier}",
        )

        require(
            result["case_id"]
            == job["case_id"],
            f"Result case mismatch: {identifier}",
        )

        require(
            result["evaluation_id"]
            == "four_state_original_iid_evaluation_v1",
            f"Evaluation identifier mismatch: {identifier}",
        )

        require(
            result["certificate_id"]
            == config["certificate_id"],
            f"Certificate identifier mismatch: {identifier}",
        )

        require(
            result["count_statistics"]["statistics_id"]
            == config["count_statistics_id"],
            f"Count-statistics identifier mismatch: {identifier}",
        )

        require(
            result["count_statistics"]["state_counts"]
            == counts,
            f"Result/sample counts differ: {identifier}",
        )

        independent_statistics = (
            weighted_statistics_binary64(
                case["p"],
                case["dynamics"],
                counts,
            )
        )

        saved_statistics = (
            result["count_statistics"]["statistics"]
        )

        require(
            saved_statistics["n_pairs"]
            == n_pairs,
            f"Statistics sample-size mismatch: {identifier}",
        )

        require(
            saved_statistics["n_classes"]
            == 2,
            f"Statistics class-count mismatch: {identifier}",
        )

        require(
            saved_statistics["class_counts"]
            == independent_statistics["class_sizes"],
            f"Class occupancy mismatch: {identifier}",
        )

        require(
            saved_statistics["simplex_sum_atol"]
            == 1e-12,
            f"Simplex tolerance mismatch: {identifier}",
        )

        for name, expected in (
            (
                "f_hat",
                independent_statistics["f_hat"],
            ),
            (
                "s_hat",
                independent_statistics["s_hat"],
            ),
            (
                "class_successor_means",
                independent_statistics["means"],
            ),
            (
                "class_successor_sse",
                independent_statistics["sses"],
            ),
            (
                "current_max_row_sum_error",
                0.0,
            ),
            (
                "successor_max_row_sum_error",
                0.0,
            ),
        ):
            compare.close(
                saved_statistics[name],
                expected,
                "statistics_" + name,
            )

        certificate = result["certificate"]

        require(
            certificate["n_pairs"]
            == n_pairs,
            f"Certificate sample-size mismatch: {identifier}",
        )

        require(
            certificate["kappa"]
            == case["kappa"],
            f"Certificate kappa mismatch: {identifier}",
        )

        require(
            certificate["delta"]
            == config["evaluation"]["delta"],
            f"Certificate delta mismatch: {identifier}",
        )

        require(
            certificate["f_hat"]
            == saved_statistics["f_hat"]
            and certificate["s_hat"]
            == saved_statistics["s_hat"],
            f"Certificate/statistics input mismatch: {identifier}",
        )

        independent = independent_certificate(
            independent_statistics["f_hat"],
            independent_statistics["s_hat"],
            n_pairs,
            case["kappa"],
            certificate["delta"],
        )

        for name, expected in independent.items():
            compare.close(
                certificate[name],
                expected,
                "certificate_" + name,
            )

        for name in (
            "V_delta",
            "L_kappa_delta",
        ):
            if independent[name] == 0.0:
                require(
                    certificate[name] == 0.0,
                    f"Expected exact zero "
                    f"{identifier}/{name}",
                )

        require(
            0.0 <= certificate["F_delta"] <= 1.0
            and 0.0 <= certificate["V_delta"] <= 1.0,
            f"Certificate component outside [0,1]: {identifier}",
        )

        lower = certificate[
            "L_kappa_delta"
        ]
        optimum = saved_reference[
            "exact_optimal_rmse"
        ]
        population_limit = saved_reference[
            "population_certificate_limit"
        ]
        comparison_atol = config[
            "evaluation"
        ]["comparison_atol"]

        require(
            result["comparison_atol"]
            == comparison_atol,
            "Comparison allowance changed",
        )

        expected_flags = {
            "strict_coverage":
                lower <= optimum,
            "tolerance_aware_coverage":
                lower <= optimum + comparison_atol,
            "zero_certificate":
                lower == 0.0,
        }

        for name, expected in expected_flags.items():
            require(
                type(result[name]) is bool
                and result[name] is expected,
                f"Incorrect flag "
                f"{identifier}/{name}",
            )

        # Signed gaps must be retained exactly in binary64 arithmetic.
        expected_gaps = {
            "gap_to_optimum":
                optimum - lower,
            "population_bound_slack":
                optimum - population_limit,
            "sampling_gap":
                population_limit - lower,
        }

        for name, expected in expected_gaps.items():
            require(
                result[name] == expected,
                f"Signed gap changed or clipped "
                f"{identifier}/{name}",
            )

        exact_population = (
            exact_population_quantities(case)
        )

        exact_optimum = exact_population[
            "exact_optimal_rmse"
        ]
        exact_limit = exact_population[
            "population_certificate_limit"
        ]

        tolerances = config[
            "evaluation"
        ]["exclusion_tolerances"]

        require(
            len(result["exclusions"])
            == len(tolerances),
            f"Missing exclusion decisions: {identifier}",
        )

        for decision, tolerance in zip(
            result["exclusions"],
            tolerances,
        ):
            require(
                decision["tolerance"]
                == tolerance,
                f"Exclusion tolerance mismatch: {identifier}",
            )

            raw_truth = (
                optimum > tolerance
            )

            require(
                decision[
                    "exact_error_exceeds_tolerance"
                ] is raw_truth,
                f"Raw-float ground-truth flag mismatch: {identifier}",
            )

            require(
                decision["excluded"]
                is (lower > tolerance),
                f"Exclusion flag mismatch: {identifier}",
            )

            require(
                decision[
                    "erroneous_exclusion"
                ]
                is (
                    lower > tolerance
                    and not raw_truth
                ),
                f"Raw erroneous-exclusion flag mismatch: {identifier}",
            )

            if decision["erroneous_exclusion"]:
                outcome_counts[
                    "raw_erroneous_exclusions"
                ] += 1

            threshold = tolerance_fraction(
                tolerance
            )

            exact_optimum_relations[
                exact_relation(
                    exact_optimum,
                    threshold,
                )
            ] += 1

            exact_population_limit_relations[
                exact_relation(
                    exact_limit,
                    threshold,
                )
            ] += 1

        outcome_counts[
            "zero_certificates"
        ] += result["zero_certificate"]

        outcome_counts[
            "strict_coverage_failures"
        ] += (
            not result["strict_coverage"]
        )

        outcome_counts[
            "tolerance_aware_coverage_failures"
        ] += (
            not result[
                "tolerance_aware_coverage"
            ]
        )

        seen.add(identifier)

    require(
        seen == set(planned),
        "Missing evaluated replicates",
    )

    require(
        campaign_summary["status"]
        == "completed",
        "Campaign summary is not completed",
    )

    require(
        campaign_summary["exit_code"] == 0,
        "Campaign exit code is nonzero",
    )

    require(
        campaign_summary["source_unchanged"]
        is True,
        "Source changed during final execution",
    )

    require(
        campaign_summary["active_attempt_id"]
        is None,
        "Campaign ended with active attempt",
    )

    for name in (
        "planned",
        "started",
        "sampled",
        "succeeded",
    ):
        require(
            campaign_summary[name]
            == EXPECTED_REPLICATES,
            f"Campaign summary mismatch: {name}",
        )

    for name in (
        "failed",
        "not_started",
        "started_without_terminal_event",
    ):
        require(
            campaign_summary[name] == 0,
            f"Unexpected nonzero summary field: {name}",
        )

    manifest_after, _ = check_inventory(
        folder
    )

    require(
        manifest_after == manifest_before,
        "Evidence inventory changed during verification",
    )

    return dict(
        verifier_id=VERIFIER_ID,
        status="passed",
        verified_replicates=len(seen),
        verified_references=len(references),
        verified_cases=len(cases),
        verified_sample_sizes=len(
            config["sampling"]["sample_sizes"]
        ),
        evidence_files_checked=file_count,
        evidence_manifest_sha256=manifest_before,
        execution_source_commit=metadata[
            "source"
        ]["execution_source_commit"],
        config_sha256=CONFIG_SHA256,
        decimal_precision=PRECISION,
        numerical_rtol=RTOL,
        numerical_atol=ATOL,
        numerical_values_checked=compare.numbers_checked,
        max_absolute_errors=compare.max_errors,
        recorded_outcome_counts=dict(
            outcome_counts
        ),
        exact_optimum_threshold_relations=dict(
            exact_optimum_relations
        ),
        exact_population_limit_threshold_relations=dict(
            exact_population_limit_relations
        ),
        random_samples_regenerated=0,
        evidence_modified=False,
    )


def verifier_provenance():
    path = Path(__file__).resolve()
    root = path.parents[1]

    def git(*args):
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.PIPE,
        )

    require(
        not git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).strip(),
        "Commit the final verifier and tests; "
        "use a clean source tree before recording verification",
    )

    commit = git(
        "rev-parse",
        "HEAD",
    ).decode().strip()

    relative = path.relative_to(
        root
    ).as_posix()

    require(
        hashlib.sha256(
            git(
                "show",
                f"{commit}:{relative}",
            )
        ).hexdigest()
        == file_sha256(path),
        "Verifier bytes are not those committed at HEAD",
    )

    required_test = (
        root
        / "tests"
        / "test_kahkm_four_state_final_numerics.py"
    )

    require(
        required_test.is_file()
        and not required_test.is_symlink(),
        "Final verifier test file is missing",
    )

    test_relative = required_test.relative_to(
        root
    ).as_posix()

    require(
        hashlib.sha256(
            git(
                "show",
                f"{commit}:{test_relative}",
            )
        ).hexdigest()
        == file_sha256(required_test),
        "Final verifier test bytes differ from HEAD",
    )

    return dict(
        verifier_source_commit=commit,
        verifier_path=relative,
        verifier_sha256=file_sha256(path),
        verifier_test_path=test_relative,
        verifier_test_sha256=file_sha256(
            required_test
        ),
        python_version=sys.version,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    try:
        provenance = verifier_provenance()
        report = verify_evidence(
            args.evidence
        )
        report.update(provenance)

        print(
            json.dumps(
                report,
                sort_keys=True,
                indent=2,
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

        return 2


if __name__ == "__main__":
    raise SystemExit(main())
