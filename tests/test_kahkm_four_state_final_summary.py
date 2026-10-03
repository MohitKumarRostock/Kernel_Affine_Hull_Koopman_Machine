"""Tests for final four-state summary generation.

The aggregation fixture contains four selected final-design cases, two sample
sizes, and four hand-constructed result records per cell. It is not experimental
evidence. Expected-size constants are patched only inside these tests.

Threshold expectations use exact Fraction arithmetic. File-pipeline tests use
temporary evidence and mocked numerical-audit/Git interfaces; they test summary
persistence and safeguards, not the scientific validity of synthetic records.
The real 117,000-replicate evidence is never opened.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_four_state_final.py"
CONFIG = ROOT / "reproduction" / "certificates" / "configs" / "four_state_final_v1.json"

spec = importlib.util.spec_from_file_location(
    "_final_summary_under_test",
    SCRIPT,
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def direct_quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)

    return (
        ordered[lower]
        + (position - lower)
        * (ordered[upper] - ordered[lower])
    )


def exact_quantities(case):
    p = Fraction(
        case["p_exact"]["numerator"],
        case["p_exact"]["denominator"],
    )

    if case["dynamics"] == "identity":
        return Fraction(0), Fraction(0)

    return (
        p - Fraction(1, 2),
        max(
            Fraction(0),
            3 * p - Fraction(5, 2),
        ),
    )


def make_fixture():
    full = json.loads(
        CONFIG.read_text(encoding="utf-8")
    )

    wanted = {
        "identity_p_31_over_32",
        "conflicting_successors_p_4_over_5",
        "conflicting_successors_p_14_over_15",
        "conflicting_successors_p_99_over_100",
    }

    cases = [
        deepcopy(case)
        for case in full["cases"]
        if case["case_id"] in wanted
    ]

    cases.sort(
        key=lambda case: case["case_key"]
    )

    config = deepcopy(full)
    config["campaign_id"] = "unit_test_final_summary_only"
    config["cases"] = cases
    config["sampling"]["sample_sizes"] = [128, 4096]
    config["sampling"]["replicates_per_cell"] = 4

    references = {}
    rows = []

    lower_values = {
        "identity_p_31_over_32": {
            128: [0.0, 0.0, 0.0, 0.0],
            4096: [0.0, 0.0, 0.0, 0.0],
        },
        "conflicting_successors_p_4_over_5": {
            128: [0.0, 0.0, 0.0, 0.0],
            4096: [0.0, 0.0, 0.0, 0.0],
        },
        "conflicting_successors_p_14_over_15": {
            128: [0.0, 0.10, 0.20, 0.25],
            4096: [0.20, 0.28, 0.30, 0.32],
        },
        "conflicting_successors_p_99_over_100": {
            128: [0.0, 0.10, 0.20, 0.30],
            4096: [0.40, 0.45, 0.48, 0.50],
        },
    }

    for case in cases:
        optimum, population_limit = exact_quantities(
            case
        )

        references[case["case_id"]] = {
            "exact_optimal_rmse": float(optimum),
            "population_certificate_limit":
                float(population_limit),
        }

        for n_pairs in (128, 4096):
            for replicate, lower in enumerate(
                lower_values[case["case_id"]][n_pairs]
            ):
                identifier = (
                    f"c{case['case_key']}_"
                    f"m{n_pairs}_"
                    f"r{replicate}"
                )

                optimum_float = float(optimum)
                population_float = float(
                    population_limit
                )

                rows.append({
                    "attempt_id": identifier,
                    "case_id": case["case_id"],
                    "certificate": {
                        "n_pairs": n_pairs,
                        "L_kappa_delta": lower,
                        "F_delta": 0.01,
                        "V_delta": 0.5,
                        "s_hat": 0.5,
                    },
                    "gap_to_optimum":
                        optimum_float - lower,
                    "sampling_gap":
                        population_float - lower,
                    "zero_certificate":
                        lower == 0.0,
                    "strict_coverage":
                        lower <= optimum_float,
                    "tolerance_aware_coverage":
                        lower <= optimum_float + 1e-12,
                    "exclusions": [
                        {
                            "tolerance": tolerance,
                            "excluded":
                                lower > tolerance,
                            "exact_error_exceeds_tolerance":
                                optimum_float > tolerance,
                            "erroneous_exclusion":
                                lower > tolerance
                                and not (
                                    optimum_float > tolerance
                                ),
                        }
                        for tolerance
                        in config["evaluation"][
                            "exclusion_tolerances"
                        ]
                    ],
                })

    return config, references, rows


class FourStateFinalSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="final-summary-test-"
        )
        self.addCleanup(temporary.cleanup)

        self.base = Path(
            temporary.name
        ).resolve()

        self.source = self.base / "source"
        self.source.mkdir()

        self.evidence = self.base / "synthetic-evidence"
        self.evidence.mkdir()

        (
            self.config,
            self.references,
            self.rows,
        ) = make_fixture()

        (
            self.evidence
            / "campaign_spec.json"
        ).write_text(
            json.dumps(self.config) + "\n",
            encoding="utf-8",
        )

        (
            self.evidence
            / "references.json"
        ).write_text(
            json.dumps(self.references) + "\n",
            encoding="utf-8",
        )

        (
            self.evidence
            / "results.jsonl"
        ).write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in self.rows
            ),
            encoding="utf-8",
        )

        self.provenance = {
            "summary_source_commit":
                "b" * 40,
            "source_sha256": {},
            "python_version": sys.version,
            "package_versions": {
                "numpy": summary.np.__version__,
                "scipy": summary.scipy.__version__,
            },
        }

        self.audit = {
            "status": "passed",
            "verified_replicates": 32,
            "verified_cases": 4,
            "verified_sample_sizes": 2,
            "evidence_manifest_sha256":
                "1" * 64,
            "execution_source_commit":
                "a" * 40,
            "config_sha256":
                summary.EXPECTED_CONFIG_SHA256,
        }

        self.api = SimpleNamespace(
            verify_evidence=Mock(
                return_value=self.audit
            ),
            document=lambda folder, name:
                json.loads(
                    (folder / name).read_text(
                        encoding="utf-8"
                    )
                ),
            records=lambda folder, name: (
                json.loads(line)
                for line in (
                    folder / name
                ).read_text(
                    encoding="utf-8"
                ).splitlines()
            ),
            check_inventory=Mock(
                return_value=("1" * 64, 3)
            ),
        )

    def patched_sizes(self):
        return ExitStack()

    def aggregate(
        self,
        config=None,
        references=None,
        rows=None,
    ):
        with patch.object(
            summary,
            "EXPECTED_CAMPAIGN_ID",
            "unit_test_final_summary_only",
        ), patch.object(
            summary,
            "EXPECTED_CASES",
            4,
        ), patch.object(
            summary,
            "EXPECTED_SAMPLE_SIZES",
            2,
        ), patch.object(
            summary,
            "EXPECTED_REPLICATES_PER_CELL",
            4,
        ), patch.object(
            summary,
            "EXPECTED_REPLICATES",
            32,
        ), patch.object(
            summary,
            "EXPECTED_CELL_ROWS",
            8,
        ), patch.object(
            summary,
            "EXPECTED_EXCLUSION_ROWS",
            48,
        ):
            return summary.summarize_cells(
                (
                    self.config
                    if config is None
                    else config
                ),
                (
                    self.references
                    if references is None
                    else references
                ),
                (
                    self.rows
                    if rows is None
                    else rows
                ),
            )

    def run_tiny(
        self,
        output,
        source_responses=None,
    ):
        source_settings = (
            {
                "return_value":
                    self.provenance
            }
            if source_responses is None
            else {
                "side_effect":
                    source_responses
            }
        )

        with patch.object(
            summary,
            "ROOT",
            self.source,
        ), patch.object(
            summary,
            "EXPECTED_CAMPAIGN_ID",
            "unit_test_final_summary_only",
        ), patch.object(
            summary,
            "EXPECTED_CASES",
            4,
        ), patch.object(
            summary,
            "EXPECTED_SAMPLE_SIZES",
            2,
        ), patch.object(
            summary,
            "EXPECTED_REPLICATES_PER_CELL",
            4,
        ), patch.object(
            summary,
            "EXPECTED_REPLICATES",
            32,
        ), patch.object(
            summary,
            "EXPECTED_CELL_ROWS",
            8,
        ), patch.object(
            summary,
            "EXPECTED_EXCLUSION_ROWS",
            48,
        ), patch.object(
            summary,
            "source_provenance",
            **source_settings,
        ), patch.object(
            summary,
            "load_verifier",
            return_value=self.api,
        ):
            return summary.summarize_evidence(
                self.evidence,
                output,
            )

    def source_fixture(
        self,
        wrong_versions=False,
    ):
        names = (
            "scripts/summarize_four_state_final.py",
            "scripts/verify_four_state_final_numerics.py",
            "tests/test_kahkm_four_state_final_summary.py",
            "requirements-lock-arm64.txt",
        )

        committed = {}

        for name in names:
            path = self.source / name

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if name.endswith(".txt"):
                scipy_version = (
                    "0.0"
                    if wrong_versions
                    else summary.scipy.__version__
                )

                data = (
                    f"numpy=={summary.np.__version__}\n"
                    f"scipy=={scipy_version}\n"
                ).encode("utf-8")
            else:
                data = (
                    "# synthetic source: "
                    + name
                    + "\n"
                ).encode("utf-8")

            path.write_bytes(data)
            committed[name] = data

        def git(command, **kwargs):
            args = command[1:]

            if args[0] == "status":
                return b""

            if args == ["rev-parse", "HEAD"]:
                return b"b" * 40 + b"\n"

            if args[0] == "show":
                commit, name = args[1].split(
                    ":",
                    1,
                )

                self.assertEqual(
                    commit,
                    "b" * 40,
                )

                return committed[name]

            raise AssertionError(
                "Unexpected Git command: "
                + repr(command)
            )

        return committed, git

    # 1
    def test_frozen_final_configuration_constants_and_design(self):
        full = json.loads(
            CONFIG.read_text(encoding="utf-8")
        )

        self.assertEqual(
            full["campaign_id"],
            summary.EXPECTED_CAMPAIGN_ID,
        )
        self.assertEqual(
            full["campaign_role"],
            summary.EXPECTED_CAMPAIGN_ROLE,
        )
        self.assertEqual(
            len(full["cases"]),
            summary.EXPECTED_CASES,
        )
        self.assertEqual(
            len(
                full["sampling"]["sample_sizes"]
            ),
            summary.EXPECTED_SAMPLE_SIZES,
        )
        self.assertEqual(
            full["sampling"][
                "replicates_per_cell"
            ],
            summary.EXPECTED_REPLICATES_PER_CELL,
        )

    # 2
    def test_exact_population_formulas(self):
        full = json.loads(
            CONFIG.read_text(encoding="utf-8")
        )

        for case in full["cases"]:
            with self.subTest(
                case=case["case_id"]
            ):
                optimum, limit = (
                    summary.exact_population_quantities(
                        case
                    ).values()
                )

                p = summary.exact_design_p(
                    case
                )

                if case["dynamics"] == "identity":
                    self.assertEqual(
                        optimum,
                        0,
                    )
                    self.assertEqual(
                        limit,
                        0,
                    )
                else:
                    self.assertEqual(
                        optimum,
                        p - Fraction(1, 2),
                    )
                    self.assertEqual(
                        limit,
                        max(
                            Fraction(0),
                            3 * p
                            - Fraction(5, 2),
                        ),
                    )

    # 3
    def test_exact_relations_identify_frozen_boundaries(self):
        full = json.loads(
            CONFIG.read_text(encoding="utf-8")
        )

        by_id = {
            case["case_id"]: case
            for case in full["cases"]
        }

        checks = (
            (
                "conflicting_successors_p_4_over_5",
                "exact_optimal_rmse",
                0.30,
            ),
            (
                "conflicting_successors_p_19_over_20",
                "exact_optimal_rmse",
                0.45,
            ),
            (
                "conflicting_successors_p_17_over_20",
                "population_certificate_limit",
                0.05,
            ),
            (
                "conflicting_successors_p_13_over_15",
                "population_certificate_limit",
                0.10,
            ),
            (
                "conflicting_successors_p_9_over_10",
                "population_certificate_limit",
                0.20,
            ),
            (
                "conflicting_successors_p_14_over_15",
                "population_certificate_limit",
                0.30,
            ),
            (
                "conflicting_successors_p_29_over_30",
                "population_certificate_limit",
                0.40,
            ),
            (
                "conflicting_successors_p_59_over_60",
                "population_certificate_limit",
                0.45,
            ),
        )

        for case_id, quantity, tolerance in checks:
            with self.subTest(
                case=case_id,
                quantity=quantity,
            ):
                value = (
                    summary.exact_population_quantities(
                        by_id[case_id]
                    )[quantity]
                )

                self.assertEqual(
                    summary.exact_relation(
                        value,
                        summary.tolerance_fraction(
                            tolerance
                        ),
                    ),
                    "boundary",
                )

    # 4
    def test_tolerance_fraction_uses_decimal_threshold_semantics(self):
        self.assertEqual(
            summary.tolerance_fraction(0.3),
            Fraction(3, 10),
        )
        self.assertEqual(
            summary.tolerance_fraction(0.45),
            Fraction(9, 20),
        )

        for value in (
            True,
            "0.3",
            math.nan,
            math.inf,
            -math.inf,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    summary.tolerance_fraction(
                        value
                    )

    # 5
    def test_binomial_endpoints(self):
        for trials in (1, 4, 1000):
            with self.subTest(trials=trials):
                zero = summary.binomial_rate(
                    [False] * trials,
                    0.95,
                )
                full = summary.binomial_rate(
                    [True] * trials,
                    0.95,
                )

                self.assertEqual(
                    zero["rate"],
                    0.0,
                )
                self.assertEqual(
                    full["rate"],
                    1.0,
                )
                self.assertEqual(
                    zero["ci_low"],
                    0.0,
                )
                self.assertEqual(
                    full["ci_high"],
                    1.0,
                )
                self.assertGreater(
                    zero["ci_high"],
                    0.0,
                )
                self.assertLess(
                    full["ci_low"],
                    1.0,
                )

    # 6
    def test_binomial_interior_tail_equations(self):
        successes = 3
        trials = 7

        result = summary.binomial_rate(
            [True] * successes
            + [False] * (
                trials - successes
            ),
            0.95,
        )

        lower = result["ci_low"]
        upper = result["ci_high"]

        lower_tail = math.fsum(
            math.comb(trials, j)
            * lower ** j
            * (1 - lower)
            ** (trials - j)
            for j in range(
                successes,
                trials + 1,
            )
        )

        upper_tail = math.fsum(
            math.comb(trials, j)
            * upper ** j
            * (1 - upper)
            ** (trials - j)
            for j in range(
                successes + 1
            )
        )

        self.assertAlmostEqual(
            lower_tail,
            0.025,
            delta=5e-12,
        )
        self.assertAlmostEqual(
            upper_tail,
            0.025,
            delta=5e-12,
        )

    # 7
    def test_binomial_invalid_inputs(self):
        invalid_flags = (
            [],
            [0, 1],
            [True, None],
            ["true"],
        )

        for values in invalid_flags:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    summary.binomial_rate(
                        values,
                        0.95,
                    )

        for confidence in (
            0.0,
            1.0,
            True,
            "0.95",
            math.nan,
        ):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    summary.binomial_rate(
                        [True],
                        confidence,
                    )

    # 8
    def test_distribution_quantiles_and_signed_values(self):
        values = [
            -3.0,
            -1.0,
            0.0,
            2.0,
            7.0,
        ]

        result = summary.distribution(
            values
        )

        for name, probability in (
            ("q05", 0.05),
            ("median", 0.5),
            ("q95", 0.95),
        ):
            self.assertAlmostEqual(
                result[name],
                direct_quantile(
                    values,
                    probability,
                ),
                delta=1e-14,
            )

        self.assertEqual(
            result["minimum"],
            -3.0,
        )
        self.assertEqual(
            result["maximum"],
            7.0,
        )

    # 9
    def test_distribution_invalid_inputs(self):
        for values in (
            [],
            [True],
            ["1"],
            [None],
            [math.nan],
            [math.inf],
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    summary.distribution(
                        values
                    )

    # 10
    def test_summary_cell_and_trial_accounting(self):
        cells, exclusions = self.aggregate()

        self.assertEqual(
            len(cells),
            8,
        )
        self.assertEqual(
            len(exclusions),
            48,
        )
        self.assertEqual(
            sum(
                row["replicates"]
                for row in cells
            ),
            32,
        )

        for row in cells:
            self.assertEqual(
                row["replicates"],
                4,
            )

            for flag in (
                "zero_certificate",
                "strict_coverage",
                "tolerance_aware_coverage",
            ):
                self.assertEqual(
                    row[
                        flag
                        + "_trials"
                    ],
                    4,
                )

        self.assertTrue(
            all(
                row["trials"] == 4
                for row in exclusions
            )
        )

    # 11
    def test_structural_detectability_is_distinct_from_finite_power(self):
        _, exclusions = self.aggregate()

        by_key = {
            (
                row["case_id"],
                row["n_pairs"],
                row["tolerance"],
            ): row
            for row in exclusions
        }

        # p=4/5: true error equals .30, while population limit is zero.
        boundary = by_key[
            (
                "conflicting_successors_p_4_over_5",
                4096,
                0.30,
            )
        ]

        self.assertEqual(
            boundary["exact_optimum_relation"],
            "boundary",
        )
        self.assertEqual(
            boundary["population_limit_relation"],
            "below",
        )
        self.assertEqual(
            boundary["interpretation"],
            "boundary_exclusion_rate",
        )
        self.assertEqual(
            boundary["population_detectability"],
            "population_bound_cannot_exclude",
        )
        self.assertEqual(
            boundary["rate"],
            0.0,
        )

        # p=99/100 at .45: population limit is .47, so exclusion is
        # population-detectable, while finite-sample rate is empirical.
        detectable = by_key[
            (
                "conflicting_successors_p_99_over_100",
                4096,
                0.45,
            )
        ]

        self.assertEqual(
            detectable["exact_optimum_relation"],
            "above",
        )
        self.assertEqual(
            detectable["population_limit_relation"],
            "above",
        )
        self.assertEqual(
            detectable["population_detectability"],
            "population_bound_can_exclude",
        )
        self.assertGreater(
            detectable["rate"],
            0.0,
        )

    # 12
    def test_exact_optimum_and_population_limit_boundaries_are_separate(self):
        _, exclusions = self.aggregate()

        by_key = {
            (
                row["case_id"],
                row["n_pairs"],
                row["tolerance"],
            ): row
            for row in exclusions
        }

        # p=14/15: optimum > .30 but population certificate limit = .30.
        row = by_key[
            (
                "conflicting_successors_p_14_over_15",
                4096,
                0.30,
            )
        ]

        self.assertEqual(
            row["exact_optimum_relation"],
            "above",
        )
        self.assertEqual(
            row["population_limit_relation"],
            "boundary",
        )
        self.assertEqual(
            row["interpretation"],
            "exclusion_power",
        )
        self.assertEqual(
            row["population_detectability"],
            "population_bound_boundary",
        )

    # 13
    def test_zero_certificates_and_signed_gaps_are_retained(self):
        cells, _ = self.aggregate()

        by_key = {
            (
                row["case_id"],
                row["n_pairs"],
            ): row
            for row in cells
        }

        identity = by_key[
            (
                "identity_p_31_over_32",
                4096,
            )
        ]

        self.assertEqual(
            identity["zero_certificate_rate"],
            1.0,
        )
        self.assertIsNone(
            identity[
                "median_certificate_over_optimum"
            ]
        )
        self.assertIsNone(
            identity[
                "median_certificate_over_population_limit"
            ]
        )

        hard = by_key[
            (
                "conflicting_successors_p_99_over_100",
                4096,
            )
        ]

        self.assertLess(
            hard["gap_to_optimum_minimum"],
            0.0,
        )
        self.assertLess(
            hard["sampling_gap_minimum"],
            0.0,
        )

    # 14
    def test_coverage_failure_is_retained_in_probability_summary(self):
        cells, _ = self.aggregate()

        row = next(
            row
            for row in cells
            if (
                row["case_id"]
                == "conflicting_successors_p_99_over_100"
                and row["n_pairs"] == 4096
            )
        )

        self.assertEqual(
            row["strict_coverage_count"],
            3,
        )
        self.assertEqual(
            row["strict_coverage_rate"],
            0.75,
        )
        self.assertGreater(
            row["strict_coverage_ci_high"],
            0.75,
        )
        self.assertLess(
            row["strict_coverage_ci_low"],
            0.75,
        )

    # 15
    def test_order_invariance(self):
        expected = self.aggregate()

        config = deepcopy(
            self.config
        )
        config["cases"].reverse()
        config["sampling"][
            "sample_sizes"
        ].reverse()

        actual = self.aggregate(
            config=config,
            rows=reversed(
                self.rows
            ),
        )

        self.assertEqual(
            actual,
            expected,
        )

    # 16
    def test_missing_duplicate_and_unexpected_results_are_rejected(self):
        cases = [
            self.rows[1:],
            self.rows + self.rows[:1],
        ]

        changed = deepcopy(
            self.rows
        )
        changed[0]["attempt_id"] = "unknown"
        cases.append(changed)

        changed = deepcopy(
            self.rows
        )
        changed[0][
            "certificate"
        ]["n_pairs"] = 999
        cases.append(changed)

        for rows in cases:
            with self.subTest(
                length=len(rows)
            ):
                with self.assertRaises(ValueError):
                    self.aggregate(
                        rows=rows
                    )

    # 17
    def test_invalid_configuration_and_reference_sets_are_rejected(self):
        changed = deepcopy(
            self.config
        )
        changed["campaign_role"] = "pilot"

        with self.assertRaises(ValueError):
            self.aggregate(
                config=changed
            )

        changed = deepcopy(
            self.config
        )
        changed["sampling"][
            "sample_sizes"
        ] = [128, 128]

        with self.assertRaises(ValueError):
            self.aggregate(
                config=changed
            )

        with self.assertRaises(ValueError):
            self.aggregate(
                references={}
            )

    # 18
    def test_csv_and_json_writers_are_exclusive(self):
        table = self.base / "table.csv"

        summary.write_csv(
            table,
            [
                {
                    "name": "test,quoted",
                    "ratio": None,
                    "value": -0.25,
                }
            ],
        )

        with table.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(
                csv.DictReader(handle)
            )

        self.assertEqual(
            rows,
            [
                {
                    "name": "test,quoted",
                    "ratio": "",
                    "value": "-0.25",
                }
            ],
        )

        original = table.read_bytes()

        with self.assertRaises(FileExistsError):
            summary.write_csv(
                table,
                [{"other": 1}],
            )

        self.assertEqual(
            table.read_bytes(),
            original,
        )

        record = self.base / "record.json"

        summary.write_json(
            record,
            {"a": 1},
        )

        with self.assertRaises(FileExistsError):
            summary.write_json(
                record,
                {"a": 2},
            )

    # 19
    def test_source_provenance_accepts_committed_bytes_and_locked_versions(self):
        committed, git = (
            self.source_fixture()
        )

        with patch.object(
            summary,
            "ROOT",
            self.source,
        ), patch.object(
            summary.subprocess,
            "check_output",
            side_effect=git,
        ):
            result = (
                summary.source_provenance()
            )

        self.assertEqual(
            result[
                "summary_source_commit"
            ],
            "b" * 40,
        )

        self.assertEqual(
            result[
                "source_sha256"
            ],
            {
                name:
                    hashlib.sha256(
                        data
                    ).hexdigest()
                for name, data
                in committed.items()
            },
        )

    # 20
    def test_source_provenance_rejects_dirty_or_dependency_mismatch(self):
        with patch.object(
            summary,
            "ROOT",
            self.source,
        ), patch.object(
            summary.subprocess,
            "check_output",
            return_value=b"?? new.py\n",
        ):
            with self.assertRaises(ValueError):
                summary.source_provenance()

        _, git = self.source_fixture(
            wrong_versions=True
        )

        with patch.object(
            summary,
            "ROOT",
            self.source,
        ), patch.object(
            summary.subprocess,
            "check_output",
            side_effect=git,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "differ from reference lockfile",
            ):
                summary.source_provenance()

    # 21
    def test_complete_tiny_summary_package_is_reproducible_and_checksummed(self):
        outputs = [
            self.base / "summary-a",
            self.base / "summary-b",
        ]

        for output in outputs:
            metadata = self.run_tiny(
                output
            )

            self.assertEqual(
                metadata["status"],
                "completed",
            )
            self.assertEqual(
                metadata["cell_rows"],
                8,
            )
            self.assertEqual(
                metadata["exclusion_rows"],
                48,
            )
            self.assertEqual(
                metadata[
                    "summarized_replicates"
                ],
                32,
            )
            self.assertIs(
                metadata[
                    "exact_rational_threshold_interpretation"
                ],
                True,
            )
            self.assertEqual(
                metadata[
                    "random_samples_generated"
                ],
                0,
            )
            self.assertIs(
                metadata[
                    "evidence_modified"
                ],
                False,
            )

            inventory = {}

            for line in (
                output
                / "SHA256SUMS"
            ).read_text(
                encoding="utf-8"
            ).splitlines():
                digest, name = line.split(
                    "  ",
                    1,
                )
                inventory[name] = digest

            self.assertEqual(
                set(inventory),
                {
                    "README.md",
                    "cell_summary.csv",
                    "exclusion_summary.csv",
                    "numerical_audit.json",
                    "summary_metadata.json",
                },
            )

            for name, digest in inventory.items():
                self.assertEqual(
                    hashlib.sha256(
                        (
                            output
                            / name
                        ).read_bytes()
                    ).hexdigest(),
                    digest,
                )

            self.assertIn(
                "exact rational p metadata",
                (
                    output
                    / "README.md"
                ).read_text(
                    encoding="utf-8"
                ),
            )

        for name in (
            "cell_summary.csv",
            "exclusion_summary.csv",
            "numerical_audit.json",
        ):
            self.assertEqual(
                (
                    outputs[0]
                    / name
                ).read_bytes(),
                (
                    outputs[1]
                    / name
                ).read_bytes(),
            )

    # 22
    def test_output_guards_reject_reuse_and_nested_locations(self):
        existing = self.base / "existing"
        existing.mkdir()

        (
            existing
            / "keep.txt"
        ).write_text(
            "keep",
            encoding="utf-8",
        )

        for output in (
            existing,
            self.source,
            self.source / "nested",
            self.evidence,
            self.evidence / "nested",
        ):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    self.run_tiny(
                        output
                    )

        self.assertEqual(
            (
                existing
                / "keep.txt"
            ).read_text(
                encoding="utf-8"
            ),
            "keep",
        )

    # 23
    def test_audit_failure_and_replicate_mismatch_abort_before_output(self):
        output = self.base / "audit-failure"

        self.api.verify_evidence.side_effect = (
            ValueError(
                "injected audit failure"
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "audit failure",
        ):
            self.run_tiny(
                output
            )

        self.assertFalse(
            output.exists()
        )

        self.api.verify_evidence.side_effect = None
        self.api.verify_evidence.return_value = dict(
            self.audit,
            verified_replicates=31,
        )

        output = self.base / "count-failure"

        with self.assertRaisesRegex(
            ValueError,
            "replicate count mismatch",
        ):
            self.run_tiny(
                output
            )

        self.assertFalse(
            output.exists()
        )

    # 24
    def test_source_or_evidence_change_during_aggregation_aborts(self):
        changed = dict(
            self.provenance,
            summary_source_commit="c" * 40,
        )

        output = self.base / "changed-source"

        with self.assertRaisesRegex(
            ValueError,
            "Source changed during aggregation",
        ):
            self.run_tiny(
                output,
                [
                    self.provenance,
                    changed,
                ],
            )

        self.assertFalse(
            output.exists()
        )

        self.api.check_inventory.return_value = (
            "9" * 64,
            3,
        )

        output = self.base / "changed-evidence"

        with self.assertRaisesRegex(
            ValueError,
            "Evidence changed during aggregation",
        ):
            self.run_tiny(
                output
            )

        self.assertFalse(
            output.exists()
        )

    # 25
    def test_cli_and_module_import_are_side_effect_free(self):
        stdout = io.StringIO()

        with patch.object(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--evidence",
                str(self.evidence),
                "--output",
                str(self.base / "cli"),
            ],
        ), patch.object(
            summary,
            "summarize_evidence",
            return_value={
                "status": "completed"
            },
        ) as run, redirect_stdout(stdout):

            self.assertEqual(
                summary.main(),
                0,
            )

        self.assertEqual(
            json.loads(
                stdout.getvalue()
            ),
            {
                "status": "completed"
            },
        )

        run.assert_called_once()

        stderr = io.StringIO()
        stdout = io.StringIO()

        with patch.object(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--evidence",
                str(self.evidence),
                "--output",
                str(self.base / "cli-failure"),
            ],
        ), patch.object(
            summary,
            "summarize_evidence",
            side_effect=OSError(
                "injected failure"
            ),
        ), redirect_stdout(stdout), \
             redirect_stderr(stderr):

            self.assertEqual(
                summary.main(),
                2,
            )

        self.assertEqual(
            stdout.getvalue(),
            "",
        )
        self.assertIn(
            "Retain any created output directory",
            stderr.getvalue(),
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    Path,
                    "read_text",
                    side_effect=AssertionError(
                        "unexpected file read"
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    Path,
                    "read_bytes",
                    side_effect=AssertionError(
                        "unexpected file read"
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    summary.subprocess,
                    "check_output",
                    side_effect=AssertionError(
                        "unexpected Git access"
                    ),
                )
            )

            for name in (
                "SeedSequence",
                "PCG64",
                "Generator",
                "default_rng",
                "multinomial",
                "seed",
            ):
                stack.enter_context(
                    patch.object(
                        summary.np.random,
                        name,
                        side_effect=AssertionError(
                            "unexpected RNG use"
                        ),
                    )
                )

            loaded = runpy.run_path(
                str(SCRIPT),
                run_name="_final_summary_import_test_",
            )

        self.assertEqual(
            loaded["SUMMARY_ID"],
            summary.SUMMARY_ID,
        )


if __name__ == "__main__":
    unittest.main()
