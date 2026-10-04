from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import scripts.summarize_dynamical_confirmatory as summary


class FakeVerifier:
    def __init__(
        self,
        evidence,
        report,
        checksums,
        results,
    ):
        self.evidence = Path(
            evidence
        )
        self.report = report
        self.checksums = checksums
        self.results = results

    def verify_evidence(
        self,
        evidence,
    ):
        self._require_evidence(
            evidence
        )

        return (
            self.report,
            self.checksums,
        )

    def jsonl_records(
        self,
        path,
    ):
        if Path(
            path
        ).name != "results.jsonl":
            raise AssertionError(
                "Unexpected JSONL path"
            )

        return list(
            self.results
        )

    def check_inventory(
        self,
        evidence,
    ):
        self._require_evidence(
            evidence
        )

        return {
            "sha256sums_fingerprint":
                summary.EXPECTED_EVIDENCE_FINGERPRINT,

            "verified_file_count":
                344,

            "files":
                {},
        }

    def _require_evidence(
        self,
        evidence,
    ):
        if Path(
            evidence
        ).resolve() != self.evidence.resolve():
            raise AssertionError(
                "Unexpected evidence directory"
            )


def certificate(
    *,
    kappa,
    L,
    f_hat,
    s_hat,
):
    return {
        "kappa":
            kappa,

        "delta":
            0.05,

        "f_hat":
            f_hat,

        "s_hat":
            s_hat,

        "r_delta":
            0.0002251513338692588,

        "F_delta":
            f_hat
            + 0.003,

        "V_delta":
            max(
                0.0,
                s_hat
                - 0.02,
            ),

        "L_kappa_delta":
            L,

        "n_pairs":
            16384,
    }


def make_record(
    *,
    mode,
    replicate,
    spectral_norm,
    L_primary,
    rmse,
):
    mode_key = (
        1
        if mode
        == "matched_stochastic"
        else 2
    )

    f_hat = (
        0.020
        + 0.0001
        * replicate
    )

    s_hat = (
        0.115
        + 0.0002
        * replicate
    )

    kappas = [
        1.0,
        spectral_norm,
        math.sqrt(
            2.0
        ),
        2.0
        * spectral_norm,
    ]

    bounds = [
        L_primary
        + 0.004,
        L_primary,
        max(
            0.0,
            L_primary
            - 0.084,
        ),
        0.0,
    ]

    budgets = []

    for index, (
        kappa,
        bound,
    ) in enumerate(
        zip(
            kappas,
            bounds,
            strict=True,
        )
    ):
        feasible = (
            index
            != 0
        )

        budgets.append(
            {
                "kappa":
                    kappa,

                "certificate":
                    certificate(
                        kappa=kappa,
                        L=bound,
                        f_hat=f_hat,
                        s_hat=s_hat,
                    ),

                "frozen_predictor_within_budget":
                    feasible,

                "frozen_predictor_rmse_minus_bound":
                    rmse
                    - bound,
            }
        )

    return {
        "attempt_id":
            (
                f"s2_q{mode_key}_"
                f"m16384_"
                f"r{replicate:03d}"
            ),

        "system":
            "vanderpol",

        "evaluation_mode":
            mode,

        "sample_size":
            16384,

        "replicate_index":
            replicate,

        "statistics":
            {
                "f_hat":
                    f_hat,

                "s_hat":
                    s_hat,

                "current_max_row_sum_error":
                    1e-16,

                "successor_max_row_sum_error":
                    1e-16,

                "simplex_sum_atol":
                    1e-12,
            },

        "frozen_predictor_spectral_norm":
            spectral_norm,

        "frozen_predictor_mse":
            rmse
            * rmse,

        "frozen_predictor_rmse":
            rmse,

        "budget_certificates":
            budgets,
    }


class DynamicalConfirmatorySummaryTests(
    unittest.TestCase
):
    def setUp(
        self,
    ):
        self.spectral_norm = (
            1.0203452494452896
        )

        self.config = {
            "campaign_id":
                summary.EXPECTED_CAMPAIGN_ID,

            "campaign_role":
                summary.EXPECTED_CAMPAIGN_ROLE,

            "certificate":
                {
                    "delta":
                        0.05,
                },

            "confirmatory_design":
                {
                    "systems":
                        [
                            "vanderpol"
                        ],

                    "sample_sizes":
                        [
                            16384
                        ],

                    "replicates_per_cell":
                        32,

                    "planned_attempts":
                        64,
                },

            "pair_sampling":
                {
                    "evaluation_modes":
                        [
                            "matched_stochastic",
                            "deterministic",
                        ],
                },
        }

        self.representation = {
            "system":
                "vanderpol",

            "omega":
                128.0,

            "spectral_norm":
                self.spectral_norm,

            "norm_budgets":
                [
                    1.0,
                    self.spectral_norm,
                    math.sqrt(
                        2.0
                    ),
                    2.0
                    * self.spectral_norm,
                ],
        }

        self.results = []

        for mode in summary.EXPECTED_MODES:
            for replicate in range(
                32
            ):
                self.results.append(
                    make_record(
                        mode=mode,
                        replicate=replicate,
                        spectral_norm=
                            self.spectral_norm,
                        L_primary=
                            0.080
                            + 0.0001
                            * replicate,
                        rmse=
                            0.270
                            + 0.0001
                            * replicate,
                    )
                )

    def test_frozen_constants(
        self,
    ):
        self.assertEqual(
            summary.EXPECTED_REPLICATES,
            64,
        )

        self.assertEqual(
            summary.EXPECTED_SAMPLE_SIZE,
            16384,
        )

        self.assertEqual(
            summary.PRIMARY_BUDGET_ROLE,
            "fitted_operator_norm",
        )

        self.assertEqual(
            summary.EXPECTED_EXECUTION_COMMIT,
            "18dc9f3b965e2cc634764e05785ff60d5fbd6422",
        )

        self.assertEqual(
            summary.EXPECTED_VERIFIER_COMMIT,
            "090eac5fa2a28879378d165c898a875f11573aac",
        )

    def test_distribution_linear_quantiles_and_signed_values(
        self,
    ):
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

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        for key, probability in (
            (
                "q05",
                0.05,
            ),
            (
                "q25",
                0.25,
            ),
            (
                "median",
                0.5,
            ),
            (
                "q75",
                0.75,
            ),
            (
                "q95",
                0.95,
            ),
        ):
            self.assertAlmostEqual(
                result[
                    key
                ],
                float(
                    np.quantile(
                        array,
                        probability,
                        method="linear",
                    )
                ),
                places=14,
            )

        self.assertEqual(
            result[
                "minimum"
            ],
            -3.0,
        )

        self.assertEqual(
            result[
                "maximum"
            ],
            7.0,
        )

    def test_binomial_endpoints_and_empty_feasible_set(
        self,
    ):
        all_pass = summary.binomial_rate(
            [
                True
            ]
            * 32
        )

        self.assertEqual(
            all_pass[
                "count"
            ],
            32,
        )

        self.assertEqual(
            all_pass[
                "trials"
            ],
            32,
        )

        self.assertEqual(
            all_pass[
                "rate"
            ],
            1.0,
        )

        self.assertLess(
            all_pass[
                "ci_low"
            ],
            1.0,
        )

        self.assertEqual(
            all_pass[
                "ci_high"
            ],
            1.0,
        )

        empty = summary.binomial_rate(
            []
        )

        self.assertEqual(
            empty[
                "count"
            ],
            0,
        )

        self.assertEqual(
            empty[
                "trials"
            ],
            0,
        )

        self.assertIsNone(
            empty[
                "rate"
            ]
        )

        self.assertIsNone(
            empty[
                "ci_low"
            ]
        )

        self.assertIsNone(
            empty[
                "ci_high"
            ]
        )

    def test_budget_roles_and_primary_index(
        self,
    ):
        roles = summary.budget_roles(
            self.spectral_norm
        )

        self.assertEqual(
            [
                role
                for role, _value
                in roles
            ],
            list(
                summary.BUDGET_ROLES
            ),
        )

        self.assertEqual(
            summary.primary_budget_index(
                self.spectral_norm
            ),
            1,
        )

    def test_complete_record_accounting(
        self,
    ):
        by_mode = summary.validate_results(
            self.results,
            spectral_norm=
                self.spectral_norm,
        )

        self.assertEqual(
            len(
                by_mode[
                    "matched_stochastic"
                ]
            ),
            32,
        )

        self.assertEqual(
            len(
                by_mode[
                    "deterministic"
                ]
            ),
            32,
        )

        self.assertEqual(
            by_mode[
                "matched_stochastic"
            ][
                0
            ][
                "attempt_id"
            ],
            "s2_q1_m16384_r000",
        )

        self.assertEqual(
            by_mode[
                "deterministic"
            ][
                -1
            ][
                "attempt_id"
            ],
            "s2_q2_m16384_r031",
        )

    def test_missing_or_duplicate_replicates_are_rejected(
        self,
    ):
        missing = self.results[
            :-1
        ]

        with self.assertRaises(
            ValueError
        ):
            summary.validate_results(
                missing,
                spectral_norm=
                    self.spectral_norm,
            )

        duplicate = list(
            self.results
        )

        duplicate[
            -1
        ] = duplicate[
            -2
        ]

        with self.assertRaises(
            ValueError
        ):
            summary.validate_results(
                duplicate,
                spectral_norm=
                    self.spectral_norm,
            )

    def test_primary_rows_retain_all_64_replicates(
        self,
    ):
        by_mode = summary.validate_results(
            self.results,
            spectral_norm=
                self.spectral_norm,
        )

        rows = summary.primary_replicate_rows(
            by_mode,
            spectral_norm=
                self.spectral_norm,
            delta=
                0.05,
        )

        self.assertEqual(
            len(
                rows
            ),
            64,
        )

        self.assertTrue(
            all(
                row[
                    "budget_role"
                ]
                == "fitted_operator_norm"
                for row in rows
            )
        )

        self.assertTrue(
            all(
                row[
                    "predictor_within_budget"
                ]
                for row in rows
            )
        )

        self.assertTrue(
            all(
                row[
                    "positive_certificate"
                ]
                for row in rows
            )
        )

        self.assertTrue(
            all(
                row[
                    "empirical_lower_bound_covered"
                ]
                for row in rows
            )
        )

    def test_budget_summary_does_not_treat_infeasible_budget_as_coverage_trials(
        self,
    ):
        (
            _mode_rows,
            budget_rows,
            _primary_rows,
        ) = summary.summarize_records(
            self.config,
            self.representation,
            self.results,
        )

        fixed_one = [
            row
            for row in budget_rows
            if row[
                "budget_role"
            ]
            == "fixed_1"
        ]

        self.assertEqual(
            len(
                fixed_one
            ),
            2,
        )

        for row in fixed_one:
            self.assertEqual(
                row[
                    "predictor_feasible_count"
                ],
                0,
            )

            self.assertEqual(
                row[
                    "empirical_lower_bound_coverage_among_feasible_trials"
                ],
                0,
            )

            self.assertIsNone(
                row[
                    "empirical_lower_bound_coverage_among_feasible_rate"
                ]
            )

            self.assertIsNone(
                row[
                    "feasible_rmse_minus_L_mean"
                ]
            )

    def test_primary_mode_summary_has_two_rows_and_exact_trial_accounting(
        self,
    ):
        (
            mode_rows,
            _budget_rows,
            _primary_rows,
        ) = summary.summarize_records(
            self.config,
            self.representation,
            self.results,
        )

        self.assertEqual(
            len(
                mode_rows
            ),
            2,
        )

        for row in mode_rows:
            self.assertEqual(
                row[
                    "replicates"
                ],
                32,
            )

            self.assertEqual(
                row[
                    "positive_certificate_count"
                ],
                32,
            )

            self.assertEqual(
                row[
                    "positive_certificate_trials"
                ],
                32,
            )

            self.assertEqual(
                row[
                    "positive_certificate_rate"
                ],
                1.0,
            )

            self.assertEqual(
                row[
                    "empirical_lower_bound_coverage_count"
                ],
                32,
            )

            self.assertEqual(
                row[
                    "empirical_lower_bound_violation_count"
                ],
                0,
            )

    def test_order_invariance(
        self,
    ):
        forward = summary.summarize_records(
            self.config,
            self.representation,
            self.results,
        )

        reverse = summary.summarize_records(
            self.config,
            self.representation,
            list(
                reversed(
                    self.results
                )
            ),
        )

        self.assertEqual(
            forward,
            reverse,
        )

    def test_writers_are_exclusive(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(
                tmp
            )

            json_path = (
                base
                / "x.json"
            )

            csv_path = (
                base
                / "x.csv"
            )

            summary.write_json(
                json_path,
                {
                    "x":
                        1
                },
            )

            summary.write_csv(
                csv_path,
                [
                    {
                        "x":
                            1
                    }
                ],
            )

            with self.assertRaises(
                FileExistsError
            ):
                summary.write_json(
                    json_path,
                    {
                        "x":
                            2
                    },
                )

            with self.assertRaises(
                FileExistsError
            ):
                summary.write_csv(
                    csv_path,
                    [
                        {
                            "x":
                                2
                        }
                    ],
                )

    def test_complete_tiny_summary_package_is_reproducible_and_checksummed(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(
                tmp
            )

            evidence = (
                base
                / "evidence"
            )

            evidence.mkdir()

            (
                evidence
                / "campaign_spec.json"
            ).write_text(
                json.dumps(
                    self.config,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            (
                evidence
                / "representation_source.json"
            ).write_text(
                json.dumps(
                    self.representation,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            (
                evidence
                / "results.jsonl"
            ).write_text(
                "\n".join(
                    json.dumps(
                        row,
                        sort_keys=True,
                    )
                    for row in self.results
                )
                + "\n",
                encoding="utf-8",
            )

            report = {
                "status":
                    "passed",

                "verified_attempts":
                    64,

                "verified_retained_pairs":
                    1_048_576,

                "execution_source_commit":
                    summary.EXPECTED_EXECUTION_COMMIT,

                "config_sha256":
                    summary.EXPECTED_CONFIG_SHA256,

                "evidence_sha256sums_fingerprint":
                    summary.EXPECTED_EVIDENCE_FINGERPRINT,

                "random_pair_datasets_regenerated":
                    0,

                "representations_refit":
                    0,
            }

            checksums = {
                "evidence_sha256sums_fingerprint":
                    summary.EXPECTED_EVIDENCE_FINGERPRINT,

                "verified_file_count":
                    344,

                "files":
                    {},
            }

            verifier = FakeVerifier(
                evidence,
                report,
                checksums,
                self.results,
            )

            provenance = {
                "summary_source_commit":
                    "a"
                    * 40,

                "source_sha256":
                    {},

                "python_version":
                    "test",

                "package_versions":
                    {
                        "numpy":
                            summary.np.__version__,

                        "scipy":
                            summary.scipy.__version__,
                    },
            }

            outputs = [
                base
                / "summary-a",

                base
                / "summary-b",
            ]

            for output in outputs:
                with patch.object(
                    summary,
                    "ROOT",
                    base
                    / "source",
                ), patch.object(
                    summary,
                    "source_provenance",
                    return_value=
                        provenance,
                ), patch.object(
                    summary,
                    "load_verifier",
                    return_value=
                        verifier,
                ):
                    metadata = (
                        summary.summarize_evidence(
                            evidence,
                            output,
                        )
                    )

                self.assertEqual(
                    metadata[
                        "status"
                    ],
                    "completed",
                )

                self.assertEqual(
                    metadata[
                        "mode_summary_rows"
                    ],
                    2,
                )

                self.assertEqual(
                    metadata[
                        "budget_summary_rows"
                    ],
                    8,
                )

                self.assertEqual(
                    metadata[
                        "primary_replicate_rows"
                    ],
                    64,
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

                    inventory[
                        name
                    ] = digest

                self.assertEqual(
                    set(
                        inventory
                    ),
                    set(
                        summary.SUMMARY_FILES
                    ),
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

            for name in (
                "README.md",
                "mode_summary.csv",
                "budget_summary.csv",
                "primary_replicates.csv",
                "numerical_audit.json",
            ):
                self.assertEqual(
                    (
                        outputs[
                            0
                        ]
                        / name
                    ).read_bytes(),
                    (
                        outputs[
                            1
                        ]
                        / name
                    ).read_bytes(),
                )

            with (
                outputs[
                    0
                ]
                / "mode_summary.csv"
            ).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                mode_rows = list(
                    csv.DictReader(
                        handle
                    )
                )

            self.assertEqual(
                len(
                    mode_rows
                ),
                2,
            )

            readme = (
                outputs[
                    0
                ]
                / "README.md"
            ).read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "kappa = 1",
                readme,
            )

            normalized_readme = " ".join(
                readme.split()
            )

            self.assertIn(
                "Zero observed violations do not establish a zero violation probability",
                normalized_readme,
            )

    def test_output_guard_rejects_existing_directory(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(
                tmp
            )

            evidence = (
                base
                / "evidence"
            )

            evidence.mkdir()

            output = (
                base
                / "output"
            )

            output.mkdir()

            with self.assertRaisesRegex(
                ValueError,
                "already exists",
            ):
                summary.summarize_evidence(
                    evidence,
                    output,
                )


if __name__ == "__main__":
    unittest.main()
