"""Tests for the independent final numerical verifier.

Temporary synthetic evidence contains one hand-constructed balanced count
vector for each of the 13 frozen final cases. It is not random experimental
evidence. Production modules create the saved synthetic answers; the verifier
then recomputes them independently with Fraction and Decimal arithmetic.

The real 117,000-replicate final evidence is never opened by these tests.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from kahkm_certificates import certificate_from_statistics
from kahkm_four_state_evaluation import evaluate_four_state_counts
from kahkm_four_state_reference import four_state_reference


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_four_state_final_numerics.py"
CONFIG = ROOT / "reproduction" / "certificates" / "configs" / "four_state_final_v1.json"

spec = importlib.util.spec_from_file_location(
    "_final_numerics_verifier_under_test",
    SCRIPT,
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def encoded(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_inventory(folder):
    lines = []

    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            lines.append(
                f"{digest}  "
                f"{path.relative_to(folder).as_posix()}\n"
            )

    (folder / "SHA256SUMS").write_text(
        "".join(lines),
        encoding="utf-8",
    )


class FourStateFinalNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(
            prefix="final-numerics-verifier-test-"
        )
        cls.addClassCleanup(temporary.cleanup)

        cls.base = Path(temporary.name).resolve()
        cls.evidence = cls.base / "synthetic-final-evidence"
        cls.evidence.mkdir()

        cls.final_bytes = CONFIG.read_bytes()
        cls.final_config = json.loads(cls.final_bytes)

        # Keep all 13 frozen cases, but reduce the synthetic schedule to
        # one sample size and one hand-constructed dataset per case.
        cls.config = deepcopy(cls.final_config)
        cls.config["sampling"]["sample_sizes"] = [128]
        cls.config["sampling"]["replicates_per_cell"] = 1

        raw = encoded(cls.config)
        cls.config_hash = hashlib.sha256(raw).hexdigest()

        (cls.evidence / "campaign_spec.json").write_bytes(raw)

        source = dict(
            execution_source_commit="a" * 40,
            git_status_porcelain="",
            source_file_sha256={},
        )

        metadata = dict(
            campaign_id=cls.config["campaign_id"],
            campaign_role="final",
            config_sha256=cls.config_hash,
            source=source,
        )

        (cls.evidence / "campaign_metadata.json").write_bytes(
            encoded(metadata)
        )

        (cls.evidence / "source_after.json").write_bytes(
            encoded(source)
        )

        references = {}
        schedules = []
        samples = []
        results = []
        attempts = []

        for case in cls.config["cases"]:
            reference = four_state_reference(
                p=case["p"],
                dynamics=case["dynamics"],
                kappa=case["kappa"],
            )

            references[case["case_id"]] = asdict(reference)

            n_pairs = 128
            replicate = 0

            identifier = (
                f"c{case['case_key']}_"
                f"m{n_pairs}_"
                f"r{replicate}"
            )

            seed = dict(
                campaign_key=cls.config["rng"]["campaign_key"],
                case_key=case["case_key"],
                n_pairs=n_pairs,
                replicate_index=replicate,
                root_seed=cls.config["rng"]["root_seed"],
            )

            schedules.append(
                dict(
                    attempt_id=identifier,
                    attempt_number=1,
                    case_id=case["case_id"],
                    seed=seed,
                )
            )

            # Hand-constructed balanced counts: not a random draw.
            counts = [32, 32, 32, 32]

            sample = dict(
                seed,
                sampler_id="four_state_multinomial_pcg64_v1",
                numpy_version="synthetic-test-fixture",
                seed_material=[
                    seed[name]
                    for name
                    in cls.config["rng"]["seed_material_order"]
                ],
                seed_material_order=cls.config["rng"]["seed_material_order"],
                bit_generator="PCG64",
                seed_sequence="SeedSequence",
                seed_sequence_pool_size=4,
                draws_per_replicate=1,
                state_order=["a", "b", "d", "e"],
                probabilities=[0.25] * 4,
                state_counts=counts,
            )

            samples.append(
                dict(
                    attempt_id=identifier,
                    sample=sample,
                )
            )

            options = cls.config["evaluation"]

            evaluated = asdict(
                evaluate_four_state_counts(
                    reference=reference,
                    state_counts=counts,
                    delta=options["delta"],
                    exclusion_tolerances=options[
                        "exclusion_tolerances"
                    ],
                    comparison_atol=options[
                        "comparison_atol"
                    ],
                )
            )

            evaluated.pop("reference")
            evaluated.update(
                attempt_id=identifier,
                case_id=case["case_id"],
            )

            results.append(evaluated)

            attempts.extend(
                (
                    dict(
                        attempt_id=identifier,
                        event="started",
                    ),
                    dict(
                        attempt_id=identifier,
                        event="completed",
                    ),
                )
            )

        (cls.evidence / "references.json").write_bytes(
            encoded(references)
        )

        for name, rows in (
            ("schedule.jsonl", schedules),
            ("samples.jsonl", samples),
            ("results.jsonl", results),
            ("attempts.jsonl", attempts),
        ):
            (cls.evidence / name).write_bytes(
                b"".join(
                    encoded(row)
                    for row in rows
                )
            )

        campaign_summary = dict(
            status="completed",
            exit_code=0,
            source_unchanged=True,
            active_attempt_id=None,
            planned=13,
            started=13,
            sampled=13,
            succeeded=13,
            failed=0,
            not_started=0,
            started_without_terminal_event=0,
        )

        (cls.evidence / "campaign_summary.json").write_bytes(
            encoded(campaign_summary)
        )

        write_inventory(cls.evidence)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            dir=self.base,
            prefix="case-",
        )
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name)

    def verify(self):
        with patch.object(
            verifier,
            "CONFIG_SHA256",
            self.config_hash,
        ), patch.object(
            verifier,
            "EXPECTED_REPLICATES",
            13,
        ), patch.object(
            verifier,
            "EXPECTED_SAMPLE_SIZES",
            1,
        ), patch.object(
            verifier,
            "EXPECTED_REPLICATES_PER_CELL",
            1,
        ):
            return verifier.verify_evidence(
                self.evidence
            )

    def changed_first_result(self, path, value):
        file = self.evidence / "results.jsonl"

        first, separator, rest = file.read_bytes().partition(
            b"\n"
        )

        self.assertEqual(separator, b"\n")

        record = json.loads(first)
        target = record

        for key in path[:-1]:
            target = target[key]

        target[path[-1]] = value

        return encoded(record) + rest

    def changed_first_sample(self, path, value):
        file = self.evidence / "samples.jsonl"

        first, separator, rest = file.read_bytes().partition(
            b"\n"
        )

        self.assertEqual(separator, b"\n")

        record = json.loads(first)
        target = record

        for key in path[:-1]:
            target = target[key]

        target[path[-1]] = value

        return encoded(record) + rest

    def temporarily_replace(self, name, data):
        path = self.evidence / name
        original = path.read_bytes()

        class Replacement:
            def __enter__(inner):
                path.write_bytes(data)
                write_inventory(self.evidence)
                return path

            def __exit__(
                inner,
                exc_type,
                exc,
                traceback,
            ):
                path.write_bytes(original)
                write_inventory(self.evidence)

        return Replacement()

    def test_frozen_configuration_fingerprint_and_constants(self):
        self.assertEqual(
            hashlib.sha256(
                self.final_bytes
            ).hexdigest(),
            verifier.CONFIG_SHA256,
        )

        self.assertEqual(
            self.final_config["campaign_id"],
            verifier.EXPECTED_CAMPAIGN_ID,
        )

        self.assertEqual(
            self.final_config["campaign_role"],
            verifier.EXPECTED_CAMPAIGN_ROLE,
        )

        self.assertEqual(
            len(self.final_config["cases"]),
            verifier.EXPECTED_CASES,
        )

        self.assertEqual(
            len(
                self.final_config["sampling"]["sample_sizes"]
            ),
            verifier.EXPECTED_SAMPLE_SIZES,
        )

        self.assertEqual(
            self.final_config["sampling"]["replicates_per_cell"],
            verifier.EXPECTED_REPLICATES_PER_CELL,
        )

    def test_strict_json_rejects_duplicates_and_nonfinite_values(self):
        invalid = (
            '{"a":1,"a":2}',
            '{"x":NaN}',
            '{"x":Infinity}',
            '{"x":-Infinity}',
        )

        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    verifier.read_json(text)

        self.assertEqual(
            verifier.read_json(
                '{"a":[1,true,null]}'
            ),
            {"a": [1, True, None]},
        )

    def test_exact_design_p_metadata_for_all_cases(self):
        for case in self.final_config["cases"]:
            with self.subTest(case=case["case_id"]):
                exact = verifier.exact_design_p(
                    case
                )

                metadata = case["p_exact"]

                self.assertEqual(
                    exact,
                    Fraction(
                        metadata["numerator"],
                        metadata["denominator"],
                    ),
                )

                self.assertEqual(
                    float(exact),
                    case["p"],
                )

    def test_exact_population_formulas_for_conflicting_cases(self):
        for case in self.final_config["cases"]:
            quantities = verifier.exact_population_quantities(
                case
            )

            p = verifier.exact_design_p(case)

            with self.subTest(case=case["case_id"]):
                if case["dynamics"] == "identity":
                    self.assertEqual(
                        quantities["exact_optimal_rmse"],
                        0,
                    )

                    self.assertEqual(
                        quantities[
                            "population_certificate_limit"
                        ],
                        0,
                    )

                else:
                    self.assertEqual(
                        quantities["exact_optimal_rmse"],
                        p - Fraction(1, 2),
                    )

                    self.assertEqual(
                        quantities[
                            "population_certificate_limit"
                        ],
                        max(
                            Fraction(0),
                            3 * p - Fraction(5, 2),
                        ),
                    )

    def test_exact_threshold_relations_include_boundaries(self):
        cases = {
            case["case_id"]: case
            for case in self.final_config["cases"]
        }

        boundary_checks = (
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
        )

        for case_id, quantity, tolerance in boundary_checks:
            with self.subTest(
                case=case_id,
                quantity=quantity,
            ):
                exact = verifier.exact_population_quantities(
                    cases[case_id]
                )[quantity]

                threshold = verifier.tolerance_fraction(
                    tolerance
                )

                self.assertEqual(
                    verifier.exact_relation(
                        exact,
                        threshold,
                    ),
                    "boundary",
                )

    def test_binary64_weighted_statistics_against_exact_manual_example(self):
        result = verifier.weighted_statistics_binary64(
            31 / 32,
            "conflicting_successors",
            (9, 1, 0, 2),
        )

        self.assertEqual(
            result["f_hat"],
            Fraction(1, 1024),
        )

        self.assertEqual(
            result["s_hat"],
            Fraction(135, 1024),
        )

        self.assertEqual(
            result["class_sizes"],
            [10, 2],
        )

        self.assertEqual(
            result["sses"],
            [
                Fraction(405, 256),
                Fraction(0),
            ],
        )

    def test_independent_decimal_certificate_matches_production_grid(self):
        cases = (
            (
                Fraction(1, 1024),
                Fraction(225, 1024),
                4096,
                math.sqrt(2),
                0.05,
            ),
            (
                0.0,
                0.5,
                8192,
                math.sqrt(2),
                0.05,
            ),
            (
                0.02,
                0.4,
                131072,
                0.75,
                0.05,
            ),
        )

        for f_hat, s_hat, n, kappa, delta in cases:
            with self.subTest(n=n):
                independent = verifier.independent_certificate(
                    f_hat,
                    s_hat,
                    n,
                    kappa,
                    delta,
                )

                production = certificate_from_statistics(
                    f_hat=f_hat,
                    s_hat=s_hat,
                    n_pairs=n,
                    kappa=kappa,
                    delta=delta,
                )

                for name, expected in independent.items():
                    self.assertTrue(
                        math.isclose(
                            getattr(production, name),
                            expected,
                            rel_tol=verifier.RTOL,
                            abs_tol=verifier.ATOL,
                        )
                    )

    def test_comparison_rejects_bad_shapes_types_and_large_errors(self):
        comparison = verifier.Comparison()

        comparison.close(
            [1.0, 2.0 + 1e-13],
            [1, 2],
            "valid",
        )

        self.assertEqual(
            comparison.numbers_checked,
            2,
        )

        invalid = (
            (True, 1),
            ("1", 1),
            (math.nan, 0),
            (math.inf, 0),
            ([1], [1, 2]),
            (1.01, 1),
        )

        for actual, expected in invalid:
            with self.subTest(actual=actual):
                with self.assertRaises(ValueError):
                    comparison.close(
                        actual,
                        expected,
                        "invalid",
                    )

    def test_verify_case_design_accepts_frozen_design(self):
        cases = verifier.verify_case_design(
            self.final_config
        )

        self.assertEqual(
            set(cases),
            set(verifier.EXPECTED_CASE_DESIGN),
        )

        self.assertEqual(
            len(cases),
            13,
        )

    def test_verify_case_design_rejects_role_or_exact_p_changes(self):
        for change in ("role", "exact_p"):
            config = deepcopy(self.final_config)

            if change == "role":
                config["cases"][1]["design_role"] = "wrong"
            else:
                config["cases"][1]["p_exact"] = {
                    "numerator": 81,
                    "denominator": 100,
                }

            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    verifier.verify_case_design(
                        config
                    )

    def test_verify_reference_accepts_all_frozen_references(self):
        comparison = verifier.Comparison()

        for case in self.final_config["cases"]:
            reference = json.loads(
                json.dumps(
                    asdict(
                        four_state_reference(
                            p=case["p"],
                            dynamics=case["dynamics"],
                            kappa=case["kappa"],
                        )
                    ),
                    allow_nan=False,
                )
            )

            with self.subTest(case=case["case_id"]):
                verifier.verify_reference(
                    reference,
                    case,
                    comparison,
                )

        self.assertGreater(
            comparison.numbers_checked,
            0,
        )

    def test_verify_reference_rejects_nonattaining_or_infeasible_matrix(self):
        case = deepcopy(
            self.final_config["cases"][0]
        )

        reference = json.loads(
            json.dumps(
                asdict(
                    four_state_reference(
                        p=case["p"],
                        dynamics=case["dynamics"],
                        kappa=case["kappa"],
                    )
                ),
                allow_nan=False,
            )
        )

        bad = deepcopy(reference)
        bad["optimal_B"] = [
            [0.0, 0.0],
            [0.0, 0.0],
        ]

        with self.assertRaises(ValueError):
            verifier.verify_reference(
                bad,
                case,
                verifier.Comparison(),
            )

        bad = deepcopy(reference)
        bad["optimal_B"] = [
            [5.0, -4.0],
            [-4.0, 5.0],
        ]

        with self.assertRaises(ValueError):
            verifier.verify_reference(
                bad,
                case,
                verifier.Comparison(),
            )

    def test_reconstruct_full_schedule_has_117000_unique_jobs(self):
        planned = verifier.reconstruct_schedule(
            self.final_config
        )

        self.assertEqual(
            len(planned),
            117000,
        )

        self.assertEqual(
            len(set(planned)),
            117000,
        )

        self.assertIn(
            "c100_m128_r0",
            planned,
        )

        self.assertIn(
            "c112_m524288_r999",
            planned,
        )

    def test_inventory_detects_changed_missing_and_unlisted_files(self):
        file = self.scratch / "a.json"
        file.write_bytes(b"{}\n")

        write_inventory(self.scratch)

        fingerprint, count = verifier.check_inventory(
            self.scratch
        )

        self.assertEqual(count, 1)
        self.assertEqual(
            fingerprint,
            hashlib.sha256(
                (self.scratch / "SHA256SUMS").read_bytes()
            ).hexdigest(),
        )

        file.write_bytes(b"changed")

        with self.assertRaises(ValueError):
            verifier.check_inventory(
                self.scratch
            )

    def test_inventory_rejects_unsafe_duplicate_and_symlink_entries(self):
        file = self.scratch / "a"
        file.write_bytes(b"x")

        checksum = hashlib.sha256(
            b"x"
        ).hexdigest()

        invalid = (
            f"{checksum}  ../a\n",
            f"{checksum}  /a\n",
            f"{checksum}  SHA256SUMS\n",
            f"{checksum}  a\n{checksum}  a\n",
        )

        for text in invalid:
            with self.subTest(text=text):
                (self.scratch / "SHA256SUMS").write_text(
                    text,
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError):
                    verifier.check_inventory(
                        self.scratch
                    )

        write_inventory(self.scratch)

        file.unlink()
        target = self.scratch / "target"
        target.write_bytes(b"x")
        file.symlink_to(target)

        with self.assertRaises(ValueError):
            verifier.check_inventory(
                self.scratch
            )

    def test_synthetic_end_to_end_evidence_passes(self):
        report = self.verify()

        self.assertEqual(
            report["status"],
            "passed",
        )

        self.assertEqual(
            report["verified_replicates"],
            13,
        )

        self.assertEqual(
            report["verified_references"],
            13,
        )

        self.assertEqual(
            report["verified_cases"],
            13,
        )

        self.assertEqual(
            report["verified_sample_sizes"],
            1,
        )

        self.assertEqual(
            report["random_samples_regenerated"],
            0,
        )

        self.assertIs(
            report["evidence_modified"],
            False,
        )

        self.assertGreater(
            report["numerical_values_checked"],
            0,
        )

        self.assertGreater(
            report[
                "exact_optimum_threshold_relations"
            ].get("boundary", 0),
            0,
        )

        self.assertGreater(
            report[
                "exact_population_limit_threshold_relations"
            ].get("boundary", 0),
            0,
        )

    def test_rehashed_numerical_corruption_is_rejected(self):
        changed = self.changed_first_result(
            [
                "certificate",
                "L_kappa_delta",
            ],
            0.123456,
        )

        with self.temporarily_replace(
            "results.jsonl",
            changed,
        ):
            verifier.check_inventory(
                self.evidence
            )

            with self.assertRaises(ValueError):
                self.verify()

    def test_rehashed_flags_gaps_and_sample_corruption_are_rejected(self):
        cases = (
            (
                "results.jsonl",
                self.changed_first_result(
                    ["strict_coverage"],
                    False,
                ),
            ),
            (
                "results.jsonl",
                self.changed_first_result(
                    ["gap_to_optimum"],
                    999.0,
                ),
            ),
            (
                "samples.jsonl",
                self.changed_first_sample(
                    ["sample", "state_counts"],
                    [33, 32, 32, 32],
                ),
            ),
            (
                "samples.jsonl",
                self.changed_first_sample(
                    ["sample", "case_key"],
                    999,
                ),
            ),
        )

        for name, data in cases:
            with self.subTest(name=name):
                with self.temporarily_replace(
                    name,
                    data,
                ):
                    with self.assertRaises(ValueError):
                        self.verify()

    def test_verifier_provenance_checks_clean_committed_verifier_and_test(self):
        commit = "b" * 40
        verifier_bytes = SCRIPT.read_bytes()
        test_bytes = Path(__file__).read_bytes()

        responses = [
            b"",
            commit.encode() + b"\n",
            verifier_bytes,
            test_bytes,
        ]

        with patch.object(
            verifier.subprocess,
            "check_output",
            side_effect=responses,
        ):
            result = verifier.verifier_provenance()

        self.assertEqual(
            result["verifier_source_commit"],
            commit,
        )

        self.assertEqual(
            result["verifier_sha256"],
            hashlib.sha256(
                verifier_bytes
            ).hexdigest(),
        )

        self.assertEqual(
            result["verifier_test_sha256"],
            hashlib.sha256(
                test_bytes
            ).hexdigest(),
        )

        with patch.object(
            verifier.subprocess,
            "check_output",
            return_value=b"?? uncommitted.py\n",
        ):
            with self.assertRaises(ValueError):
                verifier.verifier_provenance()

    def test_cli_success_failure_and_import_are_side_effect_free(self):
        stdout = io.StringIO()

        with patch.object(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--evidence",
                str(self.evidence),
            ],
        ), patch.object(
            verifier,
            "verifier_provenance",
            return_value={
                "verifier_source_commit":
                    "b" * 40
            },
        ), patch.object(
            verifier,
            "verify_evidence",
            return_value={
                "status": "passed"
            },
        ), redirect_stdout(stdout):

            self.assertEqual(
                verifier.main(),
                0,
            )

        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "status": "passed",
                "verifier_source_commit":
                    "b" * 40,
            },
        )

        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--evidence",
                str(self.evidence),
            ],
        ), patch.object(
            verifier,
            "verifier_provenance",
            side_effect=ValueError(
                "dirty verifier"
            ),
        ), redirect_stdout(stdout), \
             redirect_stderr(stderr):

            self.assertEqual(
                verifier.main(),
                2,
            )

        self.assertEqual(
            stdout.getvalue(),
            "",
        )

        self.assertIn(
            "dirty verifier",
            stderr.getvalue(),
        )

        with patch.object(
            Path,
            "read_text",
            side_effect=AssertionError(
                "unexpected file read"
            ),
        ), patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError(
                "unexpected file read"
            ),
        ), patch.object(
            verifier.subprocess,
            "check_output",
            side_effect=AssertionError(
                "unexpected Git access"
            ),
        ):
            loaded = runpy.run_path(
                str(SCRIPT),
                run_name="_final_verifier_import_test_",
            )

        self.assertEqual(
            loaded["VERIFIER_ID"],
            verifier.VERIFIER_ID,
        )


if __name__ == "__main__":
    unittest.main()
