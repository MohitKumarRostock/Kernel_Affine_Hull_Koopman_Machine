"""Integration checks for the frozen final four-state campaign runner.

The full 117,000-job schedule is constructed and inspected but never executed.
Tiny test-only configurations exercise sampling, persistence, failure handling,
and metadata using separate seed material. These unit-test draws are not final
campaign evidence. Production preflight and source checks remain mandatory.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import experiment_20_certificate_four_state as pilot
import experiment_21_certificate_four_state_final as final


class FourStateFinalCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.config = final.load_final()
        cls.jobs, cls.references = final.build_schedule(cls.config)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="final-certificate-runner-test-"
        )
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()

    def tiny_config(self):
        config = deepcopy(self.config)
        config["campaign_id"] = "unit_test_final_runner_only"
        config["rng"]["campaign_key"] = 2**32 - 4
        config["cases"] = [deepcopy(config["cases"][2])]
        config["sampling"]["sample_sizes"] = [128]
        config["sampling"]["replicates_per_cell"] = 3
        return config

    def tiny_jobs(self):
        config = self.tiny_config()
        jobs, references = final.build_schedule(config)
        return config, jobs, references

    def state(self):
        return dict(
            started=0,
            sampled=0,
            succeeded=0,
            failed=0,
            active_attempt_id=None,
        )

    def rows(self, output, name):
        path = output / name
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def run_tiny(self, output, snapshots=None):
        config = self.tiny_config()
        raw = (
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        snapshot = dict(
            execution_source_commit="f" * 40,
            git_status_porcelain="",
            source_file_sha256={},
        )

        with patch.object(final, "ROOT", self.source), \
             patch.object(final, "load_final", return_value=(raw, config)), \
             patch.object(final, "SUPPORTED_CONFIG_SHA256", digest), \
             patch.object(
                 final,
                 "source_snapshot",
                 **(
                     {"return_value": snapshot}
                     if snapshots is None
                     else {"side_effect": snapshots}
                 ),
             ), \
             patch.object(final, "run_preflight"), \
             redirect_stdout(io.StringIO()):
            code = final.run_campaign(output)

        return code, snapshot

    def test_identifiers_and_configuration_fingerprint(self):
        self.assertEqual(
            final.RUNNER_ID,
            "four_state_final_runner_v1",
        )
        self.assertEqual(
            final.CONFIG_PATH,
            "reproduction/certificates/configs/four_state_final_v1.json",
        )
        self.assertEqual(
            hashlib.sha256(self.raw).hexdigest(),
            "630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182",
        )
        self.assertEqual(
            final.SUPPORTED_CONFIG_SHA256,
            hashlib.sha256(self.raw).hexdigest(),
        )
        self.assertEqual(
            self.config["campaign_id"],
            "four_state_original_iid_final_v1",
        )
        self.assertEqual(self.config["campaign_role"], "final")

    def test_final_design_totals(self):
        self.assertEqual(len(self.config["cases"]), 13)
        self.assertEqual(
            len(self.config["sampling"]["sample_sizes"]),
            9,
        )
        self.assertEqual(
            self.config["sampling"]["replicates_per_cell"],
            1000,
        )
        self.assertEqual(
            len(self.config["cases"])
            * len(self.config["sampling"]["sample_sizes"]),
            117,
        )
        self.assertEqual(len(self.jobs), 117000)
        self.assertEqual(len(self.references), 13)

    def test_exact_rational_metadata_and_design_roles(self):
        expected_roles = {
            "identity_p_31_over_32": "exact_closure_control",
            "conflicting_successors_p_4_over_5":
                "structurally_inconclusive",
            "conflicting_successors_p_5_over_6":
                "positive_certificate_boundary",
            "conflicting_successors_p_17_over_20":
                "population_limit_0.05",
            "conflicting_successors_p_13_over_15":
                "population_limit_0.10",
            "conflicting_successors_p_9_over_10":
                "population_limit_0.20",
            "conflicting_successors_p_14_over_15":
                "population_limit_0.30",
            "conflicting_successors_p_19_over_20":
                "interior_between_0.30_and_0.40",
            "conflicting_successors_p_29_over_30":
                "population_limit_0.40",
            "conflicting_successors_p_31_over_32":
                "manuscript_reference",
            "conflicting_successors_p_59_over_60":
                "population_limit_0.45",
            "conflicting_successors_p_99_over_100":
                "above_0.45_threshold",
            "conflicting_successors_p_1":
                "hard_assignment_endpoint",
        }

        self.assertEqual(
            {case["case_id"]: case["design_role"]
             for case in self.config["cases"]},
            expected_roles,
        )

        for case in self.config["cases"]:
            exact = case["p_exact"]
            self.assertEqual(
                case["p"],
                exact["numerator"] / exact["denominator"],
            )

    def test_full_schedule_has_unique_attempts(self):
        identifiers = [job["attempt_id"] for job in self.jobs]

        self.assertEqual(len(identifiers), 117000)
        self.assertEqual(len(set(identifiers)), 117000)
        self.assertEqual(identifiers[0], "c100_m128_r0")
        self.assertEqual(
            identifiers[-1],
            "c112_m524288_r999",
        )
        self.assertTrue(
            all(job["attempt_number"] == 1 for job in self.jobs)
        )

    def test_full_schedule_seed_material_is_unique(self):
        order = self.config["rng"]["seed_material_order"]
        materials = {
            tuple(job["seed"][name] for name in order)
            for job in self.jobs
        }

        self.assertEqual(len(materials), 117000)

        for material in materials:
            self.assertTrue(
                all(type(value) is int for value in material)
            )
            self.assertEqual(
                material[0],
                self.config["rng"]["campaign_key"],
            )
            self.assertEqual(
                material[-1],
                self.config["rng"]["root_seed"],
            )

    def test_references_match_final_cases_and_population_formulas(self):
        for case in self.config["cases"]:
            with self.subTest(case=case["case_id"]):
                reference = self.references[case["case_id"]]

                self.assertEqual(reference.p, case["p"])
                self.assertEqual(reference.dynamics, case["dynamics"])
                self.assertEqual(reference.kappa, case["kappa"])

                if case["dynamics"] == "identity":
                    self.assertEqual(
                        reference.exact_optimal_rmse,
                        0.0,
                    )
                    self.assertEqual(
                        reference.population_certificate_limit,
                        0.0,
                    )
                else:
                    self.assertAlmostEqual(
                        reference.exact_optimal_rmse,
                        case["p"] - 0.5,
                        places=14,
                    )
                    self.assertAlmostEqual(
                        reference.population_certificate_limit,
                        max(0.0, 3.0 * case["p"] - 2.5),
                        places=14,
                    )

    def test_load_final_rejects_changed_configuration_bytes(self):
        path = self.source / final.CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.raw + b"\n")

        with patch.object(final, "ROOT", self.source):
            with self.assertRaisesRegex(
                ValueError,
                "differs from the supported final-v1 bytes",
            ):
                final.load_final()

    def test_tiny_execute_jobs_success(self):
        config, jobs, references = self.tiny_jobs()
        output = self.base / "tiny-jobs"
        output.mkdir()
        state = self.state()

        final.execute_jobs(
            jobs,
            references,
            config,
            output,
            state,
        )

        self.assertEqual(
            state,
            dict(
                started=3,
                sampled=3,
                succeeded=3,
                failed=0,
                active_attempt_id=None,
            ),
        )
        self.assertEqual(
            [row["event"] for row in self.rows(output, "attempts.jsonl")],
            ["started", "completed"] * 3,
        )
        self.assertEqual(len(self.rows(output, "samples.jsonl")), 3)
        self.assertEqual(len(self.rows(output, "results.jsonl")), 3)

    def test_tiny_evaluation_failure_preserves_sample(self):
        config, jobs, references = self.tiny_jobs()
        output = self.base / "tiny-failure"
        output.mkdir()
        state = self.state()
        original = final.evaluate_four_state_counts
        calls = 0

        def evaluator(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ArithmeticError(
                    "injected final-runner evaluation failure"
                )
            return original(**kwargs)

        with patch.object(
            final,
            "evaluate_four_state_counts",
            side_effect=evaluator,
        ):
            final.execute_jobs(
                jobs,
                references,
                config,
                output,
                state,
            )

        self.assertEqual(calls, 3)
        self.assertEqual(state["sampled"], 3)
        self.assertEqual(state["succeeded"], 2)
        self.assertEqual(state["failed"], 1)
        self.assertEqual(len(self.rows(output, "samples.jsonl")), 3)
        self.assertEqual(len(self.rows(output, "results.jsonl")), 2)

        failure = [
            row for row in self.rows(output, "attempts.jsonl")
            if row["event"] == "failed"
        ]
        self.assertEqual(len(failure), 1)
        self.assertEqual(failure[0]["stage"], "evaluation")

    def test_tiny_run_metadata_marks_campaign_as_final(self):
        output = self.base / "tiny-final-run"
        code, snapshot = self.run_tiny(output)

        self.assertEqual(code, 0)

        metadata = json.loads(
            (output / "campaign_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        campaign_spec = json.loads(
            (output / "campaign_spec.json").read_text(
                encoding="utf-8"
            )
        )
        summary = json.loads(
            (output / "campaign_summary.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(metadata["runner_id"], final.RUNNER_ID)
        self.assertEqual(metadata["campaign_role"], "final")
        self.assertEqual(campaign_spec["campaign_role"], "final")
        self.assertEqual(metadata["source"], snapshot)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["succeeded"], 3)
        self.assertEqual(summary["failed"], 0)

    def test_tiny_run_source_change_invalidates_completion(self):
        config = self.tiny_config()
        raw = (
            json.dumps(
                config,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        first = dict(
            execution_source_commit="f" * 40,
            git_status_porcelain="",
            source_file_sha256={},
        )
        changed = dict(
            first,
            execution_source_commit="e" * 40,
        )

        output = self.base / "changed-source"

        with patch.object(final, "ROOT", self.source), \
             patch.object(
                 final,
                 "load_final",
                 return_value=(raw, config),
             ), \
             patch.object(
                 final,
                 "SUPPORTED_CONFIG_SHA256",
                 hashlib.sha256(raw).hexdigest(),
             ), \
             patch.object(
                 final,
                 "source_snapshot",
                 side_effect=[first, first, changed],
             ), \
             patch.object(final, "run_preflight"), \
             redirect_stdout(io.StringIO()):

            code = final.run_campaign(output)

        saved = json.loads(
            (output / "campaign_summary.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(code, 2)
        self.assertEqual(saved["status"], "invalid_source")
        self.assertFalse(saved["source_unchanged"])
        self.assertEqual(saved["started"], 3)

    def test_output_guards_precede_execution(self):
        existing = self.base / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with patch.object(final, "load_final") as load:
            with self.assertRaises(FileExistsError):
                final.run_campaign(existing)
            load.assert_not_called()

        with patch.object(final, "ROOT", self.source), \
             patch.object(final, "load_final") as load:
            with self.assertRaises(ValueError):
                final.run_campaign(self.source / "nested")
            load.assert_not_called()

        self.assertEqual(
            marker.read_text(encoding="utf-8"),
            "keep",
        )

    def test_source_snapshot_requires_final_runner_test(self):
        text = Path(final.__file__).read_text(encoding="utf-8")

        self.assertIn(
            '"tests/test_kahkm_four_state_final_campaign.py"',
            text,
        )
        self.assertNotIn(
            '"tests/test_kahkm_four_state_campaign.py"',
            text,
        )

    def test_import_does_not_execute_campaign(self):
        with patch.object(
            subprocess,
            "run",
            side_effect=AssertionError(
                "campaign/preflight executed during import"
            ),
        ):
            loaded = runpy.run_path(
                str(Path(final.__file__).resolve()),
                run_name="_final_runner_import_test_",
            )

        self.assertEqual(
            loaded["RUNNER_ID"],
            "four_state_final_runner_v1",
        )

    def test_pilot_and_final_runners_remain_separate(self):
        self.assertEqual(
            pilot.RUNNER_ID,
            "four_state_pilot_runner_v1",
        )
        self.assertEqual(
            pilot.CONFIG_PATH,
            "reproduction/certificates/configs/four_state_pilot_v1.json",
        )
        self.assertNotEqual(pilot.RUNNER_ID, final.RUNNER_ID)
        self.assertNotEqual(pilot.CONFIG_PATH, final.CONFIG_PATH)
        self.assertNotEqual(
            pilot.SUPPORTED_CONFIG_SHA256,
            final.SUPPORTED_CONFIG_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
