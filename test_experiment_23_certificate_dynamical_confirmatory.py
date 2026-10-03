from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import experiment_23_certificate_dynamical_confirmatory as runner


ROOT = Path(
    runner.__file__
).resolve().parent


class ConfirmatoryCampaignRunnerTests(
    unittest.TestCase
):
    def test_frozen_config_and_schedule_are_exact(
        self,
    ):
        raw, config = (
            runner.load_confirmatory()
        )

        self.assertEqual(
            runner.file_sha256(
                ROOT
                / runner.CONFIG_PATH
            ),
            runner.SUPPORTED_CONFIG_SHA256,
        )

        self.assertEqual(
            runner.file_sha256(
                ROOT
                / runner.PROTOCOL_PATH
            ),
            runner.SUPPORTED_PROTOCOL_SHA256,
        )

        self.assertEqual(
            json.loads(
                raw
            )[
                "campaign_role"
            ],
            "confirmatory",
        )

        jobs = (
            runner.build_schedule(
                config
            )
        )

        self.assertEqual(
            len(
                jobs
            ),
            64,
        )

        self.assertEqual(
            sum(
                job[
                    "sample_size"
                ]
                for job in jobs
            ),
            1_048_576,
        )

        self.assertEqual(
            jobs[
                0
            ][
                "attempt_id"
            ],
            "s2_q1_m16384_r000",
        )

        self.assertEqual(
            jobs[
                31
            ][
                "attempt_id"
            ],
            "s2_q1_m16384_r031",
        )

        self.assertEqual(
            jobs[
                32
            ][
                "attempt_id"
            ],
            "s2_q2_m16384_r000",
        )

        self.assertEqual(
            jobs[
                -1
            ][
                "attempt_id"
            ],
            "s2_q2_m16384_r031",
        )

        self.assertEqual(
            {
                job[
                    "campaign_key"
                ]
                for job in jobs
            },
            {
                4
            },
        )

        self.assertEqual(
            {
                job[
                    "root_seed"
                ]
                for job in jobs
            },
            {
                20261003
            },
        )


    def test_norm_budgets_are_exact_binary64_deduplicated_and_sorted(
        self,
    ):
        _, config = (
            runner.load_confirmatory()
        )

        spectral_norm = (
            1.0203452494452896
        )

        actual = (
            runner._norm_budgets(
                config=
                    config,

                spectral_norm=
                    spectral_norm,
            )
        )

        self.assertEqual(
            actual,
            (
                1.0,
                spectral_norm,
                1.4142135623730951,
                2.0
                * spectral_norm,
            ),
        )


    def test_restore_real_repository_confirmatory_model_source(
        self,
    ):
        _, config = (
            runner.load_confirmatory()
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(
                tmp
            )

            bundle, metadata = (
                runner.restore_model_source(
                    output=
                        output,

                    config=
                        config,
                )
            )

            self.assertEqual(
                bundle.system,
                "vanderpol",
            )

            self.assertEqual(
                bundle.omega,
                128.0,
            )

            self.assertEqual(
                bundle.tau,
                1e-6,
            )

            self.assertEqual(
                bundle.B.shape,
                (
                    25,
                    25,
                ),
            )

            self.assertAlmostEqual(
                bundle.spectral_norm,
                1.0203452494452896,
                places=15,
            )

            self.assertEqual(
                metadata[
                    "state_space_kmeans_refits"
                ],
                0,
            )

            self.assertEqual(
                metadata[
                    "autoencoder_refits"
                ],
                0,
            )

            self.assertEqual(
                metadata[
                    "nlms_operator_refits"
                ],
                0,
            )

            self.assertEqual(
                metadata[
                    "certificate_evaluation_pairs_generated_during_restore"
                ],
                0,
            )


    def test_execute_jobs_lifecycle_without_real_sampling(
        self,
    ):
        _, config = (
            runner.load_confirmatory()
        )

        jobs = (
            runner.build_schedule(
                config
            )[
                :2
            ]
        )

        fake_dataset = object()
        fake_evaluation = object()

        model = runner.ModelBundle(
            system=
                "vanderpol",

            artifact_dir=
                Path(
                    "/tmp/fake"
                ),

            abstraction_model={
                "fake":
                    True
            },

            B=
                np.eye(
                    25,
                    dtype=np.float64,
                ),

            omega=
                128.0,

            tau=
                1e-6,

            spectral_norm=
                1.0,

            kappas=(
                1.0,
                1.4142135623730951,
                2.0,
            ),

            representation_manifest_sha256=
                "a"
                * 64,
        )

        state = {
            "started":
                0,

            "sampled":
                0,

            "evaluated":
                0,

            "succeeded":
                0,

            "failed":
                0,

            "sampled_pairs":
                0,

            "evaluated_pairs":
                0,

            "planned":
                2,

            "active_attempt_id":
                None,
        }

        source = {
            "commit":
                "a"
                * 40
        }

        sample_meta = {
            "sample_evidence_id":
                "dynamical_sample_evidence_v1",
        }

        evaluation_meta = {
            "system":
                "vanderpol",

            "evaluation_mode":
                "matched_stochastic",

            "sample_size":
                16384,

            "replicate_index":
                0,

            "frozen_predictor_spectral_norm":
                1.0203452494452896,

            "frozen_predictor_mse":
                0.01,

            "frozen_predictor_rmse":
                0.1,

            "statistics":
                {
                    "f_hat":
                        0.1,

                    "s_hat":
                        0.2,
                },

            "budget_certificates":
                [],

            "array_sha256":
                "b"
                * 64,
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(
                tmp
            )

            with (
                mock.patch.object(
                    runner,
                    "assert_source_unchanged",
                ),
                mock.patch.object(
                    runner,
                    "sample_confirmatory_dataset",
                    return_value=
                        fake_dataset,
                ) as sampler,
                mock.patch.object(
                    runner,
                    "write_confirmatory_sample_evidence",
                    return_value=
                        sample_meta,
                ) as sample_writer,
                mock.patch.object(
                    runner,
                    "evaluate_confirmatory_dataset",
                    return_value=
                        fake_evaluation,
                ) as evaluator,
                mock.patch.object(
                    runner,
                    "write_confirmatory_evaluation_evidence",
                    return_value=
                        evaluation_meta,
                ) as evaluation_writer,
            ):
                runner.execute_jobs(
                    jobs=
                        jobs,

                    model=
                        model,

                    config=
                        config,

                    output=
                        output,

                    state=
                        state,

                    source_before=
                        source,
                )

            self.assertEqual(
                sampler.call_count,
                2,
            )

            self.assertEqual(
                sample_writer.call_count,
                2,
            )

            self.assertEqual(
                evaluator.call_count,
                2,
            )

            self.assertEqual(
                evaluation_writer.call_count,
                2,
            )

            self.assertEqual(
                state[
                    "started"
                ],
                2,
            )

            self.assertEqual(
                state[
                    "sampled"
                ],
                2,
            )

            self.assertEqual(
                state[
                    "evaluated"
                ],
                2,
            )

            self.assertEqual(
                state[
                    "succeeded"
                ],
                2,
            )

            self.assertEqual(
                state[
                    "failed"
                ],
                0,
            )

            self.assertEqual(
                state[
                    "sampled_pairs"
                ],
                32768,
            )

            self.assertEqual(
                state[
                    "evaluated_pairs"
                ],
                32768,
            )

            self.assertTrue(
                (
                    output
                    / "attempts.jsonl"
                ).is_file()
            )

            self.assertTrue(
                (
                    output
                    / "samples.jsonl"
                ).is_file()
            )

            self.assertTrue(
                (
                    output
                    / "results.jsonl"
                ).is_file()
            )


    def test_sampling_failure_has_no_retry(
        self,
    ):
        _, config = (
            runner.load_confirmatory()
        )

        jobs = (
            runner.build_schedule(
                config
            )[
                :1
            ]
        )

        model = runner.ModelBundle(
            system=
                "vanderpol",

            artifact_dir=
                Path(
                    "/tmp/fake"
                ),

            abstraction_model={},

            B=
                np.eye(
                    25,
                    dtype=np.float64,
                ),

            omega=
                128.0,

            tau=
                1e-6,

            spectral_norm=
                1.0,

            kappas=(
                1.0,
            ),

            representation_manifest_sha256=
                "a"
                * 64,
        )

        state = {
            "started":
                0,

            "sampled":
                0,

            "evaluated":
                0,

            "succeeded":
                0,

            "failed":
                0,

            "sampled_pairs":
                0,

            "evaluated_pairs":
                0,

            "planned":
                1,

            "active_attempt_id":
                None,
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(
                tmp
            )

            with (
                mock.patch.object(
                    runner,
                    "assert_source_unchanged",
                ),
                mock.patch.object(
                    runner,
                    "sample_confirmatory_dataset",
                    side_effect=
                        RuntimeError(
                            "synthetic sampling failure"
                        ),
                ) as sampler,
                mock.patch.object(
                    runner,
                    "write_confirmatory_sample_evidence",
                ) as writer,
            ):
                runner.execute_jobs(
                    jobs=
                        jobs,

                    model=
                        model,

                    config=
                        config,

                    output=
                        output,

                    state=
                        state,

                    source_before={
                        "commit":
                            "a"
                            * 40
                    },
                )

            self.assertEqual(
                sampler.call_count,
                1,
            )

            writer.assert_not_called()

            self.assertEqual(
                state[
                    "failed"
                ],
                1,
            )

            self.assertEqual(
                state[
                    "succeeded"
                ],
                0,
            )

            events = [
                json.loads(
                    line
                )
                for line in (
                    output
                    / "attempts.jsonl"
                ).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertEqual(
                [
                    event[
                        "event"
                    ]
                    for event in events
                ],
                [
                    "started",
                    "failed",
                ],
            )

            self.assertEqual(
                events[
                    1
                ][
                    "stage"
                ],
                "sampling",
            )


    def test_run_campaign_contract_without_real_sampling(
        self,
    ):
        source = {
            "commit":
                "c"
                * 40,

            "git_status_porcelain":
                "",

            "required_source_sha256":
                {},
        }

        fake_bundle = runner.ModelBundle(
            system=
                "vanderpol",

            artifact_dir=
                Path(
                    "/tmp/fake"
                ),

            abstraction_model={},

            B=
                np.eye(
                    25,
                    dtype=np.float64,
                ),

            omega=
                128.0,

            tau=
                1e-6,

            spectral_norm=
                1.0203452494452896,

            kappas=(
                1.0,
                1.0203452494452896,
                1.4142135623730951,
                2.0406904988905792,
            ),

            representation_manifest_sha256=
                "d"
                * 64,
        )

        model_metadata = {
            "system":
                "vanderpol"
        }

        def fake_execute(
            *,
            jobs,
            model,
            config,
            output,
            state,
            source_before,
        ):
            self.assertEqual(
                len(
                    jobs
                ),
                64,
            )

            state[
                "started"
            ] = 64

            state[
                "sampled"
            ] = 64

            state[
                "evaluated"
            ] = 64

            state[
                "succeeded"
            ] = 64

            state[
                "sampled_pairs"
            ] = 1_048_576

            state[
                "evaluated_pairs"
            ] = 1_048_576

        with tempfile.TemporaryDirectory() as tmp:
            output = (
                Path(
                    tmp
                )
                / "run001"
            )

            with (
                mock.patch.object(
                    runner,
                    "source_snapshot",
                    return_value=
                        source,
                ),
                mock.patch.object(
                    runner,
                    "run_preflight",
                ),
                mock.patch.object(
                    runner,
                    "restore_model_source",
                    return_value=(
                        fake_bundle,
                        model_metadata,
                    ),
                ),
                mock.patch.object(
                    runner,
                    "execute_jobs",
                    side_effect=
                        fake_execute,
                ),
                mock.patch.object(
                    runner,
                    "sample_confirmatory_dataset",
                ) as sampler,
            ):
                code = (
                    runner.run_campaign(
                        output
                    )
                )

            self.assertEqual(
                code,
                0,
            )

            sampler.assert_not_called()

            summary = json.loads(
                (
                    output
                    / "campaign_summary.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                summary[
                    "status"
                ],
                "completed",
            )

            self.assertEqual(
                summary[
                    "planned"
                ],
                64,
            )

            self.assertEqual(
                summary[
                    "planned_retained_pairs"
                ],
                1_048_576,
            )

            self.assertEqual(
                summary[
                    "evaluated_pairs"
                ],
                1_048_576,
            )

            self.assertFalse(
                summary[
                    "pilot_evaluation_pairs_reused"
                ]
            )

            self.assertEqual(
                summary[
                    "representation_refits_during_campaign"
                ],
                0,
            )

            self.assertEqual(
                summary[
                    "independent_evidence_verification"
                ],
                "not_yet_performed",
            )

            self.assertTrue(
                (
                    output
                    / "SHA256SUMS"
                ).is_file()
            )


    def test_output_inside_repository_is_rejected_before_sampling(
        self,
    ):
        target = (
            ROOT
            / ".forbidden-confirmatory-output"
        )

        self.assertFalse(
            target.exists()
        )

        with mock.patch.object(
            runner,
            "sample_confirmatory_dataset",
        ) as sampler:
            with self.assertRaisesRegex(
                ValueError,
                "outside this working tree",
            ):
                runner.run_campaign(
                    target
                )

        sampler.assert_not_called()


    def test_source_change_after_preflight_aborts_before_sampling(
        self,
    ):
        first = {
            "commit":
                "1"
                * 40
        }

        changed = {
            "commit":
                "2"
                * 40
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = (
                Path(
                    tmp
                )
                / "run001"
            )

            with (
                mock.patch.object(
                    runner,
                    "source_snapshot",
                    side_effect=[
                        first,
                        changed,
                        changed,
                    ],
                ),
                mock.patch.object(
                    runner,
                    "run_preflight",
                ),
                mock.patch.object(
                    runner,
                    "restore_model_source",
                ) as restore,
                mock.patch.object(
                    runner,
                    "sample_confirmatory_dataset",
                ) as sampler,
            ):
                code = (
                    runner.run_campaign(
                        output
                    )
                )

            self.assertEqual(
                code,
                2,
            )

            restore.assert_not_called()

            sampler.assert_not_called()

            summary = json.loads(
                (
                    output
                    / "campaign_summary.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                summary[
                    "status"
                ],
                "aborted",
            )

            self.assertEqual(
                summary[
                    "aborted_stage"
                ],
                "preflight",
            )

            self.assertEqual(
                summary[
                    "started"
                ],
                0,
            )


if __name__ == "__main__":
    unittest.main()
