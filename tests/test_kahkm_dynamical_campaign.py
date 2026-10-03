"""Tests for the frozen dynamical certificate-pilot campaign runner.

The tests exercise campaign control flow, provenance locking, accounting,
persistence order, failure handling, and real restoration of the already
committed frozen representations.

No dynamical pilot pair is generated.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import experiment_22_certificate_dynamical_pilot as runner
from kahkm_dynamical_sampling import DYNAMICAL_SAMPLING_ID


class DynamicalCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def synthetic_config(self):
        return {
            "certificate": {
                "delta":
                    0.05,
            },
            "pair_sampling": {
                "horizon_steps":
                    1200,
            },
            "frozen_representation": {
                "batch_size":
                    32,
                "n_jobs":
                    1,
                "systems": {
                    "duffing": {
                        "dt":
                            0.03,
                        "vanderpol_mu":
                            1.0,
                    },
                    "vanderpol": {
                        "dt":
                            0.02,
                        "vanderpol_mu":
                            1.0,
                    },
                },
            },
        }

    def synthetic_job(self):
        return {
            "attempt_id":
                "s1_q2_m3_r000",
            "attempt_number":
                1,
            "system":
                "duffing",
            "system_key":
                1,
            "evaluation_mode":
                "deterministic",
            "mode_key":
                2,
            "sample_size":
                3,
            "replicate_index":
                0,
        }

    def synthetic_state(self):
        return {
            "status":
                "running",
            "planned":
                1,
            "planned_retained_pairs":
                3,
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
            "failed_sampling":
                0,
            "failed_evaluation":
                0,
            "active_attempt_id":
                None,
        }

    def synthetic_dataset(self):
        return SimpleNamespace(
            sampling_id=
                DYNAMICAL_SAMPLING_ID,
            current_states=
                np.zeros(
                    (
                        2,
                        3,
                    ),
                    dtype=np.float64,
                ),
            successor_states=
                np.ones(
                    (
                        2,
                        3,
                    ),
                    dtype=np.float64,
                ),
        )

    def synthetic_models(self):
        loaded = SimpleNamespace(
            abstraction_model=
                object(),
            B=
                np.eye(
                    2,
                    dtype=np.float64,
                ),
            omega=
                2.0,
            tau=
                1e-6,
        )

        return {
            "duffing":
                runner.ModelBundle(
                    system=
                        "duffing",
                    loaded=
                        loaded,
                    spectral_norm=
                        1.0,
                    kappas=(
                        1.0,
                        2.0,
                    ),
                )
        }

    def synthetic_evaluation_metadata(self):
        return {
            "system":
                "duffing",
            "evaluation_mode":
                "deterministic",
            "sample_size":
                3,
            "replicate_index":
                0,
            "statistics": {
                "f_hat":
                    0.1,
                "s_hat":
                    0.2,
            },
            "frozen_predictor_spectral_norm":
                1.0,
            "frozen_predictor_mse":
                0.09,
            "frozen_predictor_rmse":
                0.3,
            "budget_certificates": [
                {
                    "kappa":
                        1.0,
                    "certificate": {
                        "L_kappa_delta":
                            0.0,
                    },
                }
            ],
        }

    def run_execute_jobs(
        self,
        *,
        state=None,
        **patches,
    ):
        output = (
            self.base
            / "execute"
        )

        output.mkdir()

        if state is None:
            state = (
                self.synthetic_state()
            )

        patchers = {
            "assert_source_unchanged":
                mock.patch.object(
                    runner,
                    "assert_source_unchanged",
                ),
            "sample_dynamical_dataset":
                mock.patch.object(
                    runner,
                    "sample_dynamical_dataset",
                    return_value=
                        self.synthetic_dataset(),
                ),
            "write_sample_evidence":
                mock.patch.object(
                    runner,
                    "write_sample_evidence",
                    return_value={
                        "array_sha256":
                            "sample-sha",
                    },
                ),
            "evaluate_frozen_dynamical_pairs":
                mock.patch.object(
                    runner,
                    "evaluate_frozen_dynamical_pairs",
                    return_value=
                        object(),
                ),
            "write_evaluation_evidence":
                mock.patch.object(
                    runner,
                    "write_evaluation_evidence",
                    return_value=
                        self.synthetic_evaluation_metadata(),
                ),
        }

        for name, replacement in patches.items():
            patchers[
                name
            ] = mock.patch.object(
                runner,
                name,
                replacement,
            )

        entered = {}

        with (
            patchers[
                "assert_source_unchanged"
            ] as entered_guard,
            patchers[
                "sample_dynamical_dataset"
            ] as entered_sample,
            patchers[
                "write_sample_evidence"
            ] as entered_write_sample,
            patchers[
                "evaluate_frozen_dynamical_pairs"
            ] as entered_evaluate,
            patchers[
                "write_evaluation_evidence"
            ] as entered_write_evaluation,
        ):
            entered.update(
                {
                    "guard":
                        entered_guard,
                    "sample":
                        entered_sample,
                    "write_sample":
                        entered_write_sample,
                    "evaluate":
                        entered_evaluate,
                    "write_evaluation":
                        entered_write_evaluation,
                }
            )

            runner.execute_jobs(
                jobs=[
                    self.synthetic_job()
                ],
                models=
                    self.synthetic_models(),
                config=
                    self.synthetic_config(),
                output=
                    output,
                state=
                    state,
                source_before={
                    "execution_source_commit":
                        "synthetic",
                },
            )

        return (
            output,
            state,
            entered,
        )

    def campaign_patches(
        self,
        *,
        source_side_effect=None,
        execute_side_effect=None,
    ):
        before = {
            "execution_source_commit":
                "abc123",
            "git_status_porcelain":
                "",
            "source_file_sha256": {
                "runner":
                    "hash",
            },
        }

        if source_side_effect is None:
            source_side_effect = [
                before,
                before,
            ]

        config = {
            "campaign_id":
                "synthetic_dynamical_pilot",
        }

        job = (
            self.synthetic_job()
        )

        return (
            before,
            config,
            job,
            mock.patch.object(
                runner,
                "load_pilot",
                return_value=(
                    b'{"synthetic":true}\n',
                    config,
                ),
            ),
            mock.patch.object(
                runner,
                "source_snapshot",
                side_effect=
                    source_side_effect,
            ),
            mock.patch.object(
                runner,
                "build_schedule",
                return_value=[
                    job
                ],
            ),
            mock.patch.object(
                runner,
                "run_preflight",
            ),
            mock.patch.object(
                runner,
                "assert_source_unchanged",
            ),
            mock.patch.object(
                runner,
                "restore_models",
                return_value=(
                    object(),
                    {},
                ),
            ),
            mock.patch.object(
                runner,
                "execute_jobs",
                side_effect=
                    execute_side_effect,
            ),
        )

    # 1
    def test_committed_pilot_configuration_loads_with_exact_hash_and_role(self):
        raw, config = (
            runner.load_pilot()
        )

        self.assertEqual(
            hashlib.sha256(
                raw
            ).hexdigest(),
            runner.SUPPORTED_CONFIG_SHA256,
        )

        self.assertEqual(
            config[
                "campaign_role"
            ],
            "pilot",
        )

        self.assertEqual(
            config[
                "campaign_id"
            ],
            "dynamical_original_iid_pilot_v1",
        )

    # 2
    def test_complete_schedule_has_96_attempts_and_86016_pairs(self):
        _, config = (
            runner.load_pilot()
        )

        jobs = runner.build_schedule(
            config
        )

        self.assertEqual(
            len(jobs),
            96,
        )

        self.assertEqual(
            len(
                {
                    job[
                        "attempt_id"
                    ]
                    for job
                    in jobs
                }
            ),
            96,
        )

        self.assertEqual(
            sum(
                job[
                    "sample_size"
                ]
                for job
                in jobs
            ),
            86016,
        )

        cells = {
            (
                job[
                    "system"
                ],
                job[
                    "evaluation_mode"
                ],
                job[
                    "sample_size"
                ],
            )
            for job
            in jobs
        }

        self.assertEqual(
            len(cells),
            12,
        )

        for cell in cells:
            self.assertEqual(
                sum(
                    1
                    for job
                    in jobs
                    if (
                        job[
                            "system"
                        ],
                        job[
                            "evaluation_mode"
                        ],
                        job[
                            "sample_size"
                        ],
                    )
                    == cell
                ),
                8,
            )

    # 3
    def test_build_schedule_performs_no_sampling_or_evaluation(self):
        _, config = (
            runner.load_pilot()
        )

        with (
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
            ) as sample_mock,
            mock.patch.object(
                runner,
                "evaluate_frozen_dynamical_pairs",
            ) as evaluate_mock,
        ):
            jobs = (
                runner.build_schedule(
                    config
                )
            )

        self.assertEqual(
            len(jobs),
            96,
        )

        sample_mock.assert_not_called()
        evaluate_mock.assert_not_called()

    # 4
    def test_norm_budgets_follow_exact_frozen_configuration_recipe(self):
        _, config = (
            runner.load_pilot()
        )

        spectral_norm = 1.25

        actual = runner._norm_budgets(
            config=config,
            spectral_norm=
                spectral_norm,
        )

        design = config[
            "norm_budgets"
        ]

        expected = [
            float(
                value
            )
            for value
            in design[
                "fixed_constants"
            ]
        ]

        for name in design[
            "training_derived"
        ]:
            if (
                name
                == "nlms_spectral_norm"
            ):
                expected.append(
                    spectral_norm
                )

            elif (
                name
                == "twice_nlms_spectral_norm"
            ):
                expected.append(
                    2.0
                    * spectral_norm
                )

            else:
                self.fail(
                    "Unexpected frozen training-derived budget "
                    f"{name!r}."
                )

        self.assertEqual(
            actual,
            tuple(
                sorted(
                    set(
                        expected
                    )
                )
            ),
        )

    # 5
    def test_write_json_is_exclusive_and_rejects_nonfinite_values(self):
        path = (
            self.base
            / "value.json"
        )

        runner.write_json(
            path,
            {
                "x":
                    1.0,
            },
        )

        with self.assertRaises(
            FileExistsError
        ):
            runner.write_json(
                path,
                {
                    "x":
                        2.0,
                },
            )

        bad = (
            self.base
            / "bad.json"
        )

        with self.assertRaises(
            ValueError
        ):
            runner.write_json(
                bad,
                {
                    "x":
                        float(
                            "nan"
                        ),
                },
            )

        self.assertFalse(
            bad.exists()
        )

    # 6
    def test_checksum_inventory_covers_nested_files_and_excludes_itself(self):
        output = (
            self.base
            / "checksums"
        )

        nested = (
            output
            / "nested"
        )

        nested.mkdir(
            parents=True
        )

        (
            output
            / "a.txt"
        ).write_bytes(
            b"a"
        )

        (
            nested
            / "b.txt"
        ).write_bytes(
            b"b"
        )

        runner.write_checksums(
            output
        )

        lines = (
            output
            / "SHA256SUMS"
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(
            len(lines),
            2,
        )

        self.assertTrue(
            any(
                line.endswith(
                    "  a.txt"
                )
                for line
                in lines
            )
        )

        self.assertTrue(
            any(
                line.endswith(
                    "  nested/b.txt"
                )
                for line
                in lines
            )
        )

        self.assertFalse(
            any(
                "SHA256SUMS"
                in line
                for line
                in lines
            )
        )

    # 7
    def test_source_snapshot_rejects_dirty_worktree_before_byte_checks(self):
        def fake_git_bytes(
            *args,
        ):
            if args == (
                "rev-parse",
                "--show-toplevel",
            ):
                return (
                    str(
                        runner.ROOT
                    )
                    + "\n"
                ).encode()

            if args == (
                "rev-parse",
                "HEAD",
            ):
                return b"abc123\n"

            if args == (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ):
                return (
                    b"?? synthetic.txt\n"
                )

            raise AssertionError(
                args
            )

        with (
            mock.patch.object(
                runner,
                "git_bytes",
                side_effect=
                    fake_git_bytes,
            ),
            mock.patch.object(
                runner,
                "file_sha256",
            ) as hash_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "clean source",
            ):
                runner.source_snapshot()

        hash_mock.assert_not_called()

    # 8
    def test_source_snapshot_rejects_required_bytes_different_from_head(self):
        actual_hash = hashlib.sha256(
            b"actual"
        ).hexdigest()

        def fake_git_bytes(
            *args,
        ):
            if args == (
                "rev-parse",
                "--show-toplevel",
            ):
                return (
                    str(
                        runner.ROOT
                    )
                    + "\n"
                ).encode()

            if args == (
                "rev-parse",
                "HEAD",
            ):
                return b"abc123\n"

            if args == (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ):
                return b""

            if (
                len(args)
                == 2
                and args[
                    0
                ]
                == "show"
            ):
                return b"committed"

            raise AssertionError(
                args
            )

        with (
            mock.patch.object(
                runner,
                "git_bytes",
                side_effect=
                    fake_git_bytes,
            ),
            mock.patch.object(
                runner,
                "file_sha256",
                return_value=
                    actual_hash,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "differ from HEAD",
            ):
                runner.source_snapshot()

    # 9
    def test_source_guard_rejects_head_change_immediately(self):
        initial = {
            "execution_source_commit":
                "before",
            "source_file_sha256":
                {},
        }

        def fake_git_bytes(
            *args,
        ):
            if args == (
                "rev-parse",
                "--show-toplevel",
            ):
                return (
                    str(
                        runner.ROOT
                    )
                    + "\n"
                ).encode()

            if args == (
                "rev-parse",
                "HEAD",
            ):
                return b"after\n"

            raise AssertionError(
                args
            )

        with mock.patch.object(
            runner,
            "git_bytes",
            side_effect=
                fake_git_bytes,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "HEAD changed",
            ):
                runner.assert_source_unchanged(
                    initial
                )

    # 10
    def test_real_committed_frozen_models_restore_without_sampling_or_fitting(self):
        _, config = (
            runner.load_pilot()
        )

        output = (
            self.base
            / "restore"
        )

        output.mkdir()

        with (
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
            ) as sample_mock,
            mock.patch.object(
                runner,
                "evaluate_frozen_dynamical_pairs",
            ) as evaluate_mock,
        ):
            restored, models = (
                runner.restore_models(
                    output=
                        output,
                    config=
                        config,
                )
            )

        sample_mock.assert_not_called()
        evaluate_mock.assert_not_called()

        self.assertEqual(
            restored.archive_sha256,
            runner.EXPECTED_ARCHIVE_SHA256,
        )

        self.assertEqual(
            set(
                models
            ),
            {
                "duffing",
                "vanderpol",
            },
        )

        for system, bundle in models.items():
            self.assertEqual(
                bundle.system,
                system,
            )

            self.assertGreater(
                bundle.spectral_norm,
                0.0,
            )

            self.assertEqual(
                bundle.kappas,
                tuple(
                    sorted(
                        set(
                            bundle.kappas
                        )
                    )
                ),
            )

        self.assertTrue(
            (
                output
                / "frozen_model_source.json"
            ).is_file()
        )

    # 11
    def test_successful_attempt_persists_sample_before_evaluation(self):
        order = []

        def sample(**kwargs):
            order.append(
                "sample"
            )
            return (
                self.synthetic_dataset()
            )

        def persist_sample(**kwargs):
            order.append(
                "persist_sample"
            )
            return {
                "array_sha256":
                    "sample-sha",
            }

        def evaluate(**kwargs):
            order.append(
                "evaluate"
            )
            return object()

        def persist_evaluation(
            **kwargs,
        ):
            order.append(
                "persist_evaluation"
            )
            return (
                self.synthetic_evaluation_metadata()
            )

        output = (
            self.base
            / "success"
        )

        output.mkdir()

        state = (
            self.synthetic_state()
        )

        with (
            mock.patch.object(
                runner,
                "assert_source_unchanged",
            ),
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
                side_effect=
                    sample,
            ),
            mock.patch.object(
                runner,
                "write_sample_evidence",
                side_effect=
                    persist_sample,
            ),
            mock.patch.object(
                runner,
                "evaluate_frozen_dynamical_pairs",
                side_effect=
                    evaluate,
            ),
            mock.patch.object(
                runner,
                "write_evaluation_evidence",
                side_effect=
                    persist_evaluation,
            ),
        ):
            runner.execute_jobs(
                jobs=[
                    self.synthetic_job()
                ],
                models=
                    self.synthetic_models(),
                config=
                    self.synthetic_config(),
                output=
                    output,
                state=
                    state,
                source_before={
                    "execution_source_commit":
                        "synthetic",
                },
            )

        self.assertEqual(
            order,
            [
                "sample",
                "persist_sample",
                "evaluate",
                "persist_evaluation",
            ],
        )

        self.assertEqual(
            state[
                "sampled"
            ],
            1,
        )

        self.assertEqual(
            state[
                "evaluated"
            ],
            1,
        )

        self.assertEqual(
            state[
                "succeeded"
            ],
            1,
        )

        self.assertEqual(
            state[
                "failed"
            ],
            0,
        )

    # 12
    def test_sampling_failure_is_logged_without_retry(self):
        output = (
            self.base
            / "sampling_failure"
        )

        output.mkdir()

        state = (
            self.synthetic_state()
        )

        with (
            mock.patch.object(
                runner,
                "assert_source_unchanged",
            ),
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
                side_effect=
                    RuntimeError(
                        "synthetic sampling failure"
                    ),
            ) as sample_mock,
            mock.patch.object(
                runner,
                "write_sample_evidence",
            ) as persist_mock,
            mock.patch.object(
                runner,
                "evaluate_frozen_dynamical_pairs",
            ) as evaluate_mock,
        ):
            runner.execute_jobs(
                jobs=[
                    self.synthetic_job()
                ],
                models=
                    self.synthetic_models(),
                config=
                    self.synthetic_config(),
                output=
                    output,
                state=
                    state,
                source_before={
                    "execution_source_commit":
                        "synthetic",
                },
            )

        self.assertEqual(
            sample_mock.call_count,
            1,
        )

        persist_mock.assert_not_called()
        evaluate_mock.assert_not_called()

        self.assertEqual(
            state[
                "failed"
            ],
            1,
        )

        self.assertEqual(
            state[
                "failed_sampling"
            ],
            1,
        )

        events = [
            json.loads(
                line
            )
            for line
            in (
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
                for event
                in events
            ],
            [
                "started",
                "failed",
            ],
        )

        self.assertEqual(
            events[
                -1
            ][
                "stage"
            ],
            "sampling",
        )

    # 13
    def test_evaluation_failure_preserves_sample_and_is_not_retried(self):
        output = (
            self.base
            / "evaluation_failure"
        )

        output.mkdir()

        state = (
            self.synthetic_state()
        )

        with (
            mock.patch.object(
                runner,
                "assert_source_unchanged",
            ),
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
                return_value=
                    self.synthetic_dataset(),
            ) as sample_mock,
            mock.patch.object(
                runner,
                "write_sample_evidence",
                return_value={
                    "array_sha256":
                        "sample-sha",
                },
            ) as persist_sample,
            mock.patch.object(
                runner,
                "evaluate_frozen_dynamical_pairs",
                side_effect=
                    RuntimeError(
                        "synthetic evaluation failure"
                    ),
            ) as evaluate_mock,
            mock.patch.object(
                runner,
                "write_evaluation_evidence",
            ) as persist_evaluation,
        ):
            runner.execute_jobs(
                jobs=[
                    self.synthetic_job()
                ],
                models=
                    self.synthetic_models(),
                config=
                    self.synthetic_config(),
                output=
                    output,
                state=
                    state,
                source_before={
                    "execution_source_commit":
                        "synthetic",
                },
            )

        self.assertEqual(
            sample_mock.call_count,
            1,
        )

        self.assertEqual(
            persist_sample.call_count,
            1,
        )

        self.assertEqual(
            evaluate_mock.call_count,
            1,
        )

        persist_evaluation.assert_not_called()

        self.assertEqual(
            state[
                "sampled"
            ],
            1,
        )

        self.assertEqual(
            state[
                "failed_evaluation"
            ],
            1,
        )

        self.assertEqual(
            state[
                "failed"
            ],
            1,
        )

    # 14
    def test_sample_persistence_failure_aborts_instead_of_becoming_attempt_failure(self):
        output = (
            self.base
            / "sample_persistence_failure"
        )

        output.mkdir()

        state = (
            self.synthetic_state()
        )

        with (
            mock.patch.object(
                runner,
                "assert_source_unchanged",
            ),
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
                return_value=
                    self.synthetic_dataset(),
            ),
            mock.patch.object(
                runner,
                "write_sample_evidence",
                side_effect=
                    OSError(
                        "synthetic storage failure"
                    ),
            ),
            mock.patch.object(
                runner,
                "evaluate_frozen_dynamical_pairs",
            ) as evaluate_mock,
        ):
            with self.assertRaisesRegex(
                OSError,
                "synthetic storage failure",
            ):
                runner.execute_jobs(
                    jobs=[
                        self.synthetic_job()
                    ],
                    models=
                        self.synthetic_models(),
                    config=
                        self.synthetic_config(),
                    output=
                        output,
                    state=
                        state,
                    source_before={
                        "execution_source_commit":
                            "synthetic",
                    },
                )

        evaluate_mock.assert_not_called()

        self.assertEqual(
            state[
                "failed"
            ],
            0,
        )

        self.assertEqual(
            state[
                "sampled"
            ],
            0,
        )

    # 15
    def test_evaluation_persistence_failure_aborts_campaign_execution(self):
        output = (
            self.base
            / "evaluation_persistence_failure"
        )

        output.mkdir()

        state = (
            self.synthetic_state()
        )

        with (
            mock.patch.object(
                runner,
                "assert_source_unchanged",
            ),
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
                return_value=
                    self.synthetic_dataset(),
            ),
            mock.patch.object(
                runner,
                "write_sample_evidence",
                return_value={
                    "array_sha256":
                        "sample-sha",
                },
            ),
            mock.patch.object(
                runner,
                "evaluate_frozen_dynamical_pairs",
                return_value=
                    object(),
            ),
            mock.patch.object(
                runner,
                "write_evaluation_evidence",
                side_effect=
                    OSError(
                        "synthetic evaluation storage failure"
                    ),
            ),
        ):
            with self.assertRaisesRegex(
                OSError,
                "synthetic evaluation storage failure",
            ):
                runner.execute_jobs(
                    jobs=[
                        self.synthetic_job()
                    ],
                    models=
                        self.synthetic_models(),
                    config=
                        self.synthetic_config(),
                    output=
                        output,
                    state=
                        state,
                    source_before={
                        "execution_source_commit":
                            "synthetic",
                    },
                )

        self.assertEqual(
            state[
                "sampled"
            ],
            1,
        )

        self.assertEqual(
            state[
                "evaluated"
            ],
            0,
        )

        self.assertEqual(
            state[
                "failed"
            ],
            0,
        )

    # 16
    def test_source_guard_failure_before_sampling_aborts_without_a_draw(self):
        output = (
            self.base
            / "source_change"
        )

        output.mkdir()

        state = (
            self.synthetic_state()
        )

        with (
            mock.patch.object(
                runner,
                "assert_source_unchanged",
                side_effect=
                    RuntimeError(
                        "source changed"
                    ),
            ),
            mock.patch.object(
                runner,
                "sample_dynamical_dataset",
            ) as sample_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "source changed",
            ):
                runner.execute_jobs(
                    jobs=[
                        self.synthetic_job()
                    ],
                    models=
                        self.synthetic_models(),
                    config=
                        self.synthetic_config(),
                    output=
                        output,
                    state=
                        state,
                    source_before={
                        "execution_source_commit":
                            "synthetic",
                    },
                )

        sample_mock.assert_not_called()

        self.assertEqual(
            state[
                "started"
            ],
            0,
        )

    # 17
    def test_successful_campaign_finalizes_complete_accounting_and_checksums(self):
        def execute(**kwargs):
            state = kwargs[
                "state"
            ]

            state[
                "started"
            ] = 1

            state[
                "sampled"
            ] = 1

            state[
                "evaluated"
            ] = 1

            state[
                "succeeded"
            ] = 1

        (
            _before,
            _config,
            _job,
            load_patch,
            source_patch,
            schedule_patch,
            preflight_patch,
            guard_patch,
            restore_patch,
            execute_patch,
        ) = self.campaign_patches(
            execute_side_effect=
                execute,
        )

        output = (
            self.base
            / "campaign_success"
        )

        with (
            load_patch,
            source_patch,
            schedule_patch,
            preflight_patch,
            guard_patch,
            restore_patch,
            execute_patch,
        ):
            code = runner.run_campaign(
                output
            )

        self.assertEqual(
            code,
            0,
        )

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
            1,
        )

        self.assertEqual(
            summary[
                "not_started"
            ],
            0,
        )

        self.assertEqual(
            summary[
                "started_without_terminal_event"
            ],
            0,
        )

        self.assertEqual(
            summary[
                "sampled_without_success"
            ],
            0,
        )

        self.assertTrue(
            summary[
                "source_unchanged"
            ]
        )

        self.assertTrue(
            (
                output
                / "SHA256SUMS"
            ).is_file()
        )

    # 18
    def test_completed_with_failures_has_exit_one_and_complete_accounting(self):
        def execute(**kwargs):
            state = kwargs[
                "state"
            ]

            state[
                "started"
            ] = 1

            state[
                "failed"
            ] = 1

            state[
                "failed_sampling"
            ] = 1

        patches = self.campaign_patches(
            execute_side_effect=
                execute,
        )

        output = (
            self.base
            / "campaign_failure"
        )

        with (
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
        ):
            code = runner.run_campaign(
                output
            )

        self.assertEqual(
            code,
            1,
        )

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
            "completed_with_failures",
        )

        self.assertEqual(
            summary[
                "failed"
            ],
            1,
        )

        self.assertEqual(
            summary[
                "not_started"
            ],
            0,
        )

        self.assertEqual(
            summary[
                "started_without_terminal_event"
            ],
            0,
        )

    # 19
    def test_source_change_at_finalization_invalidates_completed_campaign(self):
        before = {
            "execution_source_commit":
                "before",
            "source_file_sha256":
                {
                    "x":
                        "1",
                },
        }

        after = {
            "execution_source_commit":
                "after",
            "source_file_sha256":
                {
                    "x":
                        "1",
                },
        }

        def execute(**kwargs):
            state = kwargs[
                "state"
            ]

            state[
                "started"
            ] = 1

            state[
                "sampled"
            ] = 1

            state[
                "evaluated"
            ] = 1

            state[
                "succeeded"
            ] = 1

        patches = self.campaign_patches(
            source_side_effect=[
                before,
                after,
            ],
            execute_side_effect=
                execute,
        )

        output = (
            self.base
            / "invalid_source"
        )

        with (
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
        ):
            code = runner.run_campaign(
                output
            )

        self.assertEqual(
            code,
            2,
        )

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
            "invalid_source",
        )

        self.assertFalse(
            summary[
                "source_unchanged"
            ]
        )

    # 20
    def test_interrupt_retains_unfinished_accounting_and_exit_130(self):
        def execute(**kwargs):
            state = kwargs[
                "state"
            ]

            state[
                "started"
            ] = 1

            state[
                "sampled"
            ] = 1

            state[
                "active_attempt_id"
            ] = "s1_q2_m3_r000"

            raise KeyboardInterrupt()

        patches = self.campaign_patches(
            execute_side_effect=
                execute,
        )

        output = (
            self.base
            / "interrupted"
        )

        with (
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
        ):
            code = runner.run_campaign(
                output
            )

        self.assertEqual(
            code,
            130,
        )

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
                "error_type"
            ],
            "KeyboardInterrupt",
        )

        self.assertEqual(
            summary[
                "started_without_terminal_event"
            ],
            1,
        )

        self.assertEqual(
            summary[
                "sampled_without_success"
            ],
            1,
        )

        self.assertEqual(
            summary[
                "active_attempt_id"
            ],
            "s1_q2_m3_r000",
        )


if __name__ == "__main__":
    unittest.main()
