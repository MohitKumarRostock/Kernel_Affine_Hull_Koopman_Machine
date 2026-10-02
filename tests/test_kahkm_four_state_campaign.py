"""Runner integration checks using temporary files and test-only seed keys.

The full pilot schedule is inspected, never executed. Tiny runs use a copied
configuration with a separate seed key. Git responses and preflight subprocesses
are mocked: these checks do not verify the real machine or replace production
preflight. Numerical evaluation and file persistence are real unless explicitly
fault-injected. Preflight tests mock subprocess.run to prevent recursive test
execution. All test artifacts are temporary, not retained campaign evidence.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import experiment_20_certificate_four_state as runner


class FourStateCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="certificate-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()
        self.raw, self.pilot = runner.load_pilot()
        self.config = deepcopy(self.pilot)
        self.config["campaign_id"] = "unit_test_only"
        self.config["rng"]["campaign_key"] = 2**32-2
        self.config["cases"] = [self.config["cases"][6]]
        self.config["sampling"].update(sample_sizes=[4096], replicates_per_cell=3)
        self.jobs, self.references = runner.build_schedule(self.config)
        self.snapshot = dict(execution_source_commit="1"*40,
                             git_status_porcelain="", source_file_sha256={})

    def directory(self, name):
        path = self.base / name
        path.mkdir()
        return path

    def rows(self, output, filename):
        return [json.loads(line) for line in (output / filename).read_text().splitlines()]

    def summary(self, output):
        return json.loads((output / "campaign_summary.json").read_text())

    def state(self):
        return dict(started=0, sampled=0, succeeded=0, failed=0, active_attempt_id=None)

    def execute(self, output, state):
        runner.execute_jobs(self.jobs, self.references, self.config, output, state)

    def assert_checksums(self, output):
        entries = dict(line.split("  ", 1)[::-1]
                       for line in (output / "SHA256SUMS").read_text().splitlines())
        expected = {p.relative_to(output).as_posix() for p in output.rglob("*")
                    if p.is_file() and p.name != "SHA256SUMS"}
        self.assertEqual(set(entries), expected)
        for name, digest in entries.items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)

    def isolated_run(self, output, *, snapshots=None, preflight=None):
        raw = (json.dumps(self.config, sort_keys=True) + "\n").encode()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.source))
            stack.enter_context(patch.object(runner, "load_pilot", return_value=(raw, self.config)))
            stack.enter_context(patch.object(runner, "SUPPORTED_CONFIG_SHA256",
                                             hashlib.sha256(raw).hexdigest()))
            stack.enter_context(patch.object(runner, "source_snapshot",
                **({"return_value": self.snapshot} if snapshots is None
                   else {"side_effect": snapshots})))
            stack.enter_context(patch.object(runner, "run_preflight", side_effect=preflight))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return runner.run_campaign(output)

    def source_fixture(self):
        names = [Path(runner.__file__).name, runner.CONFIG_PATH,
                 "requirements-lock-arm64.txt", "scripts/verify_environment.py",
                 "tests/test_kahkm_four_state_campaign.py",
                 *[m + ".py" for m in runner.MODULES],
                 *["tests/test_" + m + ".py" for m in runner.MODULES]]
        committed = {}
        for name in names:
            data = ("# synthetic source fixture: " + name + "\n").encode()
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            committed[name] = data
        def fake_git(*args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(self.source).encode() + b"\n"
            if args == ("rev-parse", "HEAD"):
                return b"1"*40 + b"\n"
            if args == ("status", "--porcelain=v1", "--untracked-files=all"):
                return b""
            if args[0] == "show":
                commit, name = args[1].split(":", 1)
                self.assertEqual(commit, "1"*40)
                return committed[name]
            raise AssertionError("Unexpected Git call: " + repr(args))
        return committed, fake_git

    def fake_subprocess(self, command, **kwargs):
        kwargs["stdout"].write(b"unit-test stdout\n")
        kwargs["stderr"].write(b"unit-test stderr\n")
        return subprocess.CompletedProcess(command, 0)

    def test_pilot_fingerprint_and_identifiers(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), runner.SUPPORTED_CONFIG_SHA256)
        self.assertEqual(self.pilot["campaign_role"], "pilot")
        for field, expected in (("certificate_id", runner.CERTIFICATE_ID),
                                ("reference_id", runner.FOUR_STATE_REFERENCE_ID),
                                ("count_statistics_id", runner.COUNT_STATISTICS_ID)):
            self.assertEqual(self.pilot[field], expected)

    def test_rejects_changed_config_or_implementation_identifiers(self):
        path = self.source / runner.CONFIG_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(self.raw + b"\n")
        with patch.object(runner, "ROOT", self.source), self.assertRaises(ValueError):
            runner.load_pilot()
        for name in ("CERTIFICATE_ID", "FOUR_STATE_REFERENCE_ID", "COUNT_STATISTICS_ID"):
            with self.subTest(identifier=name), patch.object(runner, name, "incompatible"):
                with self.assertRaises(ValueError):
                    runner.load_pilot()

    def test_complete_schedule_without_sampling(self):
        with patch.object(runner, "sample_four_state_counts") as sample:
            jobs, references = runner.build_schedule(self.pilot)
            sample.assert_not_called()
        self.assertEqual(len(jobs), 10240)
        self.assertEqual(len(references), 8)
        self.assertEqual(len({j["attempt_id"] for j in jobs}), 10240)
        material = set()
        cases = {c["case_id"]: c for c in self.pilot["cases"]}
        for job in jobs:
            seed = job["seed"]
            self.assertEqual(job["attempt_number"], 1)
            self.assertEqual(seed["case_key"], cases[job["case_id"]]["case_key"])
            self.assertEqual(seed["campaign_key"], self.pilot["rng"]["campaign_key"])
            self.assertEqual(seed["root_seed"], self.pilot["rng"]["root_seed"])
            self.assertIn(seed["n_pairs"], self.pilot["sampling"]["sample_sizes"])
            self.assertIn(seed["replicate_index"], range(256))
            material.add(tuple(seed[n] for n in self.pilot["rng"]["seed_material_order"]))
        self.assertEqual(len(material), 10240)

    def test_duplicate_schedule_identifiers_rejected(self):
        config = deepcopy(self.config)
        config["cases"].append(deepcopy(config["cases"][0]))
        with self.assertRaises(ValueError):
            runner.build_schedule(config)

    def test_source_snapshot_checks_required_file_bytes(self):
        committed, fake_git = self.source_fixture()
        with patch.object(runner, "ROOT", self.source), \
             patch.object(runner, "git_bytes", side_effect=fake_git):
            snapshot = runner.source_snapshot()
        self.assertEqual(snapshot["execution_source_commit"], "1"*40)
        self.assertEqual(snapshot["git_status_porcelain"], "")
        self.assertEqual(snapshot["source_file_sha256"],
                         {name: hashlib.sha256(data).hexdigest() for name, data in committed.items()})

    def test_dirty_or_wrong_repository_root_is_rejected(self):
        _, fake_git = self.source_fixture()
        for kind in ("dirty", "wrong_root"):
            def response(*args):
                if kind == "dirty" and args[0] == "status":
                    return b"?? untracked.txt\n"
                if kind == "wrong_root" and args == ("rev-parse", "--show-toplevel"):
                    return str(self.base).encode()
                return fake_git(*args)
            with self.subTest(kind=kind), patch.object(runner, "ROOT", self.source), \
                 patch.object(runner, "git_bytes", side_effect=response), self.assertRaises(RuntimeError):
                runner.source_snapshot()

    def test_missing_symlinked_or_hidden_modified_source_is_rejected(self):
        committed, fake_git = self.source_fixture()
        path = self.source / "kahkm_four_state_counts.py"
        data = committed[path.name]
        for kind in ("missing", "symlink", "modified"):
            if path.exists() or path.is_symlink():
                path.unlink()
            if kind == "symlink":
                other = self.base / "source-copy.py"
                other.write_bytes(data)
                path.symlink_to(other)
            elif kind == "modified":
                path.write_bytes(data + b"# unreported change\n")
            with self.subTest(kind=kind), patch.object(runner, "ROOT", self.source), \
                 patch.object(runner, "git_bytes", side_effect=fake_git), self.assertRaises(RuntimeError):
                runner.source_snapshot()

    def test_json_exclusivity_finite_values_and_sync_calls(self):
        path = self.base / "record.json"
        runner.write_json(path, {"b": 2, "a": 1})
        original = path.read_bytes()
        self.assertEqual(original, b'{"a":1,"b":2}\n')
        with self.assertRaises(FileExistsError):
            runner.write_json(path, {"replacement": True})
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError):
            runner.json_line({"value": float("nan")})
        handle = Mock()
        handle.fileno.return_value = 42
        with patch.object(runner.os, "fsync") as sync:
            runner.append_record(handle, {"a": 1})
        self.assertEqual(handle.mock_calls, [call.write('{"a":1}\n'), call.flush(), call.fileno()])
        sync.assert_called_once_with(42)

    def test_checksum_inventory_detects_changed_bytes_and_cannot_be_overwritten(self):
        output = self.directory("checksums")
        runner.write_json(output / "a.json", {"a": 1})
        (output / "nested").mkdir()
        (output / "nested" / "empty.txt").write_bytes(b"")
        runner.write_checksums(output)
        self.assert_checksums(output)
        inventory = (output / "SHA256SUMS").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.write_checksums(output)
        self.assertEqual((output / "SHA256SUMS").read_bytes(), inventory)
        (output / "a.json").write_bytes(b"changed\n")
        with self.assertRaises(AssertionError):
            self.assert_checksums(output)

    def test_preflight_commands_logs_and_lock_versions(self):
        (self.source / "requirements-lock-arm64.txt").write_text("example==1.0\n")
        output = self.directory("preflight")
        with patch.object(runner, "ROOT", self.source), \
             patch.object(runner.subprocess, "run", side_effect=self.fake_subprocess) as run, \
             patch.object(runner.importlib.metadata, "version", return_value="1.0"):
            runner.run_preflight(output)
        expected = [
            [sys.executable, "scripts/verify_environment.py"],
            [sys.executable, "-m", "pip", "check"],
            [sys.executable, "-m", "pip", "freeze", "--all"],
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_kahkm_*.py", "-v"],
        ]
        self.assertEqual([c.args[0] for c in run.call_args_list], expected)
        for name, command in zip(("environment", "pip_check", "pip_freeze", "unit_tests"), expected):
            folder = output / "checks"
            self.assertEqual(json.loads((folder / f"{name}.command.json").read_text())["command"], command)
            self.assertEqual(json.loads((folder / f"{name}.result.json").read_text())["returncode"], 0)
            self.assertEqual((folder / f"{name}.stdout.log").read_bytes(), b"unit-test stdout\n")
            self.assertEqual((folder / f"{name}.stderr.log").read_bytes(), b"unit-test stderr\n")
        packages = json.loads((output / "checks" / "locked_packages.json").read_text())
        self.assertEqual(packages, [dict(name="example", expected="1.0", actual="1.0", matches=True)])

    def test_preflight_command_and_dependency_failures_stop(self):
        for kind in ("command", "version", "malformed", "missing_package"):
            output = self.directory("preflight-" + kind)
            lock = "not-pinned\n" if kind == "malformed" else "example==1.0\n"
            (self.source / "requirements-lock-arm64.txt").write_text(lock)
            def command(*args, **kwargs):
                result = self.fake_subprocess(*args, **kwargs)
                result.returncode = 2 if kind == "command" else 0
                return result
            error = (ValueError if kind == "malformed" else
                     importlib.metadata.PackageNotFoundError if kind == "missing_package" else RuntimeError)
            with self.subTest(kind=kind), patch.object(runner, "ROOT", self.source), \
                 patch.object(runner.subprocess, "run", side_effect=command) as run, \
                 patch.object(runner.importlib.metadata, "version", return_value="9.9" if kind == "version" else "1.0",
                              side_effect=importlib.metadata.PackageNotFoundError("example") if kind == "missing_package" else None):
                with self.assertRaises(error):
                    runner.run_preflight(output)
                self.assertEqual(run.call_count, 1 if kind == "command" else 4)
            self.assertTrue((output / "checks" / "environment.result.json").exists())

    def test_successful_jobs_persist_samples_before_evaluation(self):
        output, state = self.directory("jobs"), self.state()
        original = runner.evaluate_four_state_counts
        def checking_evaluator(**kwargs):
            samples = self.rows(output, "samples.jsonl")
            self.assertEqual(samples[-1]["sample"]["state_counts"], list(kwargs["state_counts"]))
            self.assertEqual(self.rows(output, "attempts.jsonl")[-1]["event"], "started")
            return original(**kwargs)
        with patch.object(runner, "evaluate_four_state_counts", side_effect=checking_evaluator) as evaluate:
            self.execute(output, state)
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(state, dict(started=3, sampled=3, succeeded=3, failed=0, active_attempt_id=None))
        events = self.rows(output, "attempts.jsonl")
        self.assertEqual([e["event"] for e in events], ["started", "completed"]*3)
        results = self.rows(output, "results.jsonl")
        for job, saved in zip(self.jobs, results):
            self.assertEqual(saved["attempt_id"], job["attempt_id"])
            self.assertEqual(saved["case_id"], job["case_id"])
            self.assertNotIn("reference", saved)

    def test_sampling_failure_is_logged_without_retry(self):
        output, state = self.directory("sampling-failure"), self.state()
        original = runner.sample_four_state_counts
        def sample(**kwargs):
            if kwargs["replicate_index"] == 0:
                raise RuntimeError("injected sampling failure")
            return original(**kwargs)
        with patch.object(runner, "sample_four_state_counts", side_effect=sample) as draw:
            self.execute(output, state)
        self.assertEqual(draw.call_count, 3)
        self.assertEqual((state["sampled"], state["succeeded"], state["failed"]), (2, 2, 1))
        failure = self.rows(output, "attempts.jsonl")[1]
        self.assertEqual((failure["event"], failure["stage"]), ("failed", "sampling"))
        self.assertIn("injected sampling failure", failure["traceback"])
        self.assertEqual(len(self.rows(output, "samples.jsonl")), 2)
        self.assertEqual(len(self.rows(output, "results.jsonl")), 2)

    def test_evaluation_failure_preserves_sample_and_continues(self):
        output, state = self.directory("evaluation-failure"), self.state()
        original = runner.evaluate_four_state_counts
        calls = 0
        def evaluate(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ArithmeticError("injected evaluation failure")
            return original(**kwargs)
        with patch.object(runner, "evaluate_four_state_counts", side_effect=evaluate):
            self.execute(output, state)
        self.assertEqual(calls, 3)
        self.assertEqual((state["sampled"], state["succeeded"], state["failed"]), (3, 2, 1))
        failure = self.rows(output, "attempts.jsonl")[1]
        self.assertEqual(failure["stage"], "evaluation")
        self.assertEqual(failure["error_type"], "ArithmeticError")
        self.assertEqual(len(self.rows(output, "samples.jsonl")), 3)
        self.assertEqual(len(self.rows(output, "results.jsonl")), 2)

    def test_evidence_write_failures_abort_instead_of_skipping(self):
        original = runner.append_record
        for filename in ("attempts.jsonl", "samples.jsonl", "results.jsonl"):
            output, state = self.directory("write-" + filename), self.state()
            def writing(handle, value):
                if Path(handle.name).name == filename:
                    raise OSError("injected storage failure")
                return original(handle, value)
            with self.subTest(filename=filename), patch.object(runner, "append_record", side_effect=writing):
                with self.assertRaises(OSError):
                    self.execute(output, state)
            self.assertEqual(state["succeeded"], 0)
            self.assertEqual(state["failed"], 0)
            self.assertEqual(state["active_attempt_id"], self.jobs[0]["attempt_id"])
            if filename == "samples.jsonl":
                self.assertEqual((state["started"], state["sampled"]), (1, 0))

    def test_nonfinite_result_is_retained_as_calculation_failure(self):
        output, state = self.directory("nonfinite"), self.state()
        original = runner.evaluate_four_state_counts
        def invalid_result(**kwargs):
            return replace(original(**kwargs), gap_to_optimum=float("nan"))
        with patch.object(runner, "evaluate_four_state_counts", side_effect=invalid_result):
            self.execute(output, state)
        self.assertEqual((state["sampled"], state["failed"], state["succeeded"]), (3, 3, 0))
        self.assertEqual(self.rows(output, "results.jsonl"), [])
        self.assertTrue(all(e["stage"] == "evaluation" for e in self.rows(output, "attempts.jsonl") if e["event"] == "failed"))

    def test_zero_certificate_and_coverage_violation_are_successful_executions(self):
        self.config["cases"][0]["p"] = 1.0
        self.config["sampling"].update(sample_sizes=[8192], replicates_per_cell=2)
        self.jobs, self.references = runner.build_schedule(self.config)
        output, state = self.directory("valid-outcomes"), self.state()
        original = runner.sample_four_state_counts
        def fixture_sample(**kwargs):
            sample = original(**kwargs)
            counts = (8192, 0, 0, 0) if kwargs["replicate_index"] == 0 else (4096, 4096, 0, 0)
            return replace(sample, state_counts=counts)
        with patch.object(runner, "sample_four_state_counts", side_effect=fixture_sample):
            self.execute(output, state)
        results = self.rows(output, "results.jsonl")
        self.assertTrue(results[0]["zero_certificate"])
        self.assertFalse(results[1]["strict_coverage"])
        self.assertLess(results[1]["gap_to_optimum"], 0)
        self.assertEqual((state["succeeded"], state["failed"]), (2, 0))

    def test_complete_tiny_runs_are_reproducible_and_fully_inventoried(self):
        outputs = [self.base / "complete-a", self.base / "complete-b"]
        for output in outputs:
            self.assertEqual(self.isolated_run(output), 0)
            state = self.summary(output)
            self.assertEqual(state["status"], "completed")
            self.assertEqual((state["planned"], state["started"], state["sampled"], state["succeeded"]), (3,)*4)
            self.assertEqual((state["failed"], state["not_started"], state["started_without_terminal_event"]), (0,)*3)
            self.assertTrue(state["source_unchanged"])
            self.assertEqual(state["independent_evidence_verification"], "not_yet_performed")
            self.assertEqual(self.rows(output, "schedule.jsonl"), self.jobs)
            self.assertEqual(json.loads((output / "references.json").read_text()),
                             json.loads(json.dumps({k: asdict(v) for k, v in self.references.items()})))
            metadata = json.loads((output / "campaign_metadata.json").read_text())
            self.assertEqual(metadata["source"], self.snapshot)
            self.assertEqual(metadata["config_sha256"], hashlib.sha256((output / "campaign_spec.json").read_bytes()).hexdigest())
            self.assert_checksums(output)
        for name in ("samples.jsonl", "results.jsonl", "schedule.jsonl", "references.json"):
            self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())

    def test_output_and_dirty_source_guards_precede_execution(self):
        existing = self.directory("existing")
        marker = existing / "keep.txt"
        marker.write_text("keep")
        for output, error in ((existing, FileExistsError), (self.source, ValueError),
                               (self.source / "nested", ValueError)):
            with self.subTest(output=output), patch.object(runner, "ROOT", self.source), \
                 patch.object(runner, "load_pilot") as load, self.assertRaises(error):
                runner.run_campaign(output)
            load.assert_not_called()
        self.assertEqual(marker.read_text(), "keep")
        output = self.base / "dirty-source-run"
        with self.assertRaises(RuntimeError):
            self.isolated_run(output, snapshots=[RuntimeError("dirty source")])
        self.assertFalse(output.exists())

    def test_preflight_failure_keeps_schedule_and_aborted_summary(self):
        output = self.base / "preflight-abort"
        def preflight(path):
            self.assertEqual(self.rows(path, "schedule.jsonl"), self.jobs)
            self.assertTrue((path / "campaign_metadata.json").is_file())
            (path / "test-preflight.log").write_text("injected failure\n")
            raise RuntimeError("injected preflight failure")
        with patch.object(runner, "sample_four_state_counts") as sample:
            self.assertEqual(self.isolated_run(output, preflight=preflight), 2)
            sample.assert_not_called()
        state = self.summary(output)
        self.assertEqual((state["status"], state["aborted_stage"]), ("aborted", "preflight"))
        self.assertEqual((state["started"], state["not_started"]), (0, 3))
        self.assertFalse((output / "samples.jsonl").exists())
        self.assert_checksums(output)

    def test_source_changes_before_or_after_sampling_invalidate_run(self):
        changed = dict(self.snapshot, execution_source_commit="2"*40)
        scenarios = [([self.snapshot, changed, changed], "aborted", 0),
                     ([self.snapshot, self.snapshot, changed], "invalid_source", 3),
                     ([self.snapshot, self.snapshot, RuntimeError("inspection failure")], "invalid_source", 3)]
        for index, (snapshots, status, started) in enumerate(scenarios):
            output = self.base / f"source-change-{index}"
            with self.subTest(index=index):
                self.assertEqual(self.isolated_run(output, snapshots=snapshots), 2)
                state = self.summary(output)
                self.assertEqual((state["status"], state["started"]), (status, started))
                self.assertFalse(state["source_unchanged"])
                self.assert_checksums(output)

    def test_completed_with_failures_has_nonzero_exit_and_complete_accounting(self):
        output = self.base / "failed-calculations"
        with patch.object(runner, "evaluate_four_state_counts", side_effect=RuntimeError("injected failure")):
            self.assertEqual(self.isolated_run(output), 1)
        state = self.summary(output)
        self.assertEqual(state["status"], "completed_with_failures")
        self.assertEqual((state["started"], state["sampled"], state["failed"]), (3, 3, 3))
        self.assertEqual((state["succeeded"], state["not_started"], state["started_without_terminal_event"]), (0, 0, 0))
        self.assertEqual(len(self.rows(output, "samples.jsonl")), 3)
        self.assertEqual(len(self.rows(output, "attempts.jsonl")), 6)
        self.assert_checksums(output)

    def test_interrupt_retains_sample_and_marks_unfinished_attempt(self):
        output = self.base / "interrupted"
        with patch.object(runner, "evaluate_four_state_counts", side_effect=KeyboardInterrupt):
            self.assertEqual(self.isolated_run(output), 130)
        state = self.summary(output)
        self.assertEqual((state["status"], state["aborted_stage"]), ("aborted", "running"))
        self.assertEqual((state["started"], state["sampled"], state["started_without_terminal_event"]), (1, 1, 1))
        self.assertEqual(state["not_started"], 2)
        self.assertEqual(state["active_attempt_id"], self.jobs[0]["attempt_id"])
        self.assertEqual(len(self.rows(output, "samples.jsonl")), 1)
        self.assertEqual(self.rows(output, "results.jsonl"), [])
        self.assert_checksums(output)

    def test_cli_failure_and_import_do_not_launch_other_runs(self):
        errors = io.StringIO()
        with patch.object(sys, "argv", [runner.__file__, "--output", str(self.base / "cli")]), \
             patch.object(runner, "run_campaign", side_effect=OSError("injected finalization failure")) as run, \
             redirect_stderr(errors):
            self.assertEqual(runner.main(), 2)
        run.assert_called_once()
        self.assertIn("Retain any created output directory", errors.getvalue())
        with patch.object(runner.subprocess, "check_output", side_effect=AssertionError("Unexpected Git read")), \
             patch.object(runner.subprocess, "run", side_effect=AssertionError("Unexpected preflight")):
            loaded = runpy.run_path(str(Path(runner.__file__).resolve()), run_name="_runner_import_test_")
        self.assertEqual(loaded["RUNNER_ID"], runner.RUNNER_ID)

    def test_checksum_finalization_failure_does_not_return_success(self):
        output = self.base / "incomplete-finalization"
        with patch.object(runner, "write_checksums", side_effect=OSError("injected checksum failure")):
            with self.assertRaises(OSError):
                self.isolated_run(output)
        self.assertTrue((output / "campaign_summary.json").exists())
        self.assertFalse((output / "SHA256SUMS").exists())
        self.assertEqual(len(self.rows(output, "results.jsonl")), 3)
        # A summary alone is insufficient evidence of a finalized campaign.


if __name__ == "__main__":
    unittest.main()
