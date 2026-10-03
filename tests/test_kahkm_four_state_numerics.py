"""Tests of independent numerical verification, not new campaign evidence.

A temporary 10,240-record fixture uses hand-specified counts and a test-only
campaign key. Production modules produce its saved answers; the verifier
recomputes them independently. No random samples or actual pilot files are
used. A deliberately adverse count vector ensures legitimate coverage
failures are accepted. Corruption tests recompute fixture checksums so they
exercise checks beyond file integrity. Git and CLI tests use explicit mocks.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
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
SCRIPT = ROOT / "scripts" / "verify_four_state_pilot_numerics.py"
spec = importlib.util.spec_from_file_location("_numerics_verifier_under_test", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def inventory(folder):
    lines = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(folder).as_posix()}\n")
    (folder / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


class FourStateNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="certificate-verifier-test-")
        cls.addClassCleanup(temporary.cleanup)
        cls.base = Path(temporary.name).resolve()
        cls.evidence = cls.base / "synthetic-evidence"
        cls.evidence.mkdir()
        cls.pilot_bytes = (ROOT / "reproduction/certificates/configs/four_state_pilot_v1.json").read_bytes()
        cls.config = json.loads(cls.pilot_bytes)
        cls.config["campaign_id"] = "unit_test_numerical_verifier_only"
        cls.config["rng"]["campaign_key"] = 2**32 - 3
        raw = encoded(cls.config)
        cls.config_hash = hashlib.sha256(raw).hexdigest()
        (cls.evidence / "campaign_spec.json").write_bytes(raw)
        source = dict(execution_source_commit="a"*40, git_status_porcelain="",
                      source_file_sha256={})
        metadata = dict(config_sha256=cls.config_hash, campaign_role="pilot",
                        campaign_id=cls.config["campaign_id"], source=source)
        (cls.evidence / "campaign_metadata.json").write_bytes(encoded(metadata))
        (cls.evidence / "source_after.json").write_bytes(encoded(source))
        cls.references = {}
        cls.expected_outcomes = dict(zero_certificates=0, strict_coverage_failures=0,
                                     tolerance_aware_coverage_failures=0)
        streams = {name: [] for name in ("schedule.jsonl", "samples.jsonl",
                                        "results.jsonl", "attempts.jsonl")}
        for case in cls.config["cases"]:
            reference = four_state_reference(**{k: case[k] for k in ("p", "dynamics", "kappa")})
            cls.references[case["case_id"]] = asdict(reference)
            cache = {}
            for n in cls.config["sampling"]["sample_sizes"]:
                for rep in range(cls.config["sampling"]["replicates_per_cell"]):
                    key = f"c{case['case_key']}_m{n}_r{rep}"
                    seed = dict(campaign_key=cls.config["rng"]["campaign_key"],
                                case_key=case["case_key"], n_pairs=n,
                                replicate_index=rep, root_seed=cls.config["rng"]["root_seed"])
                    shift = rep % 3
                    counts = (n//4+shift, n//4-shift, n//4, n//4)
                    if case["case_key"] == 7 and n == 8192 and rep == 0:
                        counts = (4096, 4096, 0, 0)  # Known adverse fixture, not a draw.
                    if counts not in cache:
                        options = cls.config["evaluation"]
                        cache[counts] = asdict(evaluate_four_state_counts(
                            reference=reference, state_counts=counts,
                            delta=options["delta"], exclusion_tolerances=options["exclusion_tolerances"],
                            comparison_atol=options["comparison_atol"],
                        ))
                        cache[counts].pop("reference")
                    result = dict(cache[counts], attempt_id=key, case_id=case["case_id"])
                    sample = dict(seed, sampler_id="four_state_multinomial_pcg64_v1",
                                  numpy_version="synthetic-test-fixture-not-a-draw",
                                  seed_material_order=cls.config["rng"]["seed_material_order"],
                                  seed_material=[seed[k] for k in cls.config["rng"]["seed_material_order"]],
                                  bit_generator="PCG64", seed_sequence="SeedSequence",
                                  seed_sequence_pool_size=4, draws_per_replicate=1,
                                  state_order=["a", "b", "d", "e"], probabilities=[.25]*4,
                                  state_counts=counts)
                    streams["schedule.jsonl"].append(encoded(dict(
                        attempt_id=key, attempt_number=1, case_id=case["case_id"], seed=seed)))
                    streams["samples.jsonl"].append(encoded(dict(attempt_id=key, sample=sample)))
                    streams["results.jsonl"].append(encoded(result))
                    for event in ("started", "completed"):
                        streams["attempts.jsonl"].append(encoded(dict(attempt_id=key, event=event)))
                    cls.expected_outcomes["zero_certificates"] += result["zero_certificate"]
                    cls.expected_outcomes["strict_coverage_failures"] += not result["strict_coverage"]
                    cls.expected_outcomes["tolerance_aware_coverage_failures"] += not result["tolerance_aware_coverage"]
        for name, lines in streams.items():
            (cls.evidence / name).write_bytes(b"".join(lines))
        (cls.evidence / "references.json").write_bytes(encoded(cls.references))
        summary = dict(status="completed", exit_code=0, source_unchanged=True,
                       active_attempt_id=None, planned=10240, started=10240,
                       sampled=10240, succeeded=10240, failed=0, not_started=0,
                       started_without_terminal_event=0)
        (cls.evidence / "campaign_summary.json").write_bytes(encoded(summary))
        inventory(cls.evidence)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=self.base, prefix="case-")
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name)

    @contextmanager
    def changed(self, changes, rehash=True):
        """Only mutate the temporary fixture, restoring every byte afterward."""
        names = set(changes) | {"SHA256SUMS"}
        before = {n: (self.evidence/n).read_bytes() if (self.evidence/n).exists() else None
                  for n in names}
        try:
            for name, data in changes.items():
                path = self.evidence/name
                if data is None:
                    path.unlink()
                else:
                    path.write_bytes(data)
            if rehash:
                inventory(self.evidence)
            yield
        finally:
            for name, data in before.items():
                path = self.evidence/name
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(data)

    def first_record_change(self, name, path, value):
        first, _, rest = (self.evidence/name).read_bytes().partition(b"\n")
        record = json.loads(first)
        target = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return {name: encoded(record)+rest}

    def verify(self):
        with patch.object(verifier, "CONFIG_SHA256", self.config_hash):
            return verifier.verify_evidence(self.evidence)

    def test_committed_configuration_and_test_only_isolation(self):
        self.assertEqual(hashlib.sha256(self.pilot_bytes).hexdigest(), verifier.CONFIG_SHA256)
        original = json.loads(self.pilot_bytes)
        self.assertNotEqual(self.config["campaign_id"], original["campaign_id"])
        self.assertNotEqual(self.config["rng"]["campaign_key"], original["rng"]["campaign_key"])
        self.assertEqual(len(self.config["cases"]), 8)

    def test_strict_json_and_record_indexing(self):
        for text in ('{"a":1,"a":2}', '{"a":{"b":1,"b":2}}',
                     '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                verifier.read_json(text)
        self.assertEqual(verifier.read_json('{"a":[1,true,null]}'), {"a": [1, True, None]})
        path = self.scratch/"rows.jsonl"
        for data in (b'{"attempt_id":"x"}\n\n', b'{"attempt_id":"x"}\n'*2):
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                verifier.index_records(self.scratch, path.name)

    def test_exact_rational_statistics_and_empty_classes(self):
        s = verifier.weighted_statistics(31/32, "conflicting_successors", (9, 1, 0, 2))
        self.assertEqual(s["f"], Fraction(1, 1024))
        self.assertEqual(s["s"], Fraction(135, 1024))
        self.assertEqual(s["sizes"], [10, 2])
        self.assertEqual(s["means"], [(Fraction(7, 8), Fraction(1, 8)),
                                       (Fraction(1, 32), Fraction(31, 32))])
        self.assertEqual(s["sses"], [Fraction(405, 256), Fraction(0)])
        empty = verifier.weighted_statistics(31/32, "conflicting_successors", (0, 0, 3, 7))
        self.assertEqual(empty["means"][0], (Fraction(1, 2),)*2)
        self.assertEqual(empty["s"], 0)
        for dynamics, p in (("identity", .8), ("conflicting_successors", .5)):
            self.assertEqual(verifier.weighted_statistics(p, dynamics, (3, 1, 2, 7))["s"], 0)

    def test_decimal_calculation_matches_original_component_grid(self):
        for f, s, n, k, delta in ((1/1024, 225/1024, 4096, math.sqrt(2), .05),
                                  (0., .5, 8192, 0., .05),
                                  (1., 1., 3, 1., .05),
                                  (.02, .4, 1000000, .75, 5e-324)):
            with self.subTest(n=n, delta=delta):
                independent = verifier.independent_certificate(f, s, n, k, delta)
                original = certificate_from_statistics(f_hat=f, s_hat=s, n_pairs=n, kappa=k, delta=delta)
                for name, expected in independent.items():
                    self.assertTrue(math.isclose(getattr(original, name), expected,
                                                rel_tol=5e-12, abs_tol=5e-14))
        balanced = verifier.independent_certificate(Fraction(1, 1024), Fraction(225, 1024),
                                                      4096, math.sqrt(2), .05)
        self.assertGreater(balanced["L_kappa_delta"], .30)
        self.assertLess(balanced["L_kappa_delta"], .31)

    def test_numerical_comparison_rejects_wrong_types_shapes_and_large_errors(self):
        c = verifier.Comparison()
        c.close([1., 2.+1e-13], [1, 2], "test")
        self.assertEqual(c.numbers_checked, 2)
        self.assertGreater(c.max_errors["test"], 0.)
        for actual, expected in ((True, 1), ("1", 1), (math.nan, 0),
                                 (math.inf, 0), ([1], [1, 2]), (1.01, 1)):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                c.close(actual, expected, "invalid")

    def test_reference_checks_include_attainment_and_norm_budget(self):
        for case in self.config["cases"]:
            saved = json.loads(encoded(self.references[case["case_id"]]))
            verifier.verify_reference(saved, case, verifier.Comparison())
        case = self.config["cases"][0]  # Collapsed identity representation.
        saved = json.loads(encoded(self.references[case["case_id"]]))
        saved["optimal_B"] = [[3., -2.], [-2., 3.]]
        # Same predictions on (.5,.5), but operator norm 5 violates the budget.
        with self.assertRaisesRegex(ValueError, "exceeds norm budget"):
            verifier.verify_reference(saved, case, verifier.Comparison())
        saved["optimal_B"] = [[0., 0.], [0., 0.]]
        with self.assertRaisesRegex(ValueError, "conditional_mean_attainment"):
            verifier.verify_reference(saved, case, verifier.Comparison())

    def test_inventory_detects_changed_missing_and_unlisted_files(self):
        (self.scratch/"a.json").write_bytes(b"{}\n")
        inventory(self.scratch)
        fingerprint, count = verifier.check_inventory(self.scratch)
        self.assertEqual(count, 1)
        self.assertEqual(fingerprint, hashlib.sha256((self.scratch/"SHA256SUMS").read_bytes()).hexdigest())
        for kind in ("changed", "missing", "extra"):
            with self.subTest(kind=kind):
                (self.scratch/"a.json").write_bytes(b"{}\n")
                (self.scratch/"extra").unlink(missing_ok=True)
                if kind == "changed":
                    (self.scratch/"a.json").write_bytes(b"changed")
                elif kind == "missing":
                    (self.scratch/"a.json").unlink()
                else:
                    (self.scratch/"extra").write_bytes(b"extra")
                with self.assertRaises(ValueError):
                    verifier.check_inventory(self.scratch)

    def test_inventory_rejects_unsafe_duplicate_entries_and_symlinks(self):
        path = self.scratch/"a"
        path.write_bytes(b"x")
        checksum = hashlib.sha256(b"x").hexdigest()
        for text in (f"{checksum}  ../a\n", f"{checksum}  /a\n", f"{checksum}  SHA256SUMS\n",
                     "not-a-checksum  a\n", f"{checksum}  a\n"*2):
            with self.subTest(text=text):
                (self.scratch/"SHA256SUMS").write_text(text)
                with self.assertRaises(ValueError):
                    verifier.check_inventory(self.scratch)
        inventory(self.scratch)
        path.unlink()
        other = self.scratch/"target"
        other.write_bytes(b"x")
        path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "symlink"):
            verifier.check_inventory(self.scratch)

    def test_full_synthetic_evidence_recomputation_is_read_only(self):
        before = (self.evidence/"SHA256SUMS").read_bytes()
        report = self.verify()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["verified_replicates"], 10240)
        self.assertEqual(report["verified_references"], 8)
        self.assertEqual(report["recorded_outcome_counts"], self.expected_outcomes)
        self.assertGreater(report["recorded_outcome_counts"]["strict_coverage_failures"], 0)
        self.assertGreater(report["recorded_outcome_counts"]["zero_certificates"], 0)
        self.assertGreater(report["numerical_values_checked"], 10240)
        self.assertEqual(report["decimal_precision"], 80)
        self.assertEqual(report["random_samples_regenerated"], 0)
        self.assertIs(report["evidence_modified"], False)
        self.assertEqual((self.evidence/"SHA256SUMS").read_bytes(), before)

    def test_rehashed_numerical_corruption_is_rejected(self):
        cases = ((["count_statistics", "statistics", "f_hat"], .125),
                 (["count_statistics", "statistics", "class_successor_sse", 0], 1.),
                 (["certificate", "L_kappa_delta"], .01),
                 (["certificate", "V_delta"], 1e-15))
        for path, value in cases:
            with self.subTest(path=path), self.changed(self.first_record_change("results.jsonl", path, value)):
                verifier.check_inventory(self.evidence)  # The new checksums are consistent.
                with self.assertRaises(ValueError):
                    self.verify()

    def test_rehashed_flags_exclusions_and_signed_gaps_are_rejected(self):
        cases = ((["strict_coverage"], False), (["zero_certificate"], False),
                 (["exclusions", 0, "excluded"], True), (["comparison_atol"], .1),
                 (["gap_to_optimum"], .01), (["sampling_gap"], 1e-15))
        for path, value in cases:
            with self.subTest(path=path), self.changed(self.first_record_change("results.jsonl", path, value)):
                with self.assertRaises(ValueError):
                    self.verify()

    def test_rehashed_seed_metadata_and_invalid_counts_are_rejected(self):
        cases = ((["sample", "case_key"], 99), (["sample", "state_counts"], [128, 0, 0, 1]),
                 (["sample", "state_counts"], [True, 31, 32, 64]),
                 (["sample", "bit_generator"], "other"))
        for path, value in cases:
            with self.subTest(path=path), self.changed(self.first_record_change("samples.jsonl", path, value)):
                with self.assertRaises(ValueError):
                    self.verify()

    def test_rehashed_schedule_and_attempt_errors_are_rejected(self):
        cases = (("schedule.jsonl", ["seed", "root_seed"], 0),
                 ("attempts.jsonl", ["event"], "completed"),
                 ("results.jsonl", ["attempt_id"], "unscheduled"))
        for name, path, value in cases:
            with self.subTest(name=name), self.changed(self.first_record_change(name, path, value)):
                with self.assertRaises(ValueError):
                    self.verify()
        raw = (self.evidence/"samples.jsonl").read_bytes()
        first, _, rest = raw.partition(b"\n")
        for changed in (rest, first+b"\n"+raw):
            with self.changed({"samples.jsonl": changed}), self.assertRaises(ValueError):
                self.verify()

    def test_wrong_config_and_source_records_are_rejected(self):
        raw = (self.evidence/"campaign_spec.json").read_bytes()
        with self.changed({"campaign_spec.json": raw+b"\n"}), self.assertRaisesRegex(ValueError, "Wrong pilot configuration"):
            self.verify()
        source = json.loads((self.evidence/"source_after.json").read_bytes())
        source["execution_source_commit"] = "b"*40
        with self.changed({"source_after.json": encoded(source)}), self.assertRaisesRegex(ValueError, "Source records differ"):
            self.verify()

    def test_changed_summary_is_detected_after_recomputation(self):
        saved = json.loads((self.evidence/"campaign_summary.json").read_bytes())
        saved["succeeded"] = 10239
        with self.changed({"campaign_summary.json": encoded(saved)}):
            with self.assertRaisesRegex(ValueError, "Summary mismatch: succeeded"):
                self.verify()

    def test_inventory_change_during_verification_is_rejected(self):
        original = verifier.check_inventory
        calls = 0
        def changing(folder):
            nonlocal calls
            calls += 1
            digest, count = original(folder)
            return (digest if calls == 1 else "0"*64), count
        with patch.object(verifier, "check_inventory", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                self.verify()
        self.assertEqual(calls, 2)

    def test_verifier_provenance_checks_committed_bytes(self):
        commit = "b"*40
        raw = SCRIPT.read_bytes()
        with patch.object(verifier.subprocess, "check_output",
                          side_effect=[b"", commit.encode()+b"\n", raw]) as git:
            report = verifier.verifier_provenance()
        self.assertEqual(report["verifier_source_commit"], commit)
        self.assertEqual(report["verifier_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["verifier_path"], "scripts/verify_four_state_pilot_numerics.py")
        self.assertEqual(git.call_count, 3)
        for responses in ([b"?? uncommitted.py\n"], [b"", commit.encode(), raw+b"\n"]):
            with patch.object(verifier.subprocess, "check_output", side_effect=responses):
                with self.assertRaises(ValueError):
                    verifier.verifier_provenance()

    def test_cli_success_prints_a_separate_provenance_report(self):
        output = io.StringIO()
        with patch.object(sys, "argv", [str(SCRIPT), "--evidence", str(self.evidence)]), \
             patch.object(verifier, "verifier_provenance", return_value={"verifier_source_commit": "b"*40}), \
             patch.object(verifier, "verify_evidence", return_value={"status": "passed"}) as verify, \
             redirect_stdout(output):
            self.assertEqual(verifier.main(), 0)
        self.assertEqual(json.loads(output.getvalue()), dict(status="passed", verifier_source_commit="b"*40))
        verify.assert_called_once_with(self.evidence)

    def test_cli_failure_never_prints_a_success_report(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", [str(SCRIPT), "--evidence", str(self.evidence)]), \
             patch.object(verifier, "verifier_provenance", side_effect=ValueError("dirty verifier")), \
             patch.object(verifier, "verify_evidence") as verify, \
             redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(verifier.main(), 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("dirty verifier", errors.getvalue())
        verify.assert_not_called()

    def test_import_does_not_read_evidence_or_invoke_git(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected file read")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected file read")), \
             patch.object(verifier.subprocess, "check_output", side_effect=AssertionError("unexpected Git")):
            loaded = runpy.run_path(str(SCRIPT), run_name="_numerics_import_test_")
        self.assertEqual(loaded["VERIFIER_ID"], verifier.VERIFIER_ID)


if __name__ == "__main__":
    unittest.main()
