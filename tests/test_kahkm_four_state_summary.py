"""Summary tests with hand-constructed records, not experimental evidence.

Intervals are checked against binomial tail sums and closed-form endpoints.
Quantiles are checked by direct interpolation. Tiny file-pipeline tests mock
numerical auditing and Git provenance, which have separate tests; they test
aggregation and persistence, not the scientific validity of synthetic rows.
No actual pilot evidence or archives are opened, and no samples are drawn.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_four_state_pilot.py"
spec = importlib.util.spec_from_file_location("_pilot_summary_under_test", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture():
    cases, references, rows = [], {}, []
    definitions = (
        ("identity", 31/32, 0., 0.),
        ("conflicting_successors", .8, .3, 0.),
        ("conflicting_successors", 31/32, 15/32, 13/32),
        ("conflicting_successors", 1., .5, .5),
    )
    tolerances = [0., .3, .5]
    for key, (dynamics, p, optimum, limit) in enumerate(definitions):
        name = f"test_only_case_{key}"
        cases.append(dict(case_id=name, case_key=key, dynamics=dynamics,
                          p=p, endpoint_case=p == 1., kappa=math.sqrt(2)))
        references[name] = dict(exact_optimal_rmse=optimum,
                                population_certificate_limit=limit)
        for n in (128, 4096):
            values = [0.]*4
            if n == 4096 and key == 2:
                values = [0., .25, .3125, .5]
            elif n == 4096 and key == 3:
                values = [.125, .5, .5, .625]
            for rep, lower in enumerate(values):
                rows.append(dict(
                    attempt_id=f"c{key}_m{n}_r{rep}", case_id=name,
                    certificate=dict(n_pairs=n, L_kappa_delta=lower,
                                     F_delta=.01, V_delta=.5, s_hat=.5),
                    gap_to_optimum=optimum-lower,
                    sampling_gap=limit-lower,
                    zero_certificate=lower == 0.,
                    strict_coverage=lower <= optimum,
                    tolerance_aware_coverage=lower <= optimum+1e-12,
                    exclusions=[dict(
                        tolerance=t, excluded=lower > t,
                        exact_error_exceeds_tolerance=optimum > t,
                        erroneous_exclusion=lower > t and optimum <= t,
                    ) for t in tolerances],
                ))
    config = dict(
        campaign_id="unit_test_summary_only", campaign_role="pilot", cases=cases,
        sampling=dict(sample_sizes=[4096, 128], replicates_per_cell=4),
        evaluation=dict(delta=.05, comparison_atol=1e-12,
                        binomial_interval_method="clopper_pearson",
                        binomial_interval_confidence=.95, quantile_method="linear",
                        certificate_quantiles=[.05, .5, .95],
                        exclusion_tolerances=tolerances),
    )
    return config, references, rows


def direct_quantile(values, probability):
    ordered = sorted(values)
    index = probability*(len(ordered)-1)
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (index-lo)*(ordered[hi]-ordered[lo])


class FourStateSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="certificate-summary-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source, self.evidence = self.base/"source", self.base/"synthetic-input"
        self.source.mkdir()
        self.evidence.mkdir()
        self.config, self.references, self.rows = fixture()
        for name, value in (("campaign_spec.json", self.config),
                            ("references.json", self.references)):
            (self.evidence/name).write_text(json.dumps(value)+"\n")
        (self.evidence/"results.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in self.rows))
        self.provenance = dict(summary_source_commit="b"*40, source_sha256={})
        self.audit = dict(status="passed", verified_replicates=32,
                          evidence_manifest_sha256="1"*64,
                          execution_source_commit="a"*40, config_sha256="2"*64)
        self.api = SimpleNamespace(
            verify_evidence=Mock(return_value=self.audit),
            document=lambda folder, name: json.loads((folder/name).read_text()),
            records=lambda folder, name: (
                json.loads(line) for line in (folder/name).read_text().splitlines()
            ),
            check_inventory=Mock(return_value=("1"*64, 3)),
        )

    def aggregate(self, config=None, references=None, rows=None):
        return summary.summarize_cells(
            self.config if config is None else config,
            self.references if references is None else references,
            self.rows if rows is None else rows,
        )

    def run_tiny(self, output, source_responses=None):
        settings = ({"return_value": self.provenance} if source_responses is None
                    else {"side_effect": source_responses})
        with patch.object(summary, "ROOT", self.source), \
             patch.object(summary, "source_provenance", **settings), \
             patch.object(summary, "load_verifier", return_value=self.api):
            return summary.summarize_evidence(self.evidence, output)

    def source_fixture(self, wrong_versions=False):
        names = ("scripts/summarize_four_state_pilot.py",
                 "scripts/verify_four_state_pilot_numerics.py",
                 "tests/test_kahkm_four_state_summary.py", "requirements-lock-arm64.txt")
        committed = {}
        for name in names:
            path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ("# synthetic source: "+name+"\n").encode()
            if name.endswith(".txt"):
                version = "0.0" if wrong_versions else summary.scipy.__version__
                data = (f"numpy=={summary.np.__version__}\nscipy=={version}\n").encode()
            path.write_bytes(data)
            committed[name] = data
        def git(command, **kwargs):
            args = command[1:]
            if args[0] == "status":
                return b""
            if args == ["rev-parse", "HEAD"]:
                return b"b"*40+b"\n"
            if args[0] == "show":
                commit, name = args[1].split(":", 1)
                self.assertEqual(commit, "b"*40)
                return committed[name]
            raise AssertionError("Unexpected Git command")
        return committed, git

    def test_binomial_endpoints_have_closed_form_intervals(self):
        for n in (1, 4, 256):
            for confidence in (.8, .95):
                tail = (1-confidence)/2
                with self.subTest(n=n, confidence=confidence):
                    zero = summary.binomial_rate([False]*n, confidence)
                    full = summary.binomial_rate([True]*n, confidence)
                    self.assertEqual((zero["count"], zero["trials"], zero["rate"]), (0, n, 0.))
                    self.assertEqual((full["count"], full["trials"], full["rate"]), (n, n, 1.))
                    self.assertEqual(zero["ci_low"], 0.)
                    self.assertEqual(full["ci_high"], 1.)
                    self.assertAlmostEqual(zero["ci_high"], 1-tail**(1/n), delta=5e-12)
                    self.assertAlmostEqual(full["ci_low"], tail**(1/n), delta=5e-12)
                    self.assertGreater(zero["ci_high"], 0.)
                    self.assertLess(full["ci_low"], 1.)

    def test_interior_intervals_satisfy_binomial_tail_equations(self):
        for successes, n in ((1, 2), (3, 7), (7, 10)):
            with self.subTest(successes=successes, n=n):
                result = summary.binomial_rate([True]*successes+[False]*(n-successes), .95)
                lo, hi = result["ci_low"], result["ci_high"]
                lower_tail = math.fsum(math.comb(n, j)*lo**j*(1-lo)**(n-j)
                                      for j in range(successes, n+1))
                upper_tail = math.fsum(math.comb(n, j)*hi**j*(1-hi)**(n-j)
                                      for j in range(successes+1))
                self.assertAlmostEqual(lower_tail, .025, delta=5e-12)
                self.assertAlmostEqual(upper_tail, .025, delta=5e-12)
                self.assertEqual(result["rate"], successes/n)

    def test_binomial_inputs_are_strictly_validated(self):
        for flags in ([], [0, 1], [True, None], [summary.np.bool_(True)], ["true"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                summary.binomial_rate(flags, .95)
        for confidence in (0., 1., -.1, math.nan, math.inf, True, ".95"):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                summary.binomial_rate([True], confidence)

    def test_descriptive_quantiles_use_direct_linear_interpolation(self):
        for values in ([-3., -1., 0., 2., 7.], [0., .25, .3125, .5], [2.]):
            with self.subTest(values=values):
                result = summary.distribution(iter(values))
                for name, probability in (("q05", .05), ("median", .5), ("q95", .95)):
                    self.assertAlmostEqual(result[name], direct_quantile(values, probability), delta=1e-14)
                self.assertEqual(result["mean"], math.fsum(values)/len(values))
                self.assertEqual((result["minimum"], result["maximum"]), (min(values), max(values)))
                self.assertEqual(result, summary.distribution(reversed(values)))
        self.assertEqual(summary.distribution([1e16, 1., -1e16])["mean"], 1/3)

    def test_distribution_rejects_invalid_or_nonfinite_values(self):
        for values in ([], [True], ["1"], [None], [math.nan], [math.inf], [-math.inf], [1j]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summary.distribution(values)

    def test_cell_accounting_and_trial_denominators(self):
        cells, exclusions = self.aggregate()
        self.assertEqual((len(cells), len(exclusions)), (8, 24))
        self.assertEqual(sum(c["replicates"] for c in cells), 32)
        self.assertEqual([(c["case_key"], c["n_pairs"]) for c in cells],
                         [(key, n) for key in range(4) for n in (128, 4096)])
        for cell in cells:
            for flag in ("zero_certificate", "strict_coverage", "tolerance_aware_coverage"):
                self.assertEqual(cell[flag+"_trials"], 4)
                self.assertEqual(cell[flag+"_rate"], cell[flag+"_count"]/4)
        self.assertTrue(all(row["trials"] == 4 for row in exclusions))

    def test_zero_bounds_signed_gaps_and_undefined_ratios_are_retained(self):
        cells, _ = self.aggregate()
        by_key = {(c["case_key"], c["n_pairs"]): c for c in cells}
        for n in (128, 4096):
            self.assertIsNone(by_key[(0, n)]["median_certificate_over_optimum"])
            self.assertEqual(by_key[(0, n)]["zero_certificate_rate"], 1.)
            self.assertEqual(by_key[(1, n)]["median_certificate_over_optimum"], 0.)
        cell = by_key[(2, 4096)]
        self.assertEqual(cell["certificate_median"], .28125)
        self.assertEqual(cell["certificate_mean"], .265625)
        self.assertEqual(cell["median_certificate_over_optimum"], .6)
        self.assertEqual(cell["zero_certificate_count"], 1)
        self.assertEqual(cell["strict_coverage_count"], 3)
        self.assertEqual(cell["gap_to_optimum_minimum"], -1/32)
        self.assertEqual(cell["sampling_gap_minimum"], -3/32)
        self.assertEqual(cell["population_bound_slack"], 1/16)

    def test_exclusion_power_false_exclusion_and_boundary_are_distinguished(self):
        _, rows = self.aggregate()
        by_key = {(r["case_key"], r["n_pairs"], r["tolerance"]): r for r in rows}

        false = by_key[(3, 4096, .5)]
        self.assertEqual(false["ground_truth_relation"], "boundary")
        self.assertEqual(false["interpretation"], "boundary_exclusion_rate")
        self.assertFalse(false["raw_float_exact_error_exceeds_tolerance"])
        self.assertEqual(false["raw_saved_erroneous_exclusion_count"], 1)
        self.assertIsNone(false["scientific_false_exclusion_count"])
        self.assertEqual(false["rate"], .25)

        power = by_key[(2, 4096, .3)]
        self.assertEqual(power["ground_truth_relation"], "above")
        self.assertEqual(power["interpretation"], "exclusion_power")
        self.assertTrue(power["raw_float_exact_error_exceeds_tolerance"])
        self.assertEqual(power["raw_saved_erroneous_exclusion_count"], 0)
        self.assertIsNone(power["scientific_false_exclusion_count"])
        self.assertEqual((power["count"], power["rate"]), (2, .5))

        boundary = by_key[(1, 4096, .3)]
        self.assertEqual(boundary["ground_truth_relation"], "boundary")
        self.assertEqual(boundary["interpretation"], "boundary_exclusion_rate")
        self.assertEqual(boundary["optimum_minus_tolerance"], 0.)
        self.assertEqual(boundary["comparison_atol"], 1e-12)
        self.assertFalse(boundary["raw_float_exact_error_exceeds_tolerance"])
        self.assertIsNone(boundary["scientific_false_exclusion_count"])

        genuinely_false = by_key[(0, 4096, .3)]
        self.assertEqual(genuinely_false["ground_truth_relation"], "below")
        self.assertEqual(genuinely_false["interpretation"], "false_exclusion_rate")
        self.assertEqual(genuinely_false["scientific_false_exclusion_count"], 0)

    def test_threshold_relation_uses_reporting_tolerance_only(self):
        atol = 1e-12

        self.assertEqual(
            summary.threshold_relation(.30000000000000004, .3, atol),
            "boundary",
        )
        self.assertEqual(
            summary.threshold_relation(.3, .30000000000000004, atol),
            "boundary",
        )
        self.assertEqual(
            summary.threshold_relation(.300000000002, .3, atol),
            "above",
        )
        self.assertEqual(
            summary.threshold_relation(.299999999998, .3, atol),
            "below",
        )
        self.assertEqual(
            summary.threshold_relation(.3, .3, 0.),
            "boundary",
        )

        for values in (
            (math.nan, .3, atol),
            (.3, math.inf, atol),
            (.3, .3, -1e-12),
            (True, .3, atol),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summary.threshold_relation(*values)

    def test_order_invariance_and_inputs_are_unchanged(self):
        before = deepcopy((self.config, self.references, self.rows))
        expected = self.aggregate()
        config = deepcopy(self.config)
        config["cases"].reverse()
        config["sampling"]["sample_sizes"].reverse()
        actual = self.aggregate(config=config, rows=reversed(self.rows))
        self.assertEqual(actual, expected)
        self.assertEqual((self.config, self.references, self.rows), before)

    def test_missing_duplicate_or_unexpected_replicates_are_rejected(self):
        cases = [self.rows[1:], self.rows+self.rows[:1], self.rows[4:]]
        for field, value in (("attempt_id", "unknown"), ("case_id", "missing_case")):
            changed = deepcopy(self.rows)
            changed[0][field] = value
            cases.append(changed)
        changed = deepcopy(self.rows)
        changed[0]["certificate"]["n_pairs"] = 999
        cases.append(changed)
        for rows in cases:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                self.aggregate(rows=rows)

    def test_invalid_configurations_and_references_are_rejected(self):
        cases = []
        for field, value in (("campaign_role", "final"),):
            config = deepcopy(self.config)
            config[field] = value
            cases.append(config)
        for field, value in (("quantile_method", "nearest"), ("certificate_quantiles", [.25, .5, .75]),
                             ("binomial_interval_method", "wilson")):
            config = deepcopy(self.config)
            config["evaluation"][field] = value
            cases.append(config)
        for value in (0, True, 4.):
            config = deepcopy(self.config)
            config["sampling"]["replicates_per_cell"] = value
            cases.append(config)
        config = deepcopy(self.config)
        config["sampling"]["sample_sizes"] = [128, 128]
        cases.append(config)
        config = deepcopy(self.config)
        config["cases"].append(config["cases"][0])
        cases.append(config)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.aggregate(config=config)
        with self.assertRaises(ValueError):
            self.aggregate(references={})

    def test_inconsistent_decisions_and_nonboolean_flags_are_rejected(self):
        cases = []
        for field, value in (("tolerance", 99.), ("excluded", 1),
                             ("exact_error_exceeds_tolerance", True), ("erroneous_exclusion", True)):
            rows = deepcopy(self.rows)
            rows[0]["exclusions"][0][field] = value
            cases.append(rows)
        rows = deepcopy(self.rows)
        rows[0]["exclusions"] = []
        cases.append(rows)
        rows = deepcopy(self.rows)
        rows[0]["strict_coverage"] = 1
        cases.append(rows)
        for rows in cases:
            with self.subTest(first=rows[0]), self.assertRaises(ValueError):
                self.aggregate(rows=rows)

    def test_csv_and_json_writers_are_exclusive_and_preserve_blanks(self):
        csv_path = self.base/"table.csv"
        summary.write_csv(csv_path, [dict(name="test,quoted", ratio=None, value=-.25)])
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows, [dict(name="test,quoted", ratio="", value="-0.25")])
        original = csv_path.read_bytes()
        with self.assertRaises(FileExistsError):
            summary.write_csv(csv_path, [dict(other=1)])
        self.assertEqual(csv_path.read_bytes(), original)
        for rows in ([], [dict(a=1), dict(b=2)]):
            with self.assertRaises(ValueError):
                summary.write_csv(self.base/"invalid.csv", rows)
        json_path = self.base/"record.json"
        summary.write_json(json_path, {"a": 1})
        with self.assertRaises(FileExistsError):
            summary.write_json(json_path, {"a": 2})
        with self.assertRaises(ValueError):
            summary.write_json(self.base/"nonfinite.json", {"a": math.nan})

    def test_source_provenance_checks_committed_bytes_and_versions(self):
        committed, git = self.source_fixture()
        with patch.object(summary, "ROOT", self.source), \
             patch.object(summary.subprocess, "check_output", side_effect=git):
            result = summary.source_provenance()
        self.assertEqual(result["summary_source_commit"], "b"*40)
        self.assertEqual(result["source_sha256"],
                         {name: hashlib.sha256(data).hexdigest() for name, data in committed.items()})
        self.assertEqual(result["package_versions"],
                         dict(numpy=summary.np.__version__, scipy=summary.scipy.__version__))

    def test_dirty_missing_symlinked_or_modified_source_is_rejected(self):
        committed, git = self.source_fixture()
        with patch.object(summary, "ROOT", self.source), \
             patch.object(summary.subprocess, "check_output", return_value=b"?? new.py\n"), \
             self.assertRaises(ValueError):
            summary.source_provenance()
        path = self.source/"scripts/summarize_four_state_pilot.py"
        for kind in ("missing", "symlink", "modified"):
            path.unlink(missing_ok=True)
            if kind == "symlink":
                target = self.base/"source-copy.py"
                target.write_bytes(committed["scripts/summarize_four_state_pilot.py"])
                path.symlink_to(target)
            elif kind == "modified":
                path.write_bytes(b"uncommitted bytes\n")
            with self.subTest(kind=kind), patch.object(summary, "ROOT", self.source), \
                 patch.object(summary.subprocess, "check_output", side_effect=git), \
                 self.assertRaises(ValueError):
                summary.source_provenance()

    def test_dependency_mismatch_is_rejected_without_installing_packages(self):
        _, git = self.source_fixture(wrong_versions=True)
        with patch.object(summary, "ROOT", self.source), \
             patch.object(summary.subprocess, "check_output", side_effect=git), \
             self.assertRaisesRegex(ValueError, "differ from the reference lockfile"):
            summary.source_provenance()

    def test_complete_tiny_summary_is_reproducible_and_checksummed(self):
        before = {p.name: p.read_bytes() for p in self.evidence.iterdir()}
        outputs = [self.base/"summary-a", self.base/"summary-b"]
        for output in outputs:
            report = self.run_tiny(output)
            self.assertEqual((report["cell_rows"], report["exclusion_rows"], report["summarized_replicates"]), (8, 24, 32))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["execution_source_commit"], "a"*40)
            self.assertEqual(report["source"]["summary_source_commit"], "b"*40)
            self.assertEqual(report["random_samples_generated"], 0)
            self.assertIs(report["evidence_modified"], False)
            entries = dict(line.split("  ", 1)[::-1] for line in (output/"SHA256SUMS").read_text().splitlines())
            self.assertEqual(set(entries), {"cell_summary.csv", "exclusion_summary.csv", "numerical_audit.json",
                                            "README.md", "summary_metadata.json"})
            for name, digest in entries.items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads((output/"summary_metadata.json").read_text()), report)
            with (output/"cell_summary.csv").open(newline="") as handle:
                cells = list(csv.DictReader(handle))
            self.assertEqual(len(cells), 8)
            self.assertEqual(cells[0]["median_certificate_over_optimum"], "")
            self.assertIn("pointwise", (output/"README.md").read_text())
        for name in ("cell_summary.csv", "exclusion_summary.csv", "numerical_audit.json"):
            self.assertEqual((outputs[0]/name).read_bytes(), (outputs[1]/name).read_bytes())
        self.assertEqual({p.name: p.read_bytes() for p in self.evidence.iterdir()}, before)
        self.assertEqual(self.api.verify_evidence.call_count, 2)

    def test_output_guards_reject_reuse_and_nested_locations(self):
        existing = self.base/"existing"
        existing.mkdir()
        (existing/"keep.txt").write_text("keep")
        for path in (existing, self.source, self.source/"nested", self.evidence,
                     self.evidence/"nested", self.base):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_tiny(path)
        self.api.verify_evidence.assert_not_called()
        self.assertEqual((existing/"keep.txt").read_text(), "keep")
        with patch.object(summary, "source_provenance") as source, self.assertRaises(ValueError):
            summary.summarize_evidence(self.base/"missing", self.base/"out")
        source.assert_not_called()

    def test_audit_failure_aborts_before_output_creation(self):
        self.api.verify_evidence.side_effect = ValueError("injected numerical audit failure")
        output = self.base/"failed-audit"
        with self.assertRaisesRegex(ValueError, "audit failure"):
            self.run_tiny(output)
        self.assertFalse(output.exists())
        self.api.verify_evidence.assert_called_once_with(self.evidence)

    def test_audit_replicate_mismatch_aborts_before_output_creation(self):
        self.audit["verified_replicates"] = 31
        output = self.base/"wrong-count"
        with self.assertRaisesRegex(ValueError, "accounting differs"):
            self.run_tiny(output)
        self.assertFalse(output.exists())

    def test_source_or_inventory_change_during_aggregation_aborts(self):
        changed = dict(self.provenance, summary_source_commit="c"*40)
        output = self.base/"changed-source"
        with self.assertRaisesRegex(ValueError, "Source changed during aggregation"):
            self.run_tiny(output, [self.provenance, changed])
        self.assertFalse(output.exists())
        output = self.base/"changed-evidence"
        self.api.check_inventory.return_value = ("9"*64, 3)
        with self.assertRaisesRegex(ValueError, "Evidence changed during aggregation"):
            self.run_tiny(output)
        self.assertFalse(output.exists())

    def test_late_source_or_inventory_change_prevents_finalization(self):
        changed = dict(self.provenance, summary_source_commit="c"*40)
        for kind in ("source", "evidence"):
            output = self.base/("late-"+kind)
            if kind == "evidence":
                self.api.check_inventory.side_effect = [("1"*64, 3), ("9"*64, 3)]
            source = [self.provenance, self.provenance, changed] if kind == "source" else None
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_tiny(output, source)
            self.assertTrue((output/"cell_summary.csv").exists())
            self.assertFalse((output/"summary_metadata.json").exists())
            self.assertFalse((output/"SHA256SUMS").exists())

    def test_write_and_finalization_failures_preserve_partial_output(self):
        original = summary.write_csv
        def csv_failure(path, rows):
            if path.name == "exclusion_summary.csv":
                raise OSError("injected CSV failure")
            return original(path, rows)
        output = self.base/"write-failure"
        with patch.object(summary, "write_csv", side_effect=csv_failure), self.assertRaises(OSError):
            self.run_tiny(output)
        self.assertTrue((output/"cell_summary.csv").exists())
        self.assertFalse((output/"summary_metadata.json").exists())
        output = self.base/"checksum-failure"
        with patch.object(summary, "file_hash", side_effect=OSError("injected checksum failure")), \
             self.assertRaises(OSError):
            self.run_tiny(output)
        self.assertTrue((output/"summary_metadata.json").exists())
        self.assertEqual((output/"SHA256SUMS").read_bytes(), b"")
        # Metadata without a valid, complete inventory is not a finalized package.

    def test_cli_reports_success_and_failures_without_retry(self):
        argv = [str(SCRIPT), "--evidence", str(self.evidence), "--output", str(self.base/"cli")]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), \
             patch.object(summary, "summarize_evidence", return_value={"status": "completed"}) as run, \
             redirect_stdout(output):
            self.assertEqual(summary.main(), 0)
        run.assert_called_once_with(self.evidence, self.base/"cli")
        self.assertEqual(json.loads(output.getvalue()), {"status": "completed"})
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", argv), \
             patch.object(summary, "summarize_evidence", side_effect=OSError("injected failure")) as run, \
             redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(summary.main(), 2)
        run.assert_called_once()
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Retain any created output directory", errors.getvalue())

    def test_module_import_does_not_read_evidence_or_generate_samples(self):
        with ExitStack() as stack:
            for name in ("read_text", "read_bytes"):
                stack.enter_context(patch.object(Path, name, side_effect=AssertionError("Unexpected evidence read")))
            stack.enter_context(patch.object(summary.subprocess, "check_output",
                                             side_effect=AssertionError("Unexpected Git command")))
            for name in ("SeedSequence", "PCG64", "Generator", "default_rng", "multinomial", "seed"):
                stack.enter_context(patch.object(summary.np.random, name,
                                                 side_effect=AssertionError("Unexpected RNG use")))
            loaded = runpy.run_path(str(SCRIPT), run_name="_summary_import_test_")
        self.assertEqual(loaded["SUMMARY_ID"], summary.SUMMARY_ID)


if __name__ == "__main__":
    unittest.main()
