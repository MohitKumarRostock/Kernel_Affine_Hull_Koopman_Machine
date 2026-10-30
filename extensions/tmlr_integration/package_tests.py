"""Integrity regressions for the integrated publication export, without fits."""
from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

from . import package


class ReviewIntegrityTests(unittest.TestCase):
    def test_review_payload_cannot_omit_or_change_a_frozen_scaling_dependency(self):
        name = "extensions/natural_forecasting/package_tests.py"
        raw = b"Unchanged execution dependency\n"
        frozen = {name: package.sha256(raw)}
        self.assertEqual(package.verify_scaling_payload({name: raw}, frozen), 1)
        with self.assertRaisesRegex(ValueError, "omitted"):
            package.verify_scaling_payload({}, frozen)
        with self.assertRaisesRegex(ValueError, "changed"):
            package.verify_scaling_payload({name: b"different"}, frozen)

    def test_numeric_replay_allows_roundoff_but_rejects_changed_evidence(self):
        saved = {"tasks": 208, "means": [0.842191252179071], "phase": "post_hoc"}
        package.compare_numeric_replay(saved, {**saved, "means": [0.8421912521790711]})
        for changed in ({**saved, "tasks": 207}, {**saved, "means": [0.8422]},
                        {**saved, "means": []}, {**saved, "extra": True},
                        {**saved, "tasks": 208.0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                package.compare_numeric_replay(saved, changed)

    def test_csv_replay_keeps_schema_and_counts_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.csv"
            path.write_text("phase,tasks,loss\npost_hoc,208,0.842191252179071\n")
            parsed = package.read_numeric_csv(path)
            self.assertEqual(parsed["columns"], ["phase", "tasks", "loss"])
            self.assertEqual(parsed["rows"], [["post_hoc", 208, 0.842191252179071]])
            self.assertIs(type(parsed["rows"][0][1]), int)

    def bundle(self):
        return package.with_manifest({"README.txt": b"Anonymous review artifact\n", "science/result.json": b'{"loss": 1}\n'},
                                     "manuscript-source", {"post_study_export": True})

    def test_changed_scientific_bytes_rejected(self):
        files = self.bundle()
        files["science/result.json"] = b'{"loss": 0}\n'
        with self.assertRaisesRegex(ValueError, "changed"):
            package.verify_files(files)

    def test_unlisted_and_missing_payloads_rejected(self):
        original = self.bundle()
        for files in ({**original, "unexpected.txt": b"extra"}, {k: v for k, v in original.items() if k != "science/result.json"}):
            with self.assertRaisesRegex(ValueError, "Missing or unlisted"):
                package.verify_files(files)

    def test_manifest_digest_is_checked(self):
        files = self.bundle()
        manifest = json.loads(files[package.MANIFEST])
        manifest["content_sha256"] = "0" * 64
        files[package.MANIFEST] = package.json_bytes(manifest)
        with self.assertRaisesRegex(ValueError, "content-tree"):
            package.verify_files(files)

    def test_unsafe_paths_rejected(self):
        for name in ("../outside", "/absolute", "a/../../outside", "a\\b", "a//b", "C:/outside", "a/./b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.safe_name(name)

    def test_zip_duplicate_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, "outside")
            with self.assertRaisesRegex(ValueError, "symbolic"):
                package.read_bundle(path)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../outside", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                package.read_bundle(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("duplicate.txt", "first")
                    archive.writestr("duplicate.txt", "second")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                package.read_bundle(path)

    def test_zip_is_deterministic_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = [Path(temporary) / name for name in ("first.zip", "second.zip")]
            payload = self.bundle()
            package.write_zip(first, payload)
            package.write_zip(second, dict(reversed(list(payload.items()))))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(package.read_bundle(first), payload)
            self.assertEqual(package.verify(first)["payload_files_verified"], 2)

    def test_current_study_links_rejected_even_in_bibliography(self):
        link = "https://github.com/" + "example/research-project/tree/research/revision"
        with self.assertRaisesRegex(ValueError, "Current-study"):
            package.audit_anonymity({"references.bib": link.encode()})

    def test_identity_outside_attribution_rejected(self):
        identity = "Example" + " Person"
        with self.assertRaisesRegex(ValueError, "identity"):
            package.audit_anonymity({"README.md": identity.encode()}, [identity])
        result = package.audit_anonymity({"LICENSE": ("Copyright " + identity).encode()}, [identity])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["retained_attribution"]), 1)

    def test_in_place_bibtex_output_does_not_change_source_export(self):
        identity = "Example" + " Person"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manuscript = root / package.RELATIVE / "manuscript"
            manuscript.mkdir(parents=True)
            inputs = {
                "main.tex": "\\documentclass{article}\n\\usepackage{tmlr}\n",
                "references.bib": "@article{prior, author = {" + identity + "}, title = {Prior work}}\n",
                "tmlr.sty": "% Test style input\n",
                "tmlr.bst": "% Test BibTeX style input\n",
                "fancyhdr.sty": "% Test header input\n",
                "LICENSE_TMLR.txt": "Test fixture license\n",
            }
            for name, text in inputs.items():
                (manuscript / name).write_text(text)
            before = package.manuscript_payload(root)
            bibliography_output = ("\\bibitem{prior} " + identity + ". Prior work.\n").encode()
            (manuscript / "main.bbl").write_bytes(bibliography_output)
            after = package.manuscript_payload(root)
            self.assertEqual(after, before)
            self.assertIn("references.bib", after)
            self.assertIn("tmlr.bst", after)
            exported = package.with_manifest(after, "manuscript-source", {}, [identity])
            package.verify_files(exported)
            # Generated bibliography output is excluded, not granted an
            # attribution exemption that could hide arbitrary author metadata.
            with self.assertRaisesRegex(ValueError, "identity"):
                package.audit_anonymity({"main.bbl": bibliography_output}, [identity])

    def test_exact_inventory_prevents_hidden_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, raw in self.bundle().items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            (root / ".identity").write_text("unlisted metadata")
            with self.assertRaisesRegex(ValueError, "Missing or unlisted"):
                package.verify(root)

    def test_manuscript_source_cannot_be_scientific_audit_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.zip"
            package.write_zip(path, self.bundle())
            with self.assertRaisesRegex(ValueError, "requires the review supplement"):
                with package.repository_view(path):
                    self.fail("Source archive incorrectly accepted as scientific evidence")

    def test_review_subset_cannot_recertify_original_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / package.NATURAL / "CHECKPOINT.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({"format": "post_study_anonymous_review_inventory_v1"}))
            with self.assertRaisesRegex(ValueError, "cannot certify"):
                package.supplement_payload(root, root / "unused-run")

    def test_replay_refuses_source_tree_and_nonempty_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                package.audit(root, root / "new-output")
            output = Path(temporary) / "existing-output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "empty"):
                package.audit(root, output)
            self.assertEqual(marker.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
