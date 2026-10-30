"""Submission bundles must preserve evidence and expose any review transformation."""
from pathlib import Path
import io
import json
import tempfile
import unittest
import zipfile

from scripts import package_tmlr_submission as package


class SubmissionPackageTests(unittest.TestCase):
    def test_extracted_supplement_verifies_and_preserves_historical_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "zips"
            summary = package.build_packages(package.ROOT, output)
            self.assertLess(summary["combined_bytes"], 100_000_000)
            destination = Path(temporary) / "review"
            with zipfile.ZipFile(output / package.SUPPLEMENT_NAME) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertTrue(set(package.REQUIRED_RECORDS).issubset(names))
                for excluded in ("CITATION.cff", "manuscript/main.pdf", ".git/config"):
                    self.assertNotIn(excluded, names)
                self.assertFalse(any("__pycache__" in n or "__MACOSX" in n for n in names))
                self.assertNotIn(b"authors =", archive.read("pyproject.toml"))
                for name in names:
                    if name.startswith(("study/", "archive/")) or name == "LICENSE":
                        self.assertEqual(archive.read(name), (package.ROOT / name).read_bytes(), name)
                manifest = json.loads(archive.read(package.MANIFEST))
                metadata = manifest["files"]["pyproject.toml"]
                self.assertEqual(metadata["source_sha256"], package.sha256((package.ROOT / "pyproject.toml").read_bytes()))
                self.assertTrue(manifest["review_audit"]["retained_attribution_files"])
                archive.extractall(destination)
            self.assertEqual(package.verify_directory(destination)["status"], "verified")
            with zipfile.ZipFile(output / package.SOURCE_NAME) as archive:
                self.assertIn("main.tex", archive.namelist())
                self.assertNotIn("main.pdf", archive.namelist())
                self.assertNotIn("jmlr2e.sty", archive.namelist())
                for name in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty", "LICENSE_TMLR.txt"):
                    self.assertEqual(archive.read(name), (package.ROOT / "manuscript" / name).read_bytes())

    def test_zip_is_stable_under_input_order_and_has_fixed_metadata(self):
        files = {"z.txt": b"last", "a.txt": b"first"}
        first = package.zip_bytes(files)
        self.assertEqual(first, package.zip_bytes(dict(reversed(list(files.items())))))
        self.assertEqual(package.content_digest(files), package.content_digest(dict(reversed(list(files.items())))))
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            self.assertEqual(archive.namelist(), ["a.txt", "z.txt"])
            for entry in archive.infolist():
                self.assertEqual(entry.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(entry.external_attr >> 16, 0o100644)

    def test_missing_or_modified_payload_cannot_pass_manifest_verification(self):
        historical = package.verify_historical_inputs(package.ROOT)
        files = package.prepare_bundle(package.ROOT, "manuscript", historical)
        modified = dict(files)
        modified["main.tex"] += b"\nchanged\n"
        with self.assertRaises(ValueError):
            package.verify_bundle_files(modified)
        missing = dict(files)
        del missing["main.tex"]
        with self.assertRaises(ValueError):
            package.verify_bundle_files(missing)
        added = dict(files, unexpected=b"not listed")
        with self.assertRaises(ValueError):
            package.verify_bundle_files(added)

    def test_attribution_is_retained_but_overview_identity_is_rejected(self):
        credited = "Example Rights Holder"
        files = {"LICENSE": ("Copyright (c) 2026 " + credited).encode(), "README.md": b"Review artifact"}
        result = package.audit_review_view(files, [credited])
        self.assertEqual(result["retained_attribution_files"][0]["path"], "LICENSE")
        with self.assertRaises(ValueError):
            package.audit_review_view({"README.md": credited.encode()}, [credited])
        contact = "reviewer" + "@" + "example.org"
        with self.assertRaises(ValueError):
            package.audit_review_view({"README.md": contact.encode()})
        current = "https://" + "github.com/example/repo/tree/research/" + "kahm-predictive-geometry/path"
        with self.assertRaises(ValueError):
            package.audit_review_view({"manuscript/references.bib": current.encode()})

    def test_unsafe_paths_and_evidence_output_destinations_are_rejected(self):
        for name in ("../outside", "/absolute", "study/../../outside", "study\\outside"):
            with self.assertRaises(ValueError):
                package.safe_name(name)
        with self.assertRaises(ValueError):
            package.build_packages(package.ROOT, package.ROOT / "study" / "submission")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            outside = Path(temporary) / "outside"
            outside.write_bytes(b"outside")
            (root / "linked").symlink_to(outside)
            with self.assertRaises(ValueError):
                package.read_source(root, "linked")


if __name__ == "__main__":
    unittest.main()
