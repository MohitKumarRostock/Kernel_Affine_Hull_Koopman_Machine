"""Archive helpers must preserve deterministic bytes, safe paths, and attribution."""
from pathlib import Path
import io
import json
import tempfile
import unittest
import zipfile

from scripts import package_tmlr_submission as package


class SubmissionPackageTests(unittest.TestCase):

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
