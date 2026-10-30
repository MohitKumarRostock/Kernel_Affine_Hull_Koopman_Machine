"""Small synthetic archive-integrity tests, separate from the 53 science tests."""
from __future__ import annotations

from pathlib import Path
import json
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from . import package


class PackageIntegrityTests(unittest.TestCase):
    def test_multipart_roundtrip_and_cleanup_only_of_owned_stale_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "checkout"
            run_directory = root / "synthetic_run"
            run_directory.mkdir()
            for name in ("initial_freeze.json", "confirmation_freeze.json"):
                (run_directory / name).write_text('{"code_sha256": {}}\n')
            evidence = bytes(range(256)) * (package.REFERENCE_PART_BYTES // 256) + b"last bytes"
            (run_directory / "evidence.npz").write_bytes(evidence)
            archive = repository / package.RELATIVE_EXTENSION / "reference/natural_v1.zip"
            manifest_path = package.reference_parts_path(archive)
            with patch.object(package, "verify_run", return_value={"status": "synthetic-pass"}), \
                    patch.object(package, "_verify_scientific_hashes"):
                built = package.build_reference(repository, run_directory, archive)
                self.assertEqual(len(built["git_parts"]), 2)
                specification = json.loads(manifest_path.read_text())
                self.assertEqual(specification["parts"][0]["bytes"], 8 * 1024**2)
                combined = b"".join((archive.parent / row["filename"]).read_bytes() for row in specification["parts"])
                self.assertEqual(combined, archive.read_bytes())
                archive.unlink()  # A fresh Git checkout contains only parts.
                verified = package.verify_reference_parts(manifest_path)
                self.assertEqual(verified["multipart_parts_verified"], 2)
                restored = package.restore_reference(archive, repository)
                self.assertEqual(restored["multipart_parts_verified"], 2)
                destination = repository / package.RELATIVE_EXTENSION / "runs/natural_v1/evidence.npz"
                self.assertEqual(destination.read_bytes(), evidence)
                self.assertFalse(archive.exists())  # Reconstruction is temporary.
                sentinel = archive.parent / "natural_v1.zip.part-999"
                sentinel.write_bytes(b"unowned file must survive")
                (run_directory / "evidence.npz").write_bytes(b"small replacement")
                rebuilt = package.build_reference(repository, run_directory, archive, overwrite=True)
                self.assertEqual(len(rebuilt["git_parts"]), 1)
                self.assertFalse((archive.parent / "natural_v1.zip.part-002").exists())
                self.assertEqual(sentinel.read_bytes(), b"unowned file must survive")
                self.assertEqual(package.verify_reference_parts(manifest_path)["multipart_parts_verified"], 1)

    def test_multipart_tampering_missing_parts_and_whole_digest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "natural_v1.zip"
            package.write_zip(archive, "natural_v1", {"evidence.txt": b"saved evidence"}, kind="reference-run")
            manifest_path, specification = package._write_reference_parts(archive, root)
            part = root / specification["parts"][0]["filename"]
            original = part.read_bytes()
            part.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaisesRegex(ValueError, "part payload changed"):
                package.verify_reference_parts(manifest_path)
            part.unlink()
            with self.assertRaises(FileNotFoundError):
                package.verify_reference_parts(manifest_path)
            part.write_bytes(original)
            specification["archive"]["sha256"] = "0" * 64
            manifest_path.write_bytes(package.json_bytes(specification))
            with self.assertRaisesRegex(ValueError, "Assembled reference archive differs"):
                package.verify_reference_parts(manifest_path)

    def test_multipart_unsafe_names_and_part_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "natural_v1.zip"
            package.write_zip(archive, "natural_v1", {"evidence.txt": b"saved evidence"}, kind="reference-run")
            manifest_path, specification = package._write_reference_parts(archive, root)
            for name in ("../outside", "/absolute", "sub/file", "C:/absolute", "root\\file", "natural_v1.zip.part-002"):
                changed = json.loads(json.dumps(specification))
                changed["parts"][0]["filename"] = name
                manifest_path.write_bytes(package.json_bytes(changed))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    package.verify_reference_parts(manifest_path)
            manifest_path.write_bytes(package.json_bytes(specification))
            part = root / specification["parts"][0]["filename"]
            outside = root / "outside"
            outside.write_bytes(part.read_bytes())
            part.unlink()
            part.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                package.verify_reference_parts(manifest_path)

    def test_report_comparison_ignores_only_explicitly_remeasured_timings(self):
        saved = {"policy": "metadata", "metrics": {"scaled_test_mse": 0.25,
            "total_policy_wall_seconds": 12.0, "decision_wall_seconds": 0.001},
            "chosen_candidates": ["raw", "kahm"], "measured_node_wall_seconds": 11.999}
        recomputed = {"policy": "metadata", "metrics": {"scaled_test_mse": 0.25,
            "total_policy_wall_seconds": 12.1, "decision_wall_seconds": 0.101},
            "chosen_candidates": ["raw", "kahm"], "measured_node_wall_seconds": 11.999}
        package.compare_content(saved, recomputed, ignored_fields=package.REMEASURED_FIELDS)
        recomputed["metrics"]["scaled_test_mse"] = 0.30
        with self.assertRaisesRegex(ValueError, "numeric value differs"):
            package.compare_content(saved, recomputed, ignored_fields=package.REMEASURED_FIELDS)
        recomputed["metrics"]["scaled_test_mse"] = 0.25
        recomputed["chosen_candidates"] = ["kahm", "kahm"]
        with self.assertRaisesRegex(ValueError, "value differs"):
            package.compare_content(saved, recomputed, ignored_fields=package.REMEASURED_FIELDS)

    def test_deterministic_archives_and_complete_payload_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.zip", root / "second.zip"
            files = {"nested/source.py": b"VALUE = 3\n", "data.json": b'{"n": 4}\n'}
            package.write_zip(first, "Example", files, kind="test")
            package.write_zip(second, "Example", files, kind="test")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            verified = package.verify_zip(first)
            self.assertEqual(verified["payload_files_verified"], 2)
            self.assertEqual(set(verified["manifest"]["files"]), set(files))
            self.assertIn("self_exclusion", " ".join(verified["manifest"]))
            # The reference artifact is a run-only archive and cannot include
            # itself. Restoration never overwrites an existing evidence tree.
            run_directory = root / "synthetic_run"
            run_directory.mkdir()
            for name in ("initial_freeze.json", "confirmation_freeze.json"):
                (run_directory / name).write_text('{"code_sha256": {}}\n')
            (run_directory / "evidence.txt").write_bytes(b"saved observations")
            repository = root / "checkout"
            reference = repository / package.RELATIVE_EXTENSION / "reference/natural_v1.zip"
            with patch.object(package, "verify_run", return_value={"status": "synthetic-pass"}), \
                    patch.object(package, "_verify_scientific_hashes"):
                with self.assertRaisesRegex(ValueError, "inside the run"):
                    package.build_reference(repository, run_directory, run_directory / "recursive.zip")
                paper = run_directory / package.PDF_NAME
                paper.write_bytes(b"%PDF-synthetic")
                with self.assertRaisesRegex(ValueError, "manuscript PDF belongs in the release"):
                    package.build_reference(repository, run_directory, reference)
                paper.unlink()
                package.build_reference(repository, run_directory, reference)
                self.assertEqual(package.verify_zip(reference)["kind"], "reference-run")
                restored = package.restore_reference(reference, repository)
                self.assertEqual(restored["status"], "restored")
                destination = repository / package.RELATIVE_EXTENSION / "runs/natural_v1"
                self.assertEqual((destination / "evidence.txt").read_bytes(), b"saved observations")
                self.assertFalse((destination / package.MANIFEST).exists())
                with self.assertRaisesRegex(FileExistsError, "never overwrites"):
                    package.restore_reference(reference, repository)

    def test_payload_mutation_is_detected_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "example.zip"
            package.write_zip(archive_path, "Example", {"evidence.txt": b"original"}, kind="test")
            with zipfile.ZipFile(archive_path) as archive:
                payloads = {item.filename: archive.read(item) for item in archive.infolist()}
            payloads["Example/evidence.txt"] = b"modified"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, content in payloads.items():
                    archive.writestr(name, content)
            with self.assertRaisesRegex(ValueError, "payload changed"):
                package.verify_zip(archive_path)

    def test_unlisted_and_duplicate_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "example.zip"
            package.write_zip(archive_path, "Example", {"evidence.txt": b"original"}, kind="test")
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr("Example/unlisted.txt", b"extra")
            with self.assertRaisesRegex(ValueError, "inventory"):
                package.verify_zip(archive_path)
            package.write_zip(archive_path, "Example", {"evidence.txt": b"original"}, kind="test")
            with zipfile.ZipFile(archive_path, "a") as archive:
                with self.assertWarns(UserWarning):
                    archive.writestr("Example/evidence.txt", b"duplicate")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                package.verify_zip(archive_path)

    def test_traversal_absolute_paths_and_symlink_zip_entries_are_rejected(self):
        for name in ("../outside", "/absolute", "root/../../outside", "root//empty", "C:/absolute", "root\\file"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.safe_name(name)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.zip"
            package.write_zip(archive_path, "Example", {"evidence.txt": b"original"}, kind="test")
            info = zipfile.ZipInfo("Example/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr(info, "../../outside")
            with self.assertRaisesRegex(ValueError, "nonregular"):
                package.verify_zip(archive_path)

    def test_extracted_tree_rejects_added_files_and_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "example.zip"
            package.write_zip(archive_path, "Example", {"evidence.txt": b"original"}, kind="test")
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
            extracted = root / "Example"
            self.assertEqual(package.verify_directory(extracted)["status"], "passed")
            extra = extracted / "new.txt"
            extra.write_text("not in manifest")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                package.verify_directory(extracted)
            extra.unlink()
            outside = root / "outside"
            outside.mkdir()
            (extracted / "linked_directory").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                package.verify_directory(extracted)

    def test_repository_view_extracts_only_verified_reproducibility_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "source.zip"
            package.write_zip(archive_path, "Example", {"source.txt": b"unchanged"}, kind="manuscript-source")
            with self.assertRaisesRegex(ValueError, "reproducibility"):
                with package.repository_view(archive_path):
                    self.fail("A manuscript archive must not be executed as a source checkout")
            package.write_zip(archive_path, "Example", {"source.txt": b"unchanged"}, kind="reproducibility")
            before = package.file_digest(archive_path)
            with package.repository_view(archive_path) as extracted:
                self.assertEqual((extracted / "source.txt").read_bytes(), b"unchanged")
                temporary_path = extracted
            self.assertFalse(temporary_path.exists())
            self.assertEqual(before, package.file_digest(archive_path))


if __name__ == "__main__":
    unittest.main()
