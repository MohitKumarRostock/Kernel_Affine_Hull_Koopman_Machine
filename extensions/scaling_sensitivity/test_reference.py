"""Integrity and non-overwrite checks for the compact reference transport."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
import warnings
import zipfile

from . import reference as ref


class ReferenceIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory(prefix="scaling-reference-tests-")
        cls.base = Path(cls.storage.name)
        cls.root = cls.base / "source"
        cls.root.mkdir()
        (cls.root / "analysis.py").write_text("# Integrity-test fixture, not a scientific analysis.\n")
        sources = {"natural": {}, "controlled": {}}
        for branch, count in (("natural", 208), ("controlled", 72)):
            for index in range(count):
                names = ([f"results/natural/task{index:03d}.json", f"results/natural/task{index:03d}.npz"]
                         if branch == "natural" else
                         [f"results/controlled/jobs/job{index:03d}/result.json",
                          f"results/controlled/jobs/job{index:03d}/unit_losses.npz"])
                for name in names:
                    path = cls.root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(ref.json_bytes({"fixture": index, "name": name}))
                    sources[branch][name] = ref.digest(path)
        audit = {
            "status": "passed", "analysis_code_sha256": ref.digest(cls.root / "analysis.py"),
            "natural": {"tasks": 208, "predictions_checked": 1664, "source_files_sha256": sources["natural"]},
            "controlled": {"jobs": 72, "per_episode_loss_arrays_checked": 1440,
                           "full_prediction_arrays_checked": 1440,
                           "source_files_sha256": sources["controlled"]},
        }
        (cls.root / "analysis").mkdir()
        (cls.root / "analysis" / "AUDIT.json").write_bytes(ref.json_bytes(audit))
        cls.parts = cls.base / "reference"
        cls.built = ref.build(cls.root, cls.parts)
        with ref.assembled_reference(cls.parts / ref.PARTS_NAME) as (assembled, _):
            cls.zip_path = cls.base / ref.ARCHIVE_NAME
            shutil.copyfile(assembled, cls.zip_path)

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.base, prefix="case-")
        self.work = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _copy_parts(self):
        target = self.work / "reference"
        shutil.copytree(self.parts, target)
        return target

    def _mutated_zip(self, addition=None, replace_payload=False):
        target = self.work / "mutated.zip"
        with zipfile.ZipFile(self.zip_path) as source, zipfile.ZipFile(target, "w") as output:
            for item in source.infolist():
                value = source.read(item.filename)
                if replace_payload and item.filename == "results/natural/task000.json":
                    value = b"changed evidence"
                output.writestr(item, value)
            if addition is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    output.writestr(*addition)
        return target

    def test_deterministic_archive_and_exact_idempotent_restore(self):
        second = self.work / "second"
        self.assertEqual(ref.build(self.root, second), self.built)
        self.assertEqual((second / ref.PARTS_NAME).read_bytes(), (self.parts / ref.PARTS_NAME).read_bytes())
        restored = self.work / "restored"
        first = ref.restore(second / ref.PARTS_NAME, restored)
        self.assertEqual(first["new_files"], 560)
        self.assertEqual(ref.restore(second / ref.PARTS_NAME, restored)["new_files"], 0)
        for name, path in ref.payload_files(self.root).items():
            self.assertEqual((restored / name).read_bytes(), path.read_bytes())

    def test_corrupted_part_rejected_before_any_restore(self):
        parts = self._copy_parts()
        path = parts / (ref.ARCHIVE_NAME + ".part001")
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        destination = self.work / "restored"
        with self.assertRaises(ValueError):
            ref.restore(parts / ref.PARTS_NAME, destination)
        self.assertFalse(destination.exists())

    def test_wrong_whole_archive_hash_rejected(self):
        parts = self._copy_parts()
        path = parts / ref.PARTS_NAME
        manifest = json.loads(path.read_text())
        manifest["archive"]["sha256"] = "0" * 64
        path.write_bytes(ref.json_bytes(manifest))
        with self.assertRaises(ValueError):
            ref.verify(path)

    def test_multiple_parts_respect_four_mib_boundary_and_restore_exactly(self):
        source = self.work / "source"
        shutil.copytree(self.root, source)
        large = source / "results/natural/task000.npz"
        # NPZ payloads are stored in the transport ZIP, so this deterministically
        # crosses a part boundary without expensive random-data generation.
        large.write_bytes(b"x" * (ref.PART_BYTES + 17))
        audit_path = source / "analysis/AUDIT.json"
        audit = json.loads(audit_path.read_text())
        audit["natural"]["source_files_sha256"]["results/natural/task000.npz"] = ref.digest(large)
        audit_path.write_bytes(ref.json_bytes(audit))
        output = self.work / "parts"
        result = ref.build(source, output)
        self.assertEqual(result["parts"], 2)
        specification = json.loads((output / ref.PARTS_NAME).read_text())
        self.assertEqual(specification["parts"][0]["bytes"], ref.PART_BYTES)
        self.assertLessEqual(specification["parts"][1]["bytes"], ref.PART_BYTES)
        destination = self.work / "restored"
        ref.restore(output / ref.PARTS_NAME, destination)
        self.assertEqual(ref.digest(destination / "results/natural/task000.npz"), ref.digest(large))

    def test_unlisted_or_missing_transport_parts_rejected(self):
        parts = self._copy_parts()
        extra = parts / (ref.ARCHIVE_NAME + ".part999")
        extra.write_bytes(b"unlisted")
        with self.assertRaises(ValueError):
            ref.verify(parts / ref.PARTS_NAME)
        extra.unlink()
        (parts / (ref.ARCHIVE_NAME + ".part001")).unlink()
        with self.assertRaises(ValueError):
            ref.verify(parts / ref.PARTS_NAME)

    def test_duplicate_unsafe_symlink_and_unlisted_zip_entries_rejected(self):
        symlink = zipfile.ZipInfo("results/natural/link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        variants = [
            ("results/natural/task000.json", b"duplicate"),
            ("results/natural/../../escape", b"escape"),
            (symlink, b"target"),
            ("results/natural/unlisted.json", b"{}"),
        ]
        for addition in variants:
            with self.subTest(name=str(addition[0])):
                with self.assertRaises(ValueError):
                    ref.verify_zip(self._mutated_zip(addition=addition))

    def test_changed_payload_rejected_by_per_file_hash(self):
        with self.assertRaises(ValueError):
            ref.verify_zip(self._mutated_zip(replace_payload=True))

    def test_restore_preserves_changed_and_unlisted_existing_files(self):
        destination = self.work / "existing"
        path = destination / "results/natural/task000.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"retain my changed evidence")
        with self.assertRaises(FileExistsError):
            ref.restore(self.parts / ref.PARTS_NAME, destination)
        self.assertEqual(path.read_bytes(), b"retain my changed evidence")
        self.assertFalse((destination / "results/controlled").exists())
        path.unlink()
        unexpected = path.with_name("unexpected.json")
        unexpected.write_bytes(b"retain my additional evidence")
        with self.assertRaises(ValueError):
            ref.restore(self.parts / ref.PARTS_NAME, destination)
        self.assertEqual(unexpected.read_bytes(), b"retain my additional evidence")

    def test_source_symlink_and_stale_audit_rejected(self):
        source = self.work / "source"
        shutil.copytree(self.root, source)
        link = source / "results/natural/link.json"
        link.symlink_to(source / "results/natural/task000.json")
        with self.assertRaises(ValueError):
            ref.build(source, self.work / "out")
        link.unlink()
        (source / "results/natural/task000.json").write_bytes(b"changed after audit")
        with self.assertRaises(ValueError):
            ref.build(source, self.work / "out")

    def test_existing_different_reference_part_is_not_overwritten(self):
        output = self._copy_parts()
        path = output / (ref.ARCHIVE_NAME + ".part001")
        path.write_bytes(b"preserved old artifact")
        with self.assertRaises(FileExistsError):
            ref.build(self.root, output)
        self.assertEqual(path.read_bytes(), b"preserved old artifact")

    def test_duplicate_manifest_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            ref.read_json('{"files": {}, "files": {}}')


if __name__ == "__main__":
    unittest.main()
