"""Local Git-fixture checks for isolated scientific contexts; no network use."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "scripts/materialize_scientific_context.py"
SPEC = importlib.util.spec_from_file_location("scientific_context", HELPER)
context = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context)


class ScientificContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.run_git("init", "--quiet", "--object-format=sha1")
        self.run_git("config", "user.name", "Fixture")
        self.run_git("config", "user.email", "fixture@example.invalid")
        self.original_documents = {}
        for number, name in enumerate(context.DOCUMENTS):
            raw = f"Bound scientific documentation {number}\r\n".encode()
            self.write(name, raw)
            self.original_documents[name] = raw
        self.write("science/model.py", b"value = 1\n")
        self.write("science/data.bin", b"\x00\xff\r\nfixed inputs\x00")
        self.write("science/protocol.json", b'{"fixed":true}\n')
        self.write("science/freeze.json", b'{"source":"fixed"}\n')
        self.write(".gitattributes", b"science/data.bin export-ignore\n")
        self.commit("Scientific inputs")
        self.source_commit = self.run_git("rev-parse", "HEAD").strip()
        self.expected_documents = {
            name: {"git_blob": context.blob_identity(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in self.original_documents.items()
        }
        for name in self.original_documents:
            self.write(name, b"Current documentation\n")
        self.write("science/model.py", b"value = 2\n")
        (self.repository / "science/model.py").chmod(0o755)
        self.commit("Current committed inputs")
        self.current_commit = self.run_git("rev-parse", "HEAD").strip()
        self.write("science/model.py", b"uncommitted worktree value\n")
        self.addCleanup(patch.stopall)
        patch.object(context, "SOURCE_COMMIT", self.source_commit).start()
        patch.object(context, "DOCUMENTS", self.expected_documents).start()

    def run_git(self, *arguments):
        return subprocess.check_output(
            ["git", "-C", str(self.repository), *arguments], stderr=subprocess.PIPE,
        ).decode()

    def write(self, name, raw):
        destination = self.repository / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)

    def commit(self, message):
        self.run_git("add", ".")
        self.run_git("commit", "--quiet", "-m", message)

    def test_exact_head_science_and_three_bound_documents(self):
        output = self.base / "context"
        before = {p.relative_to(self.repository): p.read_bytes()
                  for p in self.repository.rglob("*") if p.is_file() and ".git" not in p.parts}
        result = context.materialize(self.repository, output)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["current_commit"], self.current_commit)
        self.assertEqual(result["source_commit"], self.source_commit)
        self.assertEqual(len(result["documentation_inputs"]), 3)
        tracked = self.run_git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
        self.assertEqual(len(tracked), result["verified_files"])
        for name in tracked:
            expected = self.original_documents.get(name)
            if expected is None:
                expected = subprocess.check_output(
                    ["git", "-C", str(self.repository), "show", f"HEAD:{name}"])
            self.assertEqual((output / name).read_bytes(), expected, name)
        self.assertEqual((output / "science/model.py").stat().st_mode & 0o777, 0o755)
        self.assertTrue((output / "science/data.bin").is_file())
        for name, raw in before.items():
            self.assertEqual((self.repository / name).read_bytes(), raw)

    def test_refuses_existing_output_without_modification(self):
        output = self.base / "existing"
        output.mkdir()
        (output / "keep").write_bytes(b"keep")
        with self.assertRaisesRegex(context.ContextError, "already exists"):
            context.materialize(self.repository, output)
        self.assertEqual((output / "keep").read_bytes(), b"keep")

    def test_refuses_output_inside_repository(self):
        output = self.repository / "new_context"
        with self.assertRaisesRegex(context.ContextError, "outside"):
            context.materialize(self.repository, output)
        self.assertFalse(output.exists())

    def test_refuses_bad_document_identity_before_creating_output(self):
        name = next(iter(self.expected_documents))
        self.expected_documents[name] = {"git_blob": "0" * 40, "sha256": "0" * 64}
        output = self.base / "bad_identity"
        with self.assertRaisesRegex(context.ContextError, "identity mismatch"):
            context.materialize(self.repository, output)
        self.assertFalse(output.exists())

    def test_missing_source_commit_gives_explicit_fetch_command(self):
        output = self.base / "missing_source"
        with patch.object(context, "SOURCE_COMMIT", "0" * 40):
            with self.assertRaisesRegex(context.ContextError, "fetch origin " + "0" * 40):
                context.materialize(self.repository, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
