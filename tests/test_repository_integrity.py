"""Exercise repository verification against small, independently committed Git fixtures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / 'scripts/verify_publication_tree.py'
spec = importlib.util.spec_from_file_location('repository_verifier', SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class RepositoryIntegrityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Repository verification test')
        self.git('config', 'user.email', 'verification@example.invalid')
        self.write('README.md', '# Code and evidence\n')
        self.write('extensions/study/README.md', '# Protocol\n')
        self.write('extensions/study/protocol.json', '{"repetitions": 4}\n')
        self.write('reproduction/central/README.md', '# Saved losses\n')
        self.write('reproduction/central/analysis.py', 'def mean(x):\n    return sum(x) / len(x)\n')
        self.write('reproduction/central/evidence/losses.csv', 'repetition,loss\n0,1.0\n')
        self.write('reproduction/capacity/README.md', '# Experiment source\n')
        self.write('reproduction/capacity/study/model.py', 'SEED = 7\n')
        self.write('reproduction/capacity/archive/records.json', '[{"loss": 1.0}]\n')
        self.write('papers/study/main.tex', 'Reference document.\n')
        self.write('scripts/verify_publication_tree.py', SCRIPT.read_text())
        self.manifest = {
            'protocol': 'fixture-v1',
            'files': {
                'README.md': self.identity('reproduction/central/README.md'),
                'analysis.py': self.identity('reproduction/central/analysis.py'),
                'evidence/losses.csv': self.identity('reproduction/central/evidence/losses.csv'),
            },
        }
        self.write('reproduction/central/MANIFEST.json', json.dumps(self.manifest))
        self.commit()
        self.source = self.git('rev-parse', 'HEAD').strip()
        sources = {}
        for prefix in ('extensions/study', 'reproduction/central', 'reproduction/capacity'):
            sources[prefix] = {
                'commit': self.source,
                'tree': self.git('rev-parse', f'{self.source}:{prefix}').strip(),
                'reader_documentation': [prefix + '/README.md'],
            }
        sources['reproduction/central']['documentation_manifests'] = {
            'reproduction/central/MANIFEST.json': ['README.md']}
        self.plan = {
            'repository': 'owner/project',
            'scientific_sources': sources,
            'source_link_commits': [self.source],
            'excluded_document_roots': ['papers', 'manuscript'],
            'root_entries': ['README.md', 'docs', 'extensions', 'reproduction', 'scripts'],
            'study_directories': {'extensions': ['study'], 'reproduction': ['central', 'capacity']},
            'scope': 'Code, evidence and reader documentation.',
        }
        shutil.rmtree(self.root / 'papers')
        self.write('README.md', '[Protocol](extensions/study/protocol.json)\n'
                   f'[Source](https://github.com/owner/project/blob/{self.source}/reproduction/capacity/study/model.py)\n')
        self.write('docs/PUBLICATION_SCOPE.json', json.dumps(self.plan))
        self.write('reproduction/central/README.md', '# Analyze the saved per-run losses\n')
        self.refresh_manifest()
        self.commit()
        patcher = patch.object(verifier, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, stderr=subprocess.DEVNULL).decode()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def identity(self, name):
        data = (self.root / name).read_bytes()
        return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def refresh_manifest(self):
        path = self.root / 'reproduction/central/MANIFEST.json'
        manifest = json.loads(path.read_text())
        manifest['files']['README.md'] = self.identity('reproduction/central/README.md')
        path.write_text(json.dumps(manifest))

    def commit(self):
        self.git('add', '-A')
        self.git('commit', '-qm', 'Fixture state')

    def verify(self):
        return verifier.verify_repository(self.plan)

    def test_repository_without_documents_passes(self):
        self.assertFalse((self.root / 'papers').exists())
        report = self.verify()
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['scientific_reference_files_checked'], 5)
        self.assertEqual(report['documentation_manifest_entries_checked'], 1)
        self.assertEqual(report['local_links_checked'], 1)
        self.assertEqual(report['source_paths_checked'], 1)
        self.assertFalse(report['model_fitting_performed'])

    def test_changed_scientific_source_rejected(self):
        self.write('reproduction/capacity/study/model.py', 'SEED = 8\n')
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Scientific file identity differs'):
            self.verify()

    def test_missing_evidence_rejected(self):
        (self.root / 'reproduction/central/evidence/losses.csv').unlink()
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Scientific file set differs'):
            self.verify()

    def test_added_scientific_file_rejected(self):
        self.write('extensions/study/extra.json', '{}\n')
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Scientific file set differs'):
            self.verify()

    def test_scientific_mode_change_rejected(self):
        (self.root / 'reproduction/central/analysis.py').chmod(0o755)
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Scientific file identity differs'):
            self.verify()

    def test_updated_documentation_with_matching_manifest_passes(self):
        self.write('reproduction/central/README.md', '# Instructions\n[Losses](evidence/losses.csv)\n')
        self.refresh_manifest()
        self.commit()
        self.assertEqual(self.verify()['status'], 'passed')

    def test_stale_readme_manifest_rejected(self):
        self.write('reproduction/central/README.md', '# Different instructions\n')
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Manifest differs'):
            self.verify()

    def test_other_manifest_field_change_rejected(self):
        path = self.root / 'reproduction/central/MANIFEST.json'
        manifest = json.loads(path.read_text())
        manifest['protocol'] = 'different-protocol'
        path.write_text(json.dumps(manifest))
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Manifest differs'):
            self.verify()

    def test_untracked_document_does_not_satisfy_local_link(self):
        self.write('README.md', '[Document](papers/study/main.tex)\n')
        self.commit()
        self.write('papers/study/main.tex', 'Untracked placeholder.\n')
        with self.assertRaisesRegex(AssertionError, 'Broken or untracked local link'):
            self.verify()

    def test_pinned_document_link_rejected(self):
        self.write('README.md', f'[Document](https://github.com/owner/project/blob/{self.source}/papers/study/main.tex)\n')
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'outside the code and evidence scope'):
            self.verify()

    def test_unknown_pinned_source_path_rejected(self):
        self.write('README.md', f'[Missing](https://github.com/owner/project/blob/{self.source}/absent.py)\n')
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Unknown source path'):
            self.verify()

    def test_undeclared_commit_rejected(self):
        self.write('README.md', '[Unknown](https://github.com/owner/project/tree/' + '0' * 40 + '/extensions/study)\n')
        self.commit()
        with self.assertRaisesRegex(AssertionError, 'Undeclared source commit'):
            self.verify()

    def test_uncommitted_working_change_rejected(self):
        self.write('README.md', '# Uncommitted content\n')
        with self.assertRaisesRegex(AssertionError, 'Working file differs'):
            self.verify()

    def test_wrong_reference_subtree_rejected(self):
        self.plan['scientific_sources']['extensions/study']['tree'] = '0' * 40
        with self.assertRaisesRegex(AssertionError, 'Source subtree identity differs'):
            self.verify()

    def test_cli_report_and_existing_output_protection(self):
        destination = self.root.parent / (self.root.name + '-report')
        self.addCleanup(lambda: shutil.rmtree(destination, ignore_errors=True))
        command = ['python3', '-S', str(self.root / 'scripts/verify_publication_tree.py'), '--output-dir', str(destination)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        report = json.loads((destination / 'verification.json').read_text())
        self.assertEqual(report['status'], 'passed')
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('existing reports are not overwritten', second.stderr)


if __name__ == '__main__':
    unittest.main()
