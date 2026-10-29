"""Matched head/scaler parity, legacy numerical identity, and immutable study guards."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from certified_forecasting.data import histories_targets, make_episodes
import run_forecasting as forecasting
import run_matched_geometry as study

PUBLISHED_COMMIT = "71efa873419e2b58ee89746dafd123967ff007b1"


class MatchedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT / "configs/main_v1.json").read_text())
        cls.cfg.update(max_geometry_inputs=100, max_train_per_cluster=10)
        cls.train = make_episodes("harmonic", 20, 40, 915701)
        cls.validation = make_episodes("harmonic", 12, 40, 915702)
        cls.queries = {
            history: forecasting.flat(histories_targets(cls.validation, history,
                cls.cfg["horizons"], cls.cfg["origin_stride"], max(cls.cfg["histories"]) - 1)[0])
            for history in cls.cfg["histories"]}

    def test_default_candidates_match_published_source_exactly(self):
        relative = "extensions/certified_forecasting/experiments/run_forecasting.py"
        try:
            source = subprocess.check_output(["git", "show", PUBLISHED_COMMIT + ":" + relative],
                cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("Published reference commit is unavailable in this shallow/source-only checkout")
        module = types.ModuleType("published_forecasting_reference")
        module.__file__ = str(ROOT / "experiments/run_forecasting.py")
        sys.modules[module.__name__] = module
        try:
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            expected, old_pairs = module.build_candidates(self.train, self.validation, self.cfg, 915703)
            actual, pairs = forecasting.build_candidates(self.train, self.validation, self.cfg, 915703)
            self.assertEqual(len(actual), 40)
            self.assertEqual(pairs, old_pairs)
            for old, new in zip(expected, actual):
                old_description, description = module.candidate_description(old), forecasting.candidate_description(new)
                old_description.pop("train_seconds")
                description.pop("train_seconds")
                self.assertEqual(description, old_description)
                query = self.queries[new.history]
                np.testing.assert_array_equal(new.predict(new.transform(query), self.cfg["horizons"]),
                                              old.predict(old.transform(query), self.cfg["horizons"]))
        finally:
            sys.modules.pop(module.__name__, None)

    def test_every_geometry_head_prefix_and_scaler_matches_shared_bank(self):
        for mode in study.MODES:
            with self.subTest(mode=mode):
                cfg = {**self.cfg, "input_scaling_by_kind": {kind: mode for kind in study.KINDS}}
                legacy, _ = forecasting.build_candidates(self.train, self.validation, cfg, 915703)
                actual, pairs = forecasting.build_candidates(self.train, self.validation,
                    {**cfg, "matched_geometry_heads": True}, 915703)
                self.assertEqual(len(actual), 84)
                self.assertEqual(len(pairs), 18)
                by_name = {candidate.name: candidate for candidate in actual}
                self.assertEqual(len(by_name), 84)
                for history in self.cfg["histories"]:
                    scalers, subsamples = [], []
                    for kind in study.KINDS:
                        charged = [c for c in actual if getattr(c, "geometry_bank_id", None) == f"{kind}_m{history}"]
                        self.assertEqual(sum(c.geometry_train_seconds > 0 for c in charged), 1)
                        self.assertTrue(all(c.train_seconds >= 0 for c in charged))
                        for stage, dimension in enumerate((2, 6, 14, 30), 1):
                            heads = [by_name[f"{kind}_{head}_m{history}_s{stage}"]
                                     for head in ("nested", "frobenius", "koopman")]
                            self.assertEqual([c.n_features for c in heads], [dimension] * 3)
                            expected = heads[0].transform(self.queries[history])
                            for candidate in heads:
                                np.testing.assert_array_equal(candidate.transform(self.queries[history]), expected)
                                self.assertEqual(candidate.input_scaling["mode"], mode)
                                self.assertEqual(candidate.prefix_stage, stage)
                        candidate = by_name[f"{kind}_nested_m{history}_s1"]
                        scalers.append(candidate.input_scaling)
                        subsamples.append(candidate.transform.__defaults__[0].fit_indices_)
                    self.assertEqual(scalers[0], scalers[1])
                    self.assertEqual(scalers[0], scalers[2])
                    np.testing.assert_array_equal(subsamples[0], subsamples[1])
                    np.testing.assert_array_equal(subsamples[0], subsamples[2])
                for old in legacy:
                    name = old.name + "_s4" if old.family in ("kahm_frobenius", "kahm_koopman") else old.name
                    new = by_name[name]
                    np.testing.assert_array_equal(new.validation_mse, old.validation_mse)
                    query = self.queries[old.history]
                    np.testing.assert_array_equal(new.predict(new.transform(query), cfg["horizons"]),
                                                  old.predict(old.transform(query), cfg["horizons"]))

    def test_bad_scaling_mapping_fails_before_geometry_training(self):
        for mapping in ("minmax", {"unknown": "minmax"}, {"rbf": "typo"}):
            with self.subTest(mapping=mapping), patch.object(forecasting.NestedSoftFeatures, "fit") as fit:
                with self.assertRaises(ValueError):
                    forecasting.build_candidates(self.train, self.validation,
                        {**self.cfg, "input_scaling_by_kind": mapping}, 915703)
                fit.assert_not_called()

    def test_reference_schedule_and_confidence_allocations(self):
        args = Namespace(config_dir=ROOT / "configs", campaigns=list(study.CONFIGS), modes=list(study.MODES))
        planned = study.configurations(args)
        self.assertEqual(len(planned), 12)
        jobs, candidate_fits = 0, 0
        for item in planned:
            cfg = item["configuration"]
            self.assertEqual(cfg["seeds"], list(range(5701, 5706)))
            self.assertEqual(study.candidate_count(cfg), 84)
            if item["campaign"] == "pendigits_v1":
                self.assertEqual(cfg["delta"] * 4, 0.05)
                jobs += 1
                candidate_fits += 84 * len(cfg["coordinates"]) * len(cfg["seeds"])
            else:
                self.assertAlmostEqual(cfg["delta"] * 4 + 0.01, 0.05)
                self.assertEqual(cfg["risk_delta_fraction"], 0.8)
                n = len(cfg["systems"]) * len(cfg["seeds"])
                jobs += n
                candidate_fits += n * 84
        self.assertEqual((jobs, candidate_fits), (124, 13440))

    def test_freeze_rejects_changed_protocol_source_config_and_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            cfg = json.loads((ROOT / "configs/matched_geometry_main.json").read_text())
            cfg.update(systems=["harmonic"], seeds=[915701])
            (config_dir / "matched_geometry_main.json").write_text(json.dumps(cfg))
            protocol = root / "protocol_source.md"
            protocol.write_text("Frozen test protocol\n")
            args = Namespace(output=root / "study", config_dir=config_dir, prepared=root / "unused.npz",
                             campaigns=["main_v1"], modes=["standard"], workers=1)
            with patch.object(study, "PROTOCOL", protocol), patch.object(study, "source_hashes", return_value={"test.py": "frozen"}):
                planned = study.configurations(args)
                jobs = study.freeze(args, planned)
                self.assertEqual(study.validate_freeze(args.output, planned), jobs)
                self.assertFalse(Path(jobs[0]["destination"]).exists())
                with patch.object(study, "source_hashes", return_value={"test.py": "changed"}):
                    with self.assertRaisesRegex(RuntimeError, "Source changed"):
                        study.validate_freeze(args.output, planned)
                protocol.write_text("Changed protocol\n")
                with self.assertRaisesRegex(RuntimeError, "protocol changed"):
                    study.validate_freeze(args.output, planned)
                protocol.write_text("Frozen test protocol\n")
                changed = copy.deepcopy(planned)
                changed[0]["configuration"]["seeds"] = [915702]
                with self.assertRaisesRegex(RuntimeError, "schedule differ"):
                    study.validate_freeze(args.output, changed)
                config_file = Path(jobs[0]["config_file"])
                original = config_file.read_bytes()
                config_file.write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "Frozen file changed"):
                    study.validate_freeze(args.output, planned)
                config_file.write_bytes(original)
                (args.output / "schedule.json").write_text("[]")
                with self.assertRaisesRegex(RuntimeError, "Frozen file changed"):
                    study.validate_freeze(args.output, planned)

    def test_resume_revalidates_completed_artifacts_and_preserves_partial_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "result"
            job = {"id": "example", "destination": str(destination), "log": str(root / "log.txt"),
                   "status_file": str(root / "status.json")}
            self.assertEqual(study.resume_state([job]), ([], [job]))
            destination.mkdir()
            with self.assertRaisesRegex(RuntimeError, "partial attempt"):
                study.resume_state([job])
            artifact = destination / "results.json"
            artifact.write_text('{"complete":true}')
            status = {"id": "example", "status": "completed",
                      "artifact_sha256": {"results.json": study.digest(artifact)}}
            Path(job["status_file"]).write_text(json.dumps(status))
            self.assertEqual(study.resume_state([job]), ([status], []))
            artifact.write_text('{"complete":false}')
            with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
                study.resume_state([job])

    def test_score_archive_requires_complete_finite_consistent_arrays(self):
        cfg = {"n_calibration": 4, "n_test": 3, "horizons": [1, 2]}
        losses = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]])
        record = {"results": [{"name": "candidate", "test_mse": losses.mean(axis=0).tolist()}]}
        arrays = {"candidate__cal_loss": np.zeros((4, 2)),
                  "candidate__pair_score": np.zeros((2, 2)), "candidate__test_loss": losses}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            path = destination / "episode_scores.npz"
            forecasting.write_npz_atomic(path, arrays)
            study.verify_score_archive(destination, record, cfg)
            path.unlink()
            arrays["candidate__test_loss"][0, 0] = np.nan
            forecasting.write_npz_atomic(path, arrays)
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                study.verify_score_archive(destination, record, cfg)


if __name__ == "__main__":
    unittest.main()
