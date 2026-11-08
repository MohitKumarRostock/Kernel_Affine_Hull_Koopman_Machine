"""Targeted numerical checks for the matched-geometry implementation."""
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from geometry import GeometryBank, _upstream, associations, folding_components


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(952)
        cls.X = rng.normal(size=(180, 4))
        cls.X[:, 1] += 0.6 * cls.X[:, 0] ** 2
        cls.Q = rng.normal(size=(1007, 4))
        cls.bank = GeometryBank(seed=59, cluster_counts=(2, 4), rank=2, max_refs=16).fit(cls.X)

    def test_matches_upstream_filtering_and_affinities(self):
        original = _upstream("autoencoder_filtering_extended").autoencoder_filtering_extended
        for n in (17, 1007):
            Q = self.Q[:n]
            observed = self.bank.transform(Q, diagnostics=True, include_reconstruction=True)
            Z = self.bank._scale_inputs(Q)
            score_blocks, reconstruction_vectors, phi_blocks = [], [], []
            for block in self.bank.blocks_:
                scores = []
                for model in block["models"]:
                    reconstruction, score = original(Z.T, model.ae, "folding")
                    scores.append(score)
                    reconstruction_vectors.append(reconstruction.T)
                score_block = np.column_stack(scores)
                score_blocks.append(score_block)
                # Direct upstream stable_associations formula, independently evaluated.
                logs = 16 * np.log(1 - np.clip(score_block, 0, 1) + 1e-6)
                weights = np.exp(logs - logs.max(axis=1, keepdims=True))
                phi_blocks.append(weights / weights.sum(axis=1, keepdims=True))
            np.testing.assert_allclose(observed["kahm:scores"], np.concatenate(score_blocks, axis=1), rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(observed["kahm:reconstruction"], np.concatenate(reconstruction_vectors, axis=1), rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(observed["kahm:phi"], np.concatenate(phi_blocks, axis=1), rtol=1e-13, atol=1e-14)

    def test_normalization_coordinate_recovers_scores(self):
        features = self.bank.transform(self.Q[:77])
        for geometry in ("kahm", "affine", "centroid"):
            phi = features[f"{geometry}:phi"]
            logz = features[f"{geometry}:logz"]
            recovered = []
            for b, section in enumerate(self.bank.block_slices_):
                np.testing.assert_allclose(phi[:, section].sum(axis=1), 1.0, atol=1e-14)
                recovered.append(1 + self.bank.tau - np.exp((np.log(phi[:, section]) + logz[:, b:b+1]) / self.bank.omega))
            np.testing.assert_allclose(np.concatenate(recovered, axis=1), features[f"{geometry}:scores"], rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(features[f"{geometry}:phi_raw"][:, self.bank.n_features_out_:], features["raw"])

    def test_transform_cannot_fit_on_queries(self):
        before = pickle.dumps(self.bank.state_dict(), protocol=5)
        self.bank.transform(self.Q[:23] * 30 + 500, diagnostics=True)
        after = pickle.dumps(self.bank.state_dict(), protocol=5)
        self.assertEqual(before, after)

    def test_shared_reference_geometry_and_two_dimensional_rank(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(100, 2))
        bank = GeometryBank(seed=17, cluster_counts=(2,), rank=2, max_refs=18).fit(X)
        Z = bank._scale_inputs(X)
        out = bank.transform(X[:13], include_reconstruction=True)
        affine = []
        for model in bank.blocks_[0]["models"]:
            self.assertEqual(model.requested_rank, 1)
            self.assertEqual(model.basis.shape[1], 1)
            np.testing.assert_array_equal(model.references, np.asarray(Z[model.reference_indices], dtype=np.float32))
            np.testing.assert_array_equal(model.ae["y_data"].T, model.references)
            np.testing.assert_array_equal(model.basis, model.ae["PC"])
            affine.append(model.center + ((Z[:13] - model.center) @ model.basis) @ model.basis.T)
        np.testing.assert_allclose(out["affine:reconstruction"], np.concatenate(affine, axis=1))

    def test_save_roundtrip_and_seed_reproducibility(self):
        expected = self.bank.transform(self.Q[:29])
        with tempfile.TemporaryDirectory() as directory:
            path = self.bank.save(Path(directory) / "bank.pkl")
            restored = GeometryBank.load(path)
            actual = restored.transform(self.Q[:29])
        fresh = GeometryBank(seed=59, cluster_counts=(2, 4), rank=2, max_refs=16).fit(self.X)
        repeated = fresh.transform(self.Q[:29])
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name])
            np.testing.assert_array_equal(expected[name], repeated[name])

    def test_degenerate_and_oracle_partitions_are_explicit(self):
        constant = np.ones((20, 3))
        bank = GeometryBank(cluster_counts=(1,)).fit(constant)
        self.assertEqual(bank.fit_diagnostics_["fallback_clusters"], 1)
        features = bank.transform(constant)
        np.testing.assert_array_equal(features["kahm:phi"], np.ones((20, 1)))
        labels = np.arange(len(self.X)) % 2
        oracle = GeometryBank(cluster_counts=(2,), oracle_labels=labels, max_fit=None).fit(self.X)
        np.testing.assert_array_equal(oracle.blocks_[0]["labels"], labels)
        for c, model in enumerate(oracle.blocks_[0]["models"]):
            self.assertTrue(np.all(labels[model.reference_indices] == c))
        self.assertEqual(oracle.blocks_[0]["partition_kind"], "oracle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
