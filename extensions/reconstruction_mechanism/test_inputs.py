"""Independent development-only input audit; no forecasting models are fitted.

All generated episodes use development seed 975001. The reserved seed list is
inspected for domain separation only; no reserved arrays are generated or read.
Scalar loops independently reconstruct the causal window indices and AR law.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from extensions.reconstruction_mechanism import data


def independent_seed(parts):
    return int(np.random.SeedSequence([int(value) for value in parts]).generate_state(1)[0])


def scalar_windows(episodes, row_indices, history, horizons, stride):
    """Decode each flattened row and read every scalar time index separately."""
    allowed_origins = []
    origin = history - 1
    while origin + max(horizons) < episodes.shape[1]:
        allowed_origins.append(origin)
        origin += stride
    n_origins = len(allowed_origins)
    x, y, groups, origins = [], [], [], []
    for row in row_indices:
        group, local_row = divmod(int(row), n_origins)
        origin = allowed_origins[local_row]
        history_times = list(range(origin - history + 1, origin + 1))
        future_times = [origin + int(offset) for offset in horizons]
        assert min(history_times) >= 0 and max(history_times) == origin
        assert min(future_times) > origin and max(future_times) < episodes.shape[1]
        x.append([float(episodes[group, time]) for time in history_times])
        y.append([float(episodes[group, time]) for time in future_times])
        groups.append(group)
        origins.append(origin)
    return np.asarray(x), np.asarray(y), np.asarray(groups), np.asarray(origins)


def scalar_linear_ar(n, length, seed, specification):
    """Use scalar RNG draws and recurrences, retaining the declared time range."""
    burn = int(specification["burn_in_stored_values"])
    a, b = [float(value) for value in specification["coefficients"]]
    initial = [float(value) for value in specification["initial_values"]]
    values = [[initial[0], initial[1]] for _ in range(n)]
    rng = np.random.default_rng(seed)
    for time in range(2, burn + length):
        for episode in range(n):
            innovation = float(rng.normal(0.0, float(specification["innovation_std"])))
            values[episode].append(
                a * values[episode][time - 1] + b * values[episode][time - 2] + innovation)
    return np.asarray([row[burn:burn + length] for row in values]), np.asarray(values)


def scalar_variance(values):
    flat = [float(value) for value in np.asarray(values).ravel()]
    center = math.fsum(flat) / len(flat)
    return math.fsum((value - center) ** 2 for value in flat) / len(flat)


class DevelopmentInputAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path(__file__).with_name("config.json").read_text())
        cls.seed = int(cls.config["development_seed"])
        if cls.seed != 975001 or cls.seed in cls.config["seeds"]:
            raise AssertionError("The audit is restricted to development seed 975001")
        cls.pairs = {
            law: list(data.prepare_pair(cls.config, law, cls.seed, 0))
            for law in cls.config["laws"]
        }

    def test_scalar_indexing_of_every_retained_development_row(self):
        config = self.config
        checked_rows = 0
        for law, pair in self.pairs.items():
            for arrays, metadata in pair:
                for split in data.SPLITS:
                    with self.subTest(law=law, level=metadata["noise_level"], split=split):
                        expected = scalar_windows(
                            arrays[f"{split}_episodes"], arrays[f"{split}_row_indices"],
                            config["history"], config["horizons"], config["stride"])
                        for key, value in zip(("X", "Y", "groups", "origins"), expected):
                            np.testing.assert_array_equal(arrays[f"{split}_{key}"], value)
                        checked_rows += len(expected[0])
        self.assertEqual(checked_rows, 21504)

    def test_row_caps_depend_only_on_seeded_indices(self):
        config = self.config
        for law, pair in self.pairs.items():
            law_index = config["laws"].index(law)
            for arrays, metadata in pair:
                for split_index, split in enumerate(data.SPLITS):
                    n_origins = sum(1 for origin in range(config["history"] - 1,
                                                          config["episode_length"], config["stride"])
                                    if origin + max(config["horizons"]) < config["episode_length"])
                    full_rows = config["split_episodes"][split] * n_origins
                    cap = config["split_caps"][split]
                    parts = [self.seed, law_index, split_index, config["seed_salts"]["caps"]]
                    expected = np.arange(full_rows, dtype=np.int64)
                    if cap is not None and cap < full_rows:
                        expected = np.sort(np.random.default_rng(independent_seed(parts)).choice(
                            full_rows, cap, replace=False))
                    np.testing.assert_array_equal(arrays[f"{split}_row_indices"], expected)
                    self.assertEqual(len(set(expected.tolist())), len(expected))
                    self.assertEqual(metadata["splits"][split]["uncapped_rows"], full_rows)
                    self.assertEqual(metadata["splits"][split]["retained_rows"], len(expected))
                    if split == "D":
                        self.assertEqual(len(expected), 1664)
                    else:
                        self.assertEqual(len(expected), {"A": 512, "B": 256, "C": 256}[split])

    def test_noise_levels_pair_histories_targets_and_rows(self):
        self.assertEqual(self.config["noise_levels"], [0.0, 0.1])
        for law, pair in self.pairs.items():
            clean, clean_meta = pair[0]
            noisy, noisy_meta = pair[1]
            amplitude = math.sqrt(scalar_variance(clean["A_base_episodes"]))
            self.assertAlmostEqual(float(noisy["A_base_episode_std"]), amplitude, places=14)
            self.assertAlmostEqual(float(noisy["noise_std"]), 0.1 * amplitude, places=14)
            self.assertEqual(clean_meta["fit_seed"], noisy_meta["fit_seed"])
            for split in data.SPLITS:
                for suffix in ("base_episodes", "unit_noise", "row_indices", "groups", "origins"):
                    np.testing.assert_array_equal(clean[f"{split}_{suffix}"], noisy[f"{split}_{suffix}"])
                np.testing.assert_array_equal(clean[f"{split}_episodes"], clean[f"{split}_base_episodes"])
                expected = noisy[f"{split}_base_episodes"] + 0.1 * amplitude * noisy[f"{split}_unit_noise"]
                np.testing.assert_allclose(noisy[f"{split}_episodes"], expected, rtol=2e-15, atol=2e-15)
                self.assertTrue(np.any(clean[f"{split}_X"] != noisy[f"{split}_X"]))
                self.assertTrue(np.any(clean[f"{split}_Y"] != noisy[f"{split}_Y"]))

    def test_development_split_law_and_rng_role_separation(self):
        self.assertEqual(len(set(self.config["seeds"])), len(self.config["seeds"]))
        self.assertNotIn(self.seed, self.config["seeds"])
        stream_seeds = []
        for law_index, law in enumerate(self.config["laws"]):
            arrays, metadata = self.pairs[law][0]
            for split_index, split in enumerate(data.SPLITS):
                record = metadata["splits"][split]
                for key, salt_name in (("generation", "generation"),
                                       ("unit_noise", "unit_noise"), ("cap", "caps")):
                    parts = [self.seed, law_index, split_index, self.config["seed_salts"][salt_name]]
                    expected = independent_seed(parts)
                    self.assertEqual(record[f"{key}_seed_components"], parts)
                    self.assertEqual(record[f"{key}_seed"], expected)
                    stream_seeds.append(expected)
            fit_parts = [self.seed, law_index, self.config["seed_salts"]["fit"]]
            self.assertEqual(metadata["fit_seed"], independent_seed(fit_parts))
            stream_seeds.append(metadata["fit_seed"])
            self.assertEqual(len({metadata["splits"][split]["base_episode_sha256"]
                                  for split in data.SPLITS}), 4)
            self.assertEqual(len({metadata["splits"][split]["unit_noise_sha256"]
                                  for split in data.SPLITS}), 4)
        self.assertEqual(len(set(stream_seeds)), len(stream_seeds))

    def test_scalar_ar_recurrence_and_exact_stored_burn_index(self):
        specification = self.config["linear_ar"]
        self.assertEqual(specification["burn_in_stored_values"], 256)
        self.assertEqual(specification["initial_values"], [0.0, 0.0])
        a, b = specification["coefficients"]
        eigenvalues = np.roots([1.0, -a, -b])
        self.assertTrue(np.all(np.abs(eigenvalues) < 1.0))
        np.testing.assert_allclose(np.abs(eigenvalues), np.sqrt(0.15), rtol=1e-14, atol=1e-14)
        arrays, metadata = self.pairs["linear_ar"][0]
        for split in data.SPLITS:
            n = self.config["split_episodes"][split]
            seed = metadata["splits"][split]["generation_seed"]
            expected, all_values = scalar_linear_ar(n, self.config["episode_length"], seed, specification)
            np.testing.assert_array_equal(arrays[f"{split}_base_episodes"], expected)
            np.testing.assert_array_equal(expected[:, 0], all_values[:, 256])
            np.testing.assert_array_equal(expected[:, -1], all_values[:, 327])

    def test_loss_scale_uses_only_retained_A_targets(self):
        for pair in self.pairs.values():
            for arrays, metadata in pair:
                expected = np.asarray([
                    max(scalar_variance(arrays["A_Y"][:, horizon]),
                        self.config["analysis"]["variance_floor"])
                    for horizon in range(len(self.config["horizons"]))
                ])
                np.testing.assert_allclose(arrays["loss_scale"], expected, rtol=2e-14, atol=1e-15)
                np.testing.assert_array_equal(arrays["loss_scale"], np.asarray(metadata["loss_scale"]))

    def test_perturbing_development_D_changes_neither_fit_inputs_nor_scales(self):
        original_pair = self.pairs["linear_ar"]
        original_generator = data.linear_episodes
        d_seed = original_pair[0][1]["splits"]["D"]["generation_seed"]

        def changed_D(n, length, seed, config):
            values = original_generator(n, length, seed, config)
            return 1000.0 + 100.0 * values if seed == d_seed else values

        with patch.object(data, "linear_episodes", changed_D):
            changed_pair = list(data.prepare_pair(copy.deepcopy(self.config), "linear_ar", self.seed, 0))
        for (original, _), (changed, _) in zip(original_pair, changed_pair):
            for key in original:
                if not key.startswith("D_"):
                    np.testing.assert_array_equal(original[key], changed[key])
            self.assertFalse(np.array_equal(original["D_base_episodes"], changed["D_base_episodes"]))
            np.testing.assert_array_equal(original["D_row_indices"], changed["D_row_indices"])
            np.testing.assert_array_equal(original["D_unit_noise"], changed["D_unit_noise"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
