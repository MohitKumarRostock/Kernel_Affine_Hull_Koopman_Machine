"""Reproducibility and validation checks for the count sampler.

These are unit-test draws, not campaign evidence or a distributional test.
The NumPy reference independently assembles the documented five-part seed
and call recipe; it does not independently implement the random generator.
No test assumes that different seeds must produce different count vectors.
Checks apply within the running environment, not across NumPy versions.
Mock-based tests inspect the call contract and deliberately inject failures.
The committed pilot schedule is inspected, not executed.
"""

from contextlib import ExitStack
from dataclasses import FrozenInstanceError, asdict
from fractions import Fraction
import json
from pathlib import Path
import runpy
import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np

import kahkm_four_state_sampling as sampling

COMPONENTS = (
    "campaign_key", "case_key", "n_pairs", "replicate_index", "root_seed",
)
CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "reproduction" / "certificates"
    / "configs" / "four_state_pilot_v1.json"
)


def direct_numpy_counts(params):
    """Separate seed assembly; no production helpers or shared generator."""
    sequence = np.random.SeedSequence([
        int(params["campaign_key"]), int(params["case_key"]),
        int(params["n_pairs"]), int(params["replicate_index"]),
        int(params["root_seed"]),
    ], pool_size=4)
    generator = np.random.Generator(np.random.PCG64(sequence))
    counts = generator.multinomial(int(params["n_pairs"]), (.25,)*4, size=None)
    return tuple(int(n) for n in counts)


def load_pilot():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class FourStateSamplingTests(unittest.TestCase):
    def parameters(self, **changes):
        # Test-only campaign key, distinct from the committed pilot key.
        params = dict(campaign_key=2**32-2, case_key=6, n_pairs=128,
                      replicate_index=7, root_seed=20261002)
        params.update(changes)
        return params

    def sample(self, **changes):
        return sampling.sample_four_state_counts(**self.parameters(**changes))

    def assert_valid_counts(self, result):
        self.assertIsInstance(result.state_counts, tuple)
        self.assertEqual(len(result.state_counts), 4)
        self.assertTrue(all(type(n) is int and n >= 0 for n in result.state_counts))
        self.assertEqual(sum(result.state_counts), result.n_pairs)

    def assert_same_legacy_state(self, first, second):
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2:], second[2:])

    def test_direct_numpy_recipe(self):
        for n in (1, 128, 4096, 8192):
            for case in (0, 6):
                for replicate in (0, 7, 255):
                    with self.subTest(n=n, case=case, replicate=replicate):
                        params = self.parameters(n_pairs=n, case_key=case,
                                                 replicate_index=replicate)
                        result = sampling.sample_four_state_counts(**params)
                        self.assertEqual(result.state_counts, direct_numpy_counts(params))
                        self.assert_valid_counts(result)

    def test_metadata_and_plain_integer_fields(self):
        result = self.sample()
        for name, value in self.parameters().items():
            self.assertEqual(getattr(result, name), value)
            self.assertIs(type(getattr(result, name)), int)
        self.assertEqual(result.sampler_id, "four_state_multinomial_pcg64_v1")
        self.assertEqual(result.numpy_version, np.__version__)
        self.assertEqual(result.seed_material_order, COMPONENTS)
        self.assertEqual(result.seed_material,
                         tuple(self.parameters()[name] for name in COMPONENTS))
        self.assertEqual(result.bit_generator, "PCG64")
        self.assertEqual(result.seed_sequence, "SeedSequence")
        self.assertEqual(result.seed_sequence_pool_size, 4)
        self.assertEqual(result.draws_per_replicate, 1)
        self.assertEqual(result.state_order, ("a", "b", "d", "e"))
        self.assertEqual(result.probabilities, (.25,)*4)
        self.assert_valid_counts(result)

    def test_repeatability_after_unrelated_calls(self):
        first = self.sample()
        for index in range(12):
            self.sample(case_key=index, replicate_index=index+31)
        self.assertEqual(first, self.sample())

    def test_execution_order_and_subsets(self):
        jobs = [(c, n, r) for c in (0, 6) for n in (128, 4096) for r in (0, 7, 19)]
        def execute(order):
            return {job: self.sample(case_key=job[0], n_pairs=job[1],
                                     replicate_index=job[2]) for job in order}
        forward = execute(jobs)
        self.assertEqual(forward, execute(reversed(jobs)))
        self.assertEqual({job: forward[job] for job in jobs[::3]}, execute(jobs[::3]))

    def test_legacy_global_rng_is_not_advanced(self):
        before = np.random.get_state()
        for index in range(5):
            self.sample(replicate_index=index)
        self.assert_same_legacy_state(before, np.random.get_state())

    def test_legacy_global_seed_does_not_change_results(self):
        before = np.random.get_state()
        try:
            np.random.seed(123)
            first = self.sample()
            np.random.seed(987)
            np.random.random(17)
            self.assertEqual(first, self.sample())
        finally:
            np.random.set_state(before)

    def test_each_seed_component_is_preserved(self):
        base = self.parameters()
        for position, name in enumerate(COMPONENTS):
            with self.subTest(component=name):
                changed = dict(base)
                changed[name] += 1
                result = sampling.sample_four_state_counts(**changed)
                expected = tuple(changed[key] for key in COMPONENTS)
                self.assertEqual(result.seed_material, expected)
                self.assertEqual(result.seed_material[position], base[name]+1)
                self.assertEqual(result.state_counts, direct_numpy_counts(changed))
        # Different count vectors are not required: sample coincidences are valid.

    def test_exact_constructors_and_one_multinomial_call(self):
        fake = Mock(spec=["multinomial"])
        fake.multinomial.return_value = np.array([1, 2, 3, 4], dtype=np.int64)
        with patch.object(np.random, "SeedSequence", return_value=sentinel.sequence) as seq, \
             patch.object(np.random, "PCG64", return_value=sentinel.bitgen) as bitgen, \
             patch.object(np.random, "Generator", return_value=fake) as factory:
            result = self.sample(n_pairs=10)
        seq.assert_called_once_with([2**32-2, 6, 10, 7, 20261002], pool_size=4)
        bitgen.assert_called_once_with(sentinel.sequence)
        factory.assert_called_once_with(sentinel.bitgen)
        fake.multinomial.assert_called_once_with(10, (.25,)*4, size=None)
        self.assertEqual(result.state_counts, (1, 2, 3, 4))

    def test_numpy_integer_normalization(self):
        expected = self.sample()
        for dtype in (np.int64, np.uint32, np.uint64):
            with self.subTest(dtype=dtype):
                result = sampling.sample_four_state_counts(**{
                    name: dtype(value) for name, value in self.parameters().items()
                })
                self.assertEqual(result, expected)
                self.assertTrue(all(type(x) is int for x in result.seed_material))

    def test_seed_range_boundaries_without_expanding_rows(self):
        lower = dict.fromkeys(COMPONENTS, 0)
        lower["n_pairs"] = 1
        upper = dict.fromkeys(COMPONENTS, 2**32-1)
        for params in (lower, upper):
            with self.subTest(params=params):
                result = sampling.sample_four_state_counts(**params)
                self.assert_valid_counts(result)
                self.assertEqual(result.state_counts, direct_numpy_counts(params))
        self.assertEqual(sampling.MAX_SEED_COMPONENT, 2**32-1)

    def test_pilot_configuration_matches_sampler(self):
        config = load_pilot()
        rng, law = config["rng"], config["sampling"]
        result = self.sample()
        self.assertEqual(config["campaign_role"], "pilot")
        self.assertEqual(config["schema_version"], 1)
        self.assertNotEqual(result.campaign_key, rng["campaign_key"])
        for name in ("bit_generator", "seed_sequence", "seed_sequence_pool_size"):
            self.assertEqual(rng[name], getattr(result, name))
        self.assertEqual(tuple(rng["seed_material_order"]), result.seed_material_order)
        self.assertEqual(rng["replicate_index_base"], 0)
        self.assertIs(rng["reuse_counts_across_cases"], False)
        self.assertEqual(law["count_sampler"], "numpy.random.Generator.multinomial")
        self.assertEqual(law["law"], "iid_uniform_four_state")
        self.assertEqual(law["independent_unit"], "evaluation_pair")
        self.assertEqual(law["draws_per_replicate"], result.draws_per_replicate)
        self.assertEqual(tuple(law["state_order"]), result.state_order)
        self.assertEqual(tuple(law["probabilities"]), result.probabilities)

    def test_pilot_seed_identifiers_are_unique_and_in_range(self):
        config = load_pilot()
        rng, law, cases = config["rng"], config["sampling"], config["cases"]
        self.assertEqual(len({c["case_id"] for c in cases}), len(cases))
        self.assertEqual(len({c["case_key"] for c in cases}), len(cases))
        materials = set()
        for case in cases:
            for n in law["sample_sizes"]:
                for replicate in range(law["replicates_per_cell"]):
                    material = (rng["campaign_key"], case["case_key"], n,
                                replicate, rng["root_seed"])
                    self.assertTrue(all(type(x) is int and 0 <= x < 2**32
                                        for x in material))
                    self.assertGreater(n, 0)
                    self.assertNotIn(material, materials)
                    materials.add(material)
        self.assertEqual(len(materials), 10240)
        # Only seed tuples are checked here; no pilot samples are generated.

    def test_invalid_types_rejected_before_rng_construction(self):
        invalid = (True, False, np.bool_(True), 1., np.float64(1), "1", None,
                   1j, Fraction(1), [1], np.array(1), float("nan"), float("inf"))
        for name in COMPONENTS:
            for value in invalid:
                with self.subTest(component=name, value=repr(value)):
                    with patch.object(np.random, "SeedSequence") as constructor:
                        with self.assertRaises(TypeError):
                            self.sample(**{name: value})
                        constructor.assert_not_called()

    def test_out_of_range_rejected_before_rng_construction(self):
        for name in COMPONENTS:
            values = (-1, 2**32, np.uint64(2**63))
            if name == "n_pairs":
                values += (0,)
            for value in values:
                with self.subTest(component=name, value=value):
                    with patch.object(np.random, "SeedSequence") as constructor:
                        with self.assertRaises(ValueError):
                            self.sample(**{name: value})
                        constructor.assert_not_called()

    def test_multinomial_failure_propagates_without_retry(self):
        failure = RuntimeError("injected sampling failure")
        fake = Mock(spec=["multinomial"])
        fake.multinomial.side_effect = failure
        with patch.object(np.random, "Generator", return_value=fake) as factory:
            with self.assertRaises(RuntimeError) as caught:
                self.sample()
        self.assertIs(caught.exception, failure)
        factory.assert_called_once()
        fake.multinomial.assert_called_once_with(128, (.25,)*4, size=None)

    def test_malformed_sampler_outputs_are_rejected(self):
        bad_outputs = (
            np.array([10]), np.array([[1, 2, 3, 4]]),
            np.array([1., 2., 3., 4.]), np.array([True]*4),
            np.array([1, 2, 3, 4], dtype=complex),
            np.array([-1, 2, 3, 6]), np.array([1, 2, 3, 5]),
            np.array([2**64-1, 1, 0, 0], dtype=np.uint64),
        )
        for bad in bad_outputs:
            with self.subTest(shape=bad.shape, dtype=str(bad.dtype)):
                fake = Mock(spec=["multinomial"])
                fake.multinomial.return_value = bad
                with patch.object(np.random, "Generator", return_value=fake):
                    with self.assertRaises(RuntimeError):
                        self.sample(n_pairs=10)
                fake.multinomial.assert_called_once()

    def test_constructor_failures_propagate_without_fallback(self):
        for name in ("SeedSequence", "PCG64", "Generator"):
            with self.subTest(constructor=name):
                failure = RuntimeError("injected constructor failure")
                with patch.object(np.random, name, side_effect=failure) as constructor:
                    with self.assertRaises(RuntimeError) as caught:
                        self.sample()
                self.assertIs(caught.exception, failure)
                constructor.assert_called_once()

    def test_json_roundtrip_and_reproduction_from_saved_seed(self):
        original = self.sample()
        saved = json.loads(json.dumps(asdict(original), allow_nan=False))
        self.assertEqual(saved["state_counts"], list(original.state_counts))
        self.assertEqual(saved["seed_material"], list(original.seed_material))
        rebuilt = sampling.sample_four_state_counts(**{name: saved[name] for name in COMPONENTS})
        self.assertEqual(rebuilt, original)

    def test_returned_record_is_immutable(self):
        result = self.sample()
        with self.assertRaises(FrozenInstanceError):
            result.n_pairs = 1
        with self.assertRaises(TypeError):
            result.state_counts[0] = 0
        with self.assertRaises(TypeError):
            result.seed_material[0] = 0

    def test_module_execution_does_not_draw_samples(self):
        with ExitStack() as stack:
            spies = [stack.enter_context(patch.object(
                np.random, name, side_effect=AssertionError("RNG used during import"),
            )) for name in ("SeedSequence", "PCG64", "Generator", "default_rng",
                            "seed", "multinomial")]
            loaded = runpy.run_path(str(Path(sampling.__file__).resolve()),
                                    run_name="_sampling_import_check_")
            for spy in spies:
                spy.assert_not_called()
        self.assertEqual(loaded["COUNT_SAMPLER_ID"], sampling.COUNT_SAMPLER_ID)


if __name__ == "__main__":
    unittest.main()
