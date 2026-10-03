"""Tests for independent Duffing/Van der Pol certificate pairs.

The matched-stochastic tests compare directly against the historical
benchmark generators. The deterministic tests use an independent reference
implementation local to this test module.
"""

from dataclasses import FrozenInstanceError
import math
import unittest

import numpy as np

from experiment_15_external_koopman_baselines import (
    make_vanderpol_snapshots,
)
from kahkm_dynamical_pairs import (
    PAIR_LAW_ID,
    DynamicalPair,
    sample_independent_pair,
)
from kernel_affine_hull_koopman_machines import (
    make_duffing_snapshots,
)


def reference_rk4(
    *,
    system,
    z,
    dt,
    mu,
):
    """Independent deterministic RK4 reference used only by tests."""
    z = np.asarray(z, dtype=np.float64)

    def field(state):
        q = float(state[0])
        p = float(state[1])

        if system == "duffing":
            return np.array(
                [
                    p,
                    -0.25 * p
                    + q
                    - q**3,
                ],
                dtype=np.float64,
            )

        if system == "vanderpol":
            return np.array(
                [
                    p,
                    float(mu)
                    * (1.0 - q * q)
                    * p
                    - q,
                ],
                dtype=np.float64,
            )

        raise AssertionError(system)

    k1 = field(z)
    k2 = field(
        z + 0.5 * dt * k1
    )
    k3 = field(
        z + 0.5 * dt * k2
    )
    k4 = field(
        z + dt * k3
    )

    return np.asarray(
        z
        + (dt / 6.0)
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        ),
        dtype=np.float64,
    )


def deterministic_reference_pair(
    *,
    system,
    horizon_steps,
    dt,
    trajectory_seed,
    time_seed,
    mu=1.0,
):
    """Independent implementation of the frozen deterministic pair law."""
    time_rng = np.random.default_rng(
        int(time_seed)
    )

    t = int(
        time_rng.integers(
            0,
            int(horizon_steps),
        )
    )

    trajectory_rng = np.random.default_rng(
        int(trajectory_seed)
    )

    if system == "duffing":
        center = np.array(
            [0.7, 0.0],
            dtype=np.float64,
        )
    elif system == "vanderpol":
        center = np.array(
            [2.0, 0.0],
            dtype=np.float64,
        )
    else:
        raise AssertionError(system)

    z = (
        center
        + 0.05
        * trajectory_rng.normal(size=2)
    )

    for _ in range(t):
        z = reference_rk4(
            system=system,
            z=z,
            dt=float(dt),
            mu=float(mu),
        )

    current = np.asarray(
        z,
        dtype=np.float64,
    )

    successor = reference_rk4(
        system=system,
        z=current,
        dt=float(dt),
        mu=float(mu),
    )

    return (
        t,
        current,
        successor,
    )


class DynamicalPairTests(unittest.TestCase):
    # 1
    def test_duffing_matched_stochastic_exactly_matches_historical_generator(self):
        horizon = 1200

        cases = (
            (17, 101),
            (819, 202),
            (123456, 303),
        )

        for trajectory_seed, time_seed in cases:
            with self.subTest(
                trajectory_seed=trajectory_seed,
                time_seed=time_seed,
            ):
                pair = sample_independent_pair(
                    system="duffing",
                    evaluation_mode="matched_stochastic",
                    horizon_steps=horizon,
                    dt=0.03,
                    trajectory_seed=trajectory_seed,
                    time_seed=time_seed,
                )

                X0, X1 = make_duffing_snapshots(
                    n_steps=horizon,
                    dt=0.03,
                    seed=trajectory_seed,
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.current_state
                    ),
                    X0[:, pair.time_index],
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.successor_state
                    ),
                    X1[:, pair.time_index],
                )

    # 2
    def test_vanderpol_matched_stochastic_exactly_matches_historical_generator(self):
        horizon = 1200

        cases = (
            (23, 111),
            (991, 222),
            (654321, 333),
        )

        for trajectory_seed, time_seed in cases:
            with self.subTest(
                trajectory_seed=trajectory_seed,
                time_seed=time_seed,
            ):
                pair = sample_independent_pair(
                    system="vanderpol",
                    evaluation_mode="matched_stochastic",
                    horizon_steps=horizon,
                    dt=0.02,
                    trajectory_seed=trajectory_seed,
                    time_seed=time_seed,
                    vanderpol_mu=1.0,
                )

                X0, X1 = make_vanderpol_snapshots(
                    n_steps=horizon,
                    dt=0.02,
                    seed=trajectory_seed,
                    mu=1.0,
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.current_state
                    ),
                    X0[:, pair.time_index],
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.successor_state
                    ),
                    X1[:, pair.time_index],
                )

    # 3
    def test_time_index_uses_separate_seeded_rng(self):
        horizon = 1200
        time_seed = 7001

        expected = int(
            np.random.default_rng(
                time_seed
            ).integers(
                0,
                horizon,
            )
        )

        pair = sample_independent_pair(
            system="duffing",
            evaluation_mode="matched_stochastic",
            horizon_steps=horizon,
            dt=0.03,
            trajectory_seed=5001,
            time_seed=time_seed,
        )

        self.assertEqual(
            pair.time_index,
            expected,
        )

    # 4
    def test_duffing_deterministic_matches_independent_reference(self):
        for trajectory_seed, time_seed in (
            (4, 8),
            (44, 88),
            (444, 888),
        ):
            with self.subTest(
                trajectory_seed=trajectory_seed,
                time_seed=time_seed,
            ):
                pair = sample_independent_pair(
                    system="duffing",
                    evaluation_mode="deterministic",
                    horizon_steps=1200,
                    dt=0.03,
                    trajectory_seed=trajectory_seed,
                    time_seed=time_seed,
                )

                (
                    expected_t,
                    expected_current,
                    expected_successor,
                ) = deterministic_reference_pair(
                    system="duffing",
                    horizon_steps=1200,
                    dt=0.03,
                    trajectory_seed=trajectory_seed,
                    time_seed=time_seed,
                )

                self.assertEqual(
                    pair.time_index,
                    expected_t,
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.current_state
                    ),
                    expected_current,
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.successor_state
                    ),
                    expected_successor,
                )

    # 5
    def test_vanderpol_deterministic_matches_independent_reference(self):
        for trajectory_seed, time_seed in (
            (5, 9),
            (55, 99),
            (555, 999),
        ):
            with self.subTest(
                trajectory_seed=trajectory_seed,
                time_seed=time_seed,
            ):
                pair = sample_independent_pair(
                    system="vanderpol",
                    evaluation_mode="deterministic",
                    horizon_steps=1200,
                    dt=0.02,
                    trajectory_seed=trajectory_seed,
                    time_seed=time_seed,
                    vanderpol_mu=1.0,
                )

                (
                    expected_t,
                    expected_current,
                    expected_successor,
                ) = deterministic_reference_pair(
                    system="vanderpol",
                    horizon_steps=1200,
                    dt=0.02,
                    trajectory_seed=trajectory_seed,
                    time_seed=time_seed,
                    mu=1.0,
                )

                self.assertEqual(
                    pair.time_index,
                    expected_t,
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.current_state
                    ),
                    expected_current,
                )

                np.testing.assert_array_equal(
                    np.asarray(
                        pair.successor_state
                    ),
                    expected_successor,
                )

    # 6
    def test_vanderpol_nondefault_mu_matches_historical_generator(self):
        pair = sample_independent_pair(
            system="vanderpol",
            evaluation_mode="matched_stochastic",
            horizon_steps=300,
            dt=0.02,
            trajectory_seed=321,
            time_seed=654,
            vanderpol_mu=2.25,
        )

        X0, X1 = make_vanderpol_snapshots(
            n_steps=300,
            dt=0.02,
            seed=321,
            mu=2.25,
        )

        np.testing.assert_array_equal(
            np.asarray(
                pair.current_state
            ),
            X0[:, pair.time_index],
        )

        np.testing.assert_array_equal(
            np.asarray(
                pair.successor_state
            ),
            X1[:, pair.time_index],
        )

    # 7
    def test_time_index_always_in_frozen_support(self):
        for time_seed in range(100):
            pair = sample_independent_pair(
                system="duffing",
                evaluation_mode="deterministic",
                horizon_steps=1200,
                dt=0.03,
                trajectory_seed=17,
                time_seed=time_seed,
            )

            self.assertGreaterEqual(
                pair.time_index,
                0,
            )

            self.assertLess(
                pair.time_index,
                1200,
            )

    # 8
    def test_horizon_one_forces_time_zero(self):
        pair = sample_independent_pair(
            system="vanderpol",
            evaluation_mode="deterministic",
            horizon_steps=1,
            dt=0.02,
            trajectory_seed=7,
            time_seed=8,
        )

        self.assertEqual(
            pair.time_index,
            0,
        )

    # 9
    def test_repeatability(self):
        kwargs = {
            "system": "duffing",
            "evaluation_mode":
                "matched_stochastic",
            "horizon_steps": 1200,
            "dt": 0.03,
            "trajectory_seed": 314159,
            "time_seed": 271828,
        }

        first = sample_independent_pair(
            **kwargs
        )

        second = sample_independent_pair(
            **kwargs
        )

        self.assertEqual(
            first,
            second,
        )

    # 10
    def test_pair_metadata(self):
        pair = sample_independent_pair(
            system="vanderpol",
            evaluation_mode="matched_stochastic",
            horizon_steps=1200,
            dt=0.02,
            trajectory_seed=71,
            time_seed=72,
            vanderpol_mu=1.0,
        )

        self.assertEqual(
            pair.pair_law_id,
            PAIR_LAW_ID,
        )

        self.assertEqual(
            pair.system,
            "vanderpol",
        )

        self.assertEqual(
            pair.evaluation_mode,
            "matched_stochastic",
        )

        self.assertEqual(
            pair.trajectory_seed,
            71,
        )

        self.assertEqual(
            pair.time_seed,
            72,
        )

        self.assertEqual(
            pair.horizon_steps,
            1200,
        )

        self.assertEqual(
            pair.dt,
            0.02,
        )

        self.assertEqual(
            pair.vanderpol_mu,
            1.0,
        )

    # 11
    def test_returned_states_are_finite_float_pairs(self):
        for system, dt in (
            ("duffing", 0.03),
            ("vanderpol", 0.02),
        ):
            pair = sample_independent_pair(
                system=system,
                evaluation_mode="matched_stochastic",
                horizon_steps=1200,
                dt=dt,
                trajectory_seed=9,
                time_seed=10,
            )

            self.assertEqual(
                len(pair.current_state),
                2,
            )

            self.assertEqual(
                len(pair.successor_state),
                2,
            )

            for value in (
                *pair.current_state,
                *pair.successor_state,
            ):
                self.assertIsInstance(
                    value,
                    float,
                )

                self.assertTrue(
                    math.isfinite(value)
                )

    # 12
    def test_result_is_immutable(self):
        pair = sample_independent_pair(
            system="duffing",
            evaluation_mode="deterministic",
            horizon_steps=3,
            dt=0.03,
            trajectory_seed=1,
            time_seed=2,
        )

        self.assertIsInstance(
            pair,
            DynamicalPair,
        )

        with self.assertRaises(
            FrozenInstanceError
        ):
            pair.time_index = 0

        with self.assertRaises(
            TypeError
        ):
            pair.current_state[0] = 0.0

    # 13
    def test_unknown_system_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            sample_independent_pair(
                system="lorenz",
                evaluation_mode="deterministic",
                horizon_steps=10,
                dt=0.01,
                trajectory_seed=1,
                time_seed=2,
            )

    # 14
    def test_nonstring_system_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            sample_independent_pair(
                system=123,
                evaluation_mode="deterministic",
                horizon_steps=10,
                dt=0.01,
                trajectory_seed=1,
                time_seed=2,
            )

    # 15
    def test_unknown_evaluation_mode_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            sample_independent_pair(
                system="duffing",
                evaluation_mode="stochasticish",
                horizon_steps=10,
                dt=0.03,
                trajectory_seed=1,
                time_seed=2,
            )

    # 16
    def test_nonstring_evaluation_mode_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            sample_independent_pair(
                system="duffing",
                evaluation_mode=None,
                horizon_steps=10,
                dt=0.03,
                trajectory_seed=1,
                time_seed=2,
            )

    # 17
    def test_invalid_horizon_rejected(self):
        invalid = (
            0,
            -1,
            1.5,
            True,
        )

        for value in invalid:
            with self.subTest(
                value=value
            ):
                expected = (
                    TypeError
                    if isinstance(
                        value,
                        (float, bool),
                    )
                    else ValueError
                )

                with self.assertRaises(
                    expected
                ):
                    sample_independent_pair(
                        system="duffing",
                        evaluation_mode="deterministic",
                        horizon_steps=value,
                        dt=0.03,
                        trajectory_seed=1,
                        time_seed=2,
                    )

    # 18
    def test_invalid_dt_rejected(self):
        invalid = (
            0.0,
            -0.01,
            math.nan,
            math.inf,
            -math.inf,
            True,
            "0.03",
        )

        for value in invalid:
            with self.subTest(
                value=value
            ):
                expected = (
                    TypeError
                    if isinstance(
                        value,
                        (bool, str),
                    )
                    else ValueError
                )

                with self.assertRaises(
                    expected
                ):
                    sample_independent_pair(
                        system="duffing",
                        evaluation_mode="deterministic",
                        horizon_steps=10,
                        dt=value,
                        trajectory_seed=1,
                        time_seed=2,
                    )

    # 19
    def test_invalid_seeds_rejected(self):
        for name in (
            "trajectory_seed",
            "time_seed",
        ):
            for value in (
                -1,
                1.5,
                True,
                "7",
            ):
                with self.subTest(
                    name=name,
                    value=value,
                ):
                    kwargs = {
                        "system": "duffing",
                        "evaluation_mode":
                            "deterministic",
                        "horizon_steps": 10,
                        "dt": 0.03,
                        "trajectory_seed": 1,
                        "time_seed": 2,
                    }

                    kwargs[name] = value

                    expected = (
                        ValueError
                        if value == -1
                        else TypeError
                    )

                    with self.assertRaises(
                        expected
                    ):
                        sample_independent_pair(
                            **kwargs
                        )

    # 20
    def test_invalid_vanderpol_mu_rejected(self):
        invalid = (
            0.0,
            -1.0,
            math.nan,
            math.inf,
            -math.inf,
            True,
            "1.0",
        )

        for value in invalid:
            with self.subTest(
                value=value
            ):
                expected = (
                    TypeError
                    if isinstance(
                        value,
                        (bool, str),
                    )
                    else ValueError
                )

                with self.assertRaises(
                    expected
                ):
                    sample_independent_pair(
                        system="vanderpol",
                        evaluation_mode="deterministic",
                        horizon_steps=10,
                        dt=0.02,
                        trajectory_seed=1,
                        time_seed=2,
                        vanderpol_mu=value,
                    )


if __name__ == "__main__":
    unittest.main()
