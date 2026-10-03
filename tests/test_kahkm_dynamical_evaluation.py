"""Tests for deterministic evaluation of frozen dynamical KAHM pairs.

No random pairs are generated and no representation is fitted.
"""

from dataclasses import FrozenInstanceError
import math
import unittest
from unittest import mock

import numpy as np

import kahkm_dynamical_evaluation as module
from kahkm_certificate_statistics import (
    statistics_from_pairs,
)
from kahkm_certificates import (
    certificate_from_statistics,
)
from kahkm_dynamical_evaluation import (
    DYNAMICAL_EVALUATION_ID,
    BudgetCertificate,
    DynamicalEvaluationResult,
    evaluate_frozen_dynamical_pairs,
)
from kahkm_reference_classes import (
    REFERENCE_CLASS_MAP_ID,
)


class DynamicalEvaluationTests(unittest.TestCase):
    def model(self):
        return {
            "cluster_centers": np.array(
                [
                    [-1.0, 1.0],
                    [0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "kahkm_omega": 2.0,
            "kahkm_tau": 1e-6,
        }

    def states(self):
        current = np.array(
            [
                [-0.9, 0.8, -0.2],
                [0.0, 0.0, 0.1],
            ],
            dtype=np.float64,
        )

        successor = np.array(
            [
                [-0.8, 0.7, 0.1],
                [0.1, -0.1, 0.2],
            ],
            dtype=np.float64,
        )

        return current, successor

    def associations(self):
        # KAHM convention: C x M.
        phi = np.array(
            [
                [0.8, 0.1, 0.6],
                [0.2, 0.9, 0.4],
            ],
            dtype=np.float64,
        )

        successors = np.array(
            [
                [0.7, 0.2, 0.3],
                [0.3, 0.8, 0.7],
            ],
            dtype=np.float64,
        )

        return phi, successors

    def operator(self):
        # Deliberately non-symmetric so B.T @ Phi differs from B @ Phi.
        return np.array(
            [
                [1.0, 0.0],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )

    def evaluate(
        self,
        *,
        model=None,
        B=None,
        current=None,
        successor=None,
        kappas=(0.5, 2.0),
        delta=0.05,
        phi=None,
        successor_phi=None,
    ):
        if model is None:
            model = self.model()

        if B is None:
            B = self.operator()

        if current is None or successor is None:
            default_current, default_successor = self.states()

            if current is None:
                current = default_current

            if successor is None:
                successor = default_successor

        if phi is None or successor_phi is None:
            default_phi, default_successor_phi = self.associations()

            if phi is None:
                phi = default_phi

            if successor_phi is None:
                successor_phi = default_successor_phi

        with mock.patch.object(
            module,
            "kahm_associations",
            side_effect=(
                phi,
                successor_phi,
            ),
        ) as association_mock:
            result = evaluate_frozen_dynamical_pairs(
                abstraction_model=model,
                B=B,
                omega=2.0,
                tau=1e-6,
                current_states=current,
                successor_states=successor,
                kappas=kappas,
                delta=delta,
                n_jobs=1,
                batch_size=17,
            )

        return result, association_mock

    # 1
    def test_basic_result_metadata(self):
        result, _ = self.evaluate()

        self.assertIsInstance(
            result,
            DynamicalEvaluationResult,
        )

        self.assertEqual(
            result.evaluation_id,
            DYNAMICAL_EVALUATION_ID,
        )

        self.assertEqual(
            result.reference_class_map_id,
            REFERENCE_CLASS_MAP_ID,
        )

        self.assertEqual(
            result.n_pairs,
            3,
        )

        self.assertEqual(
            result.n_classes,
            2,
        )

        self.assertEqual(
            result.state_dimension,
            2,
        )

        self.assertEqual(
            result.delta,
            0.05,
        )

    # 2
    def test_reference_classes_are_nearest_stored_centers_not_phi_argmax(self):
        # The third current state is closer to hard class 0, while its supplied
        # soft coordinate also happens to favor class 0. Force a soft map whose
        # third column favors class 1 to prove labels do not come from argmax.
        phi = np.array(
            [
                [0.8, 0.1, 0.4],
                [0.2, 0.9, 0.6],
            ],
            dtype=np.float64,
        )

        _, successor_phi = self.associations()

        result, _ = self.evaluate(
            phi=phi,
            successor_phi=successor_phi,
        )

        self.assertEqual(
            result.labels,
            (0, 1, 0),
        )

        self.assertEqual(
            int(
                np.argmax(
                    phi[:, 2]
                )
            ),
            1,
        )

    # 3
    def test_kahm_column_major_coordinates_are_transposed_for_statistics(self):
        phi, successor_phi = self.associations()

        with (
            mock.patch.object(
                module,
                "kahm_associations",
                side_effect=(
                    phi,
                    successor_phi,
                ),
            ),
            mock.patch.object(
                module,
                "statistics_from_pairs",
                wraps=statistics_from_pairs,
            ) as stats_mock,
        ):
            current, successor = self.states()

            evaluate_frozen_dynamical_pairs(
                abstraction_model=self.model(),
                B=self.operator(),
                omega=2.0,
                tau=1e-6,
                current_states=current,
                successor_states=successor,
                kappas=(0.5, 2.0),
                delta=0.05,
            )

        kwargs = stats_mock.call_args.kwargs

        np.testing.assert_array_equal(
            kwargs["phi"],
            phi.T,
        )

        np.testing.assert_array_equal(
            kwargs["successors"],
            successor_phi.T,
        )

        np.testing.assert_array_equal(
            kwargs["labels"],
            np.array(
                [0, 1, 0],
                dtype=np.int64,
            ),
        )

    # 4
    def test_statistics_match_validated_statistics_function(self):
        result, _ = self.evaluate()

        phi, successor_phi = self.associations()

        expected = statistics_from_pairs(
            phi=phi.T,
            successors=successor_phi.T,
            labels=np.array(
                [0, 1, 0],
                dtype=np.int64,
            ),
        )

        self.assertEqual(
            result.statistics,
            expected,
        )

    # 5
    def test_frozen_predictor_uses_B_transpose_orientation(self):
        result, _ = self.evaluate()

        phi, successor_phi = self.associations()
        B = self.operator()

        correct = (
            B.T
            @ phi
        )

        wrong = (
            B
            @ phi
        )

        correct_mse = float(
            np.mean(
                np.sum(
                    (
                        correct
                        - successor_phi
                    )
                    ** 2,
                    axis=0,
                )
            )
        )

        wrong_mse = float(
            np.mean(
                np.sum(
                    (
                        wrong
                        - successor_phi
                    )
                    ** 2,
                    axis=0,
                )
            )
        )

        self.assertAlmostEqual(
            result.frozen_predictor_mse,
            correct_mse,
            places=15,
        )

        self.assertNotAlmostEqual(
            result.frozen_predictor_mse,
            wrong_mse,
            places=10,
        )

    # 6
    def test_rmse_is_square_root_of_vector_mse(self):
        result, _ = self.evaluate()

        self.assertAlmostEqual(
            result.frozen_predictor_rmse,
            math.sqrt(
                result.frozen_predictor_mse
            ),
            places=15,
        )

    # 7
    def test_spectral_norm_is_computed_from_raw_frozen_operator(self):
        result, _ = self.evaluate()

        expected = float(
            np.linalg.norm(
                self.operator(),
                ord=2,
            )
        )

        self.assertAlmostEqual(
            result.frozen_predictor_spectral_norm,
            expected,
            places=15,
        )

    # 8
    def test_each_budget_certificate_matches_validated_certificate_function(self):
        result, _ = self.evaluate()

        self.assertEqual(
            tuple(
                item.kappa
                for item
                in result.budget_certificates
            ),
            (
                0.5,
                2.0,
            ),
        )

        for item in result.budget_certificates:
            expected = certificate_from_statistics(
                f_hat=
                    result.statistics.f_hat,
                s_hat=
                    result.statistics.s_hat,
                n_pairs=
                    result.statistics.n_pairs,
                kappa=
                    item.kappa,
                delta=
                    result.delta,
            )

            self.assertEqual(
                item.certificate,
                expected,
            )

    # 9
    def test_same_empirical_statistics_are_reused_for_all_budgets(self):
        result, _ = self.evaluate(
            kappas=(
                0.0,
                1.0,
                3.0,
            )
        )

        pairs = {
            (
                item.certificate.f_hat,
                item.certificate.s_hat,
                item.certificate.n_pairs,
                item.certificate.delta,
            )
            for item
            in result.budget_certificates
        }

        self.assertEqual(
            len(pairs),
            1,
        )

    # 10
    def test_budget_feasibility_flags_follow_frozen_operator_norm(self):
        result, _ = self.evaluate(
            kappas=(
                0.5,
                2.0,
            )
        )

        norm = (
            result.frozen_predictor_spectral_norm
        )

        self.assertGreater(
            norm,
            0.5,
        )

        self.assertLess(
            norm,
            2.0,
        )

        self.assertIs(
            result.budget_certificates[
                0
            ].frozen_predictor_within_budget,
            False,
        )

        self.assertIs(
            result.budget_certificates[
                1
            ].frozen_predictor_within_budget,
            True,
        )

    # 11
    def test_signed_predictor_minus_bound_gap_is_preserved(self):
        result, _ = self.evaluate()

        for item in result.budget_certificates:
            self.assertAlmostEqual(
                item.frozen_predictor_rmse_minus_bound,
                (
                    result.frozen_predictor_rmse
                    - item.certificate.L_kappa_delta
                ),
                places=15,
            )

    # 12
    def test_reference_distance_diagnostics_are_preserved(self):
        result, _ = self.evaluate()

        current, _ = self.states()

        expected = (
            (current[0, 0] + 1.0) ** 2
            + current[1, 0] ** 2,
            (current[0, 1] - 1.0) ** 2
            + current[1, 1] ** 2,
            (current[0, 2] + 1.0) ** 2
            + current[1, 2] ** 2,
        )

        np.testing.assert_allclose(
            result.selected_center_squared_distances,
            expected,
            rtol=1e-14,
            atol=1e-15,
        )

    # 13
    def test_coordinate_and_state_range_diagnostics(self):
        result, _ = self.evaluate()

        phi, successor_phi = self.associations()
        current, successor = self.states()

        self.assertEqual(
            result.phi_coordinate_min,
            float(
                np.min(phi)
            ),
        )

        self.assertEqual(
            result.phi_coordinate_max,
            float(
                np.max(phi)
            ),
        )

        self.assertEqual(
            result.successor_coordinate_min,
            float(
                np.min(
                    successor_phi
                )
            ),
        )

        self.assertEqual(
            result.successor_coordinate_max,
            float(
                np.max(
                    successor_phi
                )
            ),
        )

        self.assertEqual(
            result.current_state_min,
            tuple(
                float(x)
                for x in np.min(
                    current,
                    axis=1,
                )
            ),
        )

        self.assertEqual(
            result.current_state_max,
            tuple(
                float(x)
                for x in np.max(
                    current,
                    axis=1,
                )
            ),
        )

        self.assertEqual(
            result.successor_state_min,
            tuple(
                float(x)
                for x in np.min(
                    successor,
                    axis=1,
                )
            ),
        )

        self.assertEqual(
            result.successor_state_max,
            tuple(
                float(x)
                for x in np.max(
                    successor,
                    axis=1,
                )
            ),
        )

    # 14
    def test_kahm_association_calls_use_supplied_frozen_settings(self):
        _, association_mock = self.evaluate()

        self.assertEqual(
            association_mock.call_count,
            2,
        )

        for call in association_mock.call_args_list:
            self.assertEqual(
                call.kwargs["omega"],
                2.0,
            )

            self.assertEqual(
                call.kwargs["tau"],
                1e-6,
            )

            self.assertEqual(
                call.kwargs["n_jobs"],
                1,
            )

            self.assertEqual(
                call.kwargs["batch_size"],
                17,
            )

            self.assertIs(
                call.kwargs["show_progress"],
                False,
            )

    # 15
    def test_model_omega_or_tau_mismatch_is_rejected_before_association_evaluation(self):
        for key, value in (
            (
                "kahkm_omega",
                4.0,
            ),
            (
                "kahkm_tau",
                1e-5,
            ),
        ):
            model = self.model()
            model[key] = value

            current, successor = self.states()

            with (
                self.subTest(
                    key=key
                ),
                mock.patch.object(
                    module,
                    "kahm_associations",
                ) as association_mock,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "does not equal supplied",
                ):
                    evaluate_frozen_dynamical_pairs(
                        abstraction_model=model,
                        B=self.operator(),
                        omega=2.0,
                        tau=1e-6,
                        current_states=current,
                        successor_states=successor,
                        kappas=(1.0,),
                        delta=0.05,
                    )

                association_mock.assert_not_called()

    # 16
    def test_invalid_physical_state_inputs_are_rejected(self):
        current, successor = self.states()

        cases = (
            (
                current[:, :2],
                successor,
                ValueError,
            ),
            (
                np.array(
                    [1.0, 2.0]
                ),
                successor,
                ValueError,
            ),
            (
                np.array(
                    [
                        [np.nan],
                        [0.0],
                    ]
                ),
                np.array(
                    [
                        [0.0],
                        [0.0],
                    ]
                ),
                ValueError,
            ),
            (
                np.array(
                    [
                        [True],
                        [False],
                    ]
                ),
                np.array(
                    [
                        [0.0],
                        [0.0],
                    ]
                ),
                TypeError,
            ),
        )

        for current_value, successor_value, expected in cases:
            with self.subTest(
                expected=expected.__name__
            ):
                with self.assertRaises(
                    expected
                ):
                    evaluate_frozen_dynamical_pairs(
                        abstraction_model=self.model(),
                        B=self.operator(),
                        omega=2.0,
                        tau=1e-6,
                        current_states=current_value,
                        successor_states=successor_value,
                        kappas=(1.0,),
                        delta=0.05,
                    )

    # 17
    def test_invalid_kappa_sequences_are_rejected(self):
        current, successor = self.states()

        invalid = (
            (),
            (
                2.0,
                1.0,
            ),
            (
                1.0,
                1.0,
            ),
            (
                -1.0,
            ),
            "1.0",
        )

        for value in invalid:
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    (
                        TypeError,
                        ValueError,
                    )
                ):
                    evaluate_frozen_dynamical_pairs(
                        abstraction_model=self.model(),
                        B=self.operator(),
                        omega=2.0,
                        tau=1e-6,
                        current_states=current,
                        successor_states=successor,
                        kappas=value,
                        delta=0.05,
                    )

    # 18
    def test_bad_operator_shape_is_rejected(self):
        current, successor = self.states()
        phi, successor_phi = self.associations()

        with mock.patch.object(
            module,
            "kahm_associations",
            side_effect=(
                phi,
                successor_phi,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "shape",
            ):
                evaluate_frozen_dynamical_pairs(
                    abstraction_model=self.model(),
                    B=np.eye(3),
                    omega=2.0,
                    tau=1e-6,
                    current_states=current,
                    successor_states=successor,
                    kappas=(1.0,),
                    delta=0.05,
                )

    # 19
    def test_bad_association_outputs_are_rejected(self):
        current, successor = self.states()
        good_phi, good_successor = self.associations()

        cases = (
            (
                good_phi[:, :2],
                good_successor[:, :2],
                "pair count",
            ),
            (
                good_phi,
                np.ones(
                    (
                        3,
                        3,
                    )
                ),
                "shapes differ",
            ),
            (
                good_phi.copy(),
                good_successor.copy(),
                "nonfinite",
            ),
        )

        cases[2][0][0, 0] = np.nan

        for phi_value, successor_value, message in cases:
            with (
                self.subTest(
                    message=message
                ),
                mock.patch.object(
                    module,
                    "kahm_associations",
                    side_effect=(
                        phi_value,
                        successor_value,
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    message,
                ):
                    evaluate_frozen_dynamical_pairs(
                        abstraction_model=self.model(),
                        B=self.operator(),
                        omega=2.0,
                        tau=1e-6,
                        current_states=current,
                        successor_states=successor,
                        kappas=(1.0,),
                        delta=0.05,
                    )

    def test_retained_association_coordinates_are_exact_row_major_and_read_only(self):
        phi, successor_phi = self.associations()

        result, _ = self.evaluate(
            phi=phi,
            successor_phi=successor_phi,
        )

        self.assertEqual(
            result.phi_rows.shape,
            (3, 2),
        )

        self.assertEqual(
            result.successor_rows.shape,
            (3, 2),
        )

        np.testing.assert_array_equal(
            result.phi_rows,
            phi.T,
        )

        np.testing.assert_array_equal(
            result.successor_rows,
            successor_phi.T,
        )

        self.assertFalse(
            result.phi_rows.flags.writeable
        )

        self.assertFalse(
            result.successor_rows.flags.writeable
        )

        with self.assertRaises(
            ValueError
        ):
            result.phi_rows[
                0,
                0,
            ] = 0.0

        with self.assertRaises(
            ValueError
        ):
            result.successor_rows[
                0,
                0,
            ] = 0.0

    # 21
    def test_results_are_immutable_and_evaluation_does_not_mutate_inputs(self):
        model = self.model()
        B = self.operator()
        current, successor = self.states()
        phi, successor_phi = self.associations()

        B_before = B.copy()
        current_before = current.copy()
        successor_before = successor.copy()

        B.setflags(
            write=False
        )
        current.setflags(
            write=False
        )
        successor.setflags(
            write=False
        )

        with mock.patch.object(
            module,
            "kahm_associations",
            side_effect=(
                phi,
                successor_phi,
            ),
        ):
            result = evaluate_frozen_dynamical_pairs(
                abstraction_model=model,
                B=B,
                omega=2.0,
                tau=1e-6,
                current_states=current,
                successor_states=successor,
                kappas=(0.5, 2.0),
                delta=0.05,
            )

        np.testing.assert_array_equal(
            B,
            B_before,
        )

        np.testing.assert_array_equal(
            current,
            current_before,
        )

        np.testing.assert_array_equal(
            successor,
            successor_before,
        )

        with self.assertRaises(
            FrozenInstanceError
        ):
            result.n_pairs = 4

        self.assertIsInstance(
            result.budget_certificates[0],
            BudgetCertificate,
        )

        with self.assertRaises(
            FrozenInstanceError
        ):
            result.budget_certificates[0].kappa = 1.0


if __name__ == "__main__":
    unittest.main()
