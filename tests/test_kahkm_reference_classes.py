"""Deterministic tests for the frozen out-of-sample reference-class map.

These tests exercise only nearest stored K-means-center assignment. They do
not fit a KAHM, draw evaluation data, compute certificates, or modify the
historical KAHKM implementation.
"""

from dataclasses import FrozenInstanceError
import math
import unittest

import numpy as np

from kahkm_reference_classes import (
    REFERENCE_CLASS_MAP_ID,
    ReferenceClassResult,
    reference_class_labels,
)


def brute_force(centers, states):
    """Independent loop-based Euclidean squared-distance reference."""
    centers = np.asarray(centers, dtype=float)
    states = np.asarray(states, dtype=float)

    labels = []
    distances = []

    for column in range(states.shape[1]):
        values = []

        for cluster in range(centers.shape[1]):
            difference = (
                states[:, column]
                - centers[:, cluster]
            )

            values.append(
                float(
                    sum(
                        value * value
                        for value in difference
                    )
                )
            )

        label = min(
            range(len(values)),
            key=lambda index: values[index],
        )

        labels.append(label)
        distances.append(values[label])

    return tuple(labels), tuple(distances)


class ReferenceClassMapTests(unittest.TestCase):
    def model(self, centers=None):
        if centers is None:
            centers = np.array([
                [-1.0, 0.0, 2.0],
                [ 0.0, 2.0, 0.0],
            ])

        return {
            "cluster_centers":
                np.asarray(
                    centers,
                    dtype=float,
                )
        }

    def evaluate(self, X, centers=None):
        return reference_class_labels(
            abstraction_model=self.model(
                centers
            ),
            X=X,
        )

    # 1
    def test_basic_assignments(self):
        states = np.array([
            [-0.9, 0.1, 1.8],
            [ 0.1, 1.8, 0.2],
        ])

        result = self.evaluate(states)

        self.assertEqual(
            result.labels,
            (0, 1, 2),
        )

        self.assertEqual(
            result.class_map_id,
            REFERENCE_CLASS_MAP_ID,
        )

    # 2
    def test_matches_independent_brute_force_reference(self):
        centers = np.array([
            [-2.0, -0.5, 1.0, 3.0],
            [ 1.0,  2.0, 0.0, 1.5],
        ])

        states = np.array([
            [-1.8, -0.2, 0.9, 2.8, 0.0],
            [ 0.7,  1.7, 0.3, 1.0, 1.0],
        ])

        result = self.evaluate(
            states,
            centers,
        )

        labels, distances = brute_force(
            centers,
            states,
        )

        self.assertEqual(
            result.labels,
            labels,
        )

        np.testing.assert_allclose(
            result.selected_squared_distances,
            distances,
            rtol=1e-14,
            atol=1e-15,
        )

    # 3
    def test_exact_centers_have_zero_distance(self):
        centers = np.array([
            [-1.0, 0.0, 2.0],
            [ 0.0, 2.0, 0.0],
        ])

        result = self.evaluate(
            centers.copy(),
            centers,
        )

        self.assertEqual(
            result.labels,
            (0, 1, 2),
        )

        self.assertEqual(
            result.selected_squared_distances,
            (0.0, 0.0, 0.0),
        )

    # 4
    def test_tie_is_resolved_by_smallest_class_index(self):
        centers = np.array([
            [-1.0, 1.0],
            [ 0.0, 0.0],
        ])

        state = np.array([
            [0.0],
            [0.0],
        ])

        result = self.evaluate(
            state,
            centers,
        )

        self.assertEqual(
            result.labels,
            (0,),
        )

        self.assertEqual(
            result.selected_squared_distances,
            (1.0,),
        )

    # 5
    def test_multiple_ties_use_smallest_index(self):
        centers = np.array([
            [-1.0, 1.0, 0.0],
            [ 0.0, 0.0, 1.0],
        ])

        state = np.array([
            [0.0],
            [0.0],
        ])

        result = self.evaluate(
            state,
            centers,
        )

        self.assertEqual(
            result.labels,
            (0,),
        )

    # 6
    def test_single_class_is_supported(self):
        centers = np.array([
            [2.0],
            [-1.0],
        ])

        states = np.array([
            [2.0, 3.0, -5.0],
            [-1.0, 0.0, 10.0],
        ])

        result = self.evaluate(
            states,
            centers,
        )

        self.assertEqual(
            result.labels,
            (0, 0, 0),
        )

        labels, distances = brute_force(
            centers,
            states,
        )

        self.assertEqual(
            result.labels,
            labels,
        )

        np.testing.assert_allclose(
            result.selected_squared_distances,
            distances,
        )

    # 7
    def test_single_state_is_supported(self):
        result = self.evaluate(
            np.array([
                [1.9],
                [0.1],
            ])
        )

        self.assertEqual(
            result.n_states,
            1,
        )

        self.assertEqual(
            len(result.labels),
            1,
        )

    # 8
    def test_dimension_metadata(self):
        result = self.evaluate(
            np.array([
                [-1.0, 0.0],
                [ 0.0, 2.0],
            ])
        )

        self.assertEqual(
            result.state_dimension,
            2,
        )

        self.assertEqual(
            result.n_classes,
            3,
        )

        self.assertEqual(
            result.n_states,
            2,
        )

    # 9
    def test_translation_invariance(self):
        centers = np.array([
            [-2.0, 0.0, 3.0],
            [ 1.0, 2.0, 0.0],
        ])

        states = np.array([
            [-1.5, 0.2, 2.8],
            [ 0.7, 1.8, 0.1],
        ])

        shift = np.array([
            [100.0],
            [-37.0],
        ])

        first = self.evaluate(
            states,
            centers,
        )

        second = self.evaluate(
            states + shift,
            centers + shift,
        )

        self.assertEqual(
            first.labels,
            second.labels,
        )

        np.testing.assert_allclose(
            first.selected_squared_distances,
            second.selected_squared_distances,
            rtol=1e-12,
            atol=1e-12,
        )

    # 10
    def test_positive_uniform_scaling_preserves_labels(self):
        centers = np.array([
            [-2.0, 0.0, 3.0],
            [ 1.0, 2.0, 0.0],
        ])

        states = np.array([
            [-1.5, 0.2, 2.8],
            [ 0.7, 1.8, 0.1],
        ])

        factor = 7.5

        first = self.evaluate(
            states,
            centers,
        )

        second = self.evaluate(
            factor * states,
            factor * centers,
        )

        self.assertEqual(
            first.labels,
            second.labels,
        )

        np.testing.assert_allclose(
            second.selected_squared_distances,
            factor * factor
            * np.asarray(
                first.selected_squared_distances
            ),
            rtol=1e-13,
            atol=1e-13,
        )

    # 11
    def test_row_state_orientation_is_not_silently_accepted(self):
        states_as_rows = np.array([
            [-0.9, 0.1],
            [ 0.1, 1.8],
            [ 1.8, 0.2],
        ])

        with self.assertRaises(ValueError):
            self.evaluate(
                states_as_rows
            )

    # 12
    def test_dimension_mismatch_is_rejected(self):
        states = np.ones(
            (3, 4),
            dtype=float,
        )

        with self.assertRaises(ValueError):
            self.evaluate(states)

    # 13
    def test_missing_cluster_centers_is_rejected(self):
        with self.assertRaises(ValueError):
            reference_class_labels(
                abstraction_model={},
                X=np.ones((2, 1)),
            )

    # 14
    def test_non_dictionary_model_is_rejected(self):
        for model in (
            None,
            [],
            np.eye(2),
            "model",
        ):
            with self.subTest(model=repr(model)):
                with self.assertRaises(TypeError):
                    reference_class_labels(
                        abstraction_model=model,
                        X=np.ones((2, 1)),
                    )

    # 15
    def test_invalid_matrix_shapes_are_rejected(self):
        invalid_states = (
            np.array([1.0, 2.0]),
            np.empty((0, 3)),
            np.empty((2, 0)),
            np.ones((2, 2, 1)),
        )

        for states in invalid_states:
            with self.subTest(
                shape=states.shape
            ):
                with self.assertRaises(ValueError):
                    self.evaluate(states)

    # 16
    def test_invalid_numeric_types_are_rejected(self):
        invalid = (
            np.array([[True], [False]]),
            np.array([["1"], ["2"]]),
            np.array([[1 + 0j], [2 + 0j]]),
            np.array([[object()], [object()]], dtype=object),
        )

        for states in invalid:
            with self.subTest(
                dtype=str(states.dtype)
            ):
                with self.assertRaises(TypeError):
                    self.evaluate(states)

    # 17
    def test_nonfinite_states_are_rejected(self):
        for bad in (
            math.nan,
            math.inf,
            -math.inf,
        ):
            states = np.array([
                [bad],
                [0.0],
            ])

            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.evaluate(states)

    # 18
    def test_nonfinite_centers_are_rejected(self):
        for bad in (
            math.nan,
            math.inf,
            -math.inf,
        ):
            centers = np.array([
                [-1.0, bad],
                [ 0.0, 1.0],
            ])

            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.evaluate(
                        np.ones((2, 1)),
                        centers,
                    )

    # 19
    def test_inputs_are_not_modified(self):
        centers = np.array([
            [-1.0, 0.0, 2.0],
            [ 0.0, 2.0, 0.0],
        ])

        states = np.array([
            [-0.9, 0.1, 1.8],
            [ 0.1, 1.8, 0.2],
        ])

        centers_before = centers.copy()
        states_before = states.copy()

        centers.setflags(
            write=False
        )
        states.setflags(
            write=False
        )

        result = self.evaluate(
            states,
            centers,
        )

        np.testing.assert_array_equal(
            centers,
            centers_before,
        )

        np.testing.assert_array_equal(
            states,
            states_before,
        )

        self.assertEqual(
            result.labels,
            (0, 1, 2),
        )

    # 20
    def test_result_is_immutable(self):
        result = self.evaluate(
            np.array([
                [-0.9],
                [ 0.1],
            ])
        )

        self.assertIsInstance(
            result,
            ReferenceClassResult,
        )

        with self.assertRaises(FrozenInstanceError):
            result.n_states = 2

        with self.assertRaises(TypeError):
            result.labels[0] = 1


if __name__ == "__main__":
    unittest.main()
