"""Known-symmetry alignment and limits of finite observations."""

import itertools
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from rephix.alignment import (
    _run_demo,
    align_hidden,
    forward,
    minimum_assignment,
    symmetry_demo,
)


class AssignmentTests(unittest.TestCase):
    def test_matches_exhaustive_minimum_for_small_matrices(self):
        rng = np.random.default_rng(18)
        for size in range(1, 7):
            costs = rng.normal(size=(size, size))
            assignment = minimum_assignment(costs)
            achieved = costs[np.arange(size), assignment].sum()
            exact = min(
                costs[np.arange(size), permutation].sum()
                for permutation in itertools.permutations(range(size))
            )
            with self.subTest(size=size):
                self.assertAlmostEqual(achieved, exact, places=12)
                self.assertEqual(sorted(assignment.tolist()), list(range(size)))

    def test_equal_cost_matching_is_deterministic(self):
        first = minimum_assignment(np.zeros((4, 4)))
        np.testing.assert_array_equal(first, minimum_assignment(np.zeros((4, 4))))
        np.testing.assert_array_equal(first, np.arange(4))

    def test_invalid_cost_matrices_are_rejected(self):
        for costs in (np.zeros((2, 3)), np.array([[np.nan]]), np.empty((0, 0))):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                minimum_assignment(costs)


class HiddenAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(92)
        self.incoming = self.rng.normal(size=(4, 7))
        self.bias = self.rng.normal(size=7)
        self.outgoing = self.rng.normal(size=(7, 3))
        self.permutation = self.rng.permutation(7)
        self.features = self.rng.normal(size=(40, 4))

    def transform(self, scales):
        return (
            self.incoming[:, self.permutation] * scales,
            self.bias[self.permutation] * scales,
            self.outgoing[self.permutation] / scales[:, None],
        )

    def test_relu_permutation_and_positive_scales_recover_reference_gauge(self):
        scales = np.exp(self.rng.uniform(-3, 3, size=7))
        candidate = self.transform(scales)
        result = align_hidden(self.incoming, self.bias, self.outgoing, *candidate)
        np.testing.assert_array_equal(result.permutation, np.argsort(self.permutation))
        np.testing.assert_allclose(result.incoming_weights, self.incoming, atol=1e-12)
        np.testing.assert_allclose(result.hidden_bias, self.bias, atol=1e-12)
        np.testing.assert_allclose(result.outgoing_weights, self.outgoing, atol=1e-12)
        self.assertGreater(result.diagnostics["raw_parameter_relative_l2"], 0.1)
        self.assertLess(result.diagnostics["aligned_parameter_relative_l2"], 1e-12)
        original = forward(self.features, *candidate)
        aligned = forward(
            self.features, result.incoming_weights, result.hidden_bias, result.outgoing_weights
        )
        np.testing.assert_allclose(aligned, original, atol=1e-12)

    def test_tanh_allows_sign_but_not_arbitrary_positive_scale(self):
        signs = self.rng.choice([-1.0, 1.0], size=7)
        candidate = self.transform(signs)
        result = align_hidden(
            self.incoming, self.bias, self.outgoing, *candidate, activation="tanh"
        )
        self.assertLess(result.diagnostics["aligned_parameter_relative_l2"], 1e-12)
        np.testing.assert_allclose(
            forward(self.features, self.incoming, self.bias, self.outgoing, activation="tanh"),
            forward(self.features, *candidate, activation="tanh"),
            atol=1e-12,
        )
        scaled_candidate = self.transform(np.full(7, 2.0))
        other = align_hidden(
            self.incoming, self.bias, self.outgoing, *scaled_candidate, activation="tanh"
        )
        self.assertGreater(other.diagnostics["aligned_parameter_relative_l2"], 0.1)
        self.assertGreater(
            np.max(
                np.abs(
                    forward(
                        self.features, self.incoming, self.bias, self.outgoing, activation="tanh"
                    )
                    - forward(self.features, *scaled_candidate, activation="tanh")
                )
            ),
            0.1,
        )

    def test_extreme_finite_positive_scales_do_not_become_zero_or_infinite_norms(self):
        for scale in (1e-200, 1e200):
            candidate = self.transform(np.full(7, scale))
            with self.subTest(scale=scale):
                result = align_hidden(self.incoming, self.bias, self.outgoing, *candidate)
                self.assertEqual(result.diagnostics["candidate_zero_incoming_neurons"], 0)
                self.assertLess(result.diagnostics["aligned_parameter_relative_l2"], 1e-12)
                json.dumps(result.to_dict(), allow_nan=False)

    def test_changed_network_does_not_become_exact_after_alignment(self):
        candidate = list(self.transform(np.ones(7)))
        candidate[2] = candidate[2] + 0.5
        result = align_hidden(self.incoming, self.bias, self.outgoing, *candidate)
        self.assertGreater(result.diagnostics["aligned_parameter_relative_l2"], 0.01)
        self.assertGreater(
            np.max(
                np.abs(
                    forward(self.features, self.incoming, self.bias, self.outgoing)
                    - forward(self.features, *candidate)
                )
            ),
            0.1,
        )

    def test_zero_incoming_neurons_are_finite_and_remain_unidentifiable(self):
        reference = (
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([0.0, 0.0]),
            np.array([[1.0], [2.0]]),
        )
        candidate = (reference[0][:, ::-1], reference[1][::-1], np.array([[999.0], [1.0]]))
        result = align_hidden(*reference, *candidate)
        self.assertEqual(result.diagnostics["reference_zero_incoming_neurons"], 1)
        self.assertEqual(result.diagnostics["candidate_zero_incoming_neurons"], 1)
        self.assertGreater(result.diagnostics["aligned_parameter_l2"], 100)
        features = np.array([[1.0, 3.0], [-1.0, 2.0]])
        np.testing.assert_array_equal(forward(features, *reference), forward(features, *candidate))
        json.dumps(result.to_dict(), allow_nan=False)

    def test_all_zero_network_has_no_relative_error_denominator(self):
        network = (np.zeros((2, 3)), np.zeros(3), np.zeros((3, 1)))
        result = align_hidden(*network, *network)
        self.assertIsNone(result.diagnostics["aligned_parameter_relative_l2"])
        self.assertEqual(result.diagnostics["aligned_parameter_l2"], 0)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_invalid_networks_are_rejected(self):
        valid = (self.incoming, self.bias, self.outgoing)
        for candidate in (
            (self.incoming, self.bias[:-1], self.outgoing),
            (self.incoming, self.bias, self.outgoing[:, :1]),
            (np.full_like(self.incoming, np.inf), self.bias, self.outgoing),
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                align_hidden(*valid, *candidate)
        with self.assertRaises(ValueError):
            align_hidden(*valid, *valid, activation="sigmoid")

    def test_demo_reports_symmetry_and_finite_observation_ambiguity_separately(self):
        for activation in ("relu", "tanh"):
            report = symmetry_demo(activation=activation)
            with self.subTest(activation=activation):
                self.assertLess(
                    report["alignment"]["diagnostics"]["aligned_parameter_relative_l2"], 1e-12
                )
                self.assertLess(report["heldout_inputs"]["raw_function_max_abs_error"], 1e-12)
                self.assertEqual(
                    report["finite_observation_counterexample"]["max_observed_output_difference"], 0
                )
                self.assertEqual(
                    report["finite_observation_counterexample"]["unseen_output_difference"], 1
                )
                json.dumps(report, allow_nan=False)

    def test_cli_handler_saves_a_reproducible_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "symmetry.json"
            result = _run_demo(Namespace(seed=7, width=5, activation="relu", output=str(path)))
            self.assertEqual(json.loads(path.read_text()), result)


if __name__ == "__main__":
    unittest.main()
