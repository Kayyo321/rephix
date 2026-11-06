"""Identification and generalization checks for the affine baseline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.linear import demo, recover


class LinearRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.x = self.rng.normal(size=(80, 4))
        self.w = self.rng.normal(size=(4, 3))
        self.b = self.rng.normal(size=3)
        self.logits = self.x @ self.w + self.b

    def test_recovers_parameters_and_unseen_logits(self):
        result = recover(self.x, self.logits)
        np.testing.assert_allclose(result.weights, self.w, atol=1e-12)
        np.testing.assert_allclose(result.bias, self.b, atol=1e-12)
        unseen = self.rng.normal(size=(20, 4))
        np.testing.assert_allclose(result.predict(unseen), unseen @ self.w + self.b, atol=1e-12)
        self.assertTrue(result.diagnostics["identifiable_in_reported_gauge"])

    def test_probability_recovery_respects_softmax_gauge(self):
        logs = self.logits - np.logaddexp.reduce(self.logits, axis=1, keepdims=True)
        result = recover(self.x, np.exp(logs), kind="probabilities")
        np.testing.assert_allclose(
            result.weights, self.w - self.w.mean(axis=1, keepdims=True), atol=1e-12
        )
        np.testing.assert_allclose(result.bias, self.b - self.b.mean(), atol=1e-12)
        shared_w = self.rng.normal(size=4)
        shifted = self.logits + (self.x @ shared_w + 7)[:, None]
        shifted_logs = shifted - np.logaddexp.reduce(shifted, axis=1, keepdims=True)
        other = recover(self.x, shifted_logs, kind="log_probs")
        np.testing.assert_allclose(other.weights, result.weights, atol=1e-12)
        np.testing.assert_allclose(other.bias, result.bias, atol=1e-12)
        self.assertEqual(result.diagnostics["parameter_gauge"], "zero_class_mean")

    def test_rank_deficiency_is_not_claimed_as_weight_recovery(self):
        x = np.column_stack((self.x[:, 0], self.x[:, 0]))
        y = x @ np.array([[2, 3], [4, -1]]) + np.array([1, -2])
        for ridge in (0, 0.01):
            result = recover(x, y, ridge=ridge)
            self.assertFalse(result.diagnostics["identifiable_in_reported_gauge"])
            self.assertIsNone(result.diagnostics["condition_number"])
            self.assertEqual(result.diagnostics["design_rank"], 2)
        np.testing.assert_allclose(recover(x, y).predict(x), y, atol=1e-12)

    def test_underdetermined_samples_are_reported(self):
        result = recover(self.x[:2], self.logits[:2])
        self.assertFalse(result.diagnostics["identifiable_in_reported_gauge"])
        self.assertEqual(result.diagnostics["design_rank"], 2)

    def test_noise_is_fitted_without_claiming_exact_recovery(self):
        noisy = self.logits + self.rng.normal(scale=0.05, size=self.logits.shape)
        result = recover(self.x, noisy)
        self.assertGreater(result.diagnostics["training_target_rmse"], 0.01)
        unseen = self.rng.normal(size=(100, 4))
        self.assertLess(
            np.sqrt(np.mean((result.predict(unseen) - unseen @ self.w - self.b) ** 2)), 0.04
        )

    def test_ridge_does_not_regularize_intercept(self):
        x = self.x - self.x.mean(axis=0)
        result = recover(x, x @ self.w + self.b, ridge=1e9)
        np.testing.assert_allclose(result.bias, self.b, atol=1e-8)
        self.assertLess(np.linalg.norm(result.weights), 1e-5)

    def test_invalid_inputs(self):
        cases = [
            ((self.x[:-1], self.logits), {}),
            ((self.x, np.full_like(self.logits, np.nan)), {}),
            ((self.x.ravel(), self.logits), {}),
            ((self.x, self.logits[:, :1]), {}),
            ((self.x, self.logits), {"kind": "topk"}),
            ((self.x, self.logits), {"ridge": -1}),
            ((self.x, self.logits), {"ridge": float("inf")}),
            ((self.x, self.logits), {"ridge": True}),
            ((self.x, np.ones_like(self.logits)), {"kind": "probabilities"}),
            ((self.x, np.zeros_like(self.logits)), {"kind": "probabilities"}),
            ((self.x, np.zeros_like(self.logits)), {"kind": "log_probs"}),
        ]
        for args, kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    recover(*args, **kwargs)

    def test_pickle_free_serialization(self):
        result = recover(self.x, self.logits)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "recovered.npz"
            result.save(target)
            with np.load(target, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["weights"], result.weights)
                self.assertEqual(json.loads(str(archive["diagnostics"])), result.diagnostics)

    def test_demo_has_real_holdout_and_precise_recovery(self):
        result = demo()
        self.assertEqual(result["test_samples"], 64)
        for kind in ("full_logits", "log_probabilities"):
            self.assertLess(result[kind]["parameter_relative_error"], 1e-12)
            self.assertLess(result[kind]["heldout_logit_rmse"], 1e-12)
            self.assertEqual(result[kind]["heldout_top1_accuracy"], 1.0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
