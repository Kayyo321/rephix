"""Checks for anchored recovery from dense endpoint scores."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from rephix.head_solver import fit_anchored_head


class AnchoredHeadTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(104)
        self.features = rng.normal(size=(24, 4))
        self.heldout = rng.normal(size=(2, 4))
        self.weights = rng.normal(size=(9, 4))
        self.bias = rng.normal(size=9)
        self.anchor_id = 4
        self.logits = self.features @ self.weights.T
        self.heldout_logits = self.heldout @ self.weights.T

    def test_recovers_biased_head_from_log_probs_with_one_factorization(self):
        logits = self.logits + self.bias
        heldout_logits = self.heldout_logits + self.bias
        log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        heldout_log_probs = heldout_logits - np.logaddexp.reduce(
            heldout_logits, axis=1, keepdims=True
        )
        with patch("rephix.head_solver.np.linalg.svd", wraps=np.linalg.svd) as svd:
            weights, bias, metrics = fit_anchored_head(
                self.features,
                log_probs,
                self.heldout,
                heldout_log_probs,
                anchor_id=self.anchor_id,
                anchor_weight=self.weights[self.anchor_id],
                anchor_bias=self.bias[self.anchor_id],
                chunk_size=3,
            )
        svd.assert_called_once()
        np.testing.assert_allclose(weights, self.weights, rtol=0, atol=1e-12)
        np.testing.assert_allclose(bias, self.bias, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(weights[self.anchor_id], self.weights[self.anchor_id])
        self.assertEqual(bias[self.anchor_id], self.bias[self.anchor_id])
        self.assertEqual(metrics["design_rank"], 5)
        self.assertEqual(metrics["validation_samples"], 2)
        self.assertLess(metrics["training_contrast_rmse"], 1e-12)
        self.assertLess(metrics["validation_contrast_rmse"], 1e-12)
        json.dumps(metrics, allow_nan=False)

    def test_recovers_bias_free_head_despite_sample_wise_score_shifts(self):
        logits = self.logits + 4 * self.features[:, :1] ** 2
        heldout_logits = self.heldout_logits + 11 * self.heldout[:, :1] ** 2
        weights, bias, metrics = fit_anchored_head(
            self.features,
            logits.astype(np.float32),
            self.heldout,
            heldout_logits.astype(np.float32),
            anchor_id=np.uint64(self.anchor_id),
            anchor_weight=self.weights[self.anchor_id],
            chunk_size=np.uint64(2),
        )
        np.testing.assert_allclose(weights, self.weights, rtol=0, atol=1e-6)
        self.assertIsNone(bias)
        self.assertEqual(metrics["design_rank"], 4)

    def test_given_anchor_sets_gauge_even_when_it_is_not_the_original_row(self):
        shift = np.array([1.0, -2.0, 3.0, 1.0])
        weights, _, metrics = fit_anchored_head(
            self.features,
            self.logits,
            self.heldout,
            self.heldout_logits,
            anchor_id=self.anchor_id,
            anchor_weight=self.weights[self.anchor_id] + shift,
        )
        np.testing.assert_allclose(weights, self.weights + shift, atol=1e-12)
        self.assertLess(metrics["validation_contrast_rmse"], 1e-12)

    def test_rejects_rank_deficiency(self):
        constant = self.features.copy()
        constant[:, -1] = 1
        cases = (
            (self.features[:3], self.logits[:3], None),
            (constant, self.logits, 0.0),
        )
        for features, logits, anchor_bias in cases:
            with self.subTest(features=features.shape, anchor_bias=anchor_bias):
                with self.assertRaisesRegex(ValueError, "rank deficient"):
                    fit_anchored_head(
                        features,
                        logits,
                        self.heldout,
                        self.heldout_logits,
                        anchor_id=self.anchor_id,
                        anchor_weight=self.weights[self.anchor_id],
                        anchor_bias=anchor_bias,
                    )

    def test_rejects_rounding_noise_disguised_as_a_feature_direction(self):
        features = self.features.astype(np.float32)
        features[:, -1] = features[:, :3].sum(axis=1)
        self.assertEqual(np.linalg.matrix_rank(features.astype(np.float64)), 4)
        with self.assertRaisesRegex(ValueError, "rank deficient: rank 3 of 4"):
            fit_anchored_head(
                features,
                features @ self.weights.T,
                self.heldout,
                self.heldout_logits,
                anchor_id=self.anchor_id,
                anchor_weight=self.weights[self.anchor_id],
            )

    def test_validation_rejects_incompatible_scores_and_measures_nonanchor_rmse(self):
        corrupted = self.heldout_logits.copy()
        corrupted[1, -1] += 1.0
        args = self.features, self.logits, self.heldout, corrupted
        kwargs = {"anchor_id": self.anchor_id, "anchor_weight": self.weights[self.anchor_id]}
        with self.assertRaisesRegex(ValueError, "held-out contrast RMSE .* exceeds"):
            fit_anchored_head(*args, **kwargs)
        _, _, metrics = fit_anchored_head(*args, **kwargs, max_validation_rmse=0.3)
        self.assertAlmostEqual(metrics["validation_contrast_rmse"], 0.25)

    def test_invalid_inputs(self):
        features = self.features.copy()
        features[0, 0] = np.nan
        invalid_scores = self.logits.copy()
        invalid_scores[-1, -1] = np.inf
        invalid_heldout_scores = self.heldout_logits.copy()
        invalid_heldout_scores[0, self.anchor_id] = -np.inf
        base = {
            "features": self.features,
            "scores": self.logits,
            "validation_features": self.heldout,
            "validation_scores": self.heldout_logits,
            "anchor_id": self.anchor_id,
            "anchor_weight": self.weights[self.anchor_id],
        }
        for changes in (
            {"features": features},
            {"features": self.features.ravel()},
            {"scores": self.logits[:-1]},
            {"scores": self.logits[:, :1]},
            {"scores": self.logits.astype(complex)},
            {"scores": invalid_scores},
            {"validation_features": self.heldout[:0]},
            {"validation_features": self.heldout[:, :-1]},
            {"validation_scores": self.heldout_logits[:, :-1]},
            {"validation_scores": invalid_heldout_scores},
            {"anchor_weight": [0, 0, 0, np.nan]},
            {"anchor_weight": [[0, 0, 0, 0]]},
            {"anchor_bias": np.inf},
            {"anchor_id": -1},
            {"anchor_id": self.weights.shape[0]},
            {"anchor_id": True},
            {"chunk_size": 0},
            {"chunk_size": 1.5},
            {"max_validation_rmse": np.inf},
            {"max_validation_rmse": -1.0},
            {"max_validation_rmse": False},
        ):
            with self.subTest(argument=next(iter(changes))):
                with self.assertRaises(ValueError):
                    fit_anchored_head(**(base | changes))


if __name__ == "__main__":
    unittest.main()
