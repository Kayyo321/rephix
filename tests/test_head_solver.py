"""Checks for affine head recovery from dense endpoint scores."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from rephix.head_solver import fit_anchored_head, fit_unanchored_head


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


class UnanchoredHeadTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(121)
        self.features = rng.normal(size=(24, 4))
        self.heldout = rng.normal(size=(4, 4))
        self.base_weights = rng.normal(size=(9, 4)).astype(np.float32)
        self.base_bias = rng.normal(size=9).astype(np.float32)

    def test_raw_logits_recover_the_absolute_gauge(self):
        weights = self.base_weights + [1, -2, 3, 4]
        bias = self.base_bias.astype(np.float64) + 5
        with patch("rephix.head_solver.np.linalg.svd", wraps=np.linalg.svd) as svd:
            fitted, fitted_bias, metrics = fit_unanchored_head(
                self.features,
                self.features @ weights.T + bias,
                self.heldout,
                self.heldout @ weights.T + bias,
                mode="raw_logits",
                has_bias=True,
                chunk_size=3,
            )
        svd.assert_called_once()
        np.testing.assert_allclose(fitted, weights, rtol=0, atol=1e-12)
        np.testing.assert_allclose(fitted_bias, bias, rtol=0, atol=1e-12)
        self.assertEqual(metrics["fit_mode"], "raw_logits")
        self.assertLess(metrics["validation_rmse"], 1e-12)
        self.assertNotIn("validation_contrast_rmse", metrics)
        json.dumps(metrics, allow_nan=False)

    def test_raw_validation_detects_shared_offset(self):
        scores = self.features @ self.base_weights.T
        heldout_scores = self.heldout @ self.base_weights.T
        heldout_scores[0] += 1
        args = self.features, scores, self.heldout, heldout_scores
        with self.assertRaisesRegex(ValueError, "held-out absolute RMSE .* exceeds"):
            fit_unanchored_head(*args, mode="raw_logits", chunk_size=2)
        _, bias, metrics = fit_unanchored_head(
            *args, mode="raw_logits", chunk_size=2, max_validation_rmse=0.6
        )
        self.assertIsNone(bias)
        self.assertAlmostEqual(metrics["validation_rmse"], 0.5)

    def test_median_recovers_head_with_a_changed_reference_row(self):
        weights = self.base_weights.astype(np.float64)
        weights[[0, 2, 5, 7]] += np.arange(16).reshape(4, 4) + 1
        bias = self.base_bias.astype(np.float64)
        bias[[0, 2, 5, 7]] -= [1, 2, 3, 4]
        for has_bias in (False, True):
            with self.subTest(has_bias=has_bias):
                logits = self.features @ weights.T
                heldout_logits = self.heldout @ weights.T
                if has_bias:
                    logits += bias
                    heldout_logits += bias
                fitted, fitted_bias, metrics = fit_unanchored_head(
                    self.features,
                    logits - np.logaddexp.reduce(logits, axis=1, keepdims=True),
                    self.heldout,
                    heldout_logits - np.logaddexp.reduce(heldout_logits, axis=1, keepdims=True),
                    mode="base_median",
                    has_bias=has_bias,
                    base_weight=self.base_weights,
                    base_bias=self.base_bias if has_bias else None,
                    chunk_size=2,
                )
                np.testing.assert_allclose(fitted, weights, rtol=0, atol=1e-12)
                if has_bias:
                    np.testing.assert_allclose(fitted_bias, bias, rtol=0, atol=1e-12)
                else:
                    self.assertIsNone(fitted_bias)
                self.assertEqual(metrics["fit_mode"], "base_median")
                self.assertEqual(metrics["reference_id"], 0)
                self.assertEqual(metrics["validation_rmse"], metrics["validation_contrast_rmse"])
                self.assertLess(metrics["validation_rmse"], 1e-12)
                json.dumps(metrics, allow_nan=False)

    def test_median_cannot_detect_a_wrong_prior_with_a_shared_shift(self):
        weights = self.base_weights + [1, -2, 3, 4]
        bias = self.base_bias.astype(np.float64) + 5
        fitted, fitted_bias, metrics = fit_unanchored_head(
            self.features,
            self.features @ weights.T + bias,
            self.heldout,
            self.heldout @ weights.T + bias,
            mode="base_median",
            has_bias=True,
            base_weight=self.base_weights,
            base_bias=self.base_bias,
        )
        np.testing.assert_allclose(fitted, self.base_weights, rtol=0, atol=1e-12)
        np.testing.assert_allclose(fitted_bias, self.base_bias, rtol=0, atol=1e-12)
        self.assertGreater(np.max(np.abs(fitted - weights)), 3.0)
        self.assertLess(metrics["validation_rmse"], 1e-12)

    def test_both_modes_reject_fp32_rounding_as_a_feature_direction(self):
        features = self.features.astype(np.float32)
        features[:, -1] = features[:, :3].sum(axis=1)
        self.assertEqual(np.linalg.matrix_rank(features.astype(np.float64)), 4)
        for mode, base_weight in (("raw_logits", None), ("base_median", self.base_weights)):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "rank deficient: rank 3 of 4"):
                    fit_unanchored_head(
                        features,
                        features @ self.base_weights.T,
                        self.heldout,
                        self.heldout @ self.base_weights.T,
                        mode=mode,
                        base_weight=base_weight,
                    )

    def test_incompatible_base_evidence_is_rejected(self):
        args = (
            self.features,
            self.features @ self.base_weights.T,
            self.heldout,
            self.heldout @ self.base_weights.T,
        )
        for kwargs, message in (
            ({"mode": "raw_logits", "base_weight": self.base_weights}, "does not accept"),
            ({"mode": "base_median"}, "requires base_weight"),
            ({"mode": "base_median", "base_weight": self.base_weights.T}, "base_weight must match"),
            (
                {"mode": "base_median", "base_weight": self.base_weights, "has_bias": True},
                "requires base_bias",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fit_unanchored_head(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
