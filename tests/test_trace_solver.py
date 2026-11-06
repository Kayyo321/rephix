"""Tests for passive affine and normalization recovery."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from rephix.trace_solver import fit_trace_affine, fit_trace_normalization


class TraceAffineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(30)
        self.x = rng.normal(size=(20, 4)).astype(np.float32)
        self.val_x = rng.normal(size=(7, 4)).astype(np.float32)
        self.weights = rng.normal(size=(4, 520)).astype(np.float32)
        self.bias = rng.normal(size=520).astype(np.float32)

    def test_fits_biased_affine_trace_with_one_factorization(self):
        with patch("rephix.trace_solver.np.linalg.svd", wraps=np.linalg.svd) as svd:
            weight, bias, fit = fit_trace_affine(
                self.x,
                self.x @ self.weights + self.bias,
                self.val_x,
                self.val_x @ self.weights + self.bias,
                has_bias=True,
            )
        svd.assert_called_once()
        np.testing.assert_allclose(weight, self.weights, atol=5e-7, rtol=0)
        np.testing.assert_allclose(bias, self.bias, atol=5e-7, rtol=0)
        self.assertEqual(fit["design_width"], 5)
        self.assertEqual(fit["design_rank"], 5)
        self.assertLess(fit["validation_rmse"], 1e-6)
        json.dumps(fit, allow_nan=False)

    def test_bias_free_trace_scores_all_held_out_entries(self):
        val_y = self.val_x @ self.weights
        val_y[1, 1] += 1.0
        val_y[2, -1] -= 1.0
        args = self.x, self.x @ self.weights, self.val_x, val_y
        with self.assertRaisesRegex(ValueError, "held-out RMSE .* exceeds"):
            fit_trace_affine(*args, has_bias=False)
        weight, bias, fit = fit_trace_affine(*args, has_bias=False, max_validation_rmse=0.025)
        np.testing.assert_allclose(weight, self.weights, atol=5e-7, rtol=0)
        self.assertIsNone(bias)
        self.assertAlmostEqual(fit["validation_rmse"], np.sqrt(2 / val_y.size))

    def test_rejects_fp32_rounding_noise_in_dependent_column(self):
        self.x[:, -1] = self.x[:, :3].sum(axis=1)
        self.assertEqual(np.linalg.matrix_rank(self.x.astype(np.float64)), 4)
        with self.assertRaisesRegex(ValueError, "rank deficient: rank 3 of 4"):
            fit_trace_affine(
                self.x, self.x @ self.weights, self.val_x, self.val_x @ self.weights, has_bias=False
            )

    def test_rejects_dependent_intercept_and_too_few_rows(self):
        self.x[:, -1] = 1
        for x in (self.x, self.x[:4]):
            with (
                self.subTest(rows=x.shape[0]),
                self.assertRaisesRegex(ValueError, "rank deficient"),
            ):
                fit_trace_affine(
                    x, x @ self.weights, self.val_x, self.val_x @ self.weights, has_bias=True
                )


class TraceNormalizationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(44)
        self.x = rng.normal(size=(25, 6)) + np.arange(6)
        self.val_x = rng.normal(size=(4, 6)) + np.arange(6)
        self.gamma = np.array([0.0, 0.25, 0.5, 0.75, 1.0, -0.25])
        self.beta = np.array([5.0, 6.0, 1.0, 0.0, 2.0, 1.0])

    def test_fits_channelwise_normalization_and_reports_block_design_condition(self):
        y = self.x * self.gamma + self.beta
        val_y = self.val_x * self.gamma + self.beta
        gamma, beta, fit = fit_trace_normalization(self.x, y, self.val_x, val_y, has_bias=True)
        np.testing.assert_allclose(gamma, self.gamma, atol=1e-12, rtol=0)
        np.testing.assert_allclose(beta, self.beta, atol=1e-12, rtol=0)
        designs = np.stack((np.ones_like(self.x.T), self.x.T), axis=-1)
        singular = np.linalg.svd(designs, compute_uv=False)
        self.assertAlmostEqual(fit["condition_number"], singular.max() / singular.min())
        self.assertEqual(fit["design_rank"], 2 * self.gamma.size)
        self.assertLess(fit["validation_rmse"], 1e-12)
        json.dumps(fit, allow_nan=False)

    def test_bias_free_normalization_allows_nonzero_constant_feature(self):
        self.x[:, 1] = 2
        gamma, beta, fit = fit_trace_normalization(
            self.x, self.x * self.gamma, self.val_x, self.val_x * self.gamma, has_bias=False
        )
        np.testing.assert_allclose(gamma, self.gamma, atol=1e-12, rtol=0)
        self.assertIsNone(beta)
        self.assertEqual(fit["design_rank"], self.gamma.size)

    def test_rejects_unexcited_or_constant_coordinate(self):
        for has_bias, value in ((False, 0.0), (True, 1.0)):
            with (
                self.subTest(has_bias=has_bias),
                self.assertRaisesRegex(ValueError, "rank deficient"),
            ):
                self.x[:, -1] = value
                fit_trace_normalization(
                    self.x,
                    self.x * self.gamma,
                    self.val_x,
                    self.val_x * self.gamma,
                    has_bias=has_bias,
                )

    def test_rejects_fp32_rounding_noise_in_constant_coordinate(self):
        x = self.x.astype(np.float32)
        x[:, -1] = 1
        x[0, -1] = np.nextafter(np.float32(1), np.float32(2))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            fit_trace_normalization(x, x, self.val_x, self.val_x, has_bias=True)

    def test_rejects_incompatible_held_out_trace(self):
        y = self.x * self.gamma + self.beta
        val_y = self.val_x * self.gamma + self.beta
        val_y[:, 1] += 0.01
        with self.assertRaisesRegex(ValueError, "held-out RMSE .* exceeds"):
            fit_trace_normalization(self.x, y, self.val_x, val_y, has_bias=True)


class TraceInputTests(unittest.TestCase):
    def test_invalid_trace_arrays_and_options(self):
        x = np.eye(2)
        args = {"x": x, "y": x, "val_x": x, "val_y": x, "has_bias": False}
        for solve in (fit_trace_affine, fit_trace_normalization):
            for change in (
                {"x": x[:, 0]},
                {"x": x.astype(complex)},
                {"y": x[:, :0]},
                {"y": x[:1]},
                {"y": x * np.nan},
                {"val_x": x[:, :1]},
                {"val_y": x[:1]},
                {"val_y": [[0.0, np.inf], [0.0, 0.0]]},
                {"has_bias": 1},
                {"max_validation_rmse": -1},
                {"max_validation_rmse": np.nan},
                {"max_validation_rmse": True},
            ):
                with self.subTest(solver=solve.__name__, **change), self.assertRaises(ValueError):
                    solve(**(args | change))
