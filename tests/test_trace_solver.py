"""Tests for affine calibration and passive trace recovery."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from rephix.trace_solver import fit_trace_affine, fit_trace_normalization, plan_affine_calibration


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

    def test_calibrates_multiple_affine_directions_from_a_wide_design(self):
        x = self.x[:2].copy()
        x[:, -1] = 4
        probes, plan = plan_affine_calibration(x, has_bias=True, max_rows=3)
        self.assertEqual(probes.shape, (3, 4))
        self.assertEqual(probes.dtype, x.dtype)
        self.assertEqual(
            (plan["passive_rank"], plan["design_width"], plan["probe_rows"]), (2, 5, 3)
        )
        self.assertEqual(plan["geometry"], "projected_nullspace")
        json.dumps(plan, allow_nan=False)

        weight, bias, fit = fit_trace_affine(
            x,
            x @ self.weights + self.bias,
            self.val_x,
            self.val_x @ self.weights + self.bias,
            has_bias=True,
            calibration_x=probes,
            calibration_y=probes @ self.weights + self.bias,
        )
        np.testing.assert_allclose(weight, self.weights, atol=2e-6, rtol=0)
        np.testing.assert_allclose(bias, self.bias, atol=2e-6, rtol=0)
        self.assertEqual(fit["design_rank"], 5)
        self.assertEqual(fit["training_samples"], 5)
        self.assertEqual(fit["passive_samples"], 2)
        self.assertEqual(fit["calibration_samples"], 3)

    def test_calibrates_multiple_biasless_directions_without_doubling_targets(self):
        x = self.x.astype(np.float64)
        x[:, 2:] = 0
        val_x = self.val_x.astype(np.float64)
        probes, plan = plan_affine_calibration(x, has_bias=False, max_rows=2)
        self.assertEqual(probes.shape, (2, 4))
        self.assertEqual(probes.dtype, np.float64)
        self.assertEqual(plan["passive_rank"], 2)
        self.assertEqual(plan["geometry"], "nullspace")

        with patch("rephix.trace_solver.np.concatenate", wraps=np.concatenate) as concatenate:
            weight, bias, _ = fit_trace_affine(
                x,
                x @ self.weights,
                val_x,
                val_x @ self.weights,
                has_bias=False,
                calibration_x=probes,
                calibration_y=probes @ self.weights,
            )
        self.assertEqual(concatenate.call_count, 2)
        self.assertEqual(
            [call.args[0][1].shape for call in concatenate.call_args_list], [(2, 512), (2, 8)]
        )
        np.testing.assert_allclose(weight, self.weights, atol=1e-12, rtol=0)
        self.assertIsNone(bias)

    def test_planner_uses_the_training_cutoff_and_reports_budget(self):
        self.x[:, -1] = self.x[:, :3].sum(axis=1)
        probes, plan = plan_affine_calibration(self.x, has_bias=False, max_rows=1)
        self.assertEqual(probes.shape, (1, 4))
        self.assertEqual(plan["passive_rank"], 3)
        self.assertEqual(plan["relative_rank_tolerance"], 8 * np.finfo(np.float32).eps)
        with self.assertRaisesRegex(ValueError, "calibration row budget exceeded: need 1"):
            plan_affine_calibration(self.x, has_bias=False, max_rows=0)
        with self.assertRaisesRegex(ValueError, "max_rows must be a nonnegative integer"):
            plan_affine_calibration(self.x, has_bias=False, max_rows=True)

        x = self.x.astype(np.float64)
        probes, plan = plan_affine_calibration(x, has_bias=False, max_rows=0)
        self.assertEqual(probes.shape, (0, 4))
        self.assertEqual(probes.dtype, x.dtype)
        self.assertEqual(plan["passive_rank"], 4)
        self.assertEqual(plan["geometry"], "none")

    def test_planner_rejects_an_intercept_below_the_cutoff_at_any_probe_scale(self):
        x = np.full((3, 1), 1e10, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "could not complete rank 1 of 2"):
            plan_affine_calibration(x, has_bias=True, max_rows=1)

    def test_planner_reduces_probe_scale_to_keep_a_marginal_intercept(self):
        x = np.full((2, 1), 1e6, dtype=np.float32)
        probes, plan = plan_affine_calibration(x, has_bias=True, max_rows=1)
        self.assertEqual(probes.shape, (1, 1))
        self.assertLess(plan["probe_scale"], 1e6)
        self.assertLess(plan["condition_number"], 1 / plan["relative_rank_tolerance"])

    def test_fits_calibration_basis_without_passive_rows(self):
        for dtype in (np.float32, np.float64):
            for has_bias in (False, True):
                with self.subTest(dtype=dtype, has_bias=has_bias):
                    x = self.x[:0].astype(dtype)
                    y = np.empty((0, self.weights.shape[1]), dtype=dtype)
                    val_x = self.val_x.astype(dtype)
                    width = x.shape[1] + int(has_bias)
                    probes, plan = plan_affine_calibration(x, has_bias=has_bias, max_rows=width)
                    self.assertEqual(probes.shape, (width, x.shape[1]))
                    self.assertEqual(probes.dtype, dtype)
                    self.assertEqual(plan["passive_rank"], 0)
                    self.assertEqual(plan["design_width"], width)
                    self.assertEqual(plan["probe_rows"], width)
                    if has_bias:
                        np.testing.assert_array_equal(probes[0], 0)
                        design = np.column_stack((probes, np.ones(width)))
                        self.assertEqual(plan["geometry"], "basis_with_zero_anchor")
                    else:
                        design = probes.astype(np.float64)
                        self.assertEqual(plan["geometry"], "basis")
                    self.assertAlmostEqual(plan["condition_number"], np.linalg.cond(design))
                    self.assertLess(plan["condition_number"], 1 / plan["relative_rank_tolerance"])
                    json.dumps(plan, allow_nan=False)
                    with self.assertRaisesRegex(
                        ValueError, f"calibration row budget exceeded: need {width} rows for rank 0"
                    ):
                        plan_affine_calibration(x, has_bias=has_bias, max_rows=0)

                    intercept = self.bias if has_bias else 0
                    weight, bias, fit = fit_trace_affine(
                        x,
                        y,
                        val_x,
                        val_x @ self.weights + intercept,
                        has_bias=has_bias,
                        calibration_x=probes,
                        calibration_y=probes @ self.weights + intercept,
                    )
                    np.testing.assert_allclose(weight, self.weights, atol=5e-7, rtol=0)
                    if has_bias:
                        np.testing.assert_allclose(bias, self.bias, atol=5e-7, rtol=0)
                    else:
                        self.assertIsNone(bias)
                    self.assertEqual(fit["design_rank"], width)
                    self.assertEqual(fit["training_samples"], width)
                    self.assertEqual(fit["passive_samples"], 0)
                    self.assertEqual(fit["calibration_samples"], width)
                    self.assertLess(fit["validation_rmse"], 1e-6)

    def test_empty_passive_fit_requires_calibration_and_held_out_evidence(self):
        x = self.x[:0]
        y = np.empty((0, self.weights.shape[1]), dtype=x.dtype)
        val_y = self.val_x @ self.weights
        probes = np.eye(x.shape[1], dtype=x.dtype)
        with self.assertRaisesRegex(ValueError, "no passive or calibration rows"):
            fit_trace_affine(x, y, self.val_x, val_y, has_bias=False)
        with self.assertRaisesRegex(ValueError, "no passive or calibration rows"):
            fit_trace_affine(
                x, y, self.val_x, val_y, has_bias=False, calibration_x=x, calibration_y=y
            )
        args = {"has_bias": False, "calibration_x": probes, "calibration_y": probes @ self.weights}
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            fit_trace_affine(x, val_y, self.val_x, val_y, **args)
        with self.assertRaisesRegex(ValueError, "val_x must be a nonempty"):
            fit_trace_affine(x, y, self.val_x[:0], val_y[:0], **args)
        val_y[0, -1] += 1
        with self.assertRaisesRegex(ValueError, "held-out RMSE .* exceeds"):
            fit_trace_affine(x, y, self.val_x, val_y, **args)

    def test_calibration_requires_paired_targets_and_preserves_the_held_out_check(self):
        x = self.x.copy()
        x[:, -1] = 0
        probes, _ = plan_affine_calibration(x, has_bias=False, max_rows=1)
        y, val_y = x @ self.weights, self.val_x @ self.weights
        chosen_y = probes @ self.weights
        args = (x, y, self.val_x, val_y)
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            fit_trace_affine(*args, has_bias=False, calibration_x=probes)
        with self.assertRaisesRegex(ValueError, "must match training widths"):
            fit_trace_affine(
                *args, has_bias=False, calibration_x=probes, calibration_y=chosen_y[:, :-1]
            )
        with self.assertRaisesRegex(ValueError, "calibration_y must contain only finite"):
            fit_trace_affine(
                *args, has_bias=False, calibration_x=probes, calibration_y=chosen_y * np.nan
            )
        wrong_y = chosen_y.copy()
        wrong_y[0, -1] += 1
        wrong_val_y = val_y.copy()
        wrong_val_y[0, -1] += 1
        for calibration_y, heldout_y in ((wrong_y, val_y), (chosen_y, wrong_val_y)):
            with self.assertRaisesRegex(ValueError, "held-out RMSE .* exceeds"):
                fit_trace_affine(
                    x,
                    y,
                    self.val_x,
                    heldout_y,
                    has_bias=False,
                    calibration_x=probes,
                    calibration_y=calibration_y,
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
