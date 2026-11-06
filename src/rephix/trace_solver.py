"""Recover layer parameters from paired training and held-out activation traces."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike


def _matrix(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not all(array.shape) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty real numeric matrix")
    return array


def _fp64(array: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite FP64 values")
    return array


def _traces(
    x: ArrayLike,
    y: ArrayLike,
    val_x: ArrayLike,
    val_y: ArrayLike,
    has_bias: bool,
    max_validation_rmse: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not isinstance(has_bias, (bool, np.bool_)):
        raise ValueError("has_bias must be a boolean")
    limit = np.asarray(max_validation_rmse)
    if limit.ndim != 0 or limit.dtype.kind not in "fiu":
        raise ValueError("max_validation_rmse must be a finite nonnegative scalar")
    limit = float(limit)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("max_validation_rmse must be a finite nonnegative scalar")

    x = _matrix(x, "x")
    y = _matrix(y, "y")
    val_x = _matrix(val_x, "val_x")
    val_y = _matrix(val_y, "val_y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if val_x.shape[1] != x.shape[1] or val_y.shape != (val_x.shape[0], y.shape[1]):
        raise ValueError("held-out traces must match training widths and have paired rows")
    precision = float(np.finfo(x.dtype).eps) if x.dtype.kind == "f" else 0.0
    return _fp64(x, "x"), y, _fp64(val_x, "val_x"), val_y, precision, limit


def _rank_tolerance(rows: int, width: int, precision: float) -> float:
    # Forward rounding can make a dependent feature appear independent in FP64.
    return float(max(max(rows, width) * np.finfo(np.float64).eps, 8 * precision))


def _columns(width: int, rows: int) -> Iterator[slice]:
    chunk = max(1, min(512, 2**20 // rows))
    for start in range(0, width, chunk):
        yield slice(start, start + chunk)


def _score(residual: np.ndarray, total_entries: int) -> float:
    scale = np.max(np.abs(residual))
    if not np.isfinite(scale):
        return np.inf
    if scale == 0:
        return 0.0
    scaled = residual / scale
    return float(np.sqrt(np.vdot(scaled, scaled) / total_entries) * scale)


def _diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    val_x: np.ndarray,
    width: int,
    condition: float,
    tolerance: float,
    rmse: float,
    limit: float,
    has_bias: bool,
) -> dict[str, int | float | bool]:
    if not np.isfinite(rmse):
        raise ValueError("trace fit produced nonfinite held-out residuals")
    if rmse > limit:
        raise ValueError(f"held-out RMSE {rmse:.6g} exceeds max_validation_rmse {limit:.6g}")
    return {
        "design_rank": width,
        "design_width": width,
        "condition_number": float(condition),
        "relative_rank_tolerance": tolerance,
        "validation_rmse": float(rmse),
        "max_validation_rmse": limit,
        "has_bias": bool(has_bias),
        "features": x.shape[1],
        "outputs": y.shape[1],
        "training_samples": x.shape[0],
        "validation_samples": val_x.shape[0],
    }


def fit_trace_affine(
    x: ArrayLike,
    y: ArrayLike,
    val_x: ArrayLike,
    val_y: ArrayLike,
    *,
    has_bias: bool,
    max_validation_rmse: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int | float | bool]]:
    """Fit ``y = x @ weights + bias`` and validate on separate trace rows.

    Weights are FP64 in [input, output] order. The optional intercept is a
    separate vector. The augmented training design must have full numerical rank.
    """
    x, y, val_x, val_y, precision, limit = _traces(
        x, y, val_x, val_y, has_bias, max_validation_rmse
    )
    n_rows, n_inputs = x.shape
    width = n_inputs + int(has_bias)
    if n_rows < width:
        raise ValueError("training design is rank deficient: fewer rows than parameters per output")
    design = np.column_stack((x, np.ones(n_rows))) if has_bias else x
    try:
        left, singular, right = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("training design SVD failed to converge") from exc
    del design
    tolerance = _rank_tolerance(n_rows, width, precision)
    if not np.isfinite(singular).all():
        raise ValueError("training design has nonfinite singular values")
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    if rank < width:
        raise ValueError(f"training design is rank deficient: rank {rank} of {width}")

    parameters = np.empty((width, y.shape[1]), dtype=np.float64)
    rmse = 0.0
    for columns in _columns(y.shape[1], max(n_rows, val_x.shape[0])):
        targets = _fp64(y[:, columns], "y")
        heldout_targets = _fp64(val_y[:, columns], "val_y")
        with np.errstate(over="ignore", invalid="ignore"):
            coefficients = right.T @ ((left.T @ targets) / singular[:, None])
            residual = val_x @ coefficients[:n_inputs]
            if has_bias:
                residual += coefficients[-1]
            residual -= heldout_targets
        if not np.isfinite(coefficients).all():
            raise ValueError("trace fit produced nonfinite parameters")
        parameters[:, columns] = coefficients
        rmse = np.hypot(rmse, _score(residual, val_y.size))

    diagnostics = _diagnostics(
        x, y, val_x, width, singular[0] / singular[-1], tolerance, rmse, limit, has_bias
    )
    return parameters[:n_inputs], parameters[-1] if has_bias else None, diagnostics


def fit_trace_normalization(
    x: ArrayLike,
    y: ArrayLike,
    val_x: ArrayLike,
    val_y: ArrayLike,
    *,
    has_bias: bool,
    max_validation_rmse: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int | float | bool]]:
    """Fit elementwise gain and optional bias to already-normalized inputs.

    Each input column needs observable variation when fitting an intercept, or
    nonzero excitation otherwise. Rank and condition describe the equivalent
    block-diagonal design across all coordinates.
    """
    x, y, val_x, val_y, precision, limit = _traces(
        x, y, val_x, val_y, has_bias, max_validation_rmse
    )
    if x.shape != y.shape:
        raise ValueError("normalization input and output traces must have the same shape")
    n_rows, n_features = x.shape
    coordinate_width = 1 + int(has_bias)
    if n_rows < coordinate_width:
        raise ValueError("training normalization design is rank deficient: too few rows")
    tolerance = _rank_tolerance(n_rows, coordinate_width, precision)
    gamma = np.empty(n_features, dtype=np.float64)
    beta = np.empty(n_features, dtype=np.float64) if has_bias else None
    singular = np.empty((n_features, coordinate_width), dtype=np.float64)
    rmse = 0.0
    for columns in _columns(n_features, max(n_rows, val_x.shape[0])):
        features = x[:, columns]
        targets = _fp64(y[:, columns], "y")
        heldout_targets = _fp64(val_y[:, columns], "val_y")

        # Scale columns before computing moments, and center them before
        # fitting the gain when there is an intercept.
        feature_scale = np.max(np.abs(features), axis=0)
        target_scale = np.max(np.abs(targets), axis=0)
        scaled_x = features / np.where(feature_scale == 0, 1.0, feature_scale)
        scaled_y = targets / np.where(target_scale == 0, 1.0, target_scale)
        if has_bias:
            mean_x = scaled_x.mean(axis=0)
            mean_y = scaled_y.mean(axis=0)
            scaled_x -= mean_x
            scaled_y -= mean_y
        power = np.einsum("ij,ij->j", scaled_x, scaled_x) / n_rows
        rms = np.sqrt(power) * feature_scale

        if has_bias:
            # This triangular factor of each [1, x] block retains its
            # singular values, scaled by the common factor sqrt(n_rows).
            triangular = np.zeros((rms.size, 2, 2))
            triangular[:, 0, 0] = 1.0
            triangular[:, 0, 1] = mean_x * feature_scale
            triangular[:, 1, 1] = rms
            try:
                singular[columns] = np.linalg.svd(triangular, compute_uv=False)
            except np.linalg.LinAlgError as exc:
                raise ValueError("training normalization design SVD failed to converge") from exc
        else:
            singular[columns, 0] = rms

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            slope = (np.einsum("ij,ij->j", scaled_x, scaled_y) / n_rows) / power
            gain = (slope * target_scale) / feature_scale
            gamma[columns] = gain
            if beta is not None:
                beta[columns] = (mean_y - slope * mean_x) * target_scale
                residual = val_x[:, columns] * gain + beta[columns] - heldout_targets
            else:
                residual = val_x[:, columns] * gain - heldout_targets
        rmse = np.hypot(rmse, _score(residual, val_y.size))

    largest = np.max(singular)
    if not np.isfinite(largest):
        raise ValueError("training normalization design has nonfinite singular values")
    rank = int(np.count_nonzero(singular > tolerance * largest))
    width = singular.size
    if rank < width:
        raise ValueError(f"training normalization design is rank deficient: rank {rank} of {width}")
    if not np.isfinite(gamma).all() or (beta is not None and not np.isfinite(beta).all()):
        raise ValueError("normalization fit produced nonfinite parameters")
    diagnostics = _diagnostics(
        x, y, val_x, width, largest / np.min(singular), tolerance, rmse, limit, has_bias
    )
    return gamma, beta, diagnostics
