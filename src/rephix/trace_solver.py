"""Recover layer parameters from paired training and held-out activation traces."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike


def _matrix(value: ArrayLike, name: str, *, allow_empty_rows: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] == 0
        or (array.shape[0] == 0 and not allow_empty_rows)
        or array.dtype.kind not in "fiu"
    ):
        requirement = (
            "a real numeric matrix with positive width"
            if allow_empty_rows
            else "a nonempty real numeric matrix"
        )
        raise ValueError(f"{name} must be {requirement}")
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
    *,
    allow_empty_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not isinstance(has_bias, (bool, np.bool_)):
        raise ValueError("has_bias must be a boolean")
    limit = np.asarray(max_validation_rmse)
    if limit.ndim != 0 or limit.dtype.kind not in "fiu":
        raise ValueError("max_validation_rmse must be a finite nonnegative scalar")
    limit = float(limit)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("max_validation_rmse must be a finite nonnegative scalar")

    x = _matrix(x, "x", allow_empty_rows=allow_empty_rows)
    y = _matrix(y, "y", allow_empty_rows=allow_empty_rows)
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


def _affine_design(
    x: np.ndarray, has_bias: bool, calibration_x: np.ndarray | None = None
) -> np.ndarray:
    if calibration_x is None:
        return np.column_stack((x, np.ones(x.shape[0]))) if has_bias else x
    rows, features = x.shape
    design = np.empty((rows + calibration_x.shape[0], features + int(has_bias)))
    design[:rows, :features] = x
    design[rows:, :features] = calibration_x
    if has_bias:
        design[:, -1] = 1
    return design


def _design_rank(singular: np.ndarray, tolerance: float) -> int:
    if not np.isfinite(singular).all():
        raise ValueError("training design has nonfinite singular values")
    return int(np.count_nonzero(singular > tolerance * singular[0]))


def plan_affine_calibration(
    x: ArrayLike, *, has_bias: bool, max_rows: int
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Plan one chosen feature row per missing affine design dimension.

    The probes use the trace dtype and, for an affine layer, the implicit
    intercept coordinate of every row is one. A plan is accepted only when
    the rounded probes complete the training design at the fitting cutoff.
    With no passive rows, probe each coordinate and add a zero anchor if
    fitting an intercept.
    """
    if not isinstance(has_bias, (bool, np.bool_)):
        raise ValueError("has_bias must be a boolean")
    if (
        isinstance(max_rows, (bool, np.bool_))
        or not isinstance(max_rows, (int, np.integer))
        or max_rows < 0
    ):
        raise ValueError("max_rows must be a nonnegative integer")
    x = _matrix(x, "x", allow_empty_rows=True)
    if x.dtype.type not in (np.float32, np.float64):
        raise ValueError("calibration planning requires FP32 or FP64 inputs")
    trace_dtype = x.dtype
    x = _fp64(x, "x")
    rows, features = x.shape
    width = features + int(has_bias)
    precision = float(np.finfo(trace_dtype).eps)
    passive_rank = 0
    if rows:
        design = _affine_design(x, has_bias)
        passive_tolerance = _rank_tolerance(rows, width, precision)
        try:
            # Only a wide design needs a full factorization to obtain every right direction.
            left, singular, right = np.linalg.svd(design, full_matrices=rows < width)
        except np.linalg.LinAlgError as exc:
            raise ValueError("training design SVD failed to converge") from exc
        del left, design
        passive_rank = _design_rank(singular, passive_tolerance)
    missing = width - passive_rank
    if missing > max_rows:
        raise ValueError(
            f"calibration row budget exceeded: need {missing} rows for rank "
            f"{passive_rank} of {width}, max_rows is {max_rows}"
        )

    tolerance = _rank_tolerance(rows + missing, width, precision)
    diagnostics: dict[str, int | float | str] = {
        "passive_rank": passive_rank,
        "design_width": width,
        "probe_rows": missing,
        "relative_rank_tolerance": tolerance,
    }
    if rows == 0:
        scale = float(trace_dtype.type(np.sqrt(width) if has_bias else 1))
        geometry = "basis_with_zero_anchor" if has_bias else "basis"
        completed = np.array([scale])
        if has_bias:
            # Feature contrasts have singular value scale. The mean feature
            # direction and intercept form this block with the extreme values.
            # sqrt(width) balances the feature and intercept column norms.
            completed = np.linalg.svd([[scale, np.sqrt(features)], [0, 1]], compute_uv=False)
        if _design_rank(completed, tolerance) < completed.size:
            raise ValueError(f"calibration could not complete rank 0 of {width} at the fit cutoff")
        probes = np.zeros((width, features), dtype=trace_dtype)
        np.fill_diagonal(probes[int(has_bias) :], scale)
        diagnostics.update(
            geometry=geometry,
            probe_scale=scale,
            condition_number=float(completed[0] / completed[-1]),
        )
        return probes, diagnostics
    if missing == 0:
        diagnostics.update(
            geometry="none", probe_scale=0.0, condition_number=float(singular[0] / singular[-1])
        )
        return np.empty((0, x.shape[1]), dtype=trace_dtype), diagnostics

    directions = right[passive_rank:]
    geometry = "nullspace"
    if has_bias:
        geometry = "projected_nullspace"
        # With N = [F, b] an orthonormal null basis, FF^T = I - bb^T.
        # In singular coordinates the feature probes contribute a diagonal
        # matrix. Matching its signs to the intercept projections prevents
        # cancellation by the rank-one contribution from the fixed ones.
        try:
            rotation, _, directions = np.linalg.svd(directions[:, :-1], full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError("calibration feature projection SVD failed to converge") from exc
        intercept = rotation.T @ right[passive_rank:, -1]
        directions *= np.copysign(1.0, intercept)[:, None]

    largest = float(singular[0])
    retained = float(singular[passive_rank - 1]) if passive_rank else 0.0
    typical = largest / np.sqrt(rows) if largest else 1.0
    # Start at the passive per-row scale. A smaller probe can preserve a
    # marginal retained direction, while a larger probe can lift a weak projection.
    scales = (
        typical,
        typical / np.sqrt(2),
        retained,
        largest * np.sqrt(tolerance),
        largest,
    )
    for scale in dict.fromkeys(scales):
        if scale <= 0:
            continue
        with np.errstate(over="ignore", under="ignore"):
            probes = (scale * directions).astype(trace_dtype)
        if not np.isfinite(probes).all():
            continue
        combined = _affine_design(x, has_bias, probes)
        try:
            completed = np.linalg.svd(combined, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError("calibrated design SVD failed to converge") from exc
        del combined
        if _design_rank(completed, tolerance) == width:
            diagnostics.update(
                geometry=geometry,
                probe_scale=float(scale),
                condition_number=float(completed[0] / completed[-1]),
            )
            return probes, diagnostics
    raise ValueError(
        f"calibration could not complete rank {passive_rank} of {width} "
        f"with {missing} probe rows at the fit cutoff"
    )


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
    calibration_x: ArrayLike | None = None,
    calibration_y: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int | float | bool]]:
    """Fit ``y = x @ weights + bias`` and validate on separate trace rows.

    Weights are FP64 in [input, output] order. The optional intercept is a
    separate vector. Optional paired calibration rows join the training design,
    which must have full numerical rank. Calibration can supply the entire
    design when both passive matrices have zero rows.
    """
    x, y, val_x, val_y, precision, limit = _traces(
        x, y, val_x, val_y, has_bias, max_validation_rmse, allow_empty_rows=True
    )
    passive_rows, n_inputs = x.shape
    n_rows = passive_rows
    if (calibration_x is None) != (calibration_y is None):
        raise ValueError("calibration_x and calibration_y must be supplied together")
    if calibration_x is not None:
        calibration_x = _matrix(calibration_x, "calibration_x", allow_empty_rows=True)
        calibration_y = _matrix(calibration_y, "calibration_y", allow_empty_rows=True)
        if calibration_x.shape[1] != n_inputs or calibration_y.shape != (
            calibration_x.shape[0],
            y.shape[1],
        ):
            raise ValueError("calibration traces must match training widths and have paired rows")
        calibration_x = _fp64(calibration_x, "calibration_x")
        n_rows += calibration_x.shape[0]
    if n_rows == 0:
        raise ValueError("training design is rank deficient: no passive or calibration rows")
    width = n_inputs + int(has_bias)
    if n_rows < width:
        raise ValueError("training design is rank deficient: fewer rows than parameters per output")
    design = _affine_design(x, has_bias, calibration_x if n_rows > passive_rows else None)
    try:
        left, singular, right = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("training design SVD failed to converge") from exc
    del design
    tolerance = _rank_tolerance(n_rows, width, precision)
    rank = _design_rank(singular, tolerance)
    if rank < width:
        raise ValueError(f"training design is rank deficient: rank {rank} of {width}")

    parameters = np.empty((width, y.shape[1]), dtype=np.float64)
    rmse = 0.0
    for columns in _columns(y.shape[1], max(n_rows, val_x.shape[0])):
        targets = _fp64(y[:, columns], "y")
        if n_rows > passive_rows:
            chosen_targets = _fp64(calibration_y[:, columns], "calibration_y")
            targets = (
                np.concatenate((targets, chosen_targets), axis=0)
                if passive_rows
                else chosen_targets
            )
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
    if calibration_x is not None:
        diagnostics.update(
            training_samples=n_rows,
            passive_samples=passive_rows,
            calibration_samples=n_rows - passive_rows,
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
