"""Identify an affine output head using dense scores and a known anchor row."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _real_array(value: ArrayLike, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array")
    return array


def _design(features: ArrayLike, name: str, has_bias: bool) -> tuple[np.ndarray, float]:
    design = _real_array(features, name)
    if design.ndim != 2 or not all(design.shape):
        raise ValueError(f"{name} must be a nonempty 2D matrix")
    precision = float(np.finfo(design.dtype).eps) if design.dtype.kind == "f" else 0.0
    with np.errstate(over="ignore"):
        design = design.astype(np.float64, copy=False)
    if not np.isfinite(design).all():
        raise ValueError(f"{name} must contain only finite FP64 values")
    if has_bias:
        design = np.column_stack((design, np.ones(design.shape[0])))
    return design, precision


def _finite_scalar(value: ArrayLike, name: str) -> float:
    array = _real_array(value, name)
    if array.ndim != 0 or not np.isfinite(result := float(array)):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _contrasts(scores: np.ndarray, columns: slice, anchor: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        contrasts = np.subtract(scores[:, columns], anchor, dtype=np.float64)
    if not np.isfinite(contrasts).all():
        raise ValueError(f"{name} must contain only finite scores with finite FP64 contrasts")
    return contrasts


def fit_anchored_head(
    features: ArrayLike,
    scores: ArrayLike,
    validation_features: ArrayLike,
    validation_scores: ArrayLike,
    *,
    anchor_id: int,
    anchor_weight: ArrayLike,
    anchor_bias: float | None = None,
    chunk_size: int = 512,
    max_validation_rmse: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int | float | bool]]:
    """Fit weights in [classes, features] order and optionally a bias vector.

    Each score matrix must contain the same full vocabulary in token-ID order.
    Scores can be logits or natural log probabilities. Anchor subtraction
    cancels any additive shift shared by all classes in a sample.

    The caller supplies a known unchanged anchor row and disjoint validation
    observations. Set ``anchor_bias`` to the known bias (including zero) for a
    head with bias, or leave it as ``None`` for a bias-free head. The affine
    model and the supplied anchor determine the recovered parameters. Held-out
    contrast error checks the fit to the supplied observations. It cannot
    verify the anchor or rule out other behavior outside those observations.
    """
    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    max_validation_rmse = _finite_scalar(max_validation_rmse, "max_validation_rmse")
    if max_validation_rmse < 0:
        raise ValueError("max_validation_rmse must be nonnegative")

    has_bias = anchor_bias is not None
    design, feature_precision = _design(features, "features", has_bias)
    heldout, _ = _design(validation_features, "validation_features", has_bias)
    if design.shape[1] != heldout.shape[1]:
        raise ValueError("features and validation_features must have the same width")
    width = design.shape[1] - has_bias
    anchor = _real_array(anchor_weight, "anchor_weight")
    if anchor.shape != (width,):
        raise ValueError("anchor_weight must be a vector matching the feature width")
    with np.errstate(over="ignore"):
        anchor = anchor.astype(np.float64, copy=False)
    if not np.isfinite(anchor).all():
        raise ValueError("anchor_weight must contain only finite FP64 values")
    if has_bias:
        anchor = np.append(anchor, _finite_scalar(anchor_bias, "anchor_bias"))

    observed = _real_array(scores, "scores")
    if observed.ndim != 2 or observed.shape[0] != design.shape[0] or observed.shape[1] < 2:
        raise ValueError(
            "scores must have one row per training feature row and at least two classes"
        )
    classes = observed.shape[1]
    validation_observed = _real_array(validation_scores, "validation_scores")
    if validation_observed.shape != (heldout.shape[0], classes):
        raise ValueError("validation_scores must match the held-out rows and training vocabulary")
    if (
        isinstance(anchor_id, (bool, np.bool_))
        or not isinstance(anchor_id, (int, np.integer))
        or not 0 <= anchor_id < classes
    ):
        raise ValueError("anchor_id must be an integer within the vocabulary")
    anchor_id = int(anchor_id)

    if design.shape[0] < design.shape[1]:
        raise ValueError("training design is rank deficient: fewer rows than parameters per class")
    try:
        left, singular_values, right = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("training design SVD failed to converge") from exc
    # A nominally full rank FP32 design can consist of an exact hidden
    # hyperplane plus forward rounding. Solving in FP64 cannot make those
    # small singular values informative.
    relative_tolerance = max(max(design.shape) * np.finfo(np.float64).eps, 8 * feature_precision)
    tolerance = relative_tolerance * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != design.shape[1]:
        raise ValueError(f"training design is rank deficient: rank {rank} of {design.shape[1]}")
    condition_number = float(singular_values[0] / singular_values[-1])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse = (right.T / singular_values) @ left.T
    del left, singular_values, right
    if not np.isfinite(inverse).all():
        raise ValueError("training design pseudoinverse contains nonfinite values")

    train_anchor = observed[:, anchor_id : anchor_id + 1]
    validation_anchor = validation_observed[:, anchor_id : anchor_id + 1]
    weights = np.empty((classes, width), dtype=np.float64)
    bias = np.empty(classes, dtype=np.float64) if has_bias else None
    training_squared_error = validation_squared_error = 0.0
    for start in range(0, classes, chunk_size):
        stop = min(start + chunk_size, classes)
        columns = slice(start, stop)
        targets = _contrasts(observed, columns, train_anchor, "scores")
        heldout_targets = _contrasts(
            validation_observed, columns, validation_anchor, "validation_scores"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            parameters = inverse @ targets
            parameters += anchor[:, None]
            if start <= anchor_id < stop:
                parameters[:, anchor_id - start] = anchor
            weights[columns] = parameters[:width].T
            if bias is not None:
                bias[columns] = parameters[-1]

            # Check contrasts of the actual returned parameters, including rounding
            # when the known anchor is added back to each fitted class.
            parameters -= anchor[:, None]
            residual = design @ parameters - targets
            heldout_residual = heldout @ parameters - heldout_targets
            training_squared_error += float(np.vdot(residual, residual))
            validation_squared_error += float(np.vdot(heldout_residual, heldout_residual))

    # The anchor contributes exactly zero and is excluded from the denominator.
    training_rmse = float(np.sqrt(training_squared_error / (design.shape[0] * (classes - 1))))
    validation_rmse = float(np.sqrt(validation_squared_error / (heldout.shape[0] * (classes - 1))))
    if not np.isfinite(training_rmse) or not np.isfinite(validation_rmse):
        raise ValueError("anchored fit produced nonfinite contrast errors")
    if validation_rmse > max_validation_rmse:
        raise ValueError(
            f"held-out contrast RMSE {validation_rmse:.6g} exceeds "
            f"max_validation_rmse {max_validation_rmse:.6g}"
        )
    return (
        weights,
        bias,
        {
            "anchor_id": anchor_id,
            "has_bias": has_bias,
            "classes": classes,
            "features": width,
            "parameters_per_class": design.shape[1],
            "training_samples": design.shape[0],
            "validation_samples": heldout.shape[0],
            "design_rank": rank,
            "rank_tolerance": float(tolerance),
            "relative_rank_tolerance": float(relative_tolerance),
            "condition_number": condition_number,
            "training_contrast_rmse": training_rmse,
            "validation_contrast_rmse": validation_rmse,
            "max_validation_rmse": max_validation_rmse,
        },
    )
