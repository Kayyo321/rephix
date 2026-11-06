"""Identify an affine output head from dense scores and evidence for its gauge."""

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


def _targets(
    scores: np.ndarray, columns: slice, reference: np.ndarray | None, name: str
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        if reference is None:
            targets = scores[:, columns].astype(np.float64, copy=False)
        else:
            targets = np.subtract(scores[:, columns], reference, dtype=np.float64)
    if not np.isfinite(targets).all():
        if reference is None:
            raise ValueError(f"{name} must contain only finite FP64 scores")
        raise ValueError(f"{name} must contain only finite scores with finite FP64 contrasts")
    return targets


def _restore_median_gauge(fitted: np.ndarray, base: np.ndarray, chunk_size: int, name: str) -> None:
    # Every coordinate needs all classes, so bound the temporary by slicing features.
    feature_chunk = max(1, min(chunk_size, 2**20 // fitted.shape[0]))
    for start in range(0, fitted.shape[1], feature_chunk):
        columns = slice(start, start + feature_chunk)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.subtract(
                base[:, columns].T, fitted[:, columns].T, dtype=np.float64, order="C"
            )
            if not np.isfinite(difference).all():
                raise ValueError(f"{name} and fitted parameters must have finite FP64 differences")
            fitted[:, columns] += np.median(difference, axis=1, overwrite_input=True)
        if not np.isfinite(fitted[:, columns]).all():
            raise ValueError(f"median gauge produced nonfinite parameters for {name}")


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
    return _fit_head(
        features,
        scores,
        validation_features,
        validation_scores,
        mode="anchored",
        has_bias=anchor_bias is not None,
        anchor_id=anchor_id,
        anchor_weight=anchor_weight,
        anchor_bias=anchor_bias,
        chunk_size=chunk_size,
        max_validation_rmse=max_validation_rmse,
    )


def fit_unanchored_head(
    features: ArrayLike,
    scores: ArrayLike,
    validation_features: ArrayLike,
    validation_scores: ArrayLike,
    *,
    mode: str,
    has_bias: bool = False,
    base_weight: ArrayLike | None = None,
    base_bias: ArrayLike | None = None,
    chunk_size: int = 512,
    max_validation_rmse: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int | float | bool | str]]:
    """Fit an affine head in [classes, features] order without a known anchor.

    Each score matrix must contain the same vocabulary in token-ID order, with
    disjoint training and held-out observations. ``raw_logits`` fits absolute
    pre-softmax scores and measures absolute RMSE.

    ``base_median`` fits log odds against token 0 from logits or natural log
    probabilities. The coordinatewise median difference from the base head
    supplies the shared shift. It recovers the true gauge if strictly more than
    half the rows are unchanged from the base head, up to observation and solve
    error. Held-out contrast RMSE checks score differences and cannot establish
    the unchanged-row premise or detect an incorrect shared shift. A biased head
    requires a base bias vector in this mode.
    """
    if not isinstance(mode, str) or mode not in ("raw_logits", "base_median"):
        raise ValueError("mode must be 'raw_logits' or 'base_median'")
    if not isinstance(has_bias, (bool, np.bool_)):
        raise ValueError("has_bias must be a boolean")
    if mode == "raw_logits" and (base_weight is not None or base_bias is not None):
        raise ValueError("raw_logits mode does not accept base_weight or base_bias")
    weights, bias, fit = _fit_head(
        features,
        scores,
        validation_features,
        validation_scores,
        mode=mode,
        has_bias=bool(has_bias),
        base_weight=base_weight,
        base_bias=base_bias,
        chunk_size=chunk_size,
        max_validation_rmse=max_validation_rmse,
    )
    return weights, bias, {**fit, "fit_mode": mode}


def _fit_head(
    features: ArrayLike,
    scores: ArrayLike,
    validation_features: ArrayLike,
    validation_scores: ArrayLike,
    *,
    mode: str,
    has_bias: bool,
    anchor_id: int | None = None,
    anchor_weight: ArrayLike | None = None,
    anchor_bias: float | None = None,
    base_weight: ArrayLike | None = None,
    base_bias: ArrayLike | None = None,
    chunk_size: int,
    max_validation_rmse: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int | float | bool]]:
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

    design, feature_precision = _design(features, "features", has_bias)
    heldout, _ = _design(validation_features, "validation_features", has_bias)
    if design.shape[1] != heldout.shape[1]:
        raise ValueError("features and validation_features must have the same width")
    width = design.shape[1] - has_bias
    anchor = None
    if mode == "anchored":
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
    reference_id = None
    base_weights = base_biases = None
    if mode == "anchored":
        if (
            isinstance(anchor_id, (bool, np.bool_))
            or not isinstance(anchor_id, (int, np.integer))
            or not 0 <= anchor_id < classes
        ):
            raise ValueError("anchor_id must be an integer within the vocabulary")
        reference_id = int(anchor_id)
    elif mode == "base_median":
        reference_id = 0
        if base_weight is None:
            raise ValueError("base_median mode requires base_weight")
        base_weights = _real_array(base_weight, "base_weight")
        if base_weights.shape != (classes, width):
            raise ValueError("base_weight must match the vocabulary and feature width")
        if has_bias:
            if base_bias is None:
                raise ValueError("base_median mode requires base_bias when has_bias is True")
            base_biases = _real_array(base_bias, "base_bias")
            if base_biases.shape != (classes,):
                raise ValueError("base_bias must be a vector matching the vocabulary")
        elif base_bias is not None:
            raise ValueError("base_bias requires has_bias=True")

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

    training_reference = validation_reference = None
    if reference_id is not None:
        training_reference = observed[:, reference_id : reference_id + 1]
        validation_reference = validation_observed[:, reference_id : reference_id + 1]
    weights = np.empty((classes, width), dtype=np.float64)
    bias = np.empty(classes, dtype=np.float64) if has_bias else None
    for start in range(0, classes, chunk_size):
        columns = slice(start, start + chunk_size)
        targets = _targets(observed, columns, training_reference, "scores")
        with np.errstate(over="ignore", invalid="ignore"):
            parameters = inverse @ targets
            if anchor is not None:
                parameters += anchor[:, None]
            weights[columns] = parameters[:width].T
            if bias is not None:
                bias[columns] = parameters[-1]
    del inverse

    if anchor is not None and reference_id is not None:
        weights[reference_id] = anchor[:width]
        if bias is not None:
            bias[reference_id] = anchor[-1]
    if base_weights is not None:
        _restore_median_gauge(weights, base_weights, chunk_size, "base_weight")
    if base_biases is not None and bias is not None:
        _restore_median_gauge(bias[:, None], base_biases[:, None], chunk_size, "base_bias")

    # Score the returned parameters so the residual includes rounding when restoring the gauge.
    reference = None
    if reference_id is not None:
        reference = weights[reference_id]
        if bias is not None:
            reference = np.append(reference, bias[reference_id])
    training_squared_error = validation_squared_error = 0.0
    for start in range(0, classes, chunk_size):
        columns = slice(start, start + chunk_size)
        targets = _targets(observed, columns, training_reference, "scores")
        heldout_targets = _targets(
            validation_observed, columns, validation_reference, "validation_scores"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            parameters = weights[columns].T
            if bias is not None:
                parameters = np.vstack((parameters, bias[columns]))
            if reference is not None:
                parameters = parameters - reference[:, None]
            residual = design @ parameters - targets
            heldout_residual = heldout @ parameters - heldout_targets
            training_squared_error += float(np.vdot(residual, residual))
            validation_squared_error += float(np.vdot(heldout_residual, heldout_residual))

    # In contrast modes the reference contributes exactly zero.
    scored_classes = classes - (reference_id is not None)
    training_rmse = float(np.sqrt(training_squared_error / (design.shape[0] * scored_classes)))
    validation_rmse = float(np.sqrt(validation_squared_error / (heldout.shape[0] * scored_classes)))
    error_kind = "absolute" if reference_id is None else "contrast"
    if not np.isfinite(training_rmse) or not np.isfinite(validation_rmse):
        raise ValueError(f"{mode} fit produced nonfinite {error_kind} errors")
    if validation_rmse > max_validation_rmse:
        raise ValueError(
            f"held-out {error_kind} RMSE {validation_rmse:.6g} exceeds "
            f"max_validation_rmse {max_validation_rmse:.6g}"
        )
    fit: dict[str, int | float | bool] = {
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
        "max_validation_rmse": max_validation_rmse,
    }
    if reference_id is not None:
        fit["anchor_id" if anchor is not None else "reference_id"] = reference_id
        fit["training_contrast_rmse"] = training_rmse
        fit["validation_contrast_rmse"] = validation_rmse
    if mode != "anchored":
        fit["training_rmse"] = training_rmse
        fit["validation_rmse"] = validation_rmse
    return weights, bias, fit
