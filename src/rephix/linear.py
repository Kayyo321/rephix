"""Recover an affine output layer when its input features are observable.

This is a controlled identification baseline, not a method for recovering a
transformer's hidden layers from text. Probability observations identify only a
class-centered representative: softmax is unchanged by an arbitrary shared
affine shift to every class's logit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_save_npz


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if result.ndim != 2 or not all(result.shape):
        raise ValueError(f"{name} must be a nonempty 2D matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class RecoveryResult:
    """An affine model ``features @ weights + bias`` and its fit diagnostics."""

    weights: np.ndarray
    bias: np.ndarray
    diagnostics: dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return logits (centered logits for probability-based recovery)."""
        x = _matrix(features, "features")
        if x.shape[1] != self.weights.shape[0]:
            raise ValueError("features have a different width from the fitted model")
        return x @ self.weights + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "diagnostics": dict(self.diagnostics),
        }

    def save(self, path: str | Path) -> None:
        """Save arrays plus JSON diagnostics to a pickle-free NumPy archive."""
        atomic_save_npz(
            path,
            weights=self.weights,
            bias=self.bias,
            diagnostics=np.array(json.dumps(self.diagnostics, allow_nan=False)),
        )


def recover(
    features: np.ndarray,
    observations: np.ndarray,
    *,
    kind: str = "logits",
    ridge: float = 0.0,
) -> RecoveryResult:
    """Fit an observable affine layer by stable least squares.

    ``features`` is [samples, features], ``observations`` is [samples, classes].
    ``kind`` accepts ``logits``, ``probabilities``, or ``log_probs``. The latter
    two must be complete, normalized distributions and recover parameters in the
    zero-class-mean gauge. Top-k observations cannot be used with this solver.
    Ridge minimizes squared residuals plus ``ridge * ||weights||**2``; it never
    penalizes the intercept. Rank diagnostics always describe the unregularized
    design: regularization does not make an unidentifiable model identifiable.
    """
    x = _matrix(features, "features")
    observed = _matrix(observations, "observations")
    if x.shape[0] != observed.shape[0]:
        raise ValueError("features and observations must have the same sample count")
    if observed.shape[1] < 2:
        raise ValueError("observations must contain at least two classes")
    if kind not in {"logits", "probabilities", "log_probs"}:
        raise ValueError("kind must be logits, probabilities, or log_probs")
    if isinstance(ridge, bool):
        raise ValueError("ridge must be a finite nonnegative number")
    try:
        ridge = float(ridge)
    except (TypeError, ValueError) as exc:
        raise ValueError("ridge must be a finite nonnegative number") from exc
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite nonnegative number")

    if kind == "probabilities":
        if (observed <= 0).any():
            raise ValueError(
                "probabilities must be strictly positive; use log_probs to avoid underflow"
            )
        if not np.allclose(observed.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("probability rows must sum to one")
        targets = np.log(observed)
    elif kind == "log_probs":
        if not np.allclose(np.logaddexp.reduce(observed, axis=1), 0.0, rtol=0, atol=1e-6):
            raise ValueError("log_probs rows must be normalized (logsumexp = zero)")
        targets = observed.copy()
    else:
        targets = observed.copy()

    gauge = "full" if kind == "logits" else "zero_class_mean"
    if gauge == "zero_class_mean":
        targets -= targets.mean(axis=1, keepdims=True)

    design = np.column_stack((x, np.ones(x.shape[0])))
    # The original design determines identifiability even when a prior/penalty
    # supplies a unique numerical minimizer.
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = max(design.shape) * np.finfo(design.dtype).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    identifiable = rank == design.shape[1]
    condition = float(singular_values[0] / singular_values[-1]) if identifiable else None

    solve_design = design
    solve_targets = targets
    if ridge > 0:
        penalty = np.zeros((x.shape[1], x.shape[1] + 1))
        penalty[:, : x.shape[1]] = np.sqrt(ridge) * np.eye(x.shape[1])
        solve_design = np.concatenate((design, penalty), axis=0)
        solve_targets = np.concatenate((targets, np.zeros((x.shape[1], observed.shape[1]))), axis=0)
    parameters, _, _, _ = np.linalg.lstsq(solve_design, solve_targets, rcond=None)
    if gauge == "zero_class_mean":
        # Remove rounding-scale gauge drift introduced by the solve.
        parameters -= parameters.mean(axis=1, keepdims=True)
    residual = design @ parameters - targets
    diagnostics = {
        "observation_kind": kind,
        "parameter_gauge": gauge,
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "classes": int(observed.shape[1]),
        "design_rank": rank,
        "parameters_per_class": int(design.shape[1]),
        "identifiable_in_reported_gauge": identifiable,
        "condition_number": condition,
        "rank_tolerance": float(tolerance),
        "ridge": ridge,
        "training_target_rmse": float(np.sqrt(np.mean(residual**2))),
        "interpretation": (
            "Unique affine parameters in the reported gauge for exact observations."
            if identifiable
            else "Rank deficient: this is one minimum-norm/regularized fit, not unique parameter recovery."
        ),
    }
    return RecoveryResult(parameters[:-1], parameters[-1], diagnostics)


def demo(
    *, seed: int = 42, samples: int = 256, features: int = 8, classes: int = 5
) -> dict[str, Any]:
    """Run a deterministic planted-weight recovery with unseen test features."""
    for name, value, minimum in (
        ("samples", samples, 4),
        ("features", features, 1),
        ("classes", classes, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples, features))
    weights = rng.normal(scale=0.5, size=(features, classes))
    bias = rng.normal(scale=0.2, size=classes)
    logits = x @ weights + bias
    split = 3 * samples // 4
    exact = recover(x[:split], logits[:split])
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    centered = recover(x[:split], log_probs[:split], kind="log_probs")
    theta = np.vstack((weights, bias))
    centered_theta = theta - theta.mean(axis=1, keepdims=True)
    test_logits = logits[split:]
    centered_test_logits = test_logits - test_logits.mean(axis=1, keepdims=True)

    def metrics(
        result: RecoveryResult, target_theta: np.ndarray, targets: np.ndarray
    ) -> dict[str, Any]:
        fitted_theta = np.vstack((result.weights, result.bias))
        predictions = result.predict(x[split:])
        return {
            "parameter_relative_error": float(
                np.linalg.norm(fitted_theta - target_theta) / np.linalg.norm(target_theta)
            ),
            "heldout_logit_rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
            "heldout_top1_accuracy": float(
                np.mean(predictions.argmax(axis=1) == test_logits.argmax(axis=1))
            ),
            "diagnostics": result.diagnostics,
        }

    return {
        "experiment": "known_feature_affine_layer_recovery",
        "seed": int(seed),
        "train_samples": int(split),
        "test_samples": int(samples - split),
        "full_logits": metrics(exact, theta, test_logits),
        "log_probabilities": metrics(centered, centered_theta, centered_test_logits),
        "scope": "Known input features and an affine output layer; this does not recover hidden transformer weights.",
    }
