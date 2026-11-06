"""Estimate a fixed-coordinate output contrast subspace from dense log probabilities.

Token columns must describe the same endpoint coordinates in the same order for
all rows. The first column is the common reference. This estimator has no access
to token embeddings, model metadata, reference weights, or hidden activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _matrix(logprobs, *, min_rows=1, min_columns=2):
    values = np.asarray(logprobs, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < min_rows
        or values.shape[1] < min_columns
        or not np.isfinite(values).all()
    ):
        raise ValueError("expected dense finite log probabilities in fixed token columns")
    return values


def log_odds(logprobs):
    """Express each log probability relative to the common first column."""
    values = _matrix(logprobs)
    return values[:, 1:] - values[:, :1]


@dataclass(frozen=True)
class LogitSubspace:
    origin: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    relative_cutoff: float
    training_rows: int

    @property
    def rank(self):
        return self.basis.shape[1]

    @property
    def rank_ceiling(self):
        return min(self.training_rows - 1, self.origin.size)

    def project(self, logprobs):
        """Project known contrasts into the fitted affine span for a residual check."""
        contrasts = log_odds(logprobs)
        if contrasts.shape[1] != self.origin.size:
            raise ValueError("projection requires all fixed token columns in fit order")
        centered = contrasts - self.origin
        return (centered @ self.basis) @ self.basis.T + self.origin

    def predict(self, anchor_logprobs, anchors):
        """Predict all contrasts using only the reference and selected anchor columns.

        ``anchors`` gives indices into the fitted contrast columns (zero indexes
        the second fitted log-probability column). Reject underdetermined anchor
        designs instead of silently inventing coordinates.
        """
        odds = log_odds(anchor_logprobs)
        indices = np.asarray(anchors)
        if (
            indices.shape != (odds.shape[1],)
            or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any()
            or (indices >= self.origin.size).any()
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("anchors must be unique fitted contrast indices in evidence order")
        if not self.rank:
            return np.broadcast_to(self.origin, (odds.shape[0], self.origin.size)).copy()
        design = self.basis[indices]
        coords, _, design_rank, _ = np.linalg.lstsq(
            design, (odds - self.origin[indices]).T, rcond=None
        )
        if design_rank != self.rank:
            raise ValueError("anchor columns do not span the estimated rank")
        return coords.T @ self.basis.T + self.origin


def fit_logit_subspace(logprobs, *, relative_cutoff=1e-6):
    """Fit an affine subspace via SVD, retaining directions above a relative cutoff.

    The numerical rank depends on support, excitation, precision, noise, and
    threshold. The rank ceiling is set by both sample count and token coverage.
    """
    if not np.isfinite(relative_cutoff) or not 0 < relative_cutoff < 1:
        raise ValueError("relative cutoff must be strictly between zero and one")
    values = _matrix(logprobs, min_rows=2)
    contrasts = log_odds(values)
    origin = contrasts.mean(axis=0)
    _, singular, right = np.linalg.svd(contrasts - origin, full_matrices=False)
    rank = min(int(np.count_nonzero(singular > relative_cutoff * singular[0])), values.shape[0] - 1)
    return LogitSubspace(origin, right[:rank].T, singular, float(relative_cutoff), values.shape[0])
