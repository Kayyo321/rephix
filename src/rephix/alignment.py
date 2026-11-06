"""Permutation and scale/sign alignment for one-hidden-layer networks.

These comparisons account for a few known parameter symmetries. They neither
identify a unique network from finite observations nor align transformer layers.
All arrays use ``activation(x @ incoming + bias) @ outgoing`` conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any, name: str, dimensions: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.ndim != dimensions or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite {dimensions}D array")
    return array


def _network(incoming, bias, outgoing, label):
    incoming = _array(incoming, f"{label} incoming", 2)
    bias = _array(bias, f"{label} hidden bias", 1)
    outgoing = _array(outgoing, f"{label} outgoing", 2)
    if incoming.shape[1] != len(bias) or outgoing.shape[0] != len(bias):
        raise ValueError(f"{label} hidden widths do not match")
    return incoming, bias, outgoing


def minimum_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Return the minimum-cost column for each row of a square cost matrix.

    Implements the Hungarian shortest augmenting-path algorithm in O(n**3)
    time and O(n**2) storage, without SciPy. Equal-cost choices are deterministic.
    """
    cost = _array(cost_matrix, "cost_matrix", 2)
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("cost_matrix must be square")
    size = len(cost)
    row_potential = np.zeros(size + 1)
    column_potential = np.zeros(size + 1)
    matched_row = np.zeros(size + 1, dtype=int)
    predecessor = np.zeros(size + 1, dtype=int)
    for row in range(1, size + 1):
        matched_row[0] = row
        column = 0
        distances = np.full(size + 1, np.inf)
        visited = np.zeros(size + 1, dtype=bool)
        while True:
            visited[column] = True
            current_row = matched_row[column]
            increment = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if visited[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1, candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced < distances[candidate]:
                    distances[candidate] = reduced
                    predecessor[candidate] = column
                if distances[candidate] < increment:
                    increment = distances[candidate]
                    next_column = candidate
            if not np.isfinite(increment):
                raise ValueError("cost_matrix caused nonfinite assignment arithmetic")
            for candidate in range(size + 1):
                if visited[candidate]:
                    row_potential[matched_row[candidate]] += increment
                    column_potential[candidate] -= increment
                else:
                    distances[candidate] -= increment
            column = next_column
            if matched_row[column] == 0:
                break
        while column:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
    assignment = np.empty(size, dtype=int)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


def _distances(left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("parameters are too large for finite alignment costs")
    return result


def _coordinates(incoming, bias, outgoing):
    return np.concatenate((incoming.ravel(), bias, outgoing.ravel()))


def _l2(values):
    # hypot avoids squaring overflow/underflow for otherwise finite weights.
    result = float(np.hypot.reduce(values))
    if not np.isfinite(result):
        raise ValueError("parameters exceed the finite range for distance reporting")
    return result


@dataclass(frozen=True)
class AlignmentResult:
    """Candidate tensors transformed into the reference's hidden-neuron order.

    ``permutation[i]`` names the original candidate neuron matched to reference
    neuron ``i``. Its incoming weights and bias are multiplied by ``scales[i]``;
    its outgoing weights are divided by that value. For tanh, scales are +/-1.
    """

    permutation: np.ndarray
    scales: np.ndarray
    incoming_weights: np.ndarray
    hidden_bias: np.ndarray
    outgoing_weights: np.ndarray
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "permutation": self.permutation.tolist(),
            "scales": self.scales.tolist(),
            "incoming_weights": self.incoming_weights.tolist(),
            "hidden_bias": self.hidden_bias.tolist(),
            "outgoing_weights": self.outgoing_weights.tolist(),
            "diagnostics": dict(self.diagnostics),
        }


def align_hidden(
    reference_in: np.ndarray,
    reference_bias: np.ndarray,
    reference_out: np.ndarray,
    candidate_in: np.ndarray,
    candidate_bias: np.ndarray,
    candidate_out: np.ndarray,
    *,
    activation: str = "relu",
) -> AlignmentResult:
    """Align a single equal-width hidden layer under supported symmetries.

    ReLU matching compares scale-canonical neuron descriptors: unit-length
    incoming weights (including bias) and outgoing weights multiplied by the
    incoming norm. A positive multiplier then places matched candidate units in
    the reference gauge. Tanh matching permits sign flips and permutations only.

    This minimizes a stated neuron-descriptor assignment cost, not function
    distance or all possible neural-network symmetries. Zero incoming neurons
    use multiplier one, retain their outgoing weights, and remain unidentifiable.
    """
    if activation not in {"relu", "tanh"}:
        raise ValueError("activation must be relu or tanh")
    ref_in, ref_bias, ref_out = _network(reference_in, reference_bias, reference_out, "reference")
    cand_in, cand_bias, cand_out = _network(
        candidate_in, candidate_bias, candidate_out, "candidate"
    )
    if ref_in.shape != cand_in.shape or ref_out.shape != cand_out.shape:
        raise ValueError("reference and candidate network shapes must match")
    ref_augmented = np.column_stack((ref_in.T, ref_bias))
    cand_augmented = np.column_stack((cand_in.T, cand_bias))
    ref_norm = np.hypot.reduce(ref_augmented, axis=1)
    cand_norm = np.hypot.reduce(cand_augmented, axis=1)
    if not np.isfinite(ref_norm).all() or not np.isfinite(cand_norm).all():
        raise ValueError("parameters are too large for finite incoming norms")
    width = len(ref_bias)
    if activation == "relu":

        def canonical(augmented, outgoing, norms):
            descriptor = np.zeros((width, augmented.shape[1] + outgoing.shape[1]))
            nonzero = norms > 0
            descriptor[nonzero, : augmented.shape[1]] = augmented[nonzero] / norms[nonzero, None]
            descriptor[nonzero, augmented.shape[1] :] = outgoing[nonzero] * norms[nonzero, None]
            return descriptor

        reference_descriptor = canonical(ref_augmented, ref_out, ref_norm)
        candidate_descriptor = canonical(cand_augmented, cand_out, cand_norm)
        costs = _distances(reference_descriptor, candidate_descriptor)
        permutation = minimum_assignment(costs)
        scales = np.ones(width)
        scalable = (ref_norm > 0) & (cand_norm[permutation] > 0)
        scales[scalable] = ref_norm[scalable] / cand_norm[permutation][scalable]
        scope = "hidden-neuron permutation and positive ReLU scale symmetries"
    else:
        reference_descriptor = np.column_stack((ref_augmented, ref_out))
        candidate_descriptor = np.column_stack((cand_augmented, cand_out))
        positive_costs = _distances(reference_descriptor, candidate_descriptor)
        negative_costs = _distances(reference_descriptor, -candidate_descriptor)
        costs = np.minimum(positive_costs, negative_costs)
        permutation = minimum_assignment(costs)
        indices = np.arange(width)
        scales = np.where(
            negative_costs[indices, permutation] < positive_costs[indices, permutation], -1.0, 1.0
        )
        scope = "hidden-neuron permutation and tanh sign symmetries; no continuous scaling"
    if not np.isfinite(scales).all() or (scales == 0).any():
        raise ValueError("parameters require nonfinite or zero alignment scales")
    aligned_in = cand_in[:, permutation] * scales
    aligned_bias = cand_bias[permutation] * scales
    aligned_out = cand_out[permutation] / scales[:, None]
    if not all(np.isfinite(value).all() for value in (aligned_in, aligned_bias, aligned_out)):
        raise ValueError("alignment produced nonfinite parameters")
    reference = _coordinates(ref_in, ref_bias, ref_out)
    candidate = _coordinates(cand_in, cand_bias, cand_out)
    aligned = _coordinates(aligned_in, aligned_bias, aligned_out)
    denominator = _l2(reference)
    before = _l2(candidate - reference)
    after = _l2(aligned - reference)
    diagnostics = {
        "activation": activation,
        "hidden_width": width,
        "scope": scope,
        "assignment_cost": float(costs[np.arange(width), permutation].sum()),
        "raw_parameter_l2": before,
        "aligned_parameter_l2": after,
        "raw_parameter_relative_l2": before / denominator if denominator else None,
        "aligned_parameter_relative_l2": after / denominator if denominator else None,
        "reference_zero_incoming_neurons": int(np.count_nonzero(ref_norm == 0)),
        "candidate_zero_incoming_neurons": int(np.count_nonzero(cand_norm == 0)),
        "reference_zero_outgoing_neurons": int(np.count_nonzero(np.all(ref_out == 0, axis=1))),
        "candidate_zero_outgoing_neurons": int(np.count_nonzero(np.all(cand_out == 0, axis=1))),
        "interpretation": (
            "A comparison under stated single-layer symmetries; neither a uniqueness proof "
            "nor general transformer alignment. Inactive or duplicate neurons may admit "
            "additional ambiguities. Output bias is outside this hidden-layer comparison."
        ),
    }
    return AlignmentResult(permutation, scales, aligned_in, aligned_bias, aligned_out, diagnostics)


def forward(
    features: np.ndarray,
    incoming: np.ndarray,
    bias: np.ndarray,
    outgoing: np.ndarray,
    output_bias: np.ndarray | None = None,
    *,
    activation: str = "relu",
) -> np.ndarray:
    """Evaluate the tiny network used in the controlled symmetry experiment."""
    incoming, bias, outgoing = _network(incoming, bias, outgoing, "network")
    features = _array(features, "features", 2)
    if features.shape[1] != incoming.shape[0]:
        raise ValueError("feature width does not match incoming weights")
    hidden = features @ incoming + bias
    if activation == "relu":
        hidden = np.maximum(hidden, 0)
    elif activation == "tanh":
        hidden = np.tanh(hidden)
    else:
        raise ValueError("activation must be relu or tanh")
    result = hidden @ outgoing
    if output_bias is not None:
        output_bias = _array(output_bias, "output_bias", 1)
        if len(output_bias) != outgoing.shape[1]:
            raise ValueError("output bias width does not match outgoing weights")
        result += output_bias
    if not np.isfinite(result).all():
        raise ValueError("network evaluation produced nonfinite outputs")
    return result


def symmetry_demo(*, seed: int = 42, width: int = 8, activation: str = "relu") -> dict[str, Any]:
    """Plant known symmetry transformations and test unseen inputs separately."""
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("width must be a positive integer")
    if activation not in {"relu", "tanh"}:
        raise ValueError("activation must be relu or tanh")
    rng = np.random.default_rng(seed)
    incoming = rng.normal(size=(4, width))
    bias = rng.normal(size=width)
    outgoing = rng.normal(size=(width, 3))
    output_bias = rng.normal(size=3)
    planted_permutation = rng.permutation(width)
    planted_scales = (
        np.exp(rng.uniform(-1.5, 1.5, size=width))
        if activation == "relu"
        else rng.choice([-1.0, 1.0], size=width)
    )
    candidate_in = incoming[:, planted_permutation] * planted_scales
    candidate_bias = bias[planted_permutation] * planted_scales
    candidate_out = outgoing[planted_permutation] / planted_scales[:, None]
    result = align_hidden(
        incoming, bias, outgoing, candidate_in, candidate_bias, candidate_out, activation=activation
    )
    observed = rng.normal(size=(16, 4))
    heldout = rng.normal(size=(128, 4))

    def errors(features):
        reference = forward(features, incoming, bias, outgoing, output_bias, activation=activation)
        candidate = forward(
            features,
            candidate_in,
            candidate_bias,
            candidate_out,
            output_bias,
            activation=activation,
        )
        aligned = forward(
            features,
            result.incoming_weights,
            result.hidden_bias,
            result.outgoing_weights,
            output_bias,
            activation=activation,
        )
        return {
            "samples": len(features),
            "raw_function_max_abs_error": float(np.max(np.abs(candidate - reference))),
            "aligned_function_max_abs_error": float(np.max(np.abs(aligned - reference))),
        }

    # Two width-two ReLU models agree on all observed x <= 1 but differ at x=3.
    # This is an observation counterexample, not a symmetry transformation.
    finite_inputs = np.array([[-1.0], [0.0], [1.0]])
    first = (np.array([[1.0, 0.0]]), np.array([0.0, 0.0]), np.array([[1.0], [0.0]]))
    second = (np.array([[1.0, 1.0]]), np.array([0.0, -2.0]), np.array([[1.0], [1.0]]))
    observed_difference = np.abs(forward(finite_inputs, *first) - forward(finite_inputs, *second))
    unseen_difference = forward(np.array([[3.0]]), *second) - forward(np.array([[3.0]]), *first)
    return {
        "experiment": "single_hidden_layer_symmetry_alignment",
        "seed": int(seed),
        "activation": activation,
        "alignment": result.to_dict(),
        "observed_inputs": errors(observed),
        "heldout_inputs": errors(heldout),
        "finite_observation_counterexample": {
            "architecture": "one input, two ReLU hidden neurons, one output",
            "observed_inputs": finite_inputs.ravel().tolist(),
            "max_observed_output_difference": float(observed_difference.max()),
            "unseen_input": 3.0,
            "unseen_output_difference": float(unseen_difference.item()),
            "interpretation": "Equal finite observations can also hide different functions, beyond parameter symmetries.",
        },
        "scope": "Planted transformations of a tiny one-hidden-layer model; no transformer recovery claim.",
    }


def _run_demo(args):
    result = symmetry_demo(seed=args.seed, width=args.width, activation=args.activation)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return result


def add_subcommands(subparsers):
    parser = subparsers.add_parser(
        "symmetry-demo",
        help="Compare tiny networks under known permutation and scale/sign symmetries",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--output")
    parser.set_defaults(func=_run_demo)
